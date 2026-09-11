import itertools
import logging
import argparse
import torch
import numpy as np
import os, sys
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Ensure project root (containing `utils`, `models`, etc.) is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch.nn as nn
import torch.nn.functional as Fnn
import importlib.util as _ilu

from torchvision.datasets import MNIST

# ── poison module imports ──────────────────────────────────────────────────────
POISON_ROOT = os.path.join(PROJECT_ROOT, 'poison')
if POISON_ROOT not in sys.path:
    sys.path.append(POISON_ROOT)

def _import_poison(name):
    """Load a module from the poison sub-folder without polluting the namespace."""
    path = os.path.join(POISON_ROOT, f'{name}.py')
    spec = _ilu.spec_from_file_location(f'_poison_{name}', path)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_poison_fm   = _import_poison('FusionModel')
_poison_args = _import_poison('Arguments')
_poison_schemes = _import_poison('schemes_poison')
PoisonQNet                    = _poison_fm.QNet
poison_single_enta_to_design  = _poison_fm.single_enta_to_design
PoisonArguments               = _poison_args.Arguments
Scheme                        = _poison_schemes.Scheme
# ──────────────────────────────────────────────────────────────────────────────
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.dataset import MNISTDataLoaders, generate_mnist_concept_dataset
from utils.quantum_circuit_helpers import (
    _compute_pullback_coefficients_over_set,
    _build_xyz_zz_observables,
    _cu3_matrix,
    _data_upload_matrix,
    _embed_single_qubit_gate,
    _embed_two_qubit_gate,
    _select_top_pullback_observables_per_sample,
    _select_top_pullback_observables_over_set,
    _u3_matrix,
    build_target_observable,
    state_to_selected_observable_features_torch,
)
from utils.plot import (
    plot_concept_accuracy,
    plot_global_explanation,
    plot_grayscale_saliency,
    plot_attribution_correlation,
)
from explanations.concept import CAR, CAV
from explanations.feature import CARFeatureImportance, VanillaFeatureImportance
from sklearn.metrics import accuracy_score
from tqdm import tqdm

def _observable_column_names_for_repr(repr_mode: str, n_qubits: int) -> list[str]:
    if repr_mode == "local_x":
        return [f"X{idx}" for idx in range(n_qubits)]
    if repr_mode == "local_y":
        return [f"Y{idx}" for idx in range(n_qubits)]
    if repr_mode == "local_z":
        return [f"Z{idx}" for idx in range(n_qubits)]
    if repr_mode == "local_zz":
        return [
            f"Z{i_idx}Z{j_idx}"
            for i_idx in range(n_qubits)
            for j_idx in range(i_idx + 1, n_qubits)
        ]
    if repr_mode == "pullback":
        observables, names = _build_xyz_zz_observables(n_qubits)
        return names
    raise ValueError(f"Unsupported local observable repr_mode: {repr_mode!r}")


def _observable_matrices_for_repr(repr_mode: str, n_qubits: int) -> list[np.ndarray]:
    observables, names = _build_xyz_zz_observables(n_qubits)
    name_to_observable = {name: observable for name, observable in zip(names, observables)}
    return [name_to_observable[name] for name in _observable_column_names_for_repr(repr_mode, n_qubits)]


def _observable_feature_csv_path(run_save_dir: Path, concept_layer: int | None = None) -> Path:
    if concept_layer is not None and int(concept_layer) >= 0:
        return (
            run_save_dir.parent.parent
            / "all_features"
            / f"layer_{int(concept_layer)}"
            / run_save_dir.name
            / "observable_features.csv"
        )
    return run_save_dir.parent.parent / "all_features" / run_save_dir.name / "observable_features.csv"


def _load_repr_from_observable_csv(
    csv_path: Path,
    repr_mode: str,
    n_qubits: int,
) -> tuple[np.ndarray, np.ndarray]:
    column_names = _observable_column_names_for_repr(repr_mode, n_qubits)
    required_columns = ["Class"] + column_names
    df = pd.read_csv(csv_path, usecols=required_columns)
    features = df[column_names].to_numpy(dtype=np.float32, copy=True)
    y_test_from_csv = df["Class"].to_numpy(dtype=np.int64, copy=True)
    return features, y_test_from_csv


def _restore_mnist_loader_labels(
    labels: np.ndarray,
    digits_of_interest: list[int],
) -> np.ndarray:
    """Map MNISTDataLoaders remapped labels [0..K-1] back to original digit ids.

    If labels are already real digit ids, they are returned unchanged.
    """
    y = np.asarray(labels, dtype=np.int64)
    if y.size == 0:
        return y
    if np.all((0 <= y) & (y < len(digits_of_interest))):
        mapping = np.asarray(digits_of_interest, dtype=np.int64)
        return mapping[y]
    return y


def _safe_corrcoef(x_vals: np.ndarray, y_vals: np.ndarray) -> float:
    x_vals = np.asarray(x_vals, dtype=np.float64)
    y_vals = np.asarray(y_vals, dtype=np.float64)
    if x_vals.size < 2 or y_vals.size < 2:
        return float("nan")
    if np.isclose(np.std(x_vals), 0.0) or np.isclose(np.std(y_vals), 0.0):
        return float("nan")
    return float(np.corrcoef(x_vals, y_vals)[0, 1])


def _compute_global_explanation_correlations(
    results_df: pd.DataFrame,
    concept_names: list[str],
) -> tuple[float, float, dict[int, float], dict[int, float]]:
    scores_data = []
    classes = results_df["Class"].unique()
    methods = results_df["Method"].unique()

    for class_idx, concept, method in itertools.product(classes, concept_names, methods):
        attr = np.array(
            results_df.loc[
                (results_df.Class == class_idx)
                & (results_df.Method == method)
            ][concept]
        )
        if len(attr) == 0:
            continue
        score = float(np.mean(attr))
        scores_data.append([method, class_idx, concept, score])

    if len(scores_data) == 0:
        return float("nan"), float("nan")

    scores_df = pd.DataFrame(
        scores_data,
        columns=["Method", "Class", "Concept", "Score"],
    )

    true_scores = scores_df.loc[scores_df.Method == "True Prop."]["Score"].to_numpy()
    tcar_scores = scores_df.loc[scores_df.Method == "TCAR"]["Score"].to_numpy()
    tcav_scores = scores_df.loc[scores_df.Method == "TCAV"]["Score"].to_numpy()

    tcar_corr = _safe_corrcoef(tcar_scores, true_scores)
    tcav_corr = _safe_corrcoef(tcav_scores, true_scores)

    # Per-class correlations across concepts for each class.
    tcar_corr_per_class = {}
    tcav_corr_per_class = {}
    for class_idx in sorted(scores_df["Class"].unique()):
        class_scores = scores_df.loc[scores_df["Class"] == class_idx]
        true_cls = class_scores.loc[class_scores.Method == "True Prop."]["Score"].to_numpy()
        tcar_cls = class_scores.loc[class_scores.Method == "TCAR"]["Score"].to_numpy()
        tcav_cls = class_scores.loc[class_scores.Method == "TCAV"]["Score"].to_numpy()
        tcar_corr_per_class[int(class_idx)] = _safe_corrcoef(tcar_cls, true_cls)
        tcav_corr_per_class[int(class_idx)] = _safe_corrcoef(tcav_cls, true_cls)

    return tcar_corr, tcav_corr, tcar_corr_per_class, tcav_corr_per_class


def plot_poison_correlation_curves(
    correlation_records: list[dict],
    output_dir: Path,
) -> None:
    if len(correlation_records) == 0:
        logging.warning("No correlation records available for plotting.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    corr_df = pd.DataFrame(correlation_records)
    corr_df["x_alpha"] = corr_df["x_alpha"].round(2)
    corr_df["y_alpha"] = corr_df["y_alpha"].round(2)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    pairs = corr_df["pair"].unique().tolist()

    def _plot_single(method_col: str, title_prefix: str, file_name: str) -> None:
        plt.style.use('ggplot')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        df_x = corr_df[corr_df['y_alpha'] == 0].sort_values('x_alpha')
        for i_idx, pair in enumerate(pairs):
            color = colors[i_idx % len(colors)]
            data = df_x[df_x['pair'] == pair]
            ax1.plot(
                data['x_alpha'],
                data[method_col],
                marker='o',
                linestyle='-',
                color=color,
                label=pair,
            )
        ax1.set_title('Feature Randomization')
        ax1.set_xlabel('x_alpha')
        ax1.set_ylabel('Correlation')
        ax1.set_ylim(-1.05, 1.05)
        ax1.legend(fontsize='small')

        df_y = corr_df[corr_df['x_alpha'] == 0].sort_values('y_alpha')
        for i_idx, pair in enumerate(pairs):
            color = colors[i_idx % len(colors)]
            data = df_y[df_y['pair'] == pair]
            ax2.plot(
                data['y_alpha'],
                data[method_col],
                marker='s',
                linestyle='-',
                color=color,
                label=pair,
            )
        ax2.set_title('Label Flipping')
        ax2.set_xlabel('y_alpha')
        ax2.set_ylabel('Correlation')
        ax2.set_ylim(-1.05, 1.05)
        ax2.legend(fontsize='small')

        fig.suptitle(title_prefix)
        plt.tight_layout()
        save_path = output_dir / file_name
        plt.savefig(save_path)
        plt.close(fig)
        logging.info(f"Saved correlation curve figure: {save_path}")

    _plot_single(
        method_col='tcar_corr',
        title_prefix='TCAR Correlation vs Alpha',
        file_name='tcar_correlation_curves.png',
    )
    _plot_single(
        method_col='tcav_corr',
        title_prefix='TCAV Correlation vs Alpha',
        file_name='tcav_correlation_curves.png',
    )


def dm2vec(rho):
    """Convert a pure quantum state to a real feature vector.

    For a complex statevector/density matrix, we concatenate real and imaginary
    parts directly to preserve the pure-state information.
    """
    if hasattr(rho, 'data'):
        rho = rho.data
    rho = np.asarray(rho)

    if rho.ndim == 1:
        # Statevector: [a0, a1, ...] -> [Re(a), Im(a)]
        feature = np.concatenate([rho.real, rho.imag], axis=0).astype(np.float32)
    elif rho.ndim == 2:
        # Density/state matrix: flatten and keep both real and imaginary parts.
        if rho.shape[0] != rho.shape[1]:
            raise ValueError(f"Density matrix must be square, got shape {rho.shape}")
        flat = rho.reshape(-1)
        feature = np.concatenate([flat.real, flat.imag], axis=0).astype(np.float32)
    else:
        raise ValueError(f"Unsupported quantum state shape: {rho.shape}")

    feature = np.nan_to_num(feature, nan=0, posinf=0, neginf=0)
    return feature


def save_all_features(
    model: "PoisonQNetWrapper",
    x_input: np.ndarray | torch.Tensor,
    save_csv_path: Path,
    y_input: np.ndarray | torch.Tensor | None = None,
    batch_size: int = 256,
) -> np.ndarray:
    """Save all local and pair observable expectations for a fixed concept layer.

    The feature order is aligned with _build_xyz_zz_observables:
    [X0..X(n-1), Y0..Y(n-1), Z0..Z(n-1), X0X1, Y0Y1, Z0Z1, ...].

    Parameters
    ----------
    model
        Poison model wrapper. Layer selection follows model.concept_layer.
    x_input
        Input samples, shape [N, 1, 28, 28] or [N, 28, 28].
    save_csv_path
        Output CSV path.
    y_input
        Optional labels, shape [N]. If provided, saved as first column "Class".
    batch_size
        Mini-batch size for feature extraction.

    Returns
    -------
    np.ndarray
        Feature matrix of shape [N, 3*n_qubits + 3*C(n_qubits, 2)].
    """
    if model.repr_mode != "statevector":
        raise ValueError(
            "save_all_features requires model.repr_mode='statevector' so features "
            "are computed from quantum states in canonical observable order."
        )

    if isinstance(x_input, np.ndarray):
        x_tensor = torch.from_numpy(x_input)
    else:
        x_tensor = x_input

    if x_tensor.ndim == 3:
        x_tensor = x_tensor.unsqueeze(1)
    if x_tensor.ndim != 4:
        raise ValueError(
            f"x_input must have shape [N, 1, H, W] or [N, H, W], got {tuple(x_tensor.shape)}"
        )

    n_samples = x_tensor.shape[0]
    all_features = []
    model.eval()
    observable_family = model._pullback_observables
    observable_names = model._pullback_observable_names
    observable_tensor = torch.tensor(np.stack(observable_family), dtype=torch.complex64)


    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            x_batch = x_tensor[start:end]
            reps = model.input_to_representation(x_batch)
            reps_torch = torch.from_numpy(np.asarray(reps))
            batch_features = state_to_selected_observable_features_torch(
                reps_torch,
                observable_tensor,
            ).cpu().numpy()
            all_features.append(batch_features)

    features = np.concatenate(all_features, axis=0).astype(np.float32)
    df = pd.DataFrame(features, columns=observable_names)

    if y_input is not None:
        if isinstance(y_input, torch.Tensor):
            y_vals = y_input.detach().cpu().numpy()
        else:
            y_vals = np.asarray(y_input)
        if y_vals.shape[0] != n_samples:
            raise ValueError(
                f"y_input length ({y_vals.shape[0]}) does not match x_input size ({n_samples})"
            )
        df.insert(0, "Class", y_vals.astype(int))

    save_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_csv_path, index=False)
    logging.info(
        f"Saved all observable features to {save_csv_path} with shape {features.shape} "
        f"(layer={model.concept_layer})"
    )
    return features


# ── Wrapper for pure-quantum POISON model ─────────────────────────────────────

class PoisonQNetWrapper(nn.Module):
    """Wraps the pure-quantum POISON QNet to provide the same concept-pipeline
    interface as the hybrid model (input_to_representation / representation_to_output).

    Parameters
    ----------
    poison_model : PoisonQNet
        An instantiated POISON QNet (backend='tq').
    repr_mode : {'statevector', 'pullback'}
        'statevector' – full pure statevector after all circuit layers.
        'pullback'    – real-valued observable expectation (pullback) coefficients.
    """

    def __init__(
        self,
        poison_model,
        repr_mode: str = 'statevector',
        concept_layer: int = -1,
        pullback_target_mode: str = "z_sum",
        pullback_sparse_top_n: int = 0,
    ):
        super().__init__()
        self.model     = poison_model
        self.repr_mode = repr_mode
        self.concept_layer = concept_layer
        self.args      = poison_model.args
        self.design    = poison_model.design
        self.pullback_target_mode = pullback_target_mode
        self.pullback_sparse_top_n = int(pullback_sparse_top_n)
        self._pullback_observable_indices: np.ndarray | None = None
        pullback_observables, pullback_names = _build_xyz_zz_observables(self.args.n_qubits)
        self._pullback_observable_names: list[str] = list(pullback_names)
        self._pullback_observables = pullback_observables
        self._pullback_observable_denoms = np.array(
            [
                float(np.real(np.trace(op.conj().T @ op)))
                for op in self._pullback_observables
            ],
            dtype=np.float32,
        )

    # ── parameter pass-throughs so existing helper functions work ────────────
    @property
    def q_params_rot(self):
        return self.model.QuantumLayer.q_params_rot

    @property
    def q_params_enta(self):
        return self.model.QuantumLayer.q_params_enta

    # ── preprocessing (replicates TQLayer internal preprocessing) ───────────
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        task_name = self.args.task

        # Keep exactly the same preprocessing behavior as TQLayer.forward.
        if not task_name.startswith('QML'):
            x = self.model.QuantumLayer.adaptive_pool(x)

        x = x.view(bsz, self.args.n_qubits, 4)
        return x  # (bsz, n_qubits, n_features_per_qubit)

    # ── concept-pipeline interface ───────────────────────────────────────────
    def input_to_representation(self, x: torch.Tensor):
        """Return concept representation for a batch of raw images.

        Returns
        -------
        numpy.ndarray, shape (bsz, repr_dim)
            complex64 for 'statevector', float32 for 'pullback'.
        """
        x_proc = x
        if x_proc.ndim == 4 or (x_proc.ndim == 3 and x_proc.shape[-1] == 28):
            x_proc = self.preprocess(x_proc)
        if self.repr_mode == 'statevector':
            return self._get_statevector(x_proc)
        elif self.repr_mode == 'pullback':
            return self._get_pullback(x_proc)
        elif self.repr_mode in {"local_x", "local_y", "local_z", "local_zz"}:
            return self._get_local_observable_features(x_proc)
        else:
            raise ValueError(f"Unknown repr_mode: {self.repr_mode!r}")

    def representation_to_output(self, psi, x: torch.Tensor) -> torch.Tensor:
        """Compatibility stub – for a pure-quantum model the full forward IS the output."""
        return self.forward(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x, self.args.n_qubits, self.args.task)

    def save(self, model_dir):
        torch.save(self.model.state_dict(), Path(model_dir) / 'vqc_model.pt')

    def load_state_dict(self, state_dict, strict=True):
        return self.model.load_state_dict(state_dict, strict=strict)

    def set_pullback_observables(self, observable_indices: np.ndarray) -> None:
        """Configure which observables are used as pullback concept features."""
        all_ops, all_names = _build_xyz_zz_observables(self.args.n_qubits)
        idx = np.asarray(observable_indices, dtype=np.int64)
        if idx.ndim != 1 or idx.size == 0:
            raise ValueError("observable_indices must be a non-empty 1D array")
        if np.any(idx < 0) or np.any(idx >= len(all_ops)):
            raise ValueError(
                f"observable_indices out of range [0, {len(all_ops) - 1}]"
            )
        self._pullback_observable_indices = idx
        self._pullback_observable_names = [all_names[i] for i in idx.tolist()]

    def _selected_design(self):
        """Return ops up to concept_layer (inclusive); -1 means full circuit."""
        if self.concept_layer == -1:
            return self.design
        max_layer = self.args.n_layers - 1
        if not (0 <= self.concept_layer <= max_layer):
            raise ValueError(
                f"concept_layer must be in [-1, {max_layer}], got {self.concept_layer}"
            )
        return [op for op in self.design if int(op[2]) <= self.concept_layer]

    def _remaining_design(self):
        """Return ops after concept_layer; -1 means no remaining layers."""
        if self.concept_layer == -1:
            return []
        max_layer = self.args.n_layers - 1
        if not (0 <= self.concept_layer <= max_layer):
            raise ValueError(
                f"concept_layer must be in [-1, {max_layer}], got {self.concept_layer}"
            )
        return [op for op in self.design if int(op[2]) > self.concept_layer]

    # ── private representation builders ─────────────────────────────────────
    def _build_full_unitary(self, x_np: np.ndarray) -> np.ndarray:
        """Build the full circuit unitary U for a single preprocessed sample.

        Parameters
        ----------
        x_np : ndarray, shape (n_qubits, n_features_per_qubit)
            Already preprocessed single-sample input.

        Returns
        -------
        ndarray, shape (2**n_qubits, 2**n_qubits), complex64
        """
        n_qubits = self.args.n_qubits
        dim = 1 << n_qubits
        q_rot  = self.q_params_rot.detach().cpu().numpy()
        q_enta = self.q_params_enta.detach().cpu().numpy()

        u = np.eye(dim, dtype=np.complex64)
        for op_type, wires, layer in self._selected_design():
            if op_type == "U3":
                qubit = int(wires[0])
                theta, phi, lam = q_rot[layer][qubit]
                gate = _embed_single_qubit_gate(_u3_matrix(theta, phi, lam), qubit, n_qubits)
            elif op_type == "C(U3)":
                control, target = int(wires[0]), int(wires[1])
                theta, phi, lam = q_enta[layer][control]
                gate = _embed_two_qubit_gate(
                    _cu3_matrix(theta, phi, lam), control, target, n_qubits
                )
            elif op_type == "data":
                qubit = int(wires[0])
                gate = _embed_single_qubit_gate(
                    _data_upload_matrix(x_np[qubit]), qubit, n_qubits
                )
            else:
                continue
            u = gate @ u
        return u

    def _build_remaining_unitary(self, x_np: np.ndarray) -> np.ndarray:
        """Build sample-dependent unitary from remaining circuit after concept_layer."""
        n_qubits = self.args.n_qubits
        dim = 1 << n_qubits
        q_rot  = self.q_params_rot.detach().cpu().numpy()
        q_enta = self.q_params_enta.detach().cpu().numpy()

        u = np.eye(dim, dtype=np.complex64)
        for op_type, wires, layer in self._remaining_design():
            if op_type == "U3":
                qubit = int(wires[0])
                theta, phi, lam = q_rot[layer][qubit]
                gate = _embed_single_qubit_gate(_u3_matrix(theta, phi, lam), qubit, n_qubits)
            elif op_type == "C(U3)":
                control, target = int(wires[0]), int(wires[1])
                theta, phi, lam = q_enta[layer][control]
                gate = _embed_two_qubit_gate(
                    _cu3_matrix(theta, phi, lam), control, target, n_qubits
                )
            elif op_type == "data":
                qubit = int(wires[0])
                gate = _embed_single_qubit_gate(
                    _data_upload_matrix(x_np[qubit]), qubit, n_qubits
                )
            else:
                continue
            u = gate @ u
        return u

    def _get_statevector(self, x_proc: torch.Tensor) -> np.ndarray:
        """Full statevector after all circuit layers.  Shape: (bsz, 2**n_qubits)."""
        dim = 1 << self.args.n_qubits
        zero_state = np.zeros(dim, dtype=np.complex64)
        zero_state[0] = 1.0

        results = []
        for b in range(x_proc.shape[0]):
            x_np = x_proc[b].detach().cpu().numpy()
            u = self._build_full_unitary(x_np)
            results.append(u @ zero_state)
        return np.stack(results, axis=0)

    def _get_pullback(self, x_proc: torch.Tensor) -> np.ndarray:
        """Return per-sample pullback coefficients from each sample's remaining circuit."""
        target_obs = build_target_observable(
            self.args.n_qubits,
            mode=self.pullback_target_mode,
        )

        coeff_rows = []
        for idx in range(x_proc.shape[0]):
            x_np = x_proc[idx].detach().cpu().numpy()
            u_tail = self._build_remaining_unitary(x_np)
            o_pullback = u_tail.conj().T @ target_obs @ u_tail

            row = []
            for basis_op, denom in zip(self._pullback_observables, self._pullback_observable_denoms):
                coef = float(np.real(np.trace(basis_op.conj().T @ o_pullback))) / float(denom)
                row.append(coef)
            coeff_rows.append(np.array(row, dtype=np.float32))

        coeff_mat = np.stack(coeff_rows, axis=0)

        if self.pullback_sparse_top_n > 0:
            keep_n = min(self.pullback_sparse_top_n, coeff_mat.shape[1])
            for row_idx in range(coeff_mat.shape[0]):
                top_idx = np.argsort(-np.abs(coeff_mat[row_idx]))[:keep_n]
                mask = np.zeros(coeff_mat.shape[1], dtype=bool)
                mask[top_idx] = True
                coeff_mat[row_idx, ~mask] = 0.0

        if self._pullback_observable_indices is not None:
            coeff_mat = coeff_mat[:, self._pullback_observable_indices]

        return coeff_mat.astype(np.float32, copy=False)

    def _get_local_observable_features(self, x_proc: torch.Tensor) -> np.ndarray:
        """Selected local observable expectations for the configured repr_mode."""
        statevectors = self._get_statevector(x_proc)
        observables = _observable_matrices_for_repr(self.repr_mode, self.args.n_qubits)
        observable_tensor = torch.tensor(np.stack(observables), dtype=torch.complex64)
        state_tensor = torch.from_numpy(statevectors)
        return state_to_selected_observable_features_torch(
            state_tensor,
            observable_tensor,
        ).cpu().numpy().astype(np.float32, copy=False)


def _make_poison_model(
    nums: tuple,
    poison_x: float,
    poison_y: float,
    repr_mode: str,
    device: torch.device,
    weight_dir: str = 'poison/weights',
    concept_layer: int = -1,
    pullback_target_mode: str = "z_sum",
    pullback_sparse_top_n: int = 0,
) -> PoisonQNetWrapper:
    """Instantiate and load a poisoned model, returning a concept-pipeline wrapper."""
    task = {
        'task': 'MNIST_4',
        'option': 'mix_reg',
        'n_qubits': 6,
        'n_layers': 4,
        'fold': 1,
        'backend': 'tq',
    }
    pargs = PoisonArguments(**task)
    pargs.digits_of_interest = list(nums)

    n_layers = pargs.n_layers
    n_qubits = int(pargs.n_qubits / pargs.fold)    
    single = [[i]+[1]*2*n_layers for i in range(1,n_qubits+1)]
    enta = [[i]+[i+1]*n_layers for i in range(1,n_qubits)]+[[n_qubits]+[1]*n_layers]
    arch_code = [pargs.n_qubits, pargs.n_layers]
    pdesign   = poison_single_enta_to_design(single, enta, arch_code, pargs.fold)

    base_model = PoisonQNet(pargs, pdesign).to(device)

    weight_path = os.path.join(
        PROJECT_ROOT, weight_dir,
        f'{nums}_poison_({poison_x:.1f}, {poison_y:.1f})',
    )
    if os.path.isfile(weight_path):
        base_model.load_state_dict(torch.load(weight_path, map_location=device), strict=False)
        logging.info(f"Loaded poison weights: {weight_path}")
    else:
        logging.warning(f"Training a new model on {nums}.")
        dataloader = MNISTDataLoaders(args, task['task'])
        acc_train, acc_test = Scheme(pdesign, task, 'init', 30, nums, verbs=True, save=True,dataloader=dataloader)   

    wrapper = PoisonQNetWrapper(
        base_model,
        repr_mode=repr_mode,
        concept_layer=concept_layer,
        pullback_target_mode=pullback_target_mode,
        pullback_sparse_top_n=pullback_sparse_top_n,
    )
    wrapper.to(device)
    wrapper.eval()
    return wrapper

# ──────────────────────────────────────────────────────────────────────────────

concept_to_class = {
    "Loop": [0, 2, 6, 8, 9],
    "Vertical Line": [1, 4, 7],
    "Horizontal Line": [4, 5, 7],
    "Curvature": [0, 2, 3, 5, 6, 8, 9],
}


torch.manual_seed(42)
np.random.seed(42)


def concept_accuracy(
    model: PoisonQNetWrapper,
    random_seed: int,
    plot: bool,
    concept_bank: dict,
    eval_loader: torch.utils.data.DataLoader,
    save_dir: Path = Path.cwd() / "results/mnist_tq/concept_accuracy",
    data_dir: Path = Path.cwd() / "data/mnist",
) -> tuple[float, float, dict[str, dict[str, float]]]:
    device = torch.device("cpu")
    torch.manual_seed(random_seed)
    if not save_dir.exists():
        os.makedirs(save_dir)
    model.eval()

    if concept_bank is None:
        raise ValueError("concept_bank must be provided to evaluate concept accuracy.")

    concept_map = concept_bank["active_concept_to_class"]
    car_classifiers = concept_bank["car_classifiers"]
    cav_classifiers = concept_bank["cav_classifiers"]
    concept_to_car = {
        concept_name: car
        for concept_name, car in zip(concept_map, car_classifiers)
    }
    concept_to_cav = {
        concept_name: cav
        for concept_name, cav in zip(concept_map, cav_classifiers)
    }

    tcar_scores = []
    tcav_scores = []
    concept_scores: dict[str, dict[str, float]] = {}
    for concept_name in concept_map:
        logging.info(f"Working with concept {concept_name} and seed {random_seed}")
        pullback_uniform_indices = None

        X_test, y_test = generate_mnist_concept_dataset(
            concept_map[concept_name], eval_loader, 50, random_seed
        )
        if model.repr_mode == "pullback":
            H_test = _extract_pullback_concept_features(
                model,
                X_test,
                uniform_top_indices=pullback_uniform_indices,
            )
        else:
            H_test = model.input_to_representation(torch.from_numpy(X_test).to(device))
            if np.iscomplexobj(H_test):
                H_test = np.array([dm2vec(h) for h in H_test])

        tcar_score = float(accuracy_score(y_test, concept_to_car[concept_name].predict(H_test)))
        tcav_score = float(accuracy_score(y_test, concept_to_cav[concept_name].predict(H_test)))
        tcar_scores.append(tcar_score)
        tcav_scores.append(tcav_score)
        concept_scores[concept_name] = {
            "TCAR": tcar_score,
            "TCAV": tcav_score,
        }

    mean_tcar = float(np.mean(tcar_scores)) if len(tcar_scores) else float("nan")
    mean_tcav = float(np.mean(tcav_scores)) if len(tcav_scores) else float("nan")
    return mean_tcar, mean_tcav, concept_scores


def _sample_balanced_indices(y_train: np.ndarray, n: int) -> np.ndarray:
    """Return shuffled indices for n positive and n negative samples."""
    labels = np.asarray(y_train, dtype=np.int64)
    positive_indices = np.flatnonzero(labels == 1)
    negative_indices = np.flatnonzero(labels == 0)
    if len(positive_indices) < n or len(negative_indices) < n:
        raise ValueError(
            f"Need {n} samples per class, but found "
            f"{len(positive_indices)} positive and {len(negative_indices)} negative."
        )

    selected_indices = np.concatenate(
        [
            np.random.choice(positive_indices, size=n, replace=False),
            np.random.choice(negative_indices, size=n, replace=False),
        ]
    )
    np.random.shuffle(selected_indices)
    return selected_indices


def _extract_pullback_concept_features(
    model: "PoisonQNetWrapper",
    x_input: np.ndarray,
    uniform_top_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Extract pullback features after selecting top observable indices.

    Returns
    -------
    np.ndarray
        Feature matrix with per-sample selected pullback observable expectations.
    """
    top_n = int(getattr(model, "pullback_sparse_top_n", 0))
    target_mode = getattr(model, "pullback_target_mode", "z_sum")
    use_uniform_set = bool(getattr(model.args, "pullback_use_uniform_set", False))
    if use_uniform_set:
        if uniform_top_indices is None:
            aggregate = getattr(model.args, "pullback_aggregate", "mean_abs")
            top_indices, _ = _select_top_pullback_observables_over_set(
                model,
                x_input,
                top_n=top_n,
                target_mode=target_mode,
                aggregate=aggregate,
            )

    top_indices = _select_top_pullback_observables_per_sample(
        model,
        x_input,
        top_n=top_n,
        target_mode=target_mode,
    )

    # Map selected indices to pullback observable names and matrices.
    pullback_names = list(model._pullback_observable_names)
    pullback_observables = list(model._pullback_observables)
    selected_observable_names = [
        [pullback_names[int(idx)] for idx in row]
        for row in np.asarray(top_indices)
    ]
    selected_observable_matrices = [
        [pullback_observables[int(idx)] for idx in row]
        for row in np.asarray(top_indices)
    ]

    # Build states at the configured concept layer and evaluate selected observables.
    x_arr = np.asarray(x_input, dtype=np.float32)
    if x_arr.ndim == 3:
        x_arr = np.expand_dims(x_arr, axis=1)
    x_proc = model.preprocess(torch.from_numpy(x_arr))

    n_qubits = model.args.n_qubits
    dim = 1 << n_qubits
    zero_state = np.zeros(dim, dtype=np.complex64)
    zero_state[0] = 1.0

    feature_rows = []
    for b_idx in range(x_proc.shape[0]):
        x_np = x_proc[b_idx].detach().cpu().numpy()
        u = model._build_full_unitary(x_np)

        state_vec = u @ zero_state
        state_t = torch.from_numpy(state_vec[None, :]).to(dtype=torch.complex64)
        obs_t = torch.tensor(
            np.stack(selected_observable_matrices[b_idx]),
            dtype=torch.complex64,
        )
        feat = state_to_selected_observable_features_torch(state_t, obs_t).cpu().numpy()[0]
        feature_rows.append(feat.astype(np.float32, copy=False))

    selected_features = np.stack(feature_rows, axis=0)
    return selected_features


def _select_pullback_observable_names_per_sample(
    model: "PoisonQNetWrapper",
    x_input: np.ndarray,
) -> tuple[list[list[str]], np.ndarray]:
    """Return sample-dependent top pullback observable names and indices."""
    top_n = int(getattr(model, "pullback_sparse_top_n", 0))
    target_mode = getattr(model, "pullback_target_mode", "z_sum")
    top_indices = _select_top_pullback_observables_per_sample(
        model,
        x_input,
        top_n=top_n,
        target_mode=target_mode,
    )
    pullback_names = list(model._pullback_observable_names)
    selected_names = [
        [pullback_names[int(idx)] for idx in row]
        for row in np.asarray(top_indices)
    ]
    return selected_names, np.asarray(top_indices)


def _build_global_explanation_bank(
    model: PoisonQNetWrapper,
    random_seed: int,
    data_dir: Path,
    train_loader: torch.utils.data.DataLoader,
) -> dict:
    """Fit CAR/CAV once on the reference model and reuse them across poisoned checkpoints."""
    device = torch.device("cpu")
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    pullback_uniform_indices = None
    car_classifiers = []
    cav_classifiers = []

    for concept_name in active_concept_to_class:
        logging.info(f"Fitting classifiers for {concept_name} on seed {random_seed}")

        n_selected = 200        
        h_train, y_concept = generate_mnist_concept_dataset(
            active_concept_to_class[concept_name], train_loader, n_selected, random_seed
        )
        logging.info(
            f"{concept_name}: fitting with {n_selected} positive and "
            f"{n_selected} negative samples"
        )

        if model.repr_mode == "pullback":
            h_concept = _extract_pullback_concept_features(
                model,
                h_train,
                uniform_top_indices=None,
            )
        else:
            H_train = model.input_to_representation(torch.from_numpy(h_train).to(device))
            h_concept = (
                np.array([dm2vec(dm) for dm in H_train])
                if np.iscomplexobj(H_train)
                else H_train
            )
        car_classifier = CAR(device)
        cav_classifier = CAV(device)
        car_classifier.fit(h_concept, y_concept)
        cav_classifier.fit(h_concept, y_concept)
        car_classifiers.append(car_classifier)
        cav_classifiers.append(cav_classifier)

    return {
        "selected_digits": args.nums,
        "active_concept_to_class": active_concept_to_class,
        "car_classifiers": car_classifiers,
        "cav_classifiers": cav_classifiers,
        "pullback_uniform_indices": pullback_uniform_indices,
    }


def global_explanations(
    model: PoisonQNetWrapper,
    random_seed: int,
    batch_size: int,
    plot: bool,
    save_dir: Path = Path.cwd() / "results/mnist_tq/global_explanations",
    data_dir: Path = Path.cwd() / "data/mnist",
    concept_bank: dict | None = None,
) -> tuple[float, float, dict[int, float], dict[int, float]]:
    device = torch.device("cpu")
    torch.manual_seed(random_seed)
    if not save_dir.exists():
        os.makedirs(save_dir)
    model.eval()

    if concept_bank is None:
        concept_bank = _build_global_explanation_bank(model, random_seed, data_dir, train_loader)

    selected_digits = args.eval_digits
    active_concept_to_class = concept_bank["active_concept_to_class"]
    car_classifiers = concept_bank["car_classifiers"]
    cav_classifiers = concept_bank["cav_classifiers"]
    pullback_uniform_indices = concept_bank.get("pullback_uniform_indices")

    results_data = []    

    args = model.args
    args.digits_of_interest = selected_digits
    _, _, test_loader = MNISTDataLoaders(args, args.task)

    # just for debugging, limit to 1000 samples for now
    from torch.utils.data import Subset    
    test_loader = DataLoader(
        Subset(test_loader.dataset, range(1000)),
        batch_size=256,
        shuffle=False,
        pin_memory=getattr(test_loader, "pin_memory", False),
        num_workers=getattr(test_loader, "num_workers", 0),
        drop_last=False,
    )    
    
    test_size = len(test_loader.dataset)
    logging.info(f"Global explanations use poison MNISTDataLoaders over digits: {tuple(selected_digits)}")

    if model.repr_mode == "statevector":
        raise NotImplementedError(
            "global_explanations currently supports pullback and local observable representations only."
        )

    csv_features = None
    y_test_from_csv = None
    feature_csv_path = _observable_feature_csv_path(save_dir, model.concept_layer)    

    logging.info("Now predicting concepts on the test set")
    feature_offset = 0
    for feed_dict in tqdm(test_loader, unit="batch", leave=False):
        X_test = feed_dict["image"]
        y_test = feed_dict["digit"]
        if model.repr_mode == "pullback":
            
            if feature_csv_path.exists():
                batch_count = len(y_test)
                # batch_count = 2
                names_per_sample, top_indices = _select_pullback_observable_names_per_sample(
                model,
                X_test.detach().cpu().numpy(),
            )
                batch_df = pd.read_csv(feature_csv_path, skiprows=range(1, feature_offset+1), nrows=batch_count)
                
                batch_features = []
                for row_idx, indices in enumerate(top_indices):
                    row_values = [float(batch_df.iloc[row_idx][name]) for name in names_per_sample[row_idx]]
                    batch_features.append(row_values)
                h_test = np.asarray(batch_features, dtype=np.float32)
                feature_offset += batch_count
            else:
                h_test = _extract_pullback_concept_features(
                    model,
                    X_test.detach().cpu().numpy(),
                    uniform_top_indices=pullback_uniform_indices,
                )
        
        else:            
            repr_column_names = _observable_column_names_for_repr(model.repr_mode, model.args.n_qubits)
            required_columns = ["Class"] + repr_column_names
            if feature_csv_path.exists():
                csv_features, y_test_np = _load_repr_from_observable_csv(
                    feature_csv_path,
                    model.repr_mode,
                    model.args.n_qubits,
                )                
                h_test = np.asarray(csv_features, dtype=np.float32)

        y_test_np = y_test.detach().cpu().numpy().astype(np.int64)
        y_test_digits = _restore_mnist_loader_labels(y_test_np, selected_digits)

        car_preds = []
        cav_preds = []
        for concept_name, car, cav in zip(
            active_concept_to_class, car_classifiers, cav_classifiers
        ):
            car_preds.append(car.predict(h_test))
            cav_preds.append(cav.predict(h_test))
        targets = [
            [int(label in active_concept_to_class[concept]) for label in y_test_digits]
            for concept in active_concept_to_class
        ]

        results_data += [
            ["TCAR", int(label)] + [int(car_pred[idx]) for car_pred in car_preds]
            for idx, label in enumerate(y_test_digits)
        ]
        results_data += [
            ["TCAV", int(label)] + [int(cav_pred[idx] > 0) for cav_pred in cav_preds]
            for idx, label in enumerate(y_test_digits)
        ]
        results_data += [
            ["True Prop.", int(label)] + [target[idx] for target in targets]
            for idx, label in enumerate(y_test_digits)
        ]

    results_df = pd.DataFrame(
        results_data, columns=["Method", "Class"] + list(active_concept_to_class.keys())
    )

    tcar_corr, tcav_corr, tcar_corr_per_class, tcav_corr_per_class = _compute_global_explanation_correlations(
        results_df,
        list(active_concept_to_class.keys()),
    )

    summary_rows = [{
        "scope": "global",
        "class": "all",
        "tcar_corr": float(tcar_corr),
        "tcav_corr": float(tcav_corr),
    }]
    class_keys = sorted(set(tcar_corr_per_class.keys()) | set(tcav_corr_per_class.keys()))
    for class_idx in class_keys:
        summary_rows.append({
            "scope": "per_class",
            "class": int(class_idx),
            "tcar_corr": float(tcar_corr_per_class.get(class_idx, float("nan"))),
            "tcav_corr": float(tcav_corr_per_class.get(class_idx, float("nan"))),
        })

    csv_path = save_dir / "metrics.csv"
    pd.DataFrame(summary_rows).to_csv(csv_path, index=False)

    logging.info(
        f"Run correlations | TCAR={tcar_corr:.4f} TCAV={tcav_corr:.4f}"
    )
    if plot:
        plot_global_explanation(save_dir, "mnist")
    return tcar_corr, tcav_corr, tcar_corr_per_class, tcav_corr_per_class


def feature_importance(
    model: PoisonQNetWrapper,
    random_seed: int,
    batch_size: int,
    plot: bool,
    save_dir: Path = Path.cwd() / "results/mnist_tq/feature_importance",
    data_dir: Path = Path.cwd() / "data/mnist",
) -> None:
    device = torch.device("cpu")
    torch.manual_seed(random_seed)
    if not save_dir.exists():
        os.makedirs(save_dir)
    model.eval()

    # Fit a concept classifier and compute feature importance for each concept
    car_classifiers = [CAR(device) for _ in active_concept_to_class]
    mnist_train_set = MNIST(data_dir, train=True, download=True)
    mnist_train_set.transform = transforms.Compose([transforms.ToTensor()])
    mnist_train_loader = DataLoader(
        mnist_train_set,
        batch_size=min(1024, len(mnist_train_set)),
        shuffle=False,
    )
    test_set = MNIST(data_dir, train=False, download=True)
    test_set.transform = transforms.Compose([transforms.ToTensor()])
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False
    )
    attribution_dic = {}
    baselines = torch.zeros((1, 1, 28, 28)).to(device)
    for concept_name, car in zip(active_concept_to_class, car_classifiers):
        logging.info(f"Now fitting CAR classifier for {concept_name}")
        X_train, y_train = generate_mnist_concept_dataset(
            active_concept_to_class[concept_name], mnist_train_loader, 200, random_seed
        )
        if model.repr_mode == "pullback":
            h_train = _extract_pullback_concept_features(
                model,
                X_train,
                uniform_top_indices=None,
            )
        else:
            H_train = (
                model.input_to_representation(torch.from_numpy(X_train).to(device))
            )
            h_train = np.array([dm2vec(dm) for dm in H_train]) if np.iscomplexobj(H_train) else H_train
        car.tune_kernel_width(h_train, y_train)
        logging.info(
            f"Now computing feature importance on the test set for {concept_name}"
        )
        concept_attribution_method = CARFeatureImportance(
            "Integrated Gradient", car, model, device
        )
        attribution_dic[concept_name] = concept_attribution_method.attribute(
            test_loader, baselines=baselines
        )
        if plot:
            logging.info(f"Saving plots in {save_dir} for {concept_name}")
            X_test = test_set.data
            plot_idx = [
                torch.nonzero(test_set.targets == (n % 10))[n // 10].item()
                for n in range(100)
            ]
            for set_id in range(1, 5):
                plot_grayscale_saliency(
                    X_test,
                    attribution_dic[concept_name],
                    plot_idx[set_id * 10 : (set_id + 1) * 10],
                    save_dir,
                    f"mnist_set{set_id}",
                    concept_name.lower().replace(" ", "-"),
                )
    logging.info(f"Now computing vanilla feature importance")
    vanilla_attribution_method = VanillaFeatureImportance(
        "Integrated Gradient", model, device
    )
    attribution_dic["Vanilla"] = vanilla_attribution_method.attribute(
        test_loader, baselines=baselines
    )
    np.savez(save_dir / "attributions.npz", **attribution_dic)
    if plot:
        logging.info(f"Saving plots in {save_dir}")
        plot_attribution_correlation(save_dir, "mnist")
        X_test = test_set.data
        plot_idx = [
            torch.nonzero(test_set.targets == (n % 10))[n // 10].item()
            for n in range(100)
        ]
        for set_id in range(1, 5):
            plot_grayscale_saliency(
                X_test,
                attribution_dic["Vanilla"],
                plot_idx[set_id * 10 : (set_id + 1) * 10],
                save_dir,
                f"mnist_set{set_id}",
                "vanilla",
            )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    # Direct in-code run configuration (no CLI required for routine edits).
    args = argparse.Namespace()
    args.seeds = [1, 2, 3, 4, 5]
    args.batch_size = 120
    args.plot = False
    args.pullback_top_n = 20
    args.pullback_use_uniform_set = False
    args.pullback_sample_top_n = 12
    args.pullback_selection_per_class = 1000
    args.pullback_aggregate = "mean_abs"
    args.pullback_target = "z_sum"
    args.concept_layer = 2
    args.run_all_poison_steps = True
    args.poison_alphas = [round(a, 1) for a in np.arange(0.0, 1.0, 0.1).tolist()]
    args.poison_axis = "both"
    args.repr = "pullback"  # Options: "statevector", "pullback", "local_x", "local_y", "local_z", "local_zz"
    args.poison_x = 0.0
    args.poison_y = 0.0
    args.nums = [0, 1, 2, 3, 4, 5]
    args.eval_digits = [6, 7, 8, 9]

    args.weight_dir = "weights"
    args.task = "global_explanations"
    args.train_valid_split_ratio = [0.9, 0.1]
    args.center_crop = 24
    args.resize = 24
    args.digits_of_interest = list(args.nums)

    args.n_train_samples = 12000
    args.n_valid_samples = 1200
    args.n_test_samples = 1200
    args.same_n_samples_each_class = True
    nums   = tuple(args.nums)
    device = torch.device("cpu")
    

    axes = [("x", 1.0, 0.0), ("y", 0.0, 1.0)]
    if args.poison_axis == "x":
        axes = [("x", 1.0, 0.0)]
    elif args.poison_axis == "y":
        axes = [("y", 0.0, 1.0)]

    runs = []
    if args.run_all_poison_steps:
        for alpha in args.poison_alphas:
            for axis_name, x_mult, y_mult in axes:
                runs.append((axis_name, float(alpha) * x_mult, float(alpha) * y_mult))
    else:
        runs.append(("single", args.poison_x, args.poison_y))    

    clean_model = _make_poison_model(
        nums,
        0.0,
        0.0,
        args.repr,
        device,
        args.weight_dir,
        concept_layer=args.concept_layer,
        pullback_target_mode=args.pullback_target,
        pullback_sparse_top_n=args.pullback_sample_top_n,
    )    

    # Preload the fixed filtered test split once via MNISTDataLoaders.
    train_loader, valid_loader, test_loader = MNISTDataLoaders(args, args.task)
    feature_feed = next(iter(test_loader))
    x_feature_input = feature_feed["image"]
    y_feature_input = feature_feed["digit"].cpu().numpy()

    correlation_records = []
    pair_label = f"({nums[0]}, {nums[1]})"
    
    active_concept_to_class = {}
    allowed_digits = set(args.digits_of_interest)
    for concept_name, class_ids in concept_to_class.items():
        filtered_ids = [int(c) for c in class_ids if int(c) in allowed_digits]
        if filtered_ids:
            active_concept_to_class[concept_name] = filtered_ids

    seed_quality_rows = []
    best_seed = None
    best_tcar = -np.inf
    best_concept_bank = None

    for seed in args.seeds:
        seed_concept_bank = _build_global_explanation_bank(
            clean_model,
            seed,
            Path.cwd() / "data" / "mnist",
            train_loader
        )

        save_dir = Path.cwd() / "results" / "mnist_tq" / args.task
        mean_tcar, mean_tcav, concept_scores = concept_accuracy(
            clean_model,
            seed,
            args.plot,
            concept_bank=seed_concept_bank,
            eval_loader=test_loader,
            save_dir=save_dir,
        )
        quality_row = {
            "Seed": int(seed),
            "Mean TCAR": float(mean_tcar),
            "Mean TCAV": float(mean_tcav),
        }
        for concept_name, method_scores in concept_scores.items():
            quality_row[f"{concept_name} TCAR"] = float(method_scores["TCAR"])
            quality_row[f"{concept_name} TCAV"] = float(method_scores["TCAV"])
        seed_quality_rows.append(quality_row)
        if np.isfinite(mean_tcar) and mean_tcar > best_tcar:
            best_tcar = float(mean_tcar)
            best_seed = int(seed)
            best_concept_bank = seed_concept_bank

    concept_bank = best_concept_bank
    if concept_bank is None:
        raise RuntimeError("Failed to select a concept bank based on TCAR mean performance.")

    seed_quality_df = pd.DataFrame(seed_quality_rows)
    quality_csv = Path.cwd() / "results" / "mnist_tq" / args.task / "concept_bank_seed_quality.csv"
    quality_csv.parent.mkdir(parents=True, exist_ok=True)
    seed_quality_df.to_csv(quality_csv, index=False)
    logging.info(
        f"Selected seed {best_seed} as final concept bank based on Mean TCAR={best_tcar:.4f}. "
        f"Saved seed quality table: {quality_csv}"
    )
        


    # for axis_name, poison_x, poison_y in runs:
    #     model = _make_poison_model(
    #         nums,
    #         poison_x,
    #         poison_y,
    #         args.repr,
    #         device,
    #         args.weight_dir,
    #         concept_layer=args.concept_layer,
    #         pullback_target_mode=args.pullback_target,
    #         pullback_sparse_top_n=args.pullback_sample_top_n,
    #     )
    #     if args.repr == "pullback" and args.pullback_use_uniform_set:
    #         model.set_pullback_observables(clean_model._pullback_observable_indices)
    #     logging.info(
    #         f"Model ready | nums={nums}  poison_x={poison_x}  "
    #         f"poison_y={poison_y}  repr={args.repr}  concept_layer={args.concept_layer}"
    #     )

    #     if args.run_all_poison_steps:
    #         run_tag = (
    #             f"axis_{axis_name}_x{poison_x:.1f}_y{poison_y:.1f}"
    #             f"_layer_{args.concept_layer}"
    #         )
    #         save_dir = Path.cwd() / "results" / "mnist_tq" / args.task / run_tag
    #     else:
    #         run_tag = f"single_layer_{args.concept_layer}"
    #         save_dir = Path.cwd() / "results" / "mnist_tq" / args.task

        
    #     # concept_accuracy(model, args.seeds, args.plot, save_dir=save_dir)
    #     save_dir = Path.cwd() / "results" / "poison" / 'global_explanations' / run_tag
    #     tcar_corr, tcav_corr, tcar_corr_per_class, tcav_corr_per_class = global_explanations(
    #         model,
    #         args.seeds[0],
    #         args.batch_size,
    #         plot=False,
    #         save_dir=save_dir,
    #         concept_bank=concept_bank,
    #     )

    #     correlation_records.append({
    #         "pair": pair_label,
    #         "axis": axis_name,
    #         "x_alpha": float(poison_x),
    #         "y_alpha": float(poison_y),
    #         "tcar_corr": float(tcar_corr),
    #         "tcav_corr": float(tcav_corr),
    #     })

        # model.repr_mode = "statevector"      #save_all_features requires statevector representation
        # save_all_features(
        #     model,
        #     x_input=x_feature_input,
        #     y_input=y_feature_input,
        #     batch_size=args.batch_size,
        #     save_csv_path=save_dir / "observable_features.csv",
        # )
        
        # feature_importance(model,args.seeds[0],args.batch_size,args.plot,save_dir=save_dir)

    # corr_save_root = Path.cwd() / "results" / "poison" / "global_explanations"
    # corr_df = pd.DataFrame(correlation_records)
    # corr_csv = corr_save_root / "correlation_runs.csv"
    # corr_save_root.mkdir(parents=True, exist_ok=True)
    # corr_df.to_csv(corr_csv, index=False)
    # logging.info(f"Saved per-run correlation table: {corr_csv}")
    # plot_poison_correlation_curves(correlation_records, corr_save_root)