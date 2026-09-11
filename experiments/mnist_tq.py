import itertools
import logging
import argparse
import torch
import numpy as np
import os, sys
import pandas as pd
from pathlib import Path

# Ensure project root (containing `utils`, `models`, etc.) is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.mnist import init_trainer, get_dataloader
from models.FusionModel_mnist import QNet, single_enta_to_design
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.hooks import register_hooks, get_saved_representations, remove_all_hooks
from utils.dataset import generate_mnist_concept_dataset
from utils.plot import (
    plot_concept_accuracy,
    plot_global_explanation,
    plot_grayscale_saliency,
    plot_attribution_correlation,
    plot_kernel_sensitivity,
    plot_concept_size_impact,
    plot_tcar_inter_concepts,
)
from explanations.concept import CAR, CAV
from explanations.feature import CARFeatureImportance, VanillaFeatureImportance
from sklearn.metrics import accuracy_score
from sklearn.gaussian_process.kernels import Matern
from tqdm import tqdm
from utils.robustness import Attacker

_LOCAL_PAULI_CACHE = {}
_LOCAL_PAULI_TORCH_CACHE = {}
_XYZ_ZZ_OBSERVABLE_CACHE = {}


def _single_qubit_paulis():
    x = np.array([[0, 1], [1, 0]], dtype=np.complex64)
    y = np.array([[0, -1j], [1j, 0]], dtype=np.complex64)
    z = np.array([[1, 0], [0, -1]], dtype=np.complex64)
    i = np.eye(2, dtype=np.complex64)
    return x, y, z, i


def _build_local_xyz_observables(n_qubits):
    if n_qubits in _LOCAL_PAULI_CACHE:
        return _LOCAL_PAULI_CACHE[n_qubits]

    x, y, z, i = _single_qubit_paulis()
    observables = []
    for op in (x, y, z):
        for q in range(n_qubits):
            mat = np.array([[1]], dtype=np.complex64)
            for idx in range(n_qubits):
                mat = np.kron(mat, op if idx == q else i)
            observables.append(mat)
    _LOCAL_PAULI_CACHE[n_qubits] = observables
    return observables


def dm2vec(rho):
    """
    把 单个量子态转换为局域 X/Y/Z 可观测量期望特征。
    输入：rho (statevector 或 density matrix)
    输出：实数一维向量 [<X_1>...<X_n>, <Y_1>...<Y_n>, <Z_1>...<Z_n>]
    """
    # 转成 numpy 复数数组
    if hasattr(rho, 'data'):
        rho = rho.data
    rho = np.asarray(rho)

    if rho.ndim == 1:
        dim = rho.shape[0]
        n_qubits = int(np.log2(dim))
        if (1 << n_qubits) != dim:
            raise ValueError(f"Invalid statevector dimension: {dim}")
        density = np.outer(rho, np.conjugate(rho))
    elif rho.ndim == 2:
        if rho.shape[0] != rho.shape[1]:
            raise ValueError(f"Density matrix must be square, got shape {rho.shape}")
        dim = rho.shape[0]
        n_qubits = int(np.log2(dim))
        if (1 << n_qubits) != dim:
            raise ValueError(f"Invalid density-matrix dimension: {rho.shape}")
        density = rho
    else:
        raise ValueError(f"Unsupported quantum state shape: {rho.shape}")

    observables = _build_local_xyz_observables(n_qubits)
    feature = np.array([
        np.real(np.trace(density @ obs)) for obs in observables
    ], dtype=np.float32)

    # 去掉极小值（避免数值噪声）
    feature = np.nan_to_num(feature, nan=0, posinf=0, neginf=0)
    return feature


def _build_local_xyz_observables_torch(n_qubits: int, device: torch.device) -> torch.Tensor:
    cache_key = (n_qubits, str(device))
    if cache_key in _LOCAL_PAULI_TORCH_CACHE:
        return _LOCAL_PAULI_TORCH_CACHE[cache_key]

    observables_np = _build_local_xyz_observables(n_qubits)
    observables = torch.tensor(np.stack(observables_np), dtype=torch.complex64, device=device)
    _LOCAL_PAULI_TORCH_CACHE[cache_key] = observables
    return observables


def state_to_observable_features_torch(state: torch.Tensor) -> torch.Tensor:
    """Map batch quantum states to local X/Y/Z observable expectations."""
    if state.ndim == 2:
        dim = state.shape[-1]
        n_qubits = int(np.log2(dim))
        if (1 << n_qubits) != dim:
            raise ValueError(f"Invalid statevector dimension: {dim}")
        density = torch.einsum("bi,bj->bij", state, torch.conj(state))
    elif state.ndim == 3:
        if state.shape[-1] != state.shape[-2]:
            raise ValueError(f"Density matrix must be square, got shape {tuple(state.shape)}")
        dim = state.shape[-1]
        n_qubits = int(np.log2(dim))
        if (1 << n_qubits) != dim:
            raise ValueError(f"Invalid density-matrix dimension: {tuple(state.shape)}")
        density = state
    else:
        raise ValueError(f"Unsupported quantum state shape: {tuple(state.shape)}")

    observables = _build_local_xyz_observables_torch(n_qubits, density.device)
    features = torch.real(torch.einsum("bij,mji->bm", density, observables))
    return torch.nan_to_num(features.float(), nan=0.0, posinf=0.0, neginf=0.0)


def state_to_selected_observable_features_torch(
    state: torch.Tensor,
    observables: torch.Tensor,
) -> torch.Tensor:
    """Map batch quantum states to expectation values of a selected observable set."""
    if state.ndim == 2:
        density = torch.einsum("bi,bj->bij", state, torch.conj(state))
    elif state.ndim == 3:
        if state.shape[-1] != state.shape[-2]:
            raise ValueError(f"Density matrix must be square, got shape {tuple(state.shape)}")
        density = state
    else:
        raise ValueError(f"Unsupported quantum state shape: {tuple(state.shape)}")

    features = torch.real(torch.einsum("bij,mji->bm", density, observables))
    return torch.nan_to_num(features.float(), nan=0.0, posinf=0.0, neginf=0.0)


def _build_xyz_zz_observables(n_qubits: int):
    """Build fixed observable family [X_i, Y_i, Z_i, XX_ij, YY_ij, ZZ_ij] with stable ordering."""
    if n_qubits in _XYZ_ZZ_OBSERVABLE_CACHE:
        return _XYZ_ZZ_OBSERVABLE_CACHE[n_qubits]

    x, y, z, i = _single_qubit_paulis()
    observables = []
    names = []

    for op_name, op in (("X", x), ("Y", y), ("Z", z)):
        for q in range(n_qubits):
            mat = np.array([[1]], dtype=np.complex64)
            for idx in range(n_qubits):
                mat = np.kron(mat, op if idx == q else i)
            observables.append(mat)
            names.append(f"{op_name}{q}")

    for i_idx in range(n_qubits):
        for j_idx in range(i_idx + 1, n_qubits):
            mat_xx = np.array([[1]], dtype=np.complex64)
            mat_yy = np.array([[1]], dtype=np.complex64)
            mat_zz = np.array([[1]], dtype=np.complex64)
            for idx in range(n_qubits):
                if idx == i_idx or idx == j_idx:
                    mat_xx = np.kron(mat_xx, x)
                    mat_yy = np.kron(mat_yy, y)
                    mat_zz = np.kron(mat_zz, z)
                else:
                    mat_xx = np.kron(mat_xx, i)
                    mat_yy = np.kron(mat_yy, i)
                    mat_zz = np.kron(mat_zz, i)
            observables.append(mat_xx)
            names.append(f"X{i_idx}X{j_idx}")
            observables.append(mat_yy)
            names.append(f"Y{i_idx}Y{j_idx}")
            observables.append(mat_zz)
            names.append(f"Z{i_idx}Z{j_idx}")

    _XYZ_ZZ_OBSERVABLE_CACHE[n_qubits] = (observables, names)
    return observables, names


def build_target_observable(n_qubits: int, mode: str = "z_sum") -> np.ndarray:
    """Build a target observable in the same Hilbert space as the pullback basis."""
    observables, names = _build_xyz_zz_observables(n_qubits)
    name_to_op = {name: op for name, op in zip(names, observables)}

    if mode == "z_sum":
        selected = [name_to_op[f"Z{idx}"] for idx in range(n_qubits)]
    elif mode == "zz_sum":
        selected = [
            name_to_op[f"Z{i}Z{j}"]
            for i in range(n_qubits)
            for j in range(i + 1, n_qubits)
        ]
    elif mode in name_to_op:
        selected = [name_to_op[mode]]
    else:
        raise ValueError(
            f"Unsupported target observable mode: {mode}. "
            f"Use one of {{'z_sum', 'zz_sum'}} or an explicit basis name."
        )

    target = np.sum(selected, axis=0).astype(np.complex64)
    return target / max(len(selected), 1)


def _int_to_bits(num: int, n_qubits: int) -> list[int]:
    return [(num >> (n_qubits - 1 - idx)) & 1 for idx in range(n_qubits)]


def _bits_to_int(bits: list[int]) -> int:
    out = 0
    for bit in bits:
        out = (out << 1) | int(bit)
    return out


def _u3_matrix(theta: float, phi: float, lam: float) -> np.ndarray:
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    return np.array(
        [
            [c, -np.exp(1j * lam) * s],
            [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c],
        ],
        dtype=np.complex64,
    )


def _ry_matrix(theta: float) -> np.ndarray:
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]], dtype=np.complex64)


def _rz_matrix(theta: float) -> np.ndarray:
    return np.array(
        [[np.exp(-1j * theta / 2.0), 0], [0, np.exp(1j * theta / 2.0)]],
        dtype=np.complex64,
    )


def _rx_matrix(theta: float) -> np.ndarray:
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex64)


def _data_upload_matrix(values_4: np.ndarray) -> np.ndarray:
    # Matches the sequential encoder ops in model.data_uploading(): ry, rz, rx, ry.
    return (
        _ry_matrix(float(values_4[3]))
        @ _rx_matrix(float(values_4[2]))
        @ _rz_matrix(float(values_4[1]))
        @ _ry_matrix(float(values_4[0]))
    )


def _cu3_matrix(theta: float, phi: float, lam: float) -> np.ndarray:
    u3 = _u3_matrix(theta, phi, lam)
    out = np.eye(4, dtype=np.complex64)
    out[2:, 2:] = u3
    return out


def _embed_single_qubit_gate(gate: np.ndarray, qubit: int, n_qubits: int) -> np.ndarray:
    dim = 1 << n_qubits
    out = np.zeros((dim, dim), dtype=np.complex64)
    for in_idx in range(dim):
        bits = _int_to_bits(in_idx, n_qubits)
        sub_in = bits[qubit]
        for sub_out in (0, 1):
            coeff = gate[sub_out, sub_in]
            if coeff == 0:
                continue
            out_bits = bits.copy()
            out_bits[qubit] = sub_out
            out_idx = _bits_to_int(out_bits)
            out[out_idx, in_idx] += coeff
    return out


def _embed_two_qubit_gate(
    gate: np.ndarray, qubit_a: int, qubit_b: int, n_qubits: int
) -> np.ndarray:
    dim = 1 << n_qubits
    out = np.zeros((dim, dim), dtype=np.complex64)
    for in_idx in range(dim):
        bits = _int_to_bits(in_idx, n_qubits)
        sub_in = (bits[qubit_a] << 1) | bits[qubit_b]
        for sub_out in range(4):
            coeff = gate[sub_out, sub_in]
            if coeff == 0:
                continue
            out_bits = bits.copy()
            out_bits[qubit_a] = (sub_out >> 1) & 1
            out_bits[qubit_b] = sub_out & 1
            out_idx = _bits_to_int(out_bits)
            out[out_idx, in_idx] += coeff
    return out


def build_remaining_unitary(model: QNet, x: torch.Tensor) -> np.ndarray:
    """Build the sample-dependent unitary of the remaining circuit (layer >= represent_n)."""
    x_proc = x
    if x_proc.shape[-1] == 28:
        x_proc = model.preprocess(x_proc)
    if x_proc.ndim != 3:
        raise ValueError(f"Expected x shape [bsz, n_qubits, 4], got {tuple(x_proc.shape)}")
    if x_proc.shape[0] != 1:
        raise ValueError("build_remaining_unitary expects a single sample (bsz=1)")

    n_qubits = model.args.n_qubits
    dim = 1 << n_qubits
    x_np = x_proc[0].detach().cpu().numpy()
    q_rot = model.q_params_rot.detach().cpu().numpy()
    q_enta = model.q_params_enta.detach().cpu().numpy()

    u_remain = np.eye(dim, dtype=np.complex64)
    for op_type, wires, layer in model.design:
        if layer < model.args.represent_n:
            continue

        if op_type == "U3":
            qubit = int(wires[0])
            theta, phi, lam = q_rot[layer][qubit]
            gate = _embed_single_qubit_gate(_u3_matrix(theta, phi, lam), qubit, n_qubits)
        elif op_type == "C(U3)":
            control = int(wires[0])
            target = int(wires[1])
            theta, phi, lam = q_enta[layer][control]
            gate = _embed_two_qubit_gate(
                _cu3_matrix(theta, phi, lam), control, target, n_qubits
            )
        elif op_type == "data":
            qubit = int(wires[0])
            gate = _embed_single_qubit_gate(_data_upload_matrix(x_np[qubit]), qubit, n_qubits)
        else:
            raise ValueError(f"Unsupported gate type in design: {op_type}")

        u_remain = gate @ u_remain

    return u_remain


def decompose_pullback_observable(
    model: QNet,
    x: torch.Tensor,
    target_observable: np.ndarray,
    observable_family: list[np.ndarray] | None = None,
) -> np.ndarray:
    """Compute coefficients c_j of O_pb = U^dag O U on a fixed observable family."""
    n_qubits = model.args.n_qubits
    if observable_family is None:
        observable_family, _ = _build_xyz_zz_observables(n_qubits)

    u_remain = build_remaining_unitary(model, x)
    o_pullback = u_remain.conj().T @ target_observable @ u_remain

    coeffs = []
    for basis_op in observable_family:
        denom = np.real(np.trace(basis_op.conj().T @ basis_op))
        coef = np.real(np.trace(basis_op.conj().T @ o_pullback)) / float(denom)
        coeffs.append(float(coef))
    return np.array(coeffs, dtype=np.float32)


def select_global_top_n_pullback_observables(
    model: QNet,
    x_train: torch.Tensor,
    target_observable: np.ndarray,
    top_n: int,
    aggregate: str = "mean_abs",
) -> dict:
    """Select a global top-n observable subset using training-set pullback coefficients."""
    if x_train.ndim == 3:
        x_train = x_train.unsqueeze(1)
    if x_train.ndim != 4:
        raise ValueError(f"Expected x_train shape [N, 1, H, W], got {tuple(x_train.shape)}")

    family, names = _build_xyz_zz_observables(model.args.n_qubits)
    coeff_mat = []
    for idx in range(x_train.shape[0]):
        x_single = x_train[idx : idx + 1]
        coeff = decompose_pullback_observable(
            model=model,
            x=x_single,
            target_observable=target_observable,
            observable_family=family,
        )
        coeff_mat.append(coeff)

    coeff_mat = np.stack(coeff_mat, axis=0)
    if aggregate == "mean_abs":
        scores = np.mean(np.abs(coeff_mat), axis=0)
    elif aggregate == "mean_square":
        scores = np.mean(coeff_mat ** 2, axis=0)
    elif aggregate == "abs_mean":
        scores = np.abs(np.mean(coeff_mat, axis=0))
    else:
        raise ValueError(f"Unsupported aggregate rule: {aggregate}")

    top_n = min(int(top_n), len(names))
    top_idx = np.argsort(-scores)[:top_n]
    return {
        "top_indices": top_idx,
        "top_names": [names[i] for i in top_idx],
        "top_scores": scores[top_idx],
        "all_names": names,
        "all_scores": scores,
        "coefficients": coeff_mat,
    }


def load_observable_set_csv(
    csv_path: Path,
    n_qubits: int,
    top_n: int,
) -> tuple[np.ndarray, list[str]]:
    """Load a shared observable selection from a CSV file."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Observable CSV not found: {csv_path}")

    cached_df = pd.read_csv(csv_path)
    required_cols = {"Rank", "Observable"}
    if not required_cols.issubset(set(cached_df.columns)):
        raise ValueError(
            f"Observable CSV must contain columns {required_cols}, got {list(cached_df.columns)}"
        )

    _, all_names = _build_xyz_zz_observables(n_qubits)
    name_to_idx = {name: idx for idx, name in enumerate(all_names)}

    obs_rows = cached_df.loc[:, ["Rank", "Observable"]].sort_values("Rank")
    obs_names = obs_rows["Observable"].tolist()[:top_n]
    invalid_names = [obs_name for obs_name in obs_names if obs_name not in name_to_idx]
    if invalid_names:
        raise ValueError(f"Observable CSV has unknown observables: {invalid_names}")

    obs_indices = np.array([name_to_idx[obs_name] for obs_name in obs_names], dtype=np.int64)
    return obs_indices, obs_names


def get_observables(
    random_seed: int,
    latent_dim: int,
    save_dir: Path = Path.cwd() / "results/mnist_tq/global_explanations",
    data_dir: Path = Path.cwd() / "data/mnist",
    model_dir: Path = Path.cwd() / "results/mnist_tq",
    model_name: str = "model",
    n_observables: int = 8,
    selection_sample_size: int = 200,
    pullback_target: str = "z_sum",
    pullback_aggregate: str = "mean_abs",
) -> Path:
    """Select and store one shared observable set from the generic MNIST training split."""
    device = torch.device("cpu")
    torch.manual_seed(random_seed)
    rng = np.random.default_rng(random_seed)

    if not save_dir.exists():
        os.makedirs(save_dir)

    model_dir = model_dir / model_name
    model = QNet(myargs, design)
    model.load_state_dict(torch.load(model_dir / "vqc_model.pt"), strict=False)
    model.to(device)
    model.eval()

    target_observable = build_target_observable(model.args.n_qubits, mode=pullback_target)
    train_set = MNIST(data_dir, train=True, download=True, transform=transforms.ToTensor())
    sample_size = min(int(selection_sample_size), len(train_set))
    sample_indices = rng.choice(len(train_set), size=sample_size, replace=False)
    x_train = np.stack([train_set[int(idx)][0].numpy() for idx in sample_indices], axis=0)

    logging.info(f"Selecting {n_observables} observables from {sample_size} generic MNIST train samples")
    selection = select_global_top_n_pullback_observables(
        model=model,
        x_train=torch.from_numpy(x_train),
        target_observable=target_observable,
        top_n=n_observables,
        aggregate=pullback_aggregate,
    )
    top_obs_records = [
        {
            "Rank": rank + 1,
            "Observable": obs_name,
            "Score": float(obs_score),
        }
        for rank, (obs_name, obs_score) in enumerate(
            zip(selection["top_names"], selection["top_scores"])
        )
    ]

    csv_path = save_dir / "pullback_top_observables.csv"
    pd.DataFrame(top_obs_records).to_csv(csv_path, index=False)
    logging.info(f"Saved observable selections to {csv_path}")
    return csv_path


def get_random_observables(
    random_seed: int,
    n_observables: int,
    save_dir: Path = Path.cwd() / "results/mnist_tq/global_explanations",
    n_qubits: int | None = None,
    csv_name: str = "random_top_observables.csv",
) -> Path:
    """Randomly sample n observables from the fixed family and store them to CSV."""
    if n_qubits is None:
        n_qubits = myargs.n_qubits

    if not save_dir.exists():
        os.makedirs(save_dir)

    _, observable_names = _build_xyz_zz_observables(n_qubits)
    n_select = min(int(n_observables), len(observable_names))
    rng = np.random.default_rng(random_seed)
    chosen_indices = rng.choice(len(observable_names), size=n_select, replace=False)

    rows = [
        {"Rank": rank + 1, "Observable": observable_names[int(obs_idx)]}
        for rank, obs_idx in enumerate(chosen_indices)
    ]

    csv_path = save_dir / csv_name
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    logging.info(f"Saved random observable set to {csv_path}")
    return csv_path


def pullback_feature_matrix(
    model: QNet,
    x_np: np.ndarray,
    target_observable: np.ndarray,
    top_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Legacy helper: compute pullback coefficient features for a dataset."""
    coeffs = []
    x_tensor = torch.from_numpy(x_np)
    for idx in range(x_tensor.shape[0]):
        c = decompose_pullback_observable(
            model=model,
            x=x_tensor[idx : idx + 1],
            target_observable=target_observable,
        )
        if top_indices is not None:
            c = c[top_indices]
        coeffs.append(c)
    return np.stack(coeffs, axis=0).astype(np.float32)


def observable_expectation_feature_matrix(
    model: QNet,
    x_np: np.ndarray,
    observable_indices: np.ndarray,
    observable_family: list[np.ndarray] | None = None,
) -> np.ndarray:
    """Compute selected observable expectations on representation states for a dataset."""
    if observable_family is None:
        observable_family, _ = _build_xyz_zz_observables(model.args.n_qubits)

    x_tensor = torch.from_numpy(x_np)
    state = model.input_to_representation(x_tensor)
    selected_ops = torch.tensor(
        np.stack([observable_family[idx] for idx in observable_indices]),
        dtype=torch.complex64,
        device=state.device,
    )
    features = state_to_selected_observable_features_torch(state, selected_ops)
    return features.detach().cpu().numpy().astype(np.float32)

concept_to_class = {
    "Loop": [0, 2, 6, 8, 9],
    "Vertical Line": [1, 4, 7],
    "Horizontal Line": [4, 5, 7],
    "Curvature": [0, 2, 3, 5, 6, 8, 9],
}


from Arguments import Arguments

torch.manual_seed(42)
np.random.seed(42)

def train_mnist_model(
    latent_dim: int,
    batch_size: int,
    model_name: str = "model",
    model_dir: Path = Path.cwd() / f"results/mnist_tq/",
    data_dir: Path = Path.cwd() / "data/mnist",
) -> None:
    logging.info("Fitting MNIST classifier")
    device = torch.device("cpu")
    model_dir = model_dir / model_name
    if not model_dir.exists():
        os.makedirs(model_dir)
    model = QNet(myargs, design).to(device)
    train_set = MNIST(data_dir, train=True, download=True)
    test_set = MNIST(data_dir, train=False, download=True)
    train_transform = transforms.Compose([transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.ToTensor()])
    train_set.transform = train_transform
    test_set.transform = test_transform
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False
    )
    try:
        model.load_state_dict(torch.load(model_dir / f"vqc_model.pt"), strict=False)
        print("load success")
    except:
        pass
    model.fit(device, train_loader, test_loader, model_dir, n_epoch=50, patience=20)


def train_mnist_model_two_stage(
    latent_dim: int,
    batch_size: int,
    model_name: str = "model",
    model_dir: Path = Path.cwd() / "results/mnist_tq",
    data_dir: Path = Path.cwd() / "data/mnist",
    n_epoch: int = 100,
    lr_q: float = 1e-3,
    lr_head: float = 1e-3,
) -> None:
    """Train in two stages: quantum params first, then classification head."""
    logging.info("Fitting MNIST classifier with two-stage training")
    device = torch.device("cpu")
    model_dir = model_dir / model_name
    if not model_dir.exists():
        os.makedirs(model_dir)

    model = QNet(myargs, design).to(device)
    train_set = MNIST(data_dir, train=True, download=True)
    test_set = MNIST(data_dir, train=False, download=True)
    train_transform = transforms.Compose([transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.ToTensor()])
    train_set.transform = train_transform
    test_set.transform = test_transform
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False
    )

    ckpt_path = model_dir / "vqc_model.pt"
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path), strict=False)
        logging.info(f"Loaded checkpoint from {ckpt_path}")

    stage1_epochs = n_epoch // 2
    stage2_epochs = n_epoch - stage1_epochs

    # Stage 1: train only quantum parameters
    for p in model.out.parameters():
        p.requires_grad = False
    model.q_params_rot.requires_grad = True
    model.q_params_enta.requires_grad = True
    optim_q = torch.optim.Adam(
        [model.q_params_rot, model.q_params_enta], lr=lr_q, weight_decay=1e-5
    )

    best_test_acc = 0.0
    for epoch in range(stage1_epochs):
        train_loss = model.train_epoch(device, train_loader, optim_q)
        test_loss, test_acc = model.test_epoch(device, test_loader)
        logging.info(
            f"[Stage 1][{epoch + 1}/{stage1_epochs}] "
            f"Train Loss {train_loss:.3g} \t "
            f"Test Loss {test_loss:.3g} \t "
            f"Test Accuracy {test_acc * 100:.3g}%"
        )
        if test_acc > best_test_acc:
            best_test_acc = float(test_acc)
            model.cpu()
            model.save(model_dir)
            model.to(device)

    # Stage 2: train only classification head
    for p in model.out.parameters():
        p.requires_grad = True
    model.q_params_rot.requires_grad = False
    model.q_params_enta.requires_grad = False
    optim_head = torch.optim.Adam(model.out.parameters(), lr=lr_head, weight_decay=1e-5)

    for epoch in range(stage2_epochs):
        train_loss = model.train_epoch(device, train_loader, optim_head)
        test_loss, test_acc = model.test_epoch(device, test_loader)
        logging.info(
            f"[Stage 2][{epoch + 1}/{stage2_epochs}] "
            f"Train Loss {train_loss:.3g} \t "
            f"Test Loss {test_loss:.3g} \t "
            f"Test Accuracy {test_acc * 100:.3g}%"
        )
        if test_acc > best_test_acc:
            best_test_acc = float(test_acc)
            model.cpu()
            model.save(model_dir)
            model.to(device)

    logging.info(f"Two-stage training completed. Best test accuracy: {best_test_acc:.4f}")


def class_wise_accuracy(
    batch_size: int,
    model_dir: Path = Path.cwd() / "results/mnist_tq",
    data_dir: Path = Path.cwd() / "data/mnist",
    model_name: str = "model",
    save_dir: Path = Path.cwd() / "results/mnist_tq/class_accuracy",
) -> pd.DataFrame:
    """Evaluate original model accuracy for each MNIST class on test set."""
    device = torch.device("cpu")
    model = QNet(myargs, design)
    model_path = model_dir / model_name / "vqc_model.pt"
    model.load_state_dict(torch.load(model_path), strict=False)
    model.to(device)
    model.eval()

    test_set = MNIST(data_dir, train=False, download=True)
    test_set.transform = transforms.Compose([transforms.ToTensor()])
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False
    )

    n_classes = 10
    class_correct = torch.zeros(n_classes, dtype=torch.long)
    class_total = torch.zeros(n_classes, dtype=torch.long)

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            preds = torch.argmax(model(x_batch), dim=-1)
            for class_idx in range(n_classes):
                mask = y_batch == class_idx
                class_total[class_idx] += mask.sum().cpu()
                class_correct[class_idx] += (preds[mask] == y_batch[mask]).sum().cpu()

    overall_acc = class_correct.sum().item() / max(class_total.sum().item(), 1)
    results_df = pd.DataFrame(
        {
            "Class": list(range(n_classes)),
            "Correct": class_correct.numpy(),
            "Total": class_total.numpy(),
            "Accuracy": [
                class_correct[i].item() / max(class_total[i].item(), 1)
                for i in range(n_classes)
            ],
        }
    )

    if not save_dir.exists():
        os.makedirs(save_dir)
    csv_path = save_dir / f"{model_name}_metrics.csv"
    results_df.to_csv(csv_path, index=False)

    logging.info(f"Original network overall test accuracy: {overall_acc:.4f}")
    logging.info("Per-class accuracy on MNIST test set:\n" + results_df.to_string(index=False))
    logging.info(f"Saved class-wise accuracy to {csv_path}")
    return results_df


def concept_accuracy(
    random_seeds: list[int],
    latent_dim: int,
    plot: bool,
    save_dir: Path = Path.cwd() / "results/mnist_tq/concept_accuracy",
    data_dir: Path = Path.cwd() / "data/mnist",
    model_dir: Path = Path.cwd() / f"results/mnist_tq/",
    model_name: str = "model",
    pullback_top_n: int = 32,
    pullback_target: str = "z_sum",
    pullback_aggregate: str = "mean_abs",
) -> None:
    device = torch.device("cpu")
    torch.manual_seed(random_seeds[0])

    representation_dir = save_dir / f"{model_name}_representations"
    if not representation_dir.exists():
        os.makedirs(representation_dir)

    model_dir = model_dir / model_name
    model = QNet(myargs, design)
    model.load_state_dict(torch.load(model_dir / f"vqc_model.pt"), strict=False)
    model.to(device)
    model.eval()
    target_observable = build_target_observable(model.args.n_qubits, mode=pullback_target)
    observable_family, _ = _build_xyz_zz_observables(model.args.n_qubits)
    top_obs_records = []

    # Fit a concept classifier and test accuracy for each concept
    results_data = []
    for concept_name, random_seed in itertools.product(concept_to_class, random_seeds):
        logging.info(f"Working with concept {concept_name} and seed {random_seed}")
        # Save representations for training concept examples and then remove the hooks
        module_dic, handler_train_dic = register_hooks(
            model, representation_dir, f"{concept_name}_seed{random_seed}_train"
        )
        X_train, y_train = generate_mnist_concept_dataset(
            concept_to_class[concept_name], data_dir, True, 20, random_seed
        )
        model(torch.from_numpy(X_train).to(device))
        remove_all_hooks(handler_train_dic)
        pullback_selection = select_global_top_n_pullback_observables(
            model=model,
            x_train=torch.from_numpy(X_train),
            target_observable=target_observable,
            top_n=pullback_top_n,
            aggregate=pullback_aggregate,
        )
        top_obs_records += [
            {
                "Concept": concept_name,
                "Seed": random_seed,
                "Rank": rank + 1,
                "Observable": obs_name,
                "Score": float(obs_score),
            }
            for rank, (obs_name, obs_score) in enumerate(
                zip(pullback_selection["top_names"], pullback_selection["top_scores"])
            )
        ]
        
        # Save representations for testing concept examples and then remove the hooks
        module_dic, handler_test_dic = register_hooks(
            model, representation_dir, f"{concept_name}_seed{random_seed}_test"
        )
        X_test, y_test = generate_mnist_concept_dataset(
            concept_to_class[concept_name], data_dir, False, 50, random_seed
        )
        model(torch.from_numpy(X_test).to(device))
        remove_all_hooks(handler_test_dic)
        h_train_expectation = observable_expectation_feature_matrix(
            model=model,
            x_np=X_train,
            observable_indices=pullback_selection["top_indices"],
            observable_family=observable_family,
        )
        h_test_expectation = observable_expectation_feature_matrix(
            model=model,
            x_np=X_test,
            observable_indices=pullback_selection["top_indices"],
            observable_family=observable_family,
        )
        # Create concept classifiers, fit them and test them for each representation space
        for module_name in module_dic:
            logging.info(f"Fitting concept classifiers for {module_name}")
            car = CAR(device)
            cav = CAV(device)
            if module_name == "tqlayer_n":
                H_train = h_train_expectation
                H_test = h_test_expectation
            else:
                hook_name = f"{concept_name}_seed{random_seed}_train_{module_name}"
                H_train = get_saved_representations(hook_name, representation_dir)
                if H_train.dtype == np.complex64:
                    H_train = np.array([dm2vec(dm) for dm in H_train])
                hook_name = f"{concept_name}_seed{random_seed}_test_{module_name}"
                H_test = get_saved_representations(hook_name, representation_dir)
                if H_test.dtype == np.complex64:
                    H_test = np.array([dm2vec(dm) for dm in H_test])

            car.fit(H_train, y_train)
            cav.fit(H_train, y_train)
            results_data.append(
                [
                    concept_name,
                    module_name,
                    random_seed,
                    "CAR",
                    accuracy_score(y_train, car.predict(H_train)),
                    accuracy_score(y_test, car.predict(H_test)),
                ]
            )
            results_data.append(
                [
                    concept_name,
                    module_name,
                    random_seed,
                    "CAV",
                    accuracy_score(y_train, cav.predict(H_train)),
                    accuracy_score(y_test, cav.predict(H_test)),
                ]
            )
    results_df = pd.DataFrame(
        results_data,
        columns=["Concept", "Layer", "Seed", "Method", "Train ACC", "Test ACC"],
    )
    csv_path = save_dir / "metrics.csv"
    results_df.to_csv(csv_path, header=True, mode="w", index=False)
    if top_obs_records:
        top_obs_df = pd.DataFrame(top_obs_records)
        top_obs_df.to_csv(save_dir / "pullback_top_observables.csv", index=False)
    if plot:
        plot_concept_accuracy(save_dir, None, "mnist")
        for concept in concept_to_class:
            plot_concept_accuracy(save_dir, concept, "mnist")


def statistical_significance(
    random_seed: int,
    latent_dim: int,
    save_dir: Path = Path.cwd() / "results/mnist_tq/statistical_significance",
    data_dir: Path = Path.cwd() / "data/mnist",
    model_dir: Path = Path.cwd() / "results/mnist_tq",
    model_name: str = "model",
) -> None:
    device = torch.device("cpu")
    torch.manual_seed(random_seed)
    model_dir = model_dir / model_name
    representation_dir = save_dir / f"{model_name}_representations"
    if not representation_dir.exists():
        os.makedirs(representation_dir)

    model = QNet(myargs, design)
    model.load_state_dict(torch.load(model_dir / f"vqc_model.pt"), strict=False)
    model.to(device)
    model.eval()

    # Fit a concept classifier and test accuracy for each concept
    results_data = []
    for concept_name in concept_to_class:
        logging.info(f"Working with concept {concept_name} ")
        # Save representations for training concept examples and then remove the hooks
        module_dic, handler_train_dic = register_hooks(
            model, representation_dir, f"{concept_name}_seed{random_seed}_train"
        )
        X_train, y_train = generate_mnist_concept_dataset(
            concept_to_class[concept_name], data_dir, True, 200, random_seed
        )
        model(torch.from_numpy(X_train).to(device))
        remove_all_hooks(handler_train_dic)

        # Create concept classifiers, fit them and test them for each representation space
        for module_name in module_dic:
            logging.info(f"Testing concept classifiers for {module_name}")
            car = CAR(device)
            cav = CAV(device)
            hook_name = f"{concept_name}_seed{random_seed}_train_{module_name}"
            H_train = get_saved_representations(hook_name, representation_dir)
            if H_train.dtype == np.complex64:
                H_train=np.array([dm2vec(dm) for dm in H_train])
            results_data.append(
                [
                    concept_name,
                    module_name,
                    "CAR",
                    car.permutation_test(H_train, y_train),
                ]
            )
            results_data.append(
                [
                    concept_name,
                    module_name,
                    "CAV",
                    cav.permutation_test(H_train, y_train),
                ]
            )

    results_df = pd.DataFrame(
        results_data, columns=["Concept", "Layer", "Method", "p-value"]
    )
    csv_path = save_dir / "metrics.csv"
    results_df.to_csv(csv_path, header=True, mode="w", index=False)


def global_explanations(
    random_seed: int,
    batch_size: int,
    latent_dim: int,
    plot: bool,
    save_dir: Path = Path.cwd() / "results/mnist_tq/global_explanations",
    data_dir: Path = Path.cwd() / "data/mnist",
    model_dir: Path = Path.cwd() / f"results/mnist_tq",
    model_name: str = "model",
    pullback_top_n: int = 8,
    pullback_target: str = "z_sum",
    pullback_aggregate: str = "mean_abs",
    observable_csv_name: str = "pullback_top_observables.csv",
) -> None:
    device = torch.device("cpu")
    torch.manual_seed(random_seed)

    if not save_dir.exists():
        os.makedirs(save_dir)

    model_dir = model_dir / model_name
    model = QNet(myargs, design)
    model.load_state_dict(torch.load(model_dir / f"vqc_model.pt"), strict=False)
    model.to(device)
    model.eval()
    observable_family, _ = _build_xyz_zz_observables(model.args.n_qubits)

    # Fit a concept classifier and test accuracy for each concept
    results_data = []
    car_classifiers = [CAR(device) for _ in concept_to_class]
    cav_classifiers = [CAV(device) for _ in concept_to_class]
    pullback_csv_path = save_dir / observable_csv_name
    observable_indices, _ = load_observable_set_csv(
        csv_path=pullback_csv_path,
        n_qubits=model.args.n_qubits,
        top_n=pullback_top_n,
    )

    for concept_name, car_classifier, cav_classifier in zip(
        concept_to_class, car_classifiers, cav_classifiers
    ):
        logging.info(f"Now fitting concept classifiers for {concept_name}")
        X_train, y_train = generate_mnist_concept_dataset(
            concept_to_class[concept_name], data_dir, True, 200, random_seed
        )
        h_train = observable_expectation_feature_matrix(
            model=model,
            x_np=X_train,
            observable_indices=observable_indices,
            observable_family=observable_family,
        )
        car_classifier.fit(h_train, y_train)
        cav_classifier.fit(h_train, y_train)

    test_set = MNIST(data_dir, train=False, download=True)
    test_set.transform = transforms.Compose([transforms.ToTensor()])
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False
    )

    logging.info("Now predicting concepts on the test set")
    for X_test, y_test in tqdm(test_loader, unit="batch", leave=False):
        X_test_np = X_test.numpy()
        car_preds = []
        cav_preds = []
        for concept_name, car, cav in zip(
            concept_to_class, car_classifiers, cav_classifiers
        ):
            h_test = observable_expectation_feature_matrix(
                model=model,
                x_np=X_test_np,
                observable_indices=observable_indices,
                observable_family=observable_family,
            )
            car_preds.append(car.predict(h_test))
            cav_preds.append(cav.predict(h_test))
        targets = [
            [int(label in concept_to_class[concept]) for label in y_test]
            for concept in concept_to_class
        ]

        results_data += [
            ["TCAR", label.item()] + [int(car_pred[idx]) for car_pred in car_preds]
            for idx, label in enumerate(y_test)
        ]
        results_data += [
            ["TCAV", label.item()] + [int(cav_pred[idx] > 0) for cav_pred in cav_preds]
            for idx, label in enumerate(y_test)
        ]
        results_data += [
            ["True Prop.", label.item()] + [target[idx] for target in targets]
            for idx, label in enumerate(y_test)
        ]

    csv_path = save_dir / "metrics.csv"
    results_df = pd.DataFrame(
        results_data, columns=["Method", "Class"] + list(concept_to_class.keys())
    )
    results_df.to_csv(csv_path, index=False)
    if plot:
        plot_global_explanation(save_dir, "mnist")


def feature_importance(
    random_seed: int,
    batch_size: int,
    latent_dim: int,
    plot: bool,
    save_dir: Path = Path.cwd() / "results/mnist_tq/feature_importance",
    data_dir: Path = Path.cwd() / "data/mnist",
    model_dir: Path = Path.cwd() / f"results/mnist_tq",
    model_name: str = "model",
) -> None:
    device = torch.device("cpu")
    torch.manual_seed(random_seed)

    if not save_dir.exists():
        os.makedirs(save_dir)

    model_dir = model_dir / model_name
    model = QNet(myargs, design)
    model.load_state_dict(torch.load(model_dir / f"vqc_model.pt"), strict=False)
    model.to(device)
    model.eval()

    # Fit a concept classifier and compute feature importance for each concept
    car_classifiers = [CAR(device) for _ in concept_to_class]
    test_set = MNIST(data_dir, train=False, download=True)
    test_set.transform = transforms.Compose([transforms.ToTensor()])
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False
    )
    attribution_dic = {}
    baselines = torch.zeros((1, 1, 28, 28)).to(device)
    for concept_name, car in zip(concept_to_class, car_classifiers):
        logging.info(f"Now fitting CAR classifier for {concept_name}")
        X_train, y_train = generate_mnist_concept_dataset(
            concept_to_class[concept_name], data_dir, True, 200, random_seed
        )
        H_train = (
            model.input_to_representation(torch.from_numpy(X_train).to(device))
        )
        h_train = np.array([dm2vec(dm) for dm in H_train])
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


def kernel_sensitivity(
    random_seeds: list[int],
    latent_dim: int,
    plot: bool,
    save_dir: Path = Path.cwd() / "results/mnist_tq/kernel_sensitivity",
    data_dir: Path = Path.cwd() / "data/mnist",
    model_dir: Path = Path.cwd() / f"results/mnist_tq/",
    model_name: str = "model",
) -> None:
    device = torch.device("cpu")
    torch.manual_seed(random_seeds[0])

    if not save_dir.exists():
        os.makedirs(save_dir)

    model_dir = model_dir / model_name
    model = QNet(myargs, design)
    model.load_state_dict(torch.load(model_dir / f"vqc_model.pt"), strict=False)
    model.to(device)
    model.eval()

    # Fit a concept classifier and test accuracy for each concept
    kernels = {
        "Gaussian RBF": "rbf",
        "Linear": "linear",
        "Polynomial": "poly",
        "Sigmoid": "sigmoid",
        "Matern": Matern(),
    }
    cars = {
        kernel_name: CAR(device, kernel=kernels[kernel_name]) for kernel_name in kernels
    }
    results_data = []
    for concept_name, random_seed in itertools.product(concept_to_class, random_seeds):
        logging.info(f"Working with concept {concept_name} and seed {random_seed}")
        # Compute representation
        X_train, C_train = generate_mnist_concept_dataset(
            concept_to_class[concept_name], data_dir, True, 200, random_seed
        )
        H_train = (
            model.input_to_representation(torch.from_numpy(X_train).to(device))
        )
        h_train = np.array([dm2vec(dm) for dm in H_train])
        X_val, C_val = generate_mnist_concept_dataset(
            concept_to_class[concept_name], data_dir, True, 50, random_seed + 1
        )
        H_val = (
            model.input_to_representation(torch.from_numpy(X_val).to(device))
        )
        h_val = np.array([dm2vec(dm) for dm in H_val])
        # Create concept classifiers, fit them and test them
        for kernel_name in cars:
            car = cars[kernel_name]
            car.fit(h_train, C_train)
            results_data.append(
                [
                    "Training",
                    concept_name,
                    random_seed,
                    kernel_name,
                    accuracy_score(C_train, car.predict(h_train)),
                ]
            )
            results_data.append(
                [
                    "Validation",
                    concept_name,
                    random_seed,
                    kernel_name,
                    accuracy_score(C_val, car.predict(h_val)),
                ]
            )

    logging.info(f"Saving results in {str(save_dir)}")
    results_df = pd.DataFrame(
        results_data, columns=["Set", "Concept", "Seed", "Kernel", "Accuracy"]
    )
    csv_path = save_dir / "metrics.csv"
    results_df.to_csv(csv_path, header=True, mode="w", index=False)
    if plot:
        plot_kernel_sensitivity(save_dir, "mnist")


def concept_size_impact(
    random_seeds: list[int],
    latent_dim: int,
    concept_sizes: list[int],
    plot: bool,
    save_dir: Path = Path.cwd() / "results/mnist_tq/concept_set_impact",
    data_dir: Path = Path.cwd() / "data/mnist",
    model_dir: Path = Path.cwd() / f"results/mnist_tq/",
    model_name: str = "model",
) -> None:
    device = torch.device("cpu")
    torch.manual_seed(random_seeds[0])

    if not save_dir.exists():
        os.makedirs(save_dir)

    model_dir = model_dir / model_name
    model = QNet(myargs, design)
    model.load_state_dict(torch.load(model_dir / f"vqc_model.pt"), strict=False)
    model.to(device)
    model.eval()

    # Fit a concept classifier and test accuracy for each concept
    results_data = []
    for concept_name, random_seed in itertools.product(concept_to_class, random_seeds):
        # Compute representation
        X_test, C_test = generate_mnist_concept_dataset(
            concept_to_class[concept_name], data_dir, False, 50, random_seed
        )
        H_test = (
            model.input_to_representation(torch.from_numpy(X_test).to(device))
        )
        # Vectorize the (complex) density matrices into real feature vectors,
        # exactly like every other experiment does before fitting a classifier.
        h_test = np.array([dm2vec(dm) for dm in H_test])
        # Create concept classifiers, fit them and test them for each representation space
        prev_size = 0
        concept_sizes.sort()
        # Start with empty (0-row) accumulators that share the feature dimension
        # of the vectorized representations, so concatenation stays 2-dimensional.
        C_train = np.empty([0], dtype=int)
        H_train = np.empty([0, h_test.shape[1]])
        for concept_size in concept_sizes:
            logging.info(
                f"Working with concept {concept_name}, seed {random_seed} and a set of size {concept_size}"
            )
            n_add = concept_size - prev_size
            prev_size = concept_size
            X_add, C_add = generate_mnist_concept_dataset(
                concept_to_class[concept_name],
                data_dir,
                True,
                n_add,
                random_seed + concept_size,
            )
            H_add = (
                model.input_to_representation(torch.from_numpy(X_add).to(device))
            )
            h_add = np.array([dm2vec(dm) for dm in H_add])
            C_train = np.concatenate((C_train, C_add), axis=0)
            H_train = np.concatenate((H_train, h_add), axis=0)
            car = CAR(device)

            car.fit(H_train, C_train)
            results_data.append(
                [
                    concept_size,
                    random_seed,
                    concept_name,
                    accuracy_score(C_train, car.predict(H_train)),
                ]
            )

    logging.info(f"Saving results in {str(save_dir)}")
    results_df = pd.DataFrame(
        results_data,
        columns=["Concept Sets Size", "Random Seed", "Concept", "Test Accuracy"],
    )
    csv_path = save_dir / "metrics.csv"
    results_df.to_csv(csv_path, header=True, mode="w", index=False)
    if plot:
        plot_concept_size_impact(save_dir, "mnist")


def tcar_inter_concept(
    random_seed: int,
    batch_size: int,
    latent_dim: int,
    plot: bool,
    save_dir: Path = Path.cwd() / "results/mnist_tq/tcar_inter_concept",
    data_dir: Path = Path.cwd() / "data/mnist",
    model_dir: Path = Path.cwd() / f"results/mnist_tq",
    model_name: str = "model",
) -> None:
    device = torch.device("cpu")
    torch.manual_seed(random_seed)

    if not save_dir.exists():
        os.makedirs(save_dir)

    model_dir = model_dir / model_name
    model = QNet(myargs, design)
    model.load_state_dict(torch.load(model_dir / f"vqc_model.pt"), strict=False)
    model.to(device)
    model.eval()

    # Fit a concept classifier and test accuracy for each concept
    results_data = []
    car_classifiers = [CAR(device) for _ in concept_to_class]

    for concept_name, car_classifier in zip(concept_to_class, car_classifiers):
        logging.info(f"Now fitting concept classifiers for {concept_name}")
        X_train, y_train = generate_mnist_concept_dataset(
            concept_to_class[concept_name], data_dir, True, 200, random_seed
        )
        H_train = (
            model.input_to_representation(torch.from_numpy(X_train).to(device))
        )
        h_train = np.array([dm2vec(dm) for dm in H_train])
        car_classifier.fit(h_train, y_train)

    test_set = MNIST(data_dir, train=False, download=True)
    test_set.transform = transforms.Compose([transforms.ToTensor()])
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False
    )

    logging.info("Now predicting concepts on the test set")
    for X_test, y_test in tqdm(test_loader, unit="batch", leave=False):
        H_test = model.input_to_representation(X_test.to(device))
        h_test = np.array([dm2vec(dm) for dm in H_test])
        car_preds = [car.predict(h_test) for car in car_classifiers]
        results_data += [
            [int(car_pred[idx]) for car_pred in car_preds] for idx in range(len(y_test))
        ]

    logging.info(f"Saving results in {str(save_dir)}")
    csv_path = save_dir / "metrics.csv"
    results_df = pd.DataFrame(results_data, columns=list(concept_to_class.keys()))
    results_df.to_csv(csv_path, index=False)
    if plot:
        plot_tcar_inter_concepts(save_dir, "mnist")


def adversarial_robustness(
    random_seed: int,
    batch_size: int,
    latent_dim: int,
    save_dir: Path = Path.cwd() / "results/mnist_tq/adversarial_robustness",
    data_dir: Path = Path.cwd() / "data/mnist",
    model_dir: Path = Path.cwd() / f"results/mnist_tq",
    model_name: str = "model",
    pullback_top_n: int = 8,
    observable_csv_name: str = "pullback_top_observables.csv",
) -> None:
    device = torch.device("cpu")
    torch.manual_seed(random_seed)

    if not save_dir.exists():
        os.makedirs(save_dir)

    model_dir = model_dir / model_name
    model = QNet(myargs, design)
    model.load_state_dict(torch.load(model_dir / f"vqc_model.pt"), strict=False)
    model.to(device)
    model.eval()
    observable_family, _ = _build_xyz_zz_observables(model.args.n_qubits)
    observable_csv_path = Path.cwd() / "results/mnist_tq/global_explanations" / observable_csv_name
    observable_indices, _ = load_observable_set_csv(
        csv_path=observable_csv_path,
        n_qubits=model.args.n_qubits,
        top_n=pullback_top_n,
    )

    # Fit a concept classifier and test accuracy for each concept
    results_data = []
    car_classifiers = [CAR(device) for _ in concept_to_class]
    cav_classifiers = [CAV(device) for _ in concept_to_class]

    for concept_name, car_classifier, cav_classifier in zip(
        concept_to_class, car_classifiers, cav_classifiers
    ):
        logging.info(f"Now fitting concept classifiers for {concept_name}")
        X_train, y_train = generate_mnist_concept_dataset(
            concept_to_class[concept_name], data_dir, True, 200, random_seed
        )
        h_train = observable_expectation_feature_matrix(
            model=model,
            x_np=X_train,
            observable_indices=observable_indices,
            observable_family=observable_family,
        )
        car_classifier.fit(h_train, y_train)
        cav_classifier.fit(h_train, y_train)

    test_set = MNIST(data_dir, train=False, download=True)
    test_set.transform = transforms.Compose([transforms.ToTensor()])
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False
    )
    attacker = Attacker(model, 100, 0.1, device)

    logging.info("Now predicting concepts on the test set")
    for attack_prop in [0, 0.05, 0.1, 0.2, 0.5, 0.7, 1]:
        logging.info(f"Working with {100*attack_prop}% of adversarial samples")
        for X_test, y_test in tqdm(test_loader, unit="batch", leave=False):
            n_attacks = int(len(X_test) * attack_prop)
            X_test = X_test.to(device)
            X_adv, X_test = torch.split(X_test, [n_attacks, len(X_test) - n_attacks])
            if n_attacks > 0:
                X_adv = attacker.make_adversarial_example(X_adv, model(X_adv))
            X_test = torch.cat([X_adv, X_test])
            h_test = observable_expectation_feature_matrix(
                model=model,
                x_np=X_test.detach().cpu().numpy(),
                observable_indices=observable_indices,
                observable_family=observable_family,
            )
            car_preds = [car.predict(h_test) for car in car_classifiers]
            cav_preds = [cav.predict(h_test) for cav in cav_classifiers]
            targets = [
                [int(label in concept_to_class[concept]) for label in y_test]
                for concept in concept_to_class
            ]

            results_data += [
                [attack_prop * 100, "TCAR", label.item()]
                + [int(car_pred[idx]) for car_pred in car_preds]
                for idx, label in enumerate(y_test)
            ]
            results_data += [
                [attack_prop * 100, "TCAV", label.item()]
                + [int(cav_pred[idx] > 0) for cav_pred in cav_preds]
                for idx, label in enumerate(y_test)
            ]
            results_data += [
                [attack_prop * 100, "True Prop.", label.item()]
                + [target[idx] for target in targets]
                for idx, label in enumerate(y_test)
            ]

    csv_path = save_dir / "metrics.csv"
    results_df = pd.DataFrame(
        results_data,
        columns=["Adversarial %", "Method", "Class"] + list(concept_to_class.keys()),
    )
    results_df.to_csv(csv_path, index=False)
    scores_data = []
    classes = results_df["Class"].unique()
    methods = results_df["Method"].unique()
    adv_pcts = results_df["Adversarial %"].unique()
    concepts = concept_to_class.keys()
    for adv_pct, class_idx, concept, method in itertools.product(
        adv_pcts, classes, concepts, methods
    ):
        attr = np.array(
            results_df.loc[
                (results_df.Class == class_idx)
                & (results_df.Method == method)
                & (results_df["Adversarial %"] == adv_pct)
            ][concept]
        )
        score = np.sum(attr) / len(attr)
        scores_data.append([adv_pct, method, class_idx, concept, score])
    scores_df = pd.DataFrame(
        scores_data, columns=["Adversarial %", "Method", "Class", "Concept", "Score"]
    )
    corr_data = []
    for adv_pct in adv_pcts:
        tcar_scores = scores_df.loc[
            (scores_df.Method == "TCAR") & (scores_df["Adversarial %"] == adv_pct)
        ]["Score"]
        true_scores = scores_df.loc[
            (scores_df.Method == "True Prop.") & (scores_df["Adversarial %"] == adv_pct)
        ]["Score"]
        corr_data.append([adv_pct, "TCAR", np.corrcoef(tcar_scores, true_scores)[0, 1]])
    corr_df = pd.DataFrame(
        corr_data, columns=["Adversarial %", "Method", "Correlation"]
    )
    results_md = pd.pivot_table(
        data=corr_df,
        index="Adversarial %",
        columns="Method",
        aggfunc="mean",
        values="Correlation",
    ).to_markdown()
    logging.info(results_md)


def senn() -> None:
    from pathlib import Path
    current_file = Path(__file__).absolute()
    project_root = current_file.parent.parent
    device = torch.device("cpu")

    logging.info("Now fitting SENN model")
    senn_trainer = init_trainer(str(project_root / "configs/senn_config.json"))
    senn_trainer.run()
    senn_trainer.load_checkpoint(
        str(Path.cwd() / "results/mnist_tq/senn/checkpoints/best_model.pt")
    )
    senn = senn_trainer.model
    senn.eval()
    senn_concept_relevance = senn.parameterizer
    senn_representation = senn.conceptizer.encode

    logging.info("Now tuning CAR concept densities")
    data_dir = Path.cwd() / "data/mnist"
    car_classifiers = {concept_name: CAR(device) for concept_name in concept_to_class}
    for concept_name in concept_to_class:
        logging.info(f"Tunning {concept_name}")
        X_train, c_train = generate_mnist_concept_dataset(
            concept_to_class[concept_name], data_dir, True, 300, 1
        )
        H_train = (
            senn_representation(
                torch.from_numpy(X_train).to(senn_trainer.config.device)
            )
            .flatten(start_dim=1)
            .detach()
            .cpu()
            .numpy()
        )
        car_classifiers[concept_name].tune_kernel_width(H_train, c_train)

    logging.info("Now computing concept relevance and densities")
    _, _, test_loader = get_dataloader(senn_trainer.config)
    human_concept_importances = []
    synthetic_concept_relevances = []
    for X_test, y_test in test_loader:
        H_test = (
            senn_representation(X_test.to(senn_trainer.config.device))
            .flatten(start_dim=1)
            .detach()
        )
        C_importance = [
            car_classifiers[concept_name]
            .concept_importance(H_test)
            .view(-1, 1)
            .cpu()
            .numpy()
            for concept_name in concept_to_class
        ]
        C_importance = np.concatenate(C_importance, axis=1)
        human_concept_importances.append(C_importance)
        C_relevance = senn_concept_relevance(
            X_test.to(senn_trainer.config.device)
        ).detach()
        class_select = (
            y_test.to(senn_trainer.config.device).view(-1, 1, 1).repeat(1, 5, 1)
        )
        C_relevance = (
            torch.gather(C_relevance, -1, index=class_select)
            .flatten(start_dim=1)
            .cpu()
            .numpy()
        )
        synthetic_concept_relevances.append(C_relevance)
    human_concept_importances = np.concatenate(human_concept_importances, axis=0)
    synthetic_concept_relevances = np.concatenate(synthetic_concept_relevances, axis=0)
    corr_data = np.corrcoef(
        human_concept_importances, synthetic_concept_relevances, rowvar=False
    )[4:, :4]
    corr_df = pd.DataFrame(
        data=corr_data,
        columns=list(concept_to_class.keys()),
        index=[f"SENN Concept {i+1}" for i in range(5)],
    )
    logging.info(corr_df.to_markdown())


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="feature_importance")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(1, 11)))
    parser.add_argument("--batch_size", type=int, default=120)
    parser.add_argument("--latent_dim", type=int, default=5)
    parser.add_argument("--n_observables", type=int, default=8)
    parser.add_argument("--selection_sample_size", type=int, default=200)
    parser.add_argument(
        "--observable_set",
        type=str,
        choices=["pullback", "random"],
        default="pullback",
        help="Observable-set source for global explanations.",
    )
    parser.add_argument(
        "--random_observable_seed",
        type=int,
        default=None,
        help="Seed for random observable sampling; defaults to first seed.",
    )
    parser.add_argument("--train", action="store_true",default=False)
    parser.add_argument("--plot", action="store_true",default=True)
    parser.add_argument(
        "--concept_sizes", nargs="+", type=int, default=list(range(10, 310, 30))
    )
    args = parser.parse_args()

    model_name = f"model_{args.latent_dim}"

    myargs = Arguments(task='MNIST_10')

    n_layers = myargs.n_layers
    n_qubits = myargs.n_qubits
    single = [[i]+[1]*2*n_layers for i in range(1,n_qubits+1)]
    enta = [[i]+[i+1]*n_layers for i in range(1,n_qubits)]+[[n_qubits]+[1]*n_layers]
    arch_code = [myargs.n_qubits, myargs.n_layers]
    design = single_enta_to_design(single, enta, arch_code)

    # args.observable_set = "pullback"
    args.observable_set = "random"

    args.n_observables = 20
    args.selection_sample_size = 1000
    args.seeds = [3]
    
    if args.observable_set == "pullback":
        observable_csv_path = get_observables(
            args.seeds[0],
            args.latent_dim,
            model_name=model_name,
            n_observables=args.n_observables,
            selection_sample_size=args.selection_sample_size,
        )
    else:
        random_obs_seed = args.random_observable_seed
        if random_obs_seed is None:
            random_obs_seed = args.seeds[0]
        observable_csv_path = get_random_observables(
            random_seed=random_obs_seed,
            n_observables=args.n_observables,
        )
   
    # train_mnist_model(args.latent_dim, args.batch_size, model_name=model_name)
    # train_mnist_model_two_stage(args.latent_dim, args.batch_size, model_name=model_name, n_epoch=50)
    # class_wise_accuracy(args.batch_size, model_name=model_name)
    # concept_accuracy(args.seeds, args.latent_dim, args.plot, model_name=model_name)
    # global_explanations(
    #     args.seeds[0],
    #     args.batch_size,
    #     args.latent_dim,
    #     args.plot,
    #     model_name=model_name,
    #     pullback_top_n=args.n_observables,
    #     observable_csv_name=observable_csv_path.name,
    # )
    # statistical_significance(args.seeds[0], args.latent_dim, model_name=model_name)
    # feature_importance(args.seeds[0],args.batch_size,args.latent_dim,args.plot,model_name=model_name,)
    # kernel_sensitivity(args.seeds, args.latent_dim, args.plot, model_name=model_name)
    # concept_size_impact(args.seeds,args.latent_dim,args.concept_sizes,args.plot,model_name=model_name,)
    # tcar_inter_concept(args.seeds[0],args.batch_size,args.latent_dim,args.plot,model_name=model_name,)
    adversarial_robustness(
        args.seeds[0],
        args.batch_size,
        args.latent_dim,
        model_name=model_name,
        pullback_top_n=args.n_observables,
        observable_csv_name=observable_csv_path.name,
    )
    # senn()

