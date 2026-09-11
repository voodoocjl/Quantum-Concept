import numpy as np
import torch
import torch.nn as nn
import inspect
import random


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


def _select_top_observable_indices(
    model,
    x_input: np.ndarray,
    repr_mode: str,
    random_seed: int,
    top_n: int,
    target_mode: str,
    aggregate: str,
    y_input: np.ndarray | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Return shared observable indices for the requested representation mode."""
    all_observables, all_names = _build_xyz_zz_observables(model.args.n_qubits)

    if repr_mode == "pullback":
        return _select_top_pullback_observables_over_set(
            model,
            x_input,
            top_n=top_n,
            target_mode=target_mode,
            aggregate=aggregate,
            y_set=y_input,
        )

    if repr_mode == "random":
        rng = np.random.default_rng(int(random_seed))
        selected_idx = rng.choice(
            len(all_observables),
            size=min(int(top_n), len(all_observables)),
            replace=False,
        )
        selected_idx = np.asarray(selected_idx, dtype=np.int64)
        return selected_idx, [all_names[int(idx)] for idx in selected_idx]

    if repr_mode == "local_x":
        selected_names = [f"X{idx}" for idx in range(model.args.n_qubits)]
    elif repr_mode == "local_y":
        selected_names = [f"Y{idx}" for idx in range(model.args.n_qubits)]
    elif repr_mode == "local_z":
        selected_names = [f"Z{idx}" for idx in range(model.args.n_qubits)]
    elif repr_mode == "local_zz":
        selected_names = [
            f"Z{i_idx}Z{j_idx}"
            for i_idx in range(model.args.n_qubits)
            for j_idx in range(i_idx + 1, model.args.n_qubits)
        ]
    else:
        raise ValueError(f"Unsupported local observable repr_mode: {repr_mode!r}")

    name_to_idx = {name: idx for idx, name in enumerate(all_names)}
    selected_idx = np.asarray([name_to_idx[name] for name in selected_names], dtype=np.int64)
    return selected_idx, selected_names


def _select_top_pullback_observables_over_set(
    model,
    x_set: np.ndarray,
    top_n: int,
    target_mode: str = "z_sum",
    aggregate: str = "mean_abs",
    y_set: np.ndarray | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Select top observable indices by pullback coefficients aggregated over a set."""
    coeff_mat, names = _compute_pullback_coefficients_over_set(
        model,
        x_set,
        target_mode,
        y_set=y_set,
    )

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
    top_names = [names[idx] for idx in top_idx]
    return top_idx.astype(np.int64), top_names


def _compute_pullback_coefficients_over_set(
    model,
    x_set: np.ndarray,
    target_mode: str = "z_sum",
    y_set: np.ndarray | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Compute pullback coefficient matrix over the fixed XYZ/XX/YY/ZZ basis."""
    x_set = np.asarray(x_set)
    if x_set.ndim < 2:
        raise ValueError(
            f"Expected batched input with batch axis, got shape {tuple(x_set.shape)}"
        )

    uses_class_conditioned_target = target_mode in {"class_z", "class_margin"}
    if uses_class_conditioned_target:
        if y_set is None:
            raise ValueError(
                f"target_mode={target_mode!r} requires y_set so each sample can use its class-aligned target."
            )
        y_arr = np.asarray(y_set, dtype=np.int64)
        if y_arr.shape[0] != x_set.shape[0]:
            raise ValueError(
                f"y_set length ({y_arr.shape[0]}) does not match x_set size ({x_set.shape[0]})."
            )
    else:
        y_arr = None

    if not uses_class_conditioned_target:
        target_obs = build_target_observable(
            model.args.n_qubits,
            mode=target_mode,
            model=model,
        )
    family, names = _build_xyz_zz_observables(model.args.n_qubits)

    x_tensor = torch.from_numpy(x_set).float()
    x_proc = model.preprocess(x_tensor)
    coeff_rows = []
    for idx in range(x_proc.shape[0]):
        x_np = x_proc[idx].detach().cpu().numpy()
        if uses_class_conditioned_target:
            target_obs = build_target_observable(
                model.args.n_qubits,
                mode=target_mode,
                target_label=int(y_arr[idx]),
                digits_of_interest=list(getattr(model.args, "digits_of_interest", [])),
            )
        u_tail = model._build_remaining_unitary(x_np)
        o_pullback = u_tail.conj().T @ target_obs @ u_tail

        row = []
        for basis_op in family:
            denom = float(np.real(np.trace(basis_op.conj().T @ basis_op)))
            coef = float(np.real(np.trace(basis_op.conj().T @ o_pullback))) / denom
            row.append(coef)
        coeff_rows.append(np.array(row, dtype=np.float32))

    coeff_mat = np.stack(coeff_rows, axis=0)
    return coeff_mat.astype(np.float32, copy=False), names


def _select_top_pullback_observables_per_sample(
    model,
    x_set: np.ndarray,
    top_n: int,
    target_mode: str = "z_sum",
    y_set: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Select per-sample top indices and return sparse pullback features."""
    coeff_mat, _ = _compute_pullback_coefficients_over_set(
        model,
        x_set,
        target_mode,
        y_set=y_set,
    )
    top_n = min(int(top_n), coeff_mat.shape[1])

    top_indices = np.argsort(-np.abs(coeff_mat), axis=1)[:, :top_n]
    
    return top_indices.astype(np.int64)


class TaskConceptModelWrapper(nn.Module):
    """Shared concept wrapper for task-based quantum models.

    Differences among experiments are mainly task/data plumbing; this wrapper
    centralizes the concept representation interface.
    """

    def __init__(
        self,
        base_model,
        repr_mode: str = "pullback",
        random_seed: int = 42,
        pullback_target_mode: str = "z_sum",
        random_top_n: int | None = None,
        concept_layer: int = -1,
        pullback_sparse_top_n: int = 0,
        return_numpy: bool = False,
    ):
        super().__init__()
        self.model = base_model
        self.repr_mode = repr_mode
        self.args = base_model.args
        self.design = base_model.design
        self.concept_layer = concept_layer
        self.pullback_target_mode = pullback_target_mode
        self.random_seed = int(random_seed)
        self.pullback_sparse_top_n = int(pullback_sparse_top_n)
        self.return_numpy = bool(return_numpy)

        pullback_observables, pullback_names = _build_xyz_zz_observables(self.args.n_qubits)
        self._pullback_observables = pullback_observables
        self._pullback_observable_names = list(pullback_names)
        self._pullback_observable_indices: np.ndarray | None = None

        if random_top_n is None:
            random_top_n = self.args.n_qubits
        random_top_n = max(1, min(int(random_top_n), len(self._pullback_observables)))
        rng = np.random.default_rng(self.random_seed)
        self._random_indices = np.asarray(
            rng.choice(len(self._pullback_observables), size=random_top_n, replace=False),
            dtype=np.int64,
        )

        # Bound-method signature excludes self.
        try:
            n_params = len(inspect.signature(self.model.forward).parameters)
        except (TypeError, ValueError):
            n_params = 1
        self._forward_accepts_task_args = n_params >= 3

    @property
    def q_params_rot(self):
        return self.model.QuantumLayer.q_params_rot

    @property
    def q_params_enta(self):
        return self.model.QuantumLayer.q_params_enta

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]

        # Already preprocessed: [B, n_qubits, 4]
        if x.ndim == 3 and x.shape[1] == self.args.n_qubits and x.shape[2] == 4:
            return x

        # SEER-style tabular preprocessing.
        if x.ndim == 2 and hasattr(self.model, "fc"):
            x = self.model.fc(x)
        # MNIST-style image preprocessing.
        elif x.ndim >= 3 and hasattr(self.model, "QuantumLayer"):
            task_name = str(getattr(self.args, "task", ""))
            if not task_name.startswith("QML") and hasattr(self.model.QuantumLayer, "adaptive_pool"):
                x = self.model.QuantumLayer.adaptive_pool(x)

        return x.view(bsz, self.args.n_qubits, 4)

    def _selected_design(self) -> list:
        if self.concept_layer == -1:
            return self.design
        return [op for op in self.design if int(op[2]) <= self.concept_layer]

    def _remaining_design(self) -> list:
        if self.concept_layer == -1:
            return []
        return [op for op in self.design if int(op[2]) > self.concept_layer]

    def _build_full_unitary(self, x_np: np.ndarray) -> np.ndarray:
        n_qubits = self.args.n_qubits
        dim = 1 << n_qubits
        q_rot = self.q_params_rot.detach().cpu().numpy()
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
        n_qubits = self.args.n_qubits
        dim = 1 << n_qubits
        q_rot = self.q_params_rot.detach().cpu().numpy()
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
        dim = 1 << self.args.n_qubits
        zero_state = np.zeros(dim, dtype=np.complex64)
        zero_state[0] = 1.0

        rows = []
        for b_idx in range(x_proc.shape[0]):
            x_np = x_proc[b_idx].detach().cpu().numpy()
            u = self._build_full_unitary(x_np)
            rows.append(u @ zero_state)
        return np.stack(rows, axis=0)

    def _get_pullback(self, x_proc: torch.Tensor) -> np.ndarray:
        if self._pullback_observable_indices is None:
            raise ValueError(
                "Pullback observable indices are not configured. "
                "Call set_pullback_observables(...) before pullback inference."
            )

        selected_ops = [
            self._pullback_observables[int(idx)]
            for idx in np.asarray(self._pullback_observable_indices, dtype=np.int64)
        ]
        observable_tensor = torch.tensor(np.stack(selected_ops), dtype=torch.complex64)
        statevectors = self._get_statevector(x_proc)
        state_tensor = torch.from_numpy(statevectors)
        feat = state_to_selected_observable_features_torch(
            state_tensor,
            observable_tensor,
        ).cpu().numpy().astype(np.float32, copy=False)

        if self.pullback_sparse_top_n > 0:
            keep_n = min(self.pullback_sparse_top_n, feat.shape[1])
            for row_idx in range(feat.shape[0]):
                top_idx = np.argsort(-np.abs(feat[row_idx]))[:keep_n]
                mask = np.zeros(feat.shape[1], dtype=bool)
                mask[top_idx] = True
                feat[row_idx, ~mask] = 0.0

        return feat

    def set_pullback_observables(self, observable_indices: np.ndarray) -> None:
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

    def _observable_matrices_for_repr(self, repr_mode: str) -> list[np.ndarray]:
        n_qubits = self.args.n_qubits
        family, names = _build_xyz_zz_observables(n_qubits)
        name_to_op = {name: op for name, op in zip(names, family)}

        if repr_mode == "local_x":
            return [name_to_op[f"X{idx}"] for idx in range(n_qubits)]
        if repr_mode == "local_y":
            return [name_to_op[f"Y{idx}"] for idx in range(n_qubits)]
        if repr_mode == "local_z":
            return [name_to_op[f"Z{idx}"] for idx in range(n_qubits)]
        if repr_mode == "local_zz":
            return [
                name_to_op[f"Z{i_idx}Z{j_idx}"]
                for i_idx in range(n_qubits)
                for j_idx in range(i_idx + 1, n_qubits)
            ]
        if repr_mode == "random":
            return [family[int(i)] for i in self._random_indices]
        raise ValueError(f"Unsupported local observable repr_mode: {repr_mode!r}")

    def _get_local_observable_features(self, x_proc: torch.Tensor) -> np.ndarray:
        statevectors = self._get_statevector(x_proc)
        observables = self._observable_matrices_for_repr(self.repr_mode)
        observable_tensor = torch.tensor(np.stack(observables), dtype=torch.complex64)
        state_tensor = torch.from_numpy(statevectors)
        return state_to_selected_observable_features_torch(
            state_tensor,
            observable_tensor,
        ).cpu().numpy().astype(np.float32, copy=False)

    def forward(
        self,
        x: torch.Tensor,
        n_qubits: int | None = None,
        task: str | None = None,
    ) -> torch.Tensor:
        if self._forward_accepts_task_args:
            resolved_n_qubits = int(self.args.n_qubits if n_qubits is None else n_qubits)
            resolved_task = getattr(self.args, "task", None) if task is None else task
            return self.model(x, resolved_n_qubits, resolved_task)
        return self.model(x)

    def save(self, model_dir):
        torch.save(self.model.state_dict(), model_dir)

    def load_state_dict(self, state_dict, strict=True):
        return self.model.load_state_dict(state_dict, strict=strict)

    def input_to_representation(self, x: torch.Tensor) -> torch.Tensor:
        x_proc = self.preprocess(x)

        if self.repr_mode == "statevector":
            out = self._get_statevector(x_proc)
        elif self.repr_mode == "pullback":
            out = self._get_pullback(x_proc)
        elif self.repr_mode in {"local_x", "local_y", "local_z", "local_zz", "random"}:
            out = self._get_local_observable_features(x_proc)
        else:
            raise ValueError(f"Unsupported repr_mode: {self.repr_mode!r}")

        if self.return_numpy:
            return out
        return torch.from_numpy(out).to(device=x.device, dtype=torch.float32)

    def representation_to_output(self, psi, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


# Backward-compatible alias used by existing SEER experiment code.
SeerConceptModelWrapper = TaskConceptModelWrapper


def _resolve_target_channel(label: int, digits_of_interest: list[int] | None) -> int:
    # Class-conditioned targets are defined on output channels, so labels must be
    # loader-space indices [0..K-1], not original/raw digit IDs.
    channel_idx = int(label)

    if channel_idx < 0:
        raise ValueError(f"Loader label index must be non-negative, got {channel_idx}.")

    if digits_of_interest is not None and len(digits_of_interest) > 0:
        if channel_idx >= len(digits_of_interest):
            raise ValueError(
                f"Loader label index {channel_idx} out of range for "
                f"digits_of_interest size {len(digits_of_interest)}."
            )

    return channel_idx


def _resolve_head_weight_matrix(model) -> np.ndarray:
    """Resolve linear-head weights as a numpy matrix of shape [K, n_qubits]."""
    base_model = getattr(model, "model", model)

    head_module = None
    for attr_name in ("head", "linear_head", "classifier", "readout"):
        candidate = getattr(base_model, attr_name, None)
        if isinstance(candidate, nn.Linear):
            head_module = candidate
            break

    if head_module is None:
        raise ValueError(
            "Head-absorbed target requires a linear head (nn.Linear) on the model. "
            "Expected one of attributes: head, linear_head, classifier, readout."
        )

    weight = head_module.weight.detach().cpu().numpy().astype(np.float32, copy=False)
    if weight.ndim != 2:
        raise ValueError(f"Expected 2D linear-head weight, got shape {weight.shape}.")

    return weight


def build_target_observable(
    n_qubits: int,
    mode: str = "z_sum",
    target_label: int | None = None,
    digits_of_interest: list[int] | None = None,
    model=None,
) -> np.ndarray:
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
    elif mode == "class_z":
        if target_label is None:
            raise ValueError("target_label is required when mode='class_z'.")
        channel_idx = _resolve_target_channel(target_label, digits_of_interest)
        selected = [name_to_op[f"Z{channel_idx}"]]
    elif mode == "class_margin":
        if target_label is None:
            raise ValueError("target_label is required when mode='class_margin'.")
        channel_idx = _resolve_target_channel(target_label, digits_of_interest)
        n_outputs = len(digits_of_interest) if digits_of_interest else n_qubits
        if not (0 <= channel_idx < n_outputs):
            raise ValueError(
                f"Resolved channel index {channel_idx} out of range for n_outputs={n_outputs}."
            )

        positive = name_to_op[f"Z{channel_idx}"]
        other_indices = [idx for idx in range(n_outputs) if idx != channel_idx]
        if len(other_indices) == 0:
            return positive.astype(np.complex64, copy=False)

        negative = np.sum([name_to_op[f"Z{idx}"] for idx in other_indices], axis=0)
        return (positive - negative / len(other_indices)).astype(np.complex64, copy=False)
    elif mode == "head_absorbed":
        if model is None:
            raise ValueError("model is required when mode='head_absorbed'.")
        head_weight = _resolve_head_weight_matrix(model)
        if head_weight.shape[1] != n_qubits:
            raise ValueError(
                f"Linear-head input dim ({head_weight.shape[1]}) does not match n_qubits ({n_qubits})."
            )

        # Per-qubit contribution to the full logit vector: ||W[:, i]||_2.
        coeff = np.linalg.norm(head_weight, axis=0).astype(np.float32, copy=False)
        if np.allclose(coeff, 0.0):
            coeff = np.ones_like(coeff, dtype=np.float32)

        target = np.zeros_like(observables[0], dtype=np.complex64)
        for idx in range(n_qubits):
            target = target + np.complex64(coeff[idx]) * name_to_op[f"Z{idx}"]
        return target
    elif mode in name_to_op:
        selected = [name_to_op[mode]]
    else:
        raise ValueError(
            f"Unsupported target observable mode: {mode}. "
            f"Use one of {{'z_sum', 'zz_sum', 'class_z', 'class_margin', 'head_absorbed'}} "
            f"or an explicit basis name."
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

def generate_random_circuits(
    num_qubits: int,
    num_layers: int,
    num_circuits: int = 1,
):
    """Generate one or more random circuit designs.

    When ``num_circuits == 1`` this keeps the legacy return type
    ``(selected_single, selected_enta)``. When ``num_circuits > 1`` it returns
    a list of such pairs.
    """
    if num_circuits < 1:
        raise ValueError(f"num_circuits must be >= 1, got {num_circuits}.")

    circuits = []
    for _ in range(num_circuits):
        selected_single = []
        selected_enta = []

        for start_value in range(1, num_qubits + 1):
            single_row = [start_value] + [random.randint(0, 1) for _ in range(2 * num_layers)]
            enta_row = [start_value] + [random.randint(1, num_qubits) for _ in range(num_layers)]
            selected_single.append(single_row)
            selected_enta.append(enta_row)

        circuits.append((selected_single, selected_enta))

    if num_circuits == 1:
        return circuits[0]
    return circuits
