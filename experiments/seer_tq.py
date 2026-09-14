import os
import torch
import sys
import argparse
import logging
import pandas as pd
import numpy as np
import random

# Ensure project root (containing `utils`, `models`, etc.) is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    
from utils.dataset import SEERDataset, generate_seer_concept_dataset
from models.FusionModel import QNet, single_enta_to_design
from pathlib import Path
from torch.utils.data import DataLoader
from utils.plot import plot_seer_global_explanation, plot_seer_feature_importance
from explanations.concept import CAR, CAV
from explanations.feature import CARFeatureImportance
from utils.quantum_circuit_helpers import (
    SeerConceptModelWrapper,
    _compute_pullback_coefficients_over_set,
    _select_top_observable_indices,
    generate_random_circuits,
)
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from Arguments import Arguments

torch.manual_seed(52)
np.random.seed(52)


def _load_seer_concept_model(
    model_name: str,
    random_seed: int,
    repr_mode: str,
    pullback_target_mode: str = "head_absorbed",
    random_top_n: int | None = None,
    concept_layer: int = -1,
    model_dir: Path = Path.cwd() / "results/seer_tq",
) -> SeerConceptModelWrapper:
    device = torch.device("cpu")
    model_path = model_dir / model_name / "vqc_model.pt"

    base_model = QNet(seer_args, design)

    if model_path.exists():
        load_result = base_model.load_state_dict(
            torch.load(model_path, map_location=device),
            strict=False,
        )
        if any(key.startswith("head.") for key in getattr(load_result, "missing_keys", [])):
            logging.warning(
                "Loaded checkpoint is missing linear-head parameters. "
                "The new head will be randomly initialized; retrain this model for valid head-absorbed pullback analysis."
            )
    else:
        logging.info("Loaded shuffled model as no checkpoint was found.")
        
    base_model.to(device)
    base_model.eval()
    return SeerConceptModelWrapper(
        base_model,
        repr_mode=repr_mode,
        random_seed=random_seed,
        pullback_target_mode=pullback_target_mode,
        random_top_n=random_top_n,
        concept_layer=concept_layer,
    )


def _collect_dataset_inputs(
    dataset: torch.utils.data.Dataset,
    n_samples: int = 1024,
) -> torch.Tensor:
    n_samples = max(1, int(n_samples))
    loader = DataLoader(dataset, batch_size=min(256, len(dataset)), shuffle=False)
    all_x = []
    collected = 0
    for batch in loader:
        if isinstance(batch, (tuple, list)):
            x_batch = batch[0].detach().cpu()
        else:
            x_batch = batch.detach().cpu()

        remaining = n_samples - collected
        if remaining <= 0:
            break
        take = min(remaining, x_batch.shape[0])
        all_x.append(x_batch[:take])
        collected += take

        if collected >= n_samples:
            break

    if len(all_x) == 0:
        raise ValueError("Dataset appears empty; cannot collect inputs for pullback selection.")
    return torch.cat(all_x, dim=0)


def train_model(
    random_seed: int,
    batch_size: int,
    latent_dim: int,
    model_name: str,
    test_fraction: float = 0.1,
    model_dir: Path = Path.cwd() / "results/seer_tq",
    data_dir: Path = Path.cwd() / "data/seer",
):
    assert 0 < test_fraction < 1
    logging.info("Now fitting a SEER classifier")
    torch.manual_seed(random_seed)
    device =  torch.device("cpu")

    model_dir = model_dir / model_name
    if not model_dir.exists():
        os.makedirs(model_dir)

    train_data = SEERDataset(
        str(data_dir / "seer.csv"), random_seed, train=True, test_fraction=test_fraction
    )
    test_data = SEERDataset(
        str(data_dir / "seer.csv"),
        random_seed,
        train=False,
        test_fraction=test_fraction,
    )
    train_loader = DataLoader(train_data, batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size, shuffle=False)

    model = QNet(seer_args, design)
    try:
        model.load_state_dict(torch.load(model_dir / f"vqc_model.pt"), strict=False)
    except:
        pass
    model.fit(device, train_loader, test_loader, model_dir, n_epoch=50, patience=15)


def concept_accuracy(
    random_seed: int,
    batch_size: int,
    latent_dim: int,
    model_name: str,
    repr_mode: str = "pullback",
    pullback_top_n: int = 20,
    pullback_selection_n: int = 1024,
    concept_layer: int = -1,
    test_fraction: float = 0.1,
    model_dir: Path = Path.cwd() / "results/seer_tq",
    data_dir: Path = Path.cwd() / "data/seer",
    save_dir: Path = Path.cwd() / "results/seer_tq/concept_accuracy",
):
    torch.manual_seed(random_seed)
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device("cpu")
    if not save_dir.exists():
        os.makedirs(save_dir)

    # Load data
    train_data = SEERDataset(
        str(data_dir / "seer.csv"),
        random_seed,
        train=True,
        test_fraction=test_fraction,
        load_concept_labels=True,
    )
    test_data = SEERDataset(
        str(data_dir / "seer.csv"),
        random_seed,
        train=False,
        test_fraction=test_fraction,
        load_concept_labels=False,
    )

    test_loader = DataLoader(test_data, batch_size)

    # Load model
    model = _load_seer_concept_model(
        model_name,
        random_seed,
        repr_mode=repr_mode,
        pullback_target_mode="z_sum",
        concept_layer=concept_layer,
        model_dir=model_dir,
    )

    test_loss, test_acc = model.model.test_epoch(device, test_loader)
    logging.info(f"Circuit {circuit_idx}: test_loss={test_loss:.4f}, test_acc={test_acc:.4f}")
    
    concept_pullback_indices = [None for _ in range(5)]
    results_data = []
    car_classifiers = [CAR(device, batch_size, kernel="linear") for _ in range(5)]
    cav_classifiers = [CAV(device, batch_size) for _ in range(5)]
    for concept_id in range(5):
        logging.info(f"Now fitting a CAR classifier for Grade {concept_id+1} patients")
        X_train, C_train = generate_seer_concept_dataset(
            train_data, concept_id, 250, random_seed
        )
        if repr_mode == "pullback":
            pos_mask = (C_train == 1)
            x_concept_pos = X_train[pos_mask]
            selected_idx, _ = _select_top_observable_indices(
                model,
                x_concept_pos.detach().cpu().numpy().astype(np.float32, copy=False),
                repr_mode=repr_mode,
                random_seed=int(random_seed),
                top_n=int(pullback_top_n),
                target_mode=model.pullback_target_mode,
                aggregate="mean_abs",
            )
            model.set_pullback_observables(selected_idx)
            concept_pullback_indices[concept_id] = np.asarray(selected_idx, dtype=np.int64)
            logging.info(
                f"Grade {concept_id+1}: selected {len(selected_idx)} pullback observables from "
                f"{int(pos_mask.sum().item())} positive samples."
            )
        X_train = X_train.to(device)
        H_train = model.input_to_representation(X_train).detach().cpu().numpy()
        car = car_classifiers[concept_id]
        car.fit(H_train, C_train.numpy())
        cav = cav_classifiers[concept_id]
        cav.fit(H_train, C_train.numpy())

        X_test, C_test = generate_seer_concept_dataset(
            test_data, concept_id, 50, random_seed
        )
        X_test = X_test.to(device)
        if repr_mode == "pullback":
            model.set_pullback_observables(concept_pullback_indices[concept_id])
        H_test = model.input_to_representation(X_test).detach().cpu().numpy()
        results_data.append(
            [
                f"Grade {concept_id + 1}",
                "CAR",
                accuracy_score(C_test, car.predict(H_test)),
            ]
        )
        results_data.append(
            [
                f"Grade {concept_id + 1}",
                "CAV",
                accuracy_score(C_test, cav.predict(H_test)),
            ]
        )
    results_df = pd.DataFrame(results_data, columns=["Concept", "Method", "Test ACC"])
    logging.info(f"Saving results in {save_dir}")
    results_df.to_csv(save_dir / "metrics.csv")

    grade_order = [f"Grade {i+1}" for i in range(5)]
    method_grade_df = results_df.pivot(index="Method", columns="Concept", values="Test ACC")
    method_grade_df = method_grade_df.reindex(index=["CAR", "CAV"], columns=grade_order)
    return {
        method: {
            f"grade {grade_idx+1}": float(method_grade_df.loc[method, grade_name])
            for grade_idx, grade_name in enumerate(grade_order)
        }
        for method in ["CAR", "CAV"]
    }


def global_explanations(
    random_seed: int,
    batch_size: int,
    latent_dim: int,
    plot: bool,
    model_name: str,
    repr_mode: str = "pullback",
    pullback_top_n: int = 20,
    pullback_selection_n: int = 1024,
    concept_layer: int = -1,
    test_fraction: float = 0.1,
    model_dir: Path = Path.cwd() / "results/seer_tq",
    data_dir: Path = Path.cwd() / "data/seer",
    save_dir: Path = Path.cwd() / "results/seer_tq/global_explanations",
):
    torch.manual_seed(random_seed)
    device =torch.device("cpu")

    if not save_dir.exists():
        os.makedirs(save_dir)

    # Load data
    train_data = SEERDataset(
        str(data_dir / "seer.csv"),
        random_seed,
        train=True,
        test_fraction=test_fraction,
        load_concept_labels=True,
    )
    test_data = SEERDataset(
        str(data_dir / "seer.csv"),
        random_seed,
        train=False,
        test_fraction=test_fraction,
        load_concept_labels=False,
    )
    test_loader = DataLoader(test_data, batch_size)

    # Load model
    model = _load_seer_concept_model(
        model_name,
        random_seed,
        repr_mode=repr_mode,
        pullback_target_mode="z_sum",
        concept_layer=concept_layer,
        model_dir=model_dir,
    )

    test_loss, test_acc = model.model.test_epoch(device, test_loader)
    logging.info(f"Circuit {circuit_idx}: test_loss={test_loss:.4f}, test_acc={test_acc:.4f}")

    car_classifiers = [CAR(device, batch_size) for _ in range(5)]
    concept_pullback_indices = [None for _ in range(5)]
    for concept_id in range(5):
        logging.info(f"Now fitting a CAR classifier for Grade {concept_id+1} patients")
        X_train, C_train = generate_seer_concept_dataset(
            train_data, concept_id, 250, random_seed
        )
        if repr_mode == "pullback":
            pos_mask = (C_train == 1)
            x_concept_pos = X_train[pos_mask]
            selected_idx, _ = _select_top_observable_indices(
                model,
                x_concept_pos.detach().cpu().numpy().astype(np.float32, copy=False),
                repr_mode=repr_mode,
                random_seed=int(random_seed),
                top_n=int(pullback_top_n),
                target_mode=model.pullback_target_mode,
                aggregate="mean_abs",
            )
            model.set_pullback_observables(selected_idx)
            concept_pullback_indices[concept_id] = np.asarray(selected_idx, dtype=np.int64)
            logging.info(
                f"Grade {concept_id+1}: selected {len(selected_idx)} pullback observables from "
                f"{int(pos_mask.sum().item())} positive samples."
            )
        X_train = X_train.to(device)
        H_train = model.input_to_representation(X_train).detach().cpu().numpy()
        car = car_classifiers[concept_id]
        car.fit(H_train, C_train.numpy())

    logging.info("Producing global explanations for the test set")
    results_data = []
    for X_test, Y_test in tqdm(test_loader, unit="batch", leave=False):
        X_test = X_test.to(device)
        if repr_mode == "pullback":
            pred_concepts = []
            for concept_id, car in enumerate(car_classifiers):
                model.set_pullback_observables(concept_pullback_indices[concept_id])
                h_test_concept = model.input_to_representation(X_test).detach().cpu().numpy()
                pred_concepts.append(car.predict(h_test_concept))
        else:
            H_test = model.input_to_representation(X_test).detach().cpu().numpy()
            pred_concepts = [car.predict(H_test) for car in car_classifiers]
        results_data += [
            ["TCAR", label.item()]
            + [pred_concept[example_id] for pred_concept in pred_concepts]
            for example_id, label in enumerate(Y_test)
        ]
    results_df = pd.DataFrame(
        results_data, columns=["Method", "Class"] + [f"Grade {i+1}" for i in range(5)]
    )
    logging.info(f"Saving results in {save_dir}")
    results_df.to_csv(save_dir / "metrics.csv", index=False)
    if plot:
        plot_seer_global_explanation(
            save_dir,
            filename=f"seer_global_{repr_mode}_layer_{concept_layer}.pdf",
        )

    grade_cols = [f"Grade {i+1}" for i in range(5)]
    dies = {}
    survives = {}
    for grade in grade_cols:
        grade_active = results_df[grade] > 0.5
        if grade_active.any():
            dies[grade] = float((results_df.loc[grade_active, "Class"] == 0).mean())
            survives[grade] = float((results_df.loc[grade_active, "Class"] == 1).mean())
        else:
            dies[grade] = 0.0
            survives[grade] = 0.0
    delta = {grade: dies[grade] - survives[grade] for grade in grade_cols}
    return {
        "dies": dies,
        "survives": survives,
        "delta": delta,
    }


def quantum_concept_importance(
    random_seed: int,
    batch_size: int,
    latent_dim: int,
    model_name: str,
    repr_mode: str = "pullback",
    pullback_top_n: int = 20,
    concept_layer: int = -1,
    test_fraction: float = 0.1,
    model_dir: Path = Path.cwd() / "results/seer_tq",
    data_dir: Path = Path.cwd() / "data/seer",
    save_dir: Path = Path.cwd() / "results/seer_tq/quantum_concept_importance",
):
    torch.manual_seed(random_seed)
    device = torch.device("cpu")

    if not save_dir.exists():
        os.makedirs(save_dir)

    train_data = SEERDataset(
        str(data_dir / "seer.csv"),
        random_seed,
        train=True,
        test_fraction=test_fraction,
        load_concept_labels=True,
    )
    test_data = SEERDataset(
        str(data_dir / "seer.csv"),
        random_seed,
        train=False,
        test_fraction=test_fraction,
        load_concept_labels=False,
    )
    test_loader = DataLoader(test_data, batch_size)

    # Concept-side model uses z_sum selection to construct O_C.
    # Keep pullback feature path active even when repr_mode='random' so we can
    # explicitly set one shared random observable subset for all concepts.
    model = _load_seer_concept_model(
        model_name,
        random_seed,
        repr_mode="pullback",
        pullback_target_mode="z_sum",
        concept_layer=concept_layer,
        model_dir=model_dir,
    )

    test_loss, test_acc = model.model.test_epoch(device, test_loader)
    logging.info(f"Circuit {circuit_idx}: test_loss={test_loss:.4f}, test_acc={test_acc:.4f}")

    shared_random_idx = None
    if repr_mode == "random":
        x_seed = _collect_dataset_inputs(train_data, n_samples=1)
        shared_random_idx, shared_random_names = _select_top_observable_indices(
            model,
            x_seed.detach().cpu().numpy().astype(np.float32, copy=False),
            repr_mode="random",
            random_seed=int(random_seed),
            top_n=int(pullback_top_n),
            target_mode="z_sum",
            aggregate="mean_abs",
        )
        shared_random_idx = np.asarray(shared_random_idx, dtype=np.int64)
        logging.info(
            "Shared random observables for all concepts: "
            f"{shared_random_names}"
        )

    concept_pullback_indices = []
    concept_observable_coeffs = []
    for concept_id in range(5):
        logging.info(f"Building O_C for Grade {concept_id+1} using linear SVM")
        X_train, C_train = generate_seer_concept_dataset(
            train_data, concept_id, 250, random_seed
        )

        if repr_mode == "pullback":
            pos_mask = C_train == 1
            x_concept_pos = X_train[pos_mask]
            selected_idx, _ = _select_top_observable_indices(
                model,
                x_concept_pos.detach().cpu().numpy().astype(np.float32, copy=False),
                repr_mode="pullback",
                random_seed=int(random_seed),
                top_n=int(pullback_top_n),
                target_mode="z_sum",
                aggregate="mean_abs",
            )
            selected_idx = np.asarray(selected_idx, dtype=np.int64)
        elif repr_mode == "random":
            selected_idx = shared_random_idx
        else:
            raise ValueError(
                "quantum_concept_importance expects repr_mode in {'pullback', 'random'}."
            )

        model.set_pullback_observables(selected_idx)
        concept_pullback_indices.append(selected_idx)

        H_train = model.input_to_representation(X_train.to(device)).detach().cpu().numpy()
        car = CAR(device, batch_size, kernel="linear")
        car.fit(H_train, C_train.numpy())
        coeff = np.asarray(car.classifier.coef_, dtype=np.float32).reshape(-1)
        concept_observable_coeffs.append(coeff)

    # Decision-side O_eff uses head-absorbed pullback (patched as die-survive margin).
    concept_scores = {concept_id: [] for concept_id in range(5)}
    for X_test, _Y_test in tqdm(test_loader, unit="batch", leave=False):
        x_np = X_test.detach().cpu().numpy().astype(np.float32, copy=False)
        eff_coeff_full, _ = _compute_pullback_coefficients_over_set(
            model,
            x_np,
            target_mode="head_absorbed",
        )

        for concept_id in range(5):
            idx = concept_pullback_indices[concept_id]
            o_eff = eff_coeff_full[:, idx]
            o_c = concept_observable_coeffs[concept_id]

            denom = np.linalg.norm(o_eff, axis=1) * max(np.linalg.norm(o_c), 1e-12)
            denom = np.maximum(denom, 1e-12)
            alpha = np.einsum("bi,i->b", o_eff, o_c) / denom
            concept_scores[concept_id].append(alpha.astype(np.float32, copy=False))

    results_data = []
    summary = {}
    for concept_id in range(5):
        all_alpha = np.concatenate(concept_scores[concept_id], axis=0)
        score = float(np.mean(all_alpha)) if all_alpha.size > 0 else float("nan")
        concept_name = f"Grade {concept_id+1}"
        summary[concept_name] = score
        results_data.append([concept_name, score])

    results_df = pd.DataFrame(results_data, columns=["Concept", "HS Alignment"])
    logging.info(f"Saving quantum concept importance results in {save_dir}")
    results_df.to_csv(save_dir / "metrics.csv", index=False)
    return summary


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="global_explanations")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(1, 11)))
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--latent_dim", type=int, default=50)
    parser.add_argument(
        "--repr",
        type=str,
        default="pullback",
        choices=["pullback", "local_x", "local_y", "local_z", "random"],
    )
    parser.add_argument("--train", action="store_true", default=False)
    parser.add_argument("--plot", action="store_true", default=True)
    parser.add_argument("--pullback_top_n", type=int, default=20)
    parser.add_argument("--pullback_selection_n", type=int, default=1024)
    parser.add_argument("--concept_layer", type=int, default=-1)
    parser.add_argument("--n_circuits", type=int, default=10)
    args = parser.parse_args()
    args.concept_layer = 1
    args.repr = "random"
    args.pullback_top_n = 8

    # Explicit task-style configuration, aligned with mnist_tq_poison pattern.
    SEER_TASK = {
        "task": "SEER",
        "option": "mix_reg",
        "n_qubits": 6,
        "n_layers": 3,
        "fold": 1,
        "backend": "tq",
    }

    seer_args = Arguments(task=SEER_TASK["task"], fold=SEER_TASK["fold"])
    seer_args.option = SEER_TASK["option"]
    seer_args.n_qubits = int(SEER_TASK["n_qubits"])
    seer_args.n_layers = int(SEER_TASK["n_layers"])
    seer_args.backend = SEER_TASK["backend"]

    n_layers = seer_args.n_layers
    n_qubits = seer_args.n_qubits
    n_circuits = max(1, int(args.n_circuits))
    model_root = Path.cwd() / "results" / "seer_tq"
    all_concept_rows = []
    all_delta_rows = []
    all_pair_rows = []
    all_qci_rows = []
   
    random.seed(12) #22
    circuits = generate_random_circuits(n_qubits, n_layers, n_circuits)

    # for circuit_idx, (single, enta) in enumerate(circuits, start=1):        
    #     arch_code = [seer_args.n_qubits, seer_args.n_layers]
    #     design = single_enta_to_design(single, enta, arch_code)

    #     model_name = f"model_{args.latent_dim}_circuit_{circuit_idx}"

    #     logging.info(
    #         f"Running SEER experiment {circuit_idx}/{n_circuits} with model {model_name}"
    #     )

    #     train_model(
    #         args.seeds[0],
    #         args.batch_size,
    #         args.latent_dim,
    #         model_name=model_name,
    #         model_dir=model_root,
    #     )

    for circuit_idx, (single, enta) in enumerate(circuits, start=1):        
            arch_code = [seer_args.n_qubits, seer_args.n_layers]
            design = single_enta_to_design(single, enta, arch_code)
    
            model_name = f"model_{args.latent_dim}_circuit_{circuit_idx}"
            save_dir = model_root / "global_explanations" / f"seer_global_{circuit_idx}"
            concept_save_dir = model_root / "concept_accuracy" / f"circuit_{circuit_idx}"
            qci_save_dir = model_root / "quantum_concept_importance" / f"circuit_{circuit_idx}"

            logging.info(
                f"Running SEER experiment {circuit_idx}/{n_circuits} with model {model_name}"
            )        

            # qci_stats = quantum_concept_importance(
            #     args.seeds[0],
            #     args.batch_size,
            #     args.latent_dim,
            #     model_name=model_name,
            #     repr_mode=args.repr,
            #     pullback_top_n=args.pullback_top_n,
            #     concept_layer=args.concept_layer,
            #     save_dir=qci_save_dir,
            #     model_dir=model_root,
            # )

            # qci_row = {
            #     "circuit_id": circuit_idx,
            # }
            # for grade in range(1, 6):
            #     qci_row[f"grade {grade}"] = float(qci_stats[f"Grade {grade}"])
            # all_qci_rows.append(qci_row)

            # global_stats = global_explanations(
            #     args.seeds[0],
            #     args.batch_size,
            #     args.latent_dim,
            #     args.plot,
            #     model_name=model_name,
            #     repr_mode=args.repr,
            #     pullback_top_n=args.pullback_top_n,
            #     pullback_selection_n=args.pullback_selection_n,
            #     concept_layer=args.concept_layer,
            #     save_dir=save_dir,
            #     model_dir=model_root,
            # )            

            # pair_row = {
            #     "circuit_id": circuit_idx,
            # }
            # for grade in range(1, 6):
            #     die_score = float(global_stats["dies"][f"Grade {grade}"])
            #     survive_score = float(global_stats["survives"][f"Grade {grade}"])
            #     pair_row[f"grade {grade}"] = f"[{die_score:.2f}, {survive_score:.2f}]"
            # all_pair_rows.append(pair_row)

            concept_stats = concept_accuracy(
                args.seeds[0],
                args.batch_size,
                args.latent_dim,
                model_name=model_name,
                repr_mode=args.repr,
                pullback_top_n=args.pullback_top_n,
                pullback_selection_n=args.pullback_selection_n,
                concept_layer=args.concept_layer,
                model_dir=model_root,
                save_dir=concept_save_dir,
            )

            for method in ["CAR", "CAV"]:
                row = {
                    "circuit_id": circuit_idx,
                    "method": method,
                }
                for grade in range(1, 6):
                    row[f"grade {grade}"] = float(concept_stats[method][f"grade {grade}"])
                all_concept_rows.append(row)

    if all_concept_rows:
        concept_df = pd.DataFrame(
            all_concept_rows,
            columns=[
                "circuit_id",
                "method",
                "grade 1",
                "grade 2",
                "grade 3",
                "grade 4",
                "grade 5",
            ],
        )
        concept_df.to_csv(model_root / "concept_accuracy_all_circuits.csv", index=False)

    if all_pair_rows:
        pair_df = pd.DataFrame(
            all_pair_rows,
            columns=[
                "circuit_id",
                "grade 1",
                "grade 2",
                "grade 3",
                "grade 4",
                "grade 5",
            ],
        )
        pair_df.to_csv(
            model_root
            / f"all_circuits_{args.repr}_layer_{args.concept_layer}.csv",
            index=False,
        )

    if all_qci_rows:
        qci_df = pd.DataFrame(
            all_qci_rows,
            columns=[
                "circuit_id",
                "grade 1",
                "grade 2",
                "grade 3",
                "grade 4",
                "grade 5",
            ],
        )
        qci_df.to_csv(
            model_root
            / f"qci_all_circuits_{args.repr}_layer_{args.concept_layer}.csv",
            index=False,
        )   
