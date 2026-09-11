import argparse
import copy
import itertools
import logging
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# Ensure project root (containing `utils`, `models`, etc.) is on sys.path.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
	sys.path.insert(0, PROJECT_ROOT)

# Ensure legacy imports inside utils/poison.py can resolve dataset.py.
UTILS_ROOT = os.path.join(PROJECT_ROOT, "utils")
if UTILS_ROOT not in sys.path:
	sys.path.append(UTILS_ROOT)

from explanations.concept import CAR, CAV
from sklearn.metrics import accuracy_score
from utils.dataset import MNISTDataLoaders, generate_mnist_concept_dataset
from utils.poison import poison as poison_loader_fn
from utils.plot import plot_global_explanation


concept_to_class = {
	"Loop": [0, 2, 6, 8, 9],
	"Vertical Line": [1, 4, 7],
	"Horizontal Line": [4, 5, 7],
	"Curvature": [0, 2, 3, 5, 6, 8, 9],
}


class MLP(nn.Module):
	"""MLP(24, 5, 4) with concept representation at hidden layer (size 5)."""

	def __init__(self):
		super().__init__()
		self.pool = nn.AdaptiveAvgPool2d((4, 6))
		self.flatten = nn.Flatten()
		self.fc1 = nn.Linear(24, 5)
		self.act = nn.ReLU()
		self.fc2 = nn.Linear(5, len(args.nums))

	def _to_24d(self, x: torch.Tensor) -> torch.Tensor:
		x = self.pool(x)
		x = self.flatten(x)
		return x

	def input_to_representation(self, x: torch.Tensor) -> torch.Tensor:
		x24 = self._to_24d(x)
		h = self.act(self.fc1(x24))
		return h

	def representation_to_output(self, h: torch.Tensor, x: torch.Tensor | None = None) -> torch.Tensor:
		return self.fc2(h)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		h = self.input_to_representation(x)
		return self.representation_to_output(h)


def _safe_corrcoef(x_vals: np.ndarray, y_vals: np.ndarray) -> float:
	x_vals = np.asarray(x_vals, dtype=np.float64)
	y_vals = np.asarray(y_vals, dtype=np.float64)
	if x_vals.size < 2 or y_vals.size < 2:
		return float("nan")
	if np.isclose(np.std(x_vals), 0.0) or np.isclose(np.std(y_vals), 0.0):
		return float("nan")
	return float(np.corrcoef(x_vals, y_vals)[0, 1])


def _restore_mnist_loader_labels(labels: np.ndarray, digits_of_interest: list[int]) -> np.ndarray:
	y = np.asarray(labels, dtype=np.int64)
	if y.size == 0:
		return y
	if np.all((0 <= y) & (y < len(digits_of_interest))):
		mapping = np.asarray(digits_of_interest, dtype=np.int64)
		return mapping[y]
	return y


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
		return float("nan"), float("nan"), {}, {}

	scores_df = pd.DataFrame(
		scores_data,
		columns=["Method", "Class", "Concept", "Score"],
	)

	true_scores = scores_df.loc[scores_df.Method == "True Prop."]["Score"].to_numpy()
	tcar_scores = scores_df.loc[scores_df.Method == "TCAR"]["Score"].to_numpy()
	tcav_scores = scores_df.loc[scores_df.Method == "TCAV"]["Score"].to_numpy()

	tcar_corr = _safe_corrcoef(tcar_scores, true_scores)
	tcav_corr = _safe_corrcoef(tcav_scores, true_scores)

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


def _sample_balanced_indices(y_train: np.ndarray, n: int) -> np.ndarray:
	labels = np.asarray(y_train, dtype=np.int64)
	positive_indices = np.flatnonzero(labels == 1)
	negative_indices = np.flatnonzero(labels == 0)
	n_take = min(n, len(positive_indices), len(negative_indices))
	if n_take == 0:
		raise ValueError("Unable to sample balanced indices: one class is empty.")

	selected_indices = np.concatenate(
		[
			np.random.choice(positive_indices, size=n_take, replace=False),
			np.random.choice(negative_indices, size=n_take, replace=False),
		]
	)
	np.random.shuffle(selected_indices)
	return selected_indices


def _train_classifier(
	model: MLP,
	train_loader: DataLoader,
	epochs: int,
	lr: float,
	device: torch.device,
) -> None:
	model.train()
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)

	for _ in range(epochs):
		for feed_dict in train_loader:
			images = feed_dict["image"].to(device)
			targets = feed_dict["digit"].to(device)
			optimizer.zero_grad()
			logits = model(images)
			loss = criterion(logits, targets)
			loss.backward()
			optimizer.step()


def _build_global_explanation_bank(
	model: MLP,
	train_loader: DataLoader,
	selected_digits: list[int],
	random_seed: int,
) -> dict:
	device = torch.device("cpu")
	torch.manual_seed(random_seed)
	np.random.seed(random_seed)

	active_concept_to_class = {}
	allowed_digits = set(selected_digits)
	for concept_name, class_ids in concept_to_class.items():
		filtered_ids = [int(c) for c in class_ids if int(c) in allowed_digits]
		if filtered_ids:
			active_concept_to_class[concept_name] = filtered_ids

	if not active_concept_to_class:
		raise ValueError(
			f"No concepts left after filtering with digits_of_interest={selected_digits}"
		)

	car_classifiers = []
	cav_classifiers = []
	concept_names = list(active_concept_to_class.keys())
	model.eval()
	for concept_name in concept_names:
		logging.info(f"Fitting classifiers for {concept_name} on seed {random_seed}")

		x_train, y_concept = generate_mnist_concept_dataset(
			active_concept_to_class[concept_name], train_loader, 200, random_seed
		)
		h_repr = model.input_to_representation(
			torch.from_numpy(x_train).to(device)
		).detach().cpu().numpy().astype(np.float32)

		car = CAR(device)
		cav = CAV(device)
		car.fit(h_repr, y_concept)
		cav.fit(h_repr, y_concept)
		car_classifiers.append(car)
		cav_classifiers.append(cav)

	return {
		"selected_digits": selected_digits,
		"active_concept_to_class": active_concept_to_class,
		"concept_names": concept_names,
		"car_classifiers": car_classifiers,
		"cav_classifiers": cav_classifiers,
	}


def concept_accuracy(
	model: MLP,
	random_seed: int,
	concept_bank: dict,
	eval_loader: DataLoader,
) -> tuple[float, float, dict[str, dict[str, float]]]:
	device = torch.device("cpu")
	torch.manual_seed(random_seed)
	np.random.seed(random_seed)
	model.eval()

	if concept_bank is None:
		raise ValueError("concept_bank must be provided to evaluate concept accuracy.")

	concept_map = concept_bank["active_concept_to_class"]
	concept_names = concept_bank["concept_names"]
	car_classifiers = concept_bank["car_classifiers"]
	cav_classifiers = concept_bank["cav_classifiers"]
	concept_to_car = {
		concept_name: car
		for concept_name, car in zip(concept_names, car_classifiers)
	}
	concept_to_cav = {
		concept_name: cav
		for concept_name, cav in zip(concept_names, cav_classifiers)
	}

	tcar_scores = []
	tcav_scores = []
	concept_scores: dict[str, dict[str, float]] = {}

	for concept_name in concept_names:
		x_test, y_test = generate_mnist_concept_dataset(
			concept_map[concept_name], eval_loader, 50, random_seed
		)
		h_test = model.input_to_representation(
			torch.from_numpy(x_test).to(device)
		).detach().cpu().numpy().astype(np.float32)

		tcar_score = float(accuracy_score(y_test, concept_to_car[concept_name].predict(h_test)))
		tcav_score = float(accuracy_score(y_test, concept_to_cav[concept_name].predict(h_test)))
		tcar_scores.append(tcar_score)
		tcav_scores.append(tcav_score)
		concept_scores[concept_name] = {
			"TCAR": tcar_score,
			"TCAV": tcav_score,
		}

	mean_tcar = float(np.mean(tcar_scores)) if len(tcar_scores) else float("nan")
	mean_tcav = float(np.mean(tcav_scores)) if len(tcav_scores) else float("nan")
	return mean_tcar, mean_tcav, concept_scores


def global_explanations(
	model: MLP,
	random_seed: int,
	batch_size: int,
	plot: bool,
	eval_digits: list[int],
	save_dir: Path,
	concept_bank: dict | None = None,
) -> tuple[float, float, dict[int, float], dict[int, float]]:
	device = torch.device("cpu")
	torch.manual_seed(random_seed)
	np.random.seed(random_seed)
	save_dir.mkdir(parents=True, exist_ok=True)
	model.eval()

	if concept_bank is None:
		raise ValueError("concept_bank must be provided to preserve fixed-bank generalization.")

	concept_names = concept_bank["concept_names"]
	active_concept_to_class = concept_to_class
	car_classifiers = concept_bank["car_classifiers"]
	cav_classifiers = concept_bank["cav_classifiers"]

	eval_args = argparse.Namespace(**vars(args))
	eval_args.digits_of_interest = list(eval_digits)
	eval_args.batch_size = int(batch_size)
	_, _, test_loader = MNISTDataLoaders(eval_args, eval_args.task)

	test_loader = DataLoader(
		Subset(test_loader.dataset, range(min(1000, len(test_loader.dataset)))),
		batch_size=batch_size,
		shuffle=False,
		pin_memory=getattr(test_loader, "pin_memory", False),
		num_workers=getattr(test_loader, "num_workers", 0),
		drop_last=False,
	)

	logging.info(f"Global explanations evaluate over digits: {tuple(eval_digits)}")

	results_data = []
	for feed_dict in tqdm(test_loader, unit="batch", leave=False):
		x_test = feed_dict["image"].to(device)
		y_test = feed_dict["digit"]

		h_test = model.input_to_representation(x_test).detach().cpu().numpy().astype(np.float32)
		y_test_np = y_test.detach().cpu().numpy().astype(np.int64)
		y_test_digits = _restore_mnist_loader_labels(y_test_np, eval_digits)

		car_preds = [car.predict(h_test) for car in car_classifiers]
		cav_preds = [cav.predict(h_test) for cav in cav_classifiers]

		targets = [
			[int(label in active_concept_to_class[concept]) for label in y_test_digits]
			for concept in concept_names
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

	results_df = pd.DataFrame(results_data, columns=["Method", "Class"] + concept_names)

	tcar_corr, tcav_corr, tcar_corr_per_class, tcav_corr_per_class = _compute_global_explanation_correlations(
		results_df,
		concept_names,
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

	pd.DataFrame(summary_rows).to_csv(save_dir / "metrics.csv", index=False)

	if plot:
		plot_global_explanation(save_dir, "mnist")

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

	colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
	pairs = corr_df["pair"].unique().tolist()

	def _plot_single(method_col: str, title_prefix: str, file_name: str) -> None:
		plt.style.use("ggplot")
		fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

		df_x = corr_df[corr_df["y_alpha"] == 0].sort_values("x_alpha")
		for i_idx, pair in enumerate(pairs):
			color = colors[i_idx % len(colors)]
			data = df_x[df_x["pair"] == pair]
			ax1.plot(
				data["x_alpha"],
				data[method_col],
				marker="o",
				linestyle="-",
				color=color,
				label=pair,
			)
		ax1.set_title("Feature Randomization")
		ax1.set_xlabel("x_alpha")
		ax1.set_ylabel("Correlation")
		ax1.set_ylim(-1.05, 1.05)
		ax1.legend(fontsize="small")

		df_y = corr_df[corr_df["x_alpha"] == 0].sort_values("y_alpha")
		for i_idx, pair in enumerate(pairs):
			color = colors[i_idx % len(colors)]
			data = df_y[df_y["pair"] == pair]
			ax2.plot(
				data["y_alpha"],
				data[method_col],
				marker="s",
				linestyle="-",
				color=color,
				label=pair,
			)
		ax2.set_title("Label Flipping")
		ax2.set_xlabel("y_alpha")
		ax2.set_ylabel("Correlation")
		ax2.set_ylim(-1.05, 1.05)
		ax2.legend(fontsize="small")

		fig.suptitle(title_prefix)
		plt.tight_layout()
		save_path = output_dir / file_name
		plt.savefig(save_path)
		plt.close(fig)
		logging.info(f"Saved correlation curve figure: {save_path}")

	_plot_single(
		method_col="tcar_corr",
		title_prefix="TCAR Correlation vs Alpha",
		file_name="tcar_correlation_curves.png",
	)
	_plot_single(
		method_col="tcav_corr",
		title_prefix="TCAV Correlation vs Alpha",
		file_name="tcav_correlation_curves.png",
	)


if __name__ == "__main__":
	logging.basicConfig(
		level=logging.INFO,
		format="%(asctime)s - %(levelname)s - %(message)s",
	)

	torch.manual_seed(42)
	np.random.seed(42)

	args = argparse.Namespace()
	args.seeds = [1, 2, 3, 4, 5]
	args.batch_size = 120
	args.train_epochs = 20
	args.poison_epochs = 8
	args.lr = 1e-3
	args.plot = False
	args.run_all_poison_steps = True
	args.poison_alphas = [round(a, 1) for a in np.arange(0.0, 1.0, 0.1).tolist()]
	args.poison_axis = "both"
	args.nums = [0, 1, 2, 3, 4, 5]
	args.eval_digits = [6, 7, 8, 9]
	args.task = "MNIST_4"
	args.init_seed = 42

	# Required by MNISTDataLoaders/TQMNIST API.
	args.train_valid_split_ratio = [0.9, 0.1]
	args.center_crop = 24
	args.resize = 28
	args.digits_of_interest = list(args.nums)
	args.n_train_samples = 12000
	args.n_valid_samples = 1200
	args.n_test_samples = 1200
	args.same_n_samples_each_class = True

	nums = tuple(args.nums)
	pair_label = f"({nums[0]}, {nums[1]})"
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
		runs.append(("single", 0.0, 0.0))

	# Build one fixed random initialization and reuse it for all runs.
	torch.manual_seed(args.init_seed)
	init_model = MLP().to(device)
	init_state_dict = copy.deepcopy(init_model.state_dict())

	clean_train_loader, _, clean_test_loader = MNISTDataLoaders(args, args.task)
	clean_model = MLP().to(device)
	clean_model.load_state_dict(init_state_dict)
	_train_classifier(clean_model, clean_train_loader, args.train_epochs, args.lr, device)

	seed_quality_rows = []
	best_seed = None
	best_tcar = -np.inf
	best_concept_bank = None

	for seed in args.seeds:
		seed_concept_bank = _build_global_explanation_bank(
			clean_model,
			clean_train_loader,
			selected_digits=list(args.nums),
			random_seed=int(seed),
		)

		mean_tcar, mean_tcav, concept_scores = concept_accuracy(
			clean_model,
			int(seed),
			concept_bank=seed_concept_bank,
			eval_loader=clean_test_loader,
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
	quality_csv = Path.cwd() / "results" / "mnist_mlp" / "global_explanations" / "mlp_concept_bank_seed_quality.csv"
	quality_csv.parent.mkdir(parents=True, exist_ok=True)
	seed_quality_df.to_csv(quality_csv, index=False)
	logging.info(
		f"Selected seed {best_seed} as final concept bank based on Mean TCAR={best_tcar:.4f}. "
		f"Saved seed quality table: {quality_csv}"
	)

	# correlation_records = []

	# for axis_name, poison_x, poison_y in runs:
	# 	train_loader, _, _ = MNISTDataLoaders(args, args.task)
	# 	poisoned_train_loader = poison_loader_fn(train_loader, poison_x=poison_x, poison_y=poison_y)

	# 	model = MLP2454().to(device)
	# 	model.load_state_dict(init_state_dict)
	# 	_train_classifier(model, poisoned_train_loader, args.poison_epochs, args.lr, device)

	# 	run_tag = f"axis_{axis_name}_x{poison_x:.1f}_y{poison_y:.1f}_mlp2454"
	# 	save_dir = Path.cwd() / "results" / "poison" / "global_explanations_mlp" / run_tag

	# 	tcar_corr, tcav_corr, _, _ = global_explanations(
	# 		model,
	# 		random_seed=args.seeds[0],
	# 		batch_size=args.batch_size,
	# 		plot=args.plot,
	# 		eval_digits=list(args.eval_digits),
	# 		save_dir=save_dir,
	# 		concept_bank=concept_bank,
	# 	)

	# 	correlation_records.append(
	# 		{
	# 			"pair": pair_label,
	# 			"axis": axis_name,
	# 			"x_alpha": float(poison_x),
	# 			"y_alpha": float(poison_y),
	# 			"tcar_corr": float(tcar_corr),
	# 			"tcav_corr": float(tcav_corr),
	# 		}
	# 	)

	# 	logging.info(
	# 		f"Run done | axis={axis_name} x={poison_x:.1f} y={poison_y:.1f} "
	# 		f"TCAR={tcar_corr:.4f} TCAV={tcav_corr:.4f}"
	# 	)

	# corr_save_root = Path.cwd() / "results" / "poison" / "global_explanations_mlp"
	# corr_save_root.mkdir(parents=True, exist_ok=True)
	# corr_df = pd.DataFrame(correlation_records)
	# corr_csv = corr_save_root / "correlation_runs.csv"
	# corr_df.to_csv(corr_csv, index=False)
	# logging.info(f"Saved per-run correlation table: {corr_csv}")
	# plot_poison_correlation_curves(correlation_records, corr_save_root)
