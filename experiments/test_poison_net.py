import re
import sys
import importlib.util as ilu
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent.parent
POISON_ROOT = PROJECT_ROOT / "poison"

if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))
if str(POISON_ROOT) not in sys.path:
	sys.path.insert(0, str(POISON_ROOT))


def _import_poison_module(module_name: str):
	module_path = POISON_ROOT / f"{module_name}.py"
	spec = ilu.spec_from_file_location(f"_poison_{module_name}", module_path)
	module = ilu.module_from_spec(spec)
	spec.loader.exec_module(module)
	return module


_poison_args = _import_poison_module("Arguments")
_poison_datasets = _import_poison_module("datasets")
_poison_fm = _import_poison_module("FusionModel")

Arguments = _poison_args.Arguments
MNISTDataLoaders = _poison_datasets.MNISTDataLoaders
QNet = _poison_fm.QNet
single_enta_to_design = _poison_fm.single_enta_to_design


TASK = {
	"task": "MNIST_4",
	"option": "mix_reg",
	"n_qubits": 6,
	"n_layers": 4,
	"fold": 1,
	"backend": "tq",
}


def build_design(task: dict):
	arch_code = [task["n_qubits"], task["n_layers"]]
	args = Arguments(**task)
	n_layers = arch_code[1]
	n_qubits = int(arch_code[0] / args.fold)
	single = [[i] + [1] * 2 * n_layers for i in range(1, n_qubits + 1)]
	enta = [[i] + [i + 1] * n_layers for i in range(1, n_qubits)] + [
		[n_qubits] + [1] * n_layers
	]
	return single_enta_to_design(single, enta, arch_code, args.fold)


def parse_weight_name(weight_path: Path):
	# Matches: tmp_(1, 9)_poison_(0.0, 0.3)
	pattern = r"^tmp_\((.*?)\)_poison_\(([-0-9.]+),\s*([-0-9.]+)\)$"
	match = re.match(pattern, weight_path.name)
	if match is None:
		return None
	nums_str, poison_x_str, poison_y_str = match.groups()
	nums = tuple(int(part.strip()) for part in nums_str.split(",") if part.strip())
	return nums, float(poison_x_str), float(poison_y_str)


def evaluate_accuracy(model: QNet, data_loader, args: Arguments) -> float:
	model.eval()
	target_all = []
	output_all = []

	with torch.no_grad():
		for feed_dict in data_loader:
			images = feed_dict["image"].to(args.device)
			targets = feed_dict["digit"].to(args.device)
			output = model(images, args.n_qubits, args.task)

			target_all.append(targets)
			output_all.append(output)

	target_all = torch.cat(target_all, dim=0)
	output_all = torch.cat(output_all, dim=0)
	preds = torch.argmax(output_all, dim=1)
	return float((preds == target_all).float().mean().item())


def get_dataloaders_for_nums(nums: tuple[int, ...]):
	args = Arguments(**TASK)
	args.digits_of_interest = list(nums)
	train_loader, _, test_loader = MNISTDataLoaders(args, TASK["task"])
	return args, train_loader, test_loader


def main() -> None:
	weight_dir = PROJECT_ROOT / "poison" / "weights"
	output_csv = PROJECT_ROOT / "results" / "poison" / "all_weight_performance.csv"
	output_csv.parent.mkdir(parents=True, exist_ok=True)

	weight_files = sorted(path for path in weight_dir.iterdir() if path.is_file())

	records = []
	design = build_design(TASK)
	dataloader_cache = {}

	parsed_weights = []
	for weight_path in weight_files:
		parsed = parse_weight_name(weight_path)
		if parsed is not None:
			parsed_weights.append((weight_path, *parsed))

	for weight_path, nums, poison_x, poison_y in tqdm(parsed_weights, desc="Evaluating weights"):
		if nums not in dataloader_cache:
			dataloader_cache[nums] = get_dataloaders_for_nums(nums)

		args, train_loader, test_loader = dataloader_cache[nums]
		model = QNet(args, design).to(args.device)
		state_dict = torch.load(weight_path, map_location=args.device)
		model.load_state_dict(state_dict, strict=False)

		train_acc = evaluate_accuracy(model, train_loader, args)
		test_acc = evaluate_accuracy(model, test_loader, args)

		records.append(
			{
				"weight_file": weight_path.name,
				"nums": str(nums),
				"x_alpha": poison_x,
				"y_alpha": poison_y,
				"train_acc": train_acc,
				"test_acc": test_acc,
			}
		)

	results_df = pd.DataFrame(records)
	results_df = results_df.sort_values(["nums", "x_alpha", "y_alpha"]).reset_index(drop=True)
	results_df.to_csv(output_csv, index=False)

	print(f"Saved {len(results_df)} rows to {output_csv}")
	if len(results_df) > 0:
		print(results_df.to_string(index=False))


if __name__ == "__main__":
	main()
