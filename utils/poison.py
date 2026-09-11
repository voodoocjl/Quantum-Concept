import torch
from utils.dataset import MyDataset
import numpy as np


def _balanced_two_class_indices(labels, class_a, class_b):
    """Return balanced indices for two classes using the same sample count."""
    idx_a = np.where(labels == class_a)[0]
    idx_b = np.where(labels == class_b)[0]
    n_samples = min(len(idx_a), len(idx_b))
    return idx_a[:n_samples], idx_b[:n_samples], n_samples


def flip_labels(ys, alpha):
    # Create copy of original data
    new_ys = np.copy(ys)

    # Calculate number of samples to flip
    n_samples = len(ys)
    n_flip = int(n_samples * alpha)

    # Ensure n_flip is even for balanced allocation
    if n_flip % 2 == 1:
        n_flip -= 1

    # Split dataset into first and second halves
    first_half = np.arange(n_samples // 2)
    second_half = np.arange(n_samples // 2, n_samples)

    # Randomly select half of flip samples from each half
    flip_indices_first = np.random.choice(first_half, n_flip // 2, replace=False)
    flip_indices_second = np.random.choice(second_half, n_flip // 2, replace=False)

    # Combine flip indices from both halves
    flip_indices = np.concatenate([flip_indices_first, flip_indices_second])

    # Flip selected labels (0->1, 1->0)
    new_ys[flip_indices] = 1 - new_ys[flip_indices]

    return new_ys, flip_indices


def flip_labels_between_classes(ys, alpha, class_a, class_b):
    """Flip labels between class_a and class_b while keeping other classes unchanged."""
    new_ys = np.copy(ys)
    idx_a, idx_b, n_samples = _balanced_two_class_indices(ys, class_a, class_b)

    n_total = 2 * n_samples
    n_flip = int(n_total * alpha)

    if n_flip % 2 == 1:
        n_flip -= 1

    n_flip_per_class = n_flip // 2

    if n_flip_per_class > 0:
        flip_indices_a = np.random.choice(idx_a, n_flip_per_class, replace=False)
        flip_indices_b = np.random.choice(idx_b, n_flip_per_class, replace=False)

        new_ys[flip_indices_a] = class_b
        new_ys[flip_indices_b] = class_a
        flip_indices = np.concatenate([flip_indices_a, flip_indices_b])
    else:
        flip_indices = np.array([], dtype=np.int64)

    return new_ys, flip_indices


def data_poison(xs, alpha, xes=1, ordered=True):
    """Poison the dataset by selecting half of the samples from each half of the dataset."""
    # Create copy of original data
    poisoned_xs = np.copy(xs)

    # Calculate number of samples to poison
    n_samples = len(xs)
    n_poison = int(n_samples * alpha)

    # Ensure n_poison is even for balanced allocation
    if n_poison % 2 != 0:
        n_poison += 1

    # Split dataset into first and second halves
    mid_point = n_samples // 2
    first_half_indices = np.arange(mid_point)
    second_half_indices = np.arange(mid_point, n_samples)

    # Select half of poison samples from each half
    n_poison_per_half = n_poison // 2
    first_half_poison = np.random.choice(
        first_half_indices, n_poison_per_half, replace=False
    )
    second_half_poison = np.random.choice(
        second_half_indices, n_poison_per_half, replace=False
    )

    # Combine selected indices
    poison_indices = np.concatenate([first_half_poison, second_half_poison])

    # Replace selected samples
    if ordered:
        for idx in poison_indices:
            poisoned_xs[idx] = xes[idx]
    else:
        # Replace with random complex values
        # thetas = np.random.normal(0, 1e-1, size=[len(poison_indices), 2, len(xs[0])])
        # Replace with random values at full sample shape (pixel-level for images).
        sample_shape = xs.shape[1:]
        thetas = np.random.normal(
            0,
            1e-1,
            size=(len(poison_indices), 2, *sample_shape),
        )

        for i, idx in enumerate(poison_indices):
            if xs.dtype == "complex128":
                poisoned_xs[idx] = thetas[i, 0] + 1.0j * thetas[i, 1]
            else:
                poisoned_xs[idx] = thetas[i, 0]
            poisoned_xs[idx] = poisoned_xs[idx] / np.linalg.norm(poisoned_xs[idx])

    return poisoned_xs, poison_indices


def poison(dataloader, poison_x=0, poison_y=0):
    """
    Args:
        dataloader: The input dataloader to be poisoned.
        poison_x: alpha for data_poison (feature poisoning ratio)
        poison_y: alpha for flip_labels (label flipping ratio)
    Returns:
        poisonloader: A new DataLoader with poisoned data.
    """
    data_list = []
    labels_list = []

    for feed_dict in dataloader:
        data_list.append(feed_dict['image'])
        labels_list.append(feed_dict['digit'])

    data = torch.cat(data_list).numpy()
    labels = torch.cat(labels_list).numpy()

    unique_labels = np.sort(np.unique(labels))
    if len(unique_labels) < 2:
        raise ValueError('At least two classes are required for poisoning.')

    # Poison only the first two class ids; keep all classes in the dataset.
    class_a, class_b = unique_labels[0], unique_labels[1]
    idx_a, idx_b, n_samples = _balanced_two_class_indices(labels, class_a, class_b)
    pair_indices = np.concatenate([idx_a, idx_b])

    # Apply poisoning
    if poison_y > 0.001:
        labels, _ = flip_labels_between_classes(labels, poison_y, class_a, class_b)

    if poison_x > 0.001:
        # Poison features only on the selected two classes, in a balanced way.
        pair_data = np.copy(data[pair_indices])
        pair_data, _ = data_poison(pair_data, poison_x, ordered=False)
        data[pair_indices] = pair_data

    # Reconstruct dataset
    poisoned_dataset = MyDataset(torch.from_numpy(data), torch.from_numpy(labels))

    # Use the same batch size as the original dataloader
    batch_size = dataloader.batch_size if dataloader.batch_size is not None else 1

    poisonloader = torch.utils.data.DataLoader(
        poisoned_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True)

    return poisonloader