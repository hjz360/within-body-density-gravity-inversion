"""Reproducible split and dynamic-noise datasets for the 20k experiment."""

import numpy as np
import torch


def make_fixed_split(total_count, train_count, val_count, seed):
    """Return reproducible, disjoint training and validation indices."""
    if train_count + val_count != total_count:
        raise ValueError("train_count + val_count must equal total_count")
    indices = np.random.default_rng(seed).permutation(total_count)
    return indices[:train_count], indices[train_count:]


class AugmentedArrayDataset(torch.utils.data.Dataset):
    """Expose indexed base arrays through deterministic per-epoch noise views."""

    def __init__(self, images, masks, indices, noise_levels=(0.0,), seed=0):
        self.images = images
        self.masks = masks
        self.indices = np.asarray(indices, dtype=np.int64)
        self.noise_levels = tuple(float(level) for level in noise_levels)
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self):
        return len(self.indices) * len(self.noise_levels)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __getitem__(self, index):
        base_position = index % len(self.indices)
        view_index = index // len(self.indices)
        source_index = self.indices[base_position]
        image = self.images[source_index]
        mask = self.masks[source_index]
        noise_level = self.noise_levels[view_index]

        if noise_level == 0.0:
            return image, mask

        seed_sequence = np.random.SeedSequence(
            [self.seed, self.epoch, int(source_index), view_index]
        )
        rng = np.random.default_rng(seed_sequence)
        noise = rng.normal(0.0, 1.0, image.shape).astype(np.float32)
        noisy_image = image + noise_level * float(image.max()) * noise
        return noisy_image.astype(np.float32, copy=False), mask
