"""Pure helpers for field-data window assembly and physical diagnostics."""

from pathlib import Path

import numpy as np


def field_window_starts(full_y=36, full_x=43, window=32):
    return [
        (start_y, start_x)
        for start_x in range(full_x - window + 1)
        for start_y in range(full_y - window + 1)
    ]


def assemble_windows(window_models, starts, full_shape=(16, 36, 43)):
    windows = np.asarray(window_models, dtype=np.float64)
    sums = np.zeros(full_shape, dtype=np.float64)
    square_sums = np.zeros(full_shape, dtype=np.float64)
    counts = np.zeros(full_shape, dtype=np.int32)
    for window_model, (start_y, start_x) in zip(windows, starts):
        _, height, width = window_model.shape
        region = np.s_[:, start_y:start_y + height, start_x:start_x + width]
        sums[region] += window_model
        square_sums[region] += window_model ** 2
        counts[region] += 1
    if np.any(counts == 0):
        raise ValueError("Field assembly contains uncovered voxels")
    mean = sums / counts
    variance = np.maximum(square_sums / counts - mean ** 2, 0.0)
    return mean, counts, np.sqrt(variance)


def extract_windows(model, starts, window=32):
    volume = np.asarray(model)
    return np.stack(
        [
            volume[:, start_y:start_y + window, start_x:start_x + window]
            for start_y, start_x in starts
        ]
    )


def load_pcg(path, shape=(16, 36, 43), clip_primary=True):
    rows = np.loadtxt(Path(path))
    expected = int(np.prod(shape))
    if rows.shape != (expected, 4):
        raise ValueError(f"Expected {expected} PCG rows with four columns, got {rows.shape}")
    raw = rows[:, 3].reshape(shape)
    primary = np.clip(raw, 0.0, 1.1) if clip_primary else raw.copy()
    return raw, primary


def to_physical_density(normalized_density, scale):
    return np.asarray(normalized_density) * float(scale)


def forward_window(kernel, density_kg_m3):
    kernel = np.asarray(kernel)
    density = np.asarray(density_kg_m3)
    gravity = kernel @ density.reshape(-1)
    side = int(round(np.sqrt(gravity.size)))
    if side * side != gravity.size:
        raise ValueError("Forward output is not a square gravity grid")
    return gravity.reshape(side, side)
