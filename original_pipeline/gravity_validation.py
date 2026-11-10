"""Independent gravity-forward and validation utilities.

The training pipeline uses a precomputed voxel sensitivity matrix ``G.mat``.
This module provides a separate rectangular-prism implementation and numerical
quadrature checks so that validation is not limited to reusing that matrix.

Coordinate convention
---------------------
* x and y are horizontal coordinates in metres.
* z is positive downward; the observation plane is normally z=-0.1 m.
* density contrast is supplied in kg/m^3 to the forward routines.
* returned vertical gravity is in mGal.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import h5py
import numpy as np
from numpy.polynomial.legendre import leggauss


GRAVITATIONAL_CONSTANT = 6.67e-11
SI_TO_MGAL = 1.0e5


@dataclass(frozen=True)
class Prism:
    """Axis-aligned rectangular prism with density contrast in g/cm^3."""

    x1: float
    x2: float
    y1: float
    y2: float
    z1: float
    z2: float
    density_g_cm3: float

    @property
    def bounds(self) -> tuple[float, float, float, float, float, float]:
        return self.x1, self.x2, self.y1, self.y2, self.z1, self.z2


def load_hdf5_kernel(path: str | Path) -> np.ndarray:
    """Load MATLAB-v7.3 ``G.mat`` and return shape (1024, 16384)."""

    with h5py.File(path, "r") as handle:
        kernel = np.asarray(handle["G"][:], dtype=np.float64).T
    if kernel.shape != (1024, 16384):
        raise ValueError(f"Unexpected gravity kernel shape: {kernel.shape}")
    if not np.isfinite(kernel).all():
        raise ValueError("Gravity kernel contains non-finite values")
    return kernel


def observation_grid(
    nx: int = 32,
    ny: int = 32,
    cell_size: float = 50.0,
    z_observed: float = -0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return flattened observation coordinates using MATLAB x-fastest order."""

    x = (np.arange(nx, dtype=np.float64) + 0.5) * cell_size
    y = (np.arange(ny, dtype=np.float64) + 0.5) * cell_size
    yy, xx = np.meshgrid(y, x, indexing="ij")
    zz = np.full_like(xx, z_observed)
    return xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)


def rectangular_prism_gz(
    x_observed: np.ndarray | float,
    y_observed: np.ndarray | float,
    z_observed: np.ndarray | float,
    bounds: Sequence[float],
    density_kg_m3: float = 1.0,
) -> np.ndarray:
    """Exact vertical attraction of one rectangular prism in mGal.

    This is an independent vectorized implementation of the standard closed
    form used for rectangular-prism gravity.  It accepts arbitrary (off-grid)
    prism boundaries, which is important for avoiding a purely grid-aligned
    inverse-crime test.
    """

    x_obs, y_obs, z_obs = np.broadcast_arrays(
        np.asarray(x_observed, dtype=np.float64),
        np.asarray(y_observed, dtype=np.float64),
        np.asarray(z_observed, dtype=np.float64),
    )
    x1, x2, y1, y2, z1, z2 = map(float, bounds)
    if not (x1 < x2 and y1 < y2 and z1 < z2):
        raise ValueError(f"Invalid prism bounds: {bounds}")

    total = np.zeros_like(x_obs, dtype=np.float64)
    tiny = np.finfo(np.float64).tiny
    for ix, x_bound in enumerate((x1, x2), start=1):
        a = x_obs - x_bound
        for iy, y_bound in enumerate((y1, y2), start=1):
            b = y_obs - y_bound
            for iz, z_bound in enumerate((z1, z2), start=1):
                c = z_obs - z_bound
                radius = np.sqrt(a * a + b * b + c * c)
                log_b = np.log(np.maximum(b + radius, tiny))
                log_a = np.log(np.maximum(a + radius, tiny))
                angle = np.arctan((a * b) / (c * radius))
                primitive = a * log_b + b * log_a - c * angle
                sign = -1.0 if (ix + iy + iz) % 2 else 1.0
                total += sign * primitive

    return (
        -GRAVITATIONAL_CONSTANT
        * float(density_kg_m3)
        * total
        * SI_TO_MGAL
    )


def forward_prisms(
    prisms: Iterable[Prism],
    nx: int = 32,
    ny: int = 32,
    cell_size: float = 50.0,
    z_observed: float = -0.1,
) -> np.ndarray:
    """Forward model a collection of arbitrary rectangular prisms."""

    x_obs, y_obs, z_obs = observation_grid(nx, ny, cell_size, z_observed)
    gravity = np.zeros(nx * ny, dtype=np.float64)
    for prism in prisms:
        gravity += rectangular_prism_gz(
            x_obs,
            y_obs,
            z_obs,
            prism.bounds,
            density_kg_m3=1000.0 * prism.density_g_cm3,
        )
    return gravity.reshape(ny, nx)


def numerical_voxel_kernel(
    observation_xyz: Sequence[float],
    bounds: Sequence[float],
    order: int = 24,
) -> float:
    """Numerically integrate a unit-density voxel using Gauss-Legendre nodes."""

    x_obs, y_obs, z_obs = map(float, observation_xyz)
    x1, x2, y1, y2, z1, z2 = map(float, bounds)
    nodes, weights = leggauss(order)

    def transform(left: float, right: float) -> tuple[np.ndarray, np.ndarray]:
        points = 0.5 * (right - left) * nodes + 0.5 * (right + left)
        scaled_weights = 0.5 * (right - left) * weights
        return points, scaled_weights

    xs, wx = transform(x1, x2)
    ys, wy = transform(y1, y2)
    zs, wz = transform(z1, z2)
    dx = xs[:, None, None] - x_obs
    dy = ys[None, :, None] - y_obs
    dz = zs[None, None, :] - z_obs
    radius_squared = dx * dx + dy * dy + dz * dz
    integrand = dz / np.power(radius_squared, 1.5)
    quadrature_weights = (
        wx[:, None, None] * wy[None, :, None] * wz[None, None, :]
    )
    integral = np.sum(quadrature_weights * integrand)
    return float(GRAVITATIONAL_CONSTANT * SI_TO_MGAL * integral)


def voxelize_prisms(
    prisms: Iterable[Prism],
    nx: int = 32,
    ny: int = 32,
    nz: int = 16,
    cell_size: float = 50.0,
) -> np.ndarray:
    """Return exact cell-volume averages of arbitrary prisms in g/cm^3."""

    volume = np.zeros((nz, ny, nx), dtype=np.float64)
    x_edges = np.arange(nx + 1, dtype=np.float64) * cell_size
    y_edges = np.arange(ny + 1, dtype=np.float64) * cell_size
    z_edges = np.arange(nz + 1, dtype=np.float64) * cell_size
    cell_volume = cell_size**3

    for prism in prisms:
        overlap_x = np.maximum(
            0.0,
            np.minimum(x_edges[1:], prism.x2) - np.maximum(x_edges[:-1], prism.x1),
        )
        overlap_y = np.maximum(
            0.0,
            np.minimum(y_edges[1:], prism.y2) - np.maximum(y_edges[:-1], prism.y1),
        )
        overlap_z = np.maximum(
            0.0,
            np.minimum(z_edges[1:], prism.z2) - np.maximum(z_edges[:-1], prism.z1),
        )
        fraction = (
            overlap_z[:, None, None]
            * overlap_y[None, :, None]
            * overlap_x[None, None, :]
            / cell_volume
        )
        volume += prism.density_g_cm3 * fraction
    return volume


def forward_voxel_density(kernel: np.ndarray, density_g_cm3: np.ndarray) -> np.ndarray:
    """Apply the stored voxel kernel to a (z, y, x) density model."""

    density = np.asarray(density_g_cm3, dtype=np.float64)
    if density.shape != (16, 32, 32):
        raise ValueError(f"Expected density shape (16, 32, 32), got {density.shape}")
    gravity = np.asarray(kernel, dtype=np.float64) @ (1000.0 * density.reshape(-1))
    return gravity.reshape(32, 32)


def relative_l2(predicted: np.ndarray, reference: np.ndarray) -> float:
    reference_array = np.asarray(reference, dtype=np.float64)
    denominator = np.linalg.norm(reference_array.reshape(-1))
    if denominator <= np.finfo(np.float64).eps:
        return 0.0
    difference = np.asarray(predicted, dtype=np.float64) - reference_array
    return float(np.linalg.norm(difference.reshape(-1)) / denominator)


def rmse(predicted: np.ndarray, reference: np.ndarray) -> float:
    difference = np.asarray(predicted, dtype=np.float64) - np.asarray(
        reference, dtype=np.float64
    )
    return float(np.sqrt(np.mean(difference * difference)))


def density_center_of_mass(
    density_g_cm3: np.ndarray,
    cell_size: float = 50.0,
) -> tuple[float, float, float]:
    """Return density-weighted (x, y, depth) centre in metres."""

    density = np.clip(np.asarray(density_g_cm3, dtype=np.float64), 0.0, None)
    total = float(density.sum())
    if total <= np.finfo(np.float64).eps:
        return float("nan"), float("nan"), float("nan")
    nz, ny, nx = density.shape
    x = (np.arange(nx) + 0.5) * cell_size
    y = (np.arange(ny) + 0.5) * cell_size
    z = (np.arange(nz) + 0.5) * cell_size
    x_com = float(np.sum(density * x[None, None, :]) / total)
    y_com = float(np.sum(density * y[None, :, None]) / total)
    z_com = float(np.sum(density * z[:, None, None]) / total)
    return x_com, y_com, z_com


def model_metrics(
    prediction: np.ndarray,
    truth: np.ndarray,
    predicted_gravity: np.ndarray,
    reference_gravity: np.ndarray,
    active_threshold: float = 0.1,
) -> dict[str, float | list[float]]:
    """Joint model-domain and data-domain metrics for one reconstruction."""

    prediction = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    mask = truth > 0.05
    full_rmse = rmse(prediction, truth)
    nz_rmse = rmse(prediction[mask], truth[mask]) if np.any(mask) else 0.0
    truth_com = np.asarray(density_center_of_mass(truth))
    prediction_com = np.asarray(density_center_of_mass(prediction))
    com_error = float(np.linalg.norm(prediction_com - truth_com))
    return {
        "full_rmse_g_cm3": full_rmse,
        "nonzero_rmse_g_cm3": nz_rmse,
        "forward_relative_l2": relative_l2(predicted_gravity, reference_gravity),
        "forward_rmse_mgal": rmse(predicted_gravity, reference_gravity),
        "peak_g_cm3": float(np.nanmax(prediction)),
        "active_fraction_ge_0p1": float(np.mean(prediction >= active_threshold)),
        "center_of_mass_xyz_m": prediction_com.tolist(),
        "center_of_mass_error_m": com_error,
    }
