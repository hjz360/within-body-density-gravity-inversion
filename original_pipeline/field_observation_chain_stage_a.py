"""Stage-A utilities for the frozen San Nicolas observation chain.

This module does not train or run an inversion network.  It freezes and
validates the raw-station forward, Kriging, and uncertainty components needed
before the identifiability experiment is authorized.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np

from gravity_validation import numerical_voxel_kernel, rectangular_prism_gz


def _finite_matrix(value: np.ndarray, name: str, columns: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != columns or array.shape[0] < 1:
        raise ValueError(f"{name} must have shape (n, {columns})")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(array)


def ordinary_kriging_matrix(
    station_xy: np.ndarray,
    prediction_xy: np.ndarray,
    theta: np.ndarray,
) -> np.ndarray:
    """Return the frozen DACE ``regpoly0/corrgauss`` prediction matrix.

    DACE normalizes the station coordinates before fitting.  This function
    reproduces that geometry normalization and the tiny diagonal stabilization
    used by ``dacefit.m``.  The returned matrix maps unscaled station values to
    ordinary-Kriging predictions and therefore reproduces a constant field.
    """

    stations = _finite_matrix(station_xy, "station_xy", 2)
    predictions = _finite_matrix(prediction_xy, "prediction_xy", 2)
    theta_array = np.asarray(theta, dtype=np.float64).reshape(-1)
    if theta_array.shape != (2,) or not np.isfinite(theta_array).all():
        raise ValueError("theta must contain exactly two finite values")
    if np.any(theta_array <= 0.0):
        raise ValueError("theta values must be strictly positive")
    if np.unique(stations, axis=0).shape[0] != stations.shape[0]:
        raise ValueError("station_xy sites must be unique")

    station_mean = stations.mean(axis=0)
    station_scale = stations.std(axis=0, ddof=1)
    station_scale = np.where(station_scale == 0.0, 1.0, station_scale)
    scaled_stations = (stations - station_mean) / station_scale
    scaled_predictions = (predictions - station_mean) / station_scale
    return _ordinary_kriging_matrix_scaled(
        scaled_stations,
        scaled_predictions,
        theta_array,
    )


def _ordinary_kriging_matrix_scaled(
    scaled_stations: np.ndarray,
    scaled_predictions: np.ndarray,
    theta_array: np.ndarray,
) -> np.ndarray:
    """Ordinary-Kriging weights for coordinates already in frozen DACE scale."""

    station_delta = (
        scaled_stations[:, None, :] - scaled_stations[None, :, :]
    )
    correlation = np.exp(
        -np.sum(theta_array[None, None, :] * station_delta**2, axis=2)
    )
    station_count = scaled_stations.shape[0]
    stabilization = (10.0 + station_count) * np.finfo(np.float64).eps
    correlation.flat[:: station_count + 1] += stabilization

    prediction_delta = (
        scaled_predictions[:, None, :] - scaled_stations[None, :, :]
    )
    cross_correlation = np.exp(
        -np.sum(theta_array[None, None, :] * prediction_delta**2, axis=2)
    )

    trend = np.ones((station_count, 1), dtype=np.float64)
    solved_trend = np.linalg.solve(correlation, trend)
    solved_cross = np.linalg.solve(correlation, cross_correlation.T)
    denominator = float((trend.T @ solved_trend).item())
    correction = (
        1.0 - trend.T @ solved_cross
    ) / denominator
    weights_station_by_prediction = solved_cross + solved_trend @ correction
    return np.ascontiguousarray(weights_station_by_prediction.T)


def validate_frozen_kriging(
    matrix: np.ndarray,
    station_values: np.ndarray,
    reference_predictions: np.ndarray,
    *,
    prediction_atol: float = 1e-10,
    row_sum_atol: float = 1e-10,
) -> dict[str, Any]:
    """Validate a frozen Kriging map against an independent predictor output."""

    kriging = np.asarray(matrix, dtype=np.float64)
    values = np.asarray(station_values, dtype=np.float64).reshape(-1)
    reference = np.asarray(reference_predictions, dtype=np.float64).reshape(-1)
    if kriging.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    if kriging.shape != (reference.size, values.size):
        raise ValueError(
            "matrix shape must be (reference_predictions, station_values)"
        )
    if not (
        np.isfinite(kriging).all()
        and np.isfinite(values).all()
        and np.isfinite(reference).all()
    ):
        raise ValueError("Kriging validation inputs must be finite")

    predicted = kriging @ values
    maximum_prediction_error = float(np.max(np.abs(predicted - reference)))
    maximum_row_sum_error = float(np.max(np.abs(kriging.sum(axis=1) - 1.0)))
    prediction_match = maximum_prediction_error <= prediction_atol
    row_sum_match = maximum_row_sum_error <= row_sum_atol
    return {
        "passed": bool(prediction_match and row_sum_match),
        "prediction_match": bool(prediction_match),
        "constant_reproduction": bool(row_sum_match),
        "shape": [int(kriging.shape[0]), int(kriging.shape[1])],
        "max_abs_prediction_error": maximum_prediction_error,
        "max_abs_row_sum_error": maximum_row_sum_error,
        "prediction_atol": float(prediction_atol),
        "row_sum_atol": float(row_sum_atol),
    }


def load_frozen_kriging_artifact(path: str | Path) -> dict[str, Any]:
    """Load the MATLAB-v7.3 frozen artifact in logical Python orientation."""

    required = {
        "K",
        "station_gravity",
        "prediction_dace",
        "prediction_xy",
        "station_xy",
    }
    with h5py.File(Path(path), "r") as handle:
        missing = sorted(required.difference(handle.keys()))
        if missing:
            raise ValueError(f"Frozen Kriging artifact is missing: {missing}")
        matrix = np.asarray(handle["K"][:], dtype=np.float64).T
        station_gravity = np.asarray(
            handle["station_gravity"][:], dtype=np.float64
        ).reshape(-1)
        prediction_dace = np.asarray(
            handle["prediction_dace"][:], dtype=np.float64
        ).reshape(-1)
        prediction_xy = np.asarray(
            handle["prediction_xy"][:], dtype=np.float64
        ).T
        station_xy = np.asarray(
            handle["station_xy"][:], dtype=np.float64
        ).T

    if station_xy.shape != (station_gravity.size, 2):
        raise ValueError("station_xy shape does not match station_gravity")
    if prediction_xy.shape != (prediction_dace.size, 2):
        raise ValueError("prediction_xy shape does not match prediction_dace")
    validation = validate_frozen_kriging(
        matrix,
        station_gravity,
        prediction_dace,
    )
    return {
        "K": matrix,
        "station_gravity": station_gravity,
        "prediction_dace": prediction_dace,
        "prediction_xy": prediction_xy,
        "station_xy": station_xy,
        "validation": validation,
    }


def full_field_grid() -> dict[str, Any]:
    """Return the frozen 50 m San Nicolas field-grid geometry."""

    x_centers = np.arange(-2700.0, -600.0 + 25.0, 50.0)
    y_centers = np.arange(-1150.0, 600.0 + 25.0, 50.0)
    z_centers = np.arange(25.0, 775.0 + 25.0, 50.0)
    x_edges = np.arange(-2725.0, -575.0 + 25.0, 50.0)
    y_edges = np.arange(-1175.0, 625.0 + 25.0, 50.0)
    z_edges = np.arange(0.0, 800.0 + 25.0, 50.0)
    shape = (z_centers.size, y_centers.size, x_centers.size)
    if shape != (16, 36, 43):
        raise RuntimeError(f"Unexpected full field grid shape: {shape}")
    return {
        "shape_zyx": shape,
        "x_centers": x_centers,
        "y_centers": y_centers,
        "z_centers": z_centers,
        "x_edges": x_edges,
        "y_edges": y_edges,
        "z_edges": z_edges,
        "cell_size_m": 50.0,
        "flattening": "C-order z_y_x; x fastest",
        "z_positive": "down",
    }


def full_field_cell_bounds(
    grid: dict[str, Any] | None = None,
) -> np.ndarray:
    """Return full-grid prism bounds in C-order ``z, y, x`` flattening."""

    geometry = full_field_grid() if grid is None else grid
    x_edges = np.asarray(geometry["x_edges"], dtype=np.float64)
    y_edges = np.asarray(geometry["y_edges"], dtype=np.float64)
    z_edges = np.asarray(geometry["z_edges"], dtype=np.float64)
    if (x_edges.size, y_edges.size, z_edges.size) != (44, 37, 17):
        raise ValueError("Full field edge arrays must have sizes 44, 37, 17")

    z1, y1, x1 = np.meshgrid(
        z_edges[:-1],
        y_edges[:-1],
        x_edges[:-1],
        indexing="ij",
    )
    z2, y2, x2 = np.meshgrid(
        z_edges[1:],
        y_edges[1:],
        x_edges[1:],
        indexing="ij",
    )
    return np.ascontiguousarray(
        np.column_stack(
            [
                x1.reshape(-1),
                x2.reshape(-1),
                y1.reshape(-1),
                y2.reshape(-1),
                z1.reshape(-1),
                z2.reshape(-1),
            ]
        )
    )


def field_observation_xyz(
    pointdata: np.ndarray,
    branch: str,
    *,
    field3_authoritative: bool = False,
) -> np.ndarray:
    """Construct one explicitly labelled raw-station observation branch."""

    points = _finite_matrix(pointdata, "pointdata", 4)
    if branch == "actual_xy_flat_legacy":
        z_observed = np.full(points.shape[0], -0.1, dtype=np.float64)
    elif branch == "actual_xyz_candidate_elevation":
        z_observed = -points[:, 2]
    elif branch == "actual_xyz_authoritative":
        if not field3_authoritative:
            raise ValueError(
                "field 3 lacks an authoritative file-specific descriptor"
            )
        z_observed = -points[:, 2]
    else:
        raise ValueError(f"Unknown observation branch: {branch}")
    return np.ascontiguousarray(
        np.column_stack([points[:, 0], points[:, 1], z_observed])
    )


def build_prism_sensitivity(
    observation_xyz: np.ndarray,
    cell_bounds: np.ndarray,
    *,
    chunk_size: int = 256,
) -> np.ndarray:
    """Build an observation-by-cell exact-prism kernel in mGal/(kg/m^3)."""

    observations = _finite_matrix(observation_xyz, "observation_xyz", 3)
    bounds = _finite_matrix(cell_bounds, "cell_bounds", 6)
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError("cell_bounds x intervals must be increasing")
    if np.any(bounds[:, 2] >= bounds[:, 3]):
        raise ValueError("cell_bounds y intervals must be increasing")
    if np.any(bounds[:, 4] >= bounds[:, 5]):
        raise ValueError("cell_bounds z intervals must be increasing")

    matrix = np.empty(
        (observations.shape[0], bounds.shape[0]),
        dtype=np.float64,
    )
    for chunk_start in range(0, bounds.shape[0], chunk_size):
        chunk_stop = min(chunk_start + chunk_size, bounds.shape[0])
        for cell_id in range(chunk_start, chunk_stop):
            matrix[:, cell_id] = rectangular_prism_gz(
                observations[:, 0],
                observations[:, 1],
                observations[:, 2],
                bounds[cell_id],
                density_kg_m3=1.0,
            )
    if not np.isfinite(matrix).all():
        raise RuntimeError("Generated sensitivity matrix contains nonfinite values")
    return matrix


def generate_field_forward_operators(
    pointdata: np.ndarray,
    *,
    cell_bounds: np.ndarray | None = None,
    chunk_size: int = 256,
) -> dict[str, dict[str, Any]]:
    """Generate legacy-flat and provisional candidate-elevation operators."""

    points = _finite_matrix(pointdata, "pointdata", 4)
    bounds = (
        full_field_cell_bounds()
        if cell_bounds is None
        else _finite_matrix(cell_bounds, "cell_bounds", 6)
    )
    branches = {
        "actual_xy_flat_legacy": {
            "semantic_status": "legacy_flat_reproduction",
            "field3_authoritative": False,
        },
        "actual_xyz_candidate_elevation": {
            "semantic_status": "provisional_candidate_elevation",
            "field3_authoritative": False,
        },
    }
    result: dict[str, dict[str, Any]] = {}
    for branch, metadata in branches.items():
        observations = field_observation_xyz(points, branch)
        result[branch] = {
            **metadata,
            "observation_xyz": observations,
            "kernel": build_prism_sensitivity(
                observations,
                bounds,
                chunk_size=chunk_size,
            ),
        }
    return result


def validate_operator_elements(
    matrix: np.ndarray,
    observation_xyz: np.ndarray,
    cell_bounds: np.ndarray,
    *,
    pairs: list[tuple[int, int]],
    quadrature_order: int = 24,
    relative_tolerance: float = 1e-8,
) -> dict[str, Any]:
    """Check frozen operator elements with independent numerical quadrature."""

    observations = _finite_matrix(observation_xyz, "observation_xyz", 3)
    bounds = _finite_matrix(cell_bounds, "cell_bounds", 6)
    kernel = np.asarray(matrix, dtype=np.float64)
    if kernel.shape != (observations.shape[0], bounds.shape[0]):
        raise ValueError("matrix shape must match observations and cell bounds")
    if not np.isfinite(kernel).all():
        raise ValueError("matrix must contain only finite values")
    if quadrature_order < 2:
        raise ValueError("quadrature_order must be at least two")

    checks: list[dict[str, Any]] = []
    for observation_id, cell_id in pairs:
        if not (0 <= observation_id < observations.shape[0]):
            raise ValueError("observation index is out of range")
        if not (0 <= cell_id < bounds.shape[0]):
            raise ValueError("cell index is out of range")
        exact_value = float(kernel[observation_id, cell_id])
        numerical_value = numerical_voxel_kernel(
            observations[observation_id],
            bounds[cell_id],
            order=quadrature_order,
        )
        absolute_error = abs(exact_value - numerical_value)
        scale = max(abs(numerical_value), np.finfo(np.float64).tiny)
        relative_error = absolute_error / scale
        checks.append(
            {
                "observation_id": int(observation_id),
                "cell_id": int(cell_id),
                "exact_mgal_per_kg_m3": exact_value,
                "quadrature_mgal_per_kg_m3": numerical_value,
                "absolute_error": float(absolute_error),
                "relative_error": float(relative_error),
            }
        )
    maximum_relative_error = max(
        (check["relative_error"] for check in checks),
        default=0.0,
    )
    return {
        "passed": bool(maximum_relative_error <= relative_tolerance),
        "checked_element_count": len(checks),
        "quadrature_order": int(quadrature_order),
        "relative_tolerance": float(relative_tolerance),
        "max_relative_error": float(maximum_relative_error),
        "checks": checks,
    }


def write_field_operator_artifact(
    path: str | Path,
    operators: dict[str, dict[str, Any]],
    *,
    cell_bounds: np.ndarray,
) -> None:
    """Persist explicit observation-by-cell field operators in HDF5."""

    target = Path(path)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite field operator: {target}")
    expected = {
        "actual_xy_flat_legacy",
        "actual_xyz_candidate_elevation",
    }
    if set(operators) != expected:
        raise ValueError(f"operators must contain exactly {sorted(expected)}")
    bounds = _finite_matrix(cell_bounds, "cell_bounds", 6)
    target.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(target, "w") as handle:
        handle.attrs["logical_orientation"] = "observation_by_cell"
        handle.attrs["kernel_units"] = "mGal_per_kg_m3"
        handle.attrs["coordinate_units"] = "m"
        handle.attrs["z_positive"] = "down"
        handle.attrs["cell_flattening"] = "C-order z_y_x; x fastest"
        handle.create_dataset("cell_bounds", data=bounds)
        for branch, content in operators.items():
            observations = _finite_matrix(
                content["observation_xyz"],
                f"{branch}.observation_xyz",
                3,
            )
            kernel = np.asarray(content["kernel"], dtype=np.float64)
            if kernel.shape != (observations.shape[0], bounds.shape[0]):
                raise ValueError(f"{branch} kernel has inconsistent shape")
            if not np.isfinite(kernel).all():
                raise ValueError(f"{branch} kernel contains nonfinite values")
            group = handle.create_group(branch)
            group.attrs["semantic_status"] = content["semantic_status"]
            group.attrs["field3_authoritative"] = bool(
                content["field3_authoritative"]
            )
            group.create_dataset("observation_xyz", data=observations)
            group.create_dataset(
                "kernel",
                data=kernel,
                compression="gzip",
                shuffle=True,
                chunks=(
                    min(64, kernel.shape[0]),
                    min(1024, kernel.shape[1]),
                ),
            )


def deterministic_spatial_block_folds(
    station_xy: np.ndarray,
    *,
    n_folds: int = 5,
    axis: int = 1,
) -> list[np.ndarray]:
    """Return deterministic contiguous coordinate blocks for spatial CV."""

    stations = _finite_matrix(station_xy, "station_xy", 2)
    if axis not in (0, 1):
        raise ValueError("axis must be 0 or 1")
    if not (2 <= n_folds < stations.shape[0]):
        raise ValueError("n_folds must be between 2 and station_count - 1")
    order = np.lexsort(
        (
            np.arange(stations.shape[0]),
            stations[:, 1 - axis],
            stations[:, axis],
        )
    )
    return [
        np.ascontiguousarray(fold, dtype=np.int64)
        for fold in np.array_split(order, n_folds)
    ]


def frozen_theta_cv_residuals(
    station_xy: np.ndarray,
    station_values: np.ndarray,
    theta: np.ndarray,
    *,
    mode: str,
    n_folds: int = 5,
    block_axis: int = 1,
) -> dict[str, np.ndarray | str | int]:
    """Cross-validate with theta and full-data coordinate scaling frozen."""

    stations = _finite_matrix(station_xy, "station_xy", 2)
    values = np.asarray(station_values, dtype=np.float64).reshape(-1)
    theta_array = np.asarray(theta, dtype=np.float64).reshape(-1)
    if values.shape != (stations.shape[0],):
        raise ValueError("station_values must match station_xy rows")
    if not np.isfinite(values).all():
        raise ValueError("station_values must contain only finite values")
    if theta_array.shape != (2,) or np.any(theta_array <= 0.0):
        raise ValueError("theta must contain two strictly positive values")
    if stations.shape[0] < 3:
        raise ValueError("At least three stations are required for CV")

    if mode == "leave_one_out":
        folds = [np.array([index], dtype=np.int64) for index in range(values.size)]
    elif mode == "spatial_block":
        folds = deterministic_spatial_block_folds(
            stations,
            n_folds=n_folds,
            axis=block_axis,
        )
    else:
        raise ValueError("mode must be leave_one_out or spatial_block")

    scaling_mean = stations.mean(axis=0)
    scaling_scale = stations.std(axis=0, ddof=1)
    scaling_scale = np.where(scaling_scale == 0.0, 1.0, scaling_scale)
    scaled = (stations - scaling_mean) / scaling_scale

    prediction = np.full(values.size, np.nan, dtype=np.float64)
    fold_id = np.full(values.size, -1, dtype=np.int64)
    all_indices = np.arange(values.size)
    for current_fold_id, held_indices in enumerate(folds):
        train_mask = np.ones(values.size, dtype=bool)
        train_mask[held_indices] = False
        train_indices = all_indices[train_mask]
        if train_indices.size < 2:
            raise ValueError("Every CV fold must retain at least two stations")
        matrix = _ordinary_kriging_matrix_scaled(
            scaled[train_indices],
            scaled[held_indices],
            theta_array,
        )
        prediction[held_indices] = matrix @ values[train_indices]
        fold_id[held_indices] = current_fold_id

    if not np.isfinite(prediction).all() or np.any(fold_id < 0):
        raise RuntimeError("Cross-validation did not predict every station once")
    return {
        "mode": mode,
        "prediction": prediction,
        "residual": values - prediction,
        "fold_id": fold_id,
        "n_folds": len(folds),
    }


def _residual_summary(residuals: np.ndarray) -> dict[str, Any]:
    values = np.asarray(residuals, dtype=np.float64).reshape(-1)
    if values.size < 2 or not np.isfinite(values).all():
        raise ValueError("Residuals must contain at least two finite values")
    centered = values - np.median(values)
    robust_sigma = float(1.4826 * np.median(np.abs(centered)))
    rmse_value = float(np.sqrt(np.mean(values**2)))
    if robust_sigma <= np.finfo(np.float64).eps:
        robust_sigma = rmse_value
    return {
        "count": int(values.size),
        "mean_mgal": float(values.mean()),
        "rmse_mgal": rmse_value,
        "robust_sigma_mgal": robust_sigma,
        "quantiles_mgal": {
            label: float(value)
            for label, value in zip(
                ("q05", "q25", "q50", "q75", "q95"),
                np.quantile(values, [0.05, 0.25, 0.50, 0.75, 0.95]),
            )
        },
    }


def build_error_envelope(
    *,
    measurement_floor_mgal: float,
    loo_residuals: np.ndarray,
    spatial_block_residuals: np.ndarray,
) -> dict[str, Any]:
    """Freeze transparent measurement-plus-preprocessing error scales."""

    floor = float(measurement_floor_mgal)
    if not np.isfinite(floor) or floor <= 0.0:
        raise ValueError("measurement_floor_mgal must be finite and positive")
    loo_summary = _residual_summary(loo_residuals)
    block_summary = _residual_summary(spatial_block_residuals)
    nominal = max(floor, loo_summary["robust_sigma_mgal"])
    high = max(nominal, block_summary["robust_sigma_mgal"])
    return {
        "package_label": "measurement_and_preprocessing_error_envelope",
        "known_field_noise_truth": False,
        "published_measurement_floor": {
            "sigma_mgal": floor,
            "scope": "original_San_Nicolas_gravity_survey",
            "current_subset_error_column": False,
        },
        "empirical_residuals": {
            "frozen_theta_leave_one_out": loo_summary,
            "y_contiguous_spatial_block": block_summary,
        },
        "white_noise_sigma_mgal": {
            "low": floor,
            "nominal": float(nominal),
            "high": float(high),
        },
        "scale_rule": {
            "low": "published source-level measurement floor",
            "nominal": "max(low, robust sigma of frozen-theta LOO residuals)",
            "high": (
                "max(nominal, robust sigma of y-contiguous spatial-block "
                "CV residuals)"
            ),
        },
    }


def residual_spatial_diagnostics(
    station_xy: np.ndarray,
    residuals: np.ndarray,
    *,
    n_lag_bins: int = 8,
) -> dict[str, Any]:
    """Summarize irregular-station residual trend and semivariogram structure."""

    stations = _finite_matrix(station_xy, "station_xy", 2)
    values = np.asarray(residuals, dtype=np.float64).reshape(-1)
    if values.shape != (stations.shape[0],) or not np.isfinite(values).all():
        raise ValueError("residuals must be finite and match station rows")
    if not (2 <= n_lag_bins < stations.shape[0]):
        raise ValueError("n_lag_bins must be between 2 and station_count - 1")

    centered_xy = stations - stations.mean(axis=0)
    xy_scale = centered_xy.std(axis=0, ddof=1)
    xy_scale = np.where(xy_scale == 0.0, 1.0, xy_scale)
    design = np.column_stack(
        [np.ones(stations.shape[0]), centered_xy / xy_scale]
    )
    coefficients, _, _, _ = np.linalg.lstsq(design, values, rcond=None)
    planar = design @ coefficients

    left, right = np.triu_indices(stations.shape[0], k=1)
    distances = np.linalg.norm(stations[left] - stations[right], axis=1)
    semivariance = 0.5 * (values[left] - values[right]) ** 2
    maximum_distance = float(distances.max())
    edges = np.linspace(0.0, maximum_distance, n_lag_bins + 1)
    bins: list[dict[str, Any]] = []
    for bin_id in range(n_lag_bins):
        if bin_id == n_lag_bins - 1:
            selected = (distances >= edges[bin_id]) & (
                distances <= edges[bin_id + 1]
            )
        else:
            selected = (distances >= edges[bin_id]) & (
                distances < edges[bin_id + 1]
            )
        bins.append(
            {
                "bin_id": bin_id,
                "distance_min_m": float(edges[bin_id]),
                "distance_max_m": float(edges[bin_id + 1]),
                "pair_count": int(selected.sum()),
                "mean_distance_m": (
                    float(distances[selected].mean()) if np.any(selected) else None
                ),
                "semivariance_mgal2": (
                    float(semivariance[selected].mean())
                    if np.any(selected)
                    else None
                ),
            }
        )

    residual_variance = float(np.var(values, ddof=1))
    target_semivariance = (1.0 - np.exp(-1.0)) * residual_variance
    correlation_length = maximum_distance
    for item in bins:
        if (
            item["semivariance_mgal2"] is not None
            and item["semivariance_mgal2"] >= target_semivariance
        ):
            correlation_length = float(item["mean_distance_m"])
            break
    return {
        "station_count": int(stations.shape[0]),
        "pair_count": int(distances.size),
        "planar_coefficients_standardized_xy": coefficients.tolist(),
        "planar_component_rms_mgal": float(
            np.sqrt(np.mean((planar - planar.mean()) ** 2))
        ),
        "residual_variance_mgal2": residual_variance,
        "correlation_length_m_heuristic": correlation_length,
        "correlation_length_rule": (
            "first semivariogram bin reaching (1-exp(-1))*sample variance"
        ),
        "semivariogram": bins,
    }


def gridded_residual_spectrum(
    kriging_matrix: np.ndarray,
    station_residuals: np.ndarray,
    *,
    grid_shape_yx: tuple[int, int] = (36, 43),
    spacing_m: float = 50.0,
) -> dict[str, Any]:
    """Compute a descriptive 2-D spectrum after applying the frozen K map."""

    matrix = np.asarray(kriging_matrix, dtype=np.float64)
    residuals = np.asarray(station_residuals, dtype=np.float64).reshape(-1)
    if matrix.ndim != 2 or matrix.shape[1] != residuals.size:
        raise ValueError("kriging_matrix columns must match station_residuals")
    if not np.isfinite(matrix).all() or not np.isfinite(residuals).all():
        raise ValueError("Spectrum inputs must contain only finite values")
    ny, nx = grid_shape_yx
    if matrix.shape[0] != ny * nx:
        raise ValueError("kriging_matrix rows must equal the requested grid size")
    if spacing_m <= 0.0:
        raise ValueError("spacing_m must be positive")

    prediction_order = matrix @ residuals
    grid = prediction_order.reshape(nx, ny).T
    grid = grid - grid.mean()
    power = np.abs(np.fft.fft2(grid)) ** 2
    fy = np.fft.fftfreq(ny, d=spacing_m)
    fx = np.fft.fftfreq(nx, d=spacing_m)
    radial = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    nonzero = radial > 0.0
    maximum_radial = float(radial.max())
    normalized = radial / maximum_radial
    masks = {
        "low": nonzero & (normalized <= 1.0 / 3.0),
        "mid": (normalized > 1.0 / 3.0) & (normalized <= 2.0 / 3.0),
        "high": normalized > 2.0 / 3.0,
    }
    total = float(power[nonzero].sum())
    if total <= np.finfo(np.float64).tiny:
        fractions = {"low": 0.0, "mid": 0.0, "high": 0.0}
    else:
        fractions = {
            name: float(power[mask].sum() / total)
            for name, mask in masks.items()
        }
        normalization_error = 1.0 - sum(fractions.values())
        fractions["high"] += normalization_error
    dominant_power = np.where(nonzero, power, -np.inf)
    dominant_index = np.unravel_index(
        int(np.argmax(dominant_power)),
        dominant_power.shape,
    )
    dominant_wavenumber = float(radial[dominant_index])
    return {
        "grid_shape_yx": [int(ny), int(nx)],
        "spacing_m": float(spacing_m),
        "power_fraction": fractions,
        "dominant_wavenumber_cycles_per_m": dominant_wavenumber,
        "dominant_wavelength_m": (
            float(1.0 / dominant_wavenumber)
            if dominant_wavenumber > 0.0
            else None
        ),
        "interpretation": (
            "descriptive spectrum of frozen-K residual predictions; not "
            "independent gridded observations"
        ),
    }


def resolve_stage_a_readiness(
    *,
    field3_authoritative: bool,
    kriging_validation_passed: bool,
    field_operator_validation_passed: bool,
    error_envelope_frozen: bool,
) -> dict[str, Any]:
    """Resolve original audit blockers without erasing semantic uncertainty."""

    component_ready = bool(
        kriging_validation_passed
        and field_operator_validation_passed
        and error_envelope_frozen
    )
    checks = [
        (
            "POINTDATA_FIELD_3_SEMANTICS_UNCONFIRMED",
            bool(field3_authoritative),
        ),
        (
            "KRIGING_FITTED_MODEL_NOT_PERSISTED",
            bool(kriging_validation_passed),
        ),
        ("G_FIELD_UNAVAILABLE", bool(field_operator_validation_passed)),
        ("NOISE_METADATA_UNAVAILABLE", bool(error_envelope_frozen)),
    ]
    open_blockers = [code for code, passed in checks if not passed]
    resolved_blockers = [code for code, passed in checks if passed]
    return {
        "fully_field_realistic_stage_a_ready": bool(
            component_ready and field3_authoritative
        ),
        "open_blocker_codes": open_blockers,
        "resolved_blocker_codes": resolved_blockers,
        "branches": {
            "actual_xy_flat_legacy": {
                "authorized": component_ready,
                "authorization_class": "reproducible_legacy_chain",
                "field_geometry_claim": "flat_observation_height_legacy",
            },
            "actual_xyz_candidate_elevation": {
                "authorized": component_ready,
                "authorization_class": (
                    "confirmed_field_geometry"
                    if field3_authoritative
                    else "conditional_sensitivity_analysis"
                ),
                "field_geometry_claim": (
                    "authoritative_actual_xyz"
                    if field3_authoritative
                    else "provisional_candidate_elevation"
                ),
            },
        },
        "training_authorized": False,
        "next_authorized_action": (
            "stage_a_identifiability_on_labeled_branches"
            if component_ready
            else "resolve_remaining_chain_components_without_training"
        ),
    }
