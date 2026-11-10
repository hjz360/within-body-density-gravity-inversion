"""Read-only audit utilities for the Stage-A field observation chain."""

from __future__ import annotations

import hashlib
from pathlib import Path
import re
from typing import Iterable

import h5py
import numpy as np


FIELD_NAMES = ("x", "y", "candidate_elevation", "gravity")
FIELD_X = np.arange(-2700.0, -600.0 + 1.0, 50.0, dtype=np.float64)
FIELD_Y = np.arange(-1150.0, 600.0 + 1.0, 50.0, dtype=np.float64)
WINDOW_SIZE = 32


def sha256_file(path: Path) -> str:
    """Return the SHA-256 of one file without modifying it."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_pointdata(path: Path) -> np.ndarray:
    """Load an HDF5 MATLAB station table and normalize it to (station, field)."""

    with h5py.File(Path(path), "r") as handle:
        if "pointdata" not in handle:
            raise KeyError("MATLAB source does not contain pointdata")
        stored = np.asarray(handle["pointdata"])

    if stored.ndim != 2:
        raise ValueError("pointdata must be a two-dimensional array")
    if stored.shape == (4, 4):
        raise ValueError("pointdata orientation is ambiguous for a 4x4 array")
    if stored.shape[0] == 4 and stored.shape[1] != 4:
        logical = stored.T
    elif stored.shape[1] == 4 and stored.shape[0] != 4:
        logical = stored
    else:
        raise ValueError("pointdata must contain exactly four fields")
    if logical.shape[0] < 2:
        raise ValueError("pointdata must contain at least two stations")

    result = np.ascontiguousarray(logical, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError("pointdata values must all be finite")
    return result


def _field_summary(name: str, values: np.ndarray) -> dict:
    return {
        "name": name,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "unique_count": int(np.unique(values).size),
    }


def station_field_summary(pointdata: np.ndarray) -> dict:
    """Summarize station fields without asserting unresolved field semantics."""

    table = np.asarray(pointdata, dtype=np.float64)
    if table.ndim != 2 or table.shape[1] != 4 or table.shape[0] < 2:
        raise ValueError("pointdata must have shape (station,4)")
    if not np.isfinite(table).all():
        raise ValueError("pointdata values must all be finite")
    unique_xy = np.unique(table[:, :2], axis=0)
    return {
        "station_count": int(table.shape[0]),
        "field_count": 4,
        "field_names": list(FIELD_NAMES),
        "candidate_elevation_confirmed": False,
        "fields": [
            _field_summary(name, table[:, index])
            for index, name in enumerate(FIELD_NAMES)
        ],
        "coordinate_duplicate_count": int(table.shape[0] - unique_xy.shape[0]),
    }


def window_registry() -> list[dict]:
    """Return the exact x-major/y-minor registry used by the MATLAB slicer."""

    rows = []
    window_id = 0
    for x_start in range(FIELD_X.size - WINDOW_SIZE + 1):
        for y_start in range(FIELD_Y.size - WINDOW_SIZE + 1):
            rows.append(
                {
                    "window_id": window_id,
                    "x_start_index": x_start,
                    "y_start_index": y_start,
                    "x_min": float(FIELD_X[x_start]),
                    "x_max": float(FIELD_X[x_start + WINDOW_SIZE - 1]),
                    "y_min": float(FIELD_Y[y_start]),
                    "y_max": float(FIELD_Y[y_start + WINDOW_SIZE - 1]),
                }
            )
            window_id += 1
    return rows


def station_window_coverage(
    pointdata: np.ndarray, registry: Iterable[dict]
) -> list[dict]:
    """Add descriptive in-window counts while retaining the global fit scope."""

    table = np.asarray(pointdata, dtype=np.float64)
    if table.ndim != 2 or table.shape[1] != 4:
        raise ValueError("pointdata must have shape (station,4)")
    if not np.isfinite(table).all():
        raise ValueError("pointdata values must all be finite")

    rows = []
    x = table[:, 0]
    y = table[:, 1]
    for source in registry:
        row = dict(source)
        inside = (
            (x >= float(row["x_min"]))
            & (x <= float(row["x_max"]))
            & (y >= float(row["y_min"]))
            & (y <= float(row["y_max"]))
        )
        row["local_station_count"] = int(np.count_nonzero(inside))
        row["global_station_count"] = int(table.shape[0])
        row["local_count_is_information_count"] = False
        rows.append(row)
    return rows


def load_matlab_vector(path: Path, key: str) -> np.ndarray:
    """Load a finite MATLAB HDF5 vector stored as either (1,n) or (n,1)."""

    with h5py.File(Path(path), "r") as handle:
        if key not in handle:
            raise KeyError(f"MATLAB source does not contain {key}")
        stored = np.asarray(handle[key])
    if stored.ndim != 2 or 1 not in stored.shape:
        raise ValueError(f"{key} must be stored as a two-dimensional vector")
    vector = np.ascontiguousarray(stored.reshape(-1), dtype=np.float64)
    if vector.size == 0 or not np.isfinite(vector).all():
        raise ValueError(f"{key} vector must be nonempty and finite")
    return vector


def load_full_field_grid(path: Path) -> np.ndarray:
    """Load the current 43-by-36 MATLAB vector as a logical (y,x) grid."""

    vector = load_matlab_vector(path, "d")
    expected_size = int(FIELD_X.size * FIELD_Y.size)
    if vector.size != expected_size:
        raise ValueError(
            f"full field vector length {vector.size} != {expected_size}"
        )
    return np.ascontiguousarray(
        vector.reshape(FIELD_X.size, FIELD_Y.size).T, dtype=np.float64
    )


def expected_windows(full_grid: np.ndarray) -> np.ndarray:
    """Extract all logical windows in the exact MATLAB x-major/y-minor order."""

    full = np.asarray(full_grid, dtype=np.float64)
    expected_shape = (FIELD_Y.size, FIELD_X.size)
    if full.shape != expected_shape:
        raise ValueError(f"full grid shape {full.shape} != {expected_shape}")
    if not np.isfinite(full).all():
        raise ValueError("full grid values must all be finite")
    windows = np.empty(
        (len(window_registry()), WINDOW_SIZE, WINDOW_SIZE), dtype=np.float64
    )
    for row in window_registry():
        windows[row["window_id"]] = full[
            row["y_start_index"] : row["y_start_index"] + WINDOW_SIZE,
            row["x_start_index"] : row["x_start_index"] + WINDOW_SIZE,
        ]
    return windows


def load_saved_windows(directory: Path) -> np.ndarray:
    """Load exactly data_slice0.mat through data_slice59.mat."""

    source = Path(directory)
    rows = window_registry()
    result = np.empty((len(rows), WINDOW_SIZE, WINDOW_SIZE), dtype=np.float64)
    for row in rows:
        window_id = row["window_id"]
        path = source / f"data_slice{window_id}.mat"
        if not path.is_file():
            raise FileNotFoundError(f"missing saved field window {window_id}: {path}")
        vector = load_matlab_vector(path, "d_slice")
        if vector.size != WINDOW_SIZE * WINDOW_SIZE:
            raise ValueError(
                f"saved field window {window_id} length {vector.size} "
                f"!= {WINDOW_SIZE * WINDOW_SIZE}"
            )
        result[window_id] = vector.reshape(WINDOW_SIZE, WINDOW_SIZE)
    return result


def audit_saved_windows(
    full_grid: np.ndarray, saved_windows: np.ndarray
) -> dict:
    """Compare persisted windows with exact extraction from the persisted grid."""

    expected = expected_windows(full_grid)
    saved = np.asarray(saved_windows, dtype=np.float64)
    if saved.shape != expected.shape:
        raise ValueError(f"saved windows shape {saved.shape} != {expected.shape}")
    if not np.isfinite(saved).all():
        raise ValueError("saved window values must all be finite")
    differences = np.abs(saved - expected)
    mismatch = np.any(differences != 0.0, axis=(1, 2))
    return {
        "passed": bool(not np.any(mismatch)),
        "window_count": int(saved.shape[0]),
        "mismatch_window_ids": [
            int(index) for index in np.flatnonzero(mismatch)
        ],
        "max_abs_difference": float(np.max(differences)),
    }


def _numbers_after_assignment(text: str, name: str) -> list[float] | None:
    match = re.search(
        rf"\b{re.escape(name)}\s*=\s*\[([^\]]+)\]", text, flags=re.IGNORECASE
    )
    if match is None:
        return None
    tokens = re.findall(
        r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?",
        match.group(1),
    )
    return [float(token) for token in tokens]


def _read_source_text(path: Path) -> tuple[str, str]:
    raw = Path(path).read_bytes()
    for encoding in ("utf-8", "gb18030"):
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(
        "utf-8/gb18030",
        raw,
        0,
        min(1, len(raw)),
        "source is neither strict UTF-8 nor strict GB18030",
    )


def audit_kriging_source(path: Path) -> dict:
    """Conservatively audit the declared DACE/Kriging preprocessing behavior."""

    source = Path(path)
    text, source_encoding = _read_source_text(source)
    compact = re.sub(r"\s+", " ", text)
    used_columns = sorted(
        {
            int(value)
            for value in re.findall(
                r"pointdata\s*\(\s*:\s*,\s*(\d+)\s*\)",
                text,
                flags=re.IGNORECASE,
            )
        }
    )
    all_columns = {1, 2, 3, 4}
    data_dependent = bool(re.search(r"\bdacefit\s*\(", text, re.IGNORECASE))
    global_fit = (
        data_dependent
        and bool(re.search(r"\bS\s*=\s*\[\s*x\s*,\s*y\s*\]", compact))
        and bool(re.search(r"\bY\s*=\s*pointdata", compact, re.IGNORECASE))
    )
    trend_match = re.search(r"@(regpoly\d+)", text, re.IGNORECASE)
    correlation_match = re.search(r"@(corr\w+)", text, re.IGNORECASE)
    point_comment = re.search(
        r"(\d+)\s+measurement\s+points", text, re.IGNORECASE
    )
    persisted_candidates = [
        candidate
        for pattern in ("*dmodel*.mat", "*kriging*model*.mat", "*dace*model*.mat")
        for candidate in source.parent.glob(pattern)
        if candidate.is_file()
    ]

    xrange_declared = bool(
        re.search(r"xrange\s*=\s*-2700\s*:\s*50\s*:\s*-600", text)
    )
    yrange_declared = bool(
        re.search(r"yrange\s*=\s*-1150\s*:\s*50\s*:\s*600", text)
    )
    window_match = re.search(r"\blen\s*=\s*(\d+)", text)
    return {
        "path": str(source.resolve()),
        "sha256": sha256_file(source),
        "source_encoding": source_encoding,
        "data_dependent_fit": data_dependent,
        "fit_scope": "global_all_stations" if global_fit else "unresolved",
        "used_pointdata_columns": used_columns,
        "unused_pointdata_columns": sorted(all_columns - set(used_columns)),
        "candidate_elevation_column_used": 3 in used_columns,
        "trend_function": trend_match.group(1).lower() if trend_match else None,
        "correlation_function": (
            correlation_match.group(1).lower() if correlation_match else None
        ),
        "theta_initial": _numbers_after_assignment(text, "theta"),
        "theta_lower_bound": _numbers_after_assignment(text, "lob"),
        "theta_upper_bound": _numbers_after_assignment(text, "upb"),
        "grid_spacing_m": 50.0 if xrange_declared and yrange_declared else None,
        "grid_shape_yx": (
            [int(FIELD_Y.size), int(FIELD_X.size)]
            if xrange_declared and yrange_declared
            else None
        ),
        "window_size": int(window_match.group(1)) if window_match else None,
        "declared_measurement_count_in_comment": (
            int(point_comment.group(1)) if point_comment else None
        ),
        "fitted_model_persisted": bool(persisted_candidates),
        "persisted_model_candidates": [
            str(candidate.resolve()) for candidate in persisted_candidates
        ],
        "fixed_K_available": bool(persisted_candidates and not data_dependent),
        "prediction_variance_computed": bool(
            re.search(r"\[\s*d\s+MSE\s*\]\s*=\s*predictor", text)
        ),
    }


def _decode_attribute(value: object) -> str | None:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, str):
        return value
    return None


def _kernel_metadata(path: Path) -> dict:
    with h5py.File(path, "r") as handle:
        if "G" not in handle:
            raise KeyError(f"{path} does not contain dataset G")
        dataset = handle["G"]
        logical_shape = tuple(reversed(dataset.shape))
        geometry = _decode_attribute(dataset.attrs.get("observation_geometry"))
        dtype = str(dataset.dtype)
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "logical_shape": [int(value) for value in logical_shape],
        "dtype": dtype,
        "observation_geometry": geometry,
    }


def audit_forward_sources(root: Path, station_count: int) -> dict:
    """Classify the regular-grid kernel and require metadata for a field kernel."""

    project = Path(root)
    grid_path = project / "G.mat"
    expected_grid = (1024, 16 * 32 * 32)
    expected_field = (int(station_count), 16 * 36 * 43)

    grid = {
        "available": grid_path.is_file(),
        "expected_logical_shape": list(expected_grid),
        "classification": "regular_grid_upper_bound",
    }
    if grid_path.is_file():
        grid.update(_kernel_metadata(grid_path))
        grid["shape_matches"] = tuple(grid["logical_shape"]) == expected_grid

    candidates = [
        project / "Field_data" / "G_field.mat",
        project / "G_field.mat",
    ]
    existing = [path for path in candidates if path.is_file()]
    field = {
        "available": False,
        "expected_logical_shape": list(expected_field),
        "searched_paths": [str(path.resolve()) for path in candidates],
    }
    if existing:
        metadata = _kernel_metadata(existing[0])
        shape_matches = tuple(metadata["logical_shape"]) == expected_field
        geometry_matches = metadata["observation_geometry"] == "actual_xyz"
        field.update(metadata)
        field["shape_matches"] = shape_matches
        field["geometry_metadata_matches"] = geometry_matches
        field["available"] = bool(shape_matches and geometry_matches)

    return {
        "G_grid": grid,
        "G_field": field,
        "field_operator_ready": bool(field["available"]),
    }
