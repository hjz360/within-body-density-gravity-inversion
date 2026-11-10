"""Frozen experiment contract for the paired-density C0F0/C1F0 pilot."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, Mapping

import numpy as np
import torch


MODEL_TO_LAW = {"C0F0": "C0", "C1F0": "C1"}
MODES = ("smoke", "formal")
SEED = 20260721
BOOTSTRAP_SEED = 20260724
DATASET_MANIFEST_SHA256 = (
    "aa0c3c9c2906d655c77b943d4f1fc9e87fd2d98e8d5772b2aa7101b6a1e304a7"
)
COMMON_INITIALIZATION_SHA256 = (
    "b31a80ee5f27e7ef69b52bbe7f667f32ded425f21de2152151759502c314b6be"
)
MIN_FORMAL_FREE_BYTES = 3 * 1024**3
FORMAL_TRAIN_IDS = np.arange(0, 4500, dtype=np.int64)
FORMAL_VALIDATION_IDS = np.arange(4500, 5000, dtype=np.int64)
SMOKE_TRAIN_IDS = np.arange(0, 512, dtype=np.int64)
SMOKE_VALIDATION_IDS = np.arange(4500, 4628, dtype=np.int64)
SNAPSHOT_EPOCHS = (60, 90, 120)
BOUNDARY_CATEGORIES = (
    "interior",
    "lateral_only",
    "vertical_only",
    "mixed_lateral_vertical",
)
EXPECTED_VALIDATION_BOUNDARY_COUNTS = {
    "interior": 31,
    "lateral_only": 44,
    "vertical_only": 121,
    "mixed_lateral_vertical": 304,
}
FIXED_VISUAL_IDS = {
    "interior": (4504, 4752),
    "lateral_only": (4525, 4761),
    "vertical_only": (4503, 4773),
    "mixed_lateral_vertical": (4500, 4744),
}
FACE_KEYS = ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")


def _ids_sha256(values) -> str:
    return hashlib.sha256(np.asarray(values, dtype=np.int64).tobytes()).hexdigest()


TRAIN_IDS_SHA256 = _ids_sha256(FORMAL_TRAIN_IDS)
VALIDATION_IDS_SHA256 = _ids_sha256(FORMAL_VALIDATION_IDS)


@dataclass(frozen=True)
class PilotPaths:
    run_dir: Path
    primary_weights: Path
    best_weights: Path
    last_checkpoint: Path
    history: Path
    metadata: Path
    failure: Path
    snapshot_weights: Mapping[int, Path]


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(int(chunk_size))
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def pilot_config(model_id: str, mode: str) -> dict:
    if model_id not in MODEL_TO_LAW:
        raise ValueError(f"unknown model_id: {model_id}")
    if mode not in MODES:
        raise ValueError(f"unknown mode: {mode}")
    smoke = mode == "smoke"
    train_ids = SMOKE_TRAIN_IDS if smoke else FORMAL_TRAIN_IDS
    validation_ids = SMOKE_VALIDATION_IDS if smoke else FORMAL_VALIDATION_IDS
    max_epochs = 5 if smoke else 120
    batch_size = 32
    noise_levels = [0.0, 0.02, 0.04]
    training_views = int(len(train_ids) * len(noise_levels))
    return {
        "model_id": str(model_id),
        "density_law": MODEL_TO_LAW[model_id],
        "seed": SEED,
        "mode": str(mode),
        "train_ids": train_ids.tolist(),
        "validation_ids": validation_ids.tolist(),
        "train_ids_sha256": _ids_sha256(train_ids),
        "validation_ids_sha256": _ids_sha256(validation_ids),
        "max_epochs": max_epochs,
        "batch_size": batch_size,
        "learning_rate": 3e-4,
        "smooth_l1_beta": 0.1,
        "lambda_tv_xy": 1e-4,
        "forward_loss_weight": 0.0,
        "noise_levels": noise_levels,
        "validation_noise_levels": [0.0],
        "scheduler": None,
        "early_stopping": False,
        "freeze_any_layer": False,
        "training_views": training_views,
        "steps_per_epoch": int(math.ceil(training_views / batch_size)),
    }


def experiment_paths(root: Path, model_id: str, mode: str) -> PilotPaths:
    config = pilot_config(model_id, mode)
    root = Path(root).resolve()
    family = (
        "paired_density_c0c1_smoke"
        if mode == "smoke"
        else "paired_density_c0c1_pilot"
    )
    run_dir = root / "experiments" / family / model_id / f"seed_{SEED}"
    primary_epoch = int(config["max_epochs"])
    primary = run_dir / f"epoch_{primary_epoch:03d}_weights.pth"
    snapshot_epochs = (primary_epoch,) if mode == "smoke" else SNAPSHOT_EPOCHS
    snapshots: Dict[int, Path] = {
        int(epoch): (
            primary
            if int(epoch) == primary_epoch
            else run_dir / f"epoch_{int(epoch):03d}_weights.pth"
        )
        for epoch in snapshot_epochs
    }
    return PilotPaths(
        run_dir=run_dir,
        primary_weights=primary,
        best_weights=run_dir / "best_validation_weights.pth",
        last_checkpoint=run_dir / "last_checkpoint.pth",
        history=run_dir / "history.npz",
        metadata=run_dir / "metadata.json",
        failure=run_dir / "failure.json",
        snapshot_weights=snapshots,
    )


def boundary_category(face_contact: Mapping[str, object]) -> str:
    if set(face_contact) != set(FACE_KEYS):
        raise ValueError(f"face_contact keys must equal {FACE_KEYS}")
    lateral = any(bool(face_contact[key]) for key in FACE_KEYS[:4])
    vertical = bool(face_contact["z_min"]) or bool(face_contact["z_max"])
    if not lateral and not vertical:
        return "interior"
    if lateral and not vertical:
        return "lateral_only"
    if vertical and not lateral:
        return "vertical_only"
    return "mixed_lateral_vertical"


def validation_boundary_index(metadata_rows) -> dict:
    counts = {name: 0 for name in BOUNDARY_CATEGORIES}
    by_id = {}
    for row in metadata_rows:
        if row.get("split") != "validation":
            continue
        accepted_id = int(row["accepted_pair_id"])
        if accepted_id in by_id:
            raise RuntimeError(f"duplicate validation accepted_pair_id: {accepted_id}")
        category = boundary_category(row["support"]["face_contact"])
        by_id[accepted_id] = category
        counts[category] += 1
    return {"by_id": by_id, "counts": counts}


def _load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_jsonl(path: Path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _require_manifest_array_contract(manifest: dict) -> None:
    expected = {
        "density": ((5000, 16, 32, 32), "float32"),
        "gravity": ((5000, 1, 32, 32), "float32"),
    }
    for law in ("C0", "C1"):
        if law not in manifest.get("arrays", {}):
            raise RuntimeError(f"dataset manifest is missing {law}")
        for kind, (shape, dtype) in expected.items():
            entry = manifest["arrays"][law].get(kind, {})
            if tuple(entry.get("shape", ())) != shape or entry.get("dtype") != dtype:
                raise RuntimeError(f"dataset manifest {law}/{kind} contract mismatch")


def _metadata_identity_key(row: dict) -> dict:
    excluded = {"density_law", "density", "gravity_l2", "gravity_peak_abs"}
    return {key: value for key, value in row.items() if key not in excluded}


def verify_dataset_contract(root: Path) -> dict:
    root = Path(root).resolve()
    dataset_root = root / "data" / "paired_density_5k_v1"
    manifest_path = dataset_root / "manifest.json"
    actual_manifest_sha = sha256_file(manifest_path)
    if actual_manifest_sha != DATASET_MANIFEST_SHA256:
        raise RuntimeError(
            "dataset manifest SHA-256 mismatch: "
            f"{actual_manifest_sha} != {DATASET_MANIFEST_SHA256}"
        )
    manifest = _load_json(manifest_path)
    if manifest.get("status") != "complete":
        raise RuntimeError("dataset manifest status is not complete")
    counts = (
        int(manifest.get("accepted_pair_count", -1)),
        int(manifest.get("train_pair_count", -1)),
        int(manifest.get("validation_pair_count", -1)),
    )
    if counts != (5000, 4500, 500):
        raise RuntimeError(f"dataset count contract mismatch: {counts}")
    _require_manifest_array_contract(manifest)

    audit_path = (
        root
        / "paper_results"
        / "paired_density_5k_data_audit"
        / "paired_density_5k_audit.json"
    )
    audit = _load_json(audit_path)
    if (
        audit.get("status") != "passed"
        or not audit.get("passed")
        or int(audit.get("row_failure_count", -1)) != 0
        or audit.get("dataset_manifest_sha256") != actual_manifest_sha
    ):
        raise RuntimeError("paired-density external audit did not pass cleanly")

    for relative, expected in manifest.get("files", {}).items():
        path = dataset_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"dataset artifact missing: {path}")
        actual = sha256_file(path)
        if actual != expected.get("sha256"):
            raise RuntimeError(f"dataset artifact SHA-256 mismatch: {relative}")

    arrays = {}
    for law in ("C0", "C1"):
        density = np.load(dataset_root / law / "density.npy", mmap_mode="r")
        gravity = np.load(dataset_root / law / "gravity.npy", mmap_mode="r")
        if density.shape != (5000, 16, 32, 32) or density.dtype != np.float32:
            raise RuntimeError(f"{law} density array contract mismatch")
        if gravity.shape != (5000, 1, 32, 32) or gravity.dtype != np.float32:
            raise RuntimeError(f"{law} gravity array contract mismatch")
        if density.flags.writeable or gravity.flags.writeable:
            raise RuntimeError(f"{law} arrays must be loaded read-only")
        arrays[law] = {"density": density, "gravity": gravity}

    metadata = {
        law: _load_jsonl(dataset_root / law / "metadata.jsonl")
        for law in ("C0", "C1")
    }
    if len(metadata["C0"]) != 5000 or len(metadata["C1"]) != 5000:
        raise RuntimeError("metadata row count mismatch")
    for index, (c0, c1) in enumerate(zip(metadata["C0"], metadata["C1"])):
        if int(c0.get("accepted_pair_id", -1)) != index:
            raise RuntimeError(f"C0 accepted_pair_id discontinuity at row {index}")
        if int(c1.get("accepted_pair_id", -1)) != index:
            raise RuntimeError(f"C1 accepted_pair_id discontinuity at row {index}")
        if _metadata_identity_key(c0) != _metadata_identity_key(c1):
            raise RuntimeError(f"C0/C1 metadata identity mismatch at row {index}")

    boundary = validation_boundary_index(metadata["C0"])
    if boundary["counts"] != EXPECTED_VALIDATION_BOUNDARY_COUNTS:
        raise RuntimeError(
            "validation boundary count mismatch: "
            f"{boundary['counts']} != {EXPECTED_VALIDATION_BOUNDARY_COUNTS}"
        )
    if sorted(boundary["by_id"]) != FORMAL_VALIDATION_IDS.tolist():
        raise RuntimeError("validation metadata IDs do not match the frozen split")
    c1_boundary = validation_boundary_index(metadata["C1"])
    if boundary != c1_boundary:
        raise RuntimeError("C0/C1 validation boundary identity mismatch")

    return {
        "dataset_root": dataset_root,
        "dataset_manifest_path": manifest_path,
        "dataset_manifest_sha256": actual_manifest_sha,
        "accepted_pair_count": counts[0],
        "train_pair_count": counts[1],
        "validation_pair_count": counts[2],
        "audit_path": audit_path,
        "audit_status": audit["status"],
        "audit_row_failure_count": int(audit["row_failure_count"]),
        "validation_boundary_counts": boundary["counts"],
        "validation_boundary_by_id": boundary["by_id"],
        "metadata_identity_rows": 5000,
        "train_ids_sha256": TRAIN_IDS_SHA256,
        "validation_ids_sha256": VALIDATION_IDS_SHA256,
        "arrays": arrays,
        "metadata": metadata,
        "manifest": manifest,
    }


def verify_common_initialization(root: Path) -> dict:
    root = Path(root).resolve()
    path = (
        root
        / "experiments"
        / "factorial_5k_initial_states"
        / f"seed_{SEED}.pth"
    )
    metadata_path = path.with_suffix(".json")
    actual = sha256_file(path)
    if actual != COMMON_INITIALIZATION_SHA256:
        raise RuntimeError(
            "initialization SHA-256 mismatch: "
            f"{actual} != {COMMON_INITIALIZATION_SHA256}"
        )
    metadata = _load_json(metadata_path)
    if (
        metadata.get("sha256") != actual
        or int(metadata.get("seed", -1)) != SEED
        or int(metadata.get("parameter_tensors", -1)) != 183
        or int(metadata.get("parameter_count", -1)) != 32_463_561
    ):
        raise RuntimeError("initialization metadata contract mismatch")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or len(state) != 183:
        raise RuntimeError("initialization state_dict tensor count mismatch")
    parameter_count = sum(int(tensor.numel()) for tensor in state.values())
    if parameter_count != 32_463_561:
        raise RuntimeError("initialization state_dict parameter count mismatch")
    return {
        "path": path,
        "metadata_path": metadata_path,
        "sha256": actual,
        "seed": SEED,
        "parameter_tensors": len(state),
        "parameter_count": parameter_count,
        "state_dict": state,
    }
