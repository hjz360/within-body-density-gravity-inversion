"""Frozen source and split contract for the paired C0/C1 supervision study."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


CONFIG_RELATIVE_PATH = Path("docs/superpowers/specs/2026-07-27-c0c1-paired-supervision-v2.json")


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def load_config(root: Path) -> dict[str, Any]:
    root = Path(root).resolve()
    config = json.loads((root / CONFIG_RELATIVE_PATH).read_text(encoding="utf-8"))
    if config.get("status") != "frozen_before_training":
        raise RuntimeError("v2 configuration is not frozen")
    if config.get("input_truth_law") != "C1" or config.get("arms") != ["C0", "C1"]:
        raise RuntimeError("v2 input/arm contract is invalid")
    if config.get("training", {}).get("forward_loss_weight") != 0.0:
        raise RuntimeError("the main C0/C1 experiment must not include forward loss")
    return config


def resolve_sources(root: Path, config: Mapping[str, Any]) -> dict[str, Path]:
    root = Path(root).resolve()
    source = root / str(config["source_dataset"])
    paths = {
        "dataset_root": source,
        "manifest": source / "manifest.json",
        "c0_density": source / "C0" / "density.npy",
        "c1_density": source / "C1" / "density.npy",
        "c0_gravity": source / "C0" / "gravity.npy",
        "c1_gravity": source / "C1" / "gravity.npy",
        "metadata": source / "C1" / "metadata.jsonl",
    }
    paths["input_gravity"] = paths["c1_gravity"]
    if paths["input_gravity"] != paths["c1_gravity"]:
        raise RuntimeError("shared input must be C1 truth gravity")
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    return paths


def load_source_metadata(root: Path, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    path = resolve_sources(root, config)["metadata"]
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [int(row["accepted_pair_id"]) for row in rows]
    if ids != list(range(len(rows))):
        raise RuntimeError("metadata accepted_pair_id values must be contiguous and ordered")
    if any(row.get("density_law") != "C1" for row in rows):
        raise RuntimeError("source metadata must describe continuous true densities")
    return rows


def split_base_ids(metadata: list[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, np.ndarray]:
    split_config = config["split"]
    allowed = set(split_config["id_topologies"])
    ood_name = str(split_config["ood_topology"])
    id_pool = np.asarray([index for index, row in enumerate(metadata) if row["topology"] in allowed], dtype=np.int64)
    ood = np.asarray([index for index, row in enumerate(metadata) if row["topology"] == ood_name], dtype=np.int64)
    if id_pool.size + ood.size != len(metadata) or id_pool.size == 0 or ood.size == 0:
        raise RuntimeError("topology partition does not cover source metadata")
    permutation = np.random.default_rng(int(split_config["split_seed"])).permutation(id_pool)
    train_end = int(np.floor(float(split_config["train_fraction_of_id_pool"]) * id_pool.size))
    validation_end = train_end + int(np.floor(float(split_config["validation_fraction_of_id_pool"]) * id_pool.size))
    split = {
        "train": np.sort(permutation[:train_end]),
        "validation": np.sort(permutation[train_end:validation_end]),
        "id_test": np.sort(permutation[validation_end:]),
        "ood_layered": np.sort(ood),
    }
    if min(values.size for values in split.values()) <= 0:
        raise RuntimeError("one or more v2 split partitions are empty")
    return split


def verify_source_dataset(root: Path, config: Mapping[str, Any], max_rows: int | None = None) -> dict[str, Any]:
    paths = resolve_sources(root, config)
    expected_hashes = dict(config["source_hashes"])
    key_to_path = {
        "manifest.json": paths["manifest"],
        "C0/density.npy": paths["c0_density"],
        "C1/density.npy": paths["c1_density"],
        "C1/gravity.npy": paths["c1_gravity"],
        "C1/metadata.jsonl": paths["metadata"],
    }
    observed_hashes = {key: sha256_file(path) for key, path in key_to_path.items()}
    mismatches = [key for key, expected in expected_hashes.items() if observed_hashes.get(key) != expected]
    if mismatches:
        raise RuntimeError("source hash mismatch: " + ", ".join(mismatches))
    c0 = np.load(paths["c0_density"], mmap_mode="r")
    c1 = np.load(paths["c1_density"], mmap_mode="r")
    input_gravity = np.load(paths["input_gravity"], mmap_mode="r")
    if c0.shape != c1.shape or c0.shape[1:] != (16, 32, 32):
        raise RuntimeError("paired density shape mismatch")
    if input_gravity.shape != (c0.shape[0], 1, 32, 32):
        raise RuntimeError("shared gravity shape mismatch")
    count = c0.shape[0] if max_rows is None else min(int(max_rows), c0.shape[0])
    support_mismatch_count = 0
    max_relative_mass_difference = 0.0
    for index in range(count):
        left = np.asarray(c0[index])
        right = np.asarray(c1[index])
        support_mismatch_count += int(not np.array_equal(left > 0.0, right > 0.0))
        denominator = max(float(np.abs(left).sum()), float(np.abs(right).sum()), np.finfo(np.float64).eps)
        max_relative_mass_difference = max(max_relative_mass_difference, float(abs(left.sum() - right.sum()) / denominator))
    return {
        "source_hashes": observed_hashes,
        "input_law": "C1",
        "shared_input_identity": paths["input_gravity"] == paths["c1_gravity"],
        "checked_rows": count,
        "support_mismatch_count": support_mismatch_count,
        "max_relative_mass_difference": max_relative_mass_difference,
        "splits": {key: values.tolist() for key, values in split_base_ids(load_source_metadata(root, config), config).items()},
    }
