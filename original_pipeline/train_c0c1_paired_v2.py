"""Equal-budget paired C0/C1 training with one shared continuous-truth input."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import c0c1_paired_v2_contract as contract
from paired_density_c0c1_contract import PilotPaths
from train_formal_20k import make_unet
from train_paired_density_c0c1 import execute_training


ROOT = Path(__file__).resolve().parent
EXPERIMENT_NAMESPACE = "c0c1_paired_v2_r2"


def _ids_digest(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype=np.int64).tobytes()).hexdigest()


def _json_digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def run_paths(root: Path, arm: str, seed: int, *, epochs: int = 120) -> PilotPaths:
    if arm not in ("C0", "C1"):
        raise ValueError("arm must be C0 or C1")
    run_dir = Path(root).resolve() / "experiments" / EXPERIMENT_NAMESPACE / arm / f"seed_{int(seed)}"
    return PilotPaths(
        run_dir=run_dir,
        primary_weights=run_dir / f"epoch_{int(epochs):03d}_weights.pth",
        best_weights=run_dir / "best_validation_weights.pth",
        last_checkpoint=run_dir / "last_checkpoint.pth",
        history=run_dir / "history.npz",
        metadata=run_dir / "metadata.json",
        failure=run_dir / "failure.json",
        snapshot_weights={int(epochs): run_dir / f"epoch_{int(epochs):03d}_weights.pth"},
    )


def arm_config(root: Path, arm: str, seed: int) -> dict:
    root = Path(root).resolve()
    frozen = contract.load_config(root)
    if arm not in frozen["arms"]:
        raise ValueError("unknown arm")
    if int(seed) not in frozen["seeds"]:
        raise ValueError("seed is not frozen for this experiment")
    split = contract.split_base_ids(contract.load_source_metadata(root, frozen), frozen)
    training = frozen["training"]
    return {
        "model_id": arm,
        "density_law": arm,
        "mode": "formal",
        "seed": int(seed),
        "train_ids": split["train"].tolist(),
        "validation_ids": split["validation"].tolist(),
        "train_ids_sha256": _ids_digest(split["train"]),
        "validation_ids_sha256": _ids_digest(split["validation"]),
        "max_epochs": int(training["epochs"]),
        "batch_size": int(training["batch_size"]),
        "learning_rate": float(training["learning_rate"]),
        "smooth_l1_beta": float(training["smooth_l1_beta"]),
        "lambda_tv_xy": float(training["lambda_tv_xy"]),
        "forward_loss_weight": float(training["forward_loss_weight"]),
        "noise_levels": list(training["noise_levels"]),
        "validation_noise_levels": [0.0],
        "scheduler": None,
        "early_stopping": False,
        "freeze_any_layer": False,
    }


def load_arm_sources(root: Path, arm: str) -> dict:
    root = Path(root).resolve()
    frozen = contract.load_config(root)
    if arm not in frozen["arms"]:
        raise ValueError("unknown arm")
    paths = contract.resolve_sources(root, frozen)
    split = contract.split_base_ids(contract.load_source_metadata(root, frozen), frozen)
    return {
        "images": np.load(paths["input_gravity"], mmap_mode="r"),
        "targets": np.load(paths[f"{arm.lower()}_density"], mmap_mode="r"),
        "train_ids": split["train"],
        "validation_ids": split["validation"],
        "id_test_ids": split["id_test"],
        "ood_ids": split["ood_layered"],
        "source_paths": paths,
    }


def _initialization_path(root: Path, seed: int) -> Path:
    return Path(root).resolve() / "experiments" / EXPERIMENT_NAMESPACE / "initialization" / f"seed_{int(seed)}.pth"


def common_initial_state(root: Path, seed: int) -> tuple[dict, str, Path]:
    path = _initialization_path(root, seed)
    if path.exists():
        state = torch.load(path, map_location="cpu", weights_only=True)
        return state, contract.sha256_file(path), path
    if path.parent.exists() and any(path.parent.iterdir()):
        # A sibling seed may exist, which is expected; only this exact path is immutable.
        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        state = {key: value.detach().cpu().clone() for key, value in make_unet().state_dict().items()}
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)
    return state, contract.sha256_file(path), path


def _code_identity(root: Path) -> dict:
    root = Path(root)
    files = (
        "train_c0c1_paired_v2.py",
        "c0c1_paired_v2_contract.py",
        "train_paired_density_c0c1.py",
        "train_formal_20k.py",
        str(contract.CONFIG_RELATIVE_PATH),
    )
    hashes = {name: contract.sha256_file(root / name) for name in files}
    return {"files": hashes, "combined_sha256": _json_digest(hashes)}


def run_training(root: Path, arm: str, seed: int, *, device: str | None = None, num_workers: int = 0, resume: bool = False) -> PilotPaths:
    root = Path(root).resolve()
    frozen = contract.load_config(root)
    source_evidence = contract.verify_source_dataset(root, frozen)
    config = arm_config(root, arm, seed)
    sources = load_arm_sources(root, arm)
    initial_state, initial_sha256, initial_path = common_initial_state(root, seed)
    code = _code_identity(root)
    identity = {
        "v2_config_sha256": contract.sha256_file(root / contract.CONFIG_RELATIVE_PATH),
        "source_hashes": source_evidence["source_hashes"],
        "shared_input_sha256": source_evidence["source_hashes"]["C1/gravity.npy"],
        "input_law": "C1",
        "initial_sha256": initial_sha256,
        "initial_path": str(initial_path),
        "code_sha256": code["combined_sha256"],
        "code_files_sha256": code["files"],
        "stochastic_schedule_sha256": _json_digest({key: value for key, value in config.items() if key not in ("model_id", "density_law")}),
    }
    return execute_training(
        config,
        run_paths(root, arm, seed, epochs=int(config["max_epochs"])),
        sources["images"],
        sources["targets"],
        initial_state,
        identity,
        make_unet,
        device=device,
        num_workers=num_workers,
        resume=bool(resume),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=("C0", "C1"))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    paths = run_training(ROOT, args.arm, args.seed, device=args.device, num_workers=args.num_workers, resume=args.resume)
    print(json.dumps({"status": "complete", "run_dir": str(paths.run_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
