"""Deterministic equal-budget training for the paired C0F0/C1F0 pilot."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import shutil
import time
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn

import paired_density_c0c1_contract as contract
from train_formal_20k import make_unet


ROOT = Path(__file__).resolve().parent


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True, choices=tuple(contract.MODEL_TO_LAW))
    parser.add_argument("--mode", required=True, choices=contract.MODES)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser


def stochastic_schedule_digest(config) -> str:
    excluded = {"model_id", "density_law"}
    payload = {key: value for key, value in config.items() if key not in excluded}
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def standard_noise(shape, seed, epoch, accepted_id, view_index):
    seed_sequence = np.random.SeedSequence(
        [int(seed), int(epoch), int(accepted_id), int(view_index)]
    )
    rng = np.random.default_rng(seed_sequence)
    return rng.normal(0.0, 1.0, tuple(shape)).astype(np.float32)


class PairedDensityDataset(torch.utils.data.Dataset):
    """Read indexed arrays through deterministic relative-noise views."""

    def __init__(self, images, masks, indices, noise_levels=(0.0,), seed=0):
        self.images = images
        self.masks = masks
        self.indices = np.asarray(indices, dtype=np.int64)
        self.noise_levels = tuple(float(level) for level in noise_levels)
        if self.indices.ndim != 1 or self.indices.size == 0:
            raise ValueError("indices must be one nonempty vector")
        if not self.noise_levels or any(level < 0.0 for level in self.noise_levels):
            raise ValueError("noise_levels must be nonempty and nonnegative")
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self):
        return int(len(self.indices) * len(self.noise_levels))

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __getitem__(self, index):
        index = int(index)
        base_position = index % len(self.indices)
        view_index = index // len(self.indices)
        if view_index >= len(self.noise_levels):
            raise IndexError(index)
        accepted_id = int(self.indices[base_position])
        image = np.array(self.images[accepted_id], dtype=np.float32, copy=True)
        mask = np.array(self.masks[accepted_id], dtype=np.float32, copy=True)
        noise_level = self.noise_levels[view_index]
        if noise_level:
            noise = standard_noise(
                image.shape,
                self.seed,
                self.epoch,
                accepted_id,
                view_index,
            )
            image += np.float32(noise_level * float(image.max())) * noise
        return image, mask


def tv_xy(prediction):
    dy = torch.mean(
        torch.abs(prediction[:, :, 1:, :] - prediction[:, :, :-1, :])
    )
    dx = torch.mean(
        torch.abs(prediction[:, :, :, 1:] - prediction[:, :, :, :-1])
    )
    return dy + dx


def _finite_gradients(parameters: Iterable[torch.nn.Parameter]) -> bool:
    for parameter in parameters:
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            return False
    return True


def run_epoch(
    model,
    loader,
    criterion,
    lambda_tv_xy,
    device,
    optimizer=None,
):
    training = optimizer is not None
    model.train(training)
    sums = {"smooth_l1": 0.0, "tv_xy": 0.0, "total": 0.0}
    sample_count = 0
    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            outputs = model(images)
            if not torch.isfinite(outputs).all():
                raise FloatingPointError("model produced a non-finite output")
            density_loss = criterion(outputs, masks)
            variation = tv_xy(outputs)
            total = density_loss + float(lambda_tv_xy) * variation
            if not torch.isfinite(total):
                raise FloatingPointError("training produced a non-finite loss")
            if optimizer is not None:
                total.backward()
                if not _finite_gradients(model.parameters()):
                    raise FloatingPointError("training produced a non-finite gradient")
                optimizer.step()
        count = int(images.shape[0])
        sums["smooth_l1"] += float(density_loss.detach()) * count
        sums["tv_xy"] += float(variation.detach()) * count
        sums["total"] += float(total.detach()) * count
        sample_count += count
    if sample_count == 0:
        raise RuntimeError("epoch loader produced no samples")
    return {
        "smooth_l1": sums["smooth_l1"] / sample_count,
        "tv_xy": sums["tv_xy"] / sample_count,
        "total": sums["total"] / sample_count,
        "sample_count": sample_count,
    }


def initialize_model(root=ROOT, model_factory=make_unet):
    evidence = contract.verify_common_initialization(root)
    model = model_factory()
    model.load_state_dict(evidence["state_dict"], strict=True)
    if not all(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("all model parameters must remain trainable")
    return model, {key: value for key, value in evidence.items() if key != "state_dict"}


def _cpu_state_dict(model):
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def _atomic_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _atomic_history(path, history, config):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(
            handle,
            train_smooth_l1=np.asarray(
                [row["smooth_l1"] for row in history["train"]], dtype=np.float64
            ),
            train_tv_xy=np.asarray(
                [row["tv_xy"] for row in history["train"]], dtype=np.float64
            ),
            train_total=np.asarray(
                [row["total"] for row in history["train"]], dtype=np.float64
            ),
            validation_smooth_l1=np.asarray(
                [row["smooth_l1"] for row in history["validation"]],
                dtype=np.float64,
            ),
            validation_tv_xy=np.asarray(
                [row["tv_xy"] for row in history["validation"]], dtype=np.float64
            ),
            validation_total=np.asarray(
                [row["total"] for row in history["validation"]], dtype=np.float64
            ),
            epoch_seconds=np.asarray(history["epoch_seconds"], dtype=np.float64),
            config=json.dumps(config, sort_keys=True),
        )
    temporary.replace(path)


def atomic_torch_save_verified(payload, path) -> str:
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    torch.load(temporary, map_location="cpu", weights_only=False)
    temporary.replace(path)
    digest = contract.sha256_file(path)
    if contract.sha256_file(path) != digest:
        raise RuntimeError(f"atomic Torch artifact hash changed: {path}")
    return digest


def load_resume_checkpoint(path):
    return torch.load(Path(path), map_location="cpu", weights_only=False)


def validate_resume_contract(expected, actual):
    keys = sorted(set(expected) | set(actual))
    mismatches = [key for key in keys if expected.get(key) != actual.get(key)]
    if mismatches:
        raise ValueError("resume contract mismatch: " + ", ".join(mismatches))


def _set_reproducible(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def _resume_identity(config, identity):
    return {
        **json.loads(json.dumps(config, sort_keys=True)),
        **json.loads(json.dumps(identity, sort_keys=True)),
        "stochastic_schedule_sha256": stochastic_schedule_digest(config),
    }


def _save_metadata(
    path,
    status,
    config,
    resume_identity,
    device,
    history,
    best_epoch,
    best_validation_total,
    primary_epoch,
    elapsed_seconds,
    artifact_sha256=None,
):
    _atomic_json(
        path,
        {
            "schema_version": 1,
            "status": str(status),
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "device": str(device),
            "config": config,
            "resume_contract": resume_identity,
            "completed_epochs": len(history["train"]),
            "best_epoch": best_epoch,
            "best_validation_total": best_validation_total,
            "primary_epoch": primary_epoch,
            "final_train": history["train"][-1] if history["train"] else None,
            "final_validation": (
                history["validation"][-1] if history["validation"] else None
            ),
            "elapsed_seconds": float(elapsed_seconds),
            "artifact_sha256": artifact_sha256 or {},
        },
    )


def _rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda"):
        torch.cuda.set_rng_state_all(state["cuda"])


def _execute_training_impl(
    config,
    paths,
    images,
    masks,
    initial_state,
    identity,
    model_factory=make_unet,
    device=None,
    num_workers=0,
    resume=False,
    stop_after_epoch=None,
):
    """Execute one already-verified fit; dependency injection keeps tests small."""

    config = json.loads(json.dumps(config, sort_keys=True))
    paths = paths
    device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if paths.run_dir.exists() and not resume:
        if any(paths.run_dir.iterdir()):
            raise FileExistsError(f"experiment directory is not empty: {paths.run_dir}")
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    if resume and not paths.last_checkpoint.exists():
        raise FileNotFoundError(f"resume checkpoint not found: {paths.last_checkpoint}")

    _set_reproducible(config["seed"])
    model = model_factory()
    model.load_state_dict(initial_state, strict=True)
    if not all(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("all model parameters must remain trainable")
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(config["learning_rate"])
    )
    criterion = nn.SmoothL1Loss(beta=float(config["smooth_l1_beta"]))
    train_dataset = PairedDensityDataset(
        images,
        masks,
        config["train_ids"],
        noise_levels=config["noise_levels"],
        seed=config["seed"],
    )
    validation_dataset = PairedDensityDataset(
        images,
        masks,
        config["validation_ids"],
        noise_levels=config["validation_noise_levels"],
        seed=config["seed"],
    )
    pin_memory = device.type == "cuda"
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=pin_memory,
    )
    resume_identity = _resume_identity(config, identity)
    history = {"train": [], "validation": [], "epoch_seconds": []}
    start_epoch = 0
    best_epoch = None
    best_validation_total = float("inf")
    if resume:
        checkpoint = load_resume_checkpoint(paths.last_checkpoint)
        validate_resume_contract(
            resume_identity, checkpoint["resume_contract"]
        )
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"])
        history = checkpoint["history"]
        best_epoch = checkpoint["best_epoch"]
        best_validation_total = float(checkpoint["best_validation_total"])
        _restore_rng_state(checkpoint["rng_state"])

    started = time.time()
    _save_metadata(
        paths.metadata,
        "training",
        config,
        resume_identity,
        device,
        history,
        best_epoch,
        None if not np.isfinite(best_validation_total) else best_validation_total,
        None,
        0.0,
    )
    try:
        total_epochs = int(config["max_epochs"])
        for epoch in range(start_epoch, total_epochs):
            epoch_started = time.time()
            train_dataset.set_epoch(epoch)
            generator = torch.Generator().manual_seed(int(config["seed"]) + epoch)
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=int(config["batch_size"]),
                shuffle=True,
                generator=generator,
                num_workers=int(num_workers),
                pin_memory=pin_memory,
            )
            train_row = run_epoch(
                model,
                train_loader,
                criterion,
                config["lambda_tv_xy"],
                device,
                optimizer=optimizer,
            )
            with torch.no_grad():
                validation_row = run_epoch(
                    model,
                    validation_loader,
                    criterion,
                    config["lambda_tv_xy"],
                    device,
                )
            history["train"].append(train_row)
            history["validation"].append(validation_row)
            history["epoch_seconds"].append(time.time() - epoch_started)
            completed_epoch = epoch + 1
            if validation_row["total"] < best_validation_total:
                best_validation_total = float(validation_row["total"])
                best_epoch = completed_epoch
                atomic_torch_save_verified(
                    _cpu_state_dict(model), paths.best_weights
                )
            if completed_epoch in paths.snapshot_weights:
                atomic_torch_save_verified(
                    _cpu_state_dict(model),
                    paths.snapshot_weights[completed_epoch],
                )
            checkpoint = {
                "epoch": completed_epoch,
                "model": _cpu_state_dict(model),
                "optimizer": optimizer.state_dict(),
                "best_epoch": best_epoch,
                "best_validation_total": best_validation_total,
                "history": history,
                "resume_contract": resume_identity,
                "rng_state": _rng_state(),
            }
            atomic_torch_save_verified(checkpoint, paths.last_checkpoint)
            _atomic_history(paths.history, history, config)
            _save_metadata(
                paths.metadata,
                "training",
                config,
                resume_identity,
                device,
                history,
                best_epoch,
                best_validation_total,
                None,
                time.time() - started,
            )
            print(
                f"{config['model_id']} {config['mode']} "
                f"epoch {completed_epoch:03d}/{total_epochs}: "
                f"train={train_row['total']:.6f} "
                f"val={validation_row['total']:.6f} "
                f"best={best_epoch:03d} "
                f"time={history['epoch_seconds'][-1]:.1f}s",
                flush=True,
            )
            if stop_after_epoch is not None and completed_epoch >= int(
                stop_after_epoch
            ):
                return paths

        if len(history["train"]) != int(config["max_epochs"]):
            raise RuntimeError("fixed epoch budget was not completed")
        if not paths.primary_weights.exists():
            atomic_torch_save_verified(
                _cpu_state_dict(model), paths.primary_weights
            )
        artifact_sha256 = {
            "primary_weights": contract.sha256_file(paths.primary_weights),
            "best_weights": contract.sha256_file(paths.best_weights),
            "last_checkpoint": contract.sha256_file(paths.last_checkpoint),
            "history": contract.sha256_file(paths.history),
        }
        for epoch, path in paths.snapshot_weights.items():
            artifact_sha256[f"epoch_{int(epoch):03d}_weights"] = (
                contract.sha256_file(path)
            )
        _save_metadata(
            paths.metadata,
            "complete",
            config,
            resume_identity,
            device,
            history,
            best_epoch,
            best_validation_total,
            int(config["max_epochs"]),
            time.time() - started,
            artifact_sha256=artifact_sha256,
        )
        return paths
    except Exception as error:
        _atomic_json(
            paths.failure,
            {
                "schema_version": 1,
                "status": "failed",
                "model_id": config.get("model_id"),
                "mode": config.get("mode"),
                "completed_epochs": len(history["train"]),
                "error_type": type(error).__name__,
                "error": str(error),
                "failed_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        raise


def execute_training(*args, **kwargs):
    """Run one fit without leaking process-global RNG/backend state."""

    deterministic_enabled = torch.are_deterministic_algorithms_enabled()
    deterministic_warn_only = (
        torch.is_deterministic_algorithms_warn_only_enabled()
    )
    cudnn_benchmark = torch.backends.cudnn.benchmark
    cudnn_deterministic = torch.backends.cudnn.deterministic
    outer_rng_state = _rng_state()
    try:
        return _execute_training_impl(*args, **kwargs)
    finally:
        _restore_rng_state(outer_rng_state)
        torch.use_deterministic_algorithms(
            deterministic_enabled, warn_only=deterministic_warn_only
        )
        torch.backends.cudnn.benchmark = cudnn_benchmark
        torch.backends.cudnn.deterministic = cudnn_deterministic


def _code_identity(root):
    root = Path(root)
    names = (
        "train_paired_density_c0c1.py",
        "paired_density_c0c1_contract.py",
        "paired_density_c0c1_metrics.py",
        "train_formal_20k.py",
    )
    hashes = {name: contract.sha256_file(root / name) for name in names}
    encoded = json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return {"files": hashes, "combined_sha256": hashlib.sha256(encoded).hexdigest()}


def _formal_preflight_path(root):
    return (
        Path(root)
        / "paper_results"
        / "paired_density_c0c1_pilot"
        / "formal_preflight.json"
    )


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _verify_smoke_evidence(root, identity):
    keys = ("dataset_manifest_sha256", "initial_sha256", "code_sha256")
    evidence = {}
    schedules = set()
    for model_id in contract.MODEL_TO_LAW:
        metadata_path = contract.experiment_paths(root, model_id, "smoke").metadata
        if not metadata_path.exists():
            raise FileNotFoundError(f"smoke metadata not found: {metadata_path}")
        metadata = _read_json(metadata_path)
        resume_contract = metadata.get("resume_contract", {})
        if metadata.get("status") != "complete" or int(
            metadata.get("completed_epochs", -1)
        ) != 5:
            raise RuntimeError(f"incomplete smoke run: {model_id}")
        if resume_contract.get("audit_status") != "passed":
            raise RuntimeError(f"smoke audit did not pass: {model_id}")
        mismatches = [
            key
            for key in keys
            if resume_contract.get(key) != identity.get(key)
        ]
        if mismatches:
            raise RuntimeError(
                f"smoke identity mismatch for {model_id}: {', '.join(mismatches)}"
            )
        schedules.add(resume_contract.get("stochastic_schedule_sha256"))
        evidence[model_id] = {
            "metadata": str(metadata_path),
            "metadata_sha256": contract.sha256_file(metadata_path),
            "resume_contract": {
                key: resume_contract.get(key)
                for key in (*keys, "stochastic_schedule_sha256")
            },
        }
    if len(schedules) != 1 or None in schedules:
        raise RuntimeError("smoke stochastic schedules do not match")
    return evidence


def perform_formal_preflight(root, model_id, identity, resume=False):
    root = Path(root).resolve()
    if model_id not in contract.MODEL_TO_LAW:
        raise ValueError(f"unknown model_id: {model_id}")
    if not torch.cuda.is_available():
        raise RuntimeError("formal training requires an available CUDA device")
    smoke = _verify_smoke_evidence(root, identity)
    preflight_path = _formal_preflight_path(root)
    keys = ("dataset_manifest_sha256", "initial_sha256", "code_sha256")

    if preflight_path.exists():
        preflight = _read_json(preflight_path)
        if preflight.get("status") != "passed":
            raise RuntimeError("existing formal preflight did not pass")
        mismatches = [
            key
            for key in keys
            if preflight.get("identity", {}).get(key) != identity.get(key)
        ]
        if mismatches:
            raise RuntimeError(
                "formal preflight identity mismatch: " + ", ".join(mismatches)
            )
        for smoke_id, row in smoke.items():
            recorded = preflight.get("smoke", {}).get(smoke_id, {})
            if recorded.get("metadata_sha256") != row["metadata_sha256"]:
                raise RuntimeError(
                    f"smoke metadata changed after formal preflight: {smoke_id}"
                )
        if model_id == "C1F0":
            c0_metadata_path = contract.experiment_paths(
                root, "C0F0", "formal"
            ).metadata
            if not c0_metadata_path.exists():
                raise RuntimeError("C0F0 formal metadata is required before C1F0")
            c0_metadata = _read_json(c0_metadata_path)
            if c0_metadata.get("status") != "complete" or int(
                c0_metadata.get("completed_epochs", -1)
            ) != 120:
                raise RuntimeError("C0F0 formal training is not complete")
            c0_contract = c0_metadata.get("resume_contract", {})
            mismatches = [
                key for key in keys if c0_contract.get(key) != identity.get(key)
            ]
            if mismatches:
                raise RuntimeError(
                    "C0F0 formal identity mismatch: " + ", ".join(mismatches)
                )
        return preflight

    if model_id != "C0F0":
        raise RuntimeError("C0F0 must create the formal preflight before C1F0")
    formal_paths = {
        candidate: contract.experiment_paths(root, candidate, "formal")
        for candidate in contract.MODEL_TO_LAW
    }
    nonempty = [
        candidate
        for candidate, paths in formal_paths.items()
        if paths.run_dir.exists() and any(paths.run_dir.iterdir())
    ]
    if nonempty and not resume:
        raise RuntimeError(
            "formal output roots are not empty: " + ", ".join(nonempty)
        )
    usage = shutil.disk_usage(root)
    required_free_bytes = 3 * 1024**3
    if int(usage.free) < required_free_bytes:
        raise RuntimeError(
            "formal preflight requires at least 3 GiB free on the C drive"
        )
    preflight = {
        "schema_version": 1,
        "status": "passed",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "identity": {key: identity.get(key) for key in keys},
        "free_bytes": int(usage.free),
        "required_free_bytes": required_free_bytes,
        "cuda_available": True,
        "cuda_device": torch.cuda.get_device_name(0),
        "formal_run_dirs_empty": not nonempty,
        "smoke": smoke,
    }
    preflight_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(preflight_path, preflight)
    return preflight


def load_training_sources(root, model_id, mode):
    root = Path(root).resolve()
    evidence = contract.verify_dataset_contract(root)
    law = contract.MODEL_TO_LAW[model_id]
    return {
        "images": evidence["arrays"][law]["gravity"],
        "masks": evidence["arrays"][law]["density"],
        "source_evidence": {
            "dataset_manifest_sha256": evidence["dataset_manifest_sha256"],
            "audit_status": evidence["audit_status"],
            "audit_row_failure_count": evidence["audit_row_failure_count"],
            "train_ids_sha256": contract.pilot_config(model_id, mode)[
                "train_ids_sha256"
            ],
            "validation_ids_sha256": contract.pilot_config(model_id, mode)[
                "validation_ids_sha256"
            ],
            "validation_boundary_counts": evidence[
                "validation_boundary_counts"
            ],
        },
    }


def run_training(args, root=ROOT, model_factory=make_unet):
    root = Path(root).resolve()
    config = contract.pilot_config(args.model_id, args.mode)
    paths = contract.experiment_paths(root, args.model_id, args.mode)
    sources = load_training_sources(root, args.model_id, args.mode)
    initialization = contract.verify_common_initialization(root)
    code = _code_identity(root)
    identity = {
        **sources["source_evidence"],
        "initial_sha256": initialization["sha256"],
        "code_sha256": code["combined_sha256"],
        "code_files_sha256": code["files"],
    }
    if args.mode == "formal":
        perform_formal_preflight(
            root, args.model_id, identity, resume=args.resume
        )
    return execute_training(
        config,
        paths,
        sources["images"],
        sources["masks"],
        initialization["state_dict"],
        identity,
        model_factory=model_factory,
        num_workers=args.num_workers,
        resume=args.resume,
    )


def main():
    paths = run_training(build_parser().parse_args(), root=ROOT)
    print(f"Primary weights: {paths.primary_weights}", flush=True)


if __name__ == "__main__":
    main()
