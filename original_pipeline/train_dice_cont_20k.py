"""Reproducible fair-control 20k Dice+Sigmoid training experiment."""

import argparse
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn as nn

from train_formal_20k import (
    DEFAULT_SEED,
    DEFAULT_TOTAL_COUNT,
    DEFAULT_TRAIN_COUNT,
    DEFAULT_VAL_COUNT,
    Unet as SoftplusUnet,
    experiment_paths,
    load_base_arrays,
    mean_epoch_loss,
    save_history,
    save_metadata,
    set_reproducible,
    split_digest,
)
from training_data_20k import AugmentedArrayDataset, make_fixed_split


EXPECTED_SPLIT_SHA256 = "01991ea649898806a7a5e705a8c4bd15acf2899006d92fe5dd46359e8fe99927"


class Unet(SoftplusUnet):
    def __init__(self):
        super().__init__()
        self.final[-1] = nn.Sigmoid()


def make_unet():
    return Unet()


def dice_score(prediction, target, smooth=1.0):
    batch = prediction.shape[0]
    pred_flat = prediction.reshape(batch, -1)
    target_flat = target.reshape(batch, -1)
    numerator = 2.0 * (pred_flat * target_flat).sum(1) + smooth
    denominator = pred_flat.square().sum(1) + target_flat.square().sum(1) + smooth
    return (numerator / denominator).mean()


def dice_loss(prediction, target):
    return 1.0 - dice_score(prediction, target)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total-count", type=int, default=DEFAULT_TOTAL_COUNT)
    parser.add_argument("--train-count", type=int, default=DEFAULT_TRAIN_COUNT)
    parser.add_argument("--val-count", type=int, default=DEFAULT_VAL_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--min-delta", type=float, default=1e-6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--experiment-name", default="dice_cont_20k")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "tra&val",
    )
    parser.add_argument("--expected-split-sha256", default=EXPECTED_SPLIT_SHA256)
    parser.add_argument("--resume", action="store_true")
    return parser


def validate_split_digest(args, current_split_digest):
    is_formal_configuration = (
        args.total_count == DEFAULT_TOTAL_COUNT
        and args.train_count == DEFAULT_TRAIN_COUNT
        and args.val_count == DEFAULT_VAL_COUNT
        and args.seed == DEFAULT_SEED
    )
    if (
        is_formal_configuration
        and args.expected_split_sha256
        and current_split_digest != args.expected_split_sha256
    ):
        raise ValueError(
            "Dice split does not match the approved Softplus 20k split: "
            f"{current_split_digest}"
        )


def run_training(args):
    if args.train_count + args.val_count != args.total_count:
        raise ValueError("train-count + val-count must equal total-count")

    root = Path(__file__).resolve().parent
    paths = experiment_paths(root, args.experiment_name)
    paths["experiment_dir"].mkdir(parents=True, exist_ok=True)
    set_reproducible(args.seed)

    train_indices, val_indices = make_fixed_split(
        args.total_count, args.train_count, args.val_count, args.seed
    )
    current_split_digest = split_digest(train_indices, val_indices)
    validate_split_digest(args, current_split_digest)
    np.savez(paths["split"], train=train_indices, val=val_indices)

    config = {
        "experiment_name": args.experiment_name,
        "method": "Dice+Sigmoid",
        "objective": "dice_loss",
        "total_count": args.total_count,
        "train_count": args.train_count,
        "val_count": args.val_count,
        "training_views": args.train_count * 3,
        "noise_levels": [0.0, 0.02, 0.04],
        "seed": args.seed,
        "split_sha256": current_split_digest,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "min_delta": args.min_delta,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "lambda_tv": 0.0,
        "data_dir": str(args.data_dir.resolve()),
    }
    save_metadata(paths["metadata"], config, "loading_data", None, None, 0.0)
    images, masks = load_base_arrays(args.data_dir, args.total_count)

    train_dataset = AugmentedArrayDataset(
        images, masks, train_indices, noise_levels=(0.0, 0.02, 0.04), seed=args.seed
    )
    val_dataset = AugmentedArrayDataset(
        images, masks, val_indices, noise_levels=(0.0,), seed=args.seed
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    model = make_unet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    start_epoch = 0
    best_epoch = None
    best_val = float("inf")
    epochs_without_improvement = 0
    train_losses = []
    val_losses = []
    epoch_times = []

    if args.resume:
        if not paths["last_checkpoint"].exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {paths['last_checkpoint']}")
        checkpoint = torch.load(paths["last_checkpoint"], map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        best_epoch = checkpoint["best_epoch"]
        best_val = checkpoint["best_val"]
        epochs_without_improvement = checkpoint["epochs_without_improvement"]
        train_losses = checkpoint["train_losses"]
        val_losses = checkpoint["val_losses"]
        epoch_times = checkpoint["epoch_times"]
        print(f"Resuming at epoch {start_epoch + 1}", flush=True)

    training_started = time.time()
    save_metadata(paths["metadata"], config, "training", best_epoch, best_val, 0.0)
    for epoch in range(start_epoch, args.max_epochs):
        epoch_started = time.time()
        train_dataset.set_epoch(epoch)
        generator = torch.Generator().manual_seed(args.seed + epoch)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            generator=generator,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )
        train_loss = mean_epoch_loss(
            model, train_loader, dice_loss, 0.0, device, optimizer=optimizer
        )
        with torch.no_grad():
            val_loss = mean_epoch_loss(model, val_loader, dice_loss, 0.0, device)
        epoch_time = time.time() - epoch_started
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        epoch_times.append(epoch_time)

        if val_loss < best_val - args.min_delta:
            best_val = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), paths["weights"])
            torch.save(
                {"epoch": epoch, "model": model.state_dict(), "val_loss": val_loss, "config": config},
                paths["best_checkpoint"],
            )
        else:
            epochs_without_improvement += 1

        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_epoch": best_epoch,
            "best_val": best_val,
            "epochs_without_improvement": epochs_without_improvement,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "epoch_times": epoch_times,
            "config": config,
        }
        torch.save(checkpoint, paths["last_checkpoint"])
        save_history(paths["history"], train_losses, val_losses, epoch_times, config)
        elapsed = time.time() - training_started
        save_metadata(paths["metadata"], config, "training", best_epoch, best_val, elapsed)
        print(
            f"Epoch {epoch + 1:03d}/{args.max_epochs}: "
            f"train={train_loss:.6f} val={val_loss:.6f} "
            f"best={best_val:.6f} time={epoch_time:.1f}s",
            flush=True,
        )
        if epochs_without_improvement >= args.patience:
            print(f"Early stopping after {args.patience} epochs without improvement", flush=True)
            break

    elapsed = time.time() - training_started
    save_metadata(paths["metadata"], config, "complete", best_epoch, best_val, elapsed)
    print(f"Best epoch: {best_epoch + 1}; best val loss: {best_val:.6f}", flush=True)
    print(f"Best weights: {paths['weights']}", flush=True)
    return paths


def main():
    run_training(build_parser().parse_args())


if __name__ == "__main__":
    main()
