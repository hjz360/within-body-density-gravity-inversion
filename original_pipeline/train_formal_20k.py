"""Reproducible 20k-sample Softplus+TV training experiment."""

import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import scipy.io
import torch
import torch.nn as nn

from training_data_20k import AugmentedArrayDataset, make_fixed_split


DEFAULT_TOTAL_COUNT = 20000
DEFAULT_TRAIN_COUNT = 19000
DEFAULT_VAL_COUNT = 1000
DEFAULT_SEED = 20260611
DENSITY_SCALE = 1000.0
GRID_SIZE = 32
START_FM = 64


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ELU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ELU(inplace=True),
        )

    def forward(self, inputs):
        return self.conv(inputs)


class Unet(nn.Module):
    def __init__(self):
        super().__init__()
        s = START_FM
        self.double_conv1 = DoubleConv(1, s)
        self.maxpool1 = nn.Sequential(
            nn.Conv2d(s, s, 2, 2, 0, bias=True), nn.BatchNorm2d(s), nn.ELU(inplace=True)
        )
        self.double_conv2 = DoubleConv(s, s * 2)
        self.maxpool2 = nn.Sequential(
            nn.Conv2d(s * 2, s * 2, 2, 2, 0, bias=True),
            nn.BatchNorm2d(s * 2),
            nn.ELU(inplace=True),
        )
        self.double_conv3 = DoubleConv(s * 2, s * 4)
        self.maxpool3 = nn.Sequential(
            nn.Conv2d(s * 4, s * 4, 2, 2, 0, bias=True),
            nn.BatchNorm2d(s * 4),
            nn.ELU(inplace=True),
        )
        self.double_conv4 = DoubleConv(s * 4, s * 8)
        self.maxpool4 = nn.Sequential(
            nn.Conv2d(s * 8, s * 8, 2, 2, 0, bias=True),
            nn.BatchNorm2d(s * 8),
            nn.ELU(inplace=True),
        )
        self.double_conv5 = DoubleConv(s * 8, s * 16)
        self.t_conv4 = nn.ConvTranspose2d(s * 16, s * 8, 2, 2)
        self.ex_double_conv4 = DoubleConv(s * 16, s * 8)
        self.t_conv3 = nn.ConvTranspose2d(s * 8, s * 4, 2, 2)
        self.ex_double_conv3 = DoubleConv(s * 8, s * 4)
        self.t_conv2 = nn.ConvTranspose2d(s * 4, s * 2, 2, 2)
        self.ex_double_conv2 = DoubleConv(s * 4, s * 2)
        self.t_conv1 = nn.ConvTranspose2d(s * 2, s, 2, 2)
        self.ex_double_conv1 = DoubleConv(s * 2, s)
        self.ex_double_conv0 = DoubleConv(s, 16)
        self.dropout = nn.Dropout(0.20)
        self.final = nn.Sequential(
            nn.Conv2d(16, 16, 1, padding=0, bias=True),
            nn.BatchNorm2d(16),
            nn.Softplus(),
        )

    def forward(self, inputs):
        conv1 = self.double_conv1(inputs)
        pool1 = self.dropout(self.maxpool1(conv1))
        conv2 = self.double_conv2(pool1)
        pool2 = self.dropout(self.maxpool2(conv2))
        conv3 = self.double_conv3(pool2)
        pool3 = self.dropout(self.maxpool3(conv3))
        conv4 = self.double_conv4(pool3)
        pool4 = self.dropout(self.maxpool4(conv4))
        conv5 = self.double_conv5(pool4)
        up4 = self.dropout(self.t_conv4(conv5))
        ex4 = self.ex_double_conv4(torch.cat([conv4, up4], 1))
        up3 = self.dropout(self.t_conv3(ex4))
        ex3 = self.ex_double_conv3(torch.cat([conv3, up3], 1))
        up2 = self.dropout(self.t_conv2(ex3))
        ex2 = self.ex_double_conv2(torch.cat([conv2, up2], 1))
        up1 = self.dropout(self.t_conv1(ex2))
        ex1 = self.ex_double_conv1(torch.cat([conv1, up1], 1))
        return self.final(self.ex_double_conv0(ex1))


def make_unet():
    return Unet()


def tv_loss(prediction):
    vertical = torch.mean(torch.abs(prediction[:, :, 1:, :] - prediction[:, :, :-1, :]))
    horizontal = torch.mean(torch.abs(prediction[:, :, :, 1:] - prediction[:, :, :, :-1]))
    return vertical + horizontal


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
    parser.add_argument("--lambda-tv", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--experiment-name", default="formal_20k")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / "data" / "tra&val")
    parser.add_argument("--resume", action="store_true")
    return parser


def experiment_paths(root, experiment_name):
    root = Path(root)
    experiment_dir = root / "experiments" / experiment_name
    return {
        "experiment_dir": experiment_dir,
        "weights": root / f"parameters_GavI_{experiment_name}.pth",
        "history": root / f"history_{experiment_name}.npz",
        "split": experiment_dir / "split_indices.npz",
        "metadata": experiment_dir / "metadata.json",
        "last_checkpoint": experiment_dir / "last_checkpoint.pth",
        "best_checkpoint": experiment_dir / "best_checkpoint.pth",
    }


def set_reproducible(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def load_base_arrays(data_dir, count):
    images = np.empty((count, 1, GRID_SIZE, GRID_SIZE), dtype=np.float32)
    masks = np.empty((count, 16, GRID_SIZE, GRID_SIZE), dtype=np.float32)
    started = time.time()
    for index in range(count):
        path = data_dir / f"data{index}.mat"
        if not path.exists():
            raise FileNotFoundError(f"Missing training sample: {path}")
        sample = scipy.io.loadmat(path)
        images[index] = np.nan_to_num(sample["d"].squeeze()).reshape(1, GRID_SIZE, GRID_SIZE)
        masks[index] = (sample["m"].squeeze() / DENSITY_SCALE).reshape(16, GRID_SIZE, GRID_SIZE)
        if (index + 1) % 1000 == 0 or index + 1 == count:
            elapsed = time.time() - started
            print(f"Loaded {index + 1}/{count} base samples ({elapsed:.1f}s)", flush=True)
    return images, masks


def split_digest(train_indices, val_indices):
    digest = hashlib.sha256()
    digest.update(np.asarray(train_indices, dtype=np.int64).tobytes())
    digest.update(np.asarray(val_indices, dtype=np.int64).tobytes())
    return digest.hexdigest()


def save_history(path, train_losses, val_losses, epoch_times, config):
    np.savez(
        path,
        tra=np.asarray(train_losses, dtype=np.float64),
        val=np.asarray(val_losses, dtype=np.float64),
        epoch_time=np.asarray(epoch_times, dtype=np.float64),
        config=json.dumps(config, sort_keys=True),
    )


def save_metadata(path, config, status, best_epoch, best_val, elapsed):
    payload = {
        "status": status,
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "elapsed_seconds": elapsed,
        "config": config,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def mean_epoch_loss(model, loader, criterion, lambda_tv, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0
    sample_count = 0
    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            outputs = model(images)
            loss = criterion(outputs, masks) + lambda_tv * tv_loss(outputs)
            if training:
                loss.backward()
                optimizer.step()
        batch_count = images.shape[0]
        loss_sum += float(loss.detach()) * batch_count
        sample_count += batch_count
    return loss_sum / sample_count


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
    np.savez(paths["split"], train=train_indices, val=val_indices)

    config = {
        "experiment_name": args.experiment_name,
        "total_count": args.total_count,
        "train_count": args.train_count,
        "val_count": args.val_count,
        "training_views": args.train_count * 3,
        "noise_levels": [0.0, 0.02, 0.04],
        "seed": args.seed,
        "split_sha256": split_digest(train_indices, val_indices),
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "min_delta": args.min_delta,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "lambda_tv": args.lambda_tv,
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
    criterion = nn.SmoothL1Loss(beta=0.1)
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
            model, train_loader, criterion, args.lambda_tv, device, optimizer=optimizer
        )
        with torch.no_grad():
            val_loss = mean_epoch_loss(model, val_loader, criterion, args.lambda_tv, device)
        epoch_time = time.time() - epoch_started
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        epoch_times.append(epoch_time)

        improved = val_loss < best_val - args.min_delta
        if improved:
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
