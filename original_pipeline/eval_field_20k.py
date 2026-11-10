"""Evaluate four inversion methods on the real field gravity survey."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from field_evaluation import (
    assemble_windows,
    extract_windows,
    field_window_starts,
    forward_window,
    load_pcg,
    to_physical_density,
)
from train_dice_cont_20k import make_unet as make_sigmoid_unet
from train_formal_20k import make_unet as make_softplus_unet


ROOT = Path(__file__).resolve().parent
METHOD_CONFIGS = [
    {
        "key": "softplus",
        "label": "Softplus+TV 20k zero-shot",
        "weights": "parameters_GavI_formal_20k.pth",
        "factory": make_softplus_unet,
        "scale": 1000.0,
    },
    {
        "key": "dice_continuous",
        "label": "Dice+Sigmoid 20k continuous zero-shot",
        "weights": "parameters_GavI_dice_cont_20k.pth",
        "factory": make_sigmoid_unet,
        "scale": 1000.0,
    },
    {
        "key": "field_specific",
        "label": "Original field Dice+Sigmoid",
        "weights": "Field_data/parameters.pth",
        "factory": make_sigmoid_unet,
        "scale": 1100.0,
    },
    {
        "key": "pcg_primary",
        "label": "PCG",
        "weights": "Field_data/prediction(PCG).dat",
        "factory": None,
        "scale": 1000.0,
    },
]


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--output-dir", default="field_results")
    return parser


def load_field_windows():
    windows = np.empty((60, 1, 32, 32), dtype=np.float32)
    for index in range(60):
        path = ROOT / "Field_data" / "real" / f"data_slice{index}.mat"
        with h5py.File(path, "r") as file:
            windows[index, 0] = np.asarray(file["d_slice"][0]).reshape(32, 32)
    return windows


def load_full_observed_gravity():
    with h5py.File(ROOT / "Field_data" / "real" / "data.mat", "r") as file:
        return np.asarray(file["d"][0]).reshape(43, 36).T


def load_forward_kernel():
    with h5py.File(ROOT / "G.mat", "r") as file:
        return np.nan_to_num(file["G"][:]).T


def infer_neural(config, windows, batch_size, device):
    model = config["factory"]().to(device)
    state = torch.load(ROOT / config["weights"], map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            batch = torch.from_numpy(windows[start:start + batch_size]).to(device)
            predictions.append(model(batch).cpu().numpy())
    return np.concatenate(predictions)


def error_metrics(predicted, observed):
    predicted = np.asarray(predicted, dtype=np.float64)
    observed = np.asarray(observed, dtype=np.float64)
    difference = predicted - observed
    denominator = np.linalg.norm(observed.reshape(-1))
    return {
        "rmse": float(np.sqrt(np.mean(difference ** 2))),
        "relative_error": float(np.linalg.norm(difference.reshape(-1)) / denominator),
    }


def summarize_rows(rows):
    return {
        key: {
            "mean": float(np.mean([row[key] for row in rows])),
            "std": float(np.std([row[key] for row in rows])),
        }
        for key in rows[0]
    }


def forward_all_windows(kernel, density_windows, observed_windows):
    predictions = np.stack([forward_window(kernel, density) for density in density_windows])
    rows = [
        error_metrics(predicted, observed)
        for predicted, observed in zip(predictions, observed_windows)
    ]
    return predictions, rows, summarize_rows(rows)


def structural_metrics(model):
    model = np.asarray(model, dtype=np.float64)
    positive = np.maximum(model, 0.0)
    total = float(positive.sum())
    coordinates = np.indices(model.shape, dtype=np.float64)
    if total:
        center = [float((coordinates[axis] * positive).sum() / total) for axis in range(3)]
        depth_profile = positive.sum(axis=(1, 2))
        normalized_profile = depth_profile / depth_profile.sum()
    else:
        center = [float("nan")] * 3
        depth_profile = np.zeros(model.shape[0])
        normalized_profile = depth_profile.copy()
    return {
        "min_kg_m3": float(model.min()),
        "max_kg_m3": float(model.max()),
        "total_positive_density": total,
        "active_fraction_100": float((model >= 100.0).mean()),
        "active_fraction_300": float((model >= 300.0).mean()),
        "active_fraction_500": float((model >= 500.0).mean()),
        "total_variation_mean": float(
            np.mean(np.abs(np.diff(model, axis=0)))
            + np.mean(np.abs(np.diff(model, axis=1)))
            + np.mean(np.abs(np.diff(model, axis=2)))
        ),
        "positive_center_of_mass_zyx": center,
        "depth_positive_density": depth_profile.tolist(),
        "depth_positive_density_normalized": normalized_profile.tolist(),
    }


def pcg_agreement(model, pcg):
    left = np.asarray(model, dtype=np.float64).reshape(-1)
    right = np.asarray(pcg, dtype=np.float64).reshape(-1)
    left_scale = np.linalg.norm(left)
    right_scale = np.linalg.norm(right)
    correlation = float(np.corrcoef(left, right)[0, 1])
    normalized_rmse = float(
        np.sqrt(np.mean(((left / left_scale) - (right / right_scale)) ** 2))
    )
    return {"correlation": correlation, "normalized_rmse": normalized_rmse}


def overlap_metrics(disagreement):
    values = np.asarray(disagreement, dtype=np.float64)
    return {
        "mean_std_kg_m3": float(values.mean()),
        "median_std_kg_m3": float(np.median(values)),
        "max_std_kg_m3": float(values.max()),
    }


def build_archive_arrays(
    observed_full,
    observed_windows,
    overlap_counts,
    pcg_raw_g_cm3,
    pcg_forward_windows,
    assembled_models,
    overlap_stds,
    direct_predictions,
):
    return {
        "observed_full": observed_full,
        "observed_windows": observed_windows,
        "overlap_counts": overlap_counts,
        "pcg_raw_g_cm3": pcg_raw_g_cm3,
        "pcg_forward_windows": pcg_forward_windows,
        **assembled_models,
        **{f"{key}_overlap_std": value for key, value in overlap_stds.items()},
        **{f"{key}_direct_forward": value for key, value in direct_predictions.items()},
    }


def main():
    args = build_parser().parse_args()
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    starts = field_window_starts()
    observed_windows = load_field_windows()[:, 0]
    observed_full = load_full_observed_gravity()
    kernel = load_forward_kernel()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    assembled_models = {}
    overlap_stds = {}
    direct_predictions = {}
    results = {}

    for config in METHOD_CONFIGS[:3]:
        print(f"Evaluating {config['label']}...", flush=True)
        normalized_windows = infer_neural(config, observed_windows[:, None], args.batch_size, device)
        normalized_model, counts, normalized_std = assemble_windows(normalized_windows, starts)
        physical_windows = to_physical_density(normalized_windows, config["scale"])
        physical_model = to_physical_density(normalized_model, config["scale"])
        physical_std = to_physical_density(normalized_std, config["scale"])
        assembled_window_models = extract_windows(physical_model, starts)

        direct_gravity, direct_rows, direct_summary = forward_all_windows(
            kernel, physical_windows, observed_windows
        )
        assembled_gravity, assembled_rows, assembled_summary = forward_all_windows(
            kernel, assembled_window_models, observed_windows
        )
        key = config["key"]
        assembled_models[key] = physical_model
        overlap_stds[key] = physical_std
        direct_predictions[key] = direct_gravity
        results[key] = {
            "label": config["label"],
            "weights": config["weights"],
            "physical_scale_kg_m3": config["scale"],
            "forward_direct": direct_summary,
            "forward_assembled": assembled_summary,
            "forward_direct_per_window": direct_rows,
            "forward_assembled_per_window": assembled_rows,
            "overlap": overlap_metrics(physical_std),
            "structure": structural_metrics(physical_model),
        }

    pcg_raw_g_cm3, pcg_primary_g_cm3 = load_pcg(ROOT / METHOD_CONFIGS[3]["weights"])
    pcg_primary = to_physical_density(pcg_primary_g_cm3, METHOD_CONFIGS[3]["scale"])
    pcg_windows = extract_windows(pcg_primary, starts)
    pcg_gravity, pcg_rows, pcg_summary = forward_all_windows(
        kernel, pcg_windows, observed_windows
    )
    assembled_models["pcg_primary"] = pcg_primary
    results["pcg_primary"] = {
        "label": METHOD_CONFIGS[3]["label"],
        "weights": METHOD_CONFIGS[3]["weights"],
        "physical_scale_kg_m3": METHOD_CONFIGS[3]["scale"],
        "forward_assembled": pcg_summary,
        "forward_assembled_per_window": pcg_rows,
        "structure": structural_metrics(pcg_primary),
        "raw_structure_g_cm3": structural_metrics(pcg_raw_g_cm3),
    }

    for key, model in assembled_models.items():
        results[key]["pcg_agreement"] = pcg_agreement(model, pcg_primary)

    payload = {
        "method_order": [config["key"] for config in METHOD_CONFIGS],
        "method_labels": {config["key"]: config["label"] for config in METHOD_CONFIGS},
        "window_count": len(starts),
        "window_starts_yx": starts,
        "input_distribution": {
            "min": float(observed_windows.min()),
            "max": float(observed_windows.max()),
            "mean": float(observed_windows.mean()),
            "std": float(observed_windows.std()),
            "negative_fraction": float((observed_windows < 0).mean()),
        },
        "results": results,
    }
    json_path = output_dir / "field_comparison.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    archive_arrays = build_archive_arrays(
        observed_full,
        observed_windows,
        counts,
        pcg_raw_g_cm3,
        pcg_gravity,
        assembled_models,
        overlap_stds,
        direct_predictions,
    )
    np.savez_compressed(output_dir / "assembled_models.npz", **archive_arrays)
    print(f"Saved: {json_path}", flush=True)


if __name__ == "__main__":
    main()
