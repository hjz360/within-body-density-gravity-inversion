"""Evaluate frozen C0/C1 models against the San Nicolas raw-station chain.

This is an illustrative field-data fit check, not a voxel-level ground-truth
test.  It uses the already validated 190-station forward operators and the
same Kriging windows used by the legacy field-inference workflow.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch

from eval_field_20k import load_field_windows
from field_evaluation import assemble_windows, field_window_starts
from field_observation_chain_audit import sha256_file
from field_observation_chain_stage_a import load_frozen_kriging_artifact
from run_field_observation_chain_stage_a import write_artifact_manifest
import train_c0c1_paired_v2 as training
from train_formal_20k import make_unet


ROOT = Path(__file__).resolve().parent
NAMESPACE = "c0c1_paired_v2_r2"
STAGE_A = ROOT / "paper_results" / "field_observation_chain_stage_a"
OUTPUT = ROOT / "paper_results" / "c0c1_paired_v2_field_raw_station"


def station_forward_prediction(density_kg_m3: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    density = np.asarray(density_kg_m3, dtype=np.float64)
    operator = np.asarray(kernel, dtype=np.float64)
    if operator.ndim != 2 or operator.shape[1] != density.size:
        raise ValueError("raw-station forward kernel and field density are incompatible")
    return operator @ density.reshape(-1, order="C")


def covariance_whitened_rms(residual_mgal: np.ndarray, correlation: np.ndarray, *, sigma_mgal: float) -> float:
    residual = np.asarray(residual_mgal, dtype=np.float64).reshape(-1)
    corr = np.asarray(correlation, dtype=np.float64)
    if corr.shape != (residual.size, residual.size):
        raise ValueError("correlation shape does not match residual count")
    if not np.isfinite(residual).all() or not np.isfinite(corr).all() or sigma_mgal <= 0.0:
        raise ValueError("whitening requires finite residuals/correlation and positive sigma")
    covariance = float(sigma_mgal) ** 2 * corr
    covariance = covariance + np.eye(residual.size) * (float(sigma_mgal) ** 2 * 1e-8)
    lower = np.linalg.cholesky(covariance)
    whitened = np.linalg.solve(lower, residual)
    return float(np.sqrt(np.mean(whitened ** 2)))


def station_correlation_from_frozen_scaling(
    station_xy: np.ndarray,
    *,
    theta: np.ndarray,
    location: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    xy = np.asarray(station_xy, dtype=np.float64)
    parameters = np.asarray(theta, dtype=np.float64).reshape(-1)
    offset = np.asarray(location, dtype=np.float64).reshape(-1)
    divisor = np.asarray(scale, dtype=np.float64).reshape(-1)
    if xy.ndim != 2 or xy.shape[1] != 2 or parameters.shape != (2,) or offset.shape != (2,) or divisor.shape != (2,) or np.any(divisor <= 0.0):
        raise ValueError("station correlation requires valid frozen two-dimensional scaling")
    scaled = (xy - offset) / divisor
    delta = scaled[:, None, :] - scaled[None, :, :]
    correlation = np.exp(-np.sum(parameters[None, None, :] * delta ** 2, axis=2))
    return (correlation + correlation.T) / 2.0


def _load_theta_and_scaling(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        theta = np.asarray(handle["dmodel/theta"][:], dtype=np.float64).reshape(-1)
        scaling = np.asarray(handle["dmodel/Ssc"][:], dtype=np.float64).T
    if theta.shape != (2,) or scaling.shape != (2, 2) or not np.isfinite(theta).all() or not np.isfinite(scaling).all() or np.any(scaling[1] <= 0.0):
        raise ValueError("frozen Kriging theta/scaling is unavailable")
    return theta, scaling[0], scaling[1]


def _load_operators(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as handle:
        return {
            branch: np.asarray(handle[f"{branch}/kernel"][:], dtype=np.float64)
            for branch in ("actual_xy_flat_legacy", "actual_xyz_candidate_elevation")
        }


def _infer_full_model(weight_path: Path, windows: np.ndarray, device: torch.device) -> np.ndarray:
    model = make_unet().to(device)
    model.load_state_dict(torch.load(weight_path, map_location="cpu", weights_only=True), strict=True)
    model.eval()
    blocks = []
    with torch.no_grad():
        for start in range(0, len(windows), 12):
            batch = torch.as_tensor(windows[start:start + 12], dtype=torch.float32, device=device)
            blocks.append(model(batch).detach().cpu().numpy())
    normalized_windows = np.concatenate(blocks, axis=0)
    normalized_full, _, _ = assemble_windows(normalized_windows, field_window_starts())
    return np.asarray(normalized_full, dtype=np.float64) * 1000.0


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _figure(station_xy: np.ndarray, observed: np.ndarray, rows: list[dict], predictions: dict[str, np.ndarray], output: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 8.2), constrained_layout=True)
    for branch, marker in (("actual_xy_flat_legacy", "o"), ("actual_xyz_candidate_elevation", "s")):
        for arm, color in (("C0", "#4C78A8"), ("C1", "#F58518")):
            values = [row for row in rows if row["branch"] == branch and row["arm"] == arm]
            axes[0, 0].plot([row["seed"] for row in values], [row["rmse_mgal"] for row in values], marker=marker, color=color, label=f"{arm}, {branch}")
            axes[0, 1].plot([row["seed"] for row in values], [row["gp_nominal_whitened_rms"] for row in values], marker=marker, color=color, label=f"{arm}, {branch}")
    axes[0, 0].set_title("Raw-station residual RMSE")
    axes[0, 0].set_ylabel("mGal")
    axes[0, 1].set_title("Frozen-GP covariance-whitened residual")
    axes[0, 1].set_ylabel("RMS in nominal covariance units")
    for axis in axes[0]:
        axis.set_xlabel("paired training seed")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7, frameon=False)
    for column, arm in enumerate(("C0", "C1")):
        field_prediction = np.mean([predictions[f"{arm}_{seed}_actual_xy_flat_legacy"] for seed in (20260727, 20260728, 20260729)], axis=0)
        residual = field_prediction - observed
        scatter = axes[1, column].scatter(station_xy[:, 0], station_xy[:, 1], c=residual, s=26, cmap="coolwarm", linewidths=0)
        axes[1, column].set_title(f"{arm}: mean raw-station residual across seeds")
        axes[1, column].set_xlabel("Easting (m)")
        axes[1, column].set_ylabel("Northing (m)")
        figure.colorbar(scatter, ax=axes[1, column], label="predicted - observed (mGal)")
    figure.savefig(output / "Fig_c0c1_field_raw_station_residuals.png", dpi=240)
    figure.savefig(output / "Fig_c0c1_field_raw_station_residuals.pdf")
    plt.close(figure)


def run(root: Path = ROOT, output: Path = OUTPUT) -> dict:
    root = Path(root).resolve()
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    artifact = load_frozen_kriging_artifact(STAGE_A / "kriging" / "frozen_field_kriging.mat")
    observed = np.asarray(artifact["station_gravity"], dtype=np.float64)
    station_xy = np.asarray(artifact["station_xy"], dtype=np.float64)
    theta, scaling_location, scaling_scale = _load_theta_and_scaling(STAGE_A / "kriging" / "frozen_field_kriging.mat")
    correlation = station_correlation_from_frozen_scaling(station_xy, theta=theta, location=scaling_location, scale=scaling_scale)
    operators = _load_operators(STAGE_A / "operators" / "field_forward_operators.h5")
    envelope = json.loads((STAGE_A / "noise" / "noise_and_preprocessing_error_envelope.json").read_text(encoding="utf-8"))
    sigma = float(envelope["white_noise_sigma_mgal"]["nominal"])
    windows = load_field_windows()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows: list[dict] = []
    predictions: dict[str, np.ndarray] = {}
    for arm in ("C0", "C1"):
        for seed in (20260727, 20260728, 20260729):
            paths = training.run_paths(root, arm, seed, epochs=120)
            metadata = json.loads(paths.metadata.read_text(encoding="utf-8"))
            if metadata.get("status") != "complete" or int(metadata.get("completed_epochs", -1)) != 120:
                raise RuntimeError(f"incomplete weight: {paths.run_dir}")
            density = _infer_full_model(paths.primary_weights, windows, device)
            for branch, kernel in operators.items():
                predicted = station_forward_prediction(density, kernel)
                residual = predicted - observed
                predictions[f"{arm}_{seed}_{branch}"] = predicted
                rows.append({
                    "arm": arm,
                    "seed": seed,
                    "branch": branch,
                    "rmse_mgal": float(np.sqrt(np.mean(residual ** 2))),
                    "relative_l2": float(np.linalg.norm(residual) / max(np.linalg.norm(observed), np.finfo(np.float64).eps)),
                    "white_nominal_rms": float(np.sqrt(np.mean((residual / sigma) ** 2))),
                    "gp_nominal_whitened_rms": covariance_whitened_rms(residual, correlation, sigma_mgal=sigma),
                    "weight_sha256": sha256_file(paths.primary_weights),
                })
    np.savez_compressed(output / "raw_station_predictions_and_residuals.npz", observed=observed, station_xy=station_xy, **predictions)
    _figure(station_xy, observed, rows, predictions, output)
    summary = {
        "schema_version": 1,
        "interpretation": "field fit and sensitivity case only; not voxel-level geological ground truth",
        "station_count": int(observed.size),
        "field_input": "legacy frozen global Kriging windows",
        "covariance_model": "frozen DACE Gaussian correlation with nominal empirical residual scale; an empirical whitening convention, not known field-noise truth",
        "frozen_dace_theta": theta.tolist(),
        "frozen_dace_coordinate_location": scaling_location.tolist(),
        "frozen_dace_coordinate_scale": scaling_scale.tolist(),
        "nominal_sigma_mgal": sigma,
        "forward_branches": {
            "actual_xy_flat_legacy": "primary legacy-coordinate branch",
            "actual_xyz_candidate_elevation": "sensitivity branch because elevation-column semantics remain provisional",
        },
        "rows": rows,
    }
    _write_json(output / "field_raw_station_summary.json", summary)
    (output / "现场原始测站案例说明_CN.md").write_text(
        "# C0/C1 现场原始测站回算案例\n\n"
        "本目录将最终 C0/C1 权重作用于既有 Kriging 窗口，再用 190 个原始测站正演算子回算。它检验现场数据拟合与坐标/高程分支敏感性，不能作为地下逐体素真值恢复证据。协方差白化使用冻结 DACE 相关结构和经验残差尺度，属于透明的经验约定，而不是已知现场噪声真值。\n",
        encoding="utf-8",
    )
    write_artifact_manifest(output)
    return summary


if __name__ == "__main__":
    run()
