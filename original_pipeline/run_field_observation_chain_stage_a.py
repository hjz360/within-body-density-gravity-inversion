"""Freeze the resolved Stage-A San Nicolas observation-chain components.

This runner performs no inversion training and no counterfactual
identifiability inference.  It validates already generated Kriging and
raw-station forward components, freezes a transparent
measurement-plus-preprocessing error envelope, and writes branch-specific
readiness.
"""

from __future__ import annotations

import argparse
import csv
from datetime import date
import json
from pathlib import Path
import shutil
from typing import Any

import h5py
import numpy as np

from field_observation_chain_audit import sha256_file
from field_observation_chain_stage_a import (
    build_error_envelope,
    frozen_theta_cv_residuals,
    gridded_residual_spectrum,
    load_frozen_kriging_artifact,
    ordinary_kriging_matrix,
    residual_spatial_diagnostics,
    resolve_stage_a_readiness,
    validate_operator_elements,
)


DEFAULT_ROOT = Path(r"F:\2026.3\代码")
DEFAULT_OUTPUT = (
    DEFAULT_ROOT / "paper_results" / "field_observation_chain_stage_a"
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def prepare_formal_output(path: str | Path) -> Path:
    """Create an empty formal output directory, refusing prior content."""

    output = Path(path)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"formal output directory is nonempty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def write_artifact_manifest(path: str | Path) -> dict[str, Any]:
    """Hash every prior formal artifact and write the manifest last."""

    output = Path(path)
    manifest_path = output / "artifact_manifest.json"
    artifacts = []
    for artifact in sorted(
        (
            candidate
            for candidate in output.rglob("*")
            if candidate.is_file() and candidate != manifest_path
        ),
        key=lambda candidate: candidate.relative_to(output).as_posix(),
    ):
        artifacts.append(
            {
                "relative_path": artifact.relative_to(output).as_posix(),
                "size_bytes": int(artifact.stat().st_size),
                "sha256": sha256_file(artifact),
            }
        )
    manifest = {
        "schema_version": 1,
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "manifest_written_last": True,
    }
    _write_json(manifest_path, manifest)
    return manifest


def _load_theta(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        theta = np.asarray(handle["dmodel/theta"][:], dtype=np.float64).reshape(-1)
    if theta.shape != (2,) or not np.isfinite(theta).all():
        raise ValueError("Frozen DACE artifact has invalid theta")
    return theta


def _kriging_summary(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    artifact = load_frozen_kriging_artifact(path)
    theta = _load_theta(path)
    python_matrix = ordinary_kriging_matrix(
        artifact["station_xy"],
        artifact["prediction_xy"],
        theta,
    )
    matrix_error = float(np.max(np.abs(python_matrix - artifact["K"])))
    validation = {
        **artifact["validation"],
        "python_independent_K_match": bool(matrix_error <= 1e-10),
        "max_abs_python_vs_matlab_K": matrix_error,
        "python_vs_matlab_K_atol": 1e-10,
    }
    summary = {
        "schema_version": 1,
        "component": "frozen_global_DACE_ordinary_kriging",
        "station_count": int(artifact["station_xy"].shape[0]),
        "prediction_count": int(artifact["prediction_xy"].shape[0]),
        "grid_shape_yx": [36, 43],
        "fit_scope": "global_all_stations",
        "pointdata_columns_used": [1, 2, 4],
        "pointdata_field_3_used": False,
        "regression": "regpoly0",
        "correlation": "corrgauss",
        "theta": theta.tolist(),
        "validation": validation,
        "source_artifact_sha256": sha256_file(path),
    }
    return summary, artifact


def _operator_summary(path: Path) -> dict[str, Any]:
    expected_branches = (
        "actual_xy_flat_legacy",
        "actual_xyz_candidate_elevation",
    )
    pairs = [
        (0, 5 * 36 * 43 + 5 * 43 + 5),
        (95, 10 * 36 * 43 + 18 * 43 + 21),
        (189, 15 * 36 * 43 + 30 * 43 + 37),
    ]
    reports: dict[str, Any] = {}
    with h5py.File(path, "r") as handle:
        if handle.attrs.get("logical_orientation") != "observation_by_cell":
            raise ValueError("Field operator has invalid logical orientation")
        bounds = np.asarray(handle["cell_bounds"][:], dtype=np.float64)
        if bounds.shape != (24768, 6):
            raise ValueError(f"Unexpected full field bounds shape: {bounds.shape}")
        for branch in expected_branches:
            if branch not in handle:
                raise ValueError(f"Field operator is missing branch: {branch}")
            kernel = np.asarray(handle[f"{branch}/kernel"][:], dtype=np.float64)
            observations = np.asarray(
                handle[f"{branch}/observation_xyz"][:],
                dtype=np.float64,
            )
            if kernel.shape != (190, 24768):
                raise ValueError(f"Unexpected {branch} kernel shape: {kernel.shape}")
            report = validate_operator_elements(
                kernel,
                observations,
                bounds,
                pairs=pairs,
                quadrature_order=28,
                relative_tolerance=1e-9,
            )
            report["logical_shape"] = [190, 24768]
            report["semantic_status"] = handle[branch].attrs[
                "semantic_status"
            ]
            reports[branch] = report
    return {
        "schema_version": 1,
        "component": "full_field_raw_station_forward_operators",
        "kernel_units": "mGal_per_kg_m3",
        "coordinate_units": "m",
        "z_positive": "down",
        "cell_flattening": "C-order z_y_x; x fastest",
        "source_artifact_sha256": sha256_file(path),
        "branches": reports,
        "all_branches_passed": all(
            report["passed"] for report in reports.values()
        ),
        "independent_check": (
            "closed-form prism elements compared with Gauss-Legendre volume "
            "quadrature"
        ),
    }


def _write_residual_csv(
    path: Path,
    artifact: dict[str, Any],
    loo: dict[str, Any],
    blocked: dict[str, Any],
) -> None:
    fields = [
        "station_id",
        "x_m",
        "y_m",
        "gravity_mgal",
        "loo_prediction_mgal",
        "loo_residual_mgal",
        "block_fold_id",
        "block_prediction_mgal",
        "block_residual_mgal",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for station_id in range(artifact["station_xy"].shape[0]):
            writer.writerow(
                {
                    "station_id": station_id,
                    "x_m": artifact["station_xy"][station_id, 0],
                    "y_m": artifact["station_xy"][station_id, 1],
                    "gravity_mgal": artifact["station_gravity"][station_id],
                    "loo_prediction_mgal": loo["prediction"][station_id],
                    "loo_residual_mgal": loo["residual"][station_id],
                    "block_fold_id": blocked["fold_id"][station_id],
                    "block_prediction_mgal": blocked["prediction"][station_id],
                    "block_residual_mgal": blocked["residual"][station_id],
                }
            )


def _write_diagnostic_figure(
    path: Path,
    artifact: dict[str, Any],
    loo: dict[str, Any],
    blocked: dict[str, Any],
    loo_spatial: dict[str, Any],
    block_spatial: dict[str, Any],
    spectrum: dict[str, Any],
) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(10.5, 8.0), constrained_layout=True)
    axes[0, 0].hist(
        loo["residual"],
        bins=24,
        alpha=0.75,
        label="Frozen-theta LOO",
    )
    axes[0, 0].hist(
        blocked["residual"],
        bins=24,
        alpha=0.55,
        label="Y-block CV",
    )
    axes[0, 0].axvline(0.0, color="black", linewidth=0.8)
    axes[0, 0].set_xlabel("Residual (mGal)")
    axes[0, 0].set_ylabel("Station count")
    axes[0, 0].set_title("(a) Cross-validation residuals")
    axes[0, 0].legend(frameon=False)

    scatter = axes[0, 1].scatter(
        artifact["station_xy"][:, 0],
        artifact["station_xy"][:, 1],
        c=loo["residual"],
        cmap="coolwarm",
        s=24,
        edgecolors="none",
    )
    figure.colorbar(scatter, ax=axes[0, 1], label="LOO residual (mGal)")
    axes[0, 1].set_xlabel("Easting (m)")
    axes[0, 1].set_ylabel("Northing (m)")
    axes[0, 1].set_title("(b) Residual spatial pattern")
    axes[0, 1].set_aspect("equal", adjustable="box")

    for diagnostic, label, marker in (
        (loo_spatial, "LOO", "o"),
        (block_spatial, "Y-block CV", "s"),
    ):
        valid = [
            item
            for item in diagnostic["semivariogram"]
            if item["pair_count"] > 0
        ]
        axes[1, 0].plot(
            [item["mean_distance_m"] for item in valid],
            [item["semivariance_mgal2"] for item in valid],
            marker=marker,
            label=label,
        )
    axes[1, 0].set_xlabel("Pair distance (m)")
    axes[1, 0].set_ylabel("Semivariance (mGal²)")
    axes[1, 0].set_title("(c) Empirical semivariogram")
    axes[1, 0].legend(frameon=False)

    names = ["low", "mid", "high"]
    axes[1, 1].bar(
        names,
        [spectrum["power_fraction"][name] for name in names],
        color=["#3B82F6", "#10B981", "#F59E0B"],
    )
    axes[1, 1].set_ylim(0.0, 1.0)
    axes[1, 1].set_ylabel("Power fraction")
    axes[1, 1].set_title("(d) Frozen-K residual spectrum")
    figure.suptitle(
        "San Nicolas measurement and preprocessing error diagnostics",
        fontsize=13,
    )
    figure.savefig(path, dpi=220)
    plt.close(figure)


def _source_evidence() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "field_3_semantics": {
            "status": "strong_indirect_support_not_file_specific_confirmation",
            "project_lineage": (
                "The local project states that raw field data were supplied "
                "by UBC-GIF."
            ),
            "UBC_gravity_format": {
                "columns": ["x", "y", "elevation", "datum", "optional_error"],
                "url": (
                    "https://www.eoas.ubc.ca/courses/eosc350/content/"
                    "sftwrdocs/gm-data-viewer/gm-data-viewer_docs.htm"
                ),
            },
            "current_file_has_four_columns": True,
            "authoritative_file_specific_dictionary_found": False,
        },
        "survey_error": {
            "published_measurement_floor_mgal": 0.05,
            "scope": "original San Nicolas gravity survey",
            "url": "https://gif.eos.ubc.ca/sites/default/files/sannic-tle.pdf",
            "current_190_point_subset_has_error_column": False,
        },
        "Huang_2022": {
            "paper_url": (
                "https://agupubs.onlinelibrary.wiley.com/doi/full/"
                "10.1029/2021JB022476"
            ),
            "data_code_url": "https://zenodo.org/records/5211231",
            "reported_measurement_points": 198,
            "current_local_point_count": 190,
        },
        "Dong_2026": {
            "paper_url": "https://arxiv.org/abs/2601.02890",
            "data_code_url": "https://zenodo.org/records/17617208",
            "relevance": (
                "independent San Nicolas deep-learning case using regular "
                "gridded field input; does not resolve the current raw-file "
                "field-3 semantics"
            ),
        },
    }


def _readiness_markdown(payload: dict[str, Any]) -> str:
    readiness = payload["readiness"]
    kriging = payload["kriging"]
    operators = payload["field_forward_operators"]
    envelope = payload["error_envelope"]
    open_codes = "\n".join(
        f"- `{code}`" for code in readiness["open_blocker_codes"]
    ) or "- 无"
    resolved_codes = "\n".join(
        f"- `{code}`" for code in readiness["resolved_blocker_codes"]
    ) or "- 无"
    candidate = readiness["branches"]["actual_xyz_candidate_elevation"]
    flat = readiness["branches"]["actual_xy_flat_legacy"]
    scales = envelope["white_noise_sigma_mgal"]
    return f"""# Stage A 现场观测链阻断项解决报告

日期：{date.today().isoformat()}  
结论：**组件已可复现；真实高程语义仍未获文件级权威确认。**

## 已取得的实质进展

- 全局 DACE/Kriging 模型已冻结；实测拟合参数
  `theta={kriging["theta"]}`。
- 固定矩阵 `K` 的逻辑维度为 `1548 × 190`；`K·Y` 与 DACE predictor
  的最大绝对差为
  `{kriging["validation"]["max_abs_prediction_error"]:.3e}` mGal。
- Python 独立重建 K 与 MATLAB K 的最大绝对差为
  `{kriging["validation"]["max_abs_python_vs_matlab_K"]:.3e}`。
- 已构建两支完整场地算子，逻辑维度均为 `190 × 24768`：
  `actual_xy_flat_legacy` 与
  `actual_xyz_candidate_elevation`。
- 独立 Gauss--Legendre 体积分核对全部通过；最坏相对误差为
  `{max(report["max_relative_error"] for report in operators["branches"].values()):.3e}`。
- 冻结白噪声低/基准/高尺度为
  `{scales["low"]:.6f} / {scales["nominal"]:.6f} /
  {scales["high"]:.6f}` mGal。

## 误差尺度的含义

- 0.05 mGal 是原始 San Nicolas 调查文献给出的测量误差尺度，不是当前
  190 点文件中的逐点误差列。
- 基准尺度来自冻结 theta 的逐点留一 Kriging 残差。
- 高尺度来自按 northing 连续空间块留出的 Kriging 残差。
- 该包只能称“测量与预处理误差包络”，不能称已知现场噪声真值。

## 分支授权

- 平面旧流程复现：`authorized={flat["authorized"]}`，
  类别 `{flat["authorization_class"]}`。
- 候选高程分支：`authorized={candidate["authorized"]}`，
  类别 `{candidate["authorization_class"]}`。
- 完全真实场地几何 Stage A 就绪：
  `{readiness["fully_field_realistic_stage_a_ready"]}`。
- 网络训练授权：`False`。

候选高程分支只能作为条件性敏感性分析。不得在字段 3 获得文件级数据字典或作者
确认前，将其写成“已确认实际测点高程”。

## 已解决的原审计阻断项

{resolved_codes}

## 仍开放的阻断项

{open_codes}

## 下一步

允许在两个明确标注的观测分支上运行 Stage A 模式级可辨识性分析；主统计仍在
190 个原始测点空间完成，Kriging 网格只使用传播协方差或等价低秩表示。不得启动
网络训练或 C×F，必须先由 Stage A 的模式级门槛授权。
"""


def run_stage_a_resolution(root: Path, output: Path) -> dict[str, Any]:
    """Validate components and freeze formal branch-specific readiness."""

    project = Path(root).resolve()
    destination = prepare_formal_output(Path(output).resolve())
    component_root = project / "work" / "field_observation_chain_stage_a"
    kriging_source = component_root / "frozen_field_kriging.mat"
    operator_source = component_root / "field_forward_operators.h5"
    base_audit_path = (
        project
        / "paper_results"
        / "field_observation_chain_audit_stage_a"
        / "field_chain_audit.json"
    )
    required = (kriging_source, operator_source, base_audit_path)
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing Stage-A component(s): " + ", ".join(map(str, missing))
        )

    kriging_summary, artifact = _kriging_summary(kriging_source)
    operator_summary = _operator_summary(operator_source)
    theta = np.asarray(kriging_summary["theta"], dtype=np.float64)
    loo = frozen_theta_cv_residuals(
        artifact["station_xy"],
        artifact["station_gravity"],
        theta,
        mode="leave_one_out",
    )
    blocked = frozen_theta_cv_residuals(
        artifact["station_xy"],
        artifact["station_gravity"],
        theta,
        mode="spatial_block",
        n_folds=5,
        block_axis=1,
    )
    envelope = build_error_envelope(
        measurement_floor_mgal=0.05,
        loo_residuals=loo["residual"],
        spatial_block_residuals=blocked["residual"],
    )
    loo_spatial = residual_spatial_diagnostics(
        artifact["station_xy"],
        loo["residual"],
    )
    block_spatial = residual_spatial_diagnostics(
        artifact["station_xy"],
        blocked["residual"],
    )
    spectrum = gridded_residual_spectrum(
        artifact["K"],
        loo["residual"],
    )
    envelope["spatial_diagnostics"] = {
        "frozen_theta_leave_one_out": loo_spatial,
        "y_contiguous_spatial_block": block_spatial,
    }
    envelope["frozen_K_LOO_residual_spectrum"] = spectrum
    envelope["noise_families_to_freeze_before_stage_a"] = {
        "white": {
            "sigma_mgal": envelope["white_noise_sigma_mgal"],
        },
        "spatially_correlated": {
            "sigma_mgal": envelope["white_noise_sigma_mgal"],
            "nominal_correlation_length_m_heuristic": (
                loo_spatial["correlation_length_m_heuristic"]
            ),
            "status": "empirical_parameterization_not_known_covariance_truth",
        },
        "low_frequency_regional_residual": {
            "amplitude_mgal": {
                "low": 0.0,
                "nominal": loo_spatial["planar_component_rms_mgal"],
                "high": max(
                    loo_spatial["planar_component_rms_mgal"],
                    block_spatial["planar_component_rms_mgal"],
                ),
            },
            "status": "planar_component_envelope_not_known_regional_field_truth",
        },
    }

    kriging_passed = bool(
        kriging_summary["validation"]["passed"]
        and kriging_summary["validation"]["python_independent_K_match"]
    )
    operator_passed = bool(operator_summary["all_branches_passed"])
    readiness = resolve_stage_a_readiness(
        field3_authoritative=False,
        kriging_validation_passed=kriging_passed,
        field_operator_validation_passed=operator_passed,
        error_envelope_frozen=True,
    )
    payload = {
        "schema_version": 1,
        "analysis_type": "stage_a_field_chain_blocker_resolution",
        "training_executed": False,
        "counterfactual_identifiability_executed": False,
        "base_audit_sha256": sha256_file(base_audit_path),
        "kriging": kriging_summary,
        "field_forward_operators": operator_summary,
        "error_envelope": envelope,
        "field_3_semantics": _source_evidence()["field_3_semantics"],
        "readiness": readiness,
    }

    kriging_dir = destination / "kriging"
    operator_dir = destination / "operators"
    noise_dir = destination / "noise"
    kriging_dir.mkdir()
    operator_dir.mkdir()
    noise_dir.mkdir()
    shutil.copy2(
        kriging_source,
        kriging_dir / "frozen_field_kriging.mat",
    )
    shutil.copy2(
        operator_source,
        operator_dir / "field_forward_operators.h5",
    )
    _write_json(
        kriging_dir / "frozen_field_kriging_summary.json",
        kriging_summary,
    )
    _write_json(
        operator_dir / "field_forward_operator_validation.json",
        operator_summary,
    )
    _write_json(
        noise_dir / "noise_and_preprocessing_error_envelope.json",
        envelope,
    )
    _write_residual_csv(
        noise_dir / "cv_residuals.csv",
        artifact,
        loo,
        blocked,
    )
    _write_diagnostic_figure(
        noise_dir / "error_envelope_diagnostics.png",
        artifact,
        loo,
        blocked,
        loo_spatial,
        block_spatial,
        spectrum,
    )
    _write_json(destination / "source_evidence.json", _source_evidence())
    _write_json(destination / "stage_a_chain_resolution.json", payload)
    (destination / "stage_a_readiness_CN.md").write_text(
        _readiness_markdown(payload),
        encoding="utf-8",
    )
    write_artifact_manifest(destination)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = run_stage_a_resolution(args.root, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "readiness": payload["readiness"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
