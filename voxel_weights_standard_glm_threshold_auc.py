#!/usr/bin/env python3
"""Sweep optimization-weight percentiles against a GLM-style reference map."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from analyze_ablation_constraints import (
    DEFAULT_FULL_MODEL_HTML as DEFAULT_MOTOR_OVERLAP_HTML,
    DEFAULT_TASK_ONLY_MAP as DEFAULT_MOTOR_OVERLAP_TASK_MAP,
    DEFAULT_TASK_ONLY_Z_THRESHOLD as DEFAULT_MOTOR_OVERLAP_TASK_Z_THRESHOLD,
    _html_sprite_volumes,
)
from glm_typea_threshold_auc import (
    DEFAULT_REFERENCE_Z_THRESHOLD,
    DEFAULT_SELECTED_THRESHOLD,
    DEFAULT_STANDARD_GLM,
    DEFAULT_TYPEA_GLM,
    DEFAULT_TYPED_GLM,
    DEFAULT_TYPED_SELECTED_THRESHOLD,
    DEFAULT_WEIGHT_MAP,
    _best_row,
    _check_same_grid,
    _heatmap_support_mask,
    _load_img,
    _metrics_from_counts,
    _positive_threshold_auc,
    _row_payload,
    _save_binary_map,
)
from motor_overlap_overlay import motor_overlap_masks


DEFAULT_OUT_BASE = Path("figures/voxel_weights_vs_standard_glm_threshold_auc")
DEFAULT_PERCENTILES = tuple(np.arange(95.0, 59.9, -5.0))
DEFAULT_MARKER_PERCENTILES = (
    95.0,
    90.0,
    85.0,
    80.0,
    75.0,
    70.0,
    65.0,
    60.0,
)


def _pct_label(percentile: float) -> str:
    return f"p{percentile:g}"


def _percentile_thresholds(weight_data: np.ndarray, source_mask: np.ndarray, percentiles):
    values = weight_data[source_mask]
    if values.size == 0:
        raise RuntimeError("No nonzero finite network weights are available in the analysis mask.")
    return {float(p): float(np.percentile(values, float(p))) for p in percentiles}


def _metrics_for_percentile(
    reference_mask: np.ndarray,
    weight_data: np.ndarray,
    analysis_mask: np.ndarray,
    weight_source_mask: np.ndarray,
    percentile: float,
    threshold_value: float,
    extra_prediction_mask: np.ndarray | None = None,
) -> dict[str, object]:
    base_predicted_mask = weight_source_mask & (weight_data >= threshold_value)
    predicted_mask = base_predicted_mask
    if extra_prediction_mask is not None:
        predicted_mask = (predicted_mask | extra_prediction_mask) & analysis_mask
    gold = reference_mask & analysis_mask
    background = analysis_mask & ~reference_mask

    tp = int(np.count_nonzero(predicted_mask & gold))
    fp = int(np.count_nonzero(predicted_mask & background))
    fn = int(np.count_nonzero(~predicted_mask & gold))
    tn = int(np.count_nonzero(~predicted_mask & background))
    metrics = _metrics_from_counts(threshold_value, tp, fp, tn, fn)
    metrics["percentile"] = float(percentile)
    metrics["threshold_label"] = _pct_label(percentile)
    metrics["top_weight_fraction"] = float(100.0 - percentile)
    metrics["threshold_value"] = float(threshold_value)
    metrics["extra_prediction_voxels"] = (
        int(np.count_nonzero(extra_prediction_mask & analysis_mask))
        if extra_prediction_mask is not None
        else 0
    )
    metrics["extra_prediction_voxels_added_at_threshold"] = (
        int(np.count_nonzero(extra_prediction_mask & ~base_predicted_mask & analysis_mask))
        if extra_prediction_mask is not None
        else 0
    )
    return metrics


def _rank_tie_correction(values: np.ndarray) -> float:
    _, counts = np.unique(values, return_counts=True)
    counts = counts[counts > 1].astype(np.float64)
    return float(np.sum(counts**3 - counts))


def _kendall_w_between_maps(
    reference_data: np.ndarray,
    weight_data: np.ndarray,
    rank_mask: np.ndarray,
) -> float:
    standard_values = reference_data[rank_mask]
    weight_values = weight_data[rank_mask]
    if standard_values.size < 2:
        return np.nan
    ranks = np.vstack(
        [
            rankdata(-standard_values, method="average"),
            rankdata(-weight_values, method="average"),
        ]
    )
    n_models, n_voxels = ranks.shape
    rank_sums = np.sum(ranks, axis=0)
    sum_squares = float(np.sum((rank_sums - np.mean(rank_sums)) ** 2))
    standard_tie_correction = _rank_tie_correction(standard_values)
    weight_tie_correction = _rank_tie_correction(weight_values)
    denominator = float(
        n_models * n_models * (n_voxels**3 - n_voxels)
        - n_models * (standard_tie_correction + weight_tie_correction)
    )
    return float(12.0 * sum_squares / denominator) if denominator > 0 else np.nan


def _add_kendall_w_by_percentile(
    metrics_df: pd.DataFrame,
    reference_data: np.ndarray,
    weight_data: np.ndarray,
    analysis_mask: np.ndarray,
    reference_mask: np.ndarray,
    weight_source_mask: np.ndarray,
    extra_prediction_mask: np.ndarray | None = None,
) -> pd.DataFrame:
    rows = []
    for row in metrics_df.itertuples(index=False):
        threshold_value = float(row.threshold_value)
        predicted_mask = weight_source_mask & (weight_data >= threshold_value)
        if extra_prediction_mask is not None:
            predicted_mask = (predicted_mask | extra_prediction_mask) & analysis_mask
        rank_mask = analysis_mask & (
            reference_mask | predicted_mask
        )
        rows.append(
            {
                "percentile": float(row.percentile),
                "kendall_w": _kendall_w_between_maps(reference_data, weight_data, rank_mask),
                "kendall_w_voxels": int(np.count_nonzero(rank_mask)),
            }
        )
    kendall_df = pd.DataFrame(rows)
    return metrics_df.merge(kendall_df, on="percentile", how="left")


def _percentile_sweep(
    reference_mask: np.ndarray,
    weight_data: np.ndarray,
    analysis_mask: np.ndarray,
    percentiles,
    extra_prediction_mask: np.ndarray | None = None,
) -> tuple[pd.DataFrame, dict[float, float], np.ndarray]:
    weight_source_mask = analysis_mask & np.isfinite(weight_data) & (weight_data != 0)
    clean_percentiles = sorted({float(p) for p in percentiles}, reverse=True)
    thresholds = _percentile_thresholds(weight_data, weight_source_mask, clean_percentiles)
    rows = [
        _metrics_for_percentile(
            reference_mask,
            weight_data,
            analysis_mask,
            weight_source_mask,
            percentile,
            thresholds[percentile],
            extra_prediction_mask=extra_prediction_mask,
        )
        for percentile in clean_percentiles
    ]
    return pd.DataFrame(rows), thresholds, weight_source_mask


def _align_mask_to_affine(
    mask: np.ndarray,
    source_affine: np.ndarray,
    target_affine: np.ndarray,
) -> np.ndarray:
    aligned = mask.copy()
    for axis in range(3):
        source_step = float(source_affine[axis, axis])
        target_step = float(target_affine[axis, axis])
        if source_step != 0.0 and target_step != 0.0 and np.sign(source_step) != np.sign(target_step):
            aligned = np.flip(aligned, axis=axis)
    return aligned


def _motor_overlap_prediction_mask(
    full_model_html: Path,
    target_img,
    task_only_map: Path,
    task_z_threshold: float,
    analysis_mask: np.ndarray,
    reference_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    _, selected_mask, html_affine = _html_sprite_volumes(full_model_html)
    if selected_mask.shape != target_img.shape[:3]:
        raise RuntimeError(
            f"{full_model_html} overlay shape {selected_mask.shape} does not match "
            f"target image shape {target_img.shape[:3]}."
        )

    display_mask, shared_motor_mask = motor_overlap_masks(
        selected_mask,
        html_affine,
        task_only_map=task_only_map,
        task_z_threshold=task_z_threshold,
    )
    selected_aligned = _align_mask_to_affine(selected_mask, html_affine, target_img.affine)
    display_aligned = _align_mask_to_affine(display_mask, html_affine, target_img.affine)
    extra_mask = display_aligned & analysis_mask
    added_aligned = display_aligned & ~selected_aligned
    metadata = {
        "enabled": True,
        "definition": (
            "Dilated shared-motor display mask from the full-model HTML overlay "
            "and task-only z map, matching ablation_full_vs_task_only_anatomy."
        ),
        "full_model_html": str(full_model_html),
        "task_only_map": str(task_only_map),
        "task_only_threshold": f"z >= {task_z_threshold:g}",
        "html_selected_voxels": int(np.count_nonzero(selected_mask)),
        "shared_motor_voxels": int(np.count_nonzero(shared_motor_mask)),
        "display_voxels": int(np.count_nonzero(display_mask)),
        "display_voxels_added_beyond_html": int(np.count_nonzero(display_mask & ~selected_mask)),
        "display_voxels_in_analysis_mask": int(np.count_nonzero(extra_mask)),
        "display_voxels_excluded_by_analysis_mask": int(np.count_nonzero(display_aligned & ~analysis_mask)),
        "display_voxels_overlapping_reference": int(np.count_nonzero(extra_mask & reference_mask)),
        "added_display_voxels_in_analysis_mask": int(np.count_nonzero(added_aligned & analysis_mask)),
        "added_display_voxels_overlapping_reference": int(
            np.count_nonzero(added_aligned & analysis_mask & reference_mask)
        ),
    }
    return extra_mask, metadata


def _plot_network_metrics(
    metrics_df: pd.DataFrame,
    marker_metrics_df: pd.DataFrame,
    best_dice: pd.Series,
    auc_full: float,
    partial_auc: float,
    reference_label: str,
    out_base: Path,
) -> None:
    fig, (left_ax, middle_ax, right_ax) = plt.subplots(
        1, 3, figsize=(19.2, 5.35), facecolor="white"
    )

    marker_metrics_df = marker_metrics_df.sort_values(
        ["false_positive_rate", "sensitivity"]
    ).copy()
    finite_metrics = metrics_df[np.isfinite(metrics_df["threshold"])].copy()

    left_ax.plot(
        marker_metrics_df["false_positive_rate"],
        marker_metrics_df["sensitivity"],
        color="#1b6ca8",
        linewidth=1.8,
        marker="o",
        markersize=6,
    )
    for index, row in enumerate(marker_metrics_df.itertuples(index=False)):
        offset_y = 5 if index % 2 == 0 else -13
        left_ax.annotate(
            _pct_label(row.percentile),
            (row.false_positive_rate, row.sensitivity),
            xytext=(5, offset_y),
            textcoords="offset points",
            fontsize=8,
            color="#111111",
        )
    left_ax.set_title("Optimization network: sensitivity vs 1 - specificity")
    left_ax.set_xlabel("1 - specificity")
    left_ax.set_ylabel("Sensitivity")
    left_ax.set_xlim(
        0.0,
        min(1.0, float(marker_metrics_df["false_positive_rate"].max()) + 0.004),
    )
    left_ax.set_ylim(
        -0.02,
        min(1.0, float(marker_metrics_df["sensitivity"].max()) + 0.08),
    )
    diagonal_min = max(left_ax.get_xlim()[0], left_ax.get_ylim()[0])
    diagonal_max = min(left_ax.get_xlim()[1], left_ax.get_ylim()[1])
    left_ax.plot(
        [diagonal_min, diagonal_max],
        [diagonal_min, diagonal_max],
        color="#666666",
        linewidth=1.0,
        linestyle="--",
        alpha=0.7,
        zorder=0,
    )
    left_ax.grid(True, linewidth=0.6, alpha=0.25)

    plot_metrics = finite_metrics.sort_values("percentile", ascending=False)
    middle_ax.plot(
        plot_metrics["percentile"],
        plot_metrics["sensitivity"],
        color="#1b6ca8",
        linewidth=1.8,
        label="Sensitivity",
    )
    middle_ax.plot(
        plot_metrics["percentile"],
        plot_metrics["specificity"],
        color="#7d5fb2",
        linewidth=1.8,
        label="Specificity",
    )
    middle_ax.plot(
        plot_metrics["percentile"],
        plot_metrics["dice"],
        color="#2f8f5b",
        linewidth=1.8,
        label="Dice overlap",
    )
    for percentile in marker_metrics_df["percentile"]:
        middle_ax.axvline(percentile, color="#bbbbbb", linewidth=0.7, alpha=0.28)
    middle_ax.set_title("Optimization network: metrics across percentile thresholds")
    middle_ax.set_xlabel("Network weight percentile threshold")
    middle_ax.set_ylabel("Metric value")
    middle_ax.set_xlim(
        float(plot_metrics["percentile"].max()) + 1.0,
        float(plot_metrics["percentile"].min()) - 1.0,
    )
    middle_ax.set_ylim(-0.01, 1.01)
    middle_ax.grid(True, linewidth=0.6, alpha=0.25)
    middle_ax.legend(loc="best", fontsize=9)

    right_ax.plot(
        plot_metrics["percentile"],
        plot_metrics["kendall_w"],
        color="#b55d19",
        linewidth=1.8,
        marker="o",
        markersize=6,
    )
    right_ax.set_title(f"{reference_label} vs network rank concordance")
    right_ax.set_xlabel("Network weight percentile threshold")
    right_ax.set_ylabel("Kendall W")
    right_ax.set_xlim(
        float(plot_metrics["percentile"].max()) + 1.0,
        float(plot_metrics["percentile"].min()) - 1.0,
    )
    right_ax.set_ylim(
        max(0.0, float(plot_metrics["kendall_w"].min()) - 0.04),
        min(1.0, float(plot_metrics["kendall_w"].max()) + 0.04),
    )
    right_ax.grid(True, linewidth=0.6, alpha=0.25)

    fig.tight_layout()
    fig.savefig(f"{out_base}_sensitivity_1minus_specificity.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{out_base}_sensitivity_1minus_specificity.pdf", bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep optimization-weight percentile thresholds against the "
            "selected z-threshold reference."
        )
    )
    parser.add_argument("--standard-glm", type=Path, default=DEFAULT_STANDARD_GLM)
    parser.add_argument("--typea-glm", type=Path, default=DEFAULT_TYPEA_GLM)
    parser.add_argument("--typed-glm", type=Path, default=DEFAULT_TYPED_GLM)
    parser.add_argument("--weight-map", type=Path, default=DEFAULT_WEIGHT_MAP)
    parser.add_argument("--out-base", type=Path, default=DEFAULT_OUT_BASE)
    parser.add_argument(
        "--reference",
        choices=("standard-glm", "glmsingle-type-a", "glmsingle-type-d"),
        default="standard-glm",
        help="Reference map used as the positive class.",
    )
    parser.add_argument(
        "--reference-z-threshold",
        type=float,
        default=None,
        help=(
            "Reference z threshold. Defaults to 3.1 for standard GLM, "
            "1.5 for GLMsingle Type A, and 0.86 for GLMsingle Type D."
        ),
    )
    parser.add_argument(
        "--percentiles",
        type=float,
        nargs="+",
        default=list(DEFAULT_PERCENTILES),
        help="Network-weight percentiles to sweep. Default starts at p95 and decreases to p60.",
    )
    parser.add_argument(
        "--marker-percentiles",
        type=float,
        nargs="+",
        default=list(DEFAULT_MARKER_PERCENTILES),
        help="Percentiles to label on the sensitivity vs 1-specificity plot.",
    )
    parser.add_argument(
        "--no-binary-maps",
        action="store_true",
        help="Skip writing the reference and selected network binary NIfTI maps.",
    )
    parser.add_argument(
        "--include-motor-overlap-display",
        action="store_true",
        help=(
            "Include the same dilated shared-motor display mask used by "
            "ablation_full_vs_task_only_anatomy in each network prediction mask."
        ),
    )
    parser.add_argument(
        "--motor-overlap-html",
        type=Path,
        default=DEFAULT_MOTOR_OVERLAP_HTML,
        help="Thresholded HTML overlay used to derive the motor-overlap display mask.",
    )
    parser.add_argument(
        "--motor-overlap-task-map",
        type=Path,
        default=DEFAULT_MOTOR_OVERLAP_TASK_MAP,
        help="Task-only z map used to derive the shared motor overlap mask.",
    )
    parser.add_argument(
        "--motor-overlap-task-z-threshold",
        type=float,
        default=DEFAULT_MOTOR_OVERLAP_TASK_Z_THRESHOLD,
        help="Task-only z threshold used to derive the shared motor overlap mask.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    standard_img = _load_img(args.standard_glm)
    typea_img = _load_img(args.typea_glm)
    typed_img = _load_img(args.typed_glm)
    weight_img = _load_img(args.weight_map)
    for img in (typea_img, typed_img, weight_img):
        _check_same_grid(standard_img, img)

    standard_data = np.asarray(standard_img.get_fdata(), dtype=float)
    typea_data = np.asarray(typea_img.get_fdata(), dtype=float)
    typed_data = np.asarray(typed_img.get_fdata(), dtype=float)
    weight_data = np.asarray(weight_img.get_fdata(), dtype=float)

    if args.reference == "standard-glm":
        reference_img = standard_img
        reference_data = standard_data
        reference_label = "Standard GLM"
        reference_threshold = (
            DEFAULT_REFERENCE_Z_THRESHOLD
            if args.reference_z_threshold is None
            else float(args.reference_z_threshold)
        )
    elif args.reference == "glmsingle-type-a":
        reference_img = typea_img
        reference_data = typea_data
        reference_label = "GLMsingle Type A"
        reference_threshold = (
            DEFAULT_SELECTED_THRESHOLD
            if args.reference_z_threshold is None
            else float(args.reference_z_threshold)
        )
    elif args.reference == "glmsingle-type-d":
        reference_img = typed_img
        reference_data = typed_data
        reference_label = "GLMsingle Type D"
        reference_threshold = (
            DEFAULT_TYPED_SELECTED_THRESHOLD
            if args.reference_z_threshold is None
            else float(args.reference_z_threshold)
        )
    else:
        raise ValueError(f"Unknown reference: {args.reference}")

    analysis_mask = _heatmap_support_mask(standard_data, typea_data, typed_data, weight_data)
    analysis_mask &= (
        np.isfinite(standard_data)
        & np.isfinite(typea_data)
        & np.isfinite(typed_data)
        & np.isfinite(weight_data)
    )
    if not np.any(analysis_mask):
        raise RuntimeError("The analysis mask is empty.")

    reference_mask = analysis_mask & (reference_data >= reference_threshold)
    if not np.any(reference_mask):
        raise RuntimeError("The reference gold-standard map has no suprathreshold voxels.")
    if np.all(reference_mask[analysis_mask]):
        raise RuntimeError("The analysis mask contains no reference-negative voxels.")

    extra_prediction_mask = None
    motor_overlap_metadata: dict[str, object] = {"enabled": False}
    if args.include_motor_overlap_display:
        extra_prediction_mask, motor_overlap_metadata = _motor_overlap_prediction_mask(
            args.motor_overlap_html,
            standard_img,
            args.motor_overlap_task_map,
            float(args.motor_overlap_task_z_threshold),
            analysis_mask,
            reference_mask,
        )

    requested_percentiles = sorted({float(p) for p in args.percentiles}, reverse=True)
    if not requested_percentiles:
        raise RuntimeError("At least one percentile must be provided.")
    for percentile in requested_percentiles:
        if not 0.0 <= percentile <= 100.0:
            raise ValueError("--percentiles values must be between 0 and 100.")

    metrics_df, threshold_values, weight_source_mask = _percentile_sweep(
        reference_mask,
        weight_data,
        analysis_mask,
        requested_percentiles,
        extra_prediction_mask=extra_prediction_mask,
    )
    metrics_df = _add_kendall_w_by_percentile(
        metrics_df,
        reference_data,
        weight_data,
        analysis_mask,
        reference_mask,
        weight_source_mask,
        extra_prediction_mask=extra_prediction_mask,
    )
    best_dice = _best_row(metrics_df, "dice")
    if best_dice is None:
        raise RuntimeError("Could not identify a best Dice percentile.")

    marker_percentiles = sorted(
        ({float(p) for p in args.marker_percentiles} | {float(best_dice["percentile"])})
        & set(requested_percentiles),
        reverse=True,
    )
    marker_metrics_df = metrics_df[metrics_df["percentile"].isin(marker_percentiles)].copy()

    labels = reference_mask[analysis_mask].astype(np.uint8)
    scores = weight_data[analysis_mask]
    auc_full = float(roc_auc_score(labels, scores))
    partial_auc = _positive_threshold_auc(metrics_df)

    out_base = args.out_base
    out_base.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = Path(f"{out_base}_metrics.csv")
    marker_metrics_path = Path(f"{out_base}_marker_percentile_metrics.csv")
    summary_path = Path(f"{out_base}_summary.json")
    metrics_df.to_csv(metrics_path, index=False)
    marker_metrics_df.to_csv(marker_metrics_path, index=False)
    _plot_network_metrics(
        metrics_df,
        marker_metrics_df,
        best_dice,
        auc_full,
        partial_auc,
        reference_label,
        out_base,
    )

    binary_outputs = {}
    if not args.no_binary_maps:
        reference_path = Path(
            f"{out_base}_reference_zge{reference_threshold:g}_binary.nii.gz"
        )
        _save_binary_map(reference_mask, reference_img, reference_path)
        binary_outputs["reference"] = str(reference_path)

        best_percentile = float(best_dice["percentile"])
        best_threshold = float(best_dice["threshold_value"])
        best_mask = weight_source_mask & (weight_data >= best_threshold)
        if extra_prediction_mask is not None:
            best_mask = (best_mask | extra_prediction_mask) & analysis_mask
        best_path = Path(f"{out_base}_weights_{_pct_label(best_percentile)}_binary.nii.gz")
        _save_binary_map(best_mask, standard_img, best_path)
        binary_outputs["network_best_dice"] = str(best_path)

    summary = {
        "inputs": {
            "standard_glm": str(args.standard_glm),
            "typea_glm_for_heatmap_support_mask": str(args.typea_glm),
            "typed_glm_for_heatmap_support_mask": str(args.typed_glm),
            "weight_map": str(args.weight_map),
        },
        "reference": {
            "name": reference_label,
            "path": str(
                {
                    "standard-glm": args.standard_glm,
                    "glmsingle-type-a": args.typea_glm,
                    "glmsingle-type-d": args.typed_glm,
                }[args.reference]
            ),
            "definition": f"{reference_label} z >= {reference_threshold:g}",
            "voxels": int(np.count_nonzero(reference_mask)),
        },
        "analysis_mask": {
            "mode": "heatmap-support",
            "voxels": int(np.count_nonzero(analysis_mask)),
        },
        "network": {
            "threshold_basis": "percentiles of nonzero finite weights inside analysis mask",
            "prediction_mask": (
                "weight percentile mask OR motor-overlap display mask"
                if args.include_motor_overlap_display
                else "weight percentile mask"
            ),
            "nonzero_weight_voxels_in_analysis_mask": int(np.count_nonzero(weight_source_mask)),
            "swept_percentiles": [float(p) for p in requested_percentiles],
            "threshold_values": {f"{p:g}": threshold_values[p] for p in requested_percentiles},
            "auc_full_scores": auc_full,
            "partial_auc_percentile_thresholds": partial_auc,
            "best_dice": _row_payload(best_dice),
            "motor_overlap_display": motor_overlap_metadata,
        },
        "outputs": {
            "metrics_csv": str(metrics_path),
            "marker_percentile_metrics_csv": str(marker_metrics_path),
            "sensitivity_1minus_specificity_png": f"{out_base}_sensitivity_1minus_specificity.png",
            "sensitivity_1minus_specificity_pdf": f"{out_base}_sensitivity_1minus_specificity.pdf",
            "binary_maps": binary_outputs,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Analysis mask voxels: {summary['analysis_mask']['voxels']:,}")
    print(f"Reference voxels: {summary['reference']['voxels']:,}")
    print(f"Network full-score ROC AUC: {auc_full:.4f}")
    print(f"Network percentile-threshold AUC: {partial_auc:.4f}")
    print(
        "Network best Dice threshold: "
        f"{_pct_label(float(best_dice['percentile']))} "
        f"(weight >= {float(best_dice['threshold_value']):.6g}, "
        f"sensitivity={float(best_dice['sensitivity']):.4f}, "
        f"specificity={float(best_dice['specificity']):.4f}, "
        f"dice={float(best_dice['dice']):.4f})"
    )
    print(f"Saved {metrics_path}")
    print(f"Saved {marker_metrics_path}")
    print(f"Saved {out_base}_sensitivity_1minus_specificity.png")
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
