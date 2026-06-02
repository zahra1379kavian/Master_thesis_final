#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


DEFAULT_STANDARD_GLM = Path("data/z_valu_standard_glm.nii.gz")
DEFAULT_TYPEA_GLM = Path("data/z_value_typaA.nii.gz")
DEFAULT_OUT_BASE = Path("figures/typea_vs_standard_glm_threshold_auc")
DEFAULT_REFERENCE_Z_THRESHOLD = 3.1
DEFAULT_MARKER_THRESHOLDS = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5)
COUNT_COLUMNS = {
    "selected_voxels",
    "reference_voxels",
    "overlap_voxels",
    "tp",
    "fp",
    "tn",
    "fn",
}


def _load_img(path):
    if not path.exists():
        raise RuntimeError(f"Missing input map: {path}")
    return nib.load(str(path))


def _check_same_grid(reference_img, candidate_img):
    if candidate_img.shape[:3] != reference_img.shape[:3]:
        raise RuntimeError(
            f"Type A shape {candidate_img.shape[:3]} differs from reference {reference_img.shape[:3]}"
        )
    if not np.allclose(candidate_img.affine, reference_img.affine):
        raise RuntimeError("Type A affine differs from the reference image")


def _analysis_mask(reference_data, typea_data, mode):
    finite_both = np.isfinite(reference_data) & np.isfinite(typea_data)
    if mode == "finite-both":
        return finite_both
    if mode == "typea-nonzero":
        return finite_both & (typea_data != 0)
    if mode == "typea-positive":
        return finite_both & (typea_data > 0)
    raise ValueError(f"Unknown analysis mask mode: {mode}")


def _safe_ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else np.nan


def _metrics_for_threshold(reference_mask, scores, analysis_mask, threshold):
    predicted_mask = analysis_mask & (scores >= threshold)
    gold = reference_mask & analysis_mask
    background = analysis_mask & ~reference_mask

    tp = int(np.count_nonzero(predicted_mask & gold))
    fp = int(np.count_nonzero(predicted_mask & background))
    fn = int(np.count_nonzero(~predicted_mask & gold))
    tn = int(np.count_nonzero(~predicted_mask & background))
    selected = tp + fp
    gold_count = tp + fn
    background_count = tn + fp
    union = tp + fp + fn

    sensitivity = _safe_ratio(tp, gold_count)
    specificity = _safe_ratio(tn, background_count)
    return {
        "threshold": float(threshold),
        "selected_voxels": int(selected),
        "reference_voxels": int(gold_count),
        "overlap_voxels": int(tp),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "false_positive_rate": 1.0 - specificity if np.isfinite(specificity) else np.nan,
        "precision": _safe_ratio(tp, selected),
        "negative_predictive_value": _safe_ratio(tn, tn + fn),
        "accuracy": _safe_ratio(tp + tn, tp + fp + tn + fn),
        "balanced_accuracy": np.nanmean([sensitivity, specificity]),
        "youden_index": sensitivity + specificity - 1.0
        if np.isfinite(sensitivity) and np.isfinite(specificity)
        else np.nan,
        "dice": _safe_ratio(2 * tp, 2 * tp + fp + fn),
        "jaccard": _safe_ratio(tp, union),
        "overlap_of_reference": _safe_ratio(tp, gold_count),
        "overlap_of_typea": _safe_ratio(tp, selected),
    }


def _threshold_sweep(reference_mask, scores, analysis_mask):
    positive_scores = scores[analysis_mask & (scores > 0)]
    if positive_scores.size == 0:
        raise RuntimeError("No positive Type A z values are available in the analysis mask.")
    thresholds = np.unique(positive_scores)
    rows = [_metrics_for_threshold(reference_mask, scores, analysis_mask, np.inf)]
    rows.extend(
        _metrics_for_threshold(reference_mask, scores, analysis_mask, threshold)
        for threshold in thresholds[::-1]
    )
    return pd.DataFrame(rows)


def _best_row(metrics_df, column):
    candidates = metrics_df[np.isfinite(metrics_df["threshold"])].copy()
    candidates = candidates[np.isfinite(candidates[column])]
    if candidates.empty:
        return None
    candidates = candidates.sort_values(
        [column, "sensitivity", "specificity", "threshold"],
        ascending=[False, False, False, False],
    )
    return candidates.iloc[0]


def _positive_threshold_auc(metrics_df):
    curve = metrics_df[["false_positive_rate", "sensitivity"]].dropna().copy()
    curve = curve.sort_values(["false_positive_rate", "sensitivity"])
    curve = curve.drop_duplicates(subset=["false_positive_rate"], keep="last")
    return float(np.trapezoid(curve["sensitivity"], curve["false_positive_rate"]))


def _marker_metrics(reference_mask, scores, analysis_mask, thresholds):
    rows = [
        _metrics_for_threshold(reference_mask, scores, analysis_mask, float(threshold))
        for threshold in thresholds
    ]
    return pd.DataFrame(rows)


def _plot_metrics(
    metrics_df,
    full_fpr,
    full_tpr,
    auc_full,
    partial_auc_positive_thresholds,
    marker_metrics_df,
    best_youden,
    best_dice,
    out_base,
):
    marker_metrics_df = marker_metrics_df.sort_values("threshold").copy()

    fig, ax = plt.subplots(figsize=(7.0, 5.6), facecolor="white")
    ax.plot(
        marker_metrics_df["specificity"],
        marker_metrics_df["sensitivity"],
        color="#1b6ca8",
        linewidth=1.8,
        marker="o",
        markersize=6,
    )
    for row in marker_metrics_df.itertuples(index=False):
        ax.annotate(
            f"z >= {row.threshold:g}",
            (row.specificity, row.sensitivity),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=9,
            color="#111111",
        )
    ax.set_title("Sensitivity vs specificity across Type A thresholds")
    ax.set_xlabel("Specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_xlim(max(0.0, float(marker_metrics_df["specificity"].min()) - 0.01), 1.0)
    ax.set_ylim(0.0, min(1.0, float(marker_metrics_df["sensitivity"].max()) + 0.08))
    ax.grid(True, linewidth=0.6, alpha=0.25)

    fig.tight_layout()
    fig.savefig(f"{out_base}_sensitivity_specificity.png", dpi=220, bbox_inches="tight")
    fig.savefig(f"{out_base}_sensitivity_specificity.pdf", bbox_inches="tight")
    fig.savefig(f"{out_base}_roc.png", dpi=220, bbox_inches="tight")
    fig.savefig(f"{out_base}_roc.pdf", bbox_inches="tight")
    plt.close(fig)


def _save_binary_map(mask, reference_img, path):
    header = reference_img.header.copy()
    header.set_data_dtype(np.uint8)
    binary_img = nib.Nifti1Image(mask.astype(np.uint8), reference_img.affine, header)
    nib.save(binary_img, str(path))


def _row_payload(row):
    if row is None:
        return None
    payload = {}
    for key, value in row.to_dict().items():
        if isinstance(value, np.generic):
            value = value.item()
        if key in COUNT_COLUMNS and np.isfinite(value):
            value = int(value)
        payload[key] = value
    return payload


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Sweep positive thresholds in a GLMsingle Type A z map against a "
            "standard-GLM z-threshold gold-standard map."
        )
    )
    parser.add_argument("--standard-glm", type=Path, default=DEFAULT_STANDARD_GLM)
    parser.add_argument("--typea-glm", type=Path, default=DEFAULT_TYPEA_GLM)
    parser.add_argument("--out-base", type=Path, default=DEFAULT_OUT_BASE)
    parser.add_argument("--reference-z-threshold", type=float, default=DEFAULT_REFERENCE_Z_THRESHOLD)
    parser.add_argument(
        "--marker-thresholds",
        type=float,
        nargs="+",
        default=list(DEFAULT_MARKER_THRESHOLDS),
        help="Type A z thresholds to mark explicitly on the ROC plot.",
    )
    parser.add_argument(
        "--analysis-mask",
        choices=("finite-both", "typea-nonzero", "typea-positive"),
        default="finite-both",
        help=(
            "Voxel set used for TP/FP/TN/FN counts. Default uses every voxel finite in both maps."
        ),
    )
    parser.add_argument(
        "--no-binary-maps",
        action="store_true",
        help="Skip writing the reference and selected-threshold binary NIfTI maps.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    standard_img = _load_img(args.standard_glm)
    typea_img = _load_img(args.typea_glm)
    _check_same_grid(standard_img, typea_img)

    standard_data = np.asarray(standard_img.get_fdata(), dtype=float)
    typea_data = np.asarray(typea_img.get_fdata(), dtype=float)
    analysis_mask = _analysis_mask(standard_data, typea_data, args.analysis_mask)
    if not np.any(analysis_mask):
        raise RuntimeError("The analysis mask is empty.")

    reference_mask = analysis_mask & (standard_data >= float(args.reference_z_threshold))
    if not np.any(reference_mask):
        raise RuntimeError("The reference gold-standard map has no suprathreshold voxels.")
    if np.all(reference_mask[analysis_mask]):
        raise RuntimeError("The analysis mask contains no reference-negative voxels.")

    metrics_df = _threshold_sweep(reference_mask, typea_data, analysis_mask)
    marker_thresholds = sorted(set(float(threshold) for threshold in args.marker_thresholds))
    marker_metrics_df = _marker_metrics(reference_mask, typea_data, analysis_mask, marker_thresholds)
    best_youden = _best_row(metrics_df, "youden_index")
    best_dice = _best_row(metrics_df, "dice")

    labels = reference_mask[analysis_mask].astype(np.uint8)
    scores = typea_data[analysis_mask]
    auc_full = float(roc_auc_score(labels, scores))
    full_fpr, full_tpr, _ = roc_curve(labels, scores)
    partial_auc_positive_thresholds = _positive_threshold_auc(metrics_df)

    out_base = args.out_base
    out_base.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = Path(f"{out_base}_metrics.csv")
    marker_metrics_path = Path(f"{out_base}_marker_threshold_metrics.csv")
    summary_path = Path(f"{out_base}_summary.json")
    metrics_df.to_csv(metrics_path, index=False)
    marker_metrics_df.to_csv(marker_metrics_path, index=False)
    _plot_metrics(
        metrics_df,
        full_fpr,
        full_tpr,
        auc_full,
        partial_auc_positive_thresholds,
        marker_metrics_df,
        best_youden,
        best_dice,
        out_base,
    )

    binary_outputs = {}
    if not args.no_binary_maps:
        reference_path = Path(f"{out_base}_reference_zge{args.reference_z_threshold:g}_binary.nii.gz")
        _save_binary_map(reference_mask, standard_img, reference_path)
        binary_outputs["reference"] = str(reference_path)

        if best_youden is not None:
            threshold = float(best_youden["threshold"])
            path = Path(f"{out_base}_typeA_youden_zge{threshold:.6g}_binary.nii.gz")
            _save_binary_map(analysis_mask & (typea_data >= threshold), standard_img, path)
            binary_outputs["typeA_best_youden"] = str(path)
        if best_dice is not None:
            threshold = float(best_dice["threshold"])
            path = Path(f"{out_base}_typeA_dice_zge{threshold:.6g}_binary.nii.gz")
            _save_binary_map(analysis_mask & (typea_data >= threshold), standard_img, path)
            binary_outputs["typeA_best_dice"] = str(path)

    summary = {
        "inputs": {
            "standard_glm": str(args.standard_glm),
            "typea_glm": str(args.typea_glm),
        },
        "reference": {
            "definition": f"standard GLM z >= {float(args.reference_z_threshold):g}",
            "voxels": int(np.count_nonzero(reference_mask)),
        },
        "analysis_mask": {
            "mode": args.analysis_mask,
            "voxels": int(np.count_nonzero(analysis_mask)),
        },
        "typea_positive_voxels_in_analysis_mask": int(
            np.count_nonzero(analysis_mask & (typea_data > 0))
        ),
        "auc_full_typea_scores": auc_full,
        "partial_auc_positive_thresholds": partial_auc_positive_thresholds,
        "displayed_marker_thresholds": marker_thresholds,
        "best_youden": _row_payload(best_youden),
        "best_dice": _row_payload(best_dice),
        "outputs": {
            "metrics_csv": str(metrics_path),
            "marker_threshold_metrics_csv": str(marker_metrics_path),
            "sensitivity_specificity_png": f"{out_base}_sensitivity_specificity.png",
            "sensitivity_specificity_pdf": f"{out_base}_sensitivity_specificity.pdf",
            "binary_maps": binary_outputs,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Analysis mask voxels: {summary['analysis_mask']['voxels']:,}")
    print(f"Reference voxels: {summary['reference']['voxels']:,}")
    print(f"Full-score ROC AUC: {auc_full:.4f}")
    print(f"Positive-threshold partial ROC AUC: {partial_auc_positive_thresholds:.4f}")
    if best_youden is not None:
        print(
            "Best Youden threshold: "
            f"{float(best_youden['threshold']):.6g} "
            f"(sensitivity={float(best_youden['sensitivity']):.4f}, "
            f"specificity={float(best_youden['specificity']):.4f}, "
            f"dice={float(best_youden['dice']):.4f})"
        )
    if best_dice is not None:
        print(
            "Best Dice threshold: "
            f"{float(best_dice['threshold']):.6g} "
            f"(sensitivity={float(best_dice['sensitivity']):.4f}, "
            f"specificity={float(best_dice['specificity']):.4f}, "
            f"dice={float(best_dice['dice']):.4f})"
        )
    print(f"Saved {metrics_path}")
    print(f"Saved {marker_metrics_path}")
    print(f"Saved {out_base}_sensitivity_specificity.png")
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
