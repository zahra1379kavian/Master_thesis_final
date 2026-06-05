#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score, roc_curve


DEFAULT_STANDARD_GLM = Path("data/z_valu_standard_glm.nii.gz")
DEFAULT_TYPEA_GLM = Path("data/z_value_typaA.nii.gz")
DEFAULT_TYPED_GLM = Path("data/z_value_typeD.nii.gz")
DEFAULT_WEIGHT_MAP = Path("data/voxel_weights_task1_bold0.6_beta0.6_smooth1.25_gamma1.5.nii.gz")
DEFAULT_OUT_BASE = Path("figures/typea_vs_standard_glm_threshold_auc")
DEFAULT_REFERENCE_Z_THRESHOLD = 3.1
DEFAULT_SELECTED_THRESHOLD = 1.5
DEFAULT_TYPED_SELECTED_THRESHOLD = 0.86
DEFAULT_MARKER_THRESHOLDS = tuple(np.arange(0.5, 6.51, 0.5))
PAPER_FONT_FAMILY = "Liberation Sans"
AXIS_TICK_FONT_SIZE = 13
ANNOTATION_FONT_SIZE = 11
LEGEND_FONT_SIZE = 12
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


def _heatmap_support_mask(standard_data, typea_data, typed_data, weight_data):
    from compare_glm_region_highlights import _build_analysis_mask

    return _build_analysis_mask(
        {
            "Standard GLM": standard_data,
            "GLMsingle Type A": typea_data,
            "GLMsingle Type D": typed_data,
            "Optimization weights": weight_data,
        }
    )


def _safe_ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else np.nan


def _metrics_from_counts(threshold, tp, fp, tn, fn):
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
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "false_positive_rate": 1.0 - specificity if np.isfinite(specificity) else np.nan,
        "precision": _safe_ratio(tp, selected),
        "negative_predictive_value": _safe_ratio(tn, tn + fn),
        "accuracy": _safe_ratio(tp + tn, tp + fp + tn + fn),
        "balanced_accuracy": np.nanmean([sensitivity, specificity]),
        "dice": _safe_ratio(2 * tp, 2 * tp + fp + fn),
        "jaccard": _safe_ratio(tp, union),
        "overlap_of_reference": _safe_ratio(tp, gold_count),
        "overlap_of_typea": _safe_ratio(tp, selected),
    }


def _metrics_for_threshold(reference_mask, scores, analysis_mask, threshold):
    predicted_mask = analysis_mask & (scores >= threshold)
    gold = reference_mask & analysis_mask
    background = analysis_mask & ~reference_mask

    tp = int(np.count_nonzero(predicted_mask & gold))
    fp = int(np.count_nonzero(predicted_mask & background))
    fn = int(np.count_nonzero(~predicted_mask & gold))
    tn = int(np.count_nonzero(~predicted_mask & background))
    return _metrics_from_counts(threshold, tp, fp, tn, fn)


def _threshold_sweep(reference_mask, scores, analysis_mask):
    positive_mask = analysis_mask & (scores > 0)
    positive_scores = scores[positive_mask]
    if positive_scores.size == 0:
        raise RuntimeError("No positive Type A z values are available in the analysis mask.")

    gold_count = int(np.count_nonzero(reference_mask & analysis_mask))
    background_count = int(np.count_nonzero(analysis_mask & ~reference_mask))
    order = np.argsort(positive_scores)[::-1]
    sorted_scores = positive_scores[order]
    sorted_gold = reference_mask[positive_mask][order]
    group_ends = np.flatnonzero(np.diff(sorted_scores) != 0) + 1
    group_ends = np.r_[group_ends, sorted_scores.size]
    cumulative_tp = np.cumsum(sorted_gold, dtype=np.int64)[group_ends - 1]
    cumulative_selected = group_ends.astype(np.int64, copy=False)
    cumulative_fp = cumulative_selected - cumulative_tp

    rows = [_metrics_from_counts(np.inf, 0, 0, background_count, gold_count)]
    rows.extend(
        _metrics_from_counts(
            sorted_scores[end - 1],
            int(tp),
            int(fp),
            int(background_count - fp),
            int(gold_count - tp),
        )
        for end, tp, fp in zip(group_ends, cumulative_tp, cumulative_fp)
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


def _rank_tie_correction(values):
    _, counts = np.unique(values, return_counts=True)
    counts = counts[counts > 1].astype(np.float64)
    return float(np.sum(counts**3 - counts))


def _kendall_w_between_maps(
    standard_data,
    candidate_data,
    rank_mask,
    reference_z_threshold,
    candidate_z_threshold,
):
    standard_values = standard_data[rank_mask]
    candidate_values = candidate_data[rank_mask]
    ranks = np.vstack(
        [
            rankdata(-standard_values, method="average"),
            rankdata(-candidate_values, method="average"),
        ]
    )
    n_models, n_voxels = ranks.shape
    rank_sums = np.sum(ranks, axis=0)
    sum_squares = float(np.sum((rank_sums - np.mean(rank_sums)) ** 2))
    standard_tie_correction = _rank_tie_correction(standard_values)
    candidate_tie_correction = _rank_tie_correction(candidate_values)
    denominator = float(
        n_models * n_models * (n_voxels**3 - n_voxels)
        - n_models * (standard_tie_correction + candidate_tie_correction)
    )
    kendall_w = float(12.0 * sum_squares / denominator) if denominator > 0 else np.nan
    return {
        "models": "Standard GLM; candidate",
        "n_models": int(n_models),
        "n_voxels": int(n_voxels),
        "rank_mask_definition": (
            "heatmap-support AND "
            f"(standard GLM z >= {reference_z_threshold:g} OR candidate z >= {candidate_z_threshold:g})"
        ),
        "standard_z_threshold": float(reference_z_threshold),
        "candidate_z_threshold": float(candidate_z_threshold),
        "rank_basis": "z_value",
        "rank_direction": "descending_z",
        "rank_method": "average_ranks_for_ties",
        "tie_correction": "kendall_w_denominator",
        "kendall_w": kendall_w,
    }


def _plot_metrics(candidate_results, out_base):
    n_rows = len(candidate_results)
    fig, axes = plt.subplots(n_rows, 2, figsize=(13.8, 5.35 * n_rows), facecolor="white")
    if n_rows == 1:
        axes = np.asarray([axes])

    selected_color = "#d55e00"
    high_threshold_offsets = {
        5.0: (12, 24),
        5.5: (12, 13),
        6.0: (12, 2),
        6.5: (12, -9),
    }
    for row_index, result in enumerate(candidate_results):
        metrics_df = result["metrics_df"]
        marker_metrics_df = result["marker_metrics_df"].sort_values("threshold").copy()
        finite_metrics = metrics_df[np.isfinite(metrics_df["threshold"])].copy()
        best_dice = result["best_dice"]
        label = result["label"]
        selected_threshold = float(result["selected_threshold"])
        left_ax, right_ax = axes[row_index]

        left_ax.plot(
            marker_metrics_df["false_positive_rate"],
            marker_metrics_df["sensitivity"],
            color="#1b6ca8",
            linewidth=1.8,
            marker="o",
            markersize=6,
        )
        for index, row in enumerate(marker_metrics_df.itertuples(index=False)):
            if np.isclose(row.threshold, selected_threshold):
                continue
            offset_x = 5
            offset_y = 5 if index % 2 == 0 else -13
            if row.threshold in high_threshold_offsets:
                offset_x, offset_y = high_threshold_offsets[row.threshold]
            left_ax.annotate(
                f"{row.threshold:g}",
                (row.false_positive_rate, row.sensitivity),
                xytext=(offset_x, offset_y),
                textcoords="offset points",
                fontsize=ANNOTATION_FONT_SIZE,
                color="#111111",
            )
        selected_rows = marker_metrics_df[
            np.isclose(marker_metrics_df["threshold"], selected_threshold)
        ]
        if not selected_rows.empty:
            selected_row = selected_rows.iloc[0]
            left_ax.scatter(
                [selected_row["false_positive_rate"]],
                [selected_row["sensitivity"]],
                s=95,
                facecolors="white",
                edgecolors=selected_color,
                linewidths=1.8,
                zorder=5,
            )
            left_ax.annotate(
                f"Selected\nz = {selected_threshold:g}",
                (selected_row["false_positive_rate"], selected_row["sensitivity"]),
                xytext=(16, 2),
                textcoords="offset points",
                fontsize=ANNOTATION_FONT_SIZE,
                color=selected_color,
                ha="left",
                va="center",
                arrowprops={
                    "arrowstyle": "-",
                    "color": selected_color,
                    "linewidth": 0.9,
                    "shrinkA": 2,
                    "shrinkB": 4,
                },
            )
        rank_concordance = result.get("rank_concordance", {})
        kendall_w = rank_concordance.get("kendall_w", np.nan)
        if np.isfinite(kendall_w):
            left_ax.text(
                0.98,
                0.05,
                f"Kendall W = {kendall_w:.3f}",
                transform=left_ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=LEGEND_FONT_SIZE,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "#cccccc",
                    "boxstyle": "round,pad=0.25",
                    "alpha": 0.9,
                },
            )
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
        left_ax.grid(True, linewidth=0.6, alpha=0.25)

        right_ax.plot(
            finite_metrics["threshold"],
            finite_metrics["sensitivity"],
            color="#1b6ca8",
            linewidth=1.8,
            label="Sensitivity",
        )
        right_ax.plot(
            finite_metrics["threshold"],
            finite_metrics["specificity"],
            color="#7d5fb2",
            linewidth=1.8,
            label="Specificity",
        )
        right_ax.plot(
            finite_metrics["threshold"],
            finite_metrics["dice"],
            color="#2f8f5b",
            linewidth=1.8,
            label="Dice overlap",
        )
        for threshold in marker_metrics_df["threshold"]:
            right_ax.axvline(threshold, color="#bbbbbb", linewidth=0.7, alpha=0.28)
        right_ax.axvline(
            selected_threshold,
            color=selected_color,
            linewidth=1.6,
            linestyle="--",
            label=f"Selected threshold (z = {selected_threshold:g})",
        )
        if best_dice is not None and not np.isclose(
            float(best_dice["threshold"]), selected_threshold, atol=0.05
        ):
            best_dice_threshold = float(best_dice["threshold"])
            right_ax.axvline(
                best_dice_threshold, color="#2f8f5b", linewidth=1.3, linestyle=":"
            )
        right_ax.set_xlabel(f"{label} z threshold")
        right_ax.set_ylabel("Metric value")
        right_ax.set_xlim(
            0.0,
            max(float(finite_metrics["threshold"].max()), float(marker_metrics_df["threshold"].max())),
        )
        right_ax.set_ylim(-0.01, 1.01)
        right_ax.grid(True, linewidth=0.6, alpha=0.25)
        legend_loc = "center right" if n_rows > 1 and row_index == n_rows - 1 else "best"
        right_ax.legend(loc=legend_loc, fontsize=LEGEND_FONT_SIZE)

    for ax in np.atleast_1d(axes).ravel():
        ax.xaxis.label.set_fontsize(AXIS_TICK_FONT_SIZE)
        ax.yaxis.label.set_fontsize(AXIS_TICK_FONT_SIZE)
        ax.tick_params(labelsize=AXIS_TICK_FONT_SIZE)
    for text in fig.findobj(match=Text):
        text.set_fontfamily(PAPER_FONT_FAMILY)
    fig.tight_layout()
    with plt.rc_context({"pdf.fonttype": 42, "ps.fonttype": 42}):
        fig.savefig(f"{out_base}_sensitivity_1minus_specificity.png", dpi=300, bbox_inches="tight")
        fig.savefig(f"{out_base}_sensitivity_1minus_specificity.pdf", bbox_inches="tight")
        fig.savefig(f"{out_base}_sensitivity_specificity.png", dpi=300, bbox_inches="tight")
        fig.savefig(f"{out_base}_sensitivity_specificity.pdf", bbox_inches="tight")
        fig.savefig(f"{out_base}_roc.png", dpi=300, bbox_inches="tight")
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


def _candidate_result(
    label,
    slug,
    data,
    standard_data,
    reference_mask,
    analysis_mask,
    marker_thresholds,
    reference_z_threshold,
    selected_threshold,
):
    metrics_df = _threshold_sweep(reference_mask, data, analysis_mask)
    marker_metrics_df = _marker_metrics(reference_mask, data, analysis_mask, marker_thresholds)
    best_dice = _best_row(metrics_df, "dice")

    labels = reference_mask[analysis_mask].astype(np.uint8)
    scores = data[analysis_mask]
    auc_full = float(roc_auc_score(labels, scores))
    full_fpr, full_tpr, _ = roc_curve(labels, scores)
    return {
        "label": label,
        "slug": slug,
        "data": data,
        "metrics_df": metrics_df,
        "marker_metrics_df": marker_metrics_df,
        "best_dice": best_dice,
        "selected_threshold": float(selected_threshold),
        "auc_full": auc_full,
        "full_fpr": full_fpr,
        "full_tpr": full_tpr,
        "partial_auc_positive_thresholds": _positive_threshold_auc(metrics_df),
        "positive_voxels_in_analysis_mask": int(np.count_nonzero(analysis_mask & (data > 0))),
        "rank_concordance": _kendall_w_between_maps(
            standard_data,
            data,
            analysis_mask
            & (
                (standard_data >= float(reference_z_threshold))
                | (data >= float(selected_threshold))
            ),
            reference_z_threshold,
            selected_threshold,
        ),
    }


def _with_candidate_column(df, label):
    out = df.copy()
    out.insert(0, "candidate", label)
    return out


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Sweep positive thresholds in GLMsingle z maps against a "
            "standard-GLM z-threshold gold-standard map."
        )
    )
    parser.add_argument("--standard-glm", type=Path, default=DEFAULT_STANDARD_GLM)
    parser.add_argument("--typea-glm", type=Path, default=DEFAULT_TYPEA_GLM)
    parser.add_argument("--typed-glm", type=Path, default=DEFAULT_TYPED_GLM)
    parser.add_argument("--weight-map", type=Path, default=DEFAULT_WEIGHT_MAP)
    parser.add_argument("--out-base", type=Path, default=DEFAULT_OUT_BASE)
    parser.add_argument("--reference-z-threshold", type=float, default=DEFAULT_REFERENCE_Z_THRESHOLD)
    parser.add_argument(
        "--selected-threshold",
        type=float,
        default=DEFAULT_SELECTED_THRESHOLD,
        help="Rounded Type A z threshold to highlight as the selected paper threshold.",
    )
    parser.add_argument(
        "--typed-selected-threshold",
        type=float,
        default=DEFAULT_TYPED_SELECTED_THRESHOLD,
        help="Rounded Type D z threshold to highlight as the selected paper threshold.",
    )
    parser.add_argument(
        "--marker-thresholds",
        type=float,
        nargs="+",
        default=list(DEFAULT_MARKER_THRESHOLDS),
        help="Candidate z thresholds to mark explicitly on the sensitivity vs 1-specificity plot.",
    )
    parser.add_argument(
        "--include-typed-row",
        action="store_true",
        help="Add a second plot row and outputs for GLMsingle Type D.",
    )
    parser.add_argument(
        "--analysis-mask",
        choices=("finite-both", "typea-nonzero", "typea-positive", "heatmap-support"),
        default="finite-both",
        help=(
            "Voxel set used for TP/FP/TN/FN counts. Default uses every voxel finite in both maps. "
            "heatmap-support matches the support mask in compare_glm_region_highlights.py."
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
    typed_img = None
    typed_data = None
    if args.include_typed_row or args.analysis_mask == "heatmap-support":
        typed_img = _load_img(args.typed_glm)
        _check_same_grid(standard_img, typed_img)
        typed_data = np.asarray(typed_img.get_fdata(), dtype=float)

    if args.analysis_mask == "heatmap-support":
        weight_img = _load_img(args.weight_map)
        _check_same_grid(standard_img, weight_img)
        weight_data = np.asarray(weight_img.get_fdata(), dtype=float)
        analysis_mask = _heatmap_support_mask(standard_data, typea_data, typed_data, weight_data)
        analysis_mask &= np.isfinite(standard_data) & np.isfinite(typea_data)
    else:
        analysis_mask = _analysis_mask(standard_data, typea_data, args.analysis_mask)
    if args.include_typed_row:
        analysis_mask &= np.isfinite(typed_data)
    if not np.any(analysis_mask):
        raise RuntimeError("The analysis mask is empty.")

    reference_mask = analysis_mask & (standard_data >= float(args.reference_z_threshold))
    if not np.any(reference_mask):
        raise RuntimeError("The reference gold-standard map has no suprathreshold voxels.")
    if np.all(reference_mask[analysis_mask]):
        raise RuntimeError("The analysis mask contains no reference-negative voxels.")

    selected_threshold = float(args.selected_threshold)
    typed_selected_threshold = float(args.typed_selected_threshold)
    base_marker_thresholds = {float(threshold) for threshold in args.marker_thresholds}
    candidates = [("GLMsingle Type A", "typeA", typea_data, selected_threshold)]
    if args.include_typed_row:
        candidates.append(("GLMsingle Type D", "typeD", typed_data, typed_selected_threshold))
    candidate_results = [
        _candidate_result(
            label,
            slug,
            data,
            standard_data,
            reference_mask,
            analysis_mask,
            sorted(base_marker_thresholds | {candidate_selected_threshold}),
            args.reference_z_threshold,
            candidate_selected_threshold,
        )
        for label, slug, data, candidate_selected_threshold in candidates
    ]
    typea_result = candidate_results[0]

    out_base = args.out_base
    out_base.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = Path(f"{out_base}_metrics.csv")
    marker_metrics_path = Path(f"{out_base}_marker_threshold_metrics.csv")
    rank_concordance_path = Path(f"{out_base}_kendall_w.csv")
    summary_path = Path(f"{out_base}_summary.json")
    metrics_df = pd.concat(
        [_with_candidate_column(result["metrics_df"], result["label"]) for result in candidate_results],
        ignore_index=True,
    )
    marker_metrics_df = pd.concat(
        [
            _with_candidate_column(result["marker_metrics_df"], result["label"])
            for result in candidate_results
        ],
        ignore_index=True,
    )
    metrics_df.to_csv(metrics_path, index=False)
    marker_metrics_df.to_csv(marker_metrics_path, index=False)
    rank_concordance_df = pd.DataFrame(
        [
            {
                "candidate": result["label"],
                "models": result["rank_concordance"]["models"].replace(
                    "candidate", result["label"]
                ),
                **{
                    key: value
                    for key, value in result["rank_concordance"].items()
                    if key != "models"
                },
            }
            for result in candidate_results
        ]
    )
    rank_concordance_df.to_csv(rank_concordance_path, index=False)
    _plot_metrics(candidate_results, out_base)

    binary_outputs = {}
    if not args.no_binary_maps:
        reference_path = Path(f"{out_base}_reference_zge{args.reference_z_threshold:g}_binary.nii.gz")
        _save_binary_map(reference_mask, standard_img, reference_path)
        binary_outputs["reference"] = str(reference_path)

        for result in candidate_results:
            if result["best_dice"] is not None:
                threshold = float(result["best_dice"]["threshold"])
                path = Path(f"{out_base}_{result['slug']}_dice_zge{threshold:.6g}_binary.nii.gz")
                _save_binary_map(analysis_mask & (result["data"] >= threshold), standard_img, path)
                binary_outputs[f"{result['slug']}_best_dice"] = str(path)

    candidate_summaries = {
        result["label"]: {
            "positive_voxels_in_analysis_mask": result["positive_voxels_in_analysis_mask"],
            "auc_full_scores": result["auc_full"],
            "partial_auc_positive_thresholds": result["partial_auc_positive_thresholds"],
            "selected_threshold_for_rank_concordance": result["selected_threshold"],
            "rank_concordance": result["rank_concordance"],
            "best_dice": _row_payload(result["best_dice"]),
        }
        for result in candidate_results
    }

    summary = {
        "inputs": {
            "standard_glm": str(args.standard_glm),
            "typea_glm": str(args.typea_glm),
            "typed_glm": str(args.typed_glm)
            if args.include_typed_row or args.analysis_mask == "heatmap-support"
            else None,
            "weight_map": str(args.weight_map) if args.analysis_mask == "heatmap-support" else None,
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
        "auc_full_typea_scores": typea_result["auc_full"],
        "partial_auc_positive_thresholds": typea_result["partial_auc_positive_thresholds"],
        "displayed_marker_thresholds_by_candidate": {
            result["label"]: [
                float(threshold)
                for threshold in result["marker_metrics_df"]["threshold"].sort_values().to_list()
            ]
            for result in candidate_results
        },
        "selected_threshold_for_figure": selected_threshold,
        "typed_selected_threshold_for_figure": typed_selected_threshold
        if args.include_typed_row
        else None,
        "best_dice": _row_payload(typea_result["best_dice"]),
        "candidates": candidate_summaries,
        "outputs": {
            "metrics_csv": str(metrics_path),
            "marker_threshold_metrics_csv": str(marker_metrics_path),
            "kendall_w_csv": str(rank_concordance_path),
            "sensitivity_1minus_specificity_png": f"{out_base}_sensitivity_1minus_specificity.png",
            "sensitivity_1minus_specificity_pdf": f"{out_base}_sensitivity_1minus_specificity.pdf",
            "binary_maps": binary_outputs,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Analysis mask voxels: {summary['analysis_mask']['voxels']:,}")
    print(f"Reference voxels: {summary['reference']['voxels']:,}")
    for result in candidate_results:
        print(f"{result['label']} full-score ROC AUC: {result['auc_full']:.4f}")
        print(
            f"{result['label']} positive-threshold partial ROC AUC: "
            f"{result['partial_auc_positive_thresholds']:.4f}"
        )
        print(
            f"{result['label']} Kendall W vs Standard GLM: "
            f"{result['rank_concordance']['kendall_w']:.4f}"
        )
        if result["best_dice"] is not None:
            best_dice = result["best_dice"]
            print(
                f"{result['label']} best Dice threshold: "
                f"{float(best_dice['threshold']):.6g} "
                f"(sensitivity={float(best_dice['sensitivity']):.4f}, "
                f"specificity={float(best_dice['specificity']):.4f}, "
                f"dice={float(best_dice['dice']):.4f})"
            )
    print(f"Saved {metrics_path}")
    print(f"Saved {marker_metrics_path}")
    print(f"Saved {rank_concordance_path}")
    print(f"Saved {out_base}_sensitivity_1minus_specificity.png")
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
