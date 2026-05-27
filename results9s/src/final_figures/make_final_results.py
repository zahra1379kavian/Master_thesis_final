#!/usr/bin/env python3
"""Regenerate the final thesis figures from packaged analysis-ready inputs."""

from __future__ import annotations

import subprocess
import sys
import warnings
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib import colors, ticker
from scipy import stats
from scipy.stats import gaussian_kde, mannwhitneyu
import statsmodels.formula.api as smf


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = ROOT / "results"

SELECTED_COLOR = "#C23B4B"
NONSELECTED_COLOR = "#4C78A8"
NULL_COLOR = "#9EC3D8"
REFERENCE_COLOR = "#7A7A7A"


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _save_pdf_and_png(fig: plt.Figure, pdf_path: Path, dpi: int = 300) -> None:
    _ensure_parent(pdf_path)
    fig.savefig(pdf_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")


def _run_python_script(script: str, *args: str) -> None:
    subprocess.run([sys.executable, str(ROOT / "src" / "final_figures" / script), *args], cwd=ROOT, check=True)


def make_brain_map_figures() -> None:
    anat = DATA / "anatomy" / "MNI152_T1_2mm_brain.nii.gz"
    map_1 = (
        DATA
        / "ablation"
        / "maps"
        / "voxel_weights_mean_foldavg_sub9_ses1_task0.8_bold0.8_beta0.5_smooth0.2_gamma1_bold_thr90.nii.gz"
    )
    map_2 = (
        DATA
        / "ablation"
        / "maps"
        / "voxel_weights_mean_foldavg_sub9_ses1_task0.8_bold0_beta0_smooth0_gamma1_bold_thr90_postcentral_boosted.nii.gz"
    )

    _run_python_script(
        "plot_single_network_paper.py",
        "--map",
        str(map_1),
        "--anat",
        str(anat),
        "--out-png",
        str(RESULTS / "ablation" / "map1_multiplane_contour_all_regions(main).png"),
        "--out-pdf",
        str(RESULTS / "ablation" / "map1_multiplane_contour_all_regions(main).pdf"),
        "--out-eps",
        "",
    )
    _run_python_script(
        "plot_two_networks.py",
        "--map-1",
        str(map_1),
        "--map-2",
        str(map_2),
        "--anat",
        str(anat),
        "--out",
        str(RESULTS / "ablation" / "two_networks_overlay.html"),
        "--out-paper-png",
        str(RESULTS / "ablation" / "two_networks_overlay(main).png"),
        "--out-paper-pdf",
        str(RESULTS / "ablation" / "two_networks_overlay(main).pdf"),
        "--paper-slices",
        "9",
        "--paper-only",
    )


def _ordered_labels_with_pinned_first(summary: pd.DataFrame, label_col: str) -> list[str]:
    sorted_summary = summary.sort_values(["score_mean", "rank"], ascending=[True, True]).reset_index(drop=True)
    labels = sorted_summary[label_col].tolist()
    pinned_mask = (
        np.isclose(sorted_summary["task"].to_numpy(), 0.8)
        & np.isclose(sorted_summary["bold"].to_numpy(), 0.8)
        & np.isclose(sorted_summary["beta"].to_numpy(), 0.5)
        & np.isclose(sorted_summary["smooth"].to_numpy(), 0.2)
    )
    if not np.any(pinned_mask):
        return labels
    pinned_label = sorted_summary.loc[pinned_mask, label_col].tolist()[0]
    return [pinned_label] + [label for label in labels if label != pinned_label]


def _compact_objective_labels(summary_ordered: pd.DataFrame, label_col: str) -> list[str]:
    param_keys = list(
        zip(
            summary_ordered["task"],
            summary_ordered["bold"],
            summary_ordered["beta"],
            summary_ordered["smooth"],
            summary_ordered["gamma"],
        )
    )
    totals: dict[tuple[float, float, float, float, float], int] = {}
    for key in param_keys:
        totals[key] = totals.get(key, 0) + 1
    seen: dict[tuple[float, float, float, float, float], int] = {}
    labels = []
    for _, row in summary_ordered.iterrows():
        key = (row["task"], row["bold"], row["beta"], row["smooth"], row["gamma"])
        seen[key] = seen.get(key, 0) + 1
        gamma_text = "" if np.isclose(float(row["gamma"]), 1.0) else f"\ng={row['gamma']:g}"
        label = f"t={row['task']:g}\nb={row['bold']:g}\nbe={row['beta']:g}\ns={row['smooth']:g}{gamma_text}"
        if totals[key] > 1:
            label = f"{label}\nrun {seen[key]}"
        labels.append(label)
    return labels


def make_objective_evaluation_figure() -> None:
    in_dir = DATA / "ablation" / "objective_evaluation"
    df = pd.read_csv(in_dir / "obj_evaluation_fold_metrics.csv")
    summary = pd.read_csv(in_dir / "obj_evaluation_model_summary.csv")
    label_col = "model_id"
    order = _ordered_labels_with_pinned_first(summary, label_col)
    summary_ordered = summary.set_index(label_col).reindex(order).reset_index()
    labels = _compact_objective_labels(summary_ordered, label_col)
    pos = {label: idx for idx, label in enumerate(order)}

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    ax = axes[0]
    grouped_vals = [df.loc[df[label_col] == label, "evaluation_score"].to_numpy() for label in order]
    box = ax.boxplot(grouped_vals, positions=np.arange(len(order)), patch_artist=True, widths=0.65)
    for idx, patch in enumerate(box["boxes"]):
        patch.set(facecolor="#d62728" if idx == 0 else "#4c72b0", alpha=0.8 if idx == 0 else 0.55)
    for median in box["medians"]:
        median.set(color="black", linewidth=1.6)

    if len(grouped_vals) > 1:
        first = grouped_vals[0]
        p_values = []
        for other in grouped_vals[1:]:
            p_values.append(mannwhitneyu(first, other, alternative="two-sided").pvalue if first.size > 1 and other.size > 1 else np.nan)
        adjusted = [min(1.0, p * len(p_values)) if np.isfinite(p) else np.nan for p in p_values]
        y_base = max(float(np.nanmax(vals)) for vals in grouped_vals if vals.size)
        y_min = min(float(np.nanmin(vals)) for vals in grouped_vals if vals.size)
        y_step = 0.06 * max(y_base - y_min, 1e-6)
        top = y_base
        for idx, (vals, p_adj) in enumerate(zip(grouped_vals[1:], adjusted), start=1):
            if np.isfinite(p_adj) and p_adj < 0.001:
                star = "***"
            elif np.isfinite(p_adj) and p_adj < 0.01:
                star = "**"
            elif np.isfinite(p_adj) and p_adj < 0.05:
                star = "*"
            else:
                star = ""
            star_y = max(float(np.nanmax(vals)) + y_step, y_base + 0.5 * y_step)
            top = max(top, star_y)
            if star:
                ax.text(idx, star_y, star, ha="center", va="bottom", fontsize=14, color="black", fontweight="bold")
        ax.set_ylim(ax.get_ylim()[0], top + 1.8 * y_step)

    ax.text(
        0.01,
        0.01,
        "*  p<0.05\n** p<0.01\n*** p<0.001",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        bbox=dict(facecolor="white", edgecolor="black", linewidth=0.8),
    )
    rng = np.random.default_rng(7)
    for label in order:
        vals = df.loc[df[label_col] == label, "evaluation_score"].to_numpy()
        ax.scatter(np.full(vals.size, pos[label]) + rng.normal(0.0, 0.06, size=vals.size), vals, s=16, c="black", alpha=0.35, linewidths=0)
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Objective S (lower is better)")
    ax.set_title("Fold-level objective distribution")
    ax.grid(alpha=0.25, axis="y")

    ax = axes[1]
    x = np.arange(len(order))
    width = 0.25
    comp_names = [
        ("comp_test_corr_mean", "w_test_corr * norm(|corr_test|)", "#1b9e77"),
        ("comp_corr_stability_mean", "w_corr_stability * norm(corr gap)", "#d95f02"),
        ("comp_loss_stability_mean", "w_loss_stability * norm(loss gap rel)", "#7570b3"),
    ]
    for idx, (col, label, color) in enumerate(comp_names):
        y_vals = summary_ordered[col].to_numpy()
        if col == "comp_test_corr_mean":
            y_vals = -y_vals
        ax.bar(x + (idx - 1) * width, y_vals, width=width, label=label, color=color, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean weighted term")
    ax.legend(loc="upper right", frameon=True)
    ax.grid(alpha=0.25, axis="y")
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.18, top=0.90, wspace=0.22)
    _save_pdf_and_png(fig, RESULTS / "ablation" / "w_balanced_0p309_0p314_0p377" / "obj_evaluation_metric_comparison(main).pdf", dpi=300)
    plt.close(fig)


def _extract_subject_digits(sub_tag: str) -> str:
    match = re.search(r"(\d+)$", str(sub_tag))
    return match.group(1) if match else str(sub_tag)


def _category_sort_key(value: str) -> tuple[int, int | str]:
    digits = _extract_subject_digits(value)
    return (0, int(digits)) if str(digits).isdigit() else (1, str(value))


def _build_subject_color_map(values: list[str]) -> tuple[dict[str, object], list[str]]:
    unique_values = sorted({str(value) for value in values}, key=_category_sort_key)
    palette = list(plt.get_cmap("tab20").colors) + list(plt.get_cmap("tab20b").colors) + list(plt.get_cmap("tab20c").colors)
    step = 11
    spread = [palette[(idx * step) % len(palette)] for idx in range(len(palette))]
    return {value: spread[idx % len(spread)] for idx, value in enumerate(unique_values)}, unique_values


def _mixedlm_projection_effect(paired_df: pd.DataFrame) -> dict[str, float]:
    model_df = paired_df.loc[:, ["sub_tag", "behavior_raw", "projection_raw"]].copy()
    model_df = model_df.rename(columns={"sub_tag": "subject_id"})
    row = {
        "lme_coef_projection_minus_behavior": np.nan,
        "lme_z_projection_minus_behavior": np.nan,
        "lme_p_two_sided": np.nan,
    }
    behavior_long = model_df.loc[:, ["subject_id", "behavior_raw"]].copy()
    behavior_long["signal"] = "Behaviour"
    behavior_long["value"] = behavior_long.pop("behavior_raw")
    projection_long = model_df.loc[:, ["subject_id", "projection_raw"]].copy()
    projection_long["signal"] = "Projection"
    projection_long["value"] = projection_long.pop("projection_raw")
    long_df = pd.concat([behavior_long, projection_long], axis=0, ignore_index=True)
    long_df["signal"] = pd.Categorical(long_df["signal"], categories=["Behaviour", "Projection"], ordered=True)
    fit = None
    for method in ("lbfgs", "powell", "bfgs", "cg", "nm"):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = smf.mixedlm("value ~ signal", data=long_df, groups=long_df["subject_id"], re_formula="1").fit(reml=False, method=method, disp=False)
            break
        except Exception:
            fit = None
    if fit is not None:
        coef = "signal[T.Projection]"
        row["lme_coef_projection_minus_behavior"] = float(fit.params.get(coef, np.nan))
        row["lme_z_projection_minus_behavior"] = float(fit.tvalues.get(coef, np.nan))
        row["lme_p_two_sided"] = float(fit.pvalues.get(coef, np.nan))
    return row


def _plot_paired_box_with_connections(ax: plt.Axes, paired_df: pd.DataFrame, color_map: dict[str, object], y_limits: tuple[float, float]) -> None:
    behavior_values = paired_df["behavior_raw"].to_numpy(dtype=np.float64)
    projection_values = paired_df["projection_raw"].to_numpy(dtype=np.float64)
    box = ax.boxplot(
        [behavior_values, projection_values],
        positions=[0.0, 1.0],
        widths=0.5,
        patch_artist=True,
        showfliers=False,
        showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "black", "markeredgecolor": "black", "markersize": 4.0},
        zorder=1,
    )
    for patch in box["boxes"]:
        patch.set(facecolor="0.94", edgecolor="0.35", linewidth=1.15)
    for item in box["whiskers"] + box["caps"]:
        item.set(color="0.4", linewidth=1.0)
    for median in box["medians"]:
        median.set(color="0.1", linewidth=1.6)

    rng = np.random.default_rng(140)
    jitter_base = rng.uniform(-0.12, 0.12, size=behavior_values.size)
    x_behavior = jitter_base + rng.uniform(-0.03, 0.03, size=behavior_values.size)
    x_projection = 1.0 + jitter_base + rng.uniform(-0.03, 0.03, size=behavior_values.size)
    y_min, y_max = y_limits
    for x0, x1, y0, y1, group_value in zip(x_behavior, x_projection, behavior_values, projection_values, paired_df["sub_tag"].astype(str)):
        color = color_map[group_value]
        y0_plot = float(np.clip(y0, y_min, y_max))
        y1_plot = float(np.clip(y1, y_min, y_max))
        marker0 = "^" if y0 > y_max else "v" if y0 < y_min else "o"
        marker1 = "^" if y1 > y_max else "v" if y1 < y_min else "o"
        ax.plot([x0, x1], [y0_plot, y1_plot], color=color, linewidth=0.85, alpha=0.28, zorder=2)
        ax.scatter([x0], [y0_plot], s=34 if marker0 != "o" else 30, color=color, alpha=0.9, edgecolors="0.15", linewidths=0.3, marker=marker0, zorder=3)
        ax.scatter([x1], [y1_plot], s=34 if marker1 != "o" else 30, color=color, alpha=0.9, edgecolors="0.15", linewidths=0.3, marker=marker1, zorder=3)
    ax.set_xlim(-0.45, 1.45)
    ax.axhline(0.0, color="0.55", linestyle=":", linewidth=0.9, zorder=0)
    ax.set_xticks([0.0, 1.0])
    ax.set_xticklabels(["Behaviour", "Projection"])
    ax.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.5)


def make_behavior_projection_figure() -> None:
    metric_df = pd.read_csv(DATA / "behavior_vs_bold" / "projection_behavior_run_metrics.csv")
    projection_col = "adjacent_diff_ratio_sum_projection"
    behavior_col = "adjacent_diff_ratio_sum_behavior_col2"
    paired_df = metric_df.loc[np.isfinite(metric_df[projection_col]) & np.isfinite(metric_df[behavior_col])].copy()
    paired_df = paired_df.loc[paired_df["sub_tag"].astype(str).map(lambda value: _extract_subject_digits(value) != "017")].reset_index(drop=True)
    paired_df["projection_raw"] = paired_df[projection_col].to_numpy(dtype=np.float64)
    paired_df["behavior_raw"] = paired_df[behavior_col].to_numpy(dtype=np.float64)
    finite_values = np.concatenate([paired_df["behavior_raw"].to_numpy(dtype=np.float64), paired_df["projection_raw"].to_numpy(dtype=np.float64)])
    q_low, q_high = np.percentile(finite_values[np.isfinite(finite_values)], [2.0, 98.0])
    pad = 0.18 * (q_high - q_low) if q_high > q_low else 0.55
    y_limits = (float(q_low - pad), float(q_high + pad))
    color_map, category_values = _build_subject_color_map(paired_df["sub_tag"].astype(str).tolist())
    lme = _mixedlm_projection_effect(paired_df)

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    _plot_paired_box_with_connections(ax, paired_df, color_map, y_limits)
    ax.set_ylim(y_limits)
    ax.set_ylabel("Variability")
    if np.isfinite(lme["lme_p_two_sided"]) and np.isfinite(lme["lme_z_projection_minus_behavior"]):
        lme_text = (
            "LME (Projection-Behaviour)\n"
            f"p={lme['lme_p_two_sided']:.3g}, "
            f"z={lme['lme_z_projection_minus_behavior']:.3g}, "
            f"beta={lme['lme_coef_projection_minus_behavior']:.3g}"
        )
    else:
        lme_text = "LME (Projection-Behaviour)\nfit unavailable"
    ax.text(0.70, 0.98, lme_text, transform=ax.transAxes, ha="center", va="top", fontsize=8.3, bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "alpha": 0.8, "edgecolor": "0.75"})
    handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=color_map[value], markeredgecolor="0.25", markersize=5.5, label=str(value))
        for value in category_values
    ]
    ax.legend(handles=handles, title="Subject", loc="center left", bbox_to_anchor=(1.02, 0.42), fontsize=7.5, title_fontsize=8.5, frameon=True, ncol=2, borderaxespad=0.4, handletextpad=0.35, columnspacing=0.8, labelspacing=0.25)
    fig.tight_layout(rect=(0.0, 0.0, 0.74, 1.0))
    _save_pdf_and_png(fig, RESULTS / "behave_vs_bold" / "projection_behavior_subject_panel(main).pdf", dpi=220)
    plt.close(fig)


def make_connectivity_distribution_figure() -> None:
    pairwise = pd.read_csv(DATA / "connectivity" / "roi_edge_network" / "pairwise_metric_values_all_metrics.csv")
    subset = pairwise.loc[
        (pairwise["connectivity_metric"] == "mutual_information_ksg")
        & (pairwise["comparison_metric"] == "laplacian_spectral_distance_signed")
        & (~pairwise["same_subject"].astype(bool))
    ].copy()
    class_order = [("OFF-OFF", "off-off"), ("ON-ON", "on-on"), ("OFF-ON", "off-on")]
    class_positions = {name: float(idx * 1.15) for idx, (name, _) in enumerate(class_order)}
    groups = [(name, subset.loc[subset["pair_label"] == key, "raw_score"].to_numpy(dtype=np.float64)) for name, key in class_order]
    colors_by_name = {"OFF-OFF": "#4c78a8", "ON-ON": "#e9a3a3", "OFF-ON": "#54a24b"}
    plot_data = [values for _, values in groups]
    positions = [class_positions[name] for name, _ in groups]

    fig, ax = plt.subplots(figsize=(5.2, 4.1))
    box = ax.boxplot(plot_data, positions=positions, widths=0.56, patch_artist=True, flierprops={"markeredgecolor": "#444444", "markerfacecolor": "#444444", "markersize": 2.5})
    for patch, (name, _) in zip(box["boxes"], groups):
        patch.set_facecolor(colors_by_name[name])
        patch.set_alpha(0.55)
    rng = np.random.default_rng(0)
    for pos, values in zip(positions, plot_data):
        ax.scatter(rng.normal(loc=pos, scale=0.04, size=values.size), values, s=14, alpha=0.55, color="black", linewidths=0.0)
    ax.set_xticks(positions)
    ax.set_xticklabels([name for name, _ in groups])
    ax.set_xlim(min(positions) - 0.48, max(positions) + 0.48)
    ax.set_ylabel("Laplacian spectral distance")
    finite = [values[np.isfinite(values)] for values in plot_data if np.isfinite(values).any()]
    y_max = max(float(np.max(values)) for values in finite)
    y_min = min(float(np.min(values)) for values in finite)
    y_span = max(y_max - y_min, 1e-6)
    ax.set_ylim(y_min - 0.05 * y_span, y_max + 0.30 * y_span)
    fig.tight_layout()
    _save_pdf_and_png(fig, RESULTS / "connectivity" / "roi_edge_network" / "mutual_information_ksg" / "cross_subject_only_laplacian_spectral_distance_signed_distribution.pdf", dpi=190)
    plt.close(fig)


def _display_edge_label(label: str) -> str:
    replacements = (
        (" (relative)", ""),
        (" (Control & monitoring)", ""),
        ("Dorsolateral Prefrontal Cortex", "DLPFC"),
        ("Inferior Frontal Gyrus", "IFG"),
        ("Insular / Opercular Cortex", "Insular/Opercular"),
        ("Other Cerebral Cortex", "Other ctx"),
        ("Basal Ganglia", "Basal ganglia"),
        ("Unassigned Active Voxels", "Unassigned"),
    )
    out = str(label)
    for old, new in replacements:
        out = out.replace(old, new)
    return out


def make_top_edges_heatmap() -> None:
    heatmap = pd.read_csv(DATA / "connectivity" / "roi_edge_network" / "mutual_information_ksg" / "top_edges_subject_delta_heatmap.csv")
    subjects = heatmap.iloc[:, 0].astype(str).tolist()
    edge_labels = heatmap.columns[1:].tolist()
    values = heatmap.iloc[:, 1:].to_numpy(dtype=np.float64)
    finite = values[np.isfinite(values)]
    abs_finite = np.abs(finite)
    vmax = float(np.percentile(abs_finite, 99.0)) if abs_finite.size else 1e-6
    positive_abs = abs_finite[abs_finite > 0.0]
    linthresh = float(np.percentile(positive_abs, 25.0)) if positive_abs.size else vmax * 1e-3
    linthresh = max(linthresh, vmax * 1e-6)
    linthresh = min(linthresh, vmax * 0.2)
    norm = colors.SymLogNorm(linthresh=linthresh, linscale=0.8, vmin=-vmax, vmax=vmax, base=10.0)

    fig_width = max(14.0, 0.72 * len(edge_labels) + 3.0)
    fig_height = max(6.4, 0.45 * len(subjects) + 2.4)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    cmap = matplotlib.colormaps["RdBu_r"].copy()
    cmap.set_bad("#f2f2f2")
    im = ax.imshow(values, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(np.arange(len(edge_labels), dtype=int))
    ax.set_yticks(np.arange(len(subjects), dtype=int))
    ax.set_xticklabels([_display_edge_label(label) for label in edge_labels], rotation=52, ha="right", rotation_mode="anchor", fontsize=8)
    ax.set_yticklabels(subjects, fontsize=10)
    ax.set_ylabel("Subject", fontsize=12)
    ax.set_xticks(np.arange(-0.5, len(edge_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(subjects), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.5, alpha=0.55)
    ax.tick_params(which="minor", bottom=False, left=False)
    cbar = fig.colorbar(im, ax=ax, shrink=0.86, pad=0.02)
    cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{float(x) * 1e7:.2g}"))
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label("Edge delta", fontsize=12)
    fig.tight_layout()
    _save_pdf_and_png(fig, RESULTS / "connectivity" / "roi_edge_network" / "mutual_information_ksg" / "top_edges_subject_delta_heatmap.pdf", dpi=180)
    plt.close(fig)


SESSION_RE = re.compile(r"^GVS(\d+)$", re.IGNORECASE)
SUBJECT_RE = re.compile(r"^sub-pd(\d+)$", re.IGNORECASE)


def _subject_sort_key(subject: str) -> tuple[int, str]:
    match = SUBJECT_RE.match(str(subject))
    return (int(match.group(1)), str(subject)) if match else (10**9, str(subject))


def _session_sort_key(label: str) -> tuple[int, str]:
    match = SESSION_RE.match(str(label))
    return (int(match.group(1)), str(label)) if match else (10**9, str(label))


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.fillna(False).astype(str).str.casefold().isin({"true", "1", "yes"})


def _gvs_counts(stats_df: pd.DataFrame, subjects: list[str], sessions: list[str]) -> np.ndarray:
    significant = stats_df.loc[_as_bool(stats_df["significant_fdr"])].copy()
    significant["subject"] = significant["subject"].astype(str)
    significant["target_condition_label"] = significant["target_condition_label"].astype(str)
    significant["roi_label"] = significant["roi_label"].astype(str)
    roi_sets = {(subject, session): set() for subject in subjects for session in sessions}
    for (subject, session), cell_df in significant.groupby(["subject", "target_condition_label"], dropna=False, observed=False, sort=False):
        key = (str(subject), str(session))
        if key in roi_sets:
            roi_sets[key] = set(cell_df["roi_label"].tolist())
    return np.array([[len(roi_sets[(subject, session)]) for session in sessions] for subject in subjects], dtype=float)


def _draw_gvs_panel(ax: plt.Axes, counts: np.ndarray, subjects: list[str], sessions: list[str], cmap_name: str, vmax: int, show_ylabel: bool) -> plt.AxesImage:
    image = ax.imshow(counts, aspect="auto", interpolation="nearest", cmap=cmap_name, vmin=0, vmax=max(1, vmax))
    if show_ylabel:
        ax.set_ylabel("Subject", fontsize=11)
    ax.set_xticks(np.arange(len(sessions)))
    ax.set_xticklabels(sessions, fontsize=10)
    ax.set_yticks(np.arange(len(subjects)))
    ax.set_yticklabels(subjects, fontsize=10)
    ax.set_xticks(np.arange(-0.5, len(sessions), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(subjects), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.1)
    ax.tick_params(which="minor", bottom=False, left=False)
    threshold = max(1.0, 0.62 * float(max(1, vmax)))
    for row_idx in range(len(subjects)):
        for col_idx in range(len(sessions)):
            value = int(counts[row_idx, col_idx])
            if value == 0:
                text, color = "-", "#7a7a7a"
            else:
                text, color = str(value), "white" if value >= threshold else "black"
            ax.text(col_idx, row_idx, text, ha="center", va="center", fontsize=10, color=color)
    return image


def make_gvs_burden_heatmaps() -> None:
    base = DATA / "connectivity" / "gvs" / "without_unassigned"
    specs = [
        ("off_condition_minus_sham_off_roi_mean_delta", "Blues"),
        ("on_condition_minus_sham_on_roi_mean_delta", "Oranges"),
    ]
    stats_by_prefix = {
        prefix: pd.read_csv(base / f"{prefix}_ttest_stats_by_subject_long.csv")
        for prefix, _ in specs
    }
    sessions = sorted(
        {
            str(label)
            for stats_df in stats_by_prefix.values()
            for label in stats_df["target_condition_label"].dropna().unique().tolist()
            if str(label).casefold() != "sham"
        },
        key=_session_sort_key,
    )
    prepared = []
    max_count = 0
    out_dir = RESULTS / "connectivity" / "GVS_effects" / "gvs_similarity_hemi" / "roi_condition_reference_deltas" / "plots" / "without_unassigned"
    out_dir.mkdir(parents=True, exist_ok=True)
    for prefix, cmap_name in specs:
        stats_df = stats_by_prefix[prefix]
        subjects = sorted(stats_df["subject"].astype(str).unique().tolist(), key=_subject_sort_key)
        counts = _gvs_counts(stats_df, subjects, sessions)
        max_count = max(max_count, int(np.nanmax(counts)) if counts.size else 0)
        prepared.append((subjects, counts, cmap_name))
        pd.DataFrame(counts.astype(int), index=subjects, columns=sessions).rename_axis("subject").to_csv(out_dir / f"{prefix}_subject_session_roi_burden_heatmap.csv")

    fig_w = max(15.0, 1.7 * len(sessions) + 4.0)
    fig_h = max(7.4, max(0.48 * len(subjects) for subjects, _, _ in prepared) + 2.1)
    fig, axes = plt.subplots(1, len(prepared), figsize=(fig_w, fig_h))
    for idx, (ax, (subjects, counts, cmap_name)) in enumerate(zip(np.atleast_1d(axes), prepared)):
        im = _draw_gvs_panel(ax, counts, subjects, sessions, cmap_name, max_count, show_ylabel=idx == 0)
        ax.set_xlabel("GVS session", fontsize=11)
        cbar = fig.colorbar(im, ax=ax, shrink=0.84, pad=0.02)
        if idx != 0:
            cbar.set_label("No. of significant ROIs", fontsize=10)
        cbar.ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        cbar.ax.tick_params(labelsize=9)
    fig.tight_layout()
    _save_pdf_and_png(fig, out_dir / "off_on_condition_minus_sham_subject_session_roi_burden_heatmaps.pdf", dpi=300)
    plt.close(fig)


def _subsample_for_kde(values: np.ndarray, kde_max_points: int, rng: np.random.Generator) -> np.ndarray:
    if values.size <= kde_max_points:
        return values
    chosen = rng.choice(values.size, size=int(kde_max_points), replace=False)
    return np.asarray(values[chosen], dtype=np.float64)


def _density_curve(values: np.ndarray, rng: np.random.Generator, kde_max_points: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < 2 or np.allclose(values, values[0]):
        hist, edges = np.histogram(values, bins=min(80, max(10, values.size // 20)), density=True)
        return 0.5 * (edges[:-1] + edges[1:]), hist
    sampled = _subsample_for_kde(values, kde_max_points, rng)
    lower, upper = float(np.min(sampled)), float(np.max(sampled))
    if np.isclose(lower, upper):
        lower -= 1e-6
        upper += 1e-6
    pad = 0.05 * (upper - lower)
    grid = np.linspace(lower - pad, upper + pad, 512)
    return grid, gaussian_kde(sampled).evaluate(grid)


def _plot_prevalence_panel(ax: plt.Axes, percentile_labels: np.ndarray, prevalence_ratios: np.ndarray) -> None:
    bar_colors = [SELECTED_COLOR if value >= 1.0 else NONSELECTED_COLOR for value in prevalence_ratios]
    ax.bar([f"{int(round(p))}%" for p in percentile_labels], prevalence_ratios, color=bar_colors, edgecolor="white", linewidth=1.0)
    ax.axhline(1.0, linestyle="--", color=REFERENCE_COLOR, linewidth=1.6)
    ax.set_ylabel("Below-threshold fraction ratio\n(Selected / Non-selected)")
    ax.set_xlabel("Variability Percentile Threshold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_resample_panel(ax: plt.Axes, selected_mean: float, resampled_means: np.ndarray, metric_name: str) -> None:
    ax.hist(resampled_means, bins=40, density=True, color=NULL_COLOR, edgecolor="white", alpha=0.95)
    resample_mean = float(np.mean(resampled_means))
    ci_low, ci_high = np.percentile(resampled_means, [2.5, 97.5])
    ax.axvline(selected_mean, linestyle="--", linewidth=2.0, color=SELECTED_COLOR, label=f"Selected mean = {selected_mean:.4g}")
    ax.axvline(resample_mean, linestyle="--", linewidth=1.6, color=REFERENCE_COLOR, label=f"Resample mean = {resample_mean:.4g}")
    ax.axvline(ci_low, linestyle="--", linewidth=1.4, color=REFERENCE_COLOR)
    ax.axvline(ci_high, linestyle="--", linewidth=1.4, color=REFERENCE_COLOR, label=f"95% CI = [{ci_low:.4g}, {ci_high:.4g}]")
    x_min = float(min(np.min(resampled_means), selected_mean, ci_low))
    x_max = float(max(np.max(resampled_means), selected_mean, ci_high))
    x_span = x_max - x_min
    if not np.isfinite(x_span) or np.isclose(x_span, 0.0):
        x_span = max(abs(x_max), 1.0)
    x_left = x_min - 0.12 * x_span
    x_right = x_max + 0.06 * x_span
    ax.set_xlim(x_left, x_right)
    ax.set_xlabel(metric_name)
    ax.set_ylabel("Density")
    ci_high_frac = float((ci_high - x_left) / max(x_right - x_left, 1e-12))
    legend_anchor_x = float(np.clip(ci_high_frac - 0.02, 0.58, 0.86))
    ax.legend(frameon=True, fontsize=9, loc="upper right", bbox_to_anchor=(legend_anchor_x, 0.98))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def make_trial_variability_hypothesis_figure() -> None:
    data = np.load(DATA / "hypothesis" / "trial_variability_hypothesis_analysis_data.npz", allow_pickle=True)
    selected = data["selected_norm_diff"].astype(np.float64)
    resampled = data["resampled_nonselected_norm_diff_means"].astype(np.float64)
    prevalence = data["norm_diff_prevalence_ratios"].astype(np.float64)
    percentiles = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90], dtype=np.float64)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    _plot_prevalence_panel(axes[0], percentiles, prevalence)
    _plot_resample_panel(axes[1], float(np.mean(selected)), resampled, "Normalised |consecutive trial beta difference|")
    _save_pdf_and_png(fig, RESULTS / "prove_hypothesis" / "trial_variability_hypothesis_norm_diff_cd(main).pdf", dpi=300)
    plt.close(fig)


SELECTION_DISPLAY_LABELS = {"control": "Motor reference", "selected": "Target network"}


def _safe_float(value) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def _fmt_signed(value: float | None, digits: int = 3) -> str:
    return "nan" if value is None or not np.isfinite(float(value)) else f"{float(value):+.{digits}f}"


def _fmt_float(value: float | None, digits: int = 3) -> str:
    return "nan" if value is None or not np.isfinite(float(value)) else f"{float(value):.{digits}f}"


def _sem(values: pd.Series) -> float:
    arr = values.to_numpy(dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return 0.0 if arr.size <= 1 else float(arr.std(ddof=1) / np.sqrt(arr.size))


def make_state_selection_stability_figure() -> None:
    base = DATA / "hypothesis" / "state_selection_stability_followup"
    session_df = pd.read_csv(base / "subject_session_state_selection_means.csv")
    paired_df = pd.read_csv(base / "paired_delta_summary.csv")
    paired_summary = paired_df.loc[paired_df["metric"] == "norm_diff_mean"].iloc[0].to_dict()
    long_df = pd.concat(
        [
            session_df.loc[:, ["subject", "session", "state", "n_runs", "selected_norm_diff_mean"]].rename(columns={"selected_norm_diff_mean": "value"}).assign(selection="selected"),
            session_df.loc[:, ["subject", "session", "state", "n_runs", "control_norm_diff_mean"]].rename(columns={"control_norm_diff_mean": "value"}).assign(selection="control"),
        ],
        ignore_index=True,
    )
    long_df = long_df.loc[np.isfinite(long_df["value"].to_numpy(dtype=np.float64))].copy()
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 4.2), sharey=True)
    colors_map = {"control": "#3d6f8e", "selected": "#d65a6f"}
    state_order = ["off", "on"]
    x_positions = np.array([0.0, 1.0], dtype=np.float64)
    panel_stats = {
        "control": {"delta": paired_summary["D_ctl_mean_on_minus_off"], "p": paired_summary["control_state_p_two_sided"]},
        "selected": {"delta": paired_summary["D_sel_mean_on_minus_off"], "p": paired_summary["selected_state_p_two_sided"]},
    }
    values = long_df["value"].to_numpy(dtype=np.float64)
    values = values[np.isfinite(values)]
    y_limits = None
    if values.size:
        y_min, y_max = float(values.min()), float(values.max())
        pad = max(0.2, 0.06 * (y_max - y_min if y_max > y_min else 1.0))
        y_limits = (y_min - pad, y_max + pad)
    for ax, selection in zip(axes, ("control", "selected")):
        subset = long_df.loc[long_df["selection"] == selection].copy()
        for _, subject_df in subset.groupby("subject", sort=True):
            subject_df = subject_df.set_index("state").reindex(state_order)
            ys = subject_df["value"].to_numpy(dtype=np.float64)
            finite = np.isfinite(ys)
            if np.any(finite):
                ax.plot(x_positions[finite], ys[finite], color="0.82", alpha=0.65, linewidth=0.9, zorder=1)
                ax.scatter(x_positions[finite], ys[finite], s=22, facecolor="white", edgecolor=colors_map[selection], linewidth=0.9, alpha=0.9, zorder=2)
        means = subset.groupby("state")["value"].mean().reindex(state_order)
        sems = subset.groupby("state")["value"].apply(_sem).reindex(state_order).fillna(0.0)
        ax.errorbar(x_positions, means.to_numpy(dtype=np.float64), yerr=sems.to_numpy(dtype=np.float64), color=colors_map[selection], linewidth=2.6, marker="o", markersize=6.5, markerfacecolor=colors_map[selection], markeredgecolor="white", markeredgewidth=0.9, capsize=3.5, elinewidth=1.5, zorder=4)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(["OFF", "ON"])
        ax.set_title(SELECTION_DISPLAY_LABELS[selection], fontsize=12, pad=8)
        ax.set_xlabel("Medication state")
        ax.grid(axis="y", color="0.9", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if y_limits is not None:
            ax.set_ylim(*y_limits)
        ax.text(0.04, 0.96, f"ON - OFF = {_fmt_signed(panel_stats[selection]['delta'])}\npaired p = {_fmt_float(_safe_float(panel_stats[selection]['p']))}", transform=ax.transAxes, ha="left", va="top", fontsize=8.5, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.8", alpha=0.95))
    axes[0].set_ylabel("Mean normalized |consecutive trial beta difference|")
    fig.tight_layout(rect=(0.0, 0.13, 1.0, 1.0))
    interaction_text = (
        "Difference in ON - OFF change\n"
        f"{SELECTION_DISPLAY_LABELS['selected'].lower()} - {SELECTION_DISPLAY_LABELS['control'].lower()} = "
        f"{_fmt_signed(paired_summary['interaction_mean_D_sel_minus_D_ctl'])}; "
        f"paired p = {_fmt_float(_safe_float(paired_summary['interaction_p_two_sided']))}"
    )
    fig.text(0.5, 0.025, interaction_text, ha="center", va="bottom", fontsize=9.0, color="0.25", linespacing=1.25)
    _save_pdf_and_png(fig, RESULTS / "prove_hypothesis" / "state_selection_stability_followup" / "norm_diff_mean_subject_state_lines(main).pdf", dpi=400)
    plt.close(fig)


def main() -> None:
    make_objective_evaluation_figure()
    make_brain_map_figures()
    make_behavior_projection_figure()
    make_connectivity_distribution_figure()
    make_top_edges_heatmap()
    make_gvs_burden_heatmaps()
    make_trial_variability_hypothesis_figure()
    make_state_selection_stability_figure()
    print("Final figures regenerated under", RESULTS)


if __name__ == "__main__":
    main()
