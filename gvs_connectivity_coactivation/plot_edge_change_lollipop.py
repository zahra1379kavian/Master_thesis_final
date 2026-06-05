#!/usr/bin/env python3
"""Rank FDR-significant connectivity edges with a signed lollipop plot."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT_ROOT = ROOT / "figures" / "GVS_effects" / "main result"
SIG_CSV = OUT_ROOT / "metric_sensitivity" / "fdr_significant_edge_connectivity_metric_sensitivity.csv"
OUT_DIR = OUT_ROOT / "metric_sensitivity" / "connectogram_reports"

DEFAULT_METRIC = "mutual_info_quantile"
DEFAULT_ANALYSIS_VIEW = "all_subjects_block_pool"
DEFAULT_FDR_SCOPE = "pool=ALL_SUBJECTS_BLOCKS;gvs=ANY_GVS"
DEFAULT_TOP_N = 30

POSITIVE_COLOR = "#d95f02"
NEGATIVE_COLOR = "#2166ac"
PAPER_FONT_FAMILY = "Liberation Sans"
VALUE_FONTSIZE = 9.2
EDGE_LABEL_FONTSIZE = 9.0
AXIS_LABEL_FONTSIZE = 11.6
TICK_LABEL_FONTSIZE = 10.2
TITLE_FONTSIZE = 15.0
SUBTITLE_FONTSIZE = 11.0
LEGEND_FONTSIZE = 10.0

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": [PAPER_FONT_FAMILY, "Arial", "Helvetica", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def clean_slug(text: str) -> str:
    text = str(text)
    text = re.sub(r"[^A-Za-z0-9]+", "_", text)
    return text.strip("_").lower()


def roi_label(roi: str) -> str:
    roi = str(roi)
    if roi.endswith("_L"):
        return f"{roi[:-2].replace('_', ' ')} L"
    if roi.endswith("_R"):
        return f"{roi[:-2].replace('_', ' ')} R"
    return roi.replace("_", " ")


def edge_label(row: pd.Series) -> str:
    return f"{roi_label(row['roi_i'])} - {roi_label(row['roi_j'])}"


def pretty_label(text: str) -> str:
    return str(text).replace("_", " ")


def load_edges(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    if "sig_fdr" in df.columns:
        df = df.loc[df["sig_fdr"].astype(bool)].copy()
    for col in ["mean", "q_fdr", "p_signflip"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["mean", "roi_i", "roi_j"]).copy()
    df["abs_mean"] = df["mean"].abs()
    return df


def selected_edges(df: pd.DataFrame, metric: str, analysis_view: str, fdr_scope: str, top_n: int) -> pd.DataFrame:
    mask = (
        (df["metric"].astype(str) == metric)
        & (df["analysis_view"].astype(str) == analysis_view)
        & (df["fdr_scope"].astype(str) == fdr_scope)
    )
    out = df.loc[mask].copy()
    if out.empty:
        raise ValueError(f"No FDR-significant edges found for {metric} | {analysis_view} | {fdr_scope}")
    return out.sort_values(["abs_mean", "q_fdr", "p_signflip"], ascending=[False, True, True]).head(top_n).copy()


def plot_lollipop(
    df: pd.DataFrame,
    path_base: Path,
    title: str,
    subtitle: str,
    *,
    label_axis: str = "y",
    show_title: bool = True,
) -> None:
    plot_df = df.copy()
    plot_df["display_edge"] = plot_df.apply(edge_label, axis=1)

    if label_axis == "x":
        plot_df = plot_df.reset_index(drop=True)
        n_edges = int(plot_df.shape[0])
        fig_width = max(12.0, 0.42 * n_edges + 2.8)
        fig, ax = plt.subplots(figsize=(fig_width, 6.2))

        x = np.arange(n_edges)
        values = plot_df["mean"].astype(float).to_numpy()
        colors = np.where(values >= 0, POSITIVE_COLOR, NEGATIVE_COLOR)

        ax.axhline(0.0, color="#555555", linewidth=0.9, zorder=0)
        for xi, value, color in zip(x, values, colors, strict=True):
            ax.vlines(xi, 0.0, value, color=color, linewidth=2.0, alpha=0.72, zorder=1)
        ax.scatter(x, values, s=58, color=colors, edgecolor="white", linewidth=0.8, zorder=3)

        max_abs = float(np.nanmax(np.abs(values))) if values.size else 1.0
        if not np.isfinite(max_abs) or max_abs <= 0:
            max_abs = 1.0
        label_pad = max_abs * 0.045
        for xi, value in zip(x, values, strict=True):
            va = "bottom" if value >= 0 else "top"
            y = value + label_pad if value >= 0 else value - label_pad
            ax.text(xi, y, f"{value:+.3f}", va=va, ha="center", fontsize=VALUE_FONTSIZE, color="#222222")

        ax.set_xticks(x)
        ax.set_xticklabels(
            plot_df["display_edge"],
            rotation=62,
            ha="right",
            rotation_mode="anchor",
            fontsize=EDGE_LABEL_FONTSIZE,
        )
        for tick_label, color in zip(ax.get_xticklabels(), colors, strict=True):
            tick_label.set_color(color)
        ax.set_ylabel("Active GVS - sham edge change", fontsize=AXIS_LABEL_FONTSIZE)
        ax.grid(axis="y", color="#E1E1E1", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", length=0)
        ax.tick_params(axis="y", labelsize=TICK_LABEL_FONTSIZE)

        ylim = max_abs * 1.38
        ax.set_ylim(-ylim, ylim)
        if show_title:
            fig.text(0.02, 0.985, title, ha="left", va="top", fontsize=TITLE_FONTSIZE, fontweight="bold")
            fig.text(0.02, 0.953, subtitle, ha="left", va="top", fontsize=SUBTITLE_FONTSIZE, color="#555555")

        legend_handles = [
            Line2D([0], [0], marker="o", color=POSITIVE_COLOR, label="Active GVS > sham", markersize=6, linewidth=2),
            Line2D([0], [0], marker="o", color=NEGATIVE_COLOR, label="Active GVS < sham", markersize=6, linewidth=2),
        ]
        ax.legend(handles=legend_handles, loc="upper right", frameon=False, fontsize=LEGEND_FONTSIZE)
        fig.subplots_adjust(left=0.07, right=0.99, top=0.97 if not show_title else 0.88, bottom=0.42)
        fig.savefig(path_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
        fig.savefig(path_base.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
        return

    plot_df = plot_df.iloc[::-1].reset_index(drop=True)

    n_edges = int(plot_df.shape[0])
    fig_height = max(5.8, 0.34 * n_edges + 1.9)
    fig, ax = plt.subplots(figsize=(9.8, fig_height))

    y = np.arange(n_edges)
    values = plot_df["mean"].astype(float).to_numpy()
    colors = np.where(values >= 0, POSITIVE_COLOR, NEGATIVE_COLOR)

    ax.axvline(0.0, color="#555555", linewidth=0.9, zorder=0)
    for yi, value, color in zip(y, values, colors, strict=True):
        ax.hlines(yi, 0.0, value, color=color, linewidth=2.0, alpha=0.72, zorder=1)
    ax.scatter(values, y, s=58, color=colors, edgecolor="white", linewidth=0.8, zorder=3)

    max_abs = float(np.nanmax(np.abs(values))) if values.size else 1.0
    if not np.isfinite(max_abs) or max_abs <= 0:
        max_abs = 1.0
    label_pad = max_abs * 0.055
    for yi, value in zip(y, values, strict=True):
        ha = "left" if value >= 0 else "right"
        x = value + label_pad if value >= 0 else value - label_pad
        ax.text(x, yi, f"{value:+.3f}", va="center", ha=ha, fontsize=VALUE_FONTSIZE, color="#222222")

    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["display_edge"], fontsize=EDGE_LABEL_FONTSIZE)
    ax.set_xlabel("Active GVS - sham edge change", fontsize=AXIS_LABEL_FONTSIZE)
    ax.grid(axis="x", color="#E1E1E1", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=TICK_LABEL_FONTSIZE)

    xlim = max_abs * 1.30
    ax.set_xlim(-xlim, xlim)
    if show_title:
        fig.text(0.02, 0.985, title, ha="left", va="top", fontsize=TITLE_FONTSIZE, fontweight="bold")
        fig.text(0.02, 0.953, subtitle, ha="left", va="top", fontsize=SUBTITLE_FONTSIZE, color="#555555")

    legend_handles = [
        Line2D([0], [0], marker="o", color=POSITIVE_COLOR, label="Active GVS > sham", markersize=6, linewidth=2),
        Line2D([0], [0], marker="o", color=NEGATIVE_COLOR, label="Active GVS < sham", markersize=6, linewidth=2),
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=False, fontsize=LEGEND_FONTSIZE)
    fig.subplots_adjust(left=0.39, right=0.98, top=0.98 if not show_title else 0.90, bottom=0.08)
    fig.savefig(path_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def output_base(out_dir: Path, metric: str, analysis_view: str, fdr_scope: str, top_n: int) -> Path:
    scope_slug = clean_slug(f"{analysis_view}_{fdr_scope}")
    return out_dir / f"{clean_slug(metric)}__{scope_slug}__top{top_n}_lollipop"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric", default=DEFAULT_METRIC)
    parser.add_argument("--analysis-view", default=DEFAULT_ANALYSIS_VIEW)
    parser.add_argument("--fdr-scope", default=DEFAULT_FDR_SCOPE)
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--input", type=Path, default=SIG_CSV)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--label-axis", choices=("x", "y"), default="y")
    parser.add_argument("--hide-title", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    edges = load_edges(args.input)
    selected = selected_edges(edges, args.metric, args.analysis_view, args.fdr_scope, args.top_n)
    path_base = output_base(args.out_dir, args.metric, args.analysis_view, args.fdr_scope, int(selected.shape[0]))

    title = f"{pretty_label(args.metric)}: top FDR-significant edge changes"
    subtitle = (
        f"{pretty_label(args.analysis_view)} | {pretty_label(args.fdr_scope)} | "
        f"top {selected.shape[0]} ranked by absolute delta"
    )
    plot_lollipop(
        selected,
        path_base,
        title,
        subtitle,
        label_axis=args.label_axis,
        show_title=not args.hide_title,
    )

    selected.to_csv(path_base.with_suffix(".csv"), index=False)
    print(path_base.with_suffix(".png"))
    print(path_base.with_suffix(".pdf"))
    print(path_base.with_suffix(".csv"))


if __name__ == "__main__":
    main()
