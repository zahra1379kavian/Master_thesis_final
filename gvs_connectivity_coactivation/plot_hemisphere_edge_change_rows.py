#!/usr/bin/env python3
"""Plot significant edge changes by within- vs between-hemisphere status."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

DEFAULT_REPORTS = (
    (
        ROOT
        / "figures"
        / "GVS_effects"
        / "main result"
        / "task_activation_z3p1"
        / "metric_sensitivity"
        / "connectogram_reports"
        / "mutual_info_quantile__all_subjects_block_pool_pool_all_subjects_blocks_gvs_any_gvs.png",
        ROOT
        / "figures"
        / "GVS_effects"
        / "GPT"
        / "08_connectivity_coactivation"
        / "task_activation_z3p1"
        / "metric_sensitivity"
        / "fdr_significant_edge_connectivity_metric_sensitivity.csv",
    ),
    (
        ROOT
        / "figures"
        / "GVS_effects"
        / "main result"
        / "metric_sensitivity"
        / "connectogram_reports"
        / "mutual_info_quantile__all_subjects_block_pool_pool_all_subjects_blocks_gvs_any_gvs.png",
        ROOT
        / "figures"
        / "GVS_effects"
        / "GPT"
        / "08_connectivity_coactivation"
        / "metric_sensitivity"
        / "fdr_significant_edge_connectivity_metric_sensitivity.csv",
    ),
)

PAPER_FONT_FAMILY = "Liberation Sans"
INCREASE_COLOR = "#d95f02"
DECREASE_COLOR = "#2166ac"
ROW_ORDER = ("Between hemispheres", "Within hemisphere")
Y_POSITIONS = {"Within hemisphere": 0.0, "Between hemispheres": 1.0}

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


def parse_report_slug(report_png: Path) -> tuple[str, str]:
    if report_png.suffix.lower() != ".png":
        raise ValueError(f"Expected a PNG report path, got: {report_png}")
    if "__" not in report_png.stem:
        raise ValueError(f"Could not parse metric/scope from report name: {report_png.name}")
    metric_slug, scope_slug = report_png.stem.split("__", 1)
    return metric_slug, scope_slug


def roi_side(roi: str) -> str:
    roi = str(roi)
    if roi.endswith("_L"):
        return "L"
    if roi.endswith("_R"):
        return "R"
    return ""


def hemisphere_relation(roi_i: str, roi_j: str) -> str:
    side_i = roi_side(roi_i)
    side_j = roi_side(roi_j)
    if side_i and side_j and side_i == side_j:
        return "Within hemisphere"
    if side_i and side_j and side_i != side_j:
        return "Between hemispheres"
    return "Unclassified"


def load_report_edges(source_csv: Path, report_png: Path) -> pd.DataFrame:
    metric_slug, scope_slug = parse_report_slug(report_png)
    df = pd.read_csv(source_csv, low_memory=False)
    if "sig_fdr" in df.columns:
        df = df.loc[df["sig_fdr"].astype(bool)].copy()

    for col in ("mean", "q_fdr", "p_signflip"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["mean", "roi_i", "roi_j"]).copy()
    df["metric_slug"] = df["metric"].map(clean_slug)
    df["scope_slug"] = [clean_slug(f"{view}_{scope}") for view, scope in zip(df["analysis_view"], df["fdr_scope"], strict=True)]

    matched = df.loc[(df["metric_slug"] == metric_slug) & (df["scope_slug"] == scope_slug)].copy()
    if matched.empty:
        raise ValueError(f"No FDR-significant edges matched {report_png.name} in {source_csv}")

    matched["hemisphere_relation"] = [
        hemisphere_relation(roi_i, roi_j) for roi_i, roi_j in zip(matched["roi_i"], matched["roi_j"], strict=True)
    ]
    unknown = matched.loc[matched["hemisphere_relation"] == "Unclassified", ["roi_i", "roi_j"]]
    if not unknown.empty:
        preview = unknown.head(5).to_dict(orient="records")
        raise ValueError(f"Found edges without _L/_R hemisphere labels: {preview}")

    matched["direction"] = np.where(matched["mean"] > 0, "Improved", "Decrement")
    matched["plot_value"] = matched["mean"].abs()
    matched["abs_mean"] = matched["mean"].abs()
    return matched.sort_values(["direction", "hemisphere_relation", "abs_mean"], ascending=[True, True, True]).copy()


def load_possible_edge_denominators(source_csv: Path, report_png: Path) -> dict[str, int]:
    metric_slug, scope_slug = parse_report_slug(report_png)
    stats_csv = source_csv.with_name("edge_connectivity_metric_sensitivity_stats.csv")
    if not stats_csv.exists():
        raise FileNotFoundError(f"Missing full stats CSV needed for edge-density denominators: {stats_csv}")

    df = pd.read_csv(stats_csv, low_memory=False)
    df = df.dropna(subset=["roi_i", "roi_j"]).copy()
    df["metric_slug"] = df["metric"].map(clean_slug)
    df["scope_slug"] = [clean_slug(f"{view}_{scope}") for view, scope in zip(df["analysis_view"], df["fdr_scope"], strict=True)]

    matched = df.loc[(df["metric_slug"] == metric_slug) & (df["scope_slug"] == scope_slug)].copy()
    if matched.empty:
        raise ValueError(f"No full-stat edge rows matched {report_png.name} in {stats_csv}")

    matched["hemisphere_relation"] = [
        hemisphere_relation(roi_i, roi_j) for roi_i, roi_j in zip(matched["roi_i"], matched["roi_j"], strict=True)
    ]
    unknown = matched.loc[matched["hemisphere_relation"] == "Unclassified", ["roi_i", "roi_j"]]
    if not unknown.empty:
        preview = unknown.head(5).to_dict(orient="records")
        raise ValueError(f"Found denominator edges without _L/_R hemisphere labels: {preview}")

    return {relation: int((matched["hemisphere_relation"] == relation).sum()) for relation in ROW_ORDER}


def add_panel(ax: plt.Axes, edges: pd.DataFrame, title: str, color: str, x_max: float, denominators: dict[str, int]) -> None:
    ax.axvline(0.0, color="#555555", linewidth=0.9, zorder=0)
    rng = np.random.default_rng(20260610)

    for relation in ROW_ORDER:
        relation_edges = edges.loc[edges["hemisphere_relation"] == relation].copy()
        y_center = Y_POSITIONS[relation]
        if not relation_edges.empty:
            jitter = rng.uniform(-0.11, 0.11, size=relation_edges.shape[0])
            x = relation_edges["plot_value"].to_numpy(dtype=float)
            y = y_center + jitter
            ax.scatter(x, y, s=25, color=color, alpha=0.72, edgecolor="white", linewidth=0.45, zorder=3)
            median_value = float(np.nanmedian(x))
            ax.plot([median_value, median_value], [y_center - 0.18, y_center + 0.18], color="#202020", linewidth=1.25, zorder=4)

        denominator = denominators.get(relation, 0)
        density_label = "NA" if denominator <= 0 else f"{relation_edges.shape[0] / denominator:.1%}"
        ax.text(
            0.985,
            y_center,
            density_label,
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=9.3,
            color="#333333",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
        )

    ax.set_title(title, loc="left", fontsize=12.6, fontweight="bold", color=color, pad=5)
    ax.set_yticks([Y_POSITIONS[label] for label in ROW_ORDER])
    ax.set_yticklabels(ROW_ORDER, fontsize=10.5)
    ax.set_ylim(-0.48, 1.48)
    ax.set_xlim(0.0, x_max)
    ax.grid(axis="x", color="#E1E1E1", linewidth=0.8)
    ax.grid(axis="y", color="#EFEFEF", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=9.8)


def plot_hemisphere_rows(edges: pd.DataFrame, denominators: dict[str, int], path_base: Path) -> None:
    positive = edges.loc[edges["mean"] > 0].copy()
    negative = edges.loc[edges["mean"] < 0].copy()
    if positive.empty or negative.empty:
        raise ValueError("Expected both improved and decrement edges for the requested report.")

    max_value = float(edges["plot_value"].max())
    x_max = max(max_value * 1.12, max_value + 0.005)

    fig, axes = plt.subplots(2, 1, figsize=(7.4, 4.65), sharex=True)
    add_panel(axes[0], positive, "Improved", INCREASE_COLOR, x_max, denominators)
    add_panel(axes[1], negative, "Decrement", DECREASE_COLOR, x_max, denominators)
    fig.text(0.985, 0.982, "Significant-edge density", ha="right", va="top", fontsize=8.8, color="#555555")
    axes[1].set_xlabel("Absolute edge-change value", fontsize=11.2)
    fig.subplots_adjust(left=0.23, right=0.985, top=0.965, bottom=0.135, hspace=0.42)
    fig.savefig(path_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def output_base(report_png: Path) -> Path:
    return report_png.with_name(f"{report_png.stem}__all_significant_hemisphere_edge_changes")


def process_report(report_png: Path, source_csv: Path) -> Path:
    if not report_png.exists():
        raise FileNotFoundError(f"Missing report PNG: {report_png}")
    if not source_csv.exists():
        raise FileNotFoundError(f"Missing source CSV: {source_csv}")

    edges = load_report_edges(source_csv, report_png)
    denominators = load_possible_edge_denominators(source_csv, report_png)
    path_base = output_base(report_png)
    plot_hemisphere_rows(edges, denominators, path_base)
    export_cols = [
        "metric",
        "analysis_view",
        "fdr_scope",
        "edge_id",
        "roi_i",
        "roi_j",
        "edge_label",
        "mean",
        "plot_value",
        "q_fdr",
        "p_signflip",
        "direction",
        "hemisphere_relation",
    ]
    available_cols = [col for col in export_cols if col in edges.columns]
    edges[available_cols].to_csv(path_base.with_suffix(".csv"), index=False)
    return path_base


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        nargs=2,
        action="append",
        metavar=("REPORT_PNG", "SOURCE_CSV"),
        help="Connectogram PNG and matching FDR-significant edge CSV. May be repeated.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    reports = args.report or DEFAULT_REPORTS
    for report_png, source_csv in reports:
        path_base = process_report(Path(report_png), Path(source_csv))
        print(path_base.with_suffix(".png"))
        print(path_base.with_suffix(".pdf"))
        print(path_base.with_suffix(".csv"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
