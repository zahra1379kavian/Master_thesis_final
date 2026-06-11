#!/usr/bin/env python3
"""Plot inter/intra ranks for the same edges used in the FC-change violin figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, MaxNLocator
from scipy.stats import mannwhitneyu


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

REPORT_STEM = "mutual_info_quantile__all_subjects_block_pool_pool_all_subjects_blocks_gvs_any_gvs"
EDGE_CHANGE_SUFFIX = "__all_significant_hemisphere_edge_changes.csv"

DEFAULT_INPUTS = (
    (
        "Vigour",
        ROOT
        / "figures"
        / "GVS_effects"
        / "main result"
        / "metric_sensitivity"
        / "connectogram_reports"
        / f"{REPORT_STEM}{EDGE_CHANGE_SUFFIX}",
    ),
    (
        "Task result",
        ROOT
        / "figures"
        / "GVS_effects"
        / "main result"
        / "task_activation_z3p1"
        / "metric_sensitivity"
        / "connectogram_reports"
        / f"{REPORT_STEM}{EDGE_CHANGE_SUFFIX}",
    ),
)
DEFAULT_OUTPUT_BASE = (
    ROOT
    / "figures"
    / "GVS_effects"
    / "main result"
    / "mutual_info_quantile_vigour_task_hemisphere_edge_rank_violins"
)

PAPER_FONT_FAMILY = "Liberation Sans"
INCREMENT_COLOR = "#d95f02"
DECREMENT_COLOR = "#2166ac"
DIRECTION_ORDER = (("Improved", "Increment"), ("Decrement", "Decrement"))
RELATION_ORDER = (("Between hemispheres", "Inter"), ("Within hemisphere", "intra"))

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": [PAPER_FONT_FAMILY, "Arial", "Helvetica", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

def load_ranked_edges(label: str, path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing edge-change CSV: {path}")
    df = pd.read_csv(path, low_memory=False)
    required = {"mean", "direction", "hemisphere_relation"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required column(s): {', '.join(missing)}")

    df = df.copy()
    df["mean"] = pd.to_numeric(df["mean"], errors="coerce")
    matched = df.dropna(subset=["mean", "direction", "hemisphere_relation"]).copy()
    matched = matched.loc[matched["mean"].ne(0.0)].copy()
    if matched.empty:
        raise ValueError(f"No finite non-zero edge-change rows were found in {path}")

    expected_directions = {direction for direction, _ in DIRECTION_ORDER}
    unexpected_directions = sorted(set(matched["direction"]) - expected_directions)
    if unexpected_directions:
        raise ValueError(f"Unexpected direction value(s) in {path}: {unexpected_directions}")
    expected_relations = {relation for relation, _ in RELATION_ORDER}
    unexpected_relations = sorted(set(matched["hemisphere_relation"]) - expected_relations)
    if unexpected_relations:
        raise ValueError(f"Unexpected hemisphere_relation value(s) in {path}: {unexpected_relations}")

    matched["result_set"] = label
    matched["abs_mean"] = matched["mean"].abs()
    matched["rank_within_direction"] = matched.groupby("direction")["abs_mean"].rank(
        method="average",
        ascending=True,
    )
    matched["direction_edge_count"] = (
        matched.groupby("direction")["rank_within_direction"].transform("count").astype(int)
    )
    return matched


def build_groups(data: pd.DataFrame, result_labels: tuple[str, str]) -> list[dict[str, object]]:
    groups: list[dict[str, object]] = []
    for result_index, result_label in enumerate(result_labels):
        base_position = result_index * 4.85
        for direction_index, (direction_key, direction_label) in enumerate(DIRECTION_ORDER):
            for relation_index, (relation_key, relation_label) in enumerate(RELATION_ORDER):
                position = base_position + direction_index * 2.0 + relation_index
                values = data.loc[
                    (data["result_set"] == result_label)
                    & (data["direction"] == direction_key)
                    & (data["hemisphere_relation"] == relation_key),
                    "rank_within_direction",
                ].to_numpy(dtype=np.float64)
                values = values[np.isfinite(values)]
                if values.size == 0:
                    raise ValueError(
                        f"No rows for {result_label}, {direction_label}, {relation_key}."
                    )
                groups.append(
                    {
                        "result_set": result_label,
                        "direction_key": direction_key,
                        "direction": direction_label,
                        "hemisphere_relation": relation_key,
                        "relation_label": relation_label,
                        "position": position,
                        "values": values,
                    }
                )
    return groups


def group_color(group: dict[str, object]) -> str:
    return INCREMENT_COLOR if group["direction_key"] == "Improved" else DECREMENT_COLOR


def write_ranked_edges(data: pd.DataFrame, output_base: Path) -> Path:
    cols = [
        "result_set",
        "metric",
        "analysis_view",
        "fdr_scope",
        "edge_id",
        "roi_i",
        "roi_j",
        "edge_label",
        "mean",
        "abs_mean",
        "direction",
        "hemisphere_relation",
        "rank_within_direction",
        "direction_edge_count",
        "p_signflip",
        "q_fdr",
        "sig_fdr",
    ]
    available = [col for col in cols if col in data.columns]
    path = output_base.with_name(f"{output_base.name}_ranked_edges.csv")
    data[available].sort_values(
        ["result_set", "direction", "rank_within_direction", "hemisphere_relation", "edge_id"],
        ascending=[True, True, False, True, True],
    ).to_csv(path, index=False)
    return path


def write_group_summary(groups: list[dict[str, object]], output_base: Path) -> Path:
    rows = []
    for group in groups:
        values = np.asarray(group["values"], dtype=np.float64)
        rows.append(
            {
                "result_set": group["result_set"],
                "direction": group["direction"],
                "hemisphere_relation": group["hemisphere_relation"],
                "n_edges": int(values.size),
                "mean_rank": float(np.mean(values)),
                "median_rank": float(np.median(values)),
                "q1_rank": float(np.percentile(values, 25)),
                "q3_rank": float(np.percentile(values, 75)),
                "min_rank": float(np.min(values)),
                "max_rank": float(np.max(values)),
            }
        )
    path = output_base.with_name(f"{output_base.name}_summary.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def fdr_bh(p_values: list[float]) -> list[float]:
    p_array = np.asarray(p_values, dtype=np.float64)
    q_values = np.full(p_array.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(p_array)
    if not np.any(valid):
        return q_values.tolist()
    valid_indices = np.flatnonzero(valid)
    valid_p = p_array[valid]
    order = np.argsort(valid_p)
    ranked_p = valid_p[order]
    m = ranked_p.size
    ranked_q = np.empty(m, dtype=np.float64)
    running_min = 1.0
    for rank_index in range(m - 1, -1, -1):
        rank = rank_index + 1
        running_min = min(running_min, ranked_p[rank_index] * m / rank)
        ranked_q[rank_index] = running_min
    corrected = np.empty(m, dtype=np.float64)
    corrected[order] = np.minimum(ranked_q, 1.0)
    q_values[valid_indices] = corrected
    return q_values.tolist()


def inter_intra_tests(groups: list[dict[str, object]], edge_inclusion: str) -> list[dict[str, object]]:
    rows = []
    for left, right in zip(groups[0::2], groups[1::2], strict=True):
        left_values = np.asarray(left["values"], dtype=np.float64)
        right_values = np.asarray(right["values"], dtype=np.float64)
        result = mannwhitneyu(left_values, right_values, alternative="two-sided", method="auto")
        rows.append(
            {
                "test": "Mann-Whitney U / Wilcoxon rank-sum on edge ranks",
                "edge_inclusion": edge_inclusion,
                "rank_basis": "Higher rank values indicate larger absolute mean FC change within result_set x direction.",
                "result_set": left["result_set"],
                "direction": left["direction"],
                "left_relation": left["hemisphere_relation"],
                "right_relation": right["hemisphere_relation"],
                "left_n_edges": int(left_values.size),
                "right_n_edges": int(right_values.size),
                "left_mean_rank": float(np.mean(left_values)),
                "right_mean_rank": float(np.mean(right_values)),
                "left_median_rank": float(np.median(left_values)),
                "right_median_rank": float(np.median(right_values)),
                "median_rank_difference_inter_minus_intra": float(
                    np.median(left_values) - np.median(right_values)
                ),
                "mannwhitney_u": float(result.statistic),
                "p_value_two_sided": float(result.pvalue),
                "left_position": float(left["position"]),
                "right_position": float(right["position"]),
                "direction_key": left["direction_key"],
                "pair_max_rank": max(float(np.max(left_values)), float(np.max(right_values))),
            }
        )
    q_values = fdr_bh([float(row["p_value_two_sided"]) for row in rows])
    for row, q_value in zip(rows, q_values, strict=True):
        row["q_fdr_bh"] = float(q_value)
    return rows


def write_test_summary(test_rows: list[dict[str, object]], output_base: Path) -> Path:
    export_rows = []
    for row in test_rows:
        export_row = {
            key: value
            for key, value in row.items()
            if key not in {"left_position", "right_position", "direction_key", "pair_max_rank"}
        }
        export_rows.append(export_row)
    path = output_base.with_name(f"{output_base.name}_inter_intra_rank_tests.csv")
    pd.DataFrame(export_rows).to_csv(path, index=False)
    return path


def significance_label(q_value: float) -> str:
    if not np.isfinite(q_value):
        return ""
    if q_value < 0.001:
        return "***"
    if q_value < 0.01:
        return "**"
    if q_value < 0.05:
        return "*"
    return ""


def add_test_annotation(ax: plt.Axes, row: dict[str, object], y_limit: float) -> None:
    x_left = float(row["left_position"])
    x_right = float(row["right_position"])
    label = significance_label(float(row["q_fdr_bh"]))
    if not label:
        return
    bracket_height = y_limit * 0.018
    y = min(float(row["pair_max_rank"]) + y_limit * 0.075, y_limit * 0.9)
    text_y = y + y_limit * 0.035
    ax.plot(
        [x_left, x_left, x_right, x_right],
        [y, y + bracket_height, y + bracket_height, y],
        color="#303030",
        linewidth=0.9,
        clip_on=False,
    )
    ax.text(
        (x_left + x_right) / 2,
        text_y,
        label,
        ha="center",
        va="bottom",
        fontsize=13.0,
        color="#202020",
    )


def plot_violin_summary(
    groups: list[dict[str, object]], test_rows: list[dict[str, object]], output_base: Path
) -> tuple[Path, Path]:
    positions = [float(group["position"]) for group in groups]
    values = [np.asarray(group["values"], dtype=np.float64) for group in groups]
    colors = [group_color(group) for group in groups]

    max_rank = max(float(np.max(group_values)) for group_values in values)
    significant_pair_max = [
        float(row["pair_max_rank"])
        for row in test_rows
        if significance_label(float(row["q_fdr_bh"]))
    ]
    annotation_limit = max((pair_max / 0.86 for pair_max in significant_pair_max), default=0.0)
    y_limit = float(np.ceil(max(max_rank * 1.06, annotation_limit, 10.0)))

    fig, ax = plt.subplots(figsize=(9.6, 4.2))

    violins = ax.violinplot(
        values,
        positions=positions,
        widths=0.78,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body, color in zip(violins["bodies"], colors, strict=True):
        body.set_facecolor(color)
        body.set_edgecolor("none")
        body.set_alpha(0.18)

    box = ax.boxplot(
        values,
        positions=positions,
        widths=0.25,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#ffffff", "linewidth": 1.45},
        whiskerprops={"color": "#444444", "linewidth": 0.9},
        capprops={"color": "#444444", "linewidth": 0.9},
    )
    for patch, color in zip(box["boxes"], colors, strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.86)
        patch.set_edgecolor("#303030")
        patch.set_linewidth(0.8)

    rng = np.random.default_rng(20260610)
    for position, group_values, color in zip(positions, values, colors, strict=True):
        jitter = rng.normal(loc=position, scale=0.052, size=group_values.size)
        ax.scatter(
            jitter,
            group_values,
            s=11,
            color=color,
            alpha=0.34,
            linewidths=0.0,
            zorder=2,
        )
    ax.axvline(3.925, color="#B8B8B8", linewidth=0.85, zorder=0)
    for row in test_rows:
        add_test_annotation(ax, row, y_limit)

    tick_labels = [f"{group['relation_label']}" for group in groups]
    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels, fontsize=12.0)
    ax.set_xlim(-0.55, 8.4)
    ax.set_ylim(0.0, y_limit)
    ax.set_ylabel("edge rank", fontsize=15.0)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0f}"))
    ax.grid(axis="y", color="#DADADA", linewidth=0.75)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", length=0, pad=6)
    ax.tick_params(axis="y", labelsize=12.0)

    output_base.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_base.with_suffix(".png")
    pdf_path = output_base.with_suffix(".pdf")
    fig.savefig(png_path, dpi=320, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return png_path, pdf_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-base",
        type=Path,
        default=DEFAULT_OUTPUT_BASE,
        help="Output path without suffix. PNG, PDF, and summary CSVs are written.",
    )
    parser.add_argument(
        "--input",
        nargs=2,
        action="append",
        metavar=("LABEL", "CSV"),
        help="Result label and edge-change CSV. Repeat twice.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    inputs = tuple((label, Path(path)) for label, path in args.input) if args.input else DEFAULT_INPUTS
    if len(inputs) != 2:
        raise ValueError("Expected exactly two input CSVs: one for Vigour and one for Task result.")

    data = pd.concat(
        [
            load_ranked_edges(label, path)
            for label, path in inputs
        ],
        ignore_index=True,
    )
    result_labels = tuple(label for label, _ in inputs)
    groups = build_groups(data, result_labels)
    edge_inclusion = "Exact rows from the value figure all_significant_hemisphere_edge_changes CSVs"
    test_rows = inter_intra_tests(groups, edge_inclusion)
    png_path, pdf_path = plot_violin_summary(groups, test_rows, args.output_base)
    ranked_path = write_ranked_edges(data, args.output_base)
    summary_path = write_group_summary(groups, args.output_base)
    test_path = write_test_summary(test_rows, args.output_base)
    print(png_path)
    print(pdf_path)
    print(ranked_path)
    print(summary_path)
    print(test_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
