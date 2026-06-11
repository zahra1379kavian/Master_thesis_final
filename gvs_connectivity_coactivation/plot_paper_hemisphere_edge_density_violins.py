#!/usr/bin/env python3
"""Plot inter/intra FDR-significant edge densities as observed bars."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, MaxNLocator
from scipy.stats import fisher_exact


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

DEFAULT_METRIC = "mutual_info_quantile"
DEFAULT_ANALYSIS_VIEW = "all_subjects_block_pool"
DEFAULT_FDR_SCOPE = "pool=ALL_SUBJECTS_BLOCKS;gvs=ANY_GVS"

DEFAULT_INPUTS = (
    (
        "Vigour",
        ROOT
        / "figures"
        / "GVS_effects"
        / "GPT"
        / "08_connectivity_coactivation"
        / "metric_sensitivity"
        / "edge_connectivity_metric_sensitivity_stats.csv",
    ),
    (
        "Task result",
        ROOT
        / "figures"
        / "GVS_effects"
        / "GPT"
        / "08_connectivity_coactivation"
        / "task_activation_z3p1"
        / "metric_sensitivity"
        / "edge_connectivity_metric_sensitivity_stats.csv",
    ),
)

DEFAULT_OUTPUT_BASE = (
    ROOT
    / "figures"
    / "GVS_effects"
    / "main result"
    / "mutual_info_quantile_vigour_task_hemisphere_significant_edge_density_by_direction_violins"
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


def bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.lower().isin({"true", "1", "yes", "y"})


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


def load_edge_table(
    label: str,
    path: Path,
    metric: str,
    analysis_view: str,
    fdr_scope: str,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing full edge stats CSV: {path}")
    df = pd.read_csv(path, low_memory=False)
    required = {"metric", "analysis_view", "fdr_scope", "roi_i", "roi_j", "mean", "sig_fdr"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required column(s): {', '.join(missing)}")

    matched = df.loc[
        (df["metric"] == metric)
        & (df["analysis_view"] == analysis_view)
        & (df["fdr_scope"] == fdr_scope)
    ].copy()
    if matched.empty:
        raise ValueError(f"No rows matched {metric} | {analysis_view} | {fdr_scope} in {path}")

    matched["mean"] = pd.to_numeric(matched["mean"], errors="coerce")
    matched["sig_fdr"] = bool_series(matched["sig_fdr"])
    matched = matched.dropna(subset=["mean", "roi_i", "roi_j"]).copy()
    matched["hemisphere_relation"] = [
        hemisphere_relation(roi_i, roi_j)
        for roi_i, roi_j in zip(matched["roi_i"], matched["roi_j"])
    ]
    unknown = matched.loc[matched["hemisphere_relation"] == "Unclassified", ["roi_i", "roi_j"]]
    if not unknown.empty:
        preview = unknown.head(5).to_dict(orient="records")
        raise ValueError(f"Found edges without _L/_R hemisphere labels: {preview}")

    matched["result_set"] = label
    matched["is_significant_edge"] = matched["sig_fdr"].astype(bool)
    matched["direction"] = np.select(
        [matched["mean"].gt(0.0), matched["mean"].lt(0.0)],
        ["Improved", "Decrement"],
        default="No change",
    )
    return matched


def build_groups(data: pd.DataFrame, result_labels: tuple[str, str]) -> list[dict[str, object]]:
    groups: list[dict[str, object]] = []
    for result_index, result_label in enumerate(result_labels):
        base_position = result_index * 4.85
        for direction_index, (direction_key, direction_label) in enumerate(DIRECTION_ORDER):
            for relation_index, (relation_key, relation_label) in enumerate(RELATION_ORDER):
                position = base_position + direction_index * 2.0 + relation_index
                subset = data.loc[
                    (data["result_set"] == result_label)
                    & (data["hemisphere_relation"] == relation_key)
                ].copy()
                indicator = (
                    subset["is_significant_edge"]
                    & subset["direction"].eq(direction_key)
                ).to_numpy(dtype=np.float64)
                if indicator.size == 0:
                    raise ValueError("Cannot compute density for an empty edge group.")
                groups.append(
                    {
                        "result_set": result_label,
                        "direction_key": direction_key,
                        "direction": direction_label,
                        "hemisphere_relation": relation_key,
                        "relation_label": relation_label,
                        "position": position,
                        "n_edges_tested": int(indicator.size),
                        "n_sig_edges": int(indicator.sum()),
                        "observed_density": float(indicator.mean()),
                    }
                )
    return groups


def group_color(group: dict[str, object]) -> str:
    return INCREMENT_COLOR if group["direction_key"] == "Improved" else DECREMENT_COLOR


def density_tests(groups: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for left, right in zip(groups[0::2], groups[1::2], strict=True):
        left_success = int(left["n_sig_edges"])
        left_total = int(left["n_edges_tested"])
        right_success = int(right["n_sig_edges"])
        right_total = int(right["n_edges_tested"])
        table = [
            [left_success, left_total - left_success],
            [right_success, right_total - right_success],
        ]
        odds_ratio, p_value = fisher_exact(table, alternative="two-sided")
        rows.append(
            {
                "test": "Fisher exact test on FDR-significant edge density",
                "result_set": left["result_set"],
                "direction": left["direction"],
                "left_relation": left["hemisphere_relation"],
                "right_relation": right["hemisphere_relation"],
                "left_n_sig_edges": left_success,
                "left_n_edges_tested": left_total,
                "left_density": float(left_success / left_total),
                "right_n_sig_edges": right_success,
                "right_n_edges_tested": right_total,
                "right_density": float(right_success / right_total),
                "density_difference_inter_minus_intra": float(
                    left_success / left_total - right_success / right_total
                ),
                "odds_ratio_inter_vs_intra": float(odds_ratio),
                "p_value_two_sided": float(p_value),
                "left_position": float(left["position"]),
                "right_position": float(right["position"]),
                "direction_key": left["direction_key"],
            }
        )
    q_values = fdr_bh([float(row["p_value_two_sided"]) for row in rows])
    for row, q_value in zip(rows, q_values, strict=True):
        row["q_fdr_bh"] = float(q_value)
    return rows


def write_group_summary(groups: list[dict[str, object]], output_base: Path) -> Path:
    rows = []
    for group in groups:
        rows.append(
            {
                "result_set": group["result_set"],
                "direction": group["direction"],
                "hemisphere_relation": group["hemisphere_relation"],
                "n_sig_edges": group["n_sig_edges"],
                "n_edges_tested": group["n_edges_tested"],
                "observed_density": group["observed_density"],
            }
        )
    path = output_base.with_name(f"{output_base.name}_summary.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def write_test_summary(test_rows: list[dict[str, object]], output_base: Path) -> Path:
    export_rows = []
    for row in test_rows:
        export_rows.append(
            {
                key: value
                for key, value in row.items()
                if key not in {"left_position", "right_position", "direction_key"}
            }
        )
    path = output_base.with_name(f"{output_base.name}_inter_intra_density_tests.csv")
    pd.DataFrame(export_rows).to_csv(path, index=False)
    return path


def format_p(value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def add_test_annotation(ax: plt.Axes, row: dict[str, object], y_limit: float) -> None:
    x_left = float(row["left_position"])
    x_right = float(row["right_position"])
    y = min(
        max(float(row["left_density"]), float(row["right_density"])) + y_limit * 0.14,
        y_limit * 0.86,
    )
    bracket_height = y_limit * 0.035
    ax.plot(
        [x_left, x_left, x_right, x_right],
        [y, y + bracket_height, y + bracket_height, y],
        color="#303030",
        linewidth=0.9,
        clip_on=False,
    )
    label = f"p={format_p(float(row['p_value_two_sided']))}\nq={format_p(float(row['q_fdr_bh']))}"
    ax.text(
        (x_left + x_right) / 2,
        y + bracket_height + y_limit * 0.02,
        label,
        ha="center",
        va="bottom",
        fontsize=8.6,
        color="#202020",
        linespacing=0.95,
    )


def plot_density_bars(
    groups: list[dict[str, object]],
    test_rows: list[dict[str, object]],
    output_base: Path,
) -> tuple[Path, Path]:
    positions = [float(group["position"]) for group in groups]
    colors = [group_color(group) for group in groups]
    observed = [float(group["observed_density"]) for group in groups]

    max_observed = max(observed)
    y_limit = np.ceil(max(max_observed * 1.70, 0.08) / 0.025) * 0.025

    fig, ax = plt.subplots(figsize=(9.6, 4.2))
    ax.bar(
        x=positions,
        height=observed,
        width=0.58,
        color=colors,
        alpha=0.86,
        edgecolor="#303030",
        linewidth=0.8,
        zorder=3,
    )

    for position, density, group, color in zip(positions, observed, groups, colors, strict=True):
        count_label = f"{int(group['n_sig_edges'])}/{int(group['n_edges_tested'])}"
        ax.text(
            position,
            density + y_limit * 0.035,
            count_label,
            ha="center",
            va="bottom",
            fontsize=8.2,
            color="#303030",
        )

    ax.axvline(3.925, color="#B8B8B8", linewidth=0.85, zorder=0)
    for row in test_rows:
        add_test_annotation(ax, row, y_limit)

    ax.set_xticks(positions)
    ax.set_xticklabels([group["relation_label"] for group in groups], fontsize=11.0)
    ax.set_xlim(-0.55, 8.4)
    ax.set_ylim(0.0, y_limit)
    ax.set_ylabel("FDR-significant edge density", fontsize=13.2)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0%}"))
    ax.grid(axis="y", color="#DADADA", linewidth=0.75)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", length=0, pad=6)
    ax.tick_params(axis="y", labelsize=10.5)

    ax.text(1.5, y_limit * 0.985, "Vigour", ha="center", va="top", fontsize=11.5, fontweight="bold")
    ax.text(6.35, y_limit * 0.985, "Task result", ha="center", va="top", fontsize=11.5, fontweight="bold")

    output_base.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_base.with_suffix(".png")
    pdf_path = output_base.with_suffix(".pdf")
    fig.savefig(png_path, dpi=320, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return png_path, pdf_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric", default=DEFAULT_METRIC)
    parser.add_argument("--analysis-view", default=DEFAULT_ANALYSIS_VIEW)
    parser.add_argument("--fdr-scope", default=DEFAULT_FDR_SCOPE)
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
        help="Result label and full edge stats CSV. Repeat twice.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    inputs = tuple((label, Path(path)) for label, path in args.input) if args.input else DEFAULT_INPUTS
    if len(inputs) != 2:
        raise ValueError("Expected exactly two input CSVs: one for Vigour and one for Task result.")

    data = pd.concat(
        [
            load_edge_table(label, path, args.metric, args.analysis_view, args.fdr_scope)
            for label, path in inputs
        ],
        ignore_index=True,
    )
    result_labels = tuple(label for label, _ in inputs)
    groups = build_groups(data, result_labels)
    test_rows = density_tests(groups)
    png_path, pdf_path = plot_density_bars(groups, test_rows, args.output_base)
    summary_path = write_group_summary(groups, args.output_base)
    test_path = write_test_summary(test_rows, args.output_base)
    print(png_path)
    print(pdf_path)
    print(summary_path)
    print(test_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
