#!/usr/bin/env python3
"""Plot FDR-significant edge densities by connectogram ROI-group relation."""

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
    / "mutual_info_quantile_vigour_task_roi_group_significant_edge_density_by_direction_violins"
)

PAPER_FONT_FAMILY = "Liberation Sans"
INCREMENT_COLOR = "#d95f02"
DECREMENT_COLOR = "#2166ac"
DIRECTION_ORDER = (("Improved", "Increment"), ("Decrement", "Decrement"))
RELATION_ORDER = (("Between ROI groups", "Between"), ("Within ROI group", "Within"))
N_BOOTSTRAP = 10_000
RNG_SEED = 20260610

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": [PAPER_FONT_FAMILY, "Arial", "Helvetica", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def roi_group(roi: str) -> str:
    base = str(roi).rsplit("_", 1)[0]
    if base in {"Occipital", "Fusiform"}:
        return "Visual"
    if base in {"Precentral", "Postcentral", "Paracentral_Lobule", "Supp_Motor_Area", "Cerebellum", "Rolandic_Oper"}:
        return "Somatomotor/Cerebellar"
    if base in {"Amygdala", "Hippocampus", "ParaHippocampal", "Olfactory", "Orbitofrontal"}:
        return "Limbic/MTL-Olfactory"
    if base in {"Caudate", "Pallidum", "Putamen", "Thalamus"}:
        return "Subcortical"
    if base in {"Frontal", "Parietal"}:
        return "Frontal-Parietal"
    return "Cingulate-Temporal"


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


def load_edge_table(label: str, path: Path, metric: str, analysis_view: str, fdr_scope: str) -> pd.DataFrame:
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
    matched["roi_group_i"] = matched["roi_i"].map(roi_group)
    matched["roi_group_j"] = matched["roi_j"].map(roi_group)
    matched["roi_group_relation"] = np.where(
        matched["roi_group_i"].eq(matched["roi_group_j"]),
        "Within ROI group",
        "Between ROI groups",
    )
    matched["result_set"] = label
    matched["is_significant_edge"] = matched["sig_fdr"].astype(bool)
    matched["direction"] = np.select(
        [matched["mean"].gt(0.0), matched["mean"].lt(0.0)],
        ["Improved", "Decrement"],
        default="No change",
    )
    return matched


def bootstrap_density(indicator: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    indicator = np.asarray(indicator, dtype=np.float64)
    if indicator.size == 0:
        raise ValueError("Cannot bootstrap an empty edge group.")
    sample_indices = rng.integers(0, indicator.size, size=(N_BOOTSTRAP, indicator.size))
    return indicator[sample_indices].mean(axis=1)


def build_groups(data: pd.DataFrame, result_labels: tuple[str, str]) -> list[dict[str, object]]:
    rng = np.random.default_rng(RNG_SEED)
    groups: list[dict[str, object]] = []
    for result_index, result_label in enumerate(result_labels):
        base_position = result_index * 4.85
        for direction_index, (direction_key, direction_label) in enumerate(DIRECTION_ORDER):
            for relation_index, (relation_key, relation_label) in enumerate(RELATION_ORDER):
                position = base_position + direction_index * 2.0 + relation_index
                subset = data.loc[
                    (data["result_set"] == result_label)
                    & (data["roi_group_relation"] == relation_key)
                ].copy()
                indicator = (
                    subset["is_significant_edge"]
                    & subset["direction"].eq(direction_key)
                ).to_numpy(dtype=np.float64)
                densities = bootstrap_density(indicator, rng)
                groups.append(
                    {
                        "result_set": result_label,
                        "direction_key": direction_key,
                        "direction": direction_label,
                        "roi_group_relation": relation_key,
                        "relation_label": relation_label,
                        "position": position,
                        "values": densities,
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
        odds_ratio, p_value = fisher_exact(
            [
                [left_success, left_total - left_success],
                [right_success, right_total - right_success],
            ],
            alternative="two-sided",
        )
        rows.append(
            {
                "test": "Fisher exact test on FDR-significant edge density",
                "result_set": left["result_set"],
                "direction": left["direction"],
                "left_relation": left["roi_group_relation"],
                "right_relation": right["roi_group_relation"],
                "left_n_sig_edges": left_success,
                "left_n_edges_tested": left_total,
                "left_density": float(left_success / left_total),
                "right_n_sig_edges": right_success,
                "right_n_edges_tested": right_total,
                "right_density": float(right_success / right_total),
                "density_difference_between_minus_within": float(
                    left_success / left_total - right_success / right_total
                ),
                "odds_ratio_between_vs_within": float(odds_ratio),
                "p_value_two_sided": float(p_value),
                "q_fdr_bh": np.nan,
                "left_position": float(left["position"]),
                "right_position": float(right["position"]),
            }
        )
    q_values = fdr_bh([float(row["p_value_two_sided"]) for row in rows])
    for row, q_value in zip(rows, q_values, strict=True):
        row["q_fdr_bh"] = float(q_value)
    return rows


def write_group_summary(groups: list[dict[str, object]], output_base: Path) -> Path:
    rows = []
    for group in groups:
        values = np.asarray(group["values"], dtype=np.float64)
        rows.append(
            {
                "result_set": group["result_set"],
                "direction": group["direction"],
                "roi_group_relation": group["roi_group_relation"],
                "n_sig_edges": group["n_sig_edges"],
                "n_edges_tested": group["n_edges_tested"],
                "observed_density": group["observed_density"],
                "bootstrap_mean_density": float(np.mean(values)),
                "bootstrap_median_density": float(np.median(values)),
                "bootstrap_q1_density": float(np.percentile(values, 25)),
                "bootstrap_q3_density": float(np.percentile(values, 75)),
                "bootstrap_p025_density": float(np.percentile(values, 2.5)),
                "bootstrap_p975_density": float(np.percentile(values, 97.5)),
            }
        )
    path = output_base.with_name(f"{output_base.name}_summary.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def write_test_summary(test_rows: list[dict[str, object]], output_base: Path) -> Path:
    rows = []
    for row in test_rows:
        rows.append(
            {
                key: value
                for key, value in row.items()
                if key not in {"left_position", "right_position"}
            }
        )
    path = output_base.with_name(f"{output_base.name}_between_within_density_tests.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
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
    ax.text(
        (x_left + x_right) / 2,
        y + bracket_height + y_limit * 0.02,
        f"p={format_p(float(row['p_value_two_sided']))}\nq={format_p(float(row['q_fdr_bh']))}",
        ha="center",
        va="bottom",
        fontsize=8.6,
        color="#202020",
        linespacing=0.95,
    )


def plot_density_violins(groups: list[dict[str, object]], test_rows: list[dict[str, object]], output_base: Path) -> tuple[Path, Path]:
    positions = [float(group["position"]) for group in groups]
    values = [np.asarray(group["values"], dtype=np.float64) for group in groups]
    colors = [group_color(group) for group in groups]
    observed = [float(group["observed_density"]) for group in groups]
    max_density = max(float(np.percentile(group_values, 99.5)) for group_values in values)
    max_observed = max(observed)
    y_limit = np.ceil(max(max_density * 1.45, max_observed * 1.55, 0.08) / 0.025) * 0.025

    fig, ax = plt.subplots(figsize=(9.6, 4.2))
    violins = ax.violinplot(
        values,
        positions=positions,
        widths=0.75,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body, color in zip(violins["bodies"], colors, strict=True):
        body.set_facecolor(color)
        body.set_edgecolor("none")
        body.set_alpha(0.2)

    box = ax.boxplot(
        values,
        positions=positions,
        widths=0.22,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#ffffff", "linewidth": 1.35},
        whiskerprops={"color": "#444444", "linewidth": 0.85},
        capprops={"color": "#444444", "linewidth": 0.85},
    )
    for patch, color in zip(box["boxes"], colors, strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.86)
        patch.set_edgecolor("#303030")
        patch.set_linewidth(0.8)

    rng = np.random.default_rng(RNG_SEED + 1)
    for position, group_values, color in zip(positions, values, colors, strict=True):
        sample = rng.choice(group_values, size=min(260, group_values.size), replace=False)
        jitter = rng.normal(loc=position, scale=0.045, size=sample.size)
        ax.scatter(jitter, sample, s=6, color=color, alpha=0.12, linewidths=0.0, zorder=2)

    for position, density, group, color in zip(positions, observed, groups, colors, strict=True):
        ax.scatter([position], [density], s=36, color=color, edgecolor="white", linewidth=0.65, zorder=5)
        ax.text(
            position,
            density + y_limit * 0.035,
            f"{int(group['n_sig_edges'])}/{int(group['n_edges_tested'])}",
            ha="center",
            va="bottom",
            fontsize=8.2,
            color="#303030",
        )

    ax.axvline(3.925, color="#B8B8B8", linewidth=0.85, zorder=0)
    for row in test_rows:
        add_test_annotation(ax, row, y_limit)

    ax.set_xticks(positions)
    ax.set_xticklabels([group["relation_label"] for group in groups], fontsize=10.7)
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
    parser.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT_BASE)
    parser.add_argument("--input", nargs=2, action="append", metavar=("LABEL", "CSV"))
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
    png_path, pdf_path = plot_density_violins(groups, test_rows, args.output_base)
    summary_path = write_group_summary(groups, args.output_base)
    test_path = write_test_summary(test_rows, args.output_base)
    print(png_path)
    print(pdf_path)
    print(summary_path)
    print(test_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
