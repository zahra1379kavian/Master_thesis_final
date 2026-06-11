#!/usr/bin/env python3
"""Forest plot of between-vs-within-hemisphere odds ratios per network.

Shows the four (network x direction) between/within enrichment odds ratios with
95% CIs from ``hemisphere_enrichment.csv``. An OR > 1 means between-hemisphere
edges are over-represented among the significant changes relative to the
available edge pool; OR < 1 means within-hemisphere over-representation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

DEFAULT_INPUT = (
    ROOT
    / "figures"
    / "GVS_effects"
    / "main result"
    / "connectogram_network_comparison"
    / "hemisphere_enrichment.csv"
)
DEFAULT_OUTPUT_BASE = (
    ROOT
    / "figures"
    / "GVS_effects"
    / "main result"
    / "hemisphere_enrichment_forest"
)

PAPER_FONT_FAMILY = "Liberation Sans"
INCREMENT_COLOR = "#d95f02"
DECREMENT_COLOR = "#2166ac"

# network key -> display label
NETWORK_LABELS = {"main_result": "Vigour", "task_activation_z3p1": "Task"}
# direction key -> (display label, color)
DIRECTIONS = (("improved", "Increment", INCREMENT_COLOR), ("decreased", "Decrement", DECREMENT_COLOR))
NETWORK_ORDER = ("main_result", "task_activation_z3p1")

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": [PAPER_FONT_FAMILY, "Arial", "Helvetica", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def significance_label(p_value: float) -> str:
    if not np.isfinite(p_value):
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def build_rows(df: pd.DataFrame, roi_set: str) -> list[dict[str, object]]:
    """One row per (direction, network) entry, top-to-bottom plotting order."""
    subset = df.loc[df["roi_set"].eq(roi_set)].copy()
    rows: list[dict[str, object]] = []
    # y descends so the first listed entry sits at the top of the axis.
    y = 0.0
    for direction_key, direction_label, color in DIRECTIONS:
        for network_key in NETWORK_ORDER:
            match = subset.loc[
                subset["direction"].eq(direction_key) & subset["network"].eq(network_key)
            ]
            if match.empty:
                raise ValueError(f"Missing row for {network_key} / {direction_key} in {roi_set}.")
            record = match.iloc[0]
            rows.append(
                {
                    "y": y,
                    "direction_label": direction_label,
                    "network_label": NETWORK_LABELS.get(network_key, network_key),
                    "color": color,
                    "odds_ratio": float(record["between_vs_within_odds_ratio"]),
                    "ci_low": float(record["odds_ratio_ci95_low"]),
                    "ci_high": float(record["odds_ratio_ci95_high"]),
                    "between_sig": int(record["between_significant_edges"]),
                    "between_possible": int(record["between_possible_edges"]),
                    "within_sig": int(record["within_significant_edges"]),
                    "within_possible": int(record["within_possible_edges"]),
                    "fisher_p": float(record["fisher_p"]),
                }
            )
            y -= 1.0
        y -= 0.55  # gap between direction blocks
    return rows


def plot_forest(rows: list[dict[str, object]], output_base: Path) -> tuple[Path, Path]:
    fig, ax = plt.subplots(figsize=(7.6, 4.0))

    ys = [row["y"] for row in rows]
    ci_low = np.array([row["ci_low"] for row in rows])
    ci_high = np.array([row["ci_high"] for row in rows])

    ax.axvline(1.0, color="#9A9A9A", linewidth=1.0, linestyle="--", zorder=1)

    for row in rows:
        y = row["y"]
        color = row["color"]
        ax.plot(
            [row["ci_low"], row["ci_high"]],
            [y, y],
            color=color,
            linewidth=1.8,
            solid_capstyle="round",
            zorder=2,
        )
        ax.scatter(
            [row["odds_ratio"]],
            [y],
            s=70,
            color=color,
            edgecolors="#303030",
            linewidths=0.7,
            zorder=3,
        )

    # x-axis: log scale because odds ratios are multiplicative.
    ax.set_xscale("log")
    x_min = min(0.5, float(np.nanmin(ci_low)) * 0.85)
    x_max = max(2.2, float(np.nanmax(ci_high)) * 1.12)
    ax.set_xlim(x_min, x_max)

    # annotate each row with OR [CI], counts, and significance star.
    text_x = x_max * 1.02
    for row in rows:
        star = significance_label(row["fisher_p"])
        label = (
            f"{row['odds_ratio']:.2f} "
            f"[{row['ci_low']:.2f}, {row['ci_high']:.2f}]"
            f"{'  ' + star if star else ''}"
        )
        ax.text(text_x, row["y"], label, va="center", ha="left", fontsize=9.5, color="#202020")

    # y tick labels: network name; direction block labels added on the left.
    ax.set_yticks(ys)
    ax.set_yticklabels([row["network_label"] for row in rows], fontsize=11.0)

    # direction headers to the far left of each block (outside the axes).
    block_centers: dict[str, list[float]] = {}
    for row in rows:
        block_centers.setdefault(row["direction_label"], []).append(row["y"])
    for direction_label, block_ys in block_centers.items():
        ax.annotate(
            direction_label,
            xy=(-0.17, float(np.mean(block_ys))),
            xycoords=("axes fraction", "data"),
            va="center",
            ha="center",
            fontsize=12.0,
            fontweight="bold",
            rotation=90,
            color="#303030",
            annotation_clip=False,
        )

    ax.set_ylim(min(ys) - 0.7, max(ys) + 0.7)
    ax.set_xlabel("Between- vs within-hemisphere odds ratio  (log scale)", fontsize=12.5)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.set_xticks([0.5, 0.75, 1.0, 1.5, 2.0, 3.0])
    ax.grid(axis="x", color="#E2E2E2", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    # headroom on the right for the annotation column.
    ax.set_xlim(x_min, x_max * 1.9)

    # interpretation note as a subtitle above the panel (avoids x-label collision).
    ax.set_title(
        "OR > 1: between-hemisphere over-represented      OR < 1: within-hemisphere over-represented",
        fontsize=9.0,
        color="#5A5A5A",
        pad=10,
    )
    fig.subplots_adjust(left=0.18)

    output_base.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_base.with_suffix(".png")
    pdf_path = output_base.with_suffix(".pdf")
    fig.savefig(png_path, dpi=320, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return png_path, pdf_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="hemisphere_enrichment.csv path.")
    parser.add_argument(
        "--roi-set",
        default="full_network",
        choices=("full_network", "common_roi_set"),
        help="Which ROI set to plot.",
    )
    parser.add_argument("--output-base", type=Path, default=None, help="Output path without suffix.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Missing enrichment CSV: {args.input}")
    df = pd.read_csv(args.input)
    output_base = args.output_base or DEFAULT_OUTPUT_BASE.with_name(
        f"{DEFAULT_OUTPUT_BASE.name}_{args.roi_set}"
    )
    rows = build_rows(df, args.roi_set)
    png_path, pdf_path = plot_forest(rows, output_base)
    print(png_path)
    print(pdf_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
