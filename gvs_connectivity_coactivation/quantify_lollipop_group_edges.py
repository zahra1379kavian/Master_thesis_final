#!/usr/bin/env python3
"""Count all FDR-significant edges between broad anatomical groups."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_edge_change_lollipop import clean_slug
from plot_fdr_significant_edge_connectograms import roi_group


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

DEFAULT_REPORTS = (
    (
        "main_result",
        ROOT / "figures" / "GVS_effects" / "GPT" / "08_connectivity_coactivation" / "metric_sensitivity" / "fdr_significant_edge_connectivity_metric_sensitivity.csv",
        ROOT / "figures" / "GVS_effects" / "main result" / "metric_sensitivity" / "connectogram_reports",
    ),
)

GROUP_ORDER = [
    "Frontal-Parietal",
    "Subcortical",
    "Cingulate-Temporal",
    "Visual",
    "Limbic/MTL-Olfactory",
    "Somatomotor/Cerebellar",
]

GROUP_LABELS = {
    "Frontal-Parietal": "Frontal-Parietal",
    "Subcortical": "Subcortical",
    "Cingulate-Temporal": "Cingulate-Temporal",
    "Visual": "Visual",
    "Limbic/MTL-Olfactory": "Limbic-Olfactory",
    "Somatomotor/Cerebellar": "Somatosensory-Cerebellum",
}

PLOT_LABELS = {
    "Frontal-Parietal": "Frontal-\nParietal",
    "Subcortical": "Subcortical",
    "Cingulate-Temporal": "Cingulate-\nTemporal",
    "Visual": "Visual",
    "Limbic-Olfactory": "Limbic-\nOlfactory",
    "Somatosensory-Cerebellum": "Somatosensory-\nCerebellum",
}

DIRECTIONS = ("increase", "decrease")


@dataclass(frozen=True)
class LollipopSpec:
    metric_slug: str
    scope_slug: str
    top_n: int
    png_path: Path


def ordered_pair(group_i: str, group_j: str) -> tuple[str, str]:
    order = {group: idx for idx, group in enumerate(GROUP_ORDER)}
    if order[group_i] <= order[group_j]:
        return group_i, group_j
    return group_j, group_i


def parse_lollipop_specs(report_dir: Path) -> list[LollipopSpec]:
    specs: list[LollipopSpec] = []
    for png_path in sorted(report_dir.glob("*__top*_lollipop.png")):
        stem = png_path.stem
        if not stem.endswith("_lollipop"):
            continue
        prefix = stem.removesuffix("_lollipop")
        if "__top" not in prefix or "__" not in prefix:
            continue
        before_top, top_text = prefix.rsplit("__top", 1)
        if not top_text.isdigit():
            continue
        metric_slug, scope_slug = before_top.split("__", 1)
        specs.append(LollipopSpec(metric_slug=metric_slug, scope_slug=scope_slug, top_n=int(top_text), png_path=png_path))
    return specs


def load_significant_edges(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    if "sig_fdr" in df.columns:
        df = df.loc[df["sig_fdr"].astype(bool)].copy()
    for col in ("mean", "q_fdr", "p_signflip", "abs_mean"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["mean", "roi_i", "roi_j"]).copy()
    df["abs_mean"] = df["mean"].abs()
    df["metric_slug"] = df["metric"].map(clean_slug)
    df["scope_slug"] = [clean_slug(f"{view}_{scope}") for view, scope in zip(df["analysis_view"], df["fdr_scope"], strict=True)]
    return df


def select_lollipop_edges(sig: pd.DataFrame, spec: LollipopSpec) -> pd.DataFrame:
    matched = sig.loc[(sig["metric_slug"] == spec.metric_slug) & (sig["scope_slug"] == spec.scope_slug)].copy()
    if matched.empty:
        raise ValueError(f"No significant edges matched {spec.png_path.name}")
    return matched.sort_values(["abs_mean", "q_fdr", "p_signflip"], ascending=[False, True, True]).copy()


def empty_matrix() -> pd.DataFrame:
    labels = [GROUP_LABELS[group] for group in GROUP_ORDER]
    return pd.DataFrame(0, index=labels, columns=labels, dtype=int)


def pair_count_table(edges: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for row in edges.itertuples(index=False):
        group_i, group_j = ordered_pair(roi_group(row.roi_i), roi_group(row.roi_j))
        rows.append(
            {
                "group_i": GROUP_LABELS[group_i],
                "group_j": GROUP_LABELS[group_j],
                "mean": float(row.mean),
            }
        )
    edge_groups = pd.DataFrame(rows)

    pair_rows = []
    for i, group_i in enumerate(GROUP_ORDER):
        for group_j in GROUP_ORDER[i:]:
            label_i = GROUP_LABELS[group_i]
            label_j = GROUP_LABELS[group_j]
            if edge_groups.empty:
                pair_edges = edge_groups
            else:
                pair_edges = edge_groups.loc[(edge_groups["group_i"] == label_i) & (edge_groups["group_j"] == label_j)]
            n_edges = int(pair_edges.shape[0])
            n_positive = int((pair_edges["mean"] > 0).sum()) if n_edges else 0
            pair_rows.append(
                {
                    "group_i": label_i,
                    "group_j": label_j,
                    "n_edges": n_edges,
                    "n_positive": n_positive,
                    "n_negative": n_edges - n_positive,
                }
            )
    pairs = pd.DataFrame(pair_rows)

    matrix = empty_matrix()
    for row in pairs.itertuples(index=False):
        matrix.loc[row.group_i, row.group_j] = int(row.n_edges)
        matrix.loc[row.group_j, row.group_i] = int(row.n_edges)
    return pairs, matrix


def direction_edges(edges: pd.DataFrame, direction: str) -> pd.DataFrame:
    if direction == "increase":
        return edges.loc[edges["mean"] > 0].copy()
    if direction == "decrease":
        return edges.loc[edges["mean"] < 0].copy()
    raise ValueError(f"Unknown edge direction: {direction}")


def plot_heatmaps(matrices: list[pd.DataFrame], path_base: Path, panel_labels: list[str] | None = None) -> None:
    if not matrices:
        return
    vmax = max(int(matrix.to_numpy().max()) for matrix in matrices)
    vmax = max(vmax, 1)
    ncols = len(matrices)
    fig_width = max(6.8, 5.15 * ncols)
    fig, axes = plt.subplots(1, ncols, figsize=(fig_width, 5.9), squeeze=False)
    axes_flat = axes.ravel()

    image = None
    labels = panel_labels or [""] * len(matrices)
    for ax, matrix, label in zip(axes_flat, matrices, labels, strict=True):
        image = ax.imshow(matrix.to_numpy(dtype=float), cmap="YlGnBu", vmin=0, vmax=vmax)
        if label:
            ax.set_title(label, fontsize=11.5, pad=10)
        ax.set_xticks(np.arange(matrix.shape[1]))
        ax.set_yticks(np.arange(matrix.shape[0]))
        ax.set_xticklabels([PLOT_LABELS.get(label, label) for label in matrix.columns], fontsize=8.6, rotation=40, ha="right")
        ax.set_yticklabels([PLOT_LABELS.get(label, label) for label in matrix.index], fontsize=8.8)
        ax.tick_params(axis="both", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks(np.arange(-0.5, matrix.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, matrix.shape[0], 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.4)
        threshold = vmax * 0.56
        for row_idx in range(matrix.shape[0]):
            for col_idx in range(matrix.shape[1]):
                value = int(matrix.iat[row_idx, col_idx])
                text_color = "white" if value > threshold else "#1F2933"
                ax.text(col_idx, row_idx, str(value), ha="center", va="center", fontsize=10.2, color=text_color)

    top = 0.90 if any(labels) else 0.96
    fig.subplots_adjust(left=0.11, right=0.92, top=top, bottom=0.21, wspace=0.34)
    if image is not None:
        cbar = fig.colorbar(image, ax=axes_flat.tolist(), fraction=0.028, pad=0.02)
        cbar.set_label("Significant edges", fontsize=9.2)
        cbar.ax.tick_params(labelsize=8.6)
    fig.savefig(path_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def process_report(label: str, source_csv: Path, report_dir: Path) -> pd.DataFrame:
    if not source_csv.exists():
        raise FileNotFoundError(f"Missing source CSV: {source_csv}")
    if not report_dir.exists():
        raise FileNotFoundError(f"Missing report directory: {report_dir}")

    specs = parse_lollipop_specs(report_dir)
    if not specs:
        raise ValueError(f"No lollipop PNG files found in {report_dir}")

    sig = load_significant_edges(source_csv)
    output_pairs = []

    for spec in specs:
        edges = select_lollipop_edges(sig, spec)
        first = edges.iloc[0]
        metric = str(first["metric"])
        analysis_view = str(first["analysis_view"])
        fdr_scope = str(first["fdr_scope"])
        metadata = {
            "report": label,
            "source_csv": str(source_csv),
            "lollipop_png": str(spec.png_path),
            "metric": metric,
            "analysis_view": analysis_view,
            "fdr_scope": fdr_scope,
            "lollipop_top_n": spec.top_n,
            "n_counted_edges": int(edges.shape[0]),
        }

        signed_matrices = []
        for direction in DIRECTIONS:
            signed_edges = direction_edges(edges, direction)
            signed_pairs, signed_matrix = pair_count_table(signed_edges)
            signed_stem = f"{spec.metric_slug}__{spec.scope_slug}__all_significant_{direction}_group_edge_counts"
            signed_matrix.to_csv(spec.png_path.with_name(f"{signed_stem}_matrix.csv"))
            signed_pairs.to_csv(spec.png_path.with_name(f"{signed_stem}_pair_counts.csv"), index=False)
            signed_metadata = {**metadata, "edge_direction": direction, "n_counted_edges": int(signed_edges.shape[0])}
            signed_pairs = signed_pairs.assign(**signed_metadata)
            output_pairs.append(signed_pairs)
            signed_matrices.append(signed_matrix)

        paired_stem = f"{spec.metric_slug}__{spec.scope_slug}__all_significant_increase_decrease_group_edge_counts"
        plot_heatmaps(
            signed_matrices,
            spec.png_path.with_name(f"{paired_stem}_heatmap"),
            panel_labels=["Increase", "Decrease"],
        )

    output_df = pd.concat(output_pairs, ignore_index=True)
    output_df = output_df[
        [
            "report",
            "metric",
            "analysis_view",
            "fdr_scope",
            "edge_direction",
            "lollipop_top_n",
            "n_counted_edges",
            "group_i",
            "group_j",
            "n_edges",
            "n_positive",
            "n_negative",
            "lollipop_png",
            "source_csv",
        ]
    ]
    for direction in DIRECTIONS:
        direction_df = output_df.loc[output_df["edge_direction"] == direction].copy()
        direction_df = direction_df[
            [
                "report",
                "metric",
                "analysis_view",
                "fdr_scope",
                "edge_direction",
                "lollipop_top_n",
                "n_counted_edges",
                "group_i",
                "group_j",
                "n_edges",
                "n_positive",
                "n_negative",
                "lollipop_png",
                "source_csv",
            ]
        ]
        direction_df.to_csv(report_dir / f"lollipop_all_significant_{direction}_group_edge_counts.csv", index=False)
        direction_df.to_csv(report_dir / f"lollipop_{direction}_group_edge_counts.csv", index=False)
    return output_df


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        action="append",
        nargs=3,
        metavar=("LABEL", "SOURCE_CSV", "REPORT_DIR"),
        help="Report to quantify. May be repeated. Defaults quantify the main-result and task-activation lollipop folders.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    reports = args.report or DEFAULT_REPORTS
    outputs = []
    for label, source_csv, report_dir in reports:
        pair_df = process_report(str(label), Path(source_csv), Path(report_dir))
        outputs.append(pair_df)
        print(f"Saved sign-split heatmaps under {Path(report_dir)}")

    combined = pd.concat(outputs, ignore_index=True)
    print(combined.groupby(["report", "metric", "edge_direction"], sort=False)["n_edges"].sum().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
