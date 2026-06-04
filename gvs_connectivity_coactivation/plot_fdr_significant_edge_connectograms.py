#!/usr/bin/env python3
"""Plot FDR-significant edge-level FC results as circular connectograms."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, PathPatch, Wedge
from matplotlib.path import Path as MplPath


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ROI_DEF = ROOT / "Claude_results" / "group_analyses" / "analysis3_lme" / "per_trial_roi_betas_roi_definition.csv"
OUT_ROOT = ROOT / "figures" / "GVS_effects" / "main result"
SIG_CSV = OUT_ROOT / "metric_sensitivity" / "fdr_significant_edge_connectivity_metric_sensitivity.csv"
OUT_DIR = OUT_ROOT / "metric_sensitivity" / "connectogram_reports"
TOP_N_EDGES = 30
TOP_NODES = 15
PAPER_FONT_FAMILY = "Liberation Sans"
CONNECTOGRAM_FIGSIZE = (15.8, 5.85)
ROI_LABEL_FONTSIZE = 8.2
GROUP_LABEL_FONTSIZE = 9.4
COLORBAR_FONTSIZE = 9.0
ROI_LABEL_RADIUS = 1.23
GROUP_LABEL_RADIUS = 1.52

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": [PAPER_FONT_FAMILY, "Arial", "Helvetica", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

GROUP_ORDER = [
    "Frontal-Parietal",
    "Subcortical",
    "Cingulate-Temporal",
    "Visual",
    "Limbic/MTL-Olfactory",
    "Somatomotor/Cerebellar",
]

GROUP_COLORS = {
    "Frontal-Parietal": "#7cb342",
    "Subcortical": "#c89b1f",
    "Cingulate-Temporal": "#8e6bbd",
    "Visual": "#d65f9e",
    "Limbic/MTL-Olfactory": "#39a76d",
    "Somatomotor/Cerebellar": "#2db7b5",
}

GROUP_LABELS = {
    "Limbic/MTL-Olfactory": "Limbic-Olfactory",
}

ROI_LABELS = {
    "Amygdala": "Amygdala",
    "Caudate": "Caudate",
    "Cerebellum": "Cerebellum",
    "Cingulate": "Cingulate",
    "Frontal": "Frontal",
    "Fusiform": "Fusiform",
    "Hippocampus": "Hippocampus",
    "Insula": "Insula",
    "Occipital": "Occipital",
    "Olfactory": "Olfactory",
    "Orbitofrontal": "Orbitofrontal",
    "Pallidum": "Pallidum",
    "ParaHippocampal": "Parahippocampal",
    "Paracentral_Lobule": "Paracentral Lobule",
    "Parietal": "Parietal",
    "Postcentral": "Postcentral",
    "Precentral": "Precentral",
    "Putamen": "Putamen",
    "Rolandic_Oper": "Rolandic Operculum",
    "Supp_Motor_Area": "Supplementary Motor Area",
    "Temporal": "Temporal",
    "Thalamus": "Thalamus",
}


def clean_slug(text: str) -> str:
    text = str(text)
    text = re.sub(r"[^A-Za-z0-9]+", "_", text)
    return text.strip("_").lower()


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


def roi_base(roi: str) -> str:
    return str(roi).rsplit("_", 1)[0]


def roi_side(roi: str) -> str:
    if str(roi).endswith("_L"):
        return "L"
    if str(roi).endswith("_R"):
        return "R"
    return ""


def roi_display_label(roi: str) -> str:
    base = roi_base(roi)
    return ROI_LABELS.get(base, base.replace("_", " "))


def roi_sort_key(roi: str) -> tuple[int, str, str]:
    return GROUP_ORDER.index(roi_group(roi)), roi_base(roi), str(roi)


def add_side_layout(
    side_rois: list[str],
    start_deg: float,
    end_deg: float,
    angles: dict[str, float],
    sectors: list[tuple[str, float, float]],
) -> None:
    if not side_rois:
        return

    positions = np.deg2rad(np.linspace(start_deg, end_deg, len(side_rois)))
    step = positions[1] - positions[0] if len(positions) > 1 else np.deg2rad(end_deg - start_deg)
    if np.isclose(step, 0.0):
        step = np.deg2rad(7.0)

    for roi, angle in zip(side_rois, positions):
        angles[roi] = float(angle)

    start_idx = 0
    while start_idx < len(side_rois):
        group = roi_group(side_rois[start_idx])
        stop_idx = start_idx + 1
        while stop_idx < len(side_rois) and roi_group(side_rois[stop_idx]) == group:
            stop_idx += 1
        theta1 = positions[start_idx] - step / 2
        theta2 = positions[stop_idx - 1] + step / 2
        sectors.append((group, float(min(theta1, theta2)), float(max(theta1, theta2))))
        start_idx = stop_idx


def circular_layout(rois: list[str]) -> tuple[list[str], dict[str, float], list[tuple[str, float, float]]]:
    left_rois = sorted([roi for roi in rois if roi_side(roi) == "L"], key=roi_sort_key)
    right_rois = sorted([roi for roi in rois if roi_side(roi) == "R"], key=roi_sort_key)
    midline_rois = sorted([roi for roi in rois if roi_side(roi) not in {"L", "R"}], key=roi_sort_key)
    ordered = left_rois + right_rois + midline_rois

    angles: dict[str, float] = {}
    sectors: list[tuple[str, float, float]] = []
    add_side_layout(left_rois, 112.0, 248.0, angles, sectors)
    add_side_layout(right_rois, 68.0, -68.0, angles, sectors)
    add_side_layout(midline_rois, 268.0, 272.0, angles, sectors)
    return ordered, angles, sectors


def pol2cart(theta: float, radius: float = 1.0) -> np.ndarray:
    return np.array([radius * np.cos(theta), radius * np.sin(theta)])


def draw_chord(ax: plt.Axes, theta1: float, theta2: float, color: str, width: float, alpha: float) -> None:
    p1 = pol2cart(theta1, 0.93)
    p2 = pol2cart(theta2, 0.93)
    c1 = p1 * 0.18
    c2 = p2 * 0.18
    path = MplPath([p1, c1, c2, p2], [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4])
    patch = PathPatch(path, facecolor="none", edgecolor=color, linewidth=width, alpha=alpha, capstyle="round", zorder=1)
    ax.add_patch(patch)


def label_rotation(theta: float) -> tuple[float, str]:
    deg = np.rad2deg(theta)
    rot = deg + 180 if np.cos(theta) < 0 else deg
    rot = ((rot + 180) % 360) - 180
    if rot > 90:
        rot -= 180
    if rot < -90:
        rot += 180
    ha = "right" if np.cos(theta) < 0 else "left"
    return rot, ha


def sector_label_rotation(theta: float) -> tuple[float, str]:
    x_direction = np.cos(theta)
    if abs(x_direction) < 0.25:
        ha = "center"
    else:
        ha = "right" if x_direction < 0 else "left"
    return 0.0, ha


def draw_connectogram_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    ordered_rois: list[str],
    angles: dict[str, float],
    sectors: list[tuple[str, float, float]],
    cmap: matplotlib.colors.Colormap,
    norm: Normalize,
    max_abs: float,
    sparse_edges: bool,
    show_sector_labels: bool = True,
) -> None:
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(Circle((0, 0), 0.91, facecolor="white", edgecolor="#D0D0D0", linewidth=1.0, zorder=0))

    ordered_edges = df.sort_values("abs_mean", ascending=True)
    for row in ordered_edges.itertuples(index=False):
        theta1 = angles.get(row.roi_i)
        theta2 = angles.get(row.roi_j)
        if theta1 is None or theta2 is None:
            continue
        value = float(row.mean)
        scale = min(abs(value) / max_abs, 1.0)
        color = cmap(norm(abs(value)))
        width = (1.1 + 3.4 * scale) if sparse_edges else (0.35 + 2.8 * scale)
        alpha = (0.62 + 0.28 * scale) if sparse_edges else (0.16 + 0.34 * scale)
        draw_chord(ax, theta1, theta2, color, width, alpha)

    for group, theta1, theta2 in sectors:
        mid = (theta1 + theta2) / 2
        wedge = Wedge(
            (0, 0),
            1.17,
            np.rad2deg(theta1),
            np.rad2deg(theta2),
            width=0.075,
            facecolor=GROUP_COLORS[group],
            edgecolor="white",
            linewidth=1.4,
            alpha=0.95,
            zorder=3,
        )
        ax.add_patch(wedge)
        if not show_sector_labels or np.cos(mid) > 0.2:
            continue
        rot, ha = sector_label_rotation(mid)
        label_point = pol2cart(mid, GROUP_LABEL_RADIUS)
        if group == "Frontal-Parietal":
            label_point[1] -= 0.10
        ax.text(
            *label_point,
            GROUP_LABELS.get(group, group),
            rotation=rot,
            rotation_mode="anchor",
            ha=ha,
            va="center",
            fontsize=GROUP_LABEL_FONTSIZE,
            color=GROUP_COLORS[group],
        )

    for roi in ordered_rois:
        theta = angles[roi]
        group = roi_group(roi)
        p = pol2cart(theta, 1.01)
        ax.scatter([p[0]], [p[1]], s=42, color=GROUP_COLORS[group], edgecolor="white", linewidth=0.8, zorder=5)
        rot, ha = label_rotation(theta)
        ax.text(
            *pol2cart(theta, ROI_LABEL_RADIUS),
            roi_display_label(roi),
            rotation=rot,
            rotation_mode="anchor",
            ha=ha,
            va="center",
            fontsize=ROI_LABEL_FONTSIZE,
            color="#222222",
        )

    ax.set_xlim(-1.92, 1.92)
    ax.set_ylim(-1.42, 1.42)


def panel_abs_limits(df: pd.DataFrame) -> tuple[float, float]:
    values = pd.to_numeric(df["mean"], errors="coerce").abs().to_numpy(dtype=float)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return 0.0, 1.0
    vmin = float(np.nanmin(finite_values))
    vmax = float(np.nanmax(finite_values))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return 0.0, 1.0
    if np.isclose(vmin, vmax):
        pad = max(abs(vmax) * 0.05, 1e-6)
        vmin = max(0.0, vmin - pad)
        vmax += pad
    return vmin, vmax


def plot_connectogram(df: pd.DataFrame, roi_order: list[str], path_base: Path) -> None:
    ordered_rois, angles, sectors = circular_layout(roi_order)

    sparse_edges = len(df) <= 30

    panels = [
        (df.loc[df["mean"] > 0].copy(), plt.get_cmap("Oranges"), "Improved connectivity"),
        (df.loc[df["mean"] < 0].copy(), plt.get_cmap("Blues"), "Decreased connectivity"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=CONNECTOGRAM_FIGSIZE)
    fig.subplots_adjust(left=0.0, right=0.968, bottom=0.0, top=1.0, wspace=0.08)
    for panel_idx, (ax, (panel_df, cmap, colorbar_label)) in enumerate(zip(np.atleast_1d(axes), panels, strict=True)):
        vmin, vmax = panel_abs_limits(panel_df)
        norm = Normalize(vmin=vmin, vmax=vmax)
        max_abs = float(np.nanpercentile(panel_df["abs_mean"].astype(float), 95)) if panel_df["abs_mean"].notna().any() else 1.0
        if not np.isfinite(max_abs) or max_abs <= 0:
            max_abs = 1.0
        draw_connectogram_panel(
            ax,
            panel_df,
            ordered_rois,
            angles,
            sectors,
            cmap,
            norm,
            max_abs,
            sparse_edges,
            show_sector_labels=panel_idx == 0,
        )
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        pos = ax.get_position()
        cax = fig.add_axes([pos.x1 + 0.004, pos.y0 + pos.height * 0.2, 0.011, pos.height * 0.6])
        cbar = fig.colorbar(sm, cax=cax)
        cbar.ax.set_title(colorbar_label, fontsize=COLORBAR_FONTSIZE, pad=5)
        cbar.ax.tick_params(labelsize=COLORBAR_FONTSIZE - 1.0)

    fig.savefig(path_base.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.0)
    fig.savefig(path_base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)


def node_involvement(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in df.itertuples(index=False):
        for roi in [row.roi_i, row.roi_j]:
            rows.append(
                {
                    "roi": roi,
                    "group": roi_group(roi),
                    "edge_label": row.edge_label,
                    "mean": float(row.mean),
                    "abs_mean": abs(float(row.mean)),
                    "q_fdr": float(row.q_fdr),
                    "positive": float(row.mean) > 0,
                }
            )
    long = pd.DataFrame(rows)
    if long.empty:
        return long
    out = (
        long.groupby(["roi", "group"], dropna=False)
        .agg(
            n_sig_edges=("edge_label", "count"),
            n_positive=("positive", "sum"),
            total_abs_delta=("abs_mean", "sum"),
            mean_abs_delta=("abs_mean", "mean"),
            signed_delta_sum=("mean", "sum"),
            min_q_fdr=("q_fdr", "min"),
        )
        .reset_index()
    )
    out["n_negative"] = out["n_sig_edges"] - out["n_positive"]
    return out.sort_values(["total_abs_delta", "n_sig_edges", "mean_abs_delta"], ascending=False)


def plot_node_involvement(node_df: pd.DataFrame, path_base: Path, title: str, subtitle: str) -> None:
    plot_df = node_df.head(TOP_NODES).iloc[::-1].copy()
    fig, ax = plt.subplots(figsize=(8.6, 6.0))
    colors = [GROUP_COLORS.get(group, "#808080") for group in plot_df["group"]]
    ax.barh(plot_df["roi"].str.replace("_", " ", regex=False), plot_df["total_abs_delta"], color=colors, alpha=0.9)
    for y, row in enumerate(plot_df.itertuples(index=False)):
        ax.text(
            row.total_abs_delta,
            y,
            f"  {int(row.n_sig_edges)} edges",
            va="center",
            ha="left",
            fontsize=8.5,
            color="#333333",
        )
    ax.set_xlabel("Sum of absolute significant edge deltas")
    fig.text(0.08, 0.975, title, ha="left", va="top", fontsize=13, fontweight="bold")
    fig.text(0.08, 0.935, subtitle, ha="left", va="top", fontsize=9.5, color="#555555")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#E0E0E0", linewidth=0.8)
    ax.set_axisbelow(True)
    xmax = float(plot_df["total_abs_delta"].max()) if not plot_df.empty else 1.0
    ax.set_xlim(0, xmax * 1.32 if xmax > 0 else 1.0)
    legend_handles = [
        Line2D([0], [0], marker="s", linestyle="none", markersize=9, markerfacecolor=color, markeredgecolor="none", label=group)
        for group, color in GROUP_COLORS.items()
        if group in set(plot_df["group"])
    ]
    ax.legend(handles=legend_handles, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=8.5)
    fig.subplots_adjust(left=0.24, right=0.76, top=0.84, bottom=0.12)
    fig.savefig(path_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sig = pd.read_csv(SIG_CSV, low_memory=False)
    sig = sig.loc[sig["sig_fdr"].astype(bool)].copy()
    sig["mean"] = pd.to_numeric(sig["mean"], errors="coerce")
    sig["q_fdr"] = pd.to_numeric(sig["q_fdr"], errors="coerce")
    sig["abs_mean"] = sig["mean"].abs()
    roi_order = pd.read_csv(ROI_DEF)["roi_label"].astype(str).tolist()

    summary_rows = []
    top_rows = []
    plot_rows = []
    group_cols = ["metric", "metric_family", "analysis_view", "fdr_scope"]
    for key, group in sig.groupby(group_cols, sort=False, dropna=False):
        metric, metric_family, analysis_view, fdr_scope = key
        scope_slug = clean_slug(f"{analysis_view}_{fdr_scope}")
        path_base = OUT_DIR / f"{clean_slug(metric)}__{scope_slug}"
        top_path_base = OUT_DIR / f"{clean_slug(metric)}__{scope_slug}__top{TOP_N_EDGES}_edges"
        node_path_base = OUT_DIR / f"{clean_slug(metric)}__{scope_slug}__roi_involvement"
        subtitle = f"{analysis_view} | {fdr_scope}"
        plot_connectogram(group, roi_order, path_base)
        top = group.sort_values(["q_fdr", "p_signflip", "abs_mean"], ascending=[True, True, False]).head(TOP_N_EDGES).copy()
        plot_connectogram(top, roi_order, top_path_base)
        node_df = node_involvement(group)
        node_df["metric"] = metric
        node_df["metric_family"] = metric_family
        node_df["analysis_view"] = analysis_view
        node_df["fdr_scope"] = fdr_scope
        plot_node_involvement(node_df, node_path_base, f"{metric}: ROIs with strongest FDR edge changes", subtitle)

        n_pos = int((group["mean"] > 0).sum())
        n_neg = int((group["mean"] < 0).sum())
        best = top.iloc[0]
        summary_rows.append(
            {
                "metric": metric,
                "metric_family": metric_family,
                "analysis_view": analysis_view,
                "fdr_scope": fdr_scope,
                "n_sig_edges": int(group.shape[0]),
                "n_positive": n_pos,
                "n_negative": n_neg,
                "min_q_fdr": float(group["q_fdr"].min()),
                "best_edge": best["edge_label"],
                "best_mean_delta": float(best["mean"]),
                "best_q_fdr": float(best["q_fdr"]),
                "plot_png": str(path_base.with_suffix(".png")),
                "plot_pdf": str(path_base.with_suffix(".pdf")),
                "top_edges_plot_png": str(top_path_base.with_suffix(".png")),
                "top_edges_plot_pdf": str(top_path_base.with_suffix(".pdf")),
                "roi_involvement_plot_png": str(node_path_base.with_suffix(".png")),
                "roi_involvement_plot_pdf": str(node_path_base.with_suffix(".pdf")),
            }
        )
        top_rows.append(top)
        plot_rows.append((metric, analysis_view, fdr_scope, path_base.with_suffix(".png"), top_path_base.with_suffix(".png"), node_path_base.with_suffix(".png"), int(group.shape[0])))

    summary = pd.DataFrame(summary_rows).sort_values(["metric", "analysis_view", "fdr_scope"])
    summary.to_csv(OUT_DIR / "fdr_connectogram_summary.csv", index=False)
    top_df = pd.concat(top_rows, ignore_index=True) if top_rows else pd.DataFrame()
    top_cols = [
        "metric",
        "metric_family",
        "analysis_view",
        "fdr_scope",
        "edge_label",
        "roi_i",
        "roi_j",
        "n",
        "mean",
        "t_stat",
        "p_signflip",
        "q_fdr",
        "condition_label",
        "medication",
        "run",
    ]
    top_df[[c for c in top_cols if c in top_df.columns]].to_csv(OUT_DIR / "fdr_connectogram_top_edges.csv", index=False)
    node_df = pd.concat(
        [
            node_involvement(group).assign(metric=metric, metric_family=metric_family, analysis_view=analysis_view, fdr_scope=fdr_scope)
            for (metric, metric_family, analysis_view, fdr_scope), group in sig.groupby(group_cols, sort=False, dropna=False)
        ],
        ignore_index=True,
    )
    node_df.to_csv(OUT_DIR / "fdr_connectogram_roi_involvement.csv", index=False)

    lines = [
        "# FDR-Significant Edge Connectograms",
        "",
        f"Input: `{SIG_CSV}`",
        "",
        "Edges are grouped into broad anatomical sectors inferred from ROI names. Chords show FDR-significant active-GVS minus sham edge changes.",
        "Improved edges are shown in the Oranges panel and decreased edges are shown in the Blues panel; each panel uses its own absolute connectivity-change magnitude range.",
        "",
        "Note: the all-subjects/block-pool views are exploratory because repeated blocks from the same subject are treated as separate observations.",
        "",
        "## Sector Mapping",
        "",
    ]
    for group_name in GROUP_COLORS:
        rois = [r for r in roi_order if roi_group(r) == group_name]
        lines.append(f"- {group_name}: {', '.join(rois)}")
    lines.extend(["", "## Recommended Paper Figures", ""])
    lines.append(f"- Use the `__top{TOP_N_EDGES}_edges` connectograms for readable edge-level figures with separated improved/decreased panels.")
    lines.append("- Use the `__roi_involvement` bar plots to show which ROIs changed most.")
    lines.append("- Keep the full connectograms as supplementary figures because dense methods have many FDR-significant edges.")
    lines.extend(["", "## Plots", ""])
    for row in summary.itertuples(index=False):
        rel_png = Path(row.plot_png).relative_to(OUT_ROOT)
        top_png = Path(row.top_edges_plot_png).relative_to(OUT_ROOT)
        node_png = Path(row.roi_involvement_plot_png).relative_to(OUT_ROOT)
        lines.append(
            f"- `{row.metric}` | `{row.analysis_view}` | `{row.fdr_scope}`: "
            f"{row.n_sig_edges} edges ({row.n_positive} positive, {row.n_negative} negative), "
            f"min q={row.min_q_fdr:.4g}. Full: `{rel_png}`; top edges: `{top_png}`; ROI involvement: `{node_png}`"
        )
    lines.extend(["", "## Output Tables", ""])
    lines.append("- `fdr_connectogram_summary.csv`: one row per method/scope plot.")
    lines.append(f"- `fdr_connectogram_top_edges.csv`: top {TOP_N_EDGES} FDR-significant edges per method/scope.")
    lines.append("- `fdr_connectogram_roi_involvement.csv`: ROI-level significant-edge counts and summed absolute deltas.")
    (OUT_DIR / "FDR_CONNECTOGRAM_REPORT.md").write_text("\n".join(lines))
    print(summary.to_string(index=False), flush=True)
    print(f"Saved connectogram report under {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
