#!/usr/bin/env python3
"""Threshold-robustness analysis for the final voxel-weight network."""

from __future__ import annotations

import argparse
import json
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import datasets, image
from scipy import ndimage


DEFAULT_MAP = Path(
    "data/voxel_weights_task1_bold0.6_beta0.6_smooth1.25_gamma1.5.nii.gz"
)
DEFAULT_OUT_BASE = Path(
    "figures/voxel_weights_task1_bold0.6_beta0.6_smooth1.25_gamma1.5_threshold_robustness"
)
DEFAULT_THRESHOLDS = (80.0, 85.0, 90.0, 95.0, 97.5)
MAIN_THRESHOLDS = (85.0, 90.0, 95.0)
REFERENCE_THRESHOLD = 90.0
DEFAULT_MIN_REPORT_VOXELS = 25
DEFAULT_MIDLINE_BAND_MM = 1.0
DEFAULT_AAL_VERSION = "3v2"
DEFAULT_ATLAS_NAME = "AAL3v2 (Automated Anatomical Labeling 3)"
UNASSIGNED_ROI = "Unassigned Active Voxels"
BILATERAL_HEMISPHERE_LABEL = "Bilateral"
EXCLUDED_MAIN_PLOT_REGIONS = {"N_Acc"}
EXCLUDED_ATLAS_LABEL_REGIONS = {"Olfactory"}
COARSE_AAL_GROUPS = (
    ("Cerebellum", ("Cerebellum", "Vermis"), ()),
    ("Temporal", ("Temporal",), ("Heschl",)),
    ("Occipital", ("Occipital",), ("Calcarine", "Cuneus", "Lingual")),
    ("Parietal", ("Parietal",), ("Angular", "SupraMarginal", "Precuneus")),
    ("Frontal", ("Frontal",), ()),
    ("Orbitofrontal", ("OFC",), ("Rectus",)),
    ("Cingulate", ("Cingulate", "ACC"), ()),
    ("Thalamus", ("Thal",), ()),
    ("Raphe", ("Raphe",), ()),
)


@dataclass(frozen=True)
class ROIGroup:
    name: str
    source: str
    mask: np.ndarray
    matched_labels: tuple[str, ...]


def _pct_label(percentile: float) -> str:
    return f"p{percentile:g}".replace(".", "p")


def _coarse_aal_group_name(label_name: str) -> str:
    name = re.sub(r"_(L|R)$", "", label_name)
    for group_name, prefixes, exact_names in COARSE_AAL_GROUPS:
        if name in exact_names or any(name.startswith(prefix) for prefix in prefixes):
            return group_name
    return name


def _resample_label_img(label_img: nib.Nifti1Image, reference_img: nib.Nifti1Image) -> np.ndarray:
    if label_img.shape[:3] == reference_img.shape[:3] and np.allclose(label_img.affine, reference_img.affine):
        return np.rint(label_img.get_fdata()).astype(np.int32, copy=False)
    resampled = image.resample_to_img(
        label_img,
        reference_img,
        interpolation="nearest",
        force_resample=True,
        copy_header=True,
    )
    return np.rint(resampled.get_fdata()).astype(np.int32, copy=False)


def _build_roi_groups(
    reference_img: nib.Nifti1Image,
    aal_version: str,
    cache_dir: Path | None,
) -> tuple[list[ROIGroup], dict[str, object]]:
    data_dir = str(cache_dir) if cache_dir is not None else None
    atlas = datasets.fetch_atlas_aal(version=aal_version, data_dir=data_dir, verbose=0)
    atlas_img = atlas.maps if isinstance(atlas.maps, nib.Nifti1Image) else nib.load(atlas.maps)
    atlas_data = _resample_label_img(atlas_img, reference_img)
    atlas_source = f"AAL3v2 ({Path(str(atlas.maps)).name})" if aal_version == "3v2" else f"AAL {aal_version}"
    label_pairs = [
        (int(label_value), str(label_name))
        for label_value, label_name in zip(atlas.indices, atlas.labels)
        if int(label_value) != 0 and str(label_name).lower() != "background"
    ]
    group_masks: dict[str, np.ndarray] = {}
    group_labels: dict[str, list[str]] = {}
    for label_value, name in label_pairs:
        mask = atlas_data == label_value
        if not np.any(mask):
            continue
        group_name = _coarse_aal_group_name(name)
        if group_name in group_masks:
            group_masks[group_name] |= mask
            group_labels[group_name].append(name)
        else:
            group_masks[group_name] = mask.copy()
            group_labels[group_name] = [name]
    groups = [
        ROIGroup(
            name=name,
            source=atlas_source,
            mask=group_masks[name],
            matched_labels=tuple(group_labels[name]),
        )
        for name in group_masks
    ]

    metadata = {
        "roi_definition": "aal3_bilateral_coarse_anatomical_groups",
        "priority_order": [group.name for group in groups] + [UNASSIGNED_ROI],
        "atlas_info": {
            "name": DEFAULT_ATLAS_NAME if aal_version == "3v2" else f"AAL {aal_version}",
            "description": atlas_source,
            "version": aal_version,
            "map": str(atlas.maps),
            "n_labels": len(label_pairs),
            "n_regions": len(groups),
            "grouping": "Original AAL labels merged bilaterally into coarse anatomical groups; atlas map unchanged.",
        },
        "roi_sources": {group.name: group.source for group in groups},
        "roi_matched_labels": {group.name: group.matched_labels for group in groups},
    }
    return groups, metadata


def _assign_threshold_regions(
    weights: np.ndarray,
    affine: np.ndarray,
    mask: np.ndarray,
    groups: list[ROIGroup],
    percentile: float,
    threshold_value: float,
    min_report_voxels: int,
) -> pd.DataFrame:
    selected_ijk = np.column_stack(np.nonzero(mask)).astype(np.int32, copy=False)
    if selected_ijk.size == 0:
        return pd.DataFrame()

    x, y, z = selected_ijk.T
    assigned = np.zeros(selected_ijk.shape[0], dtype=np.int16)
    group_names = [UNASSIGNED_ROI] + [group.name for group in groups]
    group_sources = {UNASSIGNED_ROI: "Outside atlas labels"}

    for group_id, group in enumerate(groups, start=1):
        hit = group.mask[x, y, z] & (assigned == 0)
        assigned[hit] = group_id
        group_sources[group.name] = group.source

    coords_mm = nib.affines.apply_affine(affine, selected_ijk)
    selected_weights = weights[x, y, z]
    rows: list[dict[str, object]] = []
    for group_id in np.unique(assigned):
        positions = np.flatnonzero(assigned == group_id)
        if positions.size == 0:
            continue
        roi_name = group_names[int(group_id)]
        coords = coords_mm[positions]
        values = selected_weights[positions]
        x_mean = float(np.mean(coords[:, 0]))
        hemisphere = "NA" if roi_name == UNASSIGNED_ROI else BILATERAL_HEMISPHERE_LABEL
        rows.append(
            {
                "percentile": float(percentile),
                "threshold_label": _pct_label(percentile),
                "threshold_value": float(threshold_value),
                "roi_name": roi_name,
                "hemisphere": hemisphere,
                "node_name": roi_name,
                "n_voxels": int(positions.size),
                "present_for_report": bool(positions.size >= min_report_voxels and roi_name != UNASSIGNED_ROI),
                "mean_weight": float(np.mean(values)),
                "max_weight": float(np.max(values)),
                "x_mm": x_mean,
                "y_mm": float(np.mean(coords[:, 1])),
                "z_mm": float(np.mean(coords[:, 2])),
                "source": group_sources[roi_name],
            }
        )
    return pd.DataFrame(rows)


def _summarize_threshold(
    mask: np.ndarray,
    threshold_masks: dict[float, np.ndarray],
    region_df: pd.DataFrame,
    percentile: float,
    threshold_value: float,
    min_report_voxels: int,
    reference_regions: set[str],
) -> dict[str, object]:
    labels, n_components = ndimage.label(mask)
    component_sizes = np.bincount(labels.ravel())
    largest_component = int(component_sizes[1:].max()) if component_sizes.size > 1 else 0
    reference_mask = threshold_masks[REFERENCE_THRESHOLD]
    intersection = int(np.count_nonzero(mask & reference_mask))
    union = int(np.count_nonzero(mask | reference_mask))
    reportable = (region_df["n_voxels"] >= min_report_voxels) & ~region_df["roi_name"].eq(UNASSIGNED_ROI)
    present_nodes = set(region_df.loc[reportable, "node_name"])
    unassigned_voxels = int(region_df.loc[region_df["roi_name"].eq(UNASSIGNED_ROI), "n_voxels"].sum())
    n_voxels = int(np.count_nonzero(mask))
    region_union = present_nodes | reference_regions

    return {
        "percentile": float(percentile),
        "threshold_label": _pct_label(percentile),
        "threshold_value": float(threshold_value),
        "n_voxels": n_voxels,
        "n_unassigned_voxels": unassigned_voxels,
        "atlas_assigned_fraction": float(1.0 - (unassigned_voxels / n_voxels)) if n_voxels else np.nan,
        "n_components": int(n_components),
        "largest_component_voxels": largest_component,
        "n_reportable_nodes": int(len(present_nodes)),
        "jaccard_vs_p90": float(intersection / union) if union else np.nan,
        "p90_voxels_retained": float(intersection / np.count_nonzero(reference_mask)) if np.any(reference_mask) else np.nan,
        "threshold_voxels_in_p90": float(intersection / np.count_nonzero(mask)) if np.any(mask) else np.nan,
        "node_jaccard_vs_p90": float(len(present_nodes & reference_regions) / len(region_union)) if region_union else np.nan,
        "p90_nodes_retained": float(len(present_nodes & reference_regions) / len(reference_regions)) if reference_regions else np.nan,
    }


def _make_group_label_data(groups: list[ROIGroup], shape: tuple[int, int, int]) -> np.ndarray:
    label_data = np.zeros(shape, dtype=np.int16)
    for label_id, group in enumerate(groups, start=1):
        label_data[group.mask] = label_id
    return label_data


def _mode_projection(label_data: np.ndarray, axis: int) -> np.ndarray:
    moved = np.moveaxis(label_data, axis, 0)
    projected = np.zeros(moved.shape[1:], dtype=np.int16)
    for idx in np.ndindex(projected.shape):
        values = moved[(slice(None),) + idx]
        values = values[values > 0]
        if values.size:
            projected[idx] = int(np.bincount(values).argmax())
    return projected


def _axis_values_mm(affine: np.ndarray, shape: tuple[int, int, int], axis: int) -> np.ndarray:
    ijk = np.zeros((shape[axis], 3), dtype=float)
    ijk[:, axis] = np.arange(shape[axis])
    return nib.affines.apply_affine(affine, ijk)[:, axis]


def _orient_panel(data: np.ndarray, x_values: np.ndarray, y_values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    out = data.copy()
    x = x_values.copy()
    y = y_values.copy()
    if x[0] > x[-1]:
        out = out[::-1, :]
        x = x[::-1]
    if y[0] > y[-1]:
        out = out[:, ::-1]
        y = y[::-1]
    return out, x, y


def _plot_projection(
    ax: plt.Axes,
    data: np.ndarray,
    x_values: np.ndarray,
    y_values: np.ndarray,
    title: str,
    cmap: ListedColormap,
    norm: BoundaryNorm,
    square_span_mm: float | None = None,
) -> None:
    oriented, x, y = _orient_panel(data, x_values, y_values)
    ax.imshow(
        oriented.T,
        origin="lower",
        extent=[float(x[0]), float(x[-1]), float(y[0]), float(y[-1])],
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
    )
    if square_span_mm is not None:
        span = max(square_span_mm, abs(float(x[-1] - x[0])), abs(float(y[-1] - y[0])))
        x_mid = float((x[0] + x[-1]) / 2.0)
        y_mid = float((y[0] + y[-1]) / 2.0)
        half_span = span / 2.0
        ax.set_xlim(x_mid - half_span, x_mid + half_span)
        ax.set_ylim(y_mid - half_span, y_mid + half_span)
    ax.set_title(title, fontsize=11, weight="bold")
    ax.set_aspect("equal", adjustable="box")
    ax.set_box_aspect(1.0)
    ax.set_facecolor("#f5f7fb")
    ax.tick_params(labelsize=8)
    ax.grid(color="white", linewidth=0.3, alpha=0.35)


def _display_region_name(name: str) -> str:
    return name.replace("_", " ")


COMPACT_REGION_LABELS = {
    "Supp_Motor_Area": "Supp. motor",
    "ParaHippocampal": "Parahipp.",
    "Paracentral_Lobule": "Paracentral",
    "Rolandic_Oper": "Rolandic op.",
    "Orbitofrontal": "OFC",
}


def _compact_region_label(name: str, _count: int) -> str:
    return COMPACT_REGION_LABELS.get(name, _display_region_name(name))


def _reference_region_counts(region_df: pd.DataFrame, min_report_voxels: int) -> dict[str, int]:
    rows = region_df[
        np.isclose(region_df["percentile"], REFERENCE_THRESHOLD)
        & (region_df["n_voxels"] >= min_report_voxels)
        & ~region_df["roi_name"].eq(UNASSIGNED_ROI)
    ].sort_values("n_voxels", ascending=False)
    return {str(row.roi_name): int(row.n_voxels) for row in rows.itertuples(index=False)}


def _plot_atlas_regions(
    groups: list[ROIGroup],
    reference_img: nib.Nifti1Image,
    metadata: dict[str, object],
    out_base: Path,
    region_df: pd.DataFrame,
    min_report_voxels: int,
) -> None:
    shape = reference_img.shape[:3]
    reference_counts = _reference_region_counts(region_df, min_report_voxels)
    selected_groups = [group for group in groups if group.name in reference_counts]
    if not selected_groups:
        selected_groups = groups
    selected_groups = sorted(selected_groups, key=lambda group: reference_counts.get(group.name, 0), reverse=True)
    label_data = _make_group_label_data(selected_groups, shape)
    colors = plt.get_cmap("turbo")(np.linspace(0.04, 0.96, max(1, len(selected_groups))))[:, :3]
    group_colors = {group.name: colors[idx] for idx, group in enumerate(selected_groups)}
    labeled_groups = [group for group in selected_groups if group.name not in EXCLUDED_ATLAS_LABEL_REGIONS]
    cmap = ListedColormap(np.vstack([[0.96, 0.97, 0.99], colors]))
    norm = BoundaryNorm(np.arange(-0.5, len(selected_groups) + 1.5, 1), cmap.N)

    x_values = _axis_values_mm(reference_img.affine, shape, 0)
    y_values = _axis_values_mm(reference_img.affine, shape, 1)
    z_values = _axis_values_mm(reference_img.affine, shape, 2)
    projections = {
        "Axial projection": (_mode_projection(label_data, axis=2), x_values, y_values),
        "Coronal projection": (_mode_projection(label_data, axis=1), x_values, z_values),
        "Sagittal projection": (_mode_projection(label_data, axis=0), y_values, z_values),
    }

    projection_spans = [
        max(abs(float(x_axis[-1] - x_axis[0])), abs(float(y_axis[-1] - y_axis[0])))
        for _, x_axis, y_axis in projections.values()
    ]
    square_span_mm = max(projection_spans)

    fig = plt.figure(figsize=(10.8, 5.05), facecolor="white")
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.18], hspace=0.08, wspace=0.20)

    axes = [fig.add_subplot(gs[0, idx]) for idx in range(3)]
    for ax, (title, (projection, x_axis, y_axis)) in zip(axes, projections.items()):
        _plot_projection(ax, projection, x_axis, y_axis, title, cmap, norm, square_span_mm)

    centers: dict[str, np.ndarray] = {}
    for group in selected_groups:
        ijk = np.column_stack(np.nonzero(group.mask))
        centers[group.name] = nib.affines.apply_affine(reference_img.affine, ijk).mean(axis=0)

    offsets = [(-22.0, 16.0), (20.0, 14.0), (-18.0, -18.0), (20.0, -16.0), (0.0, 22.0), (0.0, -22.0)]
    for rank, group in enumerate(labeled_groups[:28]):
        center = centers[group.name]
        name_lower = group.name.lower()
        panel_idx = 2 if "cerebellum" in name_lower or "vermis" in name_lower else 1 if any(
            key in name_lower for key in ("thal", "caudate", "putamen", "pallidum", "amygdala", "hippocampus")
        ) else 0
        if panel_idx == 0:
            xy = (center[0], center[1])
        elif panel_idx == 1:
            xy = (center[0], center[2])
        else:
            xy = (center[1], center[2])
        dx, dy = offsets[rank % len(offsets)]
        axes[panel_idx].annotate(
            "\n".join(textwrap.wrap(_display_region_name(group.name), width=18)),
            xy=xy,
            xytext=(xy[0] + dx, xy[1] + dy),
            fontsize=5.5,
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.16", "facecolor": "white", "edgecolor": "#334155", "linewidth": 0.35, "alpha": 0.88},
            arrowprops={"arrowstyle": "-", "color": "#334155", "linewidth": 0.45, "alpha": 0.70},
        )

    legend_ax = fig.add_subplot(gs[1, :])
    legend_ax.axis("off")
    n_cols = 6
    rows_per_col = max(1, int(np.ceil(len(labeled_groups) / n_cols)))
    row_step = 0.31
    for idx, group in enumerate(labeled_groups, start=1):
        col = (idx - 1) // rows_per_col
        row = (idx - 1) % rows_per_col
        x = (col + 0.02) / n_cols
        y = 0.78 - row * row_step
        label = _compact_region_label(group.name, reference_counts.get(group.name, 0))
        legend_ax.add_patch(plt.Rectangle((x, y - 0.040), 0.010, 0.080, color=group_colors[group.name], transform=legend_ax.transAxes))
        legend_ax.text(x + 0.014, y, label, fontsize=7.0, va="center")

    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{out_base}_atlas_regions.png", dpi=220, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(f"{out_base}_atlas_regions.pdf", bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def _plot_robustness(summary_df: pd.DataFrame, region_df: pd.DataFrame, out_base: Path, min_report_voxels: int) -> None:
    assigned_regions = region_df.loc[~region_df["roi_name"].eq(UNASSIGNED_ROI)].copy()
    ever_reportable_nodes = set(assigned_regions.loc[assigned_regions["n_voxels"] >= min_report_voxels, "node_name"])
    assigned_regions = assigned_regions.loc[assigned_regions["node_name"].isin(ever_reportable_nodes)]
    assigned_regions = assigned_regions.loc[~assigned_regions["node_name"].isin(EXCLUDED_MAIN_PLOT_REGIONS)]
    p90_counts = (
        assigned_regions.loc[np.isclose(assigned_regions["percentile"], REFERENCE_THRESHOLD), ["node_name", "n_voxels"]]
        .set_index("node_name")["n_voxels"]
        .to_dict()
    )
    order_df = (
        assigned_regions.groupby("node_name", as_index=False)["n_voxels"]
        .max()
        .assign(p90_count=lambda df: df["node_name"].map(p90_counts).fillna(0))
        .sort_values(["p90_count", "n_voxels", "node_name"], ascending=[False, False, True])
    )
    node_order = order_df["node_name"].tolist()
    pivot = (
        assigned_regions.pivot_table(index="node_name", columns="threshold_label", values="n_voxels", aggfunc="sum", fill_value=0)
        .reindex(index=node_order, columns=[_pct_label(p) for p in summary_df["percentile"]])
        .fillna(0)
        .astype(int)
    )
    reportable = pivot >= min_report_voxels

    fig = plt.figure(figsize=(10.8, max(7.6, 0.24 * max(1, len(node_order)) + 2.8)), facecolor="white")
    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.0, 1.0, 0.028],
        height_ratios=[1.35, max(2.2, 0.17 * max(1, len(node_order)))],
        hspace=0.34,
        wspace=0.015,
    )
    ax_vox = fig.add_subplot(gs[0, 0])
    ax_stable = fig.add_subplot(gs[0, 1])
    ax_heat = fig.add_subplot(gs[1, :2])
    cax = fig.add_subplot(gs[1, 2])
    top_pos = ax_vox.get_position()
    top_left = 0.055
    top_gap = 0.095
    top_right = cax.get_position().x1
    top_width = (top_right - top_left - top_gap) / 2.0
    ax_vox.set_position([top_left, top_pos.y0, top_width, top_pos.height])
    ax_stable.set_position([top_left + top_width + top_gap, top_pos.y0, top_width, top_pos.height])

    ax_vox.plot(summary_df["percentile"], summary_df["n_voxels"], marker="o", color="#2563eb", linewidth=2.0)
    ax_vox.set_xlabel("Percentile threshold")
    ax_vox.set_ylabel("Voxels")
    ax_vox.grid(alpha=0.25)
    ax_vox.axvline(REFERENCE_THRESHOLD, color="#dc2626", linestyle="--", linewidth=1.3)

    ax_stable.plot(summary_df["percentile"], summary_df["p90_nodes_retained"], marker="o", color="#0f766e", label="p90 groups retained")
    ax_stable.plot(summary_df["percentile"], summary_df["node_jaccard_vs_p90"], marker="s", color="#7c3aed", label="group Jaccard vs p90")
    ax_stable.set_xlabel("Percentile threshold")
    ax_stable.set_ylabel("Stability")
    ax_stable.set_ylim(-0.03, 1.03)
    ax_stable.grid(alpha=0.25)
    ax_stable.axvline(REFERENCE_THRESHOLD, color="#dc2626", linestyle="--", linewidth=1.3)
    ax_stable.legend(frameon=False, fontsize=9, loc="lower left")

    heat = np.log10(pivot.to_numpy(dtype=float) + 1.0)
    im = ax_heat.imshow(heat, aspect="auto", cmap="viridis", vmin=0)
    ax_heat.set_xticks(np.arange(pivot.shape[1]))
    ax_heat.set_xticklabels(pivot.columns.tolist())
    ax_heat.set_yticks(np.arange(pivot.shape[0]))
    ax_heat.set_yticklabels([_display_region_name(name) for name in pivot.index.tolist()], fontsize=8)
    ax_heat.set_xlabel("Threshold")
    ax_heat.set_title(
        f"Rows are kept if they are reportable at any threshold; lower heatmap colors mark below-threshold cells (<{min_report_voxels} voxels).",
        fontsize=8.5,
        loc="left",
        pad=10,
        color="#374151",
    )
    for row_idx in range(pivot.shape[0]):
        for col_idx in range(pivot.shape[1]):
            if bool(reportable.iat[row_idx, col_idx]):
                value = int(pivot.iat[row_idx, col_idx])
                ax_heat.text(
                    col_idx,
                    row_idx,
                    str(value),
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="black",
                )
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("log10(voxels + 1)")
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{out_base}.png", dpi=220, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(f"{out_base}.pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def _node_set(region_df: pd.DataFrame, percentile: float, min_report_voxels: int) -> set[str]:
    rows = region_df[
        np.isclose(region_df["percentile"], percentile)
        & (region_df["n_voxels"] >= min_report_voxels)
        & ~region_df["roi_name"].eq(UNASSIGNED_ROI)
    ]
    return set(rows["node_name"].astype(str))


def _format_node_list(nodes: list[str], max_items: int = 14) -> str:
    if not nodes:
        return "None"
    shown = nodes[:max_items]
    suffix = "" if len(nodes) <= max_items else f"; plus {len(nodes) - max_items} more"
    return ", ".join(shown) + suffix


def _write_report(
    out_base: Path,
    map_path: Path,
    summary_df: pd.DataFrame,
    region_df: pd.DataFrame,
    metadata: dict[str, object],
    min_report_voxels: int,
) -> None:
    p85_nodes = _node_set(region_df, 85.0, min_report_voxels)
    p90_nodes = _node_set(region_df, 90.0, min_report_voxels)
    p95_nodes = _node_set(region_df, 95.0, min_report_voxels)
    stable_nodes = sorted(p85_nodes & p90_nodes & p95_nodes)
    relaxed_only = sorted(p85_nodes - p90_nodes)
    lost_when_tightened = sorted(p90_nodes - p95_nodes)
    p90_region_counts = (
        region_df[
            np.isclose(region_df["percentile"], REFERENCE_THRESHOLD)
            & (region_df["n_voxels"] >= min_report_voxels)
            & ~region_df["roi_name"].eq(UNASSIGNED_ROI)
        ]
        .sort_values("n_voxels", ascending=False)
    )
    top_p90 = [f"{row.node_name} ({int(row.n_voxels)})" for row in p90_region_counts.itertuples(index=False)]
    ref = summary_df[np.isclose(summary_df["percentile"], REFERENCE_THRESHOLD)].iloc[0]
    p95 = summary_df[np.isclose(summary_df["percentile"], 95.0)].iloc[0]
    p85 = summary_df[np.isclose(summary_df["percentile"], 85.0)].iloc[0]
    atlas_info = metadata.get("atlas_info", {})
    atlas_name = atlas_info.get("description", DEFAULT_ATLAS_NAME) if isinstance(atlas_info, dict) else DEFAULT_ATLAS_NAME

    verdict = (
        "The main network is robust at the anatomical-group level across p85-p95: most p90 groups remain reportable "
        "after tightening to p95, while relaxing to p85 mainly expands already-present nodes."
        if float(p95["p90_nodes_retained"]) >= 0.75
        else "The main network has a stable core, but several p90 anatomical groups are threshold-sensitive when tightened to p95."
    )

    report = f"""# Threshold-Robustness Analysis

Input map: `{map_path}`

Reference visualization: p90-style thresholding of the final voxel-weight network.

## Method

- Thresholded nonzero voxel weights at percentiles: {", ".join(_pct_label(p) for p in summary_df["percentile"])}.
- Used p90 as the reference threshold and treated p85/p95 as the main relaxed/tightened sensitivity range.
- Assigned suprathreshold voxels to the current AAL3 atlas using {atlas_name}.
- Merged left/right AAL labels and repeated subparcels into coarser bilateral anatomical groups; the atlas map itself was not changed.
- Counted a group as reportable when it contained at least {min_report_voxels} suprathreshold voxels.

## Result

{verdict}

At p90, the map contains {int(ref["n_voxels"]):,} suprathreshold voxels, {int(ref["n_components"]):,} connected components, and {int(ref["n_reportable_nodes"]):,} reportable AAL groups. The atlas assigns {float(ref["atlas_assigned_fraction"]):.1%} of p90 suprathreshold voxels; the remaining {int(ref["n_unassigned_voxels"]):,} voxels are kept in the CSV as unassigned but excluded from group-stability counts. Relaxing to p85 gives {int(p85["n_reportable_nodes"]):,} reportable groups; tightening to p95 retains {float(p95["p90_nodes_retained"]):.1%} of the p90 groups.

Stable p85-p95 groups: {_format_node_list(stable_nodes, max_items=40)}

Added when relaxed to p85: {_format_node_list(relaxed_only)}

Dropped below report threshold when tightened to p95: {_format_node_list(lost_when_tightened)}

Largest p90 groups by voxel count: {_format_node_list(top_p90, max_items=12)}

## Suggested Reporting

Report the p90 network as the primary visualization and include the robustness heatmap/table as a supplement. In text, emphasize anatomical-group stability rather than raw voxel overlap, because percentile thresholds are nested by construction. A clear phrasing is:

\"Threshold sensitivity was assessed by repeating the network definition at p85, p90, and p95 of the nonzero voxel-weight distribution. The main p90 network was stable at the bilateral anatomical-group level: the same core AAL3-derived groups persisted across p85-p95, while threshold relaxation mainly expanded the network and threshold tightening removed smaller peripheral groups.\"

Then list the stable core and the threshold-sensitive nodes from the bullets above.

## Outputs

- `{out_base}.png`
- `{out_base}.pdf`
- `{out_base}_atlas_regions.png`
- `{out_base}_atlas_regions.pdf`
- `{out_base}_summary.csv`
- `{out_base}_regions.csv`
- `{out_base}.json`
"""
    Path(f"{out_base}.md").write_text(report, encoding="utf-8")
    Path(f"{out_base}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze threshold robustness of the final voxel-weight network.")
    parser.add_argument("--map", type=Path, default=DEFAULT_MAP, help="Input unthresholded voxel-weight NIfTI map.")
    parser.add_argument("--out-base", type=Path, default=DEFAULT_OUT_BASE, help="Output path stem.")
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=list(DEFAULT_THRESHOLDS),
        help="Percentile thresholds over nonzero finite weights.",
    )
    parser.add_argument(
        "--min-report-voxels",
        type=int,
        default=DEFAULT_MIN_REPORT_VOXELS,
        help="Minimum voxels required for a region node to be considered reportable.",
    )
    parser.add_argument(
        "--midline-band-mm",
        type=float,
        default=DEFAULT_MIDLINE_BAND_MM,
        help="Retained for compatibility; AAL labels are merged bilaterally in the current grouping.",
    )
    parser.add_argument("--aal-version", default=DEFAULT_AAL_VERSION, help="AAL atlas version passed to nilearn.")
    parser.add_argument("--atlas-cache-dir", type=Path, default=None, help="Optional nilearn atlas cache directory.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if REFERENCE_THRESHOLD not in set(float(p) for p in args.thresholds):
        raise ValueError("The threshold list must include p90 because p90 is the reference.")

    img = nib.load(str(args.map))
    weights = np.asarray(img.get_fdata(), dtype=float)
    nonzero = np.isfinite(weights) & (weights != 0)
    if not np.any(nonzero):
        raise ValueError(f"No nonzero finite weights found in {args.map}")

    percentiles = sorted(float(p) for p in args.thresholds)
    values = weights[nonzero]
    threshold_values = {p: float(np.percentile(values, p)) for p in percentiles}
    threshold_masks = {p: nonzero & (weights >= threshold_values[p]) for p in percentiles}
    groups, metadata = _build_roi_groups(img, args.aal_version, args.atlas_cache_dir)
    metadata.update(
        {
            "map": str(args.map),
            "threshold_percentiles": percentiles,
            "reference_threshold": REFERENCE_THRESHOLD,
            "main_threshold_range": MAIN_THRESHOLDS,
            "min_report_voxels": int(args.min_report_voxels),
            "midline_band_mm": float(args.midline_band_mm),
        }
    )

    region_frames = []
    for percentile in percentiles:
        region_frames.append(
            _assign_threshold_regions(
                weights=weights,
                affine=img.affine,
                mask=threshold_masks[percentile],
                groups=groups,
                percentile=percentile,
                threshold_value=threshold_values[percentile],
                min_report_voxels=args.min_report_voxels,
            )
        )
    region_df = pd.concat(region_frames, ignore_index=True)
    reference_regions = _node_set(region_df, REFERENCE_THRESHOLD, args.min_report_voxels)
    region_df.attrs["reference_regions"] = reference_regions
    summary_df = pd.DataFrame(
        [
            _summarize_threshold(
                threshold_masks[percentile],
                threshold_masks,
                region_df[region_df["percentile"].eq(percentile)],
                percentile,
                threshold_values[percentile],
                args.min_report_voxels,
                reference_regions,
            )
            for percentile in percentiles
        ]
    )

    args.out_base.parent.mkdir(parents=True, exist_ok=True)
    summary_path = Path(f"{args.out_base}_summary.csv")
    regions_path = Path(f"{args.out_base}_regions.csv")
    summary_df.to_csv(summary_path, index=False)
    region_df.to_csv(regions_path, index=False)
    _plot_atlas_regions(groups, img, metadata, args.out_base, region_df, args.min_report_voxels)
    _plot_robustness(summary_df, region_df, args.out_base, args.min_report_voxels)
    _write_report(args.out_base, args.map, summary_df, region_df, metadata, args.min_report_voxels)

    print(summary_df.to_string(index=False))
    print(f"Saved {args.out_base}.png")
    print(f"Saved {args.out_base}.pdf")
    print(f"Saved {args.out_base}_atlas_regions.png")
    print(f"Saved {args.out_base}_atlas_regions.pdf")
    print(f"Saved {summary_path}")
    print(f"Saved {regions_path}")
    print(f"Saved {args.out_base}.md")
    print(f"Saved {args.out_base}.json")


if __name__ == "__main__":
    main()
