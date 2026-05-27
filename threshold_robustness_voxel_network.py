#!/usr/bin/env python3
"""Threshold-robustness analysis for the final voxel-weight network."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import datasets, image
from scipy import ndimage


DEFAULT_MAP = Path(
    "data/voxel_weights_mean_foldavg_sub9_ses1_task1_bold1_beta0.75_smooth1.8_gamma1.5.nii.gz"
)
DEFAULT_OUT_BASE = Path(
    "figures/voxel_weights_mean_foldavg_sub9_ses1_task1_bold1_beta0.75_smooth1.8_gamma1.5_threshold_robustness"
)
DEFAULT_THRESHOLDS = (80.0, 85.0, 90.0, 95.0, 97.5)
MAIN_THRESHOLDS = (85.0, 90.0, 95.0)
REFERENCE_THRESHOLD = 90.0
DEFAULT_MIN_REPORT_VOXELS = 25
DEFAULT_MIDLINE_BAND_MM = 1.0
FSL_CEREBELLUM_MAXPROB_2MM = Path(
    "/usr/local/fsl/data/atlases/Cerebellum/Cerebellum-MNIfnirt-maxprob-thr25-2mm.nii.gz"
)


@dataclass(frozen=True)
class ROIGroup:
    name: str
    source: str
    mask: np.ndarray
    matched_labels: tuple[str, ...]


def _pct_label(percentile: float) -> str:
    return f"p{percentile:g}".replace(".", "p")


def _to_label_list(labels) -> list[str]:
    out: list[str] = []
    for label in list(labels):
        out.append(label.decode("utf-8", errors="replace") if isinstance(label, bytes) else str(label))
    return out


def _fetch_ho_cort_sub(cache_dir: Path | None) -> tuple[nib.Nifti1Image, list[str], nib.Nifti1Image, list[str]]:
    data_dir = str(cache_dir) if cache_dir is not None else None
    cortical = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm", data_dir=data_dir)
    subcortical = datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr25-2mm", data_dir=data_dir)
    cortical_img = cortical.maps if isinstance(cortical.maps, nib.Nifti1Image) else nib.load(cortical.maps)
    subcortical_img = subcortical.maps if isinstance(subcortical.maps, nib.Nifti1Image) else nib.load(subcortical.maps)
    return cortical_img, _to_label_list(cortical.labels), subcortical_img, _to_label_list(subcortical.labels)


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


def _select_ids_by_patterns(
    labels: list[str],
    include: list[str],
    exclude: list[str] | None = None,
) -> tuple[list[int], tuple[str, ...]]:
    include_lower = [item.lower() for item in include]
    exclude_lower = [item.lower() for item in (exclude or [])]
    ids: list[int] = []
    names: list[str] = []
    for idx, label in enumerate(labels):
        if idx == 0:
            continue
        label_lower = label.lower()
        if not any(pattern in label_lower for pattern in include_lower):
            continue
        if any(pattern in label_lower for pattern in exclude_lower):
            continue
        ids.append(idx)
        names.append(label)
    return ids, tuple(names)


def _make_mask_from_labels(label_data: np.ndarray, label_ids: list[int], shape: tuple[int, int, int]) -> np.ndarray:
    if not label_ids:
        return np.zeros(shape, dtype=bool)
    return np.isin(label_data, label_ids)


def _build_roi_groups(reference_img: nib.Nifti1Image, cache_dir: Path | None) -> tuple[list[ROIGroup], dict[str, object]]:
    cort_img, cort_labels, sub_img, sub_labels = _fetch_ho_cort_sub(cache_dir)
    if not FSL_CEREBELLUM_MAXPROB_2MM.exists():
        raise FileNotFoundError(f"Cerebellum atlas not found: {FSL_CEREBELLUM_MAXPROB_2MM}")

    cort_data = _resample_label_img(cort_img, reference_img)
    sub_data = _resample_label_img(sub_img, reference_img)
    cereb_data = _resample_label_img(nib.load(str(FSL_CEREBELLUM_MAXPROB_2MM)), reference_img)
    shape = reference_img.shape[:3]
    groups: list[ROIGroup] = []

    def add_from_sub(name: str, patterns: list[str]) -> None:
        ids, names = _select_ids_by_patterns(sub_labels, include=patterns)
        groups.append(
            ROIGroup(
                name=name,
                source="Harvard-Oxford Subcortical (thr25, 2mm)",
                mask=_make_mask_from_labels(sub_data, ids, shape),
                matched_labels=names,
            )
        )

    def add_from_cort(name: str, patterns: list[str], exclude: list[str] | None = None) -> None:
        ids, names = _select_ids_by_patterns(cort_labels, include=patterns, exclude=exclude)
        groups.append(
            ROIGroup(
                name=name,
                source="Harvard-Oxford Cortical (thr25, 2mm)",
                mask=_make_mask_from_labels(cort_data, ids, shape),
                matched_labels=names,
            )
        )

    add_from_sub("Amygdala", ["amygdala"])
    groups.append(
        ROIGroup(
            name="Cerebellum",
            source="FSL Cerebellum MNIfnirt (maxprob thr25, 2mm)",
            mask=cereb_data > 0,
            matched_labels=("All cerebellar labels (value > 0)",),
        )
    )
    add_from_cort("Cingulate Cortex", ["cingulate gyrus", "paracingulate gyrus"])
    add_from_cort("Inferior Frontal Gyrus", ["inferior frontal gyrus"])
    add_from_cort(
        "Dorsolateral Prefrontal Cortex",
        ["middle frontal gyrus", "superior frontal gyrus", "frontal pole"],
    )
    add_from_cort(
        "vmPFC / dmPFC (Control & monitoring)",
        ["frontal medial cortex", "frontal orbital cortex", "subcallosal cortex", "paracingulate gyrus"],
    )
    add_from_cort(
        "Parietal Cortex",
        [
            "superior parietal lobule",
            "supramarginal gyrus",
            "angular gyrus",
            "precuneous cortex",
            "parietal opercular cortex",
            "postcentral gyrus",
        ],
    )
    add_from_cort("Precentral", ["precentral gyrus", "juxtapositional lobule cortex"])
    add_from_cort(
        "Temporal Cortex",
        [
            "temporal pole",
            "superior temporal gyrus",
            "middle temporal gyrus",
            "inferior temporal gyrus",
            "temporal fusiform cortex",
            "temporal occipital fusiform cortex",
            "planum polare",
            "planum temporale",
            "heschl",
            "parahippocampal gyrus",
        ],
    )
    add_from_sub("Thalamus", ["thalamus"])
    add_from_cort(
        "Occipital Cortex (relative)",
        ["lateral occipital cortex", "occipital pole", "cuneal", "lingual", "intracalcarine", "supracalcarine"],
    )
    add_from_cort(
        "Insular / Opercular Cortex (relative)",
        ["insular cortex", "opercular cortex", "frontal opercular cortex", "central opercular cortex"],
    )
    add_from_sub("Hippocampus (relative)", ["hippocampus"])
    add_from_sub("Basal Ganglia (relative)", ["caudate", "putamen", "pallidum", "accumbens"])
    add_from_sub("Other Cerebral Cortex (relative)", ["cerebral cortex"])

    metadata = {
        "roi_definition": "roi_edge_network_atlas_logic",
        "priority_order": [group.name for group in groups] + ["Unassigned Active Voxels (relative)"],
        "atlas_info": {
            "cortical_atlas": "Harvard-Oxford Cortical (thr25, 2mm)",
            "subcortical_atlas": "Harvard-Oxford Subcortical (thr25, 2mm)",
            "cerebellum_atlas": str(FSL_CEREBELLUM_MAXPROB_2MM),
        },
        "roi_sources": {group.name: group.source for group in groups},
    }
    return groups, metadata


def _split_node_positions(
    coords_x: np.ndarray,
    positions: np.ndarray,
    midline_band_mm: float,
) -> list[tuple[str, np.ndarray]]:
    left = positions[coords_x[positions] < -midline_band_mm]
    right = positions[coords_x[positions] > midline_band_mm]
    mid = positions[np.abs(coords_x[positions]) <= midline_band_mm]
    if mid.size > 0:
        if left.size >= right.size and left.size > 0:
            left = np.concatenate([left, mid])
        elif right.size > 0:
            right = np.concatenate([right, mid])
        else:
            return [("M", mid)]
    return [("L", left), ("R", right)]


def _assign_threshold_regions(
    weights: np.ndarray,
    affine: np.ndarray,
    mask: np.ndarray,
    groups: list[ROIGroup],
    percentile: float,
    threshold_value: float,
    min_report_voxels: int,
    midline_band_mm: float,
) -> pd.DataFrame:
    selected_ijk = np.column_stack(np.nonzero(mask)).astype(np.int32, copy=False)
    if selected_ijk.size == 0:
        return pd.DataFrame()

    x, y, z = selected_ijk.T
    assigned = np.zeros(selected_ijk.shape[0], dtype=np.int16)
    group_names = ["Unassigned Active Voxels (relative)"] + [group.name for group in groups]
    group_sources = {"Unassigned Active Voxels (relative)": "Data-driven fallback"}

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
        for hemisphere, member_positions in _split_node_positions(coords_mm[:, 0], positions, midline_band_mm):
            if member_positions.size == 0:
                continue
            node_name = f"{hemisphere} {roi_name}" if hemisphere in ("L", "R") else f"M {roi_name}"
            coords = coords_mm[member_positions]
            values = selected_weights[member_positions]
            rows.append(
                {
                    "percentile": float(percentile),
                    "threshold_label": _pct_label(percentile),
                    "threshold_value": float(threshold_value),
                    "roi_name": roi_name,
                    "hemisphere": hemisphere,
                    "node_name": node_name,
                    "n_voxels": int(member_positions.size),
                    "present_for_report": bool(member_positions.size >= min_report_voxels),
                    "mean_weight": float(np.mean(values)),
                    "max_weight": float(np.max(values)),
                    "x_mm": float(np.mean(coords[:, 0])),
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
    present_nodes = set(region_df.loc[region_df["n_voxels"] >= min_report_voxels, "node_name"])
    region_union = present_nodes | reference_regions

    return {
        "percentile": float(percentile),
        "threshold_label": _pct_label(percentile),
        "threshold_value": float(threshold_value),
        "n_voxels": int(np.count_nonzero(mask)),
        "n_components": int(n_components),
        "largest_component_voxels": largest_component,
        "n_reportable_nodes": int(len(present_nodes)),
        "jaccard_vs_p90": float(intersection / union) if union else np.nan,
        "p90_voxels_retained": float(intersection / np.count_nonzero(reference_mask)) if np.any(reference_mask) else np.nan,
        "threshold_voxels_in_p90": float(intersection / np.count_nonzero(mask)) if np.any(mask) else np.nan,
        "node_jaccard_vs_p90": float(len(present_nodes & reference_regions) / len(region_union)) if region_union else np.nan,
        "p90_nodes_retained": float(len(present_nodes & reference_regions) / len(reference_regions)) if reference_regions else np.nan,
    }


def _plot_robustness(summary_df: pd.DataFrame, region_df: pd.DataFrame, out_base: Path, min_report_voxels: int) -> None:
    report_regions = region_df.loc[region_df["n_voxels"] >= min_report_voxels].copy()
    p90_counts = (
        report_regions.loc[np.isclose(report_regions["percentile"], REFERENCE_THRESHOLD), ["node_name", "n_voxels"]]
        .set_index("node_name")["n_voxels"]
        .to_dict()
    )
    order_df = (
        report_regions.groupby("node_name", as_index=False)["n_voxels"]
        .max()
        .assign(p90_count=lambda df: df["node_name"].map(p90_counts).fillna(0))
        .sort_values(["p90_count", "n_voxels", "node_name"], ascending=[False, False, True])
    )
    node_order = order_df["node_name"].tolist()
    pivot = (
        report_regions.pivot_table(index="node_name", columns="threshold_label", values="n_voxels", aggfunc="sum", fill_value=0)
        .reindex(index=node_order, columns=[_pct_label(p) for p in summary_df["percentile"]])
    )

    fig = plt.figure(figsize=(10.6, max(7.4, 0.24 * max(1, len(node_order)) + 2.5)), facecolor="white")
    gs = fig.add_gridspec(2, 2, height_ratios=[0.9, max(2.2, 0.17 * max(1, len(node_order)))], hspace=0.22, wspace=0.28)
    ax_vox = fig.add_subplot(gs[0, 0])
    ax_stable = fig.add_subplot(gs[0, 1])
    ax_heat = fig.add_subplot(gs[1, :])

    ax_vox.plot(summary_df["percentile"], summary_df["n_voxels"], marker="o", color="#2563eb", linewidth=2.0)
    ax_vox.set_xlabel("Percentile threshold")
    ax_vox.set_ylabel("Voxels")
    ax_vox.grid(alpha=0.25)
    ax_vox.axvline(REFERENCE_THRESHOLD, color="#dc2626", linestyle="--", linewidth=1.3)

    ax_stable.plot(summary_df["percentile"], summary_df["p90_nodes_retained"], marker="o", color="#0f766e", label="p90 nodes retained")
    ax_stable.plot(summary_df["percentile"], summary_df["node_jaccard_vs_p90"], marker="s", color="#7c3aed", label="node Jaccard vs p90")
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
    ax_heat.set_yticklabels(pivot.index.tolist(), fontsize=8)
    ax_heat.set_xlabel("Threshold")
    for row_idx in range(pivot.shape[0]):
        for col_idx in range(pivot.shape[1]):
            value = int(pivot.iat[row_idx, col_idx])
            if value:
                ax_heat.text(col_idx, row_idx, str(value), ha="center", va="center", fontsize=6.5, color="white" if heat[row_idx, col_idx] > 2.1 else "black")
    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.018, pad=0.018)
    cbar.set_label("log10(voxels + 1)")
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{out_base}.png", dpi=220, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(f"{out_base}.pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def _node_set(region_df: pd.DataFrame, percentile: float, min_report_voxels: int) -> set[str]:
    rows = region_df[np.isclose(region_df["percentile"], percentile) & (region_df["n_voxels"] >= min_report_voxels)]
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
        region_df[np.isclose(region_df["percentile"], REFERENCE_THRESHOLD) & (region_df["n_voxels"] >= min_report_voxels)]
        .sort_values("n_voxels", ascending=False)
    )
    top_p90 = [f"{row.node_name} ({int(row.n_voxels)})" for row in p90_region_counts.itertuples(index=False)]
    ref = summary_df[np.isclose(summary_df["percentile"], REFERENCE_THRESHOLD)].iloc[0]
    p95 = summary_df[np.isclose(summary_df["percentile"], 95.0)].iloc[0]
    p85 = summary_df[np.isclose(summary_df["percentile"], 85.0)].iloc[0]

    verdict = (
        "The main network is robust at the region level across p85-p95: most p90 region nodes remain reportable "
        "after tightening to p95, while relaxing to p85 mainly expands already-present nodes."
        if float(p95["p90_nodes_retained"]) >= 0.75
        else "The main network has a stable core, but several p90 region nodes are threshold-sensitive when tightened to p95."
    )

    report = f"""# Threshold-Robustness Analysis

Input map: `{map_path}`

Reference visualization: p90-style thresholding of the final voxel-weight network.

## Method

- Thresholded nonzero voxel weights at percentiles: {", ".join(_pct_label(p) for p in summary_df["percentile"])}.
- Used p90 as the reference threshold and treated p85/p95 as the main relaxed/tightened sensitivity range.
- Assigned suprathreshold voxels to broad ROI families using Harvard-Oxford cortical/subcortical atlases plus the local FSL cerebellum atlas.
- Split ROI families into L/R nodes by MNI x coordinate.
- Counted a node as reportable when it contained at least {min_report_voxels} suprathreshold voxels.

## Result

{verdict}

At p90, the map contains {int(ref["n_voxels"]):,} suprathreshold voxels, {int(ref["n_components"]):,} connected components, and {int(ref["n_reportable_nodes"]):,} reportable L/R region nodes. Relaxing to p85 gives {int(p85["n_reportable_nodes"]):,} reportable nodes; tightening to p95 retains {float(p95["p90_nodes_retained"]):.1%} of the p90 nodes.

Stable p85-p95 nodes: {_format_node_list(stable_nodes, max_items=40)}

Added when relaxed to p85: {_format_node_list(relaxed_only)}

Dropped below report threshold when tightened to p95: {_format_node_list(lost_when_tightened)}

Largest p90 nodes by voxel count: {_format_node_list(top_p90, max_items=12)}

## Suggested Reporting

Report the p90 network as the primary visualization and include the robustness heatmap/table as a supplement. In text, emphasize region-level stability rather than raw voxel overlap, because percentile thresholds are nested by construction. A clear phrasing is:

\"Threshold sensitivity was assessed by repeating the network definition at p85, p90, and p95 of the nonzero voxel-weight distribution. The main p90 network was stable at the anatomical-region level: the same core L/R ROI nodes persisted across p85-p95, while threshold relaxation mainly expanded the network and threshold tightening removed smaller peripheral nodes.\"

Then list the stable core and the threshold-sensitive nodes from the bullets above.

## Outputs

- `{out_base}.png`
- `{out_base}.pdf`
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
        help="Absolute MNI x band assigned to the larger hemisphere node.",
    )
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
    groups, metadata = _build_roi_groups(img, args.atlas_cache_dir)
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
                midline_band_mm=args.midline_band_mm,
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
    _plot_robustness(summary_df, region_df, args.out_base, args.min_report_voxels)
    _write_report(args.out_base, args.map, summary_df, region_df, metadata, args.min_report_voxels)

    print(summary_df.to_string(index=False))
    print(f"Saved {args.out_base}.png")
    print(f"Saved {args.out_base}.pdf")
    print(f"Saved {summary_path}")
    print(f"Saved {regions_path}")
    print(f"Saved {args.out_base}.md")
    print(f"Saved {args.out_base}.json")


if __name__ == "__main__":
    main()
