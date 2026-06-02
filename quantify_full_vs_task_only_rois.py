#!/usr/bin/env python3
"""ROI quantification for the full-vs-task-only anatomy ablation figure."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import datasets, image
from scipy import ndimage

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyze_ablation_constraints import (
    DEFAULT_ATLAS_CACHE_DIR,
    DEFAULT_FULL_MODEL_HTML,
    DEFAULT_MAIN_MAP,
    DEFAULT_TASK_ONLY_MAP,
    DEFAULT_TASK_ONLY_Z_THRESHOLD,
    _brainstem_mask,
    _html_sprite_volumes,
    _load_data,
)
from threshold_robustness_voxel_network import (
    DEFAULT_AAL_VERSION,
    UNASSIGNED_ROI,
    _build_roi_groups,
)


DEFAULT_OUT_BASE = Path("figures/ablation/ablation_full_vs_task_only_roi_quantification")
HARVARD_OXFORD_CORTICAL = "cort-maxprob-thr25-2mm"
HARVARD_OXFORD_SUBCORTICAL = "sub-maxprob-thr25-2mm"
WHITE_MATTER_LABEL_FRAGMENT = "Cerebral White Matter"
COMPARISON_COLORS = {
    "vigour_only": "#0072B2",
    "task_only": "#D55E00",
    "both": "#00A676",
}


@dataclass
class RegionMask:
    name: str
    mask: np.ndarray
    sources: set[str] = field(default_factory=set)
    matched_labels: list[str] = field(default_factory=list)


def _resample_labels(label_img: nib.Nifti1Image, reference_img: nib.Nifti1Image) -> np.ndarray:
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


def _atlas_img(atlas: object) -> nib.Nifti1Image:
    maps = atlas.maps
    return maps if isinstance(maps, nib.Nifti1Image) else nib.load(maps)


def _white_matter_mask(reference_img: nib.Nifti1Image, cache_dir: Path) -> np.ndarray:
    atlas = datasets.fetch_atlas_harvard_oxford(HARVARD_OXFORD_SUBCORTICAL, data_dir=str(cache_dir), verbose=0)
    atlas_data = _resample_labels(_atlas_img(atlas), reference_img)
    label_values = [
        label_value
        for label_value, label_name in enumerate(atlas.labels)
        if WHITE_MATTER_LABEL_FRAGMENT in str(label_name)
    ]
    return np.isin(atlas_data, label_values)


def _exclude_white_matter(
    masks: dict[str, np.ndarray],
    white_matter: np.ndarray,
    metadata: dict[str, object],
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    full = masks["vigour"] & ~white_matter
    task = masks["task"] & ~white_matter
    updated = dict(masks)
    updated["white_matter"] = white_matter
    updated["vigour"] = full
    updated["task"] = task
    updated["vigour_only"] = full & ~task
    updated["task_only"] = task & ~full
    updated["both"] = full & task
    updated["union"] = full | task

    updated_metadata = dict(metadata)
    mask_definition = str(updated_metadata.get("mask_definition", "")).strip()
    updated_metadata["mask_definition"] = (
        f"{mask_definition} " if mask_definition else ""
    ) + "Harvard-Oxford cerebral white-matter voxels are excluded from the ROI quantification."
    updated_metadata["white_matter_mask_source"] = "Harvard-Oxford subcortical: Left/Right Cerebral White Matter"
    updated_metadata["white_matter_suppressed_vigour_voxels"] = int(np.count_nonzero(masks["vigour"] & white_matter))
    updated_metadata["white_matter_suppressed_task_activation_voxels"] = int(
        np.count_nonzero(masks["task"] & white_matter)
    )
    updated_metadata["white_matter_suppressed_overlap_voxels"] = int(np.count_nonzero(masks["both"] & white_matter))
    return updated, updated_metadata


def _strip_laterality(label: str) -> str:
    return re.sub(r"^(Left|Right)\s+", "", label).strip()


def _safe_unknown_ho_name(label: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", _strip_laterality(label)).strip("_")
    return f"HO_{cleaned}" if cleaned else "HO_Unmapped"


def _ho_group_name(label: str, family: str) -> str:
    base = _strip_laterality(label)
    low = base.lower()

    if family == "subcortical":
        exact = {
            "Cerebral White Matter": "Cerebral_White_Matter",
            "Cerebral Cortex": "Cerebral_Cortex",
            "Lateral Ventricle": "Lateral_Ventricle",
            "Thalamus": "Thalamus",
            "Caudate": "Caudate",
            "Putamen": "Putamen",
            "Pallidum": "Pallidum",
            "Hippocampus": "Hippocampus",
            "Amygdala": "Amygdala",
            "Accumbens": "N_Acc",
            "Brain-Stem": "Brainstem_HO",
        }
        return exact.get(base, _safe_unknown_ho_name(base))

    if "precentral" in low:
        return "Precentral"
    if "postcentral" in low:
        return "Postcentral"
    if "juxtapositional" in low or "supplementary motor" in low:
        return "Supp_Motor_Area"
    if "insular" in low:
        return "Insula"
    if "cingulate" in low or "paracingulate" in low:
        return "Cingulate"
    if "parahippocampal" in low:
        return "ParaHippocampal"
    if "fusiform" in low:
        return "Fusiform"
    if any(token in low for token in ("occipital", "intracalcarine", "cuneal", "lingual", "supracalcarine")):
        return "Occipital"
    if any(token in low for token in ("parietal", "precuneous", "angular", "supramarginal")):
        return "Parietal"
    if "frontal orbital" in low or "orbitofrontal" in low:
        return "Orbitofrontal"
    if "frontal" in low or "subcallosal" in low or "central opercular" in low:
        return "Frontal"
    if "temporal" in low or "heschl" in low or "planum temporale" in low or "planum polare" in low:
        return "Temporal"
    return _safe_unknown_ho_name(base)


def _add_region(
    regions: dict[str, RegionMask],
    name: str,
    mask: np.ndarray,
    source: str,
    matched_label: str,
) -> None:
    if not np.any(mask):
        return
    if name not in regions:
        regions[name] = RegionMask(name=name, mask=np.zeros(mask.shape, dtype=bool))
    regions[name].mask |= mask
    regions[name].sources.add(source)
    if matched_label not in regions[name].matched_labels:
        regions[name].matched_labels.append(matched_label)


def _build_aal3_regions(
    reference_img: nib.Nifti1Image,
    cache_dir: Path,
    analysis_space_mask: np.ndarray,
) -> tuple[dict[str, RegionMask], np.ndarray, dict[str, object]]:
    groups, metadata = _build_roi_groups(reference_img, DEFAULT_AAL_VERSION, cache_dir)
    regions: dict[str, RegionMask] = {}
    assigned = np.zeros(reference_img.shape[:3], dtype=bool)
    for group in groups:
        mask = group.mask & analysis_space_mask
        _add_region(regions, group.name, mask, group.source, ";".join(group.matched_labels))
        assigned |= mask
    return regions, assigned, metadata


def _add_harvard_oxford_fill(
    regions: dict[str, RegionMask],
    assigned: np.ndarray,
    reference_img: nib.Nifti1Image,
    cache_dir: Path,
    analysis_space_mask: np.ndarray,
) -> np.ndarray:
    fill_specs = (
        (HARVARD_OXFORD_CORTICAL, "cortical", "Harvard-Oxford cortical maxprob-thr25 fill after AAL3"),
        (HARVARD_OXFORD_SUBCORTICAL, "subcortical", "Harvard-Oxford subcortical maxprob-thr25 fill after AAL3"),
    )
    for atlas_name, family, source in fill_specs:
        atlas = datasets.fetch_atlas_harvard_oxford(atlas_name, data_dir=str(cache_dir), verbose=0)
        atlas_data = _resample_labels(_atlas_img(atlas), reference_img)
        for label_value, label_name in enumerate(atlas.labels):
            if label_value == 0 or str(label_name).lower() == "background":
                continue
            fill_mask = (atlas_data == label_value) & analysis_space_mask & ~assigned
            if not np.any(fill_mask):
                continue
            group_name = _ho_group_name(str(label_name), family)
            _add_region(regions, group_name, fill_mask, source, str(label_name))
            assigned |= fill_mask
    return assigned


def _add_nearest_label_fill(
    regions: dict[str, RegionMask],
    assigned: np.ndarray,
    target_mask: np.ndarray,
) -> tuple[np.ndarray, int]:
    missing = target_mask & ~assigned
    n_missing = int(np.count_nonzero(missing))
    if n_missing == 0:
        return assigned, 0
    if not np.any(assigned):
        raise RuntimeError("Cannot fill atlas-unassigned voxels because no atlas labels are assigned.")

    region_names = list(regions)
    label_volume = np.zeros(assigned.shape, dtype=np.int16)
    for label_id, roi_name in enumerate(region_names, start=1):
        label_volume[regions[roi_name].mask & assigned] = label_id

    nearest_indices = ndimage.distance_transform_edt(
        ~assigned,
        return_distances=False,
        return_indices=True,
    )
    nearest_labels = label_volume[tuple(axis_indices[missing] for axis_indices in nearest_indices)]
    for label_id in np.unique(nearest_labels):
        if label_id <= 0:
            continue
        roi_name = region_names[int(label_id) - 1]
        fill_mask = np.zeros(assigned.shape, dtype=bool)
        fill_mask[missing] = nearest_labels == label_id
        regions[roi_name].mask |= fill_mask
        regions[roi_name].sources.add("Nearest-label fill for atlas-uncovered selected voxels")
        if "nearest atlas label for uncovered selected voxels" not in regions[roi_name].matched_labels:
            regions[roi_name].matched_labels.append("nearest atlas label for uncovered selected voxels")
        assigned |= fill_mask
    return assigned, n_missing


def _build_regions(
    reference_img: nib.Nifti1Image,
    cache_dir: Path,
    analysis_space_mask: np.ndarray,
    atlas_mode: str,
    target_fill_mask: np.ndarray | None = None,
) -> tuple[dict[str, RegionMask], np.ndarray, dict[str, object]]:
    regions, assigned, metadata = _build_aal3_regions(reference_img, cache_dir, analysis_space_mask)
    if atlas_mode in {"aal3_ho_fill", "aal3_ho_nearest"}:
        assigned = _add_harvard_oxford_fill(regions, assigned, reference_img, cache_dir, analysis_space_mask)
        metadata = dict(metadata)
        metadata["roi_definition"] = "aal3_bilateral_coarse_groups_with_harvard_oxford_fill"
        metadata["fill_rule"] = (
            "AAL3 coarse bilateral regions are assigned first; only AAL3-unassigned voxels are "
            "filled from Harvard-Oxford cortical and subcortical maxprob-thr25 labels."
        )
    if atlas_mode == "aal3_ho_nearest":
        if target_fill_mask is None:
            raise RuntimeError("target_fill_mask is required for nearest-label atlas fill.")
        assigned, n_nearest_filled = _add_nearest_label_fill(regions, assigned, target_fill_mask & analysis_space_mask)
        metadata = dict(metadata)
        metadata["roi_definition"] = "aal3_bilateral_coarse_groups_with_harvard_oxford_and_nearest_selected_voxel_fill"
        metadata["fill_rule"] = (
            "AAL3 coarse bilateral regions are assigned first; AAL3-unassigned voxels are filled from "
            "Harvard-Oxford cortical and subcortical maxprob-thr25 labels; remaining selected voxels "
            "are assigned to the nearest existing anatomical label."
        )
        metadata["nearest_label_filled_selected_voxels"] = int(n_nearest_filled)
    return regions, assigned, metadata


def _load_figure_masks(
    reference_map: Path,
    full_html: Path,
    task_map: Path,
    task_z_threshold: float,
) -> tuple[nib.Nifti1Image, dict[str, np.ndarray], dict[str, object]]:
    reference_img, _ = _load_data(reference_map)
    _, full_mask, html_affine = _html_sprite_volumes(full_html)
    task_img, task_data = _load_data(task_map)
    if full_mask.shape != reference_img.shape[:3]:
        raise RuntimeError(f"{full_html} overlay shape {full_mask.shape} differs from {reference_map}.")
    if task_img.shape[:3] != full_mask.shape:
        raise RuntimeError(f"{task_map} shape {task_img.shape[:3]} differs from {full_html}.")

    task_mask = np.isfinite(task_data) & (task_data >= task_z_threshold)
    for axis in range(3):
        task_step = float(task_img.affine[axis, axis])
        html_step = float(html_affine[axis, axis])
        if task_step != 0.0 and html_step != 0.0 and np.sign(task_step) != np.sign(html_step):
            task_mask = np.flip(task_mask, axis=axis)

    display_img = nib.Nifti1Image(np.zeros(full_mask.shape, dtype=np.uint8), html_affine)
    brainstem = _brainstem_mask(display_img)
    raw_full = full_mask.copy()
    raw_task = task_mask.copy()
    full = raw_full & ~brainstem
    task = raw_task & ~brainstem
    masks = {
        "raw_vigour": raw_full,
        "raw_task": raw_task,
        "brainstem": brainstem,
        "vigour": full,
        "task": task,
        "vigour_only": full & ~task,
        "task_only": task & ~full,
        "both": full & task,
        "union": full | task,
    }
    metadata = {
        "reference_map": str(reference_map),
        "full_model_html": str(full_html),
        "task_activation_map": str(task_map),
        "task_activation_threshold": f"z >= {task_z_threshold:g}",
        "mask_definition": (
            "Vigour network uses the selected overlay embedded in the thresholded full-model HTML; "
            "task activation uses the standard-GLM z map thresholded at z >= threshold; both masks "
            "then exclude the brainstem mask used by the anatomy figure."
        ),
        "raw_vigour_voxels": int(np.count_nonzero(raw_full)),
        "raw_task_activation_voxels": int(np.count_nonzero(raw_task)),
        "brainstem_suppressed_vigour_voxels": int(np.count_nonzero(raw_full & brainstem)),
        "brainstem_suppressed_task_activation_voxels": int(np.count_nonzero(raw_task & brainstem)),
    }
    return display_img, masks, metadata


def _safe_pct(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else np.nan


def _center_mm(mask: np.ndarray, affine: np.ndarray) -> tuple[float, float, float]:
    coords = np.column_stack(np.nonzero(mask))
    if coords.size == 0:
        return (np.nan, np.nan, np.nan)
    xyz = nib.affines.apply_affine(affine, coords).mean(axis=0)
    return (float(xyz[0]), float(xyz[1]), float(xyz[2]))


def _roi_quantification(
    regions: dict[str, RegionMask],
    assigned_mask: np.ndarray,
    masks: dict[str, np.ndarray],
    reference_img: nib.Nifti1Image,
) -> pd.DataFrame:
    totals = {name: int(np.count_nonzero(mask)) for name, mask in masks.items()}
    rows: list[dict[str, object]] = []
    all_regions = list(regions.values())
    unassigned_selected = masks["union"] & ~assigned_mask
    if np.any(unassigned_selected):
        all_regions.append(
            RegionMask(
                name=UNASSIGNED_ROI,
                mask=~assigned_mask,
                sources={"Outside selected atlas labels"},
                matched_labels=[],
            )
        )

    for region in all_regions:
        region_union = masks["union"] & region.mask
        if not np.any(region_union):
            continue
        vigour_only = int(np.count_nonzero(masks["vigour_only"] & region.mask))
        task_only = int(np.count_nonzero(masks["task_only"] & region.mask))
        both = int(np.count_nonzero(masks["both"] & region.mask))
        vigour_total = vigour_only + both
        task_total = task_only + both
        union = vigour_only + task_only + both
        priority_roi_voxels = int(np.count_nonzero(region.mask)) if region.name != UNASSIGNED_ROI else np.nan
        if vigour_total > 0 and task_total > 0:
            membership = "both_maps"
        elif vigour_total > 0:
            membership = "vigour_region_not_task_region"
        elif task_total > 0:
            membership = "task_activation_region_not_vigour_region"
        else:
            membership = "neither"
        x_mm, y_mm, z_mm = _center_mm(region_union, reference_img.affine)
        rows.append(
            {
                "roi_name": region.name,
                "roi_source": "; ".join(sorted(region.sources)),
                "matched_labels": "; ".join(region.matched_labels),
                "priority_roi_voxels": priority_roi_voxels,
                "roi_membership": membership,
                "has_same_voxel_overlap": bool(both > 0),
                "vigour_network_voxels": vigour_total,
                "vigour_network_pct": _safe_pct(vigour_total, totals["vigour"]),
                "task_activation_voxels": task_total,
                "task_activation_pct": _safe_pct(task_total, totals["task"]),
                "union_voxels": union,
                "union_pct": _safe_pct(union, totals["union"]),
                "vigour_only_voxels": vigour_only,
                "vigour_only_pct_of_vigour_only": _safe_pct(vigour_only, totals["vigour_only"]),
                "task_only_voxels": task_only,
                "task_only_pct_of_task_only": _safe_pct(task_only, totals["task_only"]),
                "both_voxels": both,
                "both_pct_of_overlap": _safe_pct(both, totals["both"]),
                "vigour_only_pct_of_roi": _safe_pct(vigour_only, priority_roi_voxels)
                if region.name != UNASSIGNED_ROI
                else np.nan,
                "task_only_pct_of_roi": _safe_pct(task_only, priority_roi_voxels)
                if region.name != UNASSIGNED_ROI
                else np.nan,
                "both_pct_of_roi": _safe_pct(both, priority_roi_voxels)
                if region.name != UNASSIGNED_ROI
                else np.nan,
                "vigour_network_pct_of_roi": _safe_pct(vigour_total, priority_roi_voxels)
                if region.name != UNASSIGNED_ROI
                else np.nan,
                "task_activation_pct_of_roi": _safe_pct(task_total, priority_roi_voxels)
                if region.name != UNASSIGNED_ROI
                else np.nan,
                "union_pct_of_roi": _safe_pct(union, priority_roi_voxels) if region.name != UNASSIGNED_ROI else np.nan,
                "overlap_pct_of_vigour_in_roi": _safe_pct(both, vigour_total),
                "overlap_pct_of_task_in_roi": _safe_pct(both, task_total),
                "x_mm": x_mm,
                "y_mm": y_mm,
                "z_mm": z_mm,
            }
        )
    return pd.DataFrame(rows).sort_values(["union_voxels", "roi_name"], ascending=[False, True])


def _atlas_assigned_masks(reference_img: nib.Nifti1Image, cache_dir: Path, analysis_space_mask: np.ndarray) -> dict[str, np.ndarray]:
    groups, _ = _build_roi_groups(reference_img, DEFAULT_AAL_VERSION, cache_dir)
    aal3 = np.zeros(reference_img.shape[:3], dtype=bool)
    for group in groups:
        aal3 |= group.mask & analysis_space_mask

    cort = datasets.fetch_atlas_harvard_oxford(HARVARD_OXFORD_CORTICAL, data_dir=str(cache_dir), verbose=0)
    sub = datasets.fetch_atlas_harvard_oxford(HARVARD_OXFORD_SUBCORTICAL, data_dir=str(cache_dir), verbose=0)
    ho = ((_resample_labels(_atlas_img(cort), reference_img) > 0) | (_resample_labels(_atlas_img(sub), reference_img) > 0)) & analysis_space_mask

    schaefer = datasets.fetch_atlas_schaefer_2018(
        n_rois=100,
        yeo_networks=7,
        resolution_mm=2,
        data_dir=str(cache_dir),
        verbose=0,
    )
    schaefer_mask = (_resample_labels(_atlas_img(schaefer), reference_img) > 0) & analysis_space_mask
    return {
        "AAL3v2 coarse": aal3,
        "Harvard-Oxford cortical+subcortical": ho,
        "Schaefer2018 100 parcels": schaefer_mask,
    }


def _coverage_table(
    reference_img: nib.Nifti1Image,
    cache_dir: Path,
    analysis_space_mask: np.ndarray,
    selected_assigned_mask: np.ndarray,
    atlas_mode: str,
    masks: dict[str, np.ndarray],
) -> pd.DataFrame:
    assigned_masks = _atlas_assigned_masks(reference_img, cache_dir, analysis_space_mask)
    selected_name = _selected_atlas_name(atlas_mode)
    assigned_masks[selected_name] = selected_assigned_mask
    rows = []
    for atlas_name, assigned in assigned_masks.items():
        row: dict[str, object] = {
            "atlas_name": atlas_name,
            "atlas_voxels_in_analysis_space": int(np.count_nonzero(assigned)),
        }
        for mask_name in ("vigour", "task", "vigour_only", "task_only", "both", "union"):
            total = int(np.count_nonzero(masks[mask_name]))
            n_assigned = int(np.count_nonzero(masks[mask_name] & assigned))
            row[f"{mask_name}_assigned_voxels"] = n_assigned
            row[f"{mask_name}_total_voxels"] = total
            row[f"{mask_name}_assigned_pct"] = _safe_pct(n_assigned, total)
            row[f"{mask_name}_unassigned_voxels"] = total - n_assigned
        rows.append(row)
    return pd.DataFrame(rows).sort_values("union_assigned_pct", ascending=False)


def _set_summary(roi_df: pd.DataFrame) -> pd.DataFrame:
    specs = (
        ("regions_with_vigour_only_voxels", roi_df["vigour_only_voxels"].gt(0)),
        ("regions_with_task_only_voxels", roi_df["task_only_voxels"].gt(0)),
        ("regions_with_same_voxel_overlap", roi_df["both_voxels"].gt(0)),
        (
            "regions_with_both_maps_but_no_same_voxel_overlap",
            roi_df["roi_membership"].eq("both_maps") & roi_df["both_voxels"].eq(0),
        ),
        ("regions_present_only_in_vigour_map", roi_df["roi_membership"].eq("vigour_region_not_task_region")),
        (
            "regions_present_only_in_task_activation_map",
            roi_df["roi_membership"].eq("task_activation_region_not_vigour_region"),
        ),
    )
    rows = []
    for set_name, selector in specs:
        sub = roi_df.loc[selector].copy()
        rows.append(
            {
                "set_name": set_name,
                "n_regions": int(sub.shape[0]),
                "regions": "; ".join(sub.sort_values("union_voxels", ascending=False)["roi_name"].tolist()),
                "vigour_network_voxels": int(sub["vigour_network_voxels"].sum()),
                "task_activation_voxels": int(sub["task_activation_voxels"].sum()),
                "vigour_only_voxels": int(sub["vigour_only_voxels"].sum()),
                "task_only_voxels": int(sub["task_only_voxels"].sum()),
                "both_voxels": int(sub["both_voxels"].sum()),
                "union_voxels": int(sub["union_voxels"].sum()),
            }
        )
    return pd.DataFrame(rows)


def _fmt_pct(value: float) -> str:
    return "NA" if pd.isna(value) else f"{100.0 * float(value):.1f}%"


def _selected_atlas_name(atlas_mode: str) -> str:
    if atlas_mode == "aal3_ho_nearest":
        return "AAL3v2 + Harvard-Oxford + residual nearest-label fill"
    if atlas_mode == "aal3_ho_fill":
        return "AAL3v2 + Harvard-Oxford fill"
    return "AAL3v2 coarse"


def _roi_status(row: pd.Series) -> str:
    if int(row["both_voxels"]) > 0:
        return "Shared voxels present"
    if int(row["vigour_network_voxels"]) > 0 and int(row["task_activation_voxels"]) > 0:
        return "Both maps in ROI, no shared voxels"
    if int(row["vigour_network_voxels"]) > 0:
        return "Only vigour voxels in ROI"
    if int(row["task_activation_voxels"]) > 0:
        return "Only task voxels in ROI"
    return "No selected voxels"


def _clean_roi_table(roi_df: pd.DataFrame) -> pd.DataFrame:
    clean = roi_df.copy()
    clean["same_voxel_overlap_voxels"] = clean["both_voxels"].astype(int)
    clean["same_voxel_overlap_pct_of_roi"] = clean["both_pct_of_roi"]
    clean["roi_status"] = clean.apply(_roi_status, axis=1)
    clean = clean.rename(
        columns={
            "vigour_network_voxels": "vigour_total_voxels",
            "task_activation_voxels": "task_total_voxels",
            "priority_roi_voxels": "roi_total_voxels",
            "union_pct_of_roi": "selected_union_pct_of_roi",
        }
    )
    cols = [
        "roi_name",
        "roi_total_voxels",
        "vigour_only_voxels",
        "vigour_only_pct_of_roi",
        "task_only_voxels",
        "task_only_pct_of_roi",
        "same_voxel_overlap_voxels",
        "same_voxel_overlap_pct_of_roi",
        "vigour_total_voxels",
        "vigour_network_pct_of_roi",
        "task_total_voxels",
        "task_activation_pct_of_roi",
        "union_voxels",
        "selected_union_pct_of_roi",
        "roi_status",
    ]
    return clean[cols].sort_values(["union_voxels", "roi_name"], ascending=[False, True])


def _format_count(value: int | float) -> str:
    if pd.isna(value):
        return ""
    return f"{int(value):,}"


def _format_pct_roi(value: int | float) -> str:
    if pd.isna(value):
        return ""
    return f"{100.0 * float(value):.2f}%"


def _format_count_pct(count: int | float, pct: int | float) -> str:
    if pd.isna(count) or pd.isna(pct):
        return ""
    return f"{_format_count(count)} ({_format_pct_roi(pct)})"


def _plot_summary_image(out_path: Path, clean_df: pd.DataFrame, totals: dict[str, int], atlas_mode: str) -> None:
    plot_df = clean_df[clean_df["union_voxels"].gt(0)].sort_values(
        ["vigour_only_pct_of_roi", "roi_name"],
        ascending=[True, True],
    )
    fig_height = max(8.0, 0.34 * len(plot_df) + 2.4)
    fig, ax = plt.subplots(figsize=(11.6, fig_height), facecolor="white")
    y = np.arange(len(plot_df))
    vigour_only = 100.0 * plot_df["vigour_only_pct_of_roi"].to_numpy()
    both = 100.0 * plot_df["same_voxel_overlap_pct_of_roi"].to_numpy()
    task_only = 100.0 * plot_df["task_only_pct_of_roi"].to_numpy()

    vigour_bars = ax.barh(y, vigour_only, color=COMPARISON_COLORS["vigour_only"], label="vigour network")
    overlap_bars = ax.barh(y, both, left=vigour_only, color=COMPARISON_COLORS["both"], label="overlap")
    task_bars = ax.barh(
        y,
        task_only,
        left=vigour_only + both,
        color=COMPARISON_COLORS["task_only"],
        label="task-activation map",
    )

    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["roi_name"], fontsize=8.5)
    ax.set_xlabel("Percent of total voxels in ROI")
    ax.grid(axis="x", color="#D0D0D0", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(
        handles=[vigour_bars, task_bars, overlap_bars],
        labels=["vigour network", "task-activation map", "overlap"],
        loc="lower right",
        frameon=False,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def _plot_clean_table_image(out_path: Path, clean_df: pd.DataFrame, totals: dict[str, int], atlas_mode: str) -> None:
    display_df = clean_df[clean_df["union_voxels"].gt(0)].copy()
    display_df = display_df.sort_values(["selected_union_pct_of_roi", "roi_name"], ascending=[False, True])
    display_df = display_df[
        [
            "roi_name",
            "vigour_only_voxels",
            "vigour_only_pct_of_roi",
            "same_voxel_overlap_voxels",
            "same_voxel_overlap_pct_of_roi",
            "task_only_voxels",
            "task_only_pct_of_roi",
            "roi_status",
        ]
    ]

    table_rows = []
    for row in display_df.itertuples(index=False):
        table_rows.append(
            [
                row.roi_name,
                _format_count_pct(row.vigour_only_voxels, row.vigour_only_pct_of_roi),
                _format_count_pct(row.same_voxel_overlap_voxels, row.same_voxel_overlap_pct_of_roi),
                _format_count_pct(row.task_only_voxels, row.task_only_pct_of_roi),
                row.roi_status,
            ]
        )

    row_height = 0.26
    fig_height = max(6.3, row_height * (len(table_rows) + 2.0))
    fig, ax = plt.subplots(figsize=(11.2, fig_height), facecolor="white")
    ax.axis("off")
    table = ax.table(
        cellText=table_rows,
        colLabels=[
            "ROI",
            "Vigour only n (% ROI)",
            "Shared voxel n (% ROI)",
            "Task only n (% ROI)",
            "Voxel-level relation",
        ],
        colLoc="center",
        cellLoc="center",
        loc="upper left",
        colWidths=[0.16, 0.18, 0.20, 0.18, 0.28],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.9)
    table.scale(1.0, 1.25)

    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("#D8D8D8")
        cell.set_linewidth(0.5)
        cell.set_text_props(ha="center", va="center")
        if row_idx == 0:
            cell.set_facecolor("#F0F0F0")
            cell.set_text_props(weight="bold", color="#222222", ha="center", va="center")
        else:
            cell.set_facecolor("#FFFFFF" if row_idx % 2 else "#FAFAFA")
        if row_idx > 0 and col_idx == 1:
            cell.get_text().set_color(COMPARISON_COLORS["vigour_only"])
        if row_idx > 0 and col_idx == 2:
            cell.get_text().set_color(COMPARISON_COLORS["both"])
        if row_idx > 0 and col_idx == 3:
            cell.get_text().set_color(COMPARISON_COLORS["task_only"])

    fig.tight_layout()
    fig.savefig(out_path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def _top_region_lines(roi_df: pd.DataFrame, value_col: str, pct_col: str, n: int = 8) -> list[str]:
    sub = roi_df[roi_df[value_col].gt(0)].sort_values(value_col, ascending=False).head(n)
    return [f"- {row.roi_name}: {int(row[value_col]):,} ({_fmt_pct(row[pct_col])})" for _, row in sub.iterrows()]


def _top_roi_pct_lines(roi_df: pd.DataFrame, pct_col: str, count_col: str, n: int = 8) -> list[str]:
    sub = roi_df[roi_df[pct_col].gt(0)].sort_values(pct_col, ascending=False).head(n)
    return [f"- {row.roi_name}: {_fmt_pct(row[pct_col])} of ROI" for _, row in sub.iterrows()]


def _write_report(
    out_base: Path,
    roi_df: pd.DataFrame,
    set_df: pd.DataFrame,
    coverage_df: pd.DataFrame,
    metadata: dict[str, object],
    totals: dict[str, int],
    atlas_mode: str,
) -> None:
    selected_atlas = _selected_atlas_name(atlas_mode)
    coverage_lines = []
    for _, row in coverage_df.iterrows():
        coverage_lines.append(
            f"- {row.atlas_name}: union {int(row.union_assigned_voxels):,}/{int(row.union_total_voxels):,} "
            f"({_fmt_pct(row.union_assigned_pct)}), vigour {_fmt_pct(row.vigour_assigned_pct)}, "
            f"task {_fmt_pct(row.task_assigned_pct)}"
        )
    set_lines = []
    for _, row in set_df.iterrows():
        regions = row.regions if row.regions else "None"
        set_lines.append(f"- {row.set_name}: {int(row.n_regions)} regions - {regions}")

    lines = [
        "# Full vs Task-Only ROI Quantification",
        "",
        "## Inputs",
        f"- Vigour network: `{metadata['full_model_html']}` selected overlay, with brainstem voxels suppressed.",
        f"- Task activation map: `{metadata['task_activation_map']}` thresholded at {metadata['task_activation_threshold']}, with brainstem voxels suppressed.",
        f"- White matter excluded: {metadata.get('white_matter_mask_source', 'No')}.",
        f"- Atlas mode: {selected_atlas}.",
        "",
        "## Totals",
        f"- Vigour network: {totals['vigour']:,} non-white-matter voxels.",
        f"- Task activation map: {totals['task']:,} non-white-matter voxels.",
        f"- Same-voxel overlap: {totals['both']:,} voxels.",
        f"- Vigour-only: {totals['vigour_only']:,} voxels.",
        f"- Task-only: {totals['task_only']:,} voxels.",
        f"- Union: {totals['union']:,} voxels.",
        "",
        "## Atlas Coverage",
        *coverage_lines,
        "",
        "## Region Sets",
        "These sets are voxel-level summaries. A single ROI can contain both vigour-only and task-only voxels.",
        *set_lines,
        "",
        "## Highest Vigour-Only Percent of ROI",
        *_top_roi_pct_lines(roi_df, "vigour_only_pct_of_roi", "vigour_only_voxels"),
        "",
        "## Highest Task-Only Percent of ROI",
        *_top_roi_pct_lines(roi_df, "task_only_pct_of_roi", "task_only_voxels"),
        "",
        "## Highest Shared-Voxel Percent of ROI",
        *_top_roi_pct_lines(roi_df, "both_pct_of_roi", "both_voxels"),
        "",
        "## Outputs",
        f"- `{out_base}_by_roi.csv`",
        f"- `{out_base}_region_sets.csv`",
        f"- `{out_base}_clean_table.csv`",
        f"- `{out_base}_summary.png`",
        f"- `{out_base}_clean_table.png`",
        f"- `{out_base}_atlas_coverage.csv`",
        f"- `{out_base}_metadata.json`",
    ]
    out_base.with_name(out_base.name + "_report.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Path]:
    reference_img, masks, mask_metadata = _load_figure_masks(
        args.reference_map,
        args.full_html,
        args.task_map,
        args.task_z_threshold,
    )
    if args.exclude_white_matter:
        masks, mask_metadata = _exclude_white_matter(
            masks,
            _white_matter_mask(reference_img, args.atlas_cache_dir) & ~masks["brainstem"],
            mask_metadata,
        )
    analysis_space_mask = ~masks["brainstem"] & ~masks.get("white_matter", np.zeros(reference_img.shape[:3], dtype=bool))
    regions, assigned_mask, atlas_metadata = _build_regions(
        reference_img,
        args.atlas_cache_dir,
        analysis_space_mask,
        args.atlas_mode,
        masks["union"],
    )
    roi_df = _roi_quantification(regions, assigned_mask, masks, reference_img)
    if not roi_df["roi_name"].eq(UNASSIGNED_ROI).any() and "priority_order" in atlas_metadata:
        atlas_metadata = dict(atlas_metadata)
        atlas_metadata["priority_order"] = [
            roi_name for roi_name in atlas_metadata["priority_order"] if roi_name != UNASSIGNED_ROI
        ]
    set_df = _set_summary(roi_df)
    clean_df = _clean_roi_table(roi_df)
    coverage_df = _coverage_table(
        reference_img,
        args.atlas_cache_dir,
        analysis_space_mask,
        assigned_mask,
        args.atlas_mode,
        masks,
    )
    totals = {name: int(np.count_nonzero(mask)) for name, mask in masks.items()}

    args.out_base.parent.mkdir(parents=True, exist_ok=True)
    by_roi_path = args.out_base.with_name(args.out_base.name + "_by_roi.csv")
    sets_path = args.out_base.with_name(args.out_base.name + "_region_sets.csv")
    clean_table_path = args.out_base.with_name(args.out_base.name + "_clean_table.csv")
    summary_image_path = args.out_base.with_name(args.out_base.name + "_summary.png")
    clean_table_image_path = args.out_base.with_name(args.out_base.name + "_clean_table.png")
    coverage_path = args.out_base.with_name(args.out_base.name + "_atlas_coverage.csv")
    metadata_path = args.out_base.with_name(args.out_base.name + "_metadata.json")
    roi_df.to_csv(by_roi_path, index=False)
    set_df.to_csv(sets_path, index=False)
    clean_df.to_csv(clean_table_path, index=False)
    coverage_df.to_csv(coverage_path, index=False)
    _plot_summary_image(summary_image_path, clean_df, totals, args.atlas_mode)
    _plot_clean_table_image(clean_table_image_path, clean_df, totals, args.atlas_mode)
    metadata = {
        "inputs": mask_metadata,
        "atlas": atlas_metadata,
        "atlas_mode": args.atlas_mode,
        "exclude_white_matter": bool(args.exclude_white_matter),
        "totals": totals,
        "outputs": {
            "by_roi": str(by_roi_path),
            "region_sets": str(sets_path),
            "clean_table": str(clean_table_path),
            "summary_image": str(summary_image_path),
            "clean_table_image": str(clean_table_image_path),
            "atlas_coverage": str(coverage_path),
            "metadata": str(metadata_path),
            "report": str(args.out_base.with_name(args.out_base.name + "_report.md")),
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    _write_report(args.out_base, roi_df, set_df, coverage_df, mask_metadata, totals, args.atlas_mode)
    return {
        "by_roi": by_roi_path,
        "region_sets": sets_path,
        "clean_table": clean_table_path,
        "summary_image": summary_image_path,
        "clean_table_image": clean_table_image_path,
        "atlas_coverage": coverage_path,
        "metadata": metadata_path,
        "report": args.out_base.with_name(args.out_base.name + "_report.md"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-map", type=Path, default=DEFAULT_MAIN_MAP)
    parser.add_argument("--full-html", type=Path, default=DEFAULT_FULL_MODEL_HTML)
    parser.add_argument("--task-map", type=Path, default=DEFAULT_TASK_ONLY_MAP)
    parser.add_argument("--task-z-threshold", type=float, default=DEFAULT_TASK_ONLY_Z_THRESHOLD)
    parser.add_argument("--atlas-cache-dir", type=Path, default=DEFAULT_ATLAS_CACHE_DIR)
    parser.add_argument(
        "--atlas-mode",
        choices=("aal3", "aal3_ho_fill", "aal3_ho_nearest"),
        default="aal3_ho_nearest",
        help=(
            "Use AAL3 only, fill AAL3-unassigned voxels with Harvard-Oxford labels, or additionally "
            "assign remaining selected voxels to the nearest existing anatomical label."
        ),
    )
    parser.add_argument(
        "--include-white-matter",
        action="store_false",
        dest="exclude_white_matter",
        help="Include Harvard-Oxford cerebral white-matter voxels in the ROI quantification.",
    )
    parser.add_argument("--out-base", type=Path, default=DEFAULT_OUT_BASE)
    return parser.parse_args()


def main() -> None:
    outputs = run(parse_args())
    for label, path in outputs.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
