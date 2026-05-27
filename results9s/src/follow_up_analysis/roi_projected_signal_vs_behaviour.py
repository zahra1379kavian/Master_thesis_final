#!/usr/bin/env python3
"""Plot trial-wise weighted beta projection against behaviour RT within ROI nodes."""

from __future__ import annotations

import argparse
import json
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import datasets, image

from projected_signal_vs_behaviour import (
    DEFAULT_BEHAVIOUR_DIR,
    DEFAULT_BETA_DIR,
    DEFAULT_WEIGHT_MAP,
    ROOT,
    _discover_beta_runs,
    _load_behaviour_rt,
    _load_weights,
    _subject_digits,
)


DEFAULT_ANAT = ROOT / "data" / "anatomy" / "MNI152_T1_2mm_brain.nii.gz"
DEFAULT_OUT_DIR = ROOT / "results" / "follow_up_analysis" / "roi_projected_signal_vs_behaviour"
FSL_CEREBELLUM_MAXPROB_2MM = Path(
    "/usr/local/fsl/data/atlases/Cerebellum/Cerebellum-MNIfnirt-maxprob-thr25-2mm.nii.gz"
)
OUTLIER_MODIFIED_Z_THRESHOLD = 3.5


@dataclass
class ROIGroup:
    name: str
    source: str
    mask: np.ndarray
    matched_labels: list[str]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Project clean beta volumes through a voxel-weight map within each ROI node "
            "and scatter each ROI projection against behaviour RT."
        )
    )
    parser.add_argument("--beta-dir", type=Path, default=DEFAULT_BETA_DIR)
    parser.add_argument("--behaviour-dir", type=Path, default=DEFAULT_BEHAVIOUR_DIR)
    parser.add_argument("--weight-map", type=Path, default=DEFAULT_WEIGHT_MAP)
    parser.add_argument("--anat-path", type=Path, default=DEFAULT_ANAT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--behaviour-column", type=int, default=1, help="Zero-based RT column for 2D behaviour arrays.")
    parser.add_argument(
        "--roi-img",
        type=Path,
        default=None,
        help="Optional precomputed ROI label image. If omitted, ROI families are rebuilt from the atlas logic.",
    )
    parser.add_argument(
        "--roi-summary",
        type=Path,
        default=None,
        help="Optional JSON summary containing ROI names for --roi-img.",
    )
    parser.add_argument(
        "--atlas-cache-dir",
        type=Path,
        default=None,
        help="Optional nilearn atlas cache directory.",
    )
    parser.add_argument(
        "--min-roi-voxels",
        type=int,
        default=5,
        help="Minimum nonzero weighted voxels required to keep an ROI node.",
    )
    parser.add_argument(
        "--midline-band-mm",
        type=float,
        default=1.0,
        help="Absolute MNI x band treated as midline when splitting hemispheres.",
    )
    parser.add_argument(
        "--no-split-hemispheres",
        dest="split_hemispheres",
        action="store_false",
        help="Keep each ROI family unsplit instead of making L/R nodes.",
    )
    parser.set_defaults(split_hemispheres=True)
    parser.add_argument(
        "--exclude-roi-pattern",
        action="append",
        default=[],
        help="Case-insensitive ROI-name substring to exclude. Can be repeated.",
    )
    parser.add_argument(
        "--skip-roi-pngs",
        dest="save_roi_pngs",
        action="store_false",
        help="Skip per-ROI PNG scatter plots and save only the multipage PDF plus summaries.",
    )
    parser.set_defaults(save_roi_pngs=True)
    return parser.parse_args()


def _to_label_list(labels) -> list[str]:
    out = []
    for label in list(labels):
        if isinstance(label, bytes):
            out.append(label.decode("utf-8", errors="replace"))
        else:
            out.append(str(label))
    return out


def _fetch_ho_cort_sub(cache_dir: Path | None) -> tuple[nib.Nifti1Image, list[str], nib.Nifti1Image, list[str]]:
    data_dir = str(cache_dir) if cache_dir is not None else None
    cortical = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm", data_dir=data_dir)
    subcortical = datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr25-2mm", data_dir=data_dir)
    cortical_img = cortical.maps if isinstance(cortical.maps, nib.Nifti1Image) else nib.load(cortical.maps)
    subcortical_img = subcortical.maps if isinstance(subcortical.maps, nib.Nifti1Image) else nib.load(subcortical.maps)
    return cortical_img, _to_label_list(cortical.labels), subcortical_img, _to_label_list(subcortical.labels)


def _load_cerebellum_label_img() -> nib.Nifti1Image:
    if not FSL_CEREBELLUM_MAXPROB_2MM.exists():
        raise FileNotFoundError(f"Cerebellum atlas not found: {FSL_CEREBELLUM_MAXPROB_2MM}")
    return nib.load(str(FSL_CEREBELLUM_MAXPROB_2MM))


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
) -> tuple[list[int], list[str]]:
    include_lower = [item.lower() for item in include]
    exclude_lower = [item.lower() for item in (exclude or [])]
    ids: list[int] = []
    names: list[str] = []
    for idx, label in enumerate(labels):
        if idx == 0:
            continue
        lname = label.lower()
        if not any(pattern in lname for pattern in include_lower):
            continue
        if any(pattern in lname for pattern in exclude_lower):
            continue
        ids.append(idx)
        names.append(label)
    return ids, names


def _make_mask_from_labels(label_data: np.ndarray, label_ids: list[int], shape: tuple[int, int, int]) -> np.ndarray:
    if not label_ids:
        return np.zeros(shape, dtype=bool)
    return np.isin(label_data, label_ids)


def _build_base_requested_rois(
    reference_img: nib.Nifti1Image,
    cache_dir: Path | None,
) -> tuple[list[ROIGroup], dict[str, str], np.ndarray, list[str], np.ndarray, list[str]]:
    cort_img, cort_labels, sub_img, sub_labels = _fetch_ho_cort_sub(cache_dir)
    cereb_img = _load_cerebellum_label_img()

    cort_data = _resample_label_img(cort_img, reference_img)
    sub_data = _resample_label_img(sub_img, reference_img)
    cereb_data = _resample_label_img(cereb_img, reference_img)
    shape = reference_img.shape[:3]

    roi_groups: list[ROIGroup] = []

    def add_from_sub(name: str, patterns: list[str]) -> None:
        ids, names = _select_ids_by_patterns(sub_labels, include=patterns)
        roi_groups.append(
            ROIGroup(
                name=name,
                source="Harvard-Oxford Subcortical (thr25, 2mm)",
                mask=_make_mask_from_labels(sub_data, ids, shape),
                matched_labels=names,
            )
        )

    def add_from_cort(name: str, patterns: list[str], exclude: list[str] | None = None) -> None:
        ids, names = _select_ids_by_patterns(cort_labels, include=patterns, exclude=exclude)
        roi_groups.append(
            ROIGroup(
                name=name,
                source="Harvard-Oxford Cortical (thr25, 2mm)",
                mask=_make_mask_from_labels(cort_data, ids, shape),
                matched_labels=names,
            )
        )

    add_from_sub("Amygdala", ["amygdala"])
    roi_groups.append(
        ROIGroup(
            name="Cerebellum",
            source="FSL Cerebellum MNIfnirt (maxprob thr25, 2mm)",
            mask=cereb_data > 0,
            matched_labels=["All cerebellar labels (value > 0)"],
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

    atlas_info = {
        "cortical_atlas": "Harvard-Oxford Cortical (thr25, 2mm)",
        "subcortical_atlas": "Harvard-Oxford Subcortical (thr25, 2mm)",
        "cerebellum_atlas": str(FSL_CEREBELLUM_MAXPROB_2MM),
    }
    return roi_groups, atlas_info, cort_data, cort_labels, sub_data, sub_labels


def _selected_mask_from_groups(groups: list[ROIGroup], selected_ijk: np.ndarray) -> np.ndarray:
    x, y, z = selected_ijk.T
    covered = np.zeros(selected_ijk.shape[0], dtype=bool)
    for group in groups:
        covered |= group.mask[x, y, z]
    return covered


def _add_relative_rois(
    base_groups: list[ROIGroup],
    selected_ijk: np.ndarray,
    active_mask: np.ndarray,
    anat_shape: tuple[int, int, int],
    cort_data: np.ndarray,
    cort_labels: list[str],
    sub_data: np.ndarray,
    sub_labels: list[str],
) -> list[ROIGroup]:
    x, y, z = selected_ijk.T
    covered = _selected_mask_from_groups(base_groups, selected_ijk)
    remaining = active_mask & (~covered)

    if not np.any(remaining):
        return []

    extras: list[ROIGroup] = []
    candidates = [
        (
            "Occipital Cortex (relative)",
            "Harvard-Oxford Cortical (thr25, 2mm)",
            "cort",
            ["lateral occipital cortex", "occipital pole", "cuneal", "lingual", "intracalcarine", "supracalcarine"],
            None,
        ),
        (
            "Insular / Opercular Cortex (relative)",
            "Harvard-Oxford Cortical (thr25, 2mm)",
            "cort",
            ["insular cortex", "opercular cortex", "frontal opercular cortex", "central opercular cortex"],
            None,
        ),
        (
            "Hippocampus (relative)",
            "Harvard-Oxford Subcortical (thr25, 2mm)",
            "sub",
            ["hippocampus"],
            None,
        ),
        (
            "Basal Ganglia (relative)",
            "Harvard-Oxford Subcortical (thr25, 2mm)",
            "sub",
            ["caudate", "putamen", "pallidum", "accumbens"],
            None,
        ),
        (
            "Other Cerebral Cortex (relative)",
            "Harvard-Oxford Subcortical (thr25, 2mm)",
            "sub",
            ["cerebral cortex"],
            None,
        ),
    ]

    for roi_name, source, atlas_kind, include, exclude in candidates:
        if atlas_kind == "cort":
            ids, names = _select_ids_by_patterns(cort_labels, include=include, exclude=exclude)
            mask = _make_mask_from_labels(cort_data, ids, anat_shape)
        else:
            ids, names = _select_ids_by_patterns(sub_labels, include=include, exclude=exclude)
            mask = _make_mask_from_labels(sub_data, ids, anat_shape)

        covered_now = remaining & mask[x, y, z]
        if not np.any(covered_now):
            continue
        extras.append(ROIGroup(name=roi_name, source=source, mask=mask, matched_labels=names))
        remaining &= ~mask[x, y, z]
        if not np.any(remaining):
            break

    if np.any(remaining):
        fallback_mask = np.zeros(anat_shape, dtype=bool)
        fallback_mask[x[remaining], y[remaining], z[remaining]] = True
        extras.append(
            ROIGroup(
                name="Unassigned Active Voxels (relative)",
                source="Data-driven fallback",
                mask=fallback_mask,
                matched_labels=["No atlas label matched these active selected voxels."],
            )
        )

    return extras


def _build_label_data(groups: list[ROIGroup], shape: tuple[int, int, int]) -> np.ndarray:
    label_data = np.zeros(shape, dtype=np.int16)
    for roi_id, group in enumerate(groups, start=1):
        write_mask = group.mask & (label_data == 0)
        label_data[write_mask] = roi_id
    return label_data


def _load_roi_name_lookup(summary_path: Path | None) -> dict[int, str]:
    if summary_path is None or not summary_path.exists():
        return {}
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    names: dict[int, str] = {}
    rows = payload.get("roi_rows")
    if isinstance(rows, list):
        for row in rows:
            roi_id = row.get("roi_id")
            roi_name = row.get("roi_name")
            if isinstance(roi_id, int) and isinstance(roi_name, str) and roi_name:
                names[int(roi_id)] = roi_name
    all_names = payload.get("all_roi_names_in_order")
    if isinstance(all_names, list):
        for idx, name in enumerate(all_names, start=1):
            if idx not in names and isinstance(name, str) and name:
                names[idx] = name
    return names


def _load_or_build_roi_labels(
    weight_img: nib.Nifti1Image,
    weight_mask: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[int, str], dict[str, object]]:
    if args.roi_img is not None:
        roi_img = nib.load(str(args.roi_img))
        label_data = _resample_label_img(roi_img, weight_img)
        names = _load_roi_name_lookup(args.roi_summary)
        metadata = {
            "roi_definition": "precomputed_roi_img",
            "roi_img": str(args.roi_img),
            "roi_summary": str(args.roi_summary) if args.roi_summary else None,
        }
        return label_data, names, metadata

    if args.atlas_cache_dir is not None:
        args.atlas_cache_dir.mkdir(parents=True, exist_ok=True)

    selected_ijk = np.column_stack(np.nonzero(weight_mask)).astype(np.int32, copy=False)
    active_mask = np.ones(selected_ijk.shape[0], dtype=bool)
    base_groups, atlas_info, cort_data, cort_labels, sub_data, sub_labels = _build_base_requested_rois(
        reference_img=weight_img,
        cache_dir=args.atlas_cache_dir,
    )
    relative_groups = _add_relative_rois(
        base_groups=base_groups,
        selected_ijk=selected_ijk,
        active_mask=active_mask,
        anat_shape=weight_img.shape[:3],
        cort_data=cort_data,
        cort_labels=cort_labels,
        sub_data=sub_data,
        sub_labels=sub_labels,
    )
    groups = [*base_groups, *relative_groups]
    label_data = _build_label_data(groups, weight_img.shape[:3])
    names = {idx: group.name for idx, group in enumerate(groups, start=1)}
    metadata = {
        "roi_definition": "roi_edge_network_atlas_logic",
        "atlas_info": atlas_info,
        "n_base_requested_rois": len(base_groups),
        "n_relative_rois_added": len(relative_groups),
        "all_roi_names_in_order": [group.name for group in groups],
        "roi_sources": {group.name: group.source for group in groups},
    }
    return label_data, names, metadata


def _build_roi_nodes(
    label_data: np.ndarray,
    roi_names: dict[int, str],
    weights: np.ndarray,
    affine: np.ndarray,
    *,
    min_roi_voxels: int,
    split_hemispheres: bool,
    midline_band_mm: float,
    exclude_patterns: list[str],
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    weight_mask = np.isfinite(weights) & (weights != 0)
    selected_ijk = np.column_stack(np.nonzero(weight_mask)).astype(np.int32, copy=False)
    selected_flat = np.ravel_multi_index(selected_ijk.T, weights.shape).astype(np.int64, copy=False)
    x, y, z = selected_ijk.T
    labels_at_selected = label_data[x, y, z].astype(np.int32, copy=False)
    selected_coords_mm = nib.affines.apply_affine(affine, selected_ijk)
    selected_x = selected_coords_mm[:, 0]

    unique_ids, counts = np.unique(labels_at_selected[labels_at_selected > 0], return_counts=True)
    min_node_vox = int(max(1, min_roi_voxels))
    kept_roi_ids = unique_ids[counts >= min_node_vox].astype(np.int32, copy=False)
    exclude_lower = [pattern.strip().lower() for pattern in exclude_patterns if pattern.strip()]
    midline_band = float(max(0.0, midline_band_mm))

    nodes: list[dict[str, object]] = []
    for roi_id in kept_roi_ids.tolist():
        base_members = np.flatnonzero(labels_at_selected == int(roi_id)).astype(np.int64, copy=False)
        roi_name = roi_names.get(int(roi_id), f"ROI_{int(roi_id)}")
        if exclude_lower and any(pattern in roi_name.lower() for pattern in exclude_lower):
            continue

        node_specs: list[tuple[str, str, np.ndarray]]
        if split_hemispheres:
            roi_x = selected_x[base_members]
            left = base_members[roi_x < -midline_band]
            right = base_members[roi_x > midline_band]
            mid = base_members[np.abs(roi_x) <= midline_band]
            if mid.size > 0:
                if left.size >= right.size and left.size > 0:
                    left = np.concatenate([left, mid])
                elif right.size > 0:
                    right = np.concatenate([right, mid])
            node_specs = [("L", f"L {roi_name}", left), ("R", f"R {roi_name}", right)]
            if left.size < min_node_vox and right.size < min_node_vox and base_members.size >= min_node_vox:
                node_specs = [("M", f"M {roi_name}", base_members)]
        else:
            node_specs = [("B", roi_name, base_members)]

        for hemisphere, node_name, member_positions in node_specs:
            if member_positions.size < min_node_vox:
                continue
            flat_indices = selected_flat[member_positions]
            coords = np.mean(selected_coords_mm[member_positions], axis=0)
            nodes.append(
                {
                    "node_id": len(nodes) + 1,
                    "base_roi_id": int(roi_id),
                    "hemisphere": hemisphere,
                    "roi_name": roi_name,
                    "node_name": node_name,
                    "n_weighted_voxels": int(member_positions.size),
                    "flat_indices": flat_indices,
                    "x_mm": float(coords[0]),
                    "y_mm": float(coords[1]),
                    "z_mm": float(coords[2]),
                }
            )

    if not nodes:
        raise ValueError("No ROI nodes had enough nonzero weighted voxels.")

    rows = [
        {
            "node_id": node["node_id"],
            "base_roi_id": node["base_roi_id"],
            "hemisphere": node["hemisphere"],
            "roi_name": node["roi_name"],
            "node_name": node["node_name"],
            "n_weighted_voxels": node["n_weighted_voxels"],
            "x_mm": node["x_mm"],
            "y_mm": node["y_mm"],
            "z_mm": node["z_mm"],
        }
        for node in nodes
    ]
    return nodes, pd.DataFrame(rows)


def _align_projection_matrix(
    projection_matrix: np.ndarray,
    behaviour_rt: np.ndarray,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    n_projection = projection_matrix.shape[1]
    n_behaviour = behaviour_rt.shape[0]
    if n_projection == n_behaviour:
        return projection_matrix, behaviour_rt

    n_keep = min(n_projection, n_behaviour)
    warnings.warn(
        f"{label}: projection has {n_projection} trials and behaviour has {n_behaviour}; "
        f"truncating both to {n_keep}.",
        stacklevel=2,
    )
    return projection_matrix[:, :n_keep], behaviour_rt[:n_keep]


def _project_beta_nodes(beta_path: Path, weights: np.ndarray, nodes: list[dict[str, object]]) -> np.ndarray:
    beta = np.load(beta_path, mmap_mode="r")
    if beta.ndim != 4:
        raise ValueError(f"Expected 4D beta volume in {beta_path}, got shape {beta.shape}")
    if beta.shape[:3] != weights.shape:
        raise ValueError(
            f"Spatial shape mismatch for {beta_path}: beta {beta.shape[:3]} vs weights {weights.shape}"
        )

    n_trials = beta.shape[3]
    flat_weights = weights.reshape(-1)
    beta_2d = beta.reshape(-1, n_trials)
    projected = np.full((len(nodes), n_trials), np.nan, dtype=np.float64)

    for node_idx, node in enumerate(nodes):
        flat_indices = np.asarray(node["flat_indices"], dtype=np.int64)
        selected_weights = flat_weights[flat_indices].astype(np.float64, copy=False)
        selected_beta = np.asarray(beta_2d[flat_indices, :], dtype=np.float64)
        finite_beta = np.isfinite(selected_beta)
        numerator = np.nansum(selected_beta * selected_weights[:, None], axis=0)
        denominator = np.sum(finite_beta * selected_weights[:, None], axis=0)
        valid = denominator != 0
        projected[node_idx, valid] = numerator[valid] / denominator[valid]

    return projected


def _build_roi_projection_table(
    beta_dir: Path,
    behaviour_dir: Path,
    weights: np.ndarray,
    roi_nodes: list[dict[str, object]],
    behaviour_column: int,
) -> pd.DataFrame:
    rows = []
    for beta_run in _discover_beta_runs(beta_dir):
        sub = str(beta_run["sub"])
        ses = int(beta_run["ses"])
        run = int(beta_run["run"])
        beta_path = Path(beta_run["path"])

        behaviour_path = behaviour_dir / f"PSPD{_subject_digits(sub)}_ses_{ses}_run_{run}.npy"
        if not behaviour_path.exists():
            raise FileNotFoundError(f"Missing behaviour file for {sub} ses-{ses} run-{run}: {behaviour_path}")

        projection_matrix = _project_beta_nodes(beta_path, weights, roi_nodes)
        behaviour_rt = _load_behaviour_rt(behaviour_path, behaviour_column)
        projection_matrix, behaviour_rt = _align_projection_matrix(
            projection_matrix,
            behaviour_rt,
            f"{sub} ses-{ses} run-{run}",
        )

        for node_idx, node in enumerate(roi_nodes):
            projections = projection_matrix[node_idx]
            for trial_idx, (projection, rt) in enumerate(zip(projections, behaviour_rt), start=1):
                rows.append(
                    {
                        "sub": sub,
                        "ses": ses,
                        "run": run,
                        "trial": trial_idx,
                        "node_id": int(node["node_id"]),
                        "base_roi_id": int(node["base_roi_id"]),
                        "hemisphere": str(node["hemisphere"]),
                        "roi_name": str(node["roi_name"]),
                        "roi_label": str(node["node_name"]),
                        "n_weighted_voxels": int(node["n_weighted_voxels"]),
                        "projected_signal": projection,
                        "behaviour_rt": rt,
                    }
                )

    if not rows:
        raise ValueError("No ROI projection/behaviour rows were created.")
    return pd.DataFrame(rows)


def _remove_rt_outliers_by_trial(df: pd.DataFrame) -> pd.DataFrame:
    trial_cols = ["sub", "ses", "run", "trial"]
    unique_trials = df.loc[:, [*trial_cols, "behaviour_rt"]].drop_duplicates().copy()
    unique_trials["rt_outlier"] = False
    for _, group in unique_trials.groupby(["sub", "ses", "run"], sort=False):
        unique_trials.loc[group.index, "rt_outlier"] = _rt_outlier_mask(group["behaviour_rt"])

    n_outlier_trials = int(unique_trials["rt_outlier"].sum())
    if n_outlier_trials:
        print(f"Removed {n_outlier_trials} RT outlier trials before ROI summaries.")

    tagged = df.merge(unique_trials.loc[:, [*trial_cols, "rt_outlier"]], on=trial_cols, how="left")
    filtered = tagged.loc[~tagged["rt_outlier"].fillna(False)].drop(columns="rt_outlier")
    return filtered.copy()


def _rt_outlier_mask(rt: pd.Series) -> np.ndarray:
    values = rt.to_numpy(dtype=np.float64)
    finite = np.isfinite(values)
    outliers = np.zeros(values.shape, dtype=bool)
    if np.count_nonzero(finite) < 3:
        return outliers

    median = np.nanmedian(values[finite])
    mad = np.nanmedian(np.abs(values[finite] - median))
    if not np.isfinite(mad) or mad <= 0:
        return outliers

    modified_z = 0.6745 * (values[finite] - median) / mad
    outliers[finite] = np.abs(modified_z) > OUTLIER_MODIFIED_Z_THRESHOLD
    return outliers


def _corr_summary(group: pd.DataFrame) -> dict[str, float | int]:
    finite = np.isfinite(group["projected_signal"]) & np.isfinite(group["behaviour_rt"])
    n_finite = int(finite.sum())
    out: dict[str, float | int] = {
        "n_trials": int(len(group)),
        "n_finite": n_finite,
        "n_runs": int(group["run"].nunique()),
        "pearson_r": np.nan,
        "slope": np.nan,
        "intercept": np.nan,
    }
    if n_finite < 3:
        return out
    x = group.loc[finite, "projected_signal"].to_numpy(dtype=np.float64)
    y = group.loc[finite, "behaviour_rt"].to_numpy(dtype=np.float64)
    if np.nanmax(x) <= np.nanmin(x) or np.nanmax(y) <= np.nanmin(y):
        return out
    out["pearson_r"] = float(np.corrcoef(x, y)[0, 1])
    slope, intercept = np.polyfit(x, y, deg=1)
    out["slope"] = float(slope)
    out["intercept"] = float(intercept)
    return out


def _summarize_roi_projection(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = [
        "node_id",
        "base_roi_id",
        "hemisphere",
        "roi_name",
        "roi_label",
        "n_weighted_voxels",
        "sub",
        "ses",
    ]
    for keys, group in df.groupby(group_cols, sort=False):
        row = dict(zip(group_cols, keys))
        row.update(_corr_summary(group))
        rows.append(row)
    return pd.DataFrame(rows)


def _safe_slug(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", text.strip()).strip("_").lower()
    return slug or "roi"


def _draw_roi_scatter(
    roi_df: pd.DataFrame,
    roi_label: str,
    subjects: list[str],
    sessions: list[int],
) -> plt.Figure:
    n_rows = len(subjects)
    n_cols = len(sessions)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.2 * n_cols, 4.2 * n_rows), squeeze=False)
    session_colors = {1: "#4C78A8", 2: "#C23B4B"}

    for row_idx, sub in enumerate(subjects):
        for col_idx, ses in enumerate(sessions):
            ax = axes[row_idx, col_idx]
            panel_df = roi_df[(roi_df["sub"] == sub) & (roi_df["ses"] == ses)]
            if panel_df.empty:
                ax.axis("off")
                continue

            ax.scatter(
                panel_df["projected_signal"],
                panel_df["behaviour_rt"],
                s=22,
                alpha=0.68,
                linewidths=0,
                color=session_colors.get(int(ses), "#7A7A7A"),
            )

            finite = np.isfinite(panel_df["projected_signal"]) & np.isfinite(panel_df["behaviour_rt"])
            n_finite = int(finite.sum())
            corr_text = "r=NA"
            if n_finite >= 3:
                x = panel_df.loc[finite, "projected_signal"].to_numpy(dtype=np.float64)
                y = panel_df.loc[finite, "behaviour_rt"].to_numpy(dtype=np.float64)
                if np.nanmax(x) > np.nanmin(x) and np.nanmax(y) > np.nanmin(y):
                    corr = np.corrcoef(x, y)[0, 1]
                    corr_text = f"r={corr:.2f}"
                    slope, intercept = np.polyfit(x, y, deg=1)
                    x_fit = np.linspace(np.nanmin(x), np.nanmax(x), 100)
                    ax.plot(x_fit, slope * x_fit + intercept, color="black", linewidth=1.8)

            ax.set_title(f"{sub} ses-{int(ses)} ({corr_text}, n={n_finite})")
            ax.set_xlabel("Projected signal")
            ax.set_ylabel("Behaviour RT")
            ax.grid(alpha=0.25)

    fig.suptitle(f"{roi_label}: projected signal vs behaviour RT", y=0.995)
    fig.tight_layout()
    return fig


def _save_roi_scatter_outputs(
    df: pd.DataFrame,
    roi_nodes: pd.DataFrame,
    out_dir: Path,
    save_roi_pngs: bool,
) -> None:
    subjects = sorted(df["sub"].unique())
    sessions = sorted(int(value) for value in df["ses"].unique())
    png_dir = out_dir / "projected_signal_vs_behaviour_by_roi_pngs"
    if save_roi_pngs:
        png_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = out_dir / "projected_signal_vs_behaviour_by_roi.pdf"
    with PdfPages(pdf_path) as pdf:
        for row in roi_nodes.itertuples(index=False):
            roi_label = str(row.node_name)
            roi_df = df[df["node_id"] == int(row.node_id)]
            fig = _draw_roi_scatter(roi_df, roi_label, subjects, sessions)
            pdf.savefig(fig, bbox_inches="tight")
            if save_roi_pngs:
                fig.savefig(png_dir / f"{int(row.node_id):02d}_{_safe_slug(roi_label)}.png", dpi=250, bbox_inches="tight")
            plt.close(fig)


def _save_correlation_heatmap(summary_df: pd.DataFrame, roi_nodes: pd.DataFrame, out_dir: Path) -> None:
    plot_df = summary_df.copy()
    plot_df["subject_session"] = plot_df["sub"].astype(str) + " ses-" + plot_df["ses"].astype(int).astype(str)
    column_order = sorted(plot_df["subject_session"].unique())
    row_order = roi_nodes["node_name"].astype(str).tolist()
    heatmap = (
        plot_df.pivot_table(index="roi_label", columns="subject_session", values="pearson_r", aggfunc="mean")
        .reindex(index=row_order, columns=column_order)
    )

    values = heatmap.to_numpy(dtype=np.float64)
    fig_width = max(8.0, 0.95 * len(column_order) + 3.0)
    fig_height = max(7.0, 0.32 * len(row_order) + 2.2)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    cmap = matplotlib.colormaps["RdBu_r"].copy()
    cmap.set_bad("#eeeeee")
    im = ax.imshow(values, cmap=cmap, vmin=-1.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(column_order)))
    ax.set_xticklabels(column_order, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks(np.arange(len(row_order)))
    ax.set_yticklabels(row_order, fontsize=8)
    ax.set_xlabel("Subject/session")
    ax.set_ylabel("ROI node")
    ax.set_title("ROI projected signal vs behaviour RT correlation")
    ax.set_xticks(np.arange(-0.5, len(column_order), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_order), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.7)
    ax.tick_params(which="minor", bottom=False, left=False)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7, color="black")
    cbar = fig.colorbar(im, ax=ax, shrink=0.86, pad=0.02)
    cbar.set_label("Pearson r")
    fig.tight_layout()
    fig.savefig(out_dir / "roi_projected_signal_vs_behaviour_correlation_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    weights = _load_weights(args.weight_map)
    weight_img = nib.load(str(args.weight_map))
    if weight_img.shape[:3] != weights.shape:
        raise ValueError(f"Unexpected weight image shape: {weight_img.shape}")
    if args.anat_path.exists():
        anat_img = nib.load(str(args.anat_path))
        if anat_img.shape[:3] != weight_img.shape[:3] or not np.allclose(anat_img.affine, weight_img.affine):
            warnings.warn(
                f"Anatomy {args.anat_path} does not match the weight map grid; ROI construction uses the weight-map grid.",
                stacklevel=2,
            )

    weight_mask = np.isfinite(weights) & (weights != 0)
    label_data, roi_names, roi_metadata = _load_or_build_roi_labels(weight_img, weight_mask, args)
    roi_nodes, roi_node_df = _build_roi_nodes(
        label_data=label_data,
        roi_names=roi_names,
        weights=weights,
        affine=weight_img.affine,
        min_roi_voxels=args.min_roi_voxels,
        split_hemispheres=args.split_hemispheres,
        midline_band_mm=args.midline_band_mm,
        exclude_patterns=args.exclude_roi_pattern,
    )

    projection_df = _build_roi_projection_table(
        beta_dir=args.beta_dir,
        behaviour_dir=args.behaviour_dir,
        weights=weights,
        roi_nodes=roi_nodes,
        behaviour_column=args.behaviour_column,
    )
    projection_df = _remove_rt_outliers_by_trial(projection_df)
    summary_df = _summarize_roi_projection(projection_df)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    roi_node_df.to_csv(args.out_dir / "roi_projected_signal_vs_behaviour_nodes.csv", index=False)
    projection_df.to_csv(args.out_dir / "roi_projected_signal_vs_behaviour_trials.csv", index=False)
    summary_df.to_csv(args.out_dir / "roi_projected_signal_vs_behaviour_summary.csv", index=False)
    _save_correlation_heatmap(summary_df, roi_node_df, args.out_dir)
    _save_roi_scatter_outputs(projection_df, roi_node_df, args.out_dir, save_roi_pngs=args.save_roi_pngs)

    metadata = {
        "weight_map": str(args.weight_map),
        "beta_dir": str(args.beta_dir),
        "behaviour_dir": str(args.behaviour_dir),
        "behaviour_column": int(args.behaviour_column),
        "n_nonzero_weight_voxels": int(np.count_nonzero(weight_mask)),
        "min_roi_voxels": int(args.min_roi_voxels),
        "split_hemispheres": bool(args.split_hemispheres),
        "midline_band_mm": float(args.midline_band_mm),
        "n_roi_nodes": int(len(roi_nodes)),
        **roi_metadata,
    }
    (args.out_dir / "roi_projected_signal_vs_behaviour_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(f"Kept {len(roi_nodes)} ROI nodes from {int(np.count_nonzero(weight_mask))} nonzero weighted voxels.")
    print(f"Saved ROI projection outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
