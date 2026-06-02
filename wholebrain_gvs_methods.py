#!/usr/bin/env python3
"""Weighted p90-network GVS contrast analyses.

This script avoids ROI partitioning. It keeps only the p90 positive-weight
network voxels, multiplies beta values by the voxel weights, builds
subject-level GVS-minus-sham contrast maps from both runs within each session,
and runs second-level permutation/TFCE tests.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.maskers import NiftiMasker
from nilearn.mass_univariate import permuted_ols
from nilearn.plotting import plot_stat_map
from scipy import ndimage, stats

from GVS_effect import (
    DEFAULT_BETA_ROOT,
    DEFAULT_GVS_ORDER,
    DEFAULT_TRIALS_PER_CONDITION,
    DEFAULT_WEIGHT_MAP,
    _condition_display_name,
    _discover_run_specs,
    _iter_condition_slices,
    _load_gvs_order,
    _medication_from_session,
    _resolve_path,
    _subject_sort_key,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = ROOT / "figures" / "GVS_effects" / "wholebrain_methods"
DEFAULT_NETWORK_HTML = ROOT / "data" / "voxel_weights_task1_bold0.6_beta0.6_smooth1.25_gamma1.5_bold_thr90.html"
DEFAULT_NETWORK_WEIGHT_PERCENTILE = 90.0
ACTIVE_CONDITION_CODES = tuple(f"gvs-{index:02d}" for index in range(2, 10))
SHAM_CONDITION_CODE = "gvs-01"
CONTRAST_SPECS = {
    "active_mean": ACTIVE_CONDITION_CODES,
    "gvs2_dc_plus_1": ("gvs-03",),
    "gvs5_theta": ("gvs-06",),
    "gvs6_alpha": ("gvs-07",),
}
PRIMARY_GROUP_TESTS = (
    ("weighted_network_active_mean_OFF", "weighted_network", "active_mean", "OFF"),
    ("weighted_network_active_mean_ON", "weighted_network", "active_mean", "ON"),
    ("weighted_network_active_mean_ON_minus_OFF", "weighted_network", "active_mean", "ON_minus_OFF"),
)
EXPLORATORY_GROUP_TESTS = (
    ("weighted_network_gvs2_dc_plus_1_OFF", "weighted_network", "gvs2_dc_plus_1", "OFF"),
    ("weighted_network_gvs5_theta_OFF", "weighted_network", "gvs5_theta", "OFF"),
    ("weighted_network_gvs6_alpha_OFF", "weighted_network", "gvs6_alpha", "OFF"),
)
FWE_ALPHA = 0.05
UNCORRECTED_CLUSTER_P = 0.001
NAN_FILL = 0.0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beta-root", type=Path, default=DEFAULT_BETA_ROOT)
    parser.add_argument("--gvs-order", type=Path, default=DEFAULT_GVS_ORDER)
    parser.add_argument("--weight-map", type=Path, default=DEFAULT_WEIGHT_MAP)
    parser.add_argument("--network-html", type=Path, default=DEFAULT_NETWORK_HTML)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--trials-per-condition", type=int, default=DEFAULT_TRIALS_PER_CONDITION)
    parser.add_argument("--network-weight-percentile", type=float, default=DEFAULT_NETWORK_WEIGHT_PERCENTILE)
    parser.add_argument("--n-perm", type=int, default=2000)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--random-state", type=int, default=1301)
    parser.add_argument("--no-tfce", action="store_true")
    parser.add_argument("--searchlight-radius-mm", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--searchlight-min-voxels", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--skip-searchlight", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--include-exploratory", action="store_true")
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--subjects", nargs="+", default=None)
    return parser


def _prepare_args(args: argparse.Namespace) -> argparse.Namespace:
    args.beta_root = _resolve_path(args.beta_root)
    args.gvs_order = _resolve_path(args.gvs_order)
    args.weight_map = _resolve_path(args.weight_map)
    args.network_html = _resolve_path(args.network_html)
    args.out_dir = _resolve_path(args.out_dir)
    if int(args.trials_per_condition) <= 0:
        raise ValueError("--trials-per-condition must be positive.")
    if int(args.n_perm) <= 0:
        raise ValueError("--n-perm must be positive.")
    if not (0.0 <= float(args.network_weight_percentile) <= 100.0):
        raise ValueError("--network-weight-percentile must be between 0 and 100.")
    return args


def _flat_to_img(flat: np.ndarray, shape: tuple[int, int, int], affine: np.ndarray) -> nib.Nifti1Image:
    data = np.asarray(flat, dtype=np.float32).reshape(shape)
    return nib.Nifti1Image(data, affine)


def _save_flat_img(flat: np.ndarray, path: Path, shape: tuple[int, int, int], affine: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(_flat_to_img(flat, shape, affine), str(path))


def _load_flat_img(path: str | Path) -> np.ndarray:
    return np.asarray(nib.load(str(path)).get_fdata(), dtype=np.float32).ravel()


def _load_maps_from_manifest(path: Path) -> tuple[dict[tuple[str, int, str], np.ndarray], pd.DataFrame]:
    if not path.exists():
        raise RuntimeError(f"Missing reusable manifest: {path}")
    manifest = pd.read_csv(path)
    maps: dict[tuple[str, int, str], np.ndarray] = {}
    for row in manifest.itertuples(index=False):
        maps[(str(row.subject), int(row.session), str(row.contrast))] = _load_flat_img(str(row.path))
    return maps, manifest


def _network_weight_mask(
    weight_values: np.ndarray,
    percentile: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    weights = np.asarray(weight_values, dtype=np.float32).ravel()
    positive = np.isfinite(weights) & (weights > 0)
    if not np.any(positive):
        raise RuntimeError("Weighted-network analysis requires at least one positive finite weight.")

    threshold = float(np.percentile(weights[positive], float(percentile)))
    mask = positive & (weights >= threshold)
    if not np.any(mask):
        raise RuntimeError("Weighted-network analysis selected zero voxels.")

    metadata = {
        "network_weight_percentile": float(percentile),
        "network_weight_threshold": threshold,
        "n_positive_weight_voxels": int(np.sum(positive)),
        "n_network_voxels": int(np.sum(mask)),
    }
    return mask, weights, metadata


def _condition_means_for_run(
    beta_path: Path,
    stim_order: tuple[int, ...],
    network_mask_flat: np.ndarray,
    *,
    trials_per_condition: int,
) -> dict[str, np.ndarray]:
    beta = np.load(beta_path, mmap_mode="r")
    if beta.ndim != 4:
        raise RuntimeError(f"{beta_path} must be 4D, got shape {beta.shape}.")
    flat = beta.reshape(-1, int(beta.shape[-1]))
    if flat.shape[0] != int(network_mask_flat.size):
        raise RuntimeError(
            f"{beta_path} has {flat.shape[0]} voxels but the weight-map mask has {network_mask_flat.size}."
        )
    means: dict[str, np.ndarray] = {}
    for condition_code, start, stop in _iter_condition_slices(
        int(beta.shape[-1]),
        stim_order,
        int(trials_per_condition),
    ):
        selected = np.asarray(flat[network_mask_flat, int(start) : int(stop)], dtype=np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            means[condition_code] = np.nanmean(selected, axis=1).astype(np.float32)
    return means


def _mean_available(arrays: list[np.ndarray]) -> np.ndarray:
    if not arrays:
        raise RuntimeError("No arrays were provided for averaging.")
    finite_sum = np.zeros_like(arrays[0], dtype=np.float32)
    finite_count = np.zeros(arrays[0].shape, dtype=np.uint8)
    for array in arrays:
        finite = np.isfinite(array)
        finite_sum[finite] += array[finite].astype(np.float32, copy=False)
        finite_count[finite] += 1
    out = np.full(arrays[0].shape, np.nan, dtype=np.float32)
    valid = finite_count > 0
    out[valid] = finite_sum[valid] / finite_count[valid].astype(np.float32)
    return out


def _build_subject_contrasts(
    run_specs: list[Any],
    network_mask_flat: np.ndarray,
    network_weight_flat: np.ndarray,
    *,
    trials_per_condition: int,
) -> tuple[dict[tuple[str, int, str], np.ndarray], pd.DataFrame]:
    run_contrast_parts: dict[tuple[str, int, str], list[np.ndarray]] = defaultdict(list)
    rows: list[dict[str, Any]] = []
    selected_weights = np.asarray(network_weight_flat[network_mask_flat], dtype=np.float32)
    for index, spec in enumerate(run_specs, start=1):
        print(
            f"Computing run contrasts {index}/{len(run_specs)}: "
            f"{spec.subject} ses-{spec.session} run-{spec.run}",
            flush=True,
        )
        means = _condition_means_for_run(
            spec.beta_path,
            spec.stim_order,
            network_mask_flat,
            trials_per_condition=trials_per_condition,
        )
        sham = means.get(SHAM_CONDITION_CODE)
        if sham is None:
            raise RuntimeError(f"{spec.subject} ses-{spec.session} run-{spec.run} has no sham block.")
        for contrast_name, condition_codes in CONTRAST_SPECS.items():
            available = [means[code] for code in condition_codes if code in means]
            if not available:
                continue
            active_mean = _mean_available(available)
            weighted_delta = ((active_mean - sham) * selected_weights).astype(np.float32)
            full_delta = np.full(network_mask_flat.shape, np.nan, dtype=np.float32)
            full_delta[network_mask_flat] = weighted_delta
            run_contrast_parts[(spec.subject, int(spec.session), contrast_name)].append(full_delta)
            rows.append(
                {
                    "subject": spec.subject,
                    "session": int(spec.session),
                    "medication": _medication_from_session(int(spec.session)),
                    "run": int(spec.run),
                    "contrast": contrast_name,
                    "condition_codes": ",".join(condition_codes),
                    "condition_labels": ",".join(_condition_display_name(code) for code in condition_codes),
                    "n_condition_means": int(len(available)),
                    "n_network_voxels": int(np.sum(network_mask_flat)),
                    "beta_weighting": "voxel_beta_times_network_weight",
                    "source_beta_path": str(spec.beta_path),
                }
            )

    subject_contrasts = {
        key: _mean_available(parts)
        for key, parts in run_contrast_parts.items()
        if parts
    }
    return subject_contrasts, pd.DataFrame(rows)


def _common_finite_mask(arrays: list[np.ndarray]) -> np.ndarray:
    if not arrays:
        raise RuntimeError("Cannot build a mask from zero images.")
    mask = np.ones(arrays[0].shape, dtype=bool)
    for array in arrays:
        mask &= np.isfinite(array)
    return mask


def _write_subject_contrast_images(
    subject_contrasts: dict[tuple[str, int, str], np.ndarray],
    out_dir: Path,
    shape: tuple[int, int, int],
    affine: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    contrast_dir = out_dir / "subject_contrasts"
    for (subject, session, contrast), flat in sorted(
        subject_contrasts.items(),
        key=lambda item: (_subject_sort_key(item[0][0]), item[0][1], item[0][2]),
    ):
        path = contrast_dir / contrast / f"{subject}_ses-{session}_{contrast}_minus_sham.nii.gz"
        _save_flat_img(flat, path, shape, affine)
        rows.append(
            {
                "subject": subject,
                "session": int(session),
                "medication": _medication_from_session(int(session)),
                "contrast": contrast,
                "path": str(path),
                "finite_voxels": int(np.isfinite(flat).sum()),
                "mean": float(np.nanmean(flat)),
                "std": float(np.nanstd(flat)),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "subject_contrast_manifest.csv", index=False)
    return df


def _arrays_for_group_test(
    maps: dict[tuple[str, int, str], np.ndarray],
    contrast: str,
    group: str,
) -> tuple[list[str], list[np.ndarray]]:
    if group in {"OFF", "ON"}:
        session = 1 if group == "OFF" else 2
        pairs = [
            (subject, array)
            for (subject, map_session, map_contrast), array in maps.items()
            if int(map_session) == session and map_contrast == contrast
        ]
        pairs.sort(key=lambda item: _subject_sort_key(item[0]))
        return [subject for subject, _ in pairs], [array for _, array in pairs]

    if group == "ON_minus_OFF":
        off = {
            subject: array
            for (subject, session, map_contrast), array in maps.items()
            if int(session) == 1 and map_contrast == contrast
        }
        on = {
            subject: array
            for (subject, session, map_contrast), array in maps.items()
            if int(session) == 2 and map_contrast == contrast
        }
        subjects = sorted(set(off).intersection(on), key=_subject_sort_key)
        return subjects, [(on[subject] - off[subject]).astype(np.float32) for subject in subjects]

    raise ValueError(f"Unknown group test {group!r}.")


def _write_map_plot(stat_path: Path, out_path: Path, title: str, threshold: float | None = None) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    display = plot_stat_map(
        str(stat_path),
        threshold=threshold,
        display_mode="ortho",
        cut_coords=None,
        colorbar=True,
        title=title,
    )
    display.savefig(str(out_path), dpi=180)
    display.close()


def _uncorrected_cluster_rows(
    test_name: str,
    t_map: np.ndarray,
    mean_map: np.ndarray,
    mask_flat: np.ndarray,
    shape: tuple[int, int, int],
    affine: np.ndarray,
    n_subjects: int,
) -> list[dict[str, Any]]:
    if n_subjects < 3:
        return []
    tcrit = float(stats.t.isf(UNCORRECTED_CLUSTER_P / 2.0, df=int(n_subjects) - 1))
    t_img = t_map.reshape(shape)
    mean_img = mean_map.reshape(shape)
    mask_img = mask_flat.reshape(shape)
    rows: list[dict[str, Any]] = []
    structure = ndimage.generate_binary_structure(3, 2)
    for sign_name, sign, thresholded in (
        ("positive", 1, (t_img >= tcrit) & mask_img),
        ("negative", -1, (t_img <= -tcrit) & mask_img),
    ):
        labels, n_labels = ndimage.label(thresholded, structure=structure)
        objects = ndimage.find_objects(labels)
        for label_idx in range(1, n_labels + 1):
            loc = objects[label_idx - 1]
            if loc is None:
                continue
            cluster_mask = labels[loc] == label_idx
            size = int(cluster_mask.sum())
            if size <= 0:
                continue
            local_t = t_img[loc][cluster_mask]
            local_mean = mean_img[loc][cluster_mask]
            peak_local = int(np.argmax(local_t if sign > 0 else -local_t))
            coords_local = np.argwhere(cluster_mask)[peak_local]
            starts = np.array([sl.start for sl in loc], dtype=int)
            ijk = starts + coords_local
            xyz = nib.affines.apply_affine(affine, ijk)
            peak_t = float(local_t[peak_local])
            rows.append(
                {
                    "test": test_name,
                    "sign": sign_name,
                    "cluster_voxels": size,
                    "peak_t": peak_t,
                    "peak_abs_t": abs(peak_t),
                    "peak_mean_effect": float(local_mean[peak_local]),
                    "mni_x": float(xyz[0]),
                    "mni_y": float(xyz[1]),
                    "mni_z": float(xyz[2]),
                    "uncorrected_cluster_p": UNCORRECTED_CLUSTER_P,
                }
            )
    rows.sort(key=lambda row: (row["cluster_voxels"], row["peak_abs_t"]), reverse=True)
    return rows


def _run_group_test(
    test_name: str,
    arrays: list[np.ndarray],
    subjects: list[str],
    mask_flat: np.ndarray,
    mask_img: nib.Nifti1Image,
    out_dir: Path,
    shape: tuple[int, int, int],
    affine: np.ndarray,
    *,
    n_perm: int,
    n_jobs: int,
    random_state: int,
    use_tfce: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if len(arrays) < 3:
        raise RuntimeError(f"{test_name} has fewer than 3 subject images.")
    finite_mask = mask_flat.copy()
    for array in arrays:
        finite_mask &= np.isfinite(array)
    if int(finite_mask.sum()) < 10:
        raise RuntimeError(f"{test_name} has too few finite voxels.")

    group_dir = out_dir / "group_tests" / test_name
    group_dir.mkdir(parents=True, exist_ok=True)
    local_mask_img = nib.Nifti1Image(finite_mask.reshape(shape).astype(np.uint8), affine)
    masker = NiftiMasker(mask_img=local_mask_img, standardize=False).fit()
    target = np.vstack([array[finite_mask] for array in arrays]).astype(np.float32)
    tested = np.ones((target.shape[0], 1), dtype=np.float32)
    print(
        f"Running {'TFCE ' if use_tfce else ''}permutation test {test_name}: "
        f"n={target.shape[0]}, voxels={target.shape[1]}, permutations={n_perm}",
        flush=True,
    )
    result = permuted_ols(
        tested,
        target,
        model_intercept=False,
        n_perm=int(n_perm),
        two_sided_test=True,
        random_state=int(random_state),
        n_jobs=int(n_jobs),
        verbose=0,
        masker=masker,
        tfce=bool(use_tfce),
        output_type="dict",
    )
    mean_values = np.mean(target, axis=0)
    t_values = np.asarray(result["t"][0], dtype=np.float32)
    logp_voxel = np.asarray(result["logp_max_t"][0], dtype=np.float32)
    if use_tfce:
        logp_tfce = np.asarray(result["logp_max_tfce"][0], dtype=np.float32)
        tfce_values = np.asarray(result["tfce"][0], dtype=np.float32)
    else:
        logp_tfce = np.full_like(t_values, np.nan, dtype=np.float32)
        tfce_values = np.full_like(t_values, np.nan, dtype=np.float32)

    full_mean = np.full(mask_flat.shape, np.nan, dtype=np.float32)
    full_t = np.full(mask_flat.shape, np.nan, dtype=np.float32)
    full_logp_tfce = np.full(mask_flat.shape, np.nan, dtype=np.float32)
    full_tfce = np.full(mask_flat.shape, np.nan, dtype=np.float32)
    full_logp_voxel = np.full(mask_flat.shape, np.nan, dtype=np.float32)
    full_mean[finite_mask] = mean_values
    full_t[finite_mask] = t_values
    full_logp_tfce[finite_mask] = logp_tfce
    full_tfce[finite_mask] = tfce_values
    full_logp_voxel[finite_mask] = logp_voxel

    output_paths = {
        "mean": group_dir / f"{test_name}_mean.nii.gz",
        "t": group_dir / f"{test_name}_t.nii.gz",
        "tfce": group_dir / f"{test_name}_tfce.nii.gz",
        "neglog10p_tfce_fwe": group_dir / f"{test_name}_neglog10p_tfce_fwe.nii.gz",
        "neglog10p_voxel_fwe": group_dir / f"{test_name}_neglog10p_voxel_fwe.nii.gz",
        "mask": group_dir / f"{test_name}_mask.nii.gz",
    }
    _save_flat_img(np.nan_to_num(full_mean, nan=NAN_FILL), output_paths["mean"], shape, affine)
    _save_flat_img(np.nan_to_num(full_t, nan=NAN_FILL), output_paths["t"], shape, affine)
    if use_tfce:
        _save_flat_img(np.nan_to_num(full_tfce, nan=NAN_FILL), output_paths["tfce"], shape, affine)
        _save_flat_img(np.nan_to_num(full_logp_tfce, nan=NAN_FILL), output_paths["neglog10p_tfce_fwe"], shape, affine)
    _save_flat_img(np.nan_to_num(full_logp_voxel, nan=NAN_FILL), output_paths["neglog10p_voxel_fwe"], shape, affine)
    nib.save(local_mask_img, str(output_paths["mask"]))

    fwe_threshold = -math.log10(FWE_ALPHA)
    sig_tfce = finite_mask & (full_logp_tfce > fwe_threshold) if use_tfce else np.zeros_like(finite_mask, dtype=bool)
    sig_voxel = finite_mask & (full_logp_voxel > fwe_threshold)
    if np.any(sig_tfce):
        thresholded_t = np.where(sig_tfce, full_t, 0.0).astype(np.float32)
        thresholded_path = group_dir / f"{test_name}_t_tfce_fwe05_thresholded.nii.gz"
        _save_flat_img(thresholded_t, thresholded_path, shape, affine)
        output_paths["t_tfce_fwe05_thresholded"] = thresholded_path
        _write_map_plot(
            thresholded_path,
            group_dir / f"{test_name}_t_tfce_fwe05_thresholded.png",
            f"{test_name}: TFCE FWE p<0.05",
            threshold=0.0,
        )
    if np.any(sig_voxel):
        thresholded_t = np.where(sig_voxel, full_t, 0.0).astype(np.float32)
        thresholded_path = group_dir / f"{test_name}_t_voxel_fwe05_thresholded.nii.gz"
        _save_flat_img(thresholded_t, thresholded_path, shape, affine)
        output_paths["t_voxel_fwe05_thresholded"] = thresholded_path
        _write_map_plot(
            thresholded_path,
            group_dir / f"{test_name}_t_voxel_fwe05_thresholded.png",
            f"{test_name}: voxel FWE p<0.05",
            threshold=0.0,
        )
    _write_map_plot(
        output_paths["t"],
        group_dir / f"{test_name}_t_uncorrected_display.png",
        f"{test_name}: t map",
        threshold=3.0,
    )

    peak_idx = int(np.nanargmax(np.abs(full_t[finite_mask])))
    finite_indices = np.flatnonzero(finite_mask)
    peak_flat = int(finite_indices[peak_idx])
    peak_ijk = np.array(np.unravel_index(peak_flat, shape))
    peak_xyz = nib.affines.apply_affine(affine, peak_ijk)
    summary = {
        "test": test_name,
        "n_subjects": int(len(subjects)),
        "subjects": ";".join(subjects),
        "finite_voxels": int(finite_mask.sum()),
        "n_perm": int(n_perm),
        "tfce_used": bool(use_tfce),
        "max_abs_t": float(np.nanmax(np.abs(full_t[finite_mask]))),
        "max_neglog10p_tfce_fwe": float(np.nanmax(full_logp_tfce[finite_mask])) if use_tfce else float("nan"),
        "min_tfce_fwe_p": float(10.0 ** (-np.nanmax(full_logp_tfce[finite_mask]))) if use_tfce else float("nan"),
        "max_neglog10p_voxel_fwe": float(np.nanmax(full_logp_voxel[finite_mask])),
        "min_voxel_fwe_p": float(10.0 ** (-np.nanmax(full_logp_voxel[finite_mask]))),
        "tfce_fwe05_voxels": int(sig_tfce.sum()),
        "voxel_fwe05_voxels": int(sig_voxel.sum()),
        "peak_t": float(full_t[peak_flat]),
        "peak_mean_effect": float(full_mean[peak_flat]),
        "peak_mni_x": float(peak_xyz[0]),
        "peak_mni_y": float(peak_xyz[1]),
        "peak_mni_z": float(peak_xyz[2]),
        "mean_path": str(output_paths["mean"]),
        "t_path": str(output_paths["t"]),
        "neglog10p_tfce_fwe_path": str(output_paths["neglog10p_tfce_fwe"]) if use_tfce else "",
        "neglog10p_voxel_fwe_path": str(output_paths["neglog10p_voxel_fwe"]),
    }
    cluster_rows = _uncorrected_cluster_rows(
        test_name,
        full_t,
        full_mean,
        finite_mask,
        shape,
        affine,
        len(subjects),
    )
    pd.DataFrame(cluster_rows).to_csv(group_dir / f"{test_name}_uncorrected_p001_clusters.csv", index=False)
    return summary, cluster_rows


def main() -> int:
    args = _prepare_args(_build_parser().parse_args())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    weight_img = nib.load(str(args.weight_map))
    shape = tuple(int(value) for value in weight_img.shape[:3])
    affine = np.asarray(weight_img.affine)
    weight_values = np.asarray(weight_img.get_fdata(dtype=np.float32))
    network_mask_flat, network_weight_flat, network_metadata = _network_weight_mask(
        weight_values,
        float(args.network_weight_percentile),
    )
    network_mask_img = nib.Nifti1Image(network_mask_flat.reshape(shape).astype(np.uint8), affine)
    nib.save(network_mask_img, str(out_dir / "weighted_network_mask_p90.nii.gz"))
    _save_flat_img(
        np.where(network_mask_flat, network_weight_flat, NAN_FILL).astype(np.float32),
        out_dir / "weighted_network_weights_p90.nii.gz",
        shape,
        affine,
    )
    run_specs: list[Any] = []
    if bool(args.reuse_existing):
        subject_contrasts, subject_manifest = _load_maps_from_manifest(out_dir / "subject_contrast_manifest.csv")
        run_manifest = (
            pd.read_csv(out_dir / "run_contrast_manifest.csv")
            if (out_dir / "run_contrast_manifest.csv").exists()
            else pd.DataFrame()
        )
    else:
        gvs_order = _load_gvs_order(args.gvs_order)
        run_specs = _discover_run_specs(args.beta_root, gvs_order, args.subjects)
        subject_contrasts, run_manifest = _build_subject_contrasts(
            run_specs,
            network_mask_flat,
            network_weight_flat,
            trials_per_condition=int(args.trials_per_condition),
        )
        run_manifest.to_csv(out_dir / "run_contrast_manifest.csv", index=False)
        subject_manifest = _write_subject_contrast_images(subject_contrasts, out_dir, shape, affine)

    primary_arrays = [
        array
        for (subject, session, contrast), array in subject_contrasts.items()
        if contrast == "active_mean"
    ]
    mask_flat = network_mask_flat.copy()
    mask_flat &= _common_finite_mask(primary_arrays)
    mask_img = nib.Nifti1Image(mask_flat.reshape(shape).astype(np.uint8), affine)
    nib.save(mask_img, str(out_dir / "weighted_network_common_mask.nii.gz"))

    tests = list(PRIMARY_GROUP_TESTS)
    if bool(args.include_exploratory):
        tests.extend(EXPLORATORY_GROUP_TESTS)
    group_rows: list[dict[str, Any]] = []
    cluster_rows: list[dict[str, Any]] = []
    for offset, (test_name, map_kind, contrast, group) in enumerate(tests):
        maps = subject_contrasts
        subjects, arrays = _arrays_for_group_test(maps, contrast, group)
        if len(arrays) < 3:
            continue
        summary, clusters = _run_group_test(
            test_name,
            arrays,
            subjects,
            mask_flat,
            mask_img,
            out_dir,
            shape,
            affine,
            n_perm=int(args.n_perm),
            n_jobs=int(args.n_jobs),
            random_state=int(args.random_state) + offset,
            use_tfce=not bool(args.no_tfce),
        )
        summary["map_kind"] = map_kind
        summary["contrast"] = contrast
        summary["group"] = group
        group_rows.append(summary)
        cluster_rows.extend(clusters[:20])

    group_summary = pd.DataFrame(group_rows)
    cluster_summary = pd.DataFrame(cluster_rows)
    group_summary.to_csv(out_dir / "group_results_summary.csv", index=False)
    cluster_summary.to_csv(out_dir / "uncorrected_p001_cluster_summary_top20.csv", index=False)

    manifest = {
        "method": {
            "weighted_network": (
                "Only positive p90 voxel-weight network voxels were analyzed. Run-level "
                "condition means were computed from cleaned beta volumes inside this network. "
                "For each run, active GVS-minus-same-run-sham contrasts were multiplied by "
                "the voxel weights, then both runs were averaged within subject/session. "
                "Subject/session weighted-network maps were tested with one-sample two-sided "
                "permutation/TFCE inference."
            ),
        },
        "inputs": {
            "beta_root": str(args.beta_root),
            "gvs_order": str(args.gvs_order),
            "weight_map": str(args.weight_map),
            "network_html_reference": str(args.network_html),
        },
        "parameters": {
            "trials_per_condition": int(args.trials_per_condition),
            "network_weight_percentile": float(args.network_weight_percentile),
            "network_weight_threshold": float(network_metadata["network_weight_threshold"]),
            "n_positive_weight_voxels": int(network_metadata["n_positive_weight_voxels"]),
            "n_network_voxels": int(network_metadata["n_network_voxels"]),
            "n_perm": int(args.n_perm),
            "n_jobs": int(args.n_jobs),
            "random_state": int(args.random_state),
            "tfce": not bool(args.no_tfce),
            "fwe_alpha": FWE_ALPHA,
            "uncorrected_cluster_p": UNCORRECTED_CLUSTER_P,
            "include_exploratory": bool(args.include_exploratory),
        },
        "n_runs": int(len(run_specs)),
        "reuse_existing": bool(args.reuse_existing),
        "n_subject_session_contrasts": int(subject_manifest.shape[0]),
        "mask_voxels": int(mask_flat.sum()),
        "outputs": {
            "out_dir": str(out_dir),
            "run_contrast_manifest": str(out_dir / "run_contrast_manifest.csv"),
            "subject_contrast_manifest": str(out_dir / "subject_contrast_manifest.csv"),
            "weighted_network_mask": str(out_dir / "weighted_network_mask_p90.nii.gz"),
            "weighted_network_weights": str(out_dir / "weighted_network_weights_p90.nii.gz"),
            "weighted_network_common_mask": str(out_dir / "weighted_network_common_mask.nii.gz"),
            "group_results_summary": str(out_dir / "group_results_summary.csv"),
            "uncorrected_cluster_summary": str(out_dir / "uncorrected_p001_cluster_summary_top20.csv"),
        },
    }
    (out_dir / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved {out_dir / 'group_results_summary.csv'}")
    print(f"Saved {out_dir / 'uncorrected_p001_cluster_summary_top20.csv'}")
    print(f"Saved {out_dir / 'analysis_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
