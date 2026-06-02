#!/usr/bin/env python3
"""Whole-brain GVS contrast and searchlight analyses.

This script avoids ROI partitioning. It builds subject-level GVS-minus-sham
contrast maps from cleaned beta volumes, runs second-level permutation/TFCE
tests, and computes a run-split searchlight pattern-similarity analysis.
"""

from __future__ import annotations

import argparse
import json
import math
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
ACTIVE_CONDITION_CODES = tuple(f"gvs-{index:02d}" for index in range(2, 10))
SHAM_CONDITION_CODE = "gvs-01"
CONTRAST_SPECS = {
    "active_mean": ACTIVE_CONDITION_CODES,
    "gvs2_dc_plus_1": ("gvs-03",),
    "gvs5_theta": ("gvs-06",),
    "gvs6_alpha": ("gvs-07",),
}
PRIMARY_GROUP_TESTS = (
    ("wholebrain_active_mean_OFF", "wholebrain", "active_mean", "OFF"),
    ("wholebrain_active_mean_ON", "wholebrain", "active_mean", "ON"),
    ("wholebrain_active_mean_ON_minus_OFF", "wholebrain", "active_mean", "ON_minus_OFF"),
    ("searchlight_active_mean_OFF", "searchlight", "active_mean", "OFF"),
    ("searchlight_active_mean_ON", "searchlight", "active_mean", "ON"),
    ("searchlight_active_mean_ON_minus_OFF", "searchlight", "active_mean", "ON_minus_OFF"),
)
EXPLORATORY_GROUP_TESTS = (
    ("wholebrain_gvs2_dc_plus_1_OFF", "wholebrain", "gvs2_dc_plus_1", "OFF"),
    ("wholebrain_gvs5_theta_OFF", "wholebrain", "gvs5_theta", "OFF"),
    ("wholebrain_gvs6_alpha_OFF", "wholebrain", "gvs6_alpha", "OFF"),
)
FWE_ALPHA = 0.05
UNCORRECTED_CLUSTER_P = 0.001
NAN_FILL = 0.0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beta-root", type=Path, default=DEFAULT_BETA_ROOT)
    parser.add_argument("--gvs-order", type=Path, default=DEFAULT_GVS_ORDER)
    parser.add_argument("--weight-map", type=Path, default=DEFAULT_WEIGHT_MAP)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--trials-per-condition", type=int, default=DEFAULT_TRIALS_PER_CONDITION)
    parser.add_argument("--n-perm", type=int, default=2000)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--random-state", type=int, default=1301)
    parser.add_argument("--no-tfce", action="store_true")
    parser.add_argument("--searchlight-radius-mm", type=float, default=6.0)
    parser.add_argument("--searchlight-min-voxels", type=int, default=20)
    parser.add_argument("--skip-searchlight", action="store_true")
    parser.add_argument("--include-exploratory", action="store_true")
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--subjects", nargs="+", default=None)
    return parser


def _prepare_args(args: argparse.Namespace) -> argparse.Namespace:
    args.beta_root = _resolve_path(args.beta_root)
    args.gvs_order = _resolve_path(args.gvs_order)
    args.weight_map = _resolve_path(args.weight_map)
    args.out_dir = _resolve_path(args.out_dir)
    if int(args.trials_per_condition) <= 0:
        raise ValueError("--trials-per-condition must be positive.")
    if int(args.n_perm) <= 0:
        raise ValueError("--n-perm must be positive.")
    if float(args.searchlight_radius_mm) <= 0:
        raise ValueError("--searchlight-radius-mm must be positive.")
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


def _condition_means_for_run(
    beta_path: Path,
    stim_order: tuple[int, ...],
    *,
    trials_per_condition: int,
) -> dict[str, np.ndarray]:
    beta = np.load(beta_path, mmap_mode="r")
    if beta.ndim != 4:
        raise RuntimeError(f"{beta_path} must be 4D, got shape {beta.shape}.")
    flat = beta.reshape(-1, int(beta.shape[-1]))
    means: dict[str, np.ndarray] = {}
    for condition_code, start, stop in _iter_condition_slices(
        int(beta.shape[-1]),
        stim_order,
        int(trials_per_condition),
    ):
        with np.errstate(invalid="ignore", divide="ignore"):
            means[condition_code] = np.nanmean(flat[:, int(start) : int(stop)], axis=1).astype(np.float32)
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
    *,
    trials_per_condition: int,
) -> tuple[dict[tuple[str, int, str], np.ndarray], dict[tuple[str, int, int], np.ndarray], pd.DataFrame]:
    run_contrast_parts: dict[tuple[str, int, str], list[np.ndarray]] = defaultdict(list)
    run_active_delta: dict[tuple[str, int, int], np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    for index, spec in enumerate(run_specs, start=1):
        print(
            f"Computing run contrasts {index}/{len(run_specs)}: "
            f"{spec.subject} ses-{spec.session} run-{spec.run}",
            flush=True,
        )
        means = _condition_means_for_run(
            spec.beta_path,
            spec.stim_order,
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
            delta = (active_mean - sham).astype(np.float32)
            run_contrast_parts[(spec.subject, int(spec.session), contrast_name)].append(delta)
            if contrast_name == "active_mean":
                run_active_delta[(spec.subject, int(spec.session), int(spec.run))] = delta
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
                    "source_beta_path": str(spec.beta_path),
                }
            )

    subject_contrasts = {
        key: _mean_available(parts)
        for key, parts in run_contrast_parts.items()
        if parts
    }
    return subject_contrasts, run_active_delta, pd.DataFrame(rows)


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


def _searchlight_kernel(radius_mm: float, affine: np.ndarray) -> np.ndarray:
    voxel_sizes = np.sqrt(np.sum(np.asarray(affine[:3, :3], dtype=float) ** 2, axis=0))
    radius_voxels = np.maximum(1, np.ceil(float(radius_mm) / voxel_sizes).astype(int))
    grid = np.meshgrid(
        *[np.arange(-radius, radius + 1) for radius in radius_voxels],
        indexing="ij",
    )
    distance_mm = np.zeros_like(grid[0], dtype=float)
    for axis, coords in enumerate(grid):
        distance_mm += (coords * voxel_sizes[axis]) ** 2
    kernel = distance_mm <= float(radius_mm) ** 2
    return kernel.astype(np.float32)


def _local_pattern_similarity(
    first: np.ndarray,
    second: np.ndarray,
    mask_flat: np.ndarray,
    shape: tuple[int, int, int],
    kernel: np.ndarray,
    min_voxels: int,
) -> np.ndarray:
    first_img = first.reshape(shape)
    second_img = second.reshape(shape)
    valid_img = (mask_flat & np.isfinite(first) & np.isfinite(second)).reshape(shape)
    first_zero = np.where(valid_img, first_img, 0.0).astype(np.float32)
    second_zero = np.where(valid_img, second_img, 0.0).astype(np.float32)
    valid_float = valid_img.astype(np.float32)

    numerator = ndimage.convolve(first_zero * second_zero, kernel, mode="constant", cval=0.0)
    norm_first = ndimage.convolve(first_zero * first_zero, kernel, mode="constant", cval=0.0)
    norm_second = ndimage.convolve(second_zero * second_zero, kernel, mode="constant", cval=0.0)
    counts = ndimage.convolve(valid_float, kernel, mode="constant", cval=0.0)
    denom = np.sqrt(norm_first * norm_second)
    score = np.full(shape, np.nan, dtype=np.float32)
    ok = (counts >= int(min_voxels)) & (denom > 0.0) & valid_img
    score[ok] = numerator[ok] / denom[ok]
    score = np.clip(score, -0.999999, 0.999999)
    z_score = np.full(shape, np.nan, dtype=np.float32)
    z_score[ok] = np.arctanh(score[ok]).astype(np.float32)
    return z_score.ravel()


def _compute_searchlight_maps(
    run_active_delta: dict[tuple[str, int, int], np.ndarray],
    subject_contrasts: dict[tuple[str, int, str], np.ndarray],
    mask_flat: np.ndarray,
    out_dir: Path,
    shape: tuple[int, int, int],
    affine: np.ndarray,
    *,
    radius_mm: float,
    min_voxels: int,
) -> tuple[dict[tuple[str, int, str], np.ndarray], pd.DataFrame]:
    kernel = _searchlight_kernel(radius_mm, affine)
    searchlight_maps: dict[tuple[str, int, str], np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    session_keys = sorted(
        {(subject, session) for subject, session, _run in run_active_delta},
        key=lambda item: (_subject_sort_key(item[0]), item[1]),
    )
    for index, (subject, session) in enumerate(session_keys, start=1):
        run_keys = sorted(
            key for key in run_active_delta if key[0] == subject and key[1] == session
        )
        if len(run_keys) != 2:
            continue
        print(
            f"Computing searchlight map {index}/{len(session_keys)}: {subject} ses-{session}",
            flush=True,
        )
        first = run_active_delta[run_keys[0]]
        second = run_active_delta[run_keys[1]]
        z_map = _local_pattern_similarity(
            first,
            second,
            mask_flat,
            shape,
            kernel,
            min_voxels,
        )
        searchlight_maps[(subject, int(session), "active_mean")] = z_map
        path = out_dir / "searchlight_run_split" / f"{subject}_ses-{session}_active_mean_pattern_similarity_z.nii.gz"
        _save_flat_img(z_map, path, shape, affine)
        rows.append(
            {
                "subject": subject,
                "session": int(session),
                "medication": _medication_from_session(int(session)),
                "contrast": "active_mean",
                "path": str(path),
                "radius_mm": float(radius_mm),
                "kernel_voxels": int(kernel.sum()),
                "finite_voxels": int(np.isfinite(z_map).sum()),
                "mean_z": float(np.nanmean(z_map)),
                "std_z": float(np.nanstd(z_map)),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "searchlight_manifest.csv", index=False)
    return searchlight_maps, df


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
    run_specs: list[Any] = []
    run_active_delta: dict[tuple[str, int, int], np.ndarray] = {}
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
        subject_contrasts, run_active_delta, run_manifest = _build_subject_contrasts(
            run_specs,
            trials_per_condition=int(args.trials_per_condition),
        )
        run_manifest.to_csv(out_dir / "run_contrast_manifest.csv", index=False)
        subject_manifest = _write_subject_contrast_images(subject_contrasts, out_dir, shape, affine)

    primary_arrays = [
        array
        for (subject, session, contrast), array in subject_contrasts.items()
        if contrast == "active_mean"
    ]
    mask_flat = _common_finite_mask(primary_arrays)
    mask_img = nib.Nifti1Image(mask_flat.reshape(shape).astype(np.uint8), affine)
    nib.save(mask_img, str(out_dir / "wholebrain_common_mask.nii.gz"))

    searchlight_maps: dict[tuple[str, int, str], np.ndarray] = {}
    searchlight_manifest = pd.DataFrame()
    if not args.skip_searchlight and bool(args.reuse_existing):
        searchlight_maps, searchlight_manifest = _load_maps_from_manifest(out_dir / "searchlight_manifest.csv")
    elif not args.skip_searchlight:
        searchlight_maps, searchlight_manifest = _compute_searchlight_maps(
            run_active_delta,
            subject_contrasts,
            mask_flat,
            out_dir,
            shape,
            affine,
            radius_mm=float(args.searchlight_radius_mm),
            min_voxels=int(args.searchlight_min_voxels),
        )

    tests = list(PRIMARY_GROUP_TESTS)
    if bool(args.include_exploratory):
        tests.extend(EXPLORATORY_GROUP_TESTS)
    group_rows: list[dict[str, Any]] = []
    cluster_rows: list[dict[str, Any]] = []
    for offset, (test_name, map_kind, contrast, group) in enumerate(tests):
        if map_kind == "searchlight" and args.skip_searchlight:
            continue
        maps = subject_contrasts if map_kind == "wholebrain" else searchlight_maps
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
            "wholebrain": (
                "Run-level voxelwise condition means were computed from cleaned beta volumes. "
                "For each run, each GVS condition mean was subtracted from same-run sham. "
                "Run contrasts were averaged within subject/session, then tested with "
                "one-sample two-sided permutation/TFCE inference."
            ),
            "searchlight": (
                "For active_mean only, run 1 and run 2 GVS-minus-sham contrast patterns were "
                "compared locally with a spherical searchlight. The saved score is Fisher-z "
                "local pattern correlation, tested at group level with the same TFCE procedure."
            ),
        },
        "inputs": {
            "beta_root": str(args.beta_root),
            "gvs_order": str(args.gvs_order),
            "weight_map": str(args.weight_map),
        },
        "parameters": {
            "trials_per_condition": int(args.trials_per_condition),
            "n_perm": int(args.n_perm),
            "n_jobs": int(args.n_jobs),
            "random_state": int(args.random_state),
            "tfce": not bool(args.no_tfce),
            "searchlight_radius_mm": float(args.searchlight_radius_mm),
            "searchlight_min_voxels": int(args.searchlight_min_voxels),
            "fwe_alpha": FWE_ALPHA,
            "uncorrected_cluster_p": UNCORRECTED_CLUSTER_P,
            "include_exploratory": bool(args.include_exploratory),
        },
        "n_runs": int(len(run_specs)),
        "reuse_existing": bool(args.reuse_existing),
        "n_subject_session_contrasts": int(subject_manifest.shape[0]),
        "n_searchlight_maps": int(searchlight_manifest.shape[0]) if not searchlight_manifest.empty else 0,
        "mask_voxels": int(mask_flat.sum()),
        "outputs": {
            "out_dir": str(out_dir),
            "run_contrast_manifest": str(out_dir / "run_contrast_manifest.csv"),
            "subject_contrast_manifest": str(out_dir / "subject_contrast_manifest.csv"),
            "searchlight_manifest": str(out_dir / "searchlight_manifest.csv"),
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
