#!/usr/bin/env python3
"""Trial-to-trial variability test for selected optimization voxels.

This reproduces the Figure S3-style hypothesis check: selected voxels should
show lower normalized consecutive-trial beta variability than size-matched
samples from non-selected motor-area voxels.
"""

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
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import datasets, image
from scipy import stats


ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHT_MAP = ROOT / "data" / "voxel_weights_task1_bold0.6_beta0.6_smooth1.25_gamma1.5.nii.gz"
DEFAULT_BETA_ROOT = Path(
    "/mnt/TeamShare/Data_Masterfile/H20-00572_All-Dressed/"
    "Zahra-Thesis-Data/fmri_opt_group/results_beta_preprocessed"
)
DEFAULT_GROUP_CONCAT_DIR = Path(
    "/mnt/TeamShare/Data_Masterfile/H20-00572_All-Dressed/"
    "Zahra-Thesis-Data/fmri_opt_group/data/group_concat"
)
DEFAULT_OUT_DIR = ROOT / "figures" / "prove_hypothesis"
DEFAULT_SELECTED_PERCENTILE = 90.0
DEFAULT_NUM_RESAMPLES = 1000
DEFAULT_RANDOM_SEED = 13
DEFAULT_BATCH_SIZE = 2048
DEFAULT_MIN_ABS_VOXEL_MEAN = 1e-12
DEFAULT_SESSION_STATES = "1:off,2:on"
DEFAULT_CEREBELLUM_ATLAS = Path(
    "/usr/local/fsl/data/atlases/Cerebellum/Cerebellum-MNIfnirt-maxprob-thr25-2mm.nii.gz"
)
BETA_FILE_RE = re.compile(r"cleaned_beta_volume_(sub-[^_]+)_ses-(\d+)_run-(\d+)\.npy$")

CORTICAL_MOTOR_LABELS = (
    "Precentral Gyrus",
    "Postcentral Gyrus",
    "Frontal Medial Cortex",
    "Juxtapositional Lobule Cortex (formerly Supplementary Motor Cortex)",
)
SUBCORTICAL_MOTOR_LABELS = (
    "Left Thalamus",
    "Left Putamen",
    "Left Pallidum",
    "Right Thalamus",
    "Right Putamen",
    "Right Pallidum",
)
MOTOR_LABEL_PATTERNS = (
    "precentral gyrus",
    "juxtapositional lobule cortex",
    "supplementary motor",
    "precentral",
    "postcentral gyrus",
    "frontal medial cortex",
    "paracentral lobule",
    "thalamus",
    "caudate nucleus",
    "putamen",
    "globus pallidus",
    "pallidum",
    "cerebellum",
)


@dataclass(frozen=True)
class SegmentNorm:
    start: int
    stop: int
    mean: float
    scale: float


@dataclass(frozen=True)
class MetricUnit:
    label: str
    segment_indices: tuple[int, ...]


@dataclass(frozen=True)
class RunBetaFile:
    subject: str
    session: int
    run: int
    path: Path


def _json_ready(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _resolve_group_concat_dir(beta_root: Path, group_concat_dir: Path | None) -> Path:
    candidates = []
    if group_concat_dir is not None:
        candidates.append(group_concat_dir)
    candidates.extend(
        [
            beta_root / "group_concat",
            beta_root.parent / "data" / "group_concat",
            DEFAULT_GROUP_CONCAT_DIR,
        ]
    )
    for candidate in candidates:
        if candidate and (candidate / "cleaned_beta_volume_group.npy").exists():
            return candidate
    tried = "\n".join(f"- {candidate}" for candidate in candidates if candidate)
    raise FileNotFoundError(f"Could not find group-concat beta files. Tried:\n{tried}")


def _resample_label_img(label_img: nib.Nifti1Image, reference_img: nib.Nifti1Image) -> np.ndarray:
    if label_img.shape[:3] == reference_img.shape[:3] and np.allclose(label_img.affine, reference_img.affine):
        return np.rint(label_img.get_fdata()).astype(np.int32, copy=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        resampled = image.resample_to_img(
            label_img,
            reference_img,
            interpolation="nearest",
            force_resample=True,
            copy_header=True,
        )
    return np.rint(resampled.get_fdata()).astype(np.int32, copy=False)


def _add_harvard_oxford_labels(
    mask: np.ndarray,
    reference_img: nib.Nifti1Image,
    atlas_name: str,
    labels: tuple[str, ...],
    region_names: list[str],
    region_counts: list[int],
) -> None:
    atlas = datasets.fetch_atlas_harvard_oxford(atlas_name, verbose=0)
    atlas_img = atlas.maps if isinstance(atlas.maps, nib.Nifti1Image) else nib.load(atlas.maps)
    atlas_data = _resample_label_img(atlas_img, reference_img)
    for label in labels:
        label_index = atlas.labels.index(label)
        region_mask = atlas_data == label_index
        mask |= region_mask
        prefix = "Cortical" if atlas_name.startswith("cort") else "Subcortical"
        region_names.append(f"{prefix}: {label}")
        region_counts.append(int(np.count_nonzero(region_mask)))


def _build_motor_mask(reference_img: nib.Nifti1Image, cerebellum_atlas: Path) -> tuple[np.ndarray, dict[str, object]]:
    if not cerebellum_atlas.exists():
        raise FileNotFoundError(f"Missing FSL cerebellum atlas: {cerebellum_atlas}")

    motor_mask = np.zeros(reference_img.shape[:3], dtype=bool)
    region_names: list[str] = []
    region_counts: list[int] = []

    _add_harvard_oxford_labels(
        motor_mask,
        reference_img,
        "cort-maxprob-thr25-2mm",
        CORTICAL_MOTOR_LABELS,
        region_names,
        region_counts,
    )
    _add_harvard_oxford_labels(
        motor_mask,
        reference_img,
        "sub-maxprob-thr25-2mm",
        SUBCORTICAL_MOTOR_LABELS,
        region_names,
        region_counts,
    )

    cere_img = nib.load(str(cerebellum_atlas))
    cere_data = _resample_label_img(cere_img, reference_img)
    cere_mask = cere_data > 0
    motor_mask |= cere_mask
    region_names.append("Cerebellum (FSL maxprob)")
    region_counts.append(int(np.count_nonzero(cere_mask)))

    metadata = {
        "motor_mask_source": "harvard_oxford_thr25_plus_fsl_cerebellum_mnifnirt_thr25",
        "motor_label_patterns": MOTOR_LABEL_PATTERNS,
        "motor_region_names": region_names,
        "motor_region_counts": region_counts,
        "cerebellum_atlas": cerebellum_atlas,
    }
    return motor_mask, metadata


def _flat_indices_from_mask(mask: np.ndarray) -> np.ndarray:
    return np.flatnonzero(mask.ravel()).astype(np.int64, copy=False)


def _load_selected_flat_indices(path: Path, shape: tuple[int, int, int]) -> np.ndarray:
    if path.suffix == ".npz":
        loaded = np.load(path, allow_pickle=True)
        for key in ("flat_indices", "selected_flat_indices", "indices"):
            if key in loaded.files:
                values = loaded[key]
                break
        else:
            raise ValueError(f"No usable index array found in {path}; expected flat_indices or indices.")
    elif path.suffix == ".npy":
        values = np.load(path, allow_pickle=True)
    elif path.suffix == ".csv":
        frame = pd.read_csv(path)
        if "flat_index" in frame.columns:
            return frame["flat_index"].to_numpy(dtype=np.int64)
        coord_columns = [col for col in ("i", "j", "k") if col in frame.columns]
        if len(coord_columns) != 3:
            coord_columns = [col for col in ("x", "y", "z") if col in frame.columns]
        if len(coord_columns) != 3:
            raise ValueError(f"CSV selected-index file needs flat_index or i,j,k columns: {path}")
        values = frame[coord_columns].to_numpy(dtype=np.int64)
    else:
        raise ValueError(f"Unsupported selected-index file: {path}")

    values = np.asarray(values)
    if values.ndim == 1:
        return values.astype(np.int64, copy=False).ravel()
    if values.ndim == 2 and values.shape[1] == 3:
        return np.ravel_multi_index(values.T, shape).astype(np.int64, copy=False)
    if values.ndim == 2 and values.shape[0] == 3:
        return np.ravel_multi_index(values, shape).astype(np.int64, copy=False)
    raise ValueError(f"Unsupported selected-index array shape in {path}: {values.shape}")


def _selected_from_weight_map(
    weights: np.ndarray,
    weight_map: Path,
    percentile: float,
    selected_indices: Path | None,
) -> tuple[np.ndarray, dict[str, object]]:
    if selected_indices is not None:
        selected_flat = np.unique(_load_selected_flat_indices(selected_indices, weights.shape))
        metadata = {
            "selected_source": selected_indices,
            "selected_source_type": selected_indices.suffix.lstrip("."),
            "selected_definition": "explicit_selected_indices",
            "selected_threshold_value": None,
        }
        return selected_flat, metadata

    finite_nonzero = np.isfinite(weights) & (weights != 0)
    finite_values = weights[finite_nonzero]
    if finite_values.size == 0:
        raise ValueError("Weight map has no finite nonzero voxels.")
    threshold = float(np.percentile(finite_values, percentile))
    selected_mask = finite_nonzero & (weights >= threshold)
    selected_flat = _flat_indices_from_mask(selected_mask)
    metadata = {
        "selected_source": weight_map,
        "selected_source_type": "nifti_weight_percentile",
        "selected_definition": f"weights >= p{percentile:g} over finite nonzero weights",
        "selected_threshold_percentile": float(percentile),
        "selected_threshold_value": threshold,
        "selected_nonzero_weight_count": int(finite_values.size),
    }
    return selected_flat, metadata


def _load_group_concat(group_concat_dir: Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict[str, Path]]:
    beta_path = group_concat_dir / "cleaned_beta_volume_group.npy"
    active_flat_path = group_concat_dir / "active_flat_indices__group.npy"
    manifest_path = group_concat_dir / "concat_manifest_group.tsv"
    if not beta_path.exists() or not active_flat_path.exists() or not manifest_path.exists():
        raise FileNotFoundError(f"Missing group-concat inputs under {group_concat_dir}")

    beta = np.load(beta_path, mmap_mode="r")
    active_flat = np.asarray(np.load(active_flat_path, mmap_mode="r"), dtype=np.int64).ravel()
    manifest = pd.read_csv(manifest_path, sep="\t").sort_values("offset_start").reset_index(drop=True)
    needed = {"offset_start", "offset_end", "sub_tag", "ses", "run"}
    missing = needed - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")
    if beta.shape[0] != active_flat.size:
        raise ValueError(f"Beta rows ({beta.shape[0]}) do not match active flat indices ({active_flat.size}).")
    if int(manifest["offset_end"].max()) > beta.shape[1]:
        raise ValueError("Manifest offset_end exceeds group beta trial count.")
    paths = {"beta_path": beta_path, "active_flat_path": active_flat_path, "manifest_path": manifest_path}
    return beta, active_flat, manifest, paths


def _map_flat_to_active_rows(target_flat: np.ndarray, active_flat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    target_flat = np.asarray(target_flat, dtype=np.int64).ravel()
    order = np.argsort(active_flat)
    sorted_active = active_flat[order]
    positions = np.searchsorted(sorted_active, target_flat)
    valid = (positions < sorted_active.size) & (sorted_active[np.minimum(positions, sorted_active.size - 1)] == target_flat)
    rows = np.full(target_flat.shape, -1, dtype=np.int64)
    rows[valid] = order[positions[valid]]
    return valid, rows


def _iter_slices(n_items: int, batch_size: int):
    for start in range(0, n_items, batch_size):
        yield slice(start, min(start + batch_size, n_items))


def _compute_segment_norms(
    beta: np.ndarray,
    row_indices: np.ndarray,
    manifest: pd.DataFrame,
    batch_size: int,
    pre_normalize: bool,
) -> list[SegmentNorm]:
    segment_bounds = [
        (int(row.offset_start), int(row.offset_end))
        for row in manifest.itertuples(index=False)
    ]
    if not pre_normalize:
        return [SegmentNorm(start, stop, 0.0, 1.0) for start, stop in segment_bounds]

    means = []
    for segment_number, (start, stop) in enumerate(segment_bounds, start=1):
        finite_sum = 0.0
        finite_count = 0
        for rows_slice in _iter_slices(row_indices.size, batch_size):
            chunk = np.asarray(beta[row_indices[rows_slice], start:stop], dtype=np.float64)
            finite = np.isfinite(chunk)
            finite_sum += float(np.sum(np.where(finite, chunk, 0.0)))
            finite_count += int(np.count_nonzero(finite))
        mean_value = finite_sum / finite_count if finite_count else 0.0
        means.append(mean_value)
        if segment_number % 10 == 0 or segment_number == len(segment_bounds):
            print(f"Computed normalization means for {segment_number}/{len(segment_bounds)} manifest segments.", flush=True)

    norms: list[SegmentNorm] = []
    for segment_number, ((start, stop), mean_value) in enumerate(zip(segment_bounds, means), start=1):
        max_abs = 0.0
        for rows_slice in _iter_slices(row_indices.size, batch_size):
            chunk = np.asarray(beta[row_indices[rows_slice], start:stop], dtype=np.float64)
            finite = np.isfinite(chunk)
            if np.any(finite):
                centered = np.where(finite, chunk - mean_value, 0.0)
                max_abs = max(max_abs, float(np.max(np.abs(centered))))
        norms.append(SegmentNorm(start, stop, mean_value, max_abs if max_abs > 0 else 1.0))
        if segment_number % 10 == 0 or segment_number == len(segment_bounds):
            print(f"Computed normalization scales for {segment_number}/{len(segment_bounds)} manifest segments.", flush=True)
    return norms


def _build_metric_units(manifest: pd.DataFrame, unit: str) -> list[MetricUnit]:
    units: list[MetricUnit] = []
    if unit == "run":
        for idx, row in enumerate(manifest.itertuples(index=False)):
            label = f"{row.sub_tag}_ses-{int(row.ses)}_run-{int(row.run)}"
            units.append(MetricUnit(label, (idx,)))
        return units
    if unit != "subject_session":
        raise ValueError(f"Unknown metric unit: {unit}")

    grouped = manifest.reset_index().groupby(["sub_tag", "ses"], sort=True)
    for (sub_tag, ses), frame in grouped:
        label = f"{sub_tag}_ses-{int(ses)}"
        units.append(MetricUnit(label, tuple(int(idx) for idx in frame["index"].to_numpy())))
    return units


def _compute_voxel_norm_diff_scores(
    beta: np.ndarray,
    row_indices: np.ndarray,
    manifest: pd.DataFrame,
    batch_size: int,
    metric_unit: str,
    pre_normalize: bool,
    min_abs_voxel_mean: float,
) -> tuple[np.ndarray, np.ndarray, list[SegmentNorm], list[MetricUnit]]:
    n_voxels = row_indices.size
    segment_norms = _compute_segment_norms(beta, row_indices, manifest, batch_size, pre_normalize)
    units = _build_metric_units(manifest, metric_unit)
    score_sum = np.zeros(n_voxels, dtype=np.float64)
    score_count = np.zeros(n_voxels, dtype=np.int16)

    for unit_number, unit in enumerate(units, start=1):
        value_sum = np.zeros(n_voxels, dtype=np.float64)
        value_count = np.zeros(n_voxels, dtype=np.int32)
        diff_sum = np.zeros(n_voxels, dtype=np.float64)
        diff_count = np.zeros(n_voxels, dtype=np.int32)

        for segment_index in unit.segment_indices:
            norm = segment_norms[segment_index]
            if norm.stop - norm.start < 2:
                continue
            for rows_slice in _iter_slices(n_voxels, batch_size):
                chunk = np.asarray(beta[row_indices[rows_slice], norm.start:norm.stop], dtype=np.float64)
                chunk = (chunk - norm.mean) / norm.scale
                finite = np.isfinite(chunk)

                value_sum[rows_slice] += np.sum(np.where(finite, chunk, 0.0), axis=1)
                value_count[rows_slice] += np.count_nonzero(finite, axis=1)

                left = chunk[:, :-1]
                right = chunk[:, 1:]
                keep = np.isfinite(left) & np.isfinite(right)
                diff_sum[rows_slice] += np.sum(np.where(keep, np.abs(right - left), 0.0), axis=1)
                diff_count[rows_slice] += np.count_nonzero(keep, axis=1)

        valid = (value_count > 0) & (diff_count > 0)
        voxel_mean = np.divide(value_sum, value_count, out=np.full(n_voxels, np.nan), where=value_count > 0)
        mean_abs_diff = np.divide(diff_sum, diff_count, out=np.full(n_voxels, np.nan), where=diff_count > 0)
        denom = np.abs(voxel_mean)
        valid &= np.isfinite(mean_abs_diff) & np.isfinite(denom) & (denom > min_abs_voxel_mean)
        unit_scores = np.divide(mean_abs_diff, denom, out=np.full(n_voxels, np.nan), where=valid)
        score_sum[valid] += unit_scores[valid]
        score_count[valid] += 1

        if unit_number % 5 == 0 or unit_number == len(units):
            print(f"Computed normalized consecutive-diff scores for {unit_number}/{len(units)} {metric_unit} units.", flush=True)

    scores = np.divide(score_sum, score_count, out=np.full(n_voxels, np.nan), where=score_count > 0)
    return scores, score_count, segment_norms, units


def _resample_means(values: np.ndarray, sample_size: int, num_resamples: int, seed: int) -> tuple[np.ndarray, bool]:
    rng = np.random.default_rng(seed)
    replace = values.size < sample_size
    means = np.empty(num_resamples, dtype=np.float64)
    for idx in range(num_resamples):
        sampled = rng.choice(values, size=sample_size, replace=replace)
        means[idx] = float(np.mean(sampled))
    return means, replace


def _prevalence_ratios(
    selected: np.ndarray,
    nonselected: np.ndarray,
    percentiles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    pooled = np.concatenate([selected, nonselected])
    thresholds = np.percentile(pooled, percentiles)
    ratios = np.full(thresholds.shape, np.nan, dtype=np.float64)
    for idx, threshold in enumerate(thresholds):
        selected_fraction = float(np.mean(selected <= threshold))
        nonselected_fraction = float(np.mean(nonselected <= threshold))
        if nonselected_fraction > 0:
            ratios[idx] = selected_fraction / nonselected_fraction
    return thresholds, ratios


def _style_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.8)
    ax.tick_params(labelsize=13)


def _plot_norm_diff_figure(
    output_png: Path,
    percentiles: np.ndarray,
    ratios: np.ndarray,
    selected_scores: np.ndarray,
    resampled_means: np.ndarray,
) -> Path:
    selected_color = "#d55e00"
    resample_color = "#56b4e9"
    reference_color = "#666666"
    ci_color = "#bdbdbd"

    selected_mean = float(np.mean(selected_scores))
    resample_mean = float(np.mean(resampled_means))
    ci_low, ci_high = np.percentile(resampled_means, [2.5, 97.5])

    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"],
            "axes.labelsize": 13,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    ):
        fig, axes = plt.subplots(1, 2, figsize=(11.35, 4.3))

        axes[0].plot(
            percentiles,
            ratios,
            color=selected_color,
            marker="o",
            markersize=5.8,
            linewidth=2.2,
        )
        axes[0].axhline(
            1.0,
            color=reference_color,
            linestyle="--",
            linewidth=1.4,
            label="Equal fractions",
        )
        axes[0].set_xticks(percentiles)
        axes[0].set_xticklabels([f"{int(p)}%" for p in percentiles])
        axes[0].set_xlabel("Variability percentile threshold")
        axes[0].set_ylabel("Vigour/control fraction ratio")
        finite_ratios = ratios[np.isfinite(ratios)]
        if finite_ratios.size:
            ratio_min = min(1.0, float(np.min(finite_ratios)))
            ratio_max = max(1.0, float(np.max(finite_ratios)))
            ratio_pad = max(0.02, (ratio_max - ratio_min) * 0.18)
            axes[0].set_ylim(max(0.0, ratio_min - ratio_pad), ratio_max + ratio_pad)
        else:
            axes[0].set_ylim(0.95, 1.1)
        axes[0].set_xlim(5.0, 95.0)
        axes[0].legend(frameon=False, fontsize=10, loc="lower left")
        _style_axis(axes[0])

        bins = min(40, max(16, int(np.sqrt(resampled_means.size))))
        axes[1].axvspan(
            ci_low,
            ci_high,
            color=ci_color,
            alpha=0.24,
            linewidth=0,
            label=f"95% CI [{ci_low:.2f}, {ci_high:.2f}]",
        )
        axes[1].hist(
            resampled_means,
            bins=bins,
            density=True,
            color=resample_color,
            alpha=0.58,
            edgecolor="white",
            linewidth=0.4,
        )
        axes[1].axvline(
            selected_mean,
            color=selected_color,
            linestyle="--",
            linewidth=1.8,
            label=f"Vigour network mean = {selected_mean:.2f}",
        )
        axes[1].axvline(
            resample_mean,
            color=reference_color,
            linestyle="--",
            linewidth=1.5,
            label=f"Control network mean = {resample_mean:.2f}",
        )
        axes[1].set_xlabel("Consecutive trial variability")
        axes[1].set_ylabel("Density")
        axes[1].set_xlim(
            min(selected_mean, float(np.min(resampled_means))) - 0.07,
            max(ci_high, float(np.max(resampled_means))) + 0.07,
        )
        _style_axis(axes[1])

        axes[1].legend(
            frameon=True,
            facecolor="white",
            edgecolor="none",
            framealpha=0.78,
            fontsize=9,
            loc="upper right",
            borderaxespad=0.35,
            handlelength=2.4,
        )

        fig.tight_layout(w_pad=2.0)
        output_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_png, dpi=320, bbox_inches="tight", pad_inches=0.04)
        output_pdf = output_png.with_suffix(".pdf")
        fig.savefig(output_pdf, bbox_inches="tight", pad_inches=0.04)
        plt.close(fig)
    return output_pdf


def _parse_session_states(value: str) -> dict[int, str]:
    states: dict[int, str] = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid session-state mapping item {item!r}; expected session:state.")
        session_text, state = item.split(":", 1)
        state = state.strip().lower()
        if not state:
            raise ValueError(f"Missing state label in session-state mapping item {item!r}.")
        states[int(session_text.strip())] = state
    if not states:
        raise ValueError("At least one session-state mapping is required.")
    return states


def _discover_run_beta_files(beta_root: Path) -> list[RunBetaFile]:
    specs: list[RunBetaFile] = []
    pattern = "sub-*/cleaned_beta_volume_sub-*_ses-*_run-*.npy"
    for path in sorted(beta_root.glob(pattern)):
        match = BETA_FILE_RE.match(path.name)
        if not match:
            continue
        specs.append(
            RunBetaFile(
                subject=match.group(1),
                session=int(match.group(2)),
                run=int(match.group(3)),
                path=path,
            )
        )
    if not specs:
        raise FileNotFoundError(f"No per-run cleaned beta files found under {beta_root}")
    return specs


def _new_metric_accumulator(n_voxels: int) -> dict[str, np.ndarray]:
    return {
        "value_sum": np.zeros(n_voxels, dtype=np.float64),
        "value_sumsq": np.zeros(n_voxels, dtype=np.float64),
        "value_count": np.zeros(n_voxels, dtype=np.int32),
        "diff_sum": np.zeros(n_voxels, dtype=np.float64),
        "diff_count": np.zeros(n_voxels, dtype=np.int32),
    }


def _run_normalization(flat_view: np.ndarray, flat_indices: np.ndarray, batch_size: int) -> tuple[float, float]:
    finite_sum = 0.0
    finite_count = 0
    for rows_slice in _iter_slices(flat_indices.size, batch_size):
        chunk = np.asarray(flat_view[flat_indices[rows_slice]], dtype=np.float64)
        finite = np.isfinite(chunk)
        finite_sum += float(np.sum(np.where(finite, chunk, 0.0)))
        finite_count += int(np.count_nonzero(finite))
    mean_value = finite_sum / finite_count if finite_count else 0.0

    max_abs = 0.0
    for rows_slice in _iter_slices(flat_indices.size, batch_size):
        chunk = np.asarray(flat_view[flat_indices[rows_slice]], dtype=np.float64)
        finite = np.isfinite(chunk)
        if np.any(finite):
            centered = np.where(finite, chunk - mean_value, 0.0)
            max_abs = max(max_abs, float(np.max(np.abs(centered))))
    return mean_value, max_abs if max_abs > 0 else 1.0


def _accumulate_run_beta_metrics(
    accumulator: dict[str, np.ndarray],
    flat_view: np.ndarray,
    flat_indices: np.ndarray,
    batch_size: int,
    norm_mean: float,
    norm_scale: float,
) -> None:
    for rows_slice in _iter_slices(flat_indices.size, batch_size):
        chunk = np.asarray(flat_view[flat_indices[rows_slice]], dtype=np.float64)
        chunk = (chunk - norm_mean) / norm_scale
        finite = np.isfinite(chunk)
        filled = np.where(finite, chunk, 0.0)
        accumulator["value_sum"][rows_slice] += np.sum(filled, axis=1)
        accumulator["value_sumsq"][rows_slice] += np.sum(filled * filled, axis=1)
        accumulator["value_count"][rows_slice] += np.count_nonzero(finite, axis=1)

        if chunk.shape[1] < 2:
            continue
        left = chunk[:, :-1]
        right = chunk[:, 1:]
        keep = np.isfinite(left) & np.isfinite(right)
        accumulator["diff_sum"][rows_slice] += np.sum(np.where(keep, np.abs(right - left), 0.0), axis=1)
        accumulator["diff_count"][rows_slice] += np.count_nonzero(keep, axis=1)


def _accumulator_metric_means(
    accumulator: dict[str, np.ndarray],
    min_abs_voxel_mean: float,
) -> tuple[float, float, int, int]:
    value_sum = accumulator["value_sum"]
    value_count = accumulator["value_count"]
    with np.errstate(invalid="ignore", divide="ignore"):
        voxel_mean = np.divide(value_sum, value_count, out=np.full(value_sum.shape, np.nan), where=value_count > 0)
        voxel_second_moment = np.divide(
            accumulator["value_sumsq"],
            value_count,
            out=np.full(value_sum.shape, np.nan),
            where=value_count > 0,
        )
        variance = np.maximum(voxel_second_moment - voxel_mean * voxel_mean, 0.0)
        voxel_std = np.sqrt(variance)
        mean_abs_diff = np.divide(
            accumulator["diff_sum"],
            accumulator["diff_count"],
            out=np.full(value_sum.shape, np.nan),
            where=accumulator["diff_count"] > 0,
        )
        denom = np.abs(voxel_mean)
        cv_scores = voxel_std / denom
        norm_diff_scores = mean_abs_diff / denom

    cv_valid = np.isfinite(cv_scores) & np.isfinite(denom) & (denom > min_abs_voxel_mean) & (value_count > 1)
    norm_valid = (
        np.isfinite(norm_diff_scores)
        & np.isfinite(denom)
        & (denom > min_abs_voxel_mean)
        & (accumulator["diff_count"] > 0)
    )
    cv_mean = float(np.mean(cv_scores[cv_valid])) if np.any(cv_valid) else float("nan")
    norm_diff_mean = float(np.mean(norm_diff_scores[norm_valid])) if np.any(norm_valid) else float("nan")
    return cv_mean, norm_diff_mean, int(np.count_nonzero(cv_valid)), int(np.count_nonzero(norm_valid))


def _compute_subject_session_selection_metrics(
    run_specs: list[RunBetaFile],
    selected_flat: np.ndarray,
    control_flat: np.ndarray,
    reference_shape: tuple[int, int, int],
    batch_size: int,
    pre_normalize: bool,
    min_abs_voxel_mean: float,
) -> pd.DataFrame:
    specs_by_session: dict[tuple[str, int], list[RunBetaFile]] = {}
    for spec in run_specs:
        specs_by_session.setdefault((spec.subject, spec.session), []).append(spec)

    target_flat = np.unique(np.concatenate([selected_flat, control_flat]))
    rows: list[dict[str, object]] = []
    total_sessions = len(specs_by_session)
    for session_number, ((subject, session), specs) in enumerate(sorted(specs_by_session.items()), start=1):
        specs = sorted(specs, key=lambda item: item.run)
        selected_acc = _new_metric_accumulator(selected_flat.size)
        control_acc = _new_metric_accumulator(control_flat.size)

        for spec in specs:
            beta = np.load(spec.path, mmap_mode="r")
            if beta.ndim != 4:
                raise ValueError(f"Expected 4D beta volume for {spec.path}, got shape {beta.shape}.")
            if beta.shape[:3] != reference_shape:
                raise ValueError(
                    f"Beta volume shape mismatch for {spec.path}: {beta.shape[:3]} vs {reference_shape}."
                )
            flat_view = beta.reshape(-1, beta.shape[-1])
            norm_mean, norm_scale = _run_normalization(flat_view, target_flat, batch_size) if pre_normalize else (0.0, 1.0)
            _accumulate_run_beta_metrics(selected_acc, flat_view, selected_flat, batch_size, norm_mean, norm_scale)
            _accumulate_run_beta_metrics(control_acc, flat_view, control_flat, batch_size, norm_mean, norm_scale)

        selected_cv, selected_norm, selected_cv_n, selected_norm_n = _accumulator_metric_means(
            selected_acc,
            min_abs_voxel_mean,
        )
        control_cv, control_norm, control_cv_n, control_norm_n = _accumulator_metric_means(
            control_acc,
            min_abs_voxel_mean,
        )
        rows.append(
            {
                "subject": subject,
                "session": int(session),
                "n_runs": int(len(specs)),
                "selected_cv_mean": selected_cv,
                "control_cv_mean": control_cv,
                "selected_norm_diff_mean": selected_norm,
                "control_norm_diff_mean": control_norm,
                "selected_cv_valid_voxels": selected_cv_n,
                "control_cv_valid_voxels": control_cv_n,
                "selected_norm_diff_valid_voxels": selected_norm_n,
                "control_norm_diff_valid_voxels": control_norm_n,
            }
        )
        if session_number % 4 == 0 or session_number == total_sessions:
            print(
                f"Computed state-selection metrics for {session_number}/{total_sessions} subject/session entries.",
                flush=True,
            )

    return pd.DataFrame(rows).sort_values(["subject", "session"]).reset_index(drop=True)


def _sem(values: pd.Series) -> float:
    arr = values.to_numpy(dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return 0.0 if arr.size <= 1 else float(arr.std(ddof=1) / np.sqrt(arr.size))


def _safe_float(value) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def _fmt_signed(value: float | None, digits: int = 3) -> str:
    return "nan" if value is None or not np.isfinite(float(value)) else f"{float(value):+.{digits}f}"


def _fmt_float(value: float | None, digits: int = 3) -> str:
    return "nan" if value is None or not np.isfinite(float(value)) else f"{float(value):.{digits}f}"


def _paired_difference_stats(differences: np.ndarray) -> dict[str, float]:
    differences = np.asarray(differences, dtype=np.float64)
    differences = differences[np.isfinite(differences)]
    if differences.size < 2:
        return {
            "t": float("nan"),
            "p_two_sided": float("nan"),
            "p_one_sided_less": float("nan"),
            "wilcoxon_stat": float("nan"),
            "wilcoxon_p_two_sided": float("nan"),
        }
    t_result = stats.ttest_1samp(differences, 0.0)
    t_stat = float(t_result.statistic)
    p_two = float(t_result.pvalue)
    p_less = p_two / 2.0 if t_stat < 0 else 1.0 - p_two / 2.0
    try:
        wilcoxon_result = stats.wilcoxon(differences, alternative="two-sided")
        wilcoxon_stat = float(wilcoxon_result.statistic)
        wilcoxon_p = float(wilcoxon_result.pvalue)
    except ValueError:
        wilcoxon_stat = float("nan")
        wilcoxon_p = float("nan")
    return {
        "t": t_stat,
        "p_two_sided": p_two,
        "p_one_sided_less": float(p_less),
        "wilcoxon_stat": wilcoxon_stat,
        "wilcoxon_p_two_sided": wilcoxon_p,
    }


def _paired_delta_summary(session_df: pd.DataFrame, metric: str, metric_label: str) -> dict[str, object]:
    selected_col = f"selected_{metric}"
    control_col = f"control_{metric}"
    needed = ["subject", "state", selected_col, control_col]
    frame = session_df.loc[:, needed].copy()
    frame = frame[frame["state"].isin(["off", "on"])]
    pivot = frame.pivot_table(index="subject", columns="state", values=[selected_col, control_col], aggfunc="mean")
    required = [
        (selected_col, "off"),
        (selected_col, "on"),
        (control_col, "off"),
        (control_col, "on"),
    ]
    if pivot.empty or any(column not in pivot.columns for column in required):
        selected_off = selected_on = control_off = control_on = np.array([], dtype=np.float64)
    else:
        paired = pivot.dropna(subset=required)
        selected_off = paired[(selected_col, "off")].to_numpy(dtype=np.float64)
        selected_on = paired[(selected_col, "on")].to_numpy(dtype=np.float64)
        control_off = paired[(control_col, "off")].to_numpy(dtype=np.float64)
        control_on = paired[(control_col, "on")].to_numpy(dtype=np.float64)

    d_sel = selected_on - selected_off
    d_ctl = control_on - control_off
    interaction = d_sel - d_ctl
    selected_stats = _paired_difference_stats(d_sel)
    control_stats = _paired_difference_stats(d_ctl)
    interaction_stats = _paired_difference_stats(interaction)
    supports_on_stability = bool(
        np.isfinite(np.mean(d_sel))
        and np.mean(d_sel) < 0
        and np.isfinite(selected_stats["p_one_sided_less"])
        and selected_stats["p_one_sided_less"] < 0.05
    )
    return {
        "n_subjects_paired": int(d_sel.size),
        "selected_off_mean": float(np.mean(selected_off)) if selected_off.size else float("nan"),
        "selected_on_mean": float(np.mean(selected_on)) if selected_on.size else float("nan"),
        "control_off_mean": float(np.mean(control_off)) if control_off.size else float("nan"),
        "control_on_mean": float(np.mean(control_on)) if control_on.size else float("nan"),
        "D_sel_mean_on_minus_off": float(np.mean(d_sel)) if d_sel.size else float("nan"),
        "D_ctl_mean_on_minus_off": float(np.mean(d_ctl)) if d_ctl.size else float("nan"),
        "interaction_mean_D_sel_minus_D_ctl": float(np.mean(interaction)) if interaction.size else float("nan"),
        "selected_state_t": selected_stats["t"],
        "selected_state_p_two_sided": selected_stats["p_two_sided"],
        "selected_state_p_one_sided_less": selected_stats["p_one_sided_less"],
        "control_state_t": control_stats["t"],
        "control_state_p_two_sided": control_stats["p_two_sided"],
        "interaction_t": interaction_stats["t"],
        "interaction_p_two_sided": interaction_stats["p_two_sided"],
        "interaction_p_one_sided_less": interaction_stats["p_one_sided_less"],
        "wilcoxon_selected_state_stat": selected_stats["wilcoxon_stat"],
        "wilcoxon_selected_state_p_two_sided": selected_stats["wilcoxon_p_two_sided"],
        "wilcoxon_control_state_stat": control_stats["wilcoxon_stat"],
        "wilcoxon_control_state_p_two_sided": control_stats["wilcoxon_p_two_sided"],
        "wilcoxon_interaction_stat": interaction_stats["wilcoxon_stat"],
        "wilcoxon_interaction_p_two_sided": interaction_stats["wilcoxon_p_two_sided"],
        "claim_supports_on_preferential_stability": supports_on_stability,
        "metric": metric,
        "metric_label": metric_label,
    }


def _plot_state_selection_stability_figure(
    output_png: Path,
    session_df: pd.DataFrame,
    paired_df: pd.DataFrame,
) -> Path:
    paired_summary = paired_df.loc[paired_df["metric"].eq("norm_diff_mean")].iloc[0].to_dict()
    long_df = pd.concat(
        [
            session_df.loc[:, ["subject", "session", "state", "n_runs", "selected_norm_diff_mean"]]
            .rename(columns={"selected_norm_diff_mean": "value"})
            .assign(selection="selected"),
            session_df.loc[:, ["subject", "session", "state", "n_runs", "control_norm_diff_mean"]]
            .rename(columns={"control_norm_diff_mean": "value"})
            .assign(selection="control"),
        ],
        ignore_index=True,
    )
    long_df = long_df.loc[np.isfinite(long_df["value"].to_numpy(dtype=np.float64))].copy()

    fig, axes = plt.subplots(1, 2, figsize=(7.8, 4.3), sharey=True)
    colors_map = {"control": "#3d6f8e", "selected": "#d65a6f"}
    labels = {"control": "Motor reference", "selected": "Target network"}
    state_order = ["off", "on"]
    x_positions = np.array([0.0, 1.0], dtype=np.float64)
    panel_stats = {
        "control": {"delta": paired_summary["D_ctl_mean_on_minus_off"], "p": paired_summary["control_state_p_two_sided"]},
        "selected": {"delta": paired_summary["D_sel_mean_on_minus_off"], "p": paired_summary["selected_state_p_two_sided"]},
    }

    values = long_df["value"].to_numpy(dtype=np.float64)
    values = values[np.isfinite(values)]
    y_limits = None
    if values.size:
        y_min, y_max = float(values.min()), float(values.max())
        pad = max(0.2, 0.06 * (y_max - y_min if y_max > y_min else 1.0))
        y_limits = (y_min - pad, y_max + pad)

    for ax, selection in zip(axes, ("control", "selected")):
        subset = long_df.loc[long_df["selection"].eq(selection)].copy()
        for _, subject_df in subset.groupby("subject", sort=True):
            subject_df = subject_df.set_index("state").reindex(state_order)
            ys = subject_df["value"].to_numpy(dtype=np.float64)
            finite = np.isfinite(ys)
            if np.any(finite):
                ax.plot(x_positions[finite], ys[finite], color="0.82", alpha=0.65, linewidth=0.9, zorder=1)
                ax.scatter(
                    x_positions[finite],
                    ys[finite],
                    s=22,
                    facecolor="white",
                    edgecolor=colors_map[selection],
                    linewidth=0.9,
                    alpha=0.9,
                    zorder=2,
                )
        means = subset.groupby("state")["value"].mean().reindex(state_order)
        sems = subset.groupby("state")["value"].apply(_sem).reindex(state_order).fillna(0.0)
        ax.errorbar(
            x_positions,
            means.to_numpy(dtype=np.float64),
            yerr=sems.to_numpy(dtype=np.float64),
            color=colors_map[selection],
            linewidth=2.6,
            marker="o",
            markersize=6.5,
            markerfacecolor=colors_map[selection],
            markeredgecolor="white",
            markeredgewidth=0.9,
            capsize=3.5,
            elinewidth=1.5,
            zorder=4,
        )
        ax.set_xticks(x_positions)
        ax.set_xticklabels(["OFF", "ON"])
        ax.set_title(labels[selection], fontsize=12, pad=8)
        ax.set_xlabel("Medication state")
        ax.grid(axis="y", color="0.9", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if y_limits is not None:
            ax.set_ylim(*y_limits)
        ax.text(
            0.04,
            0.96,
            f"ON - OFF = {_fmt_signed(panel_stats[selection]['delta'])}\n"
            f"paired p = {_fmt_float(_safe_float(panel_stats[selection]['p']))}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.8", alpha=0.95),
        )

    axes[0].set_ylabel("Mean normalized |consecutive trial beta difference|")
    fig.tight_layout(rect=(0.04, 0.14, 1.0, 1.0))
    interaction_text = (
        "Difference in ON - OFF change\n"
        f"{labels['selected'].lower()} - {labels['control'].lower()} = "
        f"{_fmt_signed(paired_summary['interaction_mean_D_sel_minus_D_ctl'])}; "
        f"paired p = {_fmt_float(_safe_float(paired_summary['interaction_p_two_sided']))}"
    )
    fig.text(0.5, 0.025, interaction_text, ha="center", va="bottom", fontsize=9.0, color="0.25", linespacing=1.25)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_pdf = output_png.with_suffix(".pdf")
    fig.savefig(output_png, dpi=400, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(output_pdf, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return output_pdf


def _run_state_selection_stability(args: argparse.Namespace) -> dict[str, object]:
    weight_img = nib.load(str(args.weight_map))
    weights = np.asarray(weight_img.get_fdata(dtype=np.float32))
    selected_flat, selected_metadata = _selected_from_weight_map(
        weights,
        args.weight_map,
        args.selected_percentile,
        args.selected_indices,
    )
    motor_mask, motor_metadata = _build_motor_mask(weight_img, args.cerebellum_atlas)
    motor_flat = _flat_indices_from_mask(motor_mask)
    selected_set = set(int(value) for value in selected_flat)
    control_flat = np.asarray([idx for idx in motor_flat if int(idx) not in selected_set], dtype=np.int64)
    run_specs = _discover_run_beta_files(args.beta_root)
    session_states = _parse_session_states(args.session_states)

    print(f"Per-run beta files found: {len(run_specs):,}", flush=True)
    print(f"Selected voxels total: {selected_flat.size:,}", flush=True)
    print(f"Motor reference voxels total: {control_flat.size:,}", flush=True)

    session_df = _compute_subject_session_selection_metrics(
        run_specs=run_specs,
        selected_flat=selected_flat,
        control_flat=control_flat,
        reference_shape=weight_img.shape[:3],
        batch_size=args.batch_size,
        pre_normalize=not args.no_pre_normalize,
        min_abs_voxel_mean=args.min_abs_voxel_mean,
    )
    session_df.insert(
        session_df.columns.get_loc("n_runs"),
        "state",
        session_df["session"].map(session_states).fillna(session_df["session"].map(lambda value: f"ses-{value}")),
    )

    paired_rows = [
        _paired_delta_summary(session_df, "cv_mean", "Coefficient of Variation (std/|mean|)"),
        _paired_delta_summary(session_df, "norm_diff_mean", "Normalized |Delta| (consecutive diff / |mean|)"),
    ]
    paired_df = pd.DataFrame(paired_rows)

    out_dir = args.out_dir / "state_selection_stability_followup"
    out_dir.mkdir(parents=True, exist_ok=True)
    session_path = out_dir / "subject_session_state_selection_means.csv"
    paired_path = out_dir / "paired_delta_summary.csv"
    figure_png = out_dir / "norm_diff_mean_subject_state_lines(main).png"
    session_df.to_csv(session_path, index=False)
    paired_df.to_csv(paired_path, index=False)
    figure_pdf = _plot_state_selection_stability_figure(figure_png, session_df, paired_df)

    norm_summary = paired_df.loc[paired_df["metric"].eq("norm_diff_mean")].iloc[0].to_dict()
    summary = {
        **selected_metadata,
        **motor_metadata,
        "weight_map": args.weight_map,
        "beta_root": args.beta_root,
        "analysis": "state_selection_stability_followup",
        "metric": "subject_session_mean_abs_consecutive_trial_diff_divided_by_abs_voxel_mean",
        "pre_normalize_each_run": not args.no_pre_normalize,
        "pre_normalization_mode": (
            "demean_then_divide_by_maxabs_per_run_over_selected_and_motor_reference_voxels"
            if not args.no_pre_normalize
            else "none"
        ),
        "session_states": session_states,
        "run_file_count": int(len(run_specs)),
        "subject_session_count_total": int(session_df.shape[0]),
        "paired_subject_count": int(norm_summary["n_subjects_paired"]),
        "motor_pool_size": int(motor_flat.size),
        "selected_in_motor_count": int(np.intersect1d(selected_flat, motor_flat).size),
        "selected_count_total": int(selected_flat.size),
        "control_count_total": int(control_flat.size),
        "selected_off_mean_norm_diff": norm_summary["selected_off_mean"],
        "selected_on_mean_norm_diff": norm_summary["selected_on_mean"],
        "control_off_mean_norm_diff": norm_summary["control_off_mean"],
        "control_on_mean_norm_diff": norm_summary["control_on_mean"],
        "selected_on_minus_off_norm_diff": norm_summary["D_sel_mean_on_minus_off"],
        "control_on_minus_off_norm_diff": norm_summary["D_ctl_mean_on_minus_off"],
        "interaction_selected_minus_control_norm_diff": norm_summary["interaction_mean_D_sel_minus_D_ctl"],
        "interaction_p_two_sided_norm_diff": norm_summary["interaction_p_two_sided"],
        "session_csv_path": session_path,
        "paired_delta_csv_path": paired_path,
        "norm_diff_mean_subject_state_lines_png_path": figure_png,
        "norm_diff_mean_subject_state_lines_pdf_path": figure_pdf,
        "batch_size": int(args.batch_size),
        "min_abs_voxel_mean": float(args.min_abs_voxel_mean),
    }
    summary_path = out_dir / "state_selection_stability_summary.json"
    summary_path.write_text(json.dumps(_json_ready(summary), indent=2), encoding="utf-8")
    summary["summary_path"] = summary_path

    print(
        "Norm-diff OFF/ON means: "
        f"selected {norm_summary['selected_off_mean']:.4f}/{norm_summary['selected_on_mean']:.4f}, "
        f"control {norm_summary['control_off_mean']:.4f}/{norm_summary['control_on_mean']:.4f}",
        flush=True,
    )
    print(
        "Interaction selected-control ON-OFF: "
        f"{norm_summary['interaction_mean_D_sel_minus_D_ctl']:.4f}, "
        f"p={norm_summary['interaction_p_two_sided']:.4g}",
        flush=True,
    )
    print(f"Wrote figure: {figure_png}", flush=True)
    print(f"Wrote summary: {summary_path}", flush=True)
    return summary


def _run_analysis(args: argparse.Namespace) -> dict[str, object]:
    weight_img = nib.load(str(args.weight_map))
    weights = np.asarray(weight_img.get_fdata(dtype=np.float32))
    selected_flat, selected_metadata = _selected_from_weight_map(
        weights,
        args.weight_map,
        args.selected_percentile,
        args.selected_indices,
    )
    motor_mask, motor_metadata = _build_motor_mask(weight_img, args.cerebellum_atlas)
    motor_flat = _flat_indices_from_mask(motor_mask)
    selected_set = set(int(value) for value in selected_flat)
    nonselected_motor_flat = np.asarray([idx for idx in motor_flat if int(idx) not in selected_set], dtype=np.int64)

    group_concat_dir = _resolve_group_concat_dir(args.beta_root, args.group_concat_dir)
    beta, active_flat, manifest, input_paths = _load_group_concat(group_concat_dir)

    target_flat = np.unique(np.concatenate([selected_flat, nonselected_motor_flat]))
    active_valid, active_rows = _map_flat_to_active_rows(target_flat, active_flat)
    valid_target_flat = target_flat[active_valid]
    valid_active_rows = active_rows[active_valid]
    if valid_target_flat.size == 0:
        raise RuntimeError("None of the selected or motor-pool voxels are present in the group beta matrix.")

    print(f"Selected voxels total: {selected_flat.size:,}", flush=True)
    print(f"Motor-area pool total: {motor_flat.size:,}", flush=True)
    print(f"Non-selected motor voxels total: {nonselected_motor_flat.size:,}", flush=True)
    print(f"Voxels available in group beta matrix for metric: {valid_target_flat.size:,}", flush=True)

    scores, counts, segment_norms, units = _compute_voxel_norm_diff_scores(
        beta=beta,
        row_indices=valid_active_rows,
        manifest=manifest,
        batch_size=args.batch_size,
        metric_unit=args.metric_unit,
        pre_normalize=not args.no_pre_normalize,
        min_abs_voxel_mean=args.min_abs_voxel_mean,
    )

    selected_valid_mask = np.isin(valid_target_flat, selected_flat) & np.isfinite(scores)
    nonselected_valid_mask = np.isin(valid_target_flat, nonselected_motor_flat) & np.isfinite(scores)
    selected_scores = scores[selected_valid_mask]
    nonselected_scores = scores[nonselected_valid_mask]
    selected_counts = counts[selected_valid_mask]
    nonselected_counts = counts[nonselected_valid_mask]
    selected_valid_flat = valid_target_flat[selected_valid_mask]
    nonselected_valid_flat = valid_target_flat[nonselected_valid_mask]

    if selected_scores.size == 0 or nonselected_scores.size == 0:
        raise RuntimeError("Need at least one valid selected voxel and one valid non-selected motor voxel.")

    percentiles = np.arange(10.0, 100.0, 10.0)
    thresholds, ratios = _prevalence_ratios(selected_scores, nonselected_scores, percentiles)
    resampled_means, replacement = _resample_means(
        nonselected_scores,
        selected_scores.size,
        args.num_resamples,
        args.random_seed,
    )

    selected_mean = float(np.mean(selected_scores))
    nonselected_mean = float(np.mean(nonselected_scores))
    resample_mean = float(np.mean(resampled_means))
    resample_std = float(np.std(resampled_means, ddof=1))
    ci_low, ci_high = np.percentile(resampled_means, [2.5, 97.5])
    empirical_p = float((np.count_nonzero(resampled_means <= selected_mean) + 1) / (resampled_means.size + 1))

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    figure_png = out_dir / "trial_variability_hypothesis_norm_diff_cd(main).png"
    figure_pdf = _plot_norm_diff_figure(
        figure_png,
        percentiles,
        ratios,
        selected_scores,
        resampled_means,
    )

    npz_path = out_dir / "trial_variability_hypothesis_analysis_data.npz"
    np.savez_compressed(
        npz_path,
        motor_flat_indices=motor_flat,
        motor_region_names=np.asarray(motor_metadata["motor_region_names"], dtype=object),
        motor_region_counts=np.asarray(motor_metadata["motor_region_counts"], dtype=np.int32),
        motor_region_patterns=np.asarray(MOTOR_LABEL_PATTERNS, dtype=object),
        selected_norm_diff=selected_scores.astype(np.float32),
        nonselected_norm_diff=nonselected_scores.astype(np.float32),
        selected_subject_count_norm_diff=selected_counts.astype(np.int16),
        nonselected_subject_count_norm_diff=nonselected_counts.astype(np.int16),
        selected_flat_indices_norm_diff=selected_valid_flat.astype(np.int64),
        nonselected_flat_indices_norm_diff=nonselected_valid_flat.astype(np.int64),
        norm_diff_prevalence_percentiles=percentiles.astype(np.float32),
        norm_diff_prevalence_thresholds=thresholds.astype(np.float32),
        norm_diff_prevalence_ratios=ratios.astype(np.float32),
        resampled_nonselected_norm_diff_means=resampled_means.astype(np.float32),
    )

    summary = {
        **selected_metadata,
        **motor_metadata,
        "weight_map": args.weight_map,
        "beta_root": args.beta_root,
        "group_concat_dir": group_concat_dir,
        "beta_path": input_paths["beta_path"],
        "active_flat_path": input_paths["active_flat_path"],
        "manifest_path": input_paths["manifest_path"],
        "metric": "mean_abs_consecutive_trial_diff_divided_by_abs_voxel_mean",
        "aggregation_mode": f"{args.metric_unit}_voxelwise_nanmean",
        "pre_normalize_each_manifest_segment": not args.no_pre_normalize,
        "pre_normalization_mode": (
            "demean_then_divide_by_maxabs_per_manifest_file_over_target_voxels_and_kept_trials"
            if not args.no_pre_normalize
            else "none"
        ),
        "min_abs_voxel_mean": float(args.min_abs_voxel_mean),
        "manifest_segment_count": int(manifest.shape[0]),
        "metric_unit_count_total": int(len(units)),
        "metric_unit_labels": [unit.label for unit in units],
        "motor_pool_size": int(motor_flat.size),
        "selected_in_motor_count": int(np.intersect1d(selected_flat, motor_flat).size),
        "selected_count_total": int(selected_flat.size),
        "nonselected_count_total": int(nonselected_motor_flat.size),
        "selected_count_norm_diff_valid": int(selected_scores.size),
        "nonselected_count_norm_diff_valid": int(nonselected_scores.size),
        "selected_mean_norm_diff": selected_mean,
        "nonselected_mean_norm_diff": nonselected_mean,
        "selected_median_norm_diff": float(np.median(selected_scores)),
        "nonselected_median_norm_diff": float(np.median(nonselected_scores)),
        "selected_min_units_per_voxel_norm_diff": int(np.min(selected_counts)),
        "nonselected_min_units_per_voxel_norm_diff": int(np.min(nonselected_counts)),
        "selected_max_units_per_voxel_norm_diff": int(np.max(selected_counts)),
        "nonselected_max_units_per_voxel_norm_diff": int(np.max(nonselected_counts)),
        "norm_diff_resample_mean": resample_mean,
        "norm_diff_resample_std": resample_std,
        "norm_diff_resample_ci_2p5": float(ci_low),
        "norm_diff_resample_ci_97p5": float(ci_high),
        "norm_diff_resample_p_lower_or_equal_selected": empirical_p,
        "norm_diff_resample_with_replacement": bool(replacement),
        "num_resamples": int(args.num_resamples),
        "random_seed": int(args.random_seed),
        "batch_size": int(args.batch_size),
        "analysis_npz_path": npz_path,
        "norm_diff_cd_main_png_path": figure_png,
        "norm_diff_cd_main_pdf_path": figure_pdf,
    }
    summary_path = out_dir / "trial_variability_hypothesis_summary.json"
    summary_path.write_text(json.dumps(_json_ready(summary), indent=2), encoding="utf-8")
    summary["summary_path"] = summary_path

    print(f"Selected mean normalized diff: {selected_mean:.4f}", flush=True)
    print(f"Non-selected motor mean normalized diff: {nonselected_mean:.4f}", flush=True)
    print(f"Resample mean: {resample_mean:.4f}, 95% CI [{ci_low:.4f}, {ci_high:.4f}], p={empirical_p:.4g}", flush=True)
    print(f"Wrote figure: {figure_png}", flush=True)
    print(f"Wrote summary: {summary_path}", flush=True)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run selected-voxel stability checks against motor-area reference voxels."
    )
    parser.add_argument(
        "--analysis",
        choices=("trial_variability", "state_selection", "both"),
        default="trial_variability",
        help="trial_variability reproduces the pooled Figure S3-style check; state_selection reproduces Section 3.2/Figure S5.",
    )
    parser.add_argument("--weight-map", type=Path, default=DEFAULT_WEIGHT_MAP)
    parser.add_argument("--beta-root", type=Path, default=DEFAULT_BETA_ROOT)
    parser.add_argument("--group-concat-dir", type=Path, default=None)
    parser.add_argument("--selected-indices", type=Path, default=None)
    parser.add_argument("--selected-percentile", type=float, default=DEFAULT_SELECTED_PERCENTILE)
    parser.add_argument("--cerebellum-atlas", type=Path, default=DEFAULT_CEREBELLUM_ATLAS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--metric-unit", choices=("subject_session", "run"), default="subject_session")
    parser.add_argument("--no-pre-normalize", action="store_true")
    parser.add_argument("--min-abs-voxel-mean", type=float, default=DEFAULT_MIN_ABS_VOXEL_MEAN)
    parser.add_argument("--num-resamples", type=int, default=DEFAULT_NUM_RESAMPLES)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--session-states",
        default=DEFAULT_SESSION_STATES,
        help="Comma-separated medication-state mapping for state_selection, e.g. 1:off,2:on.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.analysis in ("trial_variability", "both"):
        _run_analysis(args)
    if args.analysis in ("state_selection", "both"):
        _run_state_selection_stability(args)


if __name__ == "__main__":
    main()
