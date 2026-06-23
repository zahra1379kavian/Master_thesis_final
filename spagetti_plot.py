#!/usr/bin/env python3
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib

warnings.filterwarnings("ignore", message="Unable to import Axes3D.*")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from scipy import stats

from gvs_behaviour_effects import _rt_seconds
from projected_sig_vs_RT import _load_behaviour_rt


DEFAULT_MED_OFF = Path("data/Projection_BOLD_trials_med_off.npy")
DEFAULT_MED_ON = Path("data/Projection_BOLD_trials_med_on.npy")
DEFAULT_TASK_MAP = Path("data/Task_map_BOLD_trials.npy")
DEFAULT_MANIFEST = Path("data/concat_manifest_group.tsv")
DEFAULT_OUT_DIR = Path("figures/med_on_off_projection_features")
DEFAULT_GVS_DIR = Path("data/GVS_projection_BOLD")
DEFAULT_GVS_OUT_DIR = Path("figures/gvs_projection_features_vs_sham")
DEFAULT_GVS_SHAM = "gvs-01"
DEFAULT_GVS_RT_RUN_METRICS = Path("figures/projected_RT/revised_behaviour_from_mat/projection_behavior_run_metrics.csv")
DEFAULT_GVS_RT_INVENTORY = Path("figures/GVS_effects/common/run_condition_inventory.csv")
DEFAULT_ACTIVE_BOLD_GROUP = Path("data/active_bold_group.npy")
DEFAULT_ACTIVE_FLAT_INDICES = Path("data/active_flat_indices__group.npy")
DEFAULT_WEIGHT_MAP = Path("data/voxel_weights_task1_bold0.6_beta0.6_smooth1.25_gamma1.5.nii.gz")
DEFAULT_WEIGHT_HTML = Path("data/voxel_weights_task1_bold0.6_beta0.6_smooth1.25_gamma1.5_bold_thr90.html")

_SIGN_FLIP_CACHE: dict[int, np.ndarray] = {}

FEATURE_NAMES = [
    "mean_level",
    "auc",
    "peak_to_peak",
    "baseline_to_peak",
    "baseline_to_trough",
    "abs_baseline_response",
    "early_late_change",
    "slope",
    "temporal_sd",
]

MED_TEST_FEATURE_NAMES = [
    "early_late_change",
    "slope",
    "abs_baseline_response",
    "peak_to_peak",
]
GVS_TEST_FEATURE_NAMES = MED_TEST_FEATURE_NAMES

FEATURE_LABELS = {
    "mean_level": "Mean level",
    "auc": "Area under curve",
    "peak_to_peak": "Peak-to-peak amplitude",
    "baseline_to_peak": "Baseline to peak",
    "baseline_to_trough": "Baseline to trough",
    "abs_baseline_response": "Max abs. baseline response",
    "early_late_change": "Late minus early",
    "slope": "Linear slope",
    "temporal_sd": "Temporal SD",
}


def _load_trials(path: Path) -> np.ndarray:
    trials = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)
    if trials.ndim != 2:
        raise ValueError(f"Expected a 2D trial-by-time array in {path}, got {trials.shape}")
    if trials.shape[0] == 0 or trials.shape[1] < 2:
        raise ValueError(f"Expected at least one trial and two time points in {path}, got {trials.shape}")
    if not np.any(np.isfinite(trials)):
        raise ValueError(f"No finite values found in {path}")
    return trials


def _load_manifest(manifest_path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path, sep="\t")
    required = {"sub_tag", "ses", "run", "n_trials"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"{manifest_path} is missing required columns: {', '.join(missing)}")
    return manifest


def _trial_labels_from_manifest(manifest_path: Path, session: int, expected_rows: int) -> pd.DataFrame:
    manifest = _load_manifest(manifest_path)
    session_rows = manifest[manifest["ses"].astype(int).eq(int(session))]
    subjects: list[str] = []
    runs: list[int] = []
    trial_in_run: list[int] = []
    manifest_rows: list[int] = []

    for row_index, row in session_rows.iterrows():
        n_trials = int(row["n_trials"])
        subjects.extend([str(row["sub_tag"])] * n_trials)
        runs.extend([int(row["run"])] * n_trials)
        trial_in_run.extend(range(n_trials))
        manifest_rows.extend([int(row_index)] * n_trials)

    if len(subjects) != int(expected_rows):
        raise ValueError(
            f"Manifest session {session} describes {len(subjects)} trials, "
            f"but the corresponding array has {expected_rows} rows"
        )

    return pd.DataFrame(
        {
            "subject": subjects,
            "session": int(session),
            "run": runs,
            "trial_in_run": trial_in_run,
            "manifest_row": manifest_rows,
        }
    )


def _reduce_task_map_bold_trials(
    path: Path,
    manifest_path: Path,
    reducer: str,
    chunk_size: int,
) -> np.ndarray:
    trials = np.load(path, mmap_mode="r")
    if trials.ndim != 3:
        raise ValueError(f"Expected a 3D voxel-by-trial-by-time array in {path}, got {trials.shape}")

    manifest = _load_manifest(manifest_path)
    n_trials = int(manifest["n_trials"].sum())
    if trials.shape[1] != n_trials:
        raise ValueError(
            f"Expected {path} axis 1 to match {n_trials} manifest trials, got shape {trials.shape}"
        )
    if chunk_size <= 0:
        raise ValueError("--task-map-chunk-size must be positive")
    if reducer != "mean":
        raise ValueError(f"Unsupported task-map reducer: {reducer}")

    reduced_sum = np.zeros((trials.shape[1], trials.shape[2]), dtype=np.float64)
    reduced_count = np.zeros((trials.shape[1], trials.shape[2]), dtype=np.int64)
    for start in range(0, trials.shape[0], chunk_size):
        stop = min(start + chunk_size, trials.shape[0])
        chunk = np.asarray(trials[start:stop], dtype=np.float32)
        finite = np.isfinite(chunk)
        reduced_sum += np.sum(np.where(finite, chunk, 0.0), axis=0, dtype=np.float64)
        reduced_count += np.count_nonzero(finite, axis=0)

    reduced = np.divide(
        reduced_sum,
        reduced_count,
        out=np.full_like(reduced_sum, np.nan, dtype=np.float64),
        where=reduced_count > 0,
    )
    if not np.any(np.isfinite(reduced)):
        raise ValueError(f"No finite reduced values found in {path}")
    return reduced


def _align_mask_to_affine(mask: np.ndarray, source_affine: np.ndarray, target_affine: np.ndarray) -> np.ndarray:
    aligned = mask.copy()
    for axis in range(3):
        source_step = float(source_affine[axis, axis])
        target_step = float(target_affine[axis, axis])
        if source_step != 0.0 and target_step != 0.0 and np.sign(source_step) != np.sign(target_step):
            aligned = np.flip(aligned, axis=axis)
    return aligned


def _load_html_selected_mask(html_path: Path, reference_img: nib.Nifti1Image) -> np.ndarray:
    from plot_weight_multiplane_colorbar import load_html_display

    _, selected_mask, html_affine = load_html_display(html_path)
    if selected_mask.shape != reference_img.shape[:3]:
        raise ValueError(
            f"{html_path} selected mask shape {selected_mask.shape} does not match "
            f"weight map shape {reference_img.shape[:3]}"
        )
    return _align_mask_to_affine(selected_mask, html_affine, reference_img.affine)


def _weighted_html_projection_from_active_bold(
    active_bold_path: Path,
    active_flat_indices_path: Path,
    weight_map: Path,
    html_mask: Path,
    chunk_size: int,
) -> tuple[np.ndarray, dict[str, object]]:
    if chunk_size <= 0:
        raise ValueError("--projection-voxel-chunk-size must be positive")

    active_bold = np.load(active_bold_path, mmap_mode="r")
    if active_bold.ndim != 3:
        raise ValueError(f"Expected a 3D voxel-by-trial-by-time array in {active_bold_path}, got {active_bold.shape}")

    active_flat = np.asarray(np.load(active_flat_indices_path, allow_pickle=False), dtype=np.int64).ravel()
    if active_flat.size != active_bold.shape[0]:
        raise ValueError(
            f"Active index count mismatch: {active_flat_indices_path} has {active_flat.size} rows, "
            f"but {active_bold_path} has {active_bold.shape[0]} voxels"
        )

    weight_img = nib.load(str(weight_map))
    weights = weight_img.get_fdata(dtype=np.float32)
    if active_flat.size and (int(np.nanmin(active_flat)) < 0 or int(np.nanmax(active_flat)) >= weights.size):
        raise ValueError(f"{active_flat_indices_path} contains indices outside {weight_map}")

    html_selected = _load_html_selected_mask(html_mask, weight_img).ravel()
    flat_weights = weights.ravel()[active_flat].astype(np.float64, copy=False)
    keep = html_selected[active_flat] & np.isfinite(flat_weights) & (flat_weights != 0)
    selected_rows = np.flatnonzero(keep)
    if selected_rows.size == 0:
        raise ValueError(f"No active BOLD voxels overlap nonzero weights in {html_mask}")

    selected_weights = flat_weights[selected_rows]
    n_trials, trial_length = active_bold.shape[1], active_bold.shape[2]
    n_timepoints = n_trials * trial_length
    projection = np.zeros(n_timepoints, dtype=np.float64)
    any_finite = np.zeros(n_timepoints, dtype=bool)

    for start in range(0, selected_rows.size, chunk_size):
        stop = min(start + chunk_size, selected_rows.size)
        rows = selected_rows[start:stop]
        chunk = np.asarray(active_bold[rows], dtype=np.float64).reshape(rows.size, -1)
        finite = np.isfinite(chunk)
        finite_counts = np.count_nonzero(finite, axis=1, keepdims=True)
        sums = np.sum(np.where(finite, chunk, 0.0), axis=1, keepdims=True)
        means = np.divide(sums, finite_counts, out=np.zeros_like(sums), where=finite_counts > 0)
        centered = np.where(finite, chunk - means, 0.0)
        projection += selected_weights[start:stop] @ centered
        any_finite |= np.any(finite, axis=0)

    projection[~any_finite] = np.nan
    metadata = {
        "active_bold": str(active_bold_path),
        "active_flat_indices": str(active_flat_indices_path),
        "weight_map": str(weight_map),
        "html_mask": str(html_mask),
        "html_selected_voxels": int(np.count_nonzero(html_selected)),
        "selected_active_voxels": int(selected_rows.size),
        "missed_html_voxels_not_in_active_bold": int(np.count_nonzero(html_selected) - selected_rows.size),
        "chunk_size": int(chunk_size),
    }
    return projection.reshape(n_trials, trial_length), metadata


def _session_trials_from_full_trials(
    full_trials: np.ndarray,
    manifest_path: Path,
    session: int,
) -> np.ndarray:
    manifest = _load_manifest(manifest_path)
    required = {"offset_start", "offset_end"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"{manifest_path} is missing required columns: {', '.join(missing)}")

    session_rows = manifest[manifest["ses"].astype(int).eq(int(session))]
    chunks = [
        full_trials[int(row["offset_start"]) : int(row["offset_end"])]
        for _, row in session_rows.iterrows()
    ]
    if not chunks:
        raise ValueError(f"No manifest rows found for session {session}")
    return np.vstack(chunks)


def _linear_slope(trials: np.ndarray) -> np.ndarray:
    time = np.arange(trials.shape[1], dtype=np.float64)
    centered_time = time - np.mean(time)
    denom = float(np.sum(centered_time**2))
    centered_trials = trials - np.nanmean(trials, axis=1, keepdims=True)
    return np.nansum(centered_trials * centered_time[None, :], axis=1) / denom


def _compute_trial_features(trials: np.ndarray) -> pd.DataFrame:
    baseline = trials[:, 0]
    centered = trials - baseline[:, None]
    window = min(3, trials.shape[1])
    trapezoid = getattr(np, "trapezoid", np.trapz)

    features = {
        "mean_level": np.nanmean(trials, axis=1),
        "auc": trapezoid(trials, dx=1.0, axis=1),
        "peak_to_peak": np.nanmax(trials, axis=1) - np.nanmin(trials, axis=1),
        "baseline_to_peak": np.nanmax(centered, axis=1),
        "baseline_to_trough": np.nanmin(centered, axis=1),
        "abs_baseline_response": np.nanmax(np.abs(centered), axis=1),
        "early_late_change": np.nanmean(trials[:, -window:], axis=1) - np.nanmean(trials[:, :window], axis=1),
        "slope": _linear_slope(trials),
        "temporal_sd": np.nanstd(trials, axis=1, ddof=1),
    }
    return pd.DataFrame(features)


def _make_trial_frame(
    med_off: np.ndarray,
    med_on: np.ndarray,
    off_labels: pd.DataFrame,
    on_labels: pd.DataFrame,
) -> pd.DataFrame:
    off_features = pd.concat([off_labels.reset_index(drop=True), _compute_trial_features(med_off)], axis=1)
    off_features["medication"] = "OFF"
    on_features = pd.concat([on_labels.reset_index(drop=True), _compute_trial_features(med_on)], axis=1)
    on_features["medication"] = "ON"
    return pd.concat([off_features, on_features], ignore_index=True)


def _paired_subject_features(
    trial_features: pd.DataFrame,
    feature_names: list[str] | None = None,
) -> pd.DataFrame:
    feature_names = feature_names if feature_names is not None else FEATURE_NAMES
    subject_features = (
        trial_features.groupby(["subject", "medication"], as_index=False)[feature_names]
        .mean()
        .merge(
            trial_features.groupby(["subject", "medication"], as_index=False)
            .size()
            .rename(columns={"size": "n_trials"}),
            on=["subject", "medication"],
            how="left",
        )
    )
    return subject_features


def _fdr_bh(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=np.float64)
    q_values = np.full_like(p_values, np.nan, dtype=np.float64)
    valid = np.isfinite(p_values)
    if not np.any(valid):
        return q_values

    valid_p = p_values[valid]
    order = np.argsort(valid_p)
    ranked = valid_p[order]
    m = ranked.size
    adjusted = ranked * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)

    valid_indices = np.flatnonzero(valid)
    q_values[valid_indices[order]] = adjusted
    return q_values


def _sign_flip_matrix(n_values: int) -> np.ndarray:
    if n_values not in _SIGN_FLIP_CACHE:
        bits = np.arange(1 << n_values, dtype=np.uint64)[:, None]
        shifts = np.arange(n_values, dtype=np.uint64)[None, :]
        _SIGN_FLIP_CACHE[n_values] = np.where(((bits >> shifts) & 1) == 1, 1, -1).astype(np.int8)
    return _SIGN_FLIP_CACHE[n_values]


def _exact_sign_flip_pvalue(diff: np.ndarray, exact_limit: int = 18) -> tuple[float, bool]:
    diff = np.asarray(diff, dtype=np.float64)
    diff = diff[np.isfinite(diff)]
    if diff.size == 0:
        return np.nan, False

    observed = abs(float(np.mean(diff)))
    if observed == 0.0:
        return 1.0, diff.size <= exact_limit

    if diff.size <= exact_limit:
        signs = _sign_flip_matrix(diff.size)
        null_means = signs @ diff / diff.size
        p_value = np.mean(np.abs(null_means) >= observed - 1e-15)
        return float(p_value), True

    rng = np.random.default_rng(0)
    signs = rng.choice((-1.0, 1.0), size=(100000, diff.size))
    null_means = signs @ diff / diff.size
    p_value = np.mean(np.abs(null_means) >= observed - 1e-15)
    return float(p_value), False


def _bootstrap_mean_ci(diff: np.ndarray, rng: np.random.Generator, n_bootstrap: int) -> tuple[float, float]:
    diff = np.asarray(diff, dtype=np.float64)
    diff = diff[np.isfinite(diff)]
    if diff.size == 0:
        return np.nan, np.nan
    if diff.size == 1 or n_bootstrap <= 0:
        value = float(np.mean(diff))
        return value, value
    sample_indices = rng.integers(0, diff.size, size=(int(n_bootstrap), diff.size))
    bootstrap_means = np.mean(diff[sample_indices], axis=1)
    low, high = np.percentile(bootstrap_means, [2.5, 97.5])
    return float(low), float(high)


def _safe_wilcoxon(diff: np.ndarray) -> float:
    diff = np.asarray(diff, dtype=np.float64)
    diff = diff[np.isfinite(diff)]
    if diff.size == 0 or np.allclose(diff, 0.0):
        return np.nan
    try:
        return float(stats.wilcoxon(diff, alternative="two-sided").pvalue)
    except ValueError:
        return np.nan


def _feature_stats(
    subject_features: pd.DataFrame,
    n_bootstrap: int,
    random_state: int,
    feature_names: list[str] | None = None,
) -> pd.DataFrame:
    feature_names = feature_names if feature_names is not None else FEATURE_NAMES
    rng = np.random.default_rng(random_state)
    rows = []

    for feature in feature_names:
        wide = subject_features.pivot(index="subject", columns="medication", values=feature)
        if "OFF" not in wide.columns or "ON" not in wide.columns:
            continue
        paired = wide[["OFF", "ON"]].dropna()
        off = paired["OFF"].to_numpy(dtype=np.float64)
        on = paired["ON"].to_numpy(dtype=np.float64)
        diff = on - off

        p_perm, exact = _exact_sign_flip_pvalue(diff)
        ci_low, ci_high = _bootstrap_mean_ci(diff, rng, n_bootstrap)
        t_test = stats.ttest_1samp(diff, 0.0, nan_policy="omit")
        sd_diff = float(np.nanstd(diff, ddof=1)) if diff.size > 1 else np.nan
        cohen_dz = float(np.nanmean(diff) / sd_diff) if np.isfinite(sd_diff) and sd_diff > 0 else np.nan

        rows.append(
            {
                "feature": feature,
                "label": FEATURE_LABELS[feature],
                "n_subjects": int(diff.size),
                "mean_off": float(np.nanmean(off)),
                "mean_on": float(np.nanmean(on)),
                "mean_on_minus_off": float(np.nanmean(diff)),
                "median_on_minus_off": float(np.nanmedian(diff)),
                "mean_diff_ci95_low": ci_low,
                "mean_diff_ci95_high": ci_high,
                "cohen_dz": cohen_dz,
                "p_perm": p_perm,
                "p_perm_exact": bool(exact),
                "p_wilcoxon": _safe_wilcoxon(diff),
                "p_paired_t": float(t_test.pvalue),
            }
        )

    stats_df = pd.DataFrame(rows)
    if not stats_df.empty:
        stats_df["q_perm_fdr"] = _fdr_bh(stats_df["p_perm"].to_numpy())
        stats_df = stats_df.sort_values(["q_perm_fdr", "p_perm", "feature"], na_position="last")
    return stats_df


def _subject_timecourses(
    med_off: np.ndarray,
    med_on: np.ndarray,
    off_labels: pd.DataFrame,
    on_labels: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    time_cols = [f"t{idx}" for idx in range(med_off.shape[1])]
    off_df = pd.concat([off_labels[["subject"]].reset_index(drop=True), pd.DataFrame(med_off, columns=time_cols)], axis=1)
    on_df = pd.concat([on_labels[["subject"]].reset_index(drop=True), pd.DataFrame(med_on, columns=time_cols)], axis=1)
    off_subject = off_df.groupby("subject")[time_cols].mean()
    on_subject = on_df.groupby("subject")[time_cols].mean()
    return off_subject, on_subject


def _timepoint_stats(off_subject: pd.DataFrame, on_subject: pd.DataFrame) -> pd.DataFrame:
    common_subjects = off_subject.index.intersection(on_subject.index)
    off = off_subject.loc[common_subjects]
    on = on_subject.loc[common_subjects]

    rows = []
    for idx, col in enumerate(off.columns):
        diff = on[col].to_numpy(dtype=np.float64) - off[col].to_numpy(dtype=np.float64)
        p_perm, exact = _exact_sign_flip_pvalue(diff)
        rows.append(
            {
                "time_index": idx,
                "n_subjects": int(diff.size),
                "mean_off": float(off[col].mean()),
                "mean_on": float(on[col].mean()),
                "mean_on_minus_off": float(np.mean(diff)),
                "median_on_minus_off": float(np.median(diff)),
                "p_perm": p_perm,
                "p_perm_exact": bool(exact),
                "p_wilcoxon": _safe_wilcoxon(diff),
                "p_paired_t": float(stats.ttest_1samp(diff, 0.0, nan_policy="omit").pvalue),
            }
        )

    stats_df = pd.DataFrame(rows)
    stats_df["q_perm_fdr"] = _fdr_bh(stats_df["p_perm"].to_numpy())
    return stats_df


def _plot_feature_spaghetti(
    subject_features: pd.DataFrame,
    stats_df: pd.DataFrame,
    output_path: Path,
    signal_label: str = "projected BOLD",
    features: list[str] | None = None,
) -> None:
    plot_features = features if features is not None else FEATURE_NAMES
    if len(plot_features) == 4:
        n_cols = 2
        figsize = (8.6, 8.4)
    else:
        n_cols = min(3, max(1, len(plot_features)))
        figsize = (12, 10) if len(plot_features) == len(FEATURE_NAMES) else (4.3 * n_cols, 4.2 * int(np.ceil(len(plot_features) / n_cols)))
    n_rows = int(np.ceil(len(plot_features) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, constrained_layout=True)
    axes = np.asarray(axes).ravel()
    stats_lookup = stats_df.set_index("feature") if not stats_df.empty else pd.DataFrame()

    for ax, feature in zip(axes, plot_features):
        wide = subject_features.pivot(index="subject", columns="medication", values=feature)
        if "OFF" not in wide.columns or "ON" not in wide.columns:
            ax.set_axis_off()
            continue
        paired = wide[["OFF", "ON"]].dropna()

        for _, row in paired.iterrows():
            ax.plot([0, 1], [row["OFF"], row["ON"]], color="0.65", lw=0.9, alpha=0.65, zorder=1)
        ax.scatter(np.zeros(len(paired)), paired["OFF"], s=24, color="#2f6fbb", alpha=0.9, zorder=2)
        ax.scatter(np.ones(len(paired)), paired["ON"], s=24, color="#c05a2b", alpha=0.9, zorder=2)
        ax.plot([0, 1], [paired["OFF"].mean(), paired["ON"].mean()], color="black", lw=2.0, zorder=3)

        q_value = np.nan
        mean_diff = np.nan
        if feature in stats_lookup.index:
            q_value = float(stats_lookup.loc[feature, "q_perm_fdr"])
            mean_diff = float(stats_lookup.loc[feature, "mean_on_minus_off"])
        title = FEATURE_LABELS[feature]
        subtitle = f"ON-OFF mean={mean_diff:.3g}, q={q_value:.3g}" if np.isfinite(q_value) else ""
        ax.set_title(f"{title}\n{subtitle}", fontsize=10)
        ax.set_xticks([0, 1], ["OFF", "ON"])
        ax.tick_params(axis="both", labelsize=9)
        ax.grid(axis="y", color="0.9", lw=0.8)

    for ax in axes[len(plot_features) :]:
        ax.set_axis_off()

    fig.suptitle(f"Subject-level medication effect on {signal_label} signal features", fontsize=14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _plot_timecourses(
    off_subject: pd.DataFrame,
    on_subject: pd.DataFrame,
    timepoint_stats: pd.DataFrame,
    output_path: Path,
    random_state: int,
    n_bootstrap: int,
    signal_label: str = "projected BOLD",
) -> None:
    common_subjects = off_subject.index.intersection(on_subject.index)
    off = off_subject.loc[common_subjects]
    on = on_subject.loc[common_subjects]
    diff = on.to_numpy(dtype=np.float64) - off.to_numpy(dtype=np.float64)
    time = np.arange(off.shape[1])

    rng = np.random.default_rng(random_state)
    if len(common_subjects) > 1 and n_bootstrap > 0:
        sample_indices = rng.integers(0, len(common_subjects), size=(int(n_bootstrap), len(common_subjects)))
        bootstrap_diff = diff[sample_indices].mean(axis=1)
        diff_low, diff_high = np.percentile(bootstrap_diff, [2.5, 97.5], axis=0)
    else:
        diff_low = diff.mean(axis=0)
        diff_high = diff.mean(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)

    ax = axes[0]
    for subject in common_subjects:
        ax.plot(time, off.loc[subject], color="#2f6fbb", alpha=0.18, lw=1)
        ax.plot(time, on.loc[subject], color="#c05a2b", alpha=0.18, lw=1)
    ax.plot(time, off.mean(axis=0), color="#2f6fbb", lw=2.5, label="OFF subject mean")
    ax.plot(time, on.mean(axis=0), color="#c05a2b", lw=2.5, label="ON subject mean")
    ax.set_title(f"{signal_label.capitalize()} timecourse", fontsize=11)
    ax.set_xlabel("Time index")
    ax.set_ylabel("Projection value")
    ax.grid(color="0.9", lw=0.8)
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1]
    mean_diff = diff.mean(axis=0)
    ax.axhline(0.0, color="0.35", lw=1)
    ax.fill_between(time, diff_low, diff_high, color="#7a7a7a", alpha=0.22, label="95% bootstrap CI")
    ax.plot(time, mean_diff, color="black", lw=2.5, label="Mean ON-OFF")
    significant = timepoint_stats["q_perm_fdr"].to_numpy(dtype=np.float64) < 0.05
    if np.any(significant):
        y_marker = np.nanmin(diff_low) - 0.05 * (np.nanmax(diff_high) - np.nanmin(diff_low))
        ax.scatter(time[significant], np.full(np.count_nonzero(significant), y_marker), color="black", s=28, marker="*")
    ax.set_title("Paired difference by time point", fontsize=11)
    ax.set_xlabel("Time index")
    ax.set_ylabel("ON minus OFF")
    ax.grid(color="0.9", lw=0.8)
    ax.legend(frameon=False, fontsize=9)

    fig.suptitle(f"Medication effect on {signal_label} timecourse", fontsize=14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _write_report(
    path: Path,
    med_off: np.ndarray,
    med_on: np.ndarray,
    subject_features: pd.DataFrame,
    feature_stats: pd.DataFrame,
    timepoint_stats: pd.DataFrame,
    signal_label: str = "projected BOLD",
    input_note: str | None = None,
) -> None:
    common_subjects = (
        subject_features.pivot(index="subject", columns="medication", values="n_trials")
        .dropna()
        .index
        .tolist()
    )
    significant_features = feature_stats[feature_stats["q_perm_fdr"] < 0.05]
    significant_timepoints = timepoint_stats[timepoint_stats["q_perm_fdr"] < 0.05]

    lines = [
        f"Medication OFF vs ON {signal_label} feature analysis",
        "",
        f"OFF array shape: {med_off.shape}",
        f"ON array shape: {med_on.shape}",
        f"Paired subjects used for inference: {len(common_subjects)}",
    ]
    if input_note is not None:
        lines.append(f"Input source: {input_note}")
    lines.extend(
        [
            "",
            "Primary statistical unit: subject.",
            "Each trial is summarized first, then features are averaged within subject and medication state.",
            "Primary p-values are paired ON-minus-OFF exact sign-flip permutation tests on subject means.",
            "q-values are Benjamini-Hochberg FDR corrections across the tested features.",
            "Tested features: "
            + ", ".join(FEATURE_LABELS[feature] for feature in feature_stats["feature"].astype(str).tolist()),
            "",
            "Feature results sorted by FDR q-value:",
        ]
    )

    for _, row in feature_stats.iterrows():
        lines.append(
            f"- {row['label']}: mean ON-OFF={row['mean_on_minus_off']:.6g}, "
            f"95% CI [{row['mean_diff_ci95_low']:.6g}, {row['mean_diff_ci95_high']:.6g}], "
            f"p_perm={row['p_perm']:.6g}, q={row['q_perm_fdr']:.6g}, dz={row['cohen_dz']:.3g}"
        )

    lines.extend(["", "Features passing q < 0.05:"])
    if significant_features.empty:
        lines.append("- None")
    else:
        for _, row in significant_features.iterrows():
            lines.append(f"- {row['label']} ({row['feature']})")

    lines.extend(["", "Time points passing q < 0.05:"])
    if significant_timepoints.empty:
        lines.append("- None")
    else:
        for _, row in significant_timepoints.iterrows():
            lines.append(f"- time_index {int(row['time_index'])}: q={row['q_perm_fdr']:.6g}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _gvs_code_from_path(path: Path) -> str:
    prefix = "Projection_BOLD_trials_"
    stem = path.stem
    if not stem.startswith(prefix):
        raise ValueError(f"Unexpected GVS projection filename: {path.name}")
    return stem[len(prefix) :]


def _load_gvs_trials(gvs_dir: Path) -> dict[str, np.ndarray]:
    paths = sorted(gvs_dir.glob("Projection_BOLD_trials_gvs-*.npy"))
    if not paths:
        raise FileNotFoundError(f"No Projection_BOLD_trials_gvs-*.npy files found in {gvs_dir}")

    trials_by_code = {}
    n_timepoints = None
    for path in paths:
        code = _gvs_code_from_path(path)
        trials = _load_trials(path)
        if n_timepoints is None:
            n_timepoints = trials.shape[1]
        elif trials.shape[1] != n_timepoints:
            raise ValueError(f"{path} has {trials.shape[1]} time points; expected {n_timepoints}")
        trials_by_code[code] = trials
    return trials_by_code


def _gvs_trials_from_full_trials(full_trials: np.ndarray, gvs_dir: Path) -> dict[str, np.ndarray]:
    metadata = _load_gvs_metadata(gvs_dir)
    if "projected_trial_index" not in metadata.columns:
        raise ValueError(f"{gvs_dir / 'gvs_projection_trial_metadata.tsv'} is missing projected_trial_index")

    trials_by_code = {}
    for code in sorted(metadata["gvs_code"].unique()):
        indices = metadata.loc[metadata["gvs_code"].eq(code), "projected_trial_index"].to_numpy(dtype=np.int64)
        if indices.size == 0:
            continue
        if np.nanmin(indices) < 0 or np.nanmax(indices) >= full_trials.shape[0]:
            raise ValueError(f"{code} projected_trial_index values are outside full trial range {full_trials.shape[0]}")
        trials_by_code[str(code)] = np.asarray(full_trials[indices], dtype=np.float64)

    if not trials_by_code:
        raise RuntimeError(f"No GVS trial arrays could be derived from {gvs_dir}")
    return trials_by_code


def _load_gvs_metadata(gvs_dir: Path) -> pd.DataFrame:
    metadata_path = gvs_dir / "gvs_projection_trial_metadata.tsv"
    metadata = pd.read_csv(metadata_path, sep="\t")
    required = {
        "subject",
        "session",
        "medication",
        "run",
        "block_index",
        "trial_in_block",
        "gvs_id",
        "gvs_code",
    }
    missing = sorted(required - set(metadata.columns))
    if missing:
        raise ValueError(f"{metadata_path} is missing required columns: {', '.join(missing)}")
    return metadata


def _make_gvs_frames(
    gvs_dir: Path,
    trials_by_code: dict[str, np.ndarray] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    if trials_by_code is None:
        trials_by_code = _load_gvs_trials(gvs_dir)
    metadata = _load_gvs_metadata(gvs_dir)
    feature_frames = []
    signal_frames = []

    for code, trials in sorted(trials_by_code.items()):
        labels = metadata.loc[metadata["gvs_code"].eq(code)].reset_index(drop=True).copy()
        if labels.shape[0] != trials.shape[0]:
            raise ValueError(
                f"{code} metadata has {labels.shape[0]} rows, but its projection array has {trials.shape[0]} rows"
            )

        feature_frames.append(pd.concat([labels, _compute_trial_features(trials)], axis=1))
        time_cols = [f"t{idx}" for idx in range(trials.shape[1])]
        signal_frames.append(pd.concat([labels, pd.DataFrame(trials, columns=time_cols)], axis=1))

    return pd.concat(feature_frames, ignore_index=True), pd.concat(signal_frames, ignore_index=True), trials_by_code


def _gvs_run_feature_means(trial_features: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["subject", "session", "medication", "run", "gvs_code"]
    run_features = (
        trial_features.groupby(group_cols, as_index=False)[FEATURE_NAMES]
        .mean()
        .merge(
            trial_features.groupby(group_cols, as_index=False).size().rename(columns={"size": "n_trials"}),
            on=group_cols,
            how="left",
        )
    )
    return run_features.sort_values(group_cols).reset_index(drop=True)


def _gvs_subject_feature_pairs(
    run_features: pd.DataFrame,
    sham_code: str,
    feature_names: list[str] | None = None,
) -> pd.DataFrame:
    feature_names = feature_names if feature_names is not None else FEATURE_NAMES
    index_cols = ["subject", "session", "medication", "run"]
    sham_runs = run_features.loc[run_features["gvs_code"].eq(sham_code)].set_index(index_cols)
    active_codes = sorted(code for code in run_features["gvs_code"].unique() if code != sham_code)
    rows = []

    for active_code in active_codes:
        active_runs = run_features.loc[run_features["gvs_code"].eq(active_code)].set_index(index_cols)
        paired_runs = sham_runs.join(active_runs, how="inner", lsuffix="_sham", rsuffix="_active")
        if paired_runs.empty:
            continue

        for feature in feature_names:
            per_run = pd.DataFrame(
                {
                    "subject": paired_runs.index.get_level_values("subject"),
                    "sham_value": paired_runs[f"{feature}_sham"].to_numpy(dtype=np.float64),
                    "active_value": paired_runs[f"{feature}_active"].to_numpy(dtype=np.float64),
                    "n_sham_trials": paired_runs["n_trials_sham"].to_numpy(dtype=np.float64),
                    "n_active_trials": paired_runs["n_trials_active"].to_numpy(dtype=np.float64),
                }
            )
            per_run["delta_active_minus_sham"] = per_run["active_value"] - per_run["sham_value"]
            per_subject = (
                per_run.replace([np.inf, -np.inf], np.nan)
                .dropna(subset=["sham_value", "active_value", "delta_active_minus_sham"])
                .groupby("subject", as_index=False)
                .agg(
                    sham_value=("sham_value", "mean"),
                    active_value=("active_value", "mean"),
                    delta_active_minus_sham=("delta_active_minus_sham", "mean"),
                    n_runs=("delta_active_minus_sham", "size"),
                    n_sham_trials=("n_sham_trials", "sum"),
                    n_active_trials=("n_active_trials", "sum"),
                )
            )
            per_subject.insert(0, "feature", feature)
            per_subject.insert(0, "label", FEATURE_LABELS[feature])
            per_subject.insert(0, "sham_gvs", sham_code)
            per_subject.insert(0, "active_gvs", active_code)
            rows.append(per_subject)

    if not rows:
        raise RuntimeError(f"No run-paired GVS comparisons could be built against {sham_code}")
    return pd.concat(rows, ignore_index=True).sort_values(["active_gvs", "feature", "subject"]).reset_index(drop=True)


def _gvs_feature_stats(
    subject_pairs: pd.DataFrame,
    n_bootstrap: int,
    random_state: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    rows = []

    for (active_gvs, feature), group in subject_pairs.groupby(["active_gvs", "feature"], sort=True):
        diff = group["delta_active_minus_sham"].to_numpy(dtype=np.float64)
        sham = group["sham_value"].to_numpy(dtype=np.float64)
        active = group["active_value"].to_numpy(dtype=np.float64)
        p_perm, exact = _exact_sign_flip_pvalue(diff)
        ci_low, ci_high = _bootstrap_mean_ci(diff, rng, n_bootstrap)
        t_test = stats.ttest_1samp(diff, 0.0, nan_policy="omit")
        sd_diff = float(np.nanstd(diff, ddof=1)) if diff.size > 1 else np.nan
        cohen_dz = float(np.nanmean(diff) / sd_diff) if np.isfinite(sd_diff) and sd_diff > 0 else np.nan

        rows.append(
            {
                "active_gvs": active_gvs,
                "sham_gvs": str(group["sham_gvs"].iloc[0]),
                "feature": feature,
                "label": FEATURE_LABELS[feature],
                "n_subjects": int(np.count_nonzero(np.isfinite(diff))),
                "mean_sham": float(np.nanmean(sham)),
                "mean_active": float(np.nanmean(active)),
                "mean_active_minus_sham": float(np.nanmean(diff)),
                "median_active_minus_sham": float(np.nanmedian(diff)),
                "mean_diff_ci95_low": ci_low,
                "mean_diff_ci95_high": ci_high,
                "cohen_dz": cohen_dz,
                "p_perm": p_perm,
                "p_perm_exact": bool(exact),
                "p_wilcoxon": _safe_wilcoxon(diff),
                "p_paired_t": float(t_test.pvalue),
            }
        )

    stats_df = pd.DataFrame(rows)
    stats_df["q_perm_fdr"] = _fdr_bh(stats_df["p_perm"].to_numpy())
    return stats_df.sort_values(["active_gvs", "q_perm_fdr", "p_perm", "feature"]).reset_index(drop=True)


def _gvs_subject_run_feature_stats(
    run_features: pd.DataFrame,
    sham_code: str,
    feature_names: list[str] | None = None,
) -> pd.DataFrame:
    feature_names = feature_names if feature_names is not None else FEATURE_NAMES
    index_cols = ["subject", "session", "medication", "run"]
    sham_runs = run_features.loc[run_features["gvs_code"].eq(sham_code)].set_index(index_cols)
    active_codes = sorted(code for code in run_features["gvs_code"].unique() if code != sham_code)
    rows = []

    for active_code in active_codes:
        active_runs = run_features.loc[run_features["gvs_code"].eq(active_code)].set_index(index_cols)
        paired_runs = sham_runs.join(active_runs, how="inner", lsuffix="_sham", rsuffix="_active")
        if paired_runs.empty:
            continue

        for subject, group in paired_runs.groupby(level="subject", sort=True):
            for feature in feature_names:
                sham = group[f"{feature}_sham"].to_numpy(dtype=np.float64)
                active = group[f"{feature}_active"].to_numpy(dtype=np.float64)
                diff = active - sham
                diff = diff[np.isfinite(diff)]
                p_perm, exact = _exact_sign_flip_pvalue(diff)
                t_test_p = float(stats.ttest_1samp(diff, 0.0, nan_policy="omit").pvalue) if diff.size > 1 else np.nan
                rows.append(
                    {
                        "subject": subject,
                        "active_gvs": active_code,
                        "sham_gvs": sham_code,
                        "feature": feature,
                        "label": FEATURE_LABELS[feature],
                        "n_runs": int(diff.size),
                        "mean_sham": float(np.nanmean(sham)),
                        "mean_active": float(np.nanmean(active)),
                        "mean_active_minus_sham": float(np.nanmean(diff)),
                        "median_active_minus_sham": float(np.nanmedian(diff)),
                        "p_perm": p_perm,
                        "p_perm_exact": bool(exact),
                        "p_wilcoxon": _safe_wilcoxon(diff),
                        "p_paired_t": t_test_p,
                    }
                )

    if not rows:
        raise RuntimeError(f"No subject run-level GVS comparisons could be built against {sham_code}")

    stats_df = pd.DataFrame(rows)
    stats_df["q_perm_fdr"] = np.nan
    for _, index in stats_df.groupby("subject", sort=True).groups.items():
        stats_df.loc[index, "q_perm_fdr"] = _fdr_bh(stats_df.loc[index, "p_perm"].to_numpy())
    return stats_df.sort_values(["subject", "active_gvs", "q_perm_fdr", "p_perm", "feature"]).reset_index(drop=True)


def _gvs_subject_timecourse_pairs(signal_frame: pd.DataFrame, sham_code: str) -> pd.DataFrame:
    group_cols = ["subject", "session", "medication", "run", "gvs_code"]
    time_cols = [col for col in signal_frame.columns if col.startswith("t") and col[1:].isdigit()]
    run_time = (
        signal_frame.groupby(group_cols, as_index=False)[time_cols]
        .mean()
        .merge(
            signal_frame.groupby(group_cols, as_index=False).size().rename(columns={"size": "n_trials"}),
            on=group_cols,
            how="left",
        )
    )

    index_cols = ["subject", "session", "medication", "run"]
    sham_runs = run_time.loc[run_time["gvs_code"].eq(sham_code)].set_index(index_cols)
    active_codes = sorted(code for code in run_time["gvs_code"].unique() if code != sham_code)
    rows = []

    for active_code in active_codes:
        active_runs = run_time.loc[run_time["gvs_code"].eq(active_code)].set_index(index_cols)
        paired_runs = sham_runs.join(active_runs, how="inner", lsuffix="_sham", rsuffix="_active")
        if paired_runs.empty:
            continue

        for subject, group in paired_runs.groupby(level="subject", sort=True):
            record: dict[str, object] = {
                "active_gvs": active_code,
                "sham_gvs": sham_code,
                "subject": subject,
                "n_runs": int(group.shape[0]),
                "n_sham_trials": int(group["n_trials_sham"].sum()),
                "n_active_trials": int(group["n_trials_active"].sum()),
            }
            for col in time_cols:
                sham_values = group[f"{col}_sham"].to_numpy(dtype=np.float64)
                active_values = group[f"{col}_active"].to_numpy(dtype=np.float64)
                record[f"{col}_sham"] = float(np.nanmean(sham_values))
                record[f"{col}_active"] = float(np.nanmean(active_values))
                record[f"{col}_delta"] = float(np.nanmean(active_values - sham_values))
            rows.append(record)

    if not rows:
        raise RuntimeError(f"No subject-level GVS timecourse pairs could be built against {sham_code}")
    return pd.DataFrame(rows).sort_values(["active_gvs", "subject"]).reset_index(drop=True)


def _gvs_run_timecourse_pairs(signal_frame: pd.DataFrame, sham_code: str) -> pd.DataFrame:
    group_cols = ["subject", "session", "medication", "run", "gvs_code"]
    time_cols = [col for col in signal_frame.columns if col.startswith("t") and col[1:].isdigit()]
    run_time = (
        signal_frame.groupby(group_cols, as_index=False)[time_cols]
        .mean()
        .merge(
            signal_frame.groupby(group_cols, as_index=False).size().rename(columns={"size": "n_trials"}),
            on=group_cols,
            how="left",
        )
    )

    index_cols = ["subject", "session", "medication", "run"]
    sham_runs = run_time.loc[run_time["gvs_code"].eq(sham_code)].set_index(index_cols)
    active_codes = sorted(code for code in run_time["gvs_code"].unique() if code != sham_code)
    rows = []

    for active_code in active_codes:
        active_runs = run_time.loc[run_time["gvs_code"].eq(active_code)].set_index(index_cols)
        paired_runs = sham_runs.join(active_runs, how="inner", lsuffix="_sham", rsuffix="_active")
        if paired_runs.empty:
            continue

        for index_values, row in paired_runs.iterrows():
            subject, session, medication, run = index_values
            record: dict[str, object] = {
                "active_gvs": active_code,
                "sham_gvs": sham_code,
                "subject": subject,
                "session": int(session),
                "medication": medication,
                "run": int(run),
                "n_sham_trials": int(row["n_trials_sham"]),
                "n_active_trials": int(row["n_trials_active"]),
            }
            for col in time_cols:
                time_index = int(col[1:])
                sham_value = float(row[f"{col}_sham"])
                active_value = float(row[f"{col}_active"])
                record[f"t{time_index}_sham"] = sham_value
                record[f"t{time_index}_active"] = active_value
                record[f"t{time_index}_delta"] = active_value - sham_value
            rows.append(record)

    if not rows:
        raise RuntimeError(f"No run-level GVS timecourse pairs could be built against {sham_code}")
    return pd.DataFrame(rows).sort_values(["subject", "active_gvs", "session", "run"]).reset_index(drop=True)


def _gvs_timepoint_stats(timecourse_pairs: pd.DataFrame) -> pd.DataFrame:
    delta_cols = sorted(
        [col for col in timecourse_pairs.columns if col.startswith("t") and col.endswith("_delta")],
        key=lambda col: int(col[1:].split("_", 1)[0]),
    )
    rows = []

    for active_gvs, group in timecourse_pairs.groupby("active_gvs", sort=True):
        for col in delta_cols:
            time_index = int(col[1:].split("_", 1)[0])
            diff = group[col].to_numpy(dtype=np.float64)
            p_perm, exact = _exact_sign_flip_pvalue(diff)
            sham_col = f"t{time_index}_sham"
            active_col = f"t{time_index}_active"
            rows.append(
                {
                    "active_gvs": active_gvs,
                    "sham_gvs": str(group["sham_gvs"].iloc[0]),
                    "time_index": time_index,
                    "n_subjects": int(np.count_nonzero(np.isfinite(diff))),
                    "mean_sham": float(np.nanmean(group[sham_col].to_numpy(dtype=np.float64))),
                    "mean_active": float(np.nanmean(group[active_col].to_numpy(dtype=np.float64))),
                    "mean_active_minus_sham": float(np.nanmean(diff)),
                    "median_active_minus_sham": float(np.nanmedian(diff)),
                    "p_perm": p_perm,
                    "p_perm_exact": bool(exact),
                    "p_wilcoxon": _safe_wilcoxon(diff),
                    "p_paired_t": float(stats.ttest_1samp(diff, 0.0, nan_policy="omit").pvalue),
                }
            )

    stats_df = pd.DataFrame(rows)
    stats_df["q_perm_fdr"] = _fdr_bh(stats_df["p_perm"].to_numpy())
    return stats_df.sort_values(["active_gvs", "time_index"]).reset_index(drop=True)


def _safe_name(value: str) -> str:
    return value.replace("-", "_")


def _gvs_display_label(gvs_code: str) -> str:
    raw_code = str(gvs_code).strip().lower()
    code = raw_code.replace("_", "-").replace(" ", "-")
    if code in {"gvs-01", "gvs-01-sham", "gvs1-sham", "sham"}:
        return "sham"
    if code.startswith("gvs-"):
        try:
            gvs_index = int(code.split("-", 1)[1])
        except ValueError:
            return str(gvs_code)
        if gvs_index == 1:
            return "sham"
        return f"gvs{gvs_index - 1}"
    return str(gvs_code)


def _resolve_existing_path(path: Path) -> Path:
    path = Path(path)
    if path.is_absolute() or path.exists():
        return path
    return Path(__file__).resolve().parent / path


def _gvs_condition_order_by_subject_from_rt(
    run_metrics_path: Path,
    inventory_path: Path,
    behaviour_column: int,
) -> dict[str, list[str]]:
    run_metrics_path = _resolve_existing_path(run_metrics_path)
    inventory_path = _resolve_existing_path(inventory_path)
    if not run_metrics_path.exists() or not inventory_path.exists():
        warnings.warn(
            f"Could not find RT ordering inputs: {run_metrics_path} and/or {inventory_path}; "
            "using default GVS panel order.",
            stacklevel=2,
        )
        return {}

    run_metrics = pd.read_csv(run_metrics_path)
    inventory = pd.read_csv(inventory_path)
    required_metrics = {"sub_tag", "ses", "run", "behaviour_path"}
    required_inventory = {"subject", "session", "run", "condition_code", "trial_start", "trial_stop"}
    missing_metrics = sorted(required_metrics - set(run_metrics.columns))
    missing_inventory = sorted(required_inventory - set(inventory.columns))
    if missing_metrics or missing_inventory:
        raise RuntimeError(
            f"RT order inputs are missing columns: "
            f"{run_metrics_path}: {', '.join(missing_metrics) or 'ok'}; "
            f"{inventory_path}: {', '.join(missing_inventory) or 'ok'}"
        )

    run_metrics = run_metrics.rename(columns={"sub_tag": "subject", "ses": "session"})
    metric_lookup = {
        (str(row.subject), int(row.session), int(row.run)): row
        for row in run_metrics.itertuples(index=False)
    }
    behaviour_cache: dict[tuple[str, int, int], np.ndarray] = {}
    rows: list[dict[str, object]] = []

    for row in inventory.itertuples(index=False):
        subject = str(row.subject)
        session = int(row.session)
        run = int(row.run)
        key = (subject, session, run)
        metric_row = metric_lookup.get(key)
        if metric_row is None:
            continue

        if key not in behaviour_cache:
            behaviour_path = _resolve_existing_path(Path(getattr(metric_row, "behaviour_path")))
            behaviour_metric = _load_behaviour_rt(behaviour_path, behaviour_column)
            behaviour_cache[key] = _rt_seconds(behaviour_metric, input_is_inverse_rt=True) * 1000.0

        rt_ms = behaviour_cache[key]
        start = int(row.trial_start)
        stop = min(int(row.trial_stop), rt_ms.shape[0])
        if start >= stop:
            continue

        values = rt_ms[start:stop]
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            continue
        rows.append(
            {
                "subject": subject,
                "condition_code": str(row.condition_code),
                "rt_sum": float(np.sum(finite)),
                "n_rt": int(finite.size),
            }
        )

    if not rows:
        warnings.warn("No finite RT values were available for GVS panel ordering; using default order.", stacklevel=2)
        return {}

    order_df = pd.DataFrame(rows)
    subject_condition = (
        order_df.groupby(["subject", "condition_code"], as_index=False)
        .agg(rt_sum=("rt_sum", "sum"), n_rt=("n_rt", "sum"))
        .query("n_rt > 0")
    )
    subject_condition["mean_rt_ms"] = subject_condition["rt_sum"] / subject_condition["n_rt"]

    return {
        subject: group.sort_values(["mean_rt_ms", "condition_code"])["condition_code"].astype(str).tolist()
        for subject, group in subject_condition.groupby("subject", sort=True)
    }


def _plot_gvs_feature_spaghetti(
    subject_pairs: pd.DataFrame,
    stats_df: pd.DataFrame,
    active_gvs: str,
    sham_code: str,
    output_path: Path,
    signal_label: str = "projected BOLD",
    features: list[str] | None = None,
) -> None:
    plot_features = features if features is not None else FEATURE_NAMES
    data = subject_pairs.loc[subject_pairs["active_gvs"].eq(active_gvs)].copy()
    stats_lookup = stats_df.loc[stats_df["active_gvs"].eq(active_gvs)].set_index("feature")
    if len(plot_features) == 4:
        n_cols = 2
        figsize = (8.6, 8.4)
    else:
        n_cols = min(3, max(1, len(plot_features)))
        figsize = (12, 10) if len(plot_features) == len(FEATURE_NAMES) else (4.3 * n_cols, 4.2 * int(np.ceil(len(plot_features) / n_cols)))
    n_rows = int(np.ceil(len(plot_features) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, constrained_layout=True)
    axes = np.asarray(axes).ravel()

    for ax, feature in zip(axes, plot_features):
        feature_data = data.loc[data["feature"].eq(feature)].sort_values("delta_active_minus_sham")
        sham = feature_data["sham_value"].to_numpy(dtype=np.float64)
        active = feature_data["active_value"].to_numpy(dtype=np.float64)

        for y0, y1 in zip(sham, active):
            ax.plot([0, 1], [y0, y1], color="0.65", lw=0.9, alpha=0.65, zorder=1)
        ax.scatter(np.zeros(len(feature_data)), sham, s=24, color="#777777", alpha=0.9, zorder=2)
        ax.scatter(np.ones(len(feature_data)), active, s=24, color="#2a8f5a", alpha=0.9, zorder=2)
        ax.plot([0, 1], [np.nanmean(sham), np.nanmean(active)], color="black", lw=2.0, zorder=3)

        q_value = float(stats_lookup.loc[feature, "q_perm_fdr"]) if feature in stats_lookup.index else np.nan
        mean_diff = (
            float(stats_lookup.loc[feature, "mean_active_minus_sham"]) if feature in stats_lookup.index else np.nan
        )
        subtitle = f"{active_gvs}-sham mean={mean_diff:.3g}, q={q_value:.3g}" if np.isfinite(q_value) else ""
        ax.set_title(f"{FEATURE_LABELS[feature]}\n{subtitle}", fontsize=10)
        ax.set_xticks([0, 1], [f"{sham_code}\nsham", active_gvs])
        ax.tick_params(axis="both", labelsize=9)
        ax.grid(axis="y", color="0.9", lw=0.8)

    for ax in axes[len(plot_features) :]:
        ax.set_axis_off()

    fig.suptitle(f"{active_gvs} vs {sham_code} {signal_label} signal features", fontsize=14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _plot_gvs_timecourse_spaghetti(
    timecourse_pairs: pd.DataFrame,
    timepoint_stats: pd.DataFrame,
    active_gvs: str,
    sham_code: str,
    output_path: Path,
    random_state: int,
    n_bootstrap: int,
    signal_label: str = "projected BOLD",
) -> None:
    data = timecourse_pairs.loc[timecourse_pairs["active_gvs"].eq(active_gvs)].copy()
    time_indices = sorted(
        int(col[1:].split("_", 1)[0]) for col in data.columns if col.startswith("t") and col.endswith("_delta")
    )
    time = np.asarray(time_indices, dtype=int)
    sham = data[[f"t{idx}_sham" for idx in time_indices]].to_numpy(dtype=np.float64)
    active = data[[f"t{idx}_active" for idx in time_indices]].to_numpy(dtype=np.float64)
    diff = data[[f"t{idx}_delta" for idx in time_indices]].to_numpy(dtype=np.float64)

    rng = np.random.default_rng(random_state)
    if data.shape[0] > 1 and n_bootstrap > 0:
        sample_indices = rng.integers(0, data.shape[0], size=(int(n_bootstrap), data.shape[0]))
        bootstrap_diff = diff[sample_indices].mean(axis=1)
        diff_low, diff_high = np.percentile(bootstrap_diff, [2.5, 97.5], axis=0)
    else:
        diff_low = diff.mean(axis=0)
        diff_high = diff.mean(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)

    ax = axes[0]
    for row_index in range(data.shape[0]):
        ax.plot(time, sham[row_index], color="#777777", alpha=0.18, lw=1)
        ax.plot(time, active[row_index], color="#2a8f5a", alpha=0.18, lw=1)
    ax.plot(time, np.nanmean(sham, axis=0), color="#555555", lw=2.5, label=f"{sham_code} sham")
    ax.plot(time, np.nanmean(active, axis=0), color="#2a8f5a", lw=2.5, label=active_gvs)
    ax.set_title(f"{signal_label.capitalize()} timecourse", fontsize=11)
    ax.set_xlabel("Time index")
    ax.set_ylabel("Projection value")
    ax.grid(color="0.9", lw=0.8)
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1]
    mean_diff = np.nanmean(diff, axis=0)
    ax.axhline(0.0, color="0.35", lw=1)
    ax.fill_between(time, diff_low, diff_high, color="#7a7a7a", alpha=0.22, label="95% bootstrap CI")
    ax.plot(time, mean_diff, color="black", lw=2.5, label=f"{active_gvs} minus sham")
    stats_subset = timepoint_stats.loc[timepoint_stats["active_gvs"].eq(active_gvs)].sort_values("time_index")
    significant = stats_subset["q_perm_fdr"].to_numpy(dtype=np.float64) < 0.05
    if np.any(significant):
        y_marker = np.nanmin(diff_low) - 0.05 * (np.nanmax(diff_high) - np.nanmin(diff_low))
        ax.scatter(time[significant], np.full(np.count_nonzero(significant), y_marker), color="black", s=28, marker="*")
    ax.set_title("Paired difference by time point", fontsize=11)
    ax.set_xlabel("Time index")
    ax.set_ylabel(f"{active_gvs} minus sham")
    ax.grid(color="0.9", lw=0.8)
    ax.legend(frameon=False, fontsize=9)

    fig.suptitle(f"{active_gvs} vs {sham_code} {signal_label} timecourse", fontsize=14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _plot_gvs_timecourse_delta_grid(
    timecourse_pairs: pd.DataFrame,
    active_codes: list[str],
    output_path: Path,
    random_state: int,
    n_bootstrap: int,
) -> None:
    n_cols = 4
    n_rows = int(np.ceil(len(active_codes) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.7 * n_cols, 4.7 * n_rows), constrained_layout=True)
    axes = np.asarray(axes).ravel()

    for ax, active_gvs in zip(axes, active_codes):
        active_label = _gvs_display_label(active_gvs)
        data = timecourse_pairs.loc[timecourse_pairs["active_gvs"].eq(active_gvs)].copy()
        if data.empty:
            ax.set_axis_off()
            continue
        time_indices = sorted(
            int(col[1:].split("_", 1)[0]) for col in data.columns if col.startswith("t") and col.endswith("_delta")
        )
        if not time_indices:
            ax.set_axis_off()
            continue
        time = np.asarray(time_indices, dtype=int)
        diff = data[[f"t{idx}_delta" for idx in time_indices]].to_numpy(dtype=np.float64)

        rng = np.random.default_rng(random_state)
        if data.shape[0] > 1 and n_bootstrap > 0:
            sample_indices = rng.integers(0, data.shape[0], size=(int(n_bootstrap), data.shape[0]))
            bootstrap_diff = diff[sample_indices].mean(axis=1)
            diff_low, diff_high = np.percentile(bootstrap_diff, [2.5, 97.5], axis=0)
        else:
            diff_low = diff.mean(axis=0)
            diff_high = diff.mean(axis=0)

        mean_diff = np.nanmean(diff, axis=0)
        ax.axhline(0.0, color="0.35", lw=1)
        ax.fill_between(time, diff_low, diff_high, color="#7a7a7a", alpha=0.22, label="95% bootstrap CI")
        ax.plot(time, mean_diff, color="black", lw=2.5, label=f"{active_label} minus sham")
        ax.set_title(active_label, fontsize=11)
        ax.set_xlabel("Time index")
        ax.set_ylabel(f"{active_label} minus sham")
        ax.grid(color="0.9", lw=0.8)
        ax.legend(frameon=False, fontsize=9)

    for ax in axes[len(active_codes) :]:
        ax.set_axis_off()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _plot_gvs_timecourse_original_grid(
    timecourse_pairs: pd.DataFrame,
    active_codes: list[str],
    sham_code: str,
    output_path: Path,
    random_state: int,
    n_bootstrap: int,
    show_ci: bool = True,
    change_from_t0: bool = False,
    condition_order: list[str] | None = None,
) -> None:
    time_indices = sorted(
        int(col[1:].split("_", 1)[0]) for col in timecourse_pairs.columns if col.startswith("t") and col.endswith("_active")
    )
    if not time_indices:
        raise ValueError("No active timecourse columns were found")

    time = np.asarray(time_indices, dtype=int)
    n_cols = 3
    n_rows = 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.7 * n_cols, 4.7 * n_rows), constrained_layout=True)
    axes = np.asarray(axes).ravel()

    panels_by_code: dict[str, tuple[str, np.ndarray]] = {}
    dedupe_cols = [col for col in ["subject", "session", "medication", "run"] if col in timecourse_pairs.columns]
    sham_cols = [f"t{idx}_sham" for idx in time_indices]
    if dedupe_cols:
        sham_data = timecourse_pairs.drop_duplicates(dedupe_cols)
    else:
        sham_data = timecourse_pairs
    panels_by_code[str(sham_code)] = (_gvs_display_label(sham_code), sham_data[sham_cols].to_numpy(dtype=np.float64))

    for active_gvs in active_codes:
        active_label = _gvs_display_label(active_gvs)
        data = timecourse_pairs.loc[timecourse_pairs["active_gvs"].eq(active_gvs)].copy()
        active_cols = [f"t{idx}_active" for idx in time_indices]
        panels_by_code[str(active_gvs)] = (active_label, data[active_cols].to_numpy(dtype=np.float64))

    default_order = [str(sham_code), *[str(code) for code in active_codes]]
    if condition_order is None:
        panel_order = default_order
    else:
        seen: set[str] = set()
        panel_order = []
        for code in [*condition_order, *default_order]:
            code = str(code)
            if code in panels_by_code and code not in seen:
                panel_order.append(code)
                seen.add(code)
    panels = [panels_by_code[code] for code in panel_order]

    y_label = "Change from t0" if change_from_t0 else "Original projection signal"
    line_prefix = f"{'change from t0' if change_from_t0 else 'mean'}"

    for panel_index, (ax, (label, values)) in enumerate(zip(axes, panels)):
        if values.size == 0:
            ax.set_axis_off()
            continue

        plot_values = values - values[:, [0]] if change_from_t0 else values

        if show_ci:
            rng = np.random.default_rng(random_state + panel_index)
            if plot_values.shape[0] > 1 and n_bootstrap > 0:
                sample_indices = rng.integers(0, plot_values.shape[0], size=(int(n_bootstrap), plot_values.shape[0]))
                bootstrap_mean = np.nanmean(plot_values[sample_indices], axis=1)
                low, high = np.nanpercentile(bootstrap_mean, [2.5, 97.5], axis=0)
            else:
                low = np.nanmean(plot_values, axis=0)
                high = np.nanmean(plot_values, axis=0)
            ax.fill_between(time, low, high, color="#7a7a7a", alpha=0.22, label="95% bootstrap CI")

        mean_signal = np.nanmean(plot_values, axis=0)
        if change_from_t0:
            ax.axhline(0.0, color="0.55", lw=0.9)
        ax.plot(time, mean_signal, color="black", lw=2.5, label=f"{label} {line_prefix}")
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Time index")
        ax.set_ylabel(y_label)
        ax.grid(color="0.9", lw=0.8)
        ax.legend(frameon=False, fontsize=8)

    for ax in axes[len(panels) :]:
        ax.set_axis_off()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _plot_gvs_feature_q_heatmap(
    stats_df: pd.DataFrame,
    output_path: Path,
    title: str | None = "GVS vs sham feature evidence (-log10 FDR q)",
    y_tick_labels: list[str] | None = None,
    features: list[str] | None = None,
) -> None:
    plot_features = features if features is not None else FEATURE_NAMES
    active_codes = sorted(stats_df["active_gvs"].unique())
    if y_tick_labels is not None and len(y_tick_labels) != len(active_codes):
        raise ValueError("y_tick_labels must match the number of active GVS conditions")
    q_matrix = (
        stats_df.pivot(index="active_gvs", columns="feature", values="q_perm_fdr")
        .reindex(index=active_codes, columns=plot_features)
        .to_numpy(dtype=np.float64)
    )
    p_matrix = (
        stats_df.pivot(index="active_gvs", columns="feature", values="p_perm")
        .reindex(index=active_codes, columns=plot_features)
        .to_numpy(dtype=np.float64)
    )
    score = -np.log10(np.clip(q_matrix, 1e-6, 1.0))

    fig_width = 12 if len(plot_features) == len(FEATURE_NAMES) else max(8.2, 1.6 * len(plot_features) + 2.4)
    fig, ax = plt.subplots(figsize=(fig_width, 5.2), constrained_layout=True)
    image = ax.imshow(score, aspect="auto", cmap="viridis", vmin=0.0, vmax=max(2.0, float(np.nanmax(score))))
    ax.set_xticks(np.arange(len(plot_features)), [FEATURE_LABELS[name] for name in plot_features], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(active_codes)), y_tick_labels if y_tick_labels is not None else active_codes)
    if title:
        ax.set_title(title, fontsize=13)
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("-log10(q)")

    for row in range(q_matrix.shape[0]):
        for col in range(q_matrix.shape[1]):
            q_value = q_matrix[row, col]
            p_value = p_matrix[row, col]
            if np.isfinite(q_value):
                text = f"q={q_value:.2g}\np={p_value:.2g}"
                red, green, blue, _ = image.cmap(image.norm(score[row, col]))
                luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
                text_color = "black" if luminance > 0.55 else "white"
                ax.text(col, row, text, ha="center", va="center", fontsize=7, color=text_color)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _write_gvs_report(
    path: Path,
    trials_by_code: dict[str, np.ndarray],
    subject_pairs: pd.DataFrame,
    feature_stats: pd.DataFrame,
    timepoint_stats: pd.DataFrame,
    sham_code: str,
    signal_label: str = "projected BOLD",
    input_note: str | None = None,
) -> None:
    lines = [
        f"GVS {signal_label} feature analysis",
        "",
        f"Sham/no-GVS reference: {sham_code}",
        "Primary statistical unit: subject.",
        "Within each subject/session/run, each active GVS block is paired with that run's sham block.",
        "Run-level active-minus-sham deltas are averaged within subject before inference.",
        "Primary p-values are paired exact sign-flip permutation tests on subject mean deltas.",
        "Feature q-values are Benjamini-Hochberg FDR corrections across all GVS-by-tested-feature tests.",
        "Tested features: "
        + ", ".join(FEATURE_LABELS[feature] for feature in feature_stats["feature"].drop_duplicates().astype(str).tolist()),
        "Time-point q-values are FDR corrections across all GVS-by-time-point tests.",
        "",
        "Input arrays:",
    ]
    if input_note is not None:
        lines.insert(3, f"Input source: {input_note}")
        lines.insert(4, "")
    for code, trials in sorted(trials_by_code.items()):
        lines.append(f"- {code}: shape={trials.shape}")

    lines.extend(["", "Feature tests passing q < 0.05:"])
    significant = feature_stats[feature_stats["q_perm_fdr"] < 0.05]
    if significant.empty:
        lines.append("- None")
    else:
        for _, row in significant.sort_values(["active_gvs", "q_perm_fdr", "feature"]).iterrows():
            lines.append(
                f"- {row['active_gvs']} {row['label']}: mean active-sham={row['mean_active_minus_sham']:.6g}, "
                f"95% CI [{row['mean_diff_ci95_low']:.6g}, {row['mean_diff_ci95_high']:.6g}], "
                f"p_perm={row['p_perm']:.6g}, q={row['q_perm_fdr']:.6g}, dz={row['cohen_dz']:.3g}"
            )

    lines.extend(["", "Best feature result per active GVS condition:"])
    for active_gvs, group in feature_stats.groupby("active_gvs", sort=True):
        best = group.sort_values(["q_perm_fdr", "p_perm"]).iloc[0]
        n_subjects = int(subject_pairs.loc[subject_pairs["active_gvs"].eq(active_gvs), "subject"].nunique())
        lines.append(
            f"- {active_gvs}: {best['label']}, mean active-sham={best['mean_active_minus_sham']:.6g}, "
            f"p_perm={best['p_perm']:.6g}, q={best['q_perm_fdr']:.6g}, n_subjects={n_subjects}"
        )

    lines.extend(["", "Time points passing q < 0.05:"])
    significant_time = timepoint_stats[timepoint_stats["q_perm_fdr"] < 0.05]
    if significant_time.empty:
        lines.append("- None")
    else:
        for _, row in significant_time.sort_values(["active_gvs", "time_index"]).iterrows():
            lines.append(
                f"- {row['active_gvs']} time_index {int(row['time_index'])}: "
                f"mean active-sham={row['mean_active_minus_sham']:.6g}, q={row['q_perm_fdr']:.6g}"
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_med_analysis(args: argparse.Namespace) -> None:
    if args.active_bold_group is not None and args.task_map_bold_trials is not None:
        raise ValueError("--active-bold-group and --task-map-bold-trials are mutually exclusive medication input sources")

    if args.active_bold_group is not None:
        full_trials, projection_metadata = _weighted_html_projection_from_active_bold(
            args.active_bold_group,
            args.active_flat_indices,
            args.projection_weight_map,
            args.projection_html_mask,
            args.projection_voxel_chunk_size,
        )
        med_off = _session_trials_from_full_trials(full_trials, args.manifest, session=1)
        med_on = _session_trials_from_full_trials(full_trials, args.manifest, session=2)
        signal_label = "HTML-mask weighted BOLD"
        input_note = (
            f"{projection_metadata['active_bold']}; weights={projection_metadata['weight_map']}; "
            f"html_mask={projection_metadata['html_mask']}; "
            f"selected_active_voxels={projection_metadata['selected_active_voxels']}/"
            f"{projection_metadata['html_selected_voxels']}"
        )
    elif args.task_map_bold_trials is None:
        med_off = _load_trials(args.med_off)
        med_on = _load_trials(args.med_on)
        signal_label = "projected BOLD"
        input_note = f"OFF={args.med_off}; ON={args.med_on}"
    else:
        full_trials = _reduce_task_map_bold_trials(
            args.task_map_bold_trials,
            args.manifest,
            args.task_map_reducer,
            args.task_map_chunk_size,
        )
        med_off = _session_trials_from_full_trials(full_trials, args.manifest, session=1)
        med_on = _session_trials_from_full_trials(full_trials, args.manifest, session=2)
        signal_label = "task-map voxel-mean BOLD"
        input_note = (
            f"{args.task_map_bold_trials}; reducer=voxel {args.task_map_reducer}; "
            f"session split from {args.manifest}"
        )
    if med_off.shape[1] != med_on.shape[1]:
        raise ValueError(f"OFF and ON arrays must have the same number of time points: {med_off.shape}, {med_on.shape}")

    off_labels = _trial_labels_from_manifest(args.manifest, session=1, expected_rows=med_off.shape[0])
    on_labels = _trial_labels_from_manifest(args.manifest, session=2, expected_rows=med_on.shape[0])

    trial_features = _make_trial_frame(med_off, med_on, off_labels, on_labels)
    all_subject_features = _paired_subject_features(trial_features)
    all_feature_stats = _feature_stats(all_subject_features, args.n_bootstrap, args.random_state)
    subject_features = _paired_subject_features(trial_features, MED_TEST_FEATURE_NAMES)
    feature_stats = _feature_stats(subject_features, args.n_bootstrap, args.random_state, MED_TEST_FEATURE_NAMES)
    off_subject, on_subject = _subject_timecourses(med_off, med_on, off_labels, on_labels)
    timepoint_stats = _timepoint_stats(off_subject, on_subject)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    subject_features.to_csv(args.out_dir / "subject_signal_features.csv", index=False)
    feature_stats.to_csv(args.out_dir / "paired_signal_feature_stats.csv", index=False)
    timepoint_stats.to_csv(args.out_dir / "paired_timepoint_stats.csv", index=False)
    _plot_feature_spaghetti(
        subject_features,
        feature_stats,
        args.out_dir / "signal_feature_spaghetti.png",
        signal_label=signal_label,
        features=MED_TEST_FEATURE_NAMES,
    )
    _plot_feature_spaghetti(
        all_subject_features,
        all_feature_stats,
        args.out_dir / "signal_feature_spaghetti_all_features.png",
        signal_label=signal_label,
    )
    all_significant_feature_names = [
        feature
        for feature in FEATURE_NAMES
        if feature in set(all_feature_stats.loc[all_feature_stats["q_perm_fdr"].lt(0.05), "feature"])
    ]
    if all_significant_feature_names:
        _plot_feature_spaghetti(
            all_subject_features,
            all_feature_stats,
            args.out_dir / "signal_feature_spaghetti_all_features_significant.png",
            signal_label=signal_label,
            features=all_significant_feature_names,
        )
    significant_feature_names = [
        feature
        for feature in MED_TEST_FEATURE_NAMES
        if feature in set(feature_stats.loc[feature_stats["q_perm_fdr"].lt(0.05), "feature"])
    ]
    if significant_feature_names:
        _plot_feature_spaghetti(
            subject_features,
            feature_stats,
            args.out_dir / "signal_feature_spaghetti_significant.png",
            signal_label=signal_label,
            features=significant_feature_names,
        )
    _plot_timecourses(
        off_subject,
        on_subject,
        timepoint_stats,
        args.out_dir / "signal_timecourse_spaghetti.png",
        random_state=args.random_state,
        n_bootstrap=args.n_bootstrap,
        signal_label=signal_label,
    )
    _write_report(
        args.out_dir / "med_on_off_signal_feature_report.txt",
        med_off,
        med_on,
        subject_features,
        feature_stats,
        timepoint_stats,
        signal_label=signal_label,
        input_note=input_note,
    )

    n_subjects = (
        subject_features.pivot(index="subject", columns="medication", values="n_trials")
        .dropna()
        .shape[0]
    )
    if args.active_bold_group is not None:
        print(
            f"Loaded {args.active_bold_group}, applied HTML-mask weighted projection, "
            f"and split it into OFF {med_off.shape} and ON {med_on.shape} trial-by-time arrays using {args.manifest}."
        )
    elif args.task_map_bold_trials is None:
        print(f"Loaded OFF {med_off.shape} and ON {med_on.shape} trial-by-time arrays.")
    else:
        print(
            f"Loaded {args.task_map_bold_trials} and reduced it with voxel {args.task_map_reducer} "
            f"to OFF {med_off.shape} and ON {med_on.shape} trial-by-time arrays."
        )
    print(f"Using {n_subjects} paired subjects as the statistical unit.")
    print()
    print("Medication feature tests, sorted by FDR q-value:")
    columns = ["label", "mean_on_minus_off", "mean_diff_ci95_low", "mean_diff_ci95_high", "p_perm", "q_perm_fdr"]
    print(feature_stats[columns].to_string(index=False, float_format=lambda value: f"{value:.6g}"))
    print()
    print(f"Wrote medication outputs to {args.out_dir}")


def _run_gvs_analysis(args: argparse.Namespace) -> None:
    if args.active_bold_group is not None and args.task_map_bold_trials is not None:
        raise ValueError("--active-bold-group and --task-map-bold-trials are mutually exclusive GVS input sources")

    if args.active_bold_group is not None:
        full_trials, projection_metadata = _weighted_html_projection_from_active_bold(
            args.active_bold_group,
            args.active_flat_indices,
            args.projection_weight_map,
            args.projection_html_mask,
            args.projection_voxel_chunk_size,
        )
        trials_by_code = _gvs_trials_from_full_trials(full_trials, args.gvs_dir)
        feature_frame, signal_frame, trials_by_code = _make_gvs_frames(args.gvs_dir, trials_by_code=trials_by_code)
        signal_label = "HTML-mask weighted BOLD"
        input_note = (
            f"{projection_metadata['active_bold']}; weights={projection_metadata['weight_map']}; "
            f"html_mask={projection_metadata['html_mask']}; "
            f"selected_active_voxels={projection_metadata['selected_active_voxels']}/"
            f"{projection_metadata['html_selected_voxels']}"
        )
    elif args.task_map_bold_trials is None:
        feature_frame, signal_frame, trials_by_code = _make_gvs_frames(args.gvs_dir)
        signal_label = "projected BOLD"
        input_note = f"{args.gvs_dir}"
    else:
        full_trials = _reduce_task_map_bold_trials(
            args.task_map_bold_trials,
            args.manifest,
            args.task_map_reducer,
            args.task_map_chunk_size,
        )
        trials_by_code = _gvs_trials_from_full_trials(full_trials, args.gvs_dir)
        feature_frame, signal_frame, trials_by_code = _make_gvs_frames(args.gvs_dir, trials_by_code=trials_by_code)
        signal_label = "task-map voxel-mean BOLD"
        input_note = (
            f"{args.task_map_bold_trials}; reducer=voxel {args.task_map_reducer}; "
            f"GVS split from {args.gvs_dir / 'gvs_projection_trial_metadata.tsv'}"
        )
    if args.gvs_sham not in trials_by_code:
        raise ValueError(f"Sham GVS code {args.gvs_sham} was not found in {args.gvs_dir}")

    run_features = _gvs_run_feature_means(feature_frame)
    all_subject_pairs = _gvs_subject_feature_pairs(run_features, args.gvs_sham)
    subject_pairs = _gvs_subject_feature_pairs(run_features, args.gvs_sham, GVS_TEST_FEATURE_NAMES)
    all_feature_stats = _gvs_feature_stats(all_subject_pairs, args.n_bootstrap, args.random_state)
    feature_stats = _gvs_feature_stats(subject_pairs, args.n_bootstrap, args.random_state)
    all_subject_run_feature_stats = _gvs_subject_run_feature_stats(run_features, args.gvs_sham)
    subject_run_feature_stats = _gvs_subject_run_feature_stats(run_features, args.gvs_sham, GVS_TEST_FEATURE_NAMES)
    timecourse_pairs = _gvs_subject_timecourse_pairs(signal_frame, args.gvs_sham)
    run_timecourse_pairs = _gvs_run_timecourse_pairs(signal_frame, args.gvs_sham)
    timepoint_stats = _gvs_timepoint_stats(timecourse_pairs)
    rt_condition_orders = _gvs_condition_order_by_subject_from_rt(
        args.gvs_rt_run_metrics,
        args.gvs_rt_inventory,
        args.gvs_rt_behaviour_column,
    )

    args.gvs_out_dir.mkdir(parents=True, exist_ok=True)
    feature_frame.to_csv(args.gvs_out_dir / "gvs_trial_signal_features.csv", index=False)
    run_features.to_csv(args.gvs_out_dir / "gvs_run_signal_features.csv", index=False)
    subject_pairs.to_csv(args.gvs_out_dir / "gvs_vs_sham_subject_signal_feature_pairs.csv", index=False)
    all_subject_pairs.to_csv(args.gvs_out_dir / "gvs_vs_sham_subject_signal_feature_pairs_all_features.csv", index=False)
    feature_stats.to_csv(args.gvs_out_dir / "gvs_vs_sham_signal_feature_stats.csv", index=False)
    all_feature_stats.to_csv(args.gvs_out_dir / "gvs_vs_sham_signal_feature_stats_all_features.csv", index=False)
    subject_run_feature_stats.to_csv(args.gvs_out_dir / "gvs_vs_sham_subject_runlevel_feature_stats.csv", index=False)
    all_subject_run_feature_stats.to_csv(
        args.gvs_out_dir / "gvs_vs_sham_subject_runlevel_feature_stats_all_features.csv",
        index=False,
    )
    timecourse_pairs.to_csv(args.gvs_out_dir / "gvs_vs_sham_subject_timecourse_pairs.csv", index=False)
    timepoint_stats.to_csv(args.gvs_out_dir / "gvs_vs_sham_timepoint_stats.csv", index=False)

    active_codes = sorted(code for code in trials_by_code if code != args.gvs_sham)
    for active_code in active_codes:
        safe_code = _safe_name(active_code)
        _plot_gvs_feature_spaghetti(
            subject_pairs,
            feature_stats,
            active_code,
            args.gvs_sham,
            args.gvs_out_dir / f"signal_feature_spaghetti_{safe_code}_vs_sham.png",
            signal_label=signal_label,
            features=GVS_TEST_FEATURE_NAMES,
        )
        _plot_gvs_feature_spaghetti(
            all_subject_pairs,
            all_feature_stats,
            active_code,
            args.gvs_sham,
            args.gvs_out_dir / f"signal_feature_spaghetti_{safe_code}_vs_sham_all_features.png",
            signal_label=signal_label,
        )
        _plot_gvs_timecourse_spaghetti(
            timecourse_pairs,
            timepoint_stats,
            active_code,
            args.gvs_sham,
            args.gvs_out_dir / f"signal_timecourse_spaghetti_{safe_code}_vs_sham.png",
            random_state=args.random_state,
            n_bootstrap=args.n_bootstrap,
            signal_label=signal_label,
        )

    _plot_gvs_timecourse_delta_grid(
        timecourse_pairs,
        active_codes,
        args.gvs_out_dir / "second_subplots_gvs_02_to_09_vs_sham.png",
        random_state=args.random_state,
        n_bootstrap=args.n_bootstrap,
    )
    _plot_gvs_feature_q_heatmap(
        feature_stats,
        args.gvs_out_dir / "gvs_vs_sham_feature_q_heatmap.png",
        title=None,
        y_tick_labels=[f"gvs{idx + 1}" for idx in range(len(active_codes))],
        features=GVS_TEST_FEATURE_NAMES,
    )
    _plot_gvs_feature_q_heatmap(
        all_feature_stats,
        args.gvs_out_dir / "gvs_vs_sham_feature_q_heatmap_all_features.png",
        title=None,
        y_tick_labels=[f"gvs{idx + 1}" for idx in range(len(active_codes))],
    )
    subject_heatmap_dir = args.gvs_out_dir / "subject_feature_q_heatmaps"
    for subject, subject_stats in subject_run_feature_stats.groupby("subject", sort=True):
        safe_subject = _safe_name(str(subject))
        _plot_gvs_feature_q_heatmap(
            subject_stats,
            subject_heatmap_dir / f"gvs_vs_sham_feature_q_heatmap_{safe_subject}.png",
            title=f"{subject} run-paired GVS vs sham feature evidence (-log10 FDR q)",
            features=GVS_TEST_FEATURE_NAMES,
        )
    for subject, subject_stats in all_subject_run_feature_stats.groupby("subject", sort=True):
        safe_subject = _safe_name(str(subject))
        _plot_gvs_feature_q_heatmap(
            subject_stats,
            subject_heatmap_dir / f"gvs_vs_sham_feature_q_heatmap_{safe_subject}_all_features.png",
            title=f"{subject} run-paired GVS vs sham feature evidence (-log10 FDR q)",
        )
    for subject, subject_timecourse_pairs in run_timecourse_pairs.groupby("subject", sort=True):
        safe_subject = _safe_name(str(subject))
        _plot_gvs_timecourse_original_grid(
            subject_timecourse_pairs,
            active_codes,
            args.gvs_sham,
            subject_heatmap_dir / f"second_subplots_gvs_02_to_09_vs_sham_{safe_subject}.png",
            random_state=args.random_state,
            n_bootstrap=args.n_bootstrap,
            show_ci=True,
            change_from_t0=False,
        )
        _plot_gvs_timecourse_original_grid(
            subject_timecourse_pairs,
            active_codes,
            args.gvs_sham,
            subject_heatmap_dir / f"second_subplots_gvs_01_to_09_original_signal_no_ci_{safe_subject}.png",
            random_state=args.random_state,
            n_bootstrap=args.n_bootstrap,
            show_ci=False,
            change_from_t0=False,
            condition_order=rt_condition_orders.get(str(subject)),
        )
    _write_gvs_report(
        args.gvs_out_dir / "gvs_vs_sham_signal_feature_report.txt",
        trials_by_code,
        subject_pairs,
        feature_stats,
        timepoint_stats,
        args.gvs_sham,
        signal_label=signal_label,
        input_note=input_note,
    )

    n_subjects = int(subject_pairs["subject"].nunique())
    if args.active_bold_group is not None:
        print(
            f"Loaded {args.active_bold_group}, applied HTML-mask weighted projection, "
            f"and split it into {len(trials_by_code)} GVS trial-by-time arrays using {args.gvs_dir} metadata."
        )
    elif args.task_map_bold_trials is None:
        print(f"Loaded {len(trials_by_code)} GVS trial-by-time arrays from {args.gvs_dir}.")
    else:
        print(
            f"Loaded {args.task_map_bold_trials}, reduced it with voxel {args.task_map_reducer}, "
            f"and split it into {len(trials_by_code)} GVS trial-by-time arrays using {args.gvs_dir} metadata."
        )
    print(f"Using {n_subjects} subjects; active GVS blocks are paired to {args.gvs_sham} within subject/session/run.")
    print()
    print("Best feature result per active GVS condition:")
    best_rows = feature_stats.sort_values(["active_gvs", "q_perm_fdr", "p_perm"]).groupby("active_gvs", as_index=False).first()
    columns = ["active_gvs", "label", "mean_active_minus_sham", "mean_diff_ci95_low", "mean_diff_ci95_high", "p_perm", "q_perm_fdr"]
    print(best_rows[columns].to_string(index=False, float_format=lambda value: f"{value:.6g}"))
    print()
    significant = feature_stats[feature_stats["q_perm_fdr"] < 0.05]
    print(f"Feature tests passing global FDR q < 0.05: {significant.shape[0]}")
    print(f"Wrote {subject_run_feature_stats['subject'].nunique()} per-subject feature q heatmaps to {subject_heatmap_dir}")
    print(f"Wrote {run_timecourse_pairs['subject'].nunique()} per-subject raw CI and raw no-CI grids to {subject_heatmap_dir}")
    print(f"Wrote GVS outputs to {args.gvs_out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare projected BOLD signal features between medication states or between active GVS and sham. "
            "Inference is paired at the subject level, not at the pooled-trial level."
        )
    )
    parser.add_argument("--analysis", choices=("med", "gvs", "both"), default="med")
    parser.add_argument("--med-off", type=Path, default=DEFAULT_MED_OFF)
    parser.add_argument("--med-on", type=Path, default=DEFAULT_MED_ON)
    parser.add_argument(
        "--task-map-bold-trials",
        type=Path,
        default=None,
        help=(
            "Optional 3D voxel-by-trial-by-time task-map BOLD array. "
            "When provided, session OFF/ON arrays are derived from it using manifest offsets."
        ),
    )
    parser.add_argument("--task-map-reducer", choices=("mean",), default="mean")
    parser.add_argument("--task-map-chunk-size", type=int, default=512)
    parser.add_argument(
        "--active-bold-group",
        type=Path,
        default=None,
        help=(
            "Optional 3D voxel-by-trial-by-time active BOLD group array. "
            "For medication and GVS analyses this recomputes weighted projections using --projection-html-mask."
        ),
    )
    parser.add_argument("--active-flat-indices", type=Path, default=DEFAULT_ACTIVE_FLAT_INDICES)
    parser.add_argument("--projection-weight-map", type=Path, default=DEFAULT_WEIGHT_MAP)
    parser.add_argument("--projection-html-mask", type=Path, default=DEFAULT_WEIGHT_HTML)
    parser.add_argument("--projection-voxel-chunk-size", type=int, default=512)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--gvs-dir", type=Path, default=DEFAULT_GVS_DIR)
    parser.add_argument("--gvs-sham", default=DEFAULT_GVS_SHAM)
    parser.add_argument("--gvs-out-dir", type=Path, default=DEFAULT_GVS_OUT_DIR)
    parser.add_argument("--gvs-rt-run-metrics", type=Path, default=DEFAULT_GVS_RT_RUN_METRICS)
    parser.add_argument("--gvs-rt-inventory", type=Path, default=DEFAULT_GVS_RT_INVENTORY)
    parser.add_argument(
        "--gvs-rt-behaviour-column",
        type=int,
        default=1,
        help="Zero-based RT column for sorting GVS/sham panels in subject original-signal grids.",
    )
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--random-state", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.analysis in {"med", "both"}:
        _run_med_analysis(args)
    if args.analysis == "both":
        print()
    if args.analysis in {"gvs", "both"}:
        _run_gvs_analysis(args)


if __name__ == "__main__":
    main()
