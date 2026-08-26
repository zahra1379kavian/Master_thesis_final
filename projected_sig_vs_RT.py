#!/usr/bin/env python3
"""Compare adjacent-trial variability in weighted projection and behaviour RT."""

from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path

import matplotlib

warnings.filterwarnings("ignore", message="Unable to import Axes3D.*")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text
import nibabel as nib
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = Path(
    "/mnt/TeamShare/Data_Masterfile/H20-00572_All-Dressed/Zahra-Thesis-Data/fmri_opt_group/results_beta_preprocessed"
)
DEFAULT_BEHAVIOUR_DIR = Path(
    "/mnt/TeamShare/Data_Masterfile/H20-00572_All-Dressed/Zahra-Thesis-Data/fmri_opt_group/behaviour"
)
DEFAULT_WEIGHT_MAP = ROOT / "data" / "voxel_weights_task1_bold0.6_beta0.6_smooth1.25_gamma1.5.nii.gz"
DEFAULT_OUT_DIR = ROOT / "figures" / "projected_RT"
DEFAULT_FIGURE_STEM = "projection_behavior_subject_panel(main)"
DEFAULT_CORRELATION_FIGURE_STEM = "projection_behavior_subject_correlation_histogram"
SESSION_FIGURE_SPECS = {
    1: "medication_off",
    2: "medication_on",
}
PROJECTION_TRIAL_CHUNK_SIZE = 8
PROJECTION_VOXEL_CHUNK_SIZE = 4096
VARIABILITY_AXIS_LABEL = "Consecutive-trial variability"
BEHAVIOUR_COLOR = "#0072B2"
PROJECTION_COLOR = "#D55E00"
DIFFERENCE_COLOR = "#009E73"
SUBJECT_MARKER_SIZE = 15
MEAN_MARKER = "D"
PAPER_FONT_FAMILY = "Liberation Sans"
TITLE_FONT_SIZE = 18
TAKEAWAY_SUBTITLE_FONT_SIZE = 13
AXIS_TICK_FONT_SIZE = 13
CELL_VALUE_FONT_SIZE = 12
FOOTER_NOTE_FONT_SIZE = 11

BETA_RE = re.compile(
    r"cleaned_beta_volume_(?P<sub>sub-pd\d+)_ses-(?P<ses>\d+)_run-(?P<run>\d+)\.npy$"
)
ACTIVE_BOLD_RE = re.compile(
    r"active_bold_(?P<sub>sub-pd\d+)_ses-(?P<ses>\d+)_run-(?P<run>\d+)\.npy\.npy$"
)
BEHAVIOUR_RE = re.compile(r"PSPD(?P<digits>\d+)_ses_(?P<ses>\d+)_run_(?P<run>\d+)\.npy$")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Project BOLD or beta data through a voxel-weight map, then compare "
            "adjacent-trial variability with behaviour RT."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--behaviour-dir", type=Path, default=DEFAULT_BEHAVIOUR_DIR)
    parser.add_argument("--weight-map", type=Path, default=DEFAULT_WEIGHT_MAP)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--projection-source",
        choices=("bold", "beta"),
        default="bold",
        help="Use active BOLD trial time series or cleaned beta volumes for the weighted projection.",
    )
    parser.add_argument(
        "--bold-trial-reducer",
        choices=("median", "mean"),
        default="median",
        help="Reducer applied across each trial's BOLD time points after projection.",
    )
    parser.add_argument(
        "--behaviour-column",
        type=int,
        default=1,
        help="Zero-based RT column for 2D behaviour arrays; default 1 uses the second column.",
    )
    parser.add_argument(
        "--save-session-figures",
        action="store_true",
        help="Also save separate session 1 medication-off and session 2 medication-on figures.",
    )
    return parser.parse_args()


def _subject_digits(sub_tag: str) -> str:
    match = re.search(r"(\d+)$", sub_tag)
    if not match:
        raise ValueError(f"Could not parse subject digits from {sub_tag!r}")
    return match.group(1)


def _category_sort_key(value: str) -> tuple[int, int | str]:
    digits = _subject_digits(str(value))
    return (0, int(digits)) if digits.isdigit() else (1, str(value))


def _load_weights(weight_map: Path) -> np.ndarray:
    weights = nib.load(str(weight_map)).get_fdata(dtype=np.float32)
    mask = np.isfinite(weights) & (weights != 0)
    if not np.any(mask):
        raise ValueError(f"No nonzero finite weights found in {weight_map}")
    return weights


def _load_behaviour_rt(path: Path, column: int) -> np.ndarray:
    behaviour = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)
    if behaviour.ndim == 1:
        return behaviour
    if behaviour.ndim != 2:
        raise ValueError(f"Expected 1D or 2D behaviour array in {path}, got shape {behaviour.shape}")
    if column >= behaviour.shape[1]:
        raise ValueError(f"Behaviour column {column} is not available in {path} with shape {behaviour.shape}")
    return behaviour[:, column]


def _align_trials(projected_signal: np.ndarray, behaviour_rt: np.ndarray, label: str) -> tuple[np.ndarray, np.ndarray]:
    n_projection = projected_signal.shape[0]
    n_behaviour = behaviour_rt.shape[0]
    if n_projection == n_behaviour:
        return projected_signal, behaviour_rt

    n_keep = min(n_projection, n_behaviour)
    warnings.warn(
        f"{label}: projection has {n_projection} trials and behaviour has {n_behaviour}; "
        f"truncating both to {n_keep}.",
        stacklevel=2,
    )
    return projected_signal[:n_keep], behaviour_rt[:n_keep]


def _adjacent_diff_ratio_sum(values: np.ndarray) -> tuple[float, int]:
    values = np.asarray(values, dtype=np.float64)
    if values.size < 2:
        return np.nan, 0
    finite_values = np.isfinite(values)
    if not np.any(finite_values):
        return np.nan, 0
    values = values.copy()
    values[finite_values] -= np.nanmean(values[finite_values])
    x0 = values[:-1]
    x1 = values[1:]
    denominator = x0 * x0 + x1 * x1
    keep = np.isfinite(x0) & np.isfinite(x1) & np.isfinite(denominator) & (denominator > 0)
    if not np.any(keep):
        return np.nan, 0
    score = np.sum(((x0[keep] - x1[keep]) ** 2) / denominator[keep])
    return float(score), int(np.count_nonzero(keep))


def _project_beta(beta_path: Path, weights: np.ndarray) -> np.ndarray:
    beta = np.load(beta_path, mmap_mode="r")
    if beta.ndim != 4:
        raise ValueError(f"Expected 4D beta volume in {beta_path}, got shape {beta.shape}")
    if beta.shape[:3] != weights.shape:
        raise ValueError(f"Spatial shape mismatch for {beta_path}: beta {beta.shape[:3]} vs weights {weights.shape}")

    weight_mask = np.isfinite(weights) & (weights != 0)
    selected_weights = weights[weight_mask].astype(np.float64)
    projected_signal = np.full(beta.shape[3], np.nan, dtype=np.float64)

    for start in range(0, beta.shape[3], PROJECTION_TRIAL_CHUNK_SIZE):
        stop = min(start + PROJECTION_TRIAL_CHUNK_SIZE, beta.shape[3])
        selected_beta = np.asarray(beta[weight_mask, start:stop], dtype=np.float64)
        finite_beta = np.isfinite(selected_beta)
        filled_beta = np.nan_to_num(selected_beta, nan=0.0, posinf=0.0, neginf=0.0)
        chunk_projection = selected_weights @ filled_beta
        chunk_projection[~np.any(finite_beta, axis=0)] = np.nan
        projected_signal[start:stop] = chunk_projection

    return projected_signal


def _active_flat_indices_path(data_dir: Path, sub: str, ses: int, run: int) -> Path:
    return data_dir / sub / f"active_flat_indices__{sub}_ses-{ses}_run-{run}.npy"


def _reduce_bold_trials(projected_timepoints: np.ndarray, reducer: str) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        if reducer == "median":
            return np.nanmedian(projected_timepoints, axis=1)
        if reducer == "mean":
            return np.nanmean(projected_timepoints, axis=1)
    raise ValueError(f"Unknown BOLD trial reducer: {reducer}")


def _project_bold(active_bold_path: Path, active_flat_indices_path: Path, weights: np.ndarray, reducer: str) -> np.ndarray:
    active_bold = np.load(active_bold_path, mmap_mode="r")
    if active_bold.ndim != 3:
        raise ValueError(f"Expected 3D active BOLD array in {active_bold_path}, got shape {active_bold.shape}")

    flat_indices = np.asarray(np.load(active_flat_indices_path, allow_pickle=False), dtype=np.int64).ravel()
    if flat_indices.size != active_bold.shape[0]:
        raise ValueError(
            f"Active index count mismatch for {active_bold_path}: "
            f"{flat_indices.size} indices vs {active_bold.shape[0]} BOLD voxels"
        )

    flat_weights = weights.ravel()[flat_indices].astype(np.float64)
    n_voxels, n_trials, trial_length = active_bold.shape
    n_timepoints = n_trials * trial_length
    projection = np.zeros(n_timepoints, dtype=np.float64)
    any_finite = np.zeros(n_timepoints, dtype=bool)

    for start in range(0, n_voxels, PROJECTION_VOXEL_CHUNK_SIZE):
        stop = min(start + PROJECTION_VOXEL_CHUNK_SIZE, n_voxels)
        chunk_weights = flat_weights[start:stop]
        weight_keep = np.isfinite(chunk_weights) & (chunk_weights != 0)
        if not np.any(weight_keep):
            continue

        chunk = np.asarray(active_bold[start:stop][weight_keep].reshape(np.count_nonzero(weight_keep), -1), dtype=np.float64)
        finite = np.isfinite(chunk)
        finite_counts = np.count_nonzero(finite, axis=1, keepdims=True)
        sums = np.sum(np.where(finite, chunk, 0.0), axis=1, keepdims=True)
        means = np.divide(sums, finite_counts, out=np.zeros_like(sums), where=finite_counts > 0)
        centered = np.where(finite, chunk - means, 0.0)

        projection += chunk_weights[weight_keep] @ centered
        any_finite |= np.any(finite, axis=0)

    projection[~any_finite] = np.nan
    projected_trials = projection.reshape(n_trials, trial_length)
    return _reduce_bold_trials(projected_trials, reducer)


def _discover_beta_runs(data_dir: Path) -> list[dict[str, object]]:
    runs = []
    for path in sorted(data_dir.glob("sub-*/cleaned_beta_volume_sub-pd*_ses-*_run-*.npy")):
        match = BETA_RE.match(path.name)
        if match:
            runs.append({"path": path, "sub": match.group("sub"), "ses": int(match.group("ses")), "run": int(match.group("run"))})
    return runs


def _discover_bold_runs(data_dir: Path) -> list[dict[str, object]]:
    runs = []
    for path in sorted(data_dir.glob("sub-*/active_bold_sub-pd*_ses-*_run-*.npy.npy")):
        match = ACTIVE_BOLD_RE.match(path.name)
        if match:
            runs.append({"path": path, "sub": match.group("sub"), "ses": int(match.group("ses")), "run": int(match.group("run"))})
    return runs


def _discover_runs(data_dir: Path, projection_source: str) -> list[dict[str, object]]:
    if projection_source == "bold":
        runs = _discover_bold_runs(data_dir)
        label = "active BOLD"
    elif projection_source == "beta":
        runs = _discover_beta_runs(data_dir)
        label = "clean beta"
    else:
        raise ValueError(f"Unknown projection source: {projection_source}")
    if not runs:
        raise FileNotFoundError(f"No {label} files found under {data_dir}")
    return runs


def _behaviour_path(behaviour_dir: Path, sub: str, ses: int, run: int) -> Path:
    return behaviour_dir / f"PSPD{_subject_digits(sub)}_ses_{ses}_run_{run}.npy"


def _warn_unmatched_behaviour_runs(behaviour_dir: Path, projection_runs: list[dict[str, object]]) -> None:
    projection_keys = {
        (_subject_digits(str(item["sub"])), int(item["ses"]), int(item["run"]))
        for item in projection_runs
    }
    projection_subjects = {key[0] for key in projection_keys}
    unmatched = []
    for path in sorted(behaviour_dir.glob("PSPD*_ses_*_run_*.npy")):
        match = BEHAVIOUR_RE.match(path.name)
        if not match:
            continue
        key = (match.group("digits"), int(match.group("ses")), int(match.group("run")))
        if key[0] in projection_subjects and key not in projection_keys:
            unmatched.append(path.name)
    if unmatched:
        print("Behaviour files without matching projection data:")
        for name in unmatched:
            print(f"- {name}")


def _pearsonr(x_values: np.ndarray, y_values: np.ndarray) -> tuple[float, int]:
    x_values = np.asarray(x_values, dtype=np.float64)
    y_values = np.asarray(y_values, dtype=np.float64)
    keep = np.isfinite(x_values) & np.isfinite(y_values)
    n_values = int(np.count_nonzero(keep))
    if n_values < 3:
        return np.nan, n_values

    x_keep = x_values[keep]
    y_keep = y_values[keep]
    if np.ptp(x_keep) == 0 or np.ptp(y_keep) == 0:
        return np.nan, n_values

    result = stats.pearsonr(x_keep, y_keep)
    return float(result.statistic), n_values


def _build_run_metric_table(
    data_dir: Path,
    behaviour_dir: Path,
    weights: np.ndarray,
    projection_source: str,
    behaviour_column: int,
    bold_trial_reducer: str,
    include_subjects: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    runs = _discover_runs(data_dir, projection_source)
    if include_subjects is not None:
        runs = [item for item in runs if str(item["sub"]) in include_subjects]
        found_subjects = {str(item["sub"]) for item in runs}
        missing_subjects = sorted(include_subjects - found_subjects, key=_category_sort_key)
        if missing_subjects:
            raise FileNotFoundError(
                "No projection runs found for requested subjects: " + ", ".join(missing_subjects)
            )
    _warn_unmatched_behaviour_runs(behaviour_dir, runs)

    missing = []
    for item in runs:
        sub = str(item["sub"])
        ses = int(item["ses"])
        run = int(item["run"])
        behaviour_path = _behaviour_path(behaviour_dir, sub, ses, run)
        if not behaviour_path.exists():
            missing.append(f"{sub} ses-{ses} run-{run}: {behaviour_path}")
        if projection_source == "bold":
            active_flat_path = _active_flat_indices_path(data_dir, sub, ses, run)
            if not active_flat_path.exists():
                missing.append(f"{sub} ses-{ses} run-{run}: {active_flat_path}")
    if missing:
        missing_lines = "\n".join(f"- {item}" for item in missing)
        raise FileNotFoundError(f"Missing datasets:\n{missing_lines}")

    rows = []
    subject_projection_values: dict[str, list[np.ndarray]] = {}
    subject_behaviour_values: dict[str, list[np.ndarray]] = {}
    subject_run_keys: dict[str, set[tuple[int, int]]] = {}
    for item in runs:
        sub = str(item["sub"])
        ses = int(item["ses"])
        run = int(item["run"])
        projection_path = Path(item["path"])
        label = f"{sub} ses-{ses} run-{run}"

        if projection_source == "bold":
            active_flat_path = _active_flat_indices_path(data_dir, sub, ses, run)
            projected_signal = _project_bold(projection_path, active_flat_path, weights, bold_trial_reducer)
        else:
            projected_signal = _project_beta(projection_path, weights)

        behaviour_rt = _load_behaviour_rt(_behaviour_path(behaviour_dir, sub, ses, run), behaviour_column)
        projected_signal, behaviour_rt = _align_trials(projected_signal, behaviour_rt, label)
        paired_finite = np.isfinite(projected_signal) & np.isfinite(behaviour_rt)
        projection_values = projected_signal[paired_finite]
        behaviour_values = behaviour_rt[paired_finite]
        subject_projection_values.setdefault(sub, []).append(projection_values)
        subject_behaviour_values.setdefault(sub, []).append(behaviour_values)
        if projection_values.size:
            subject_run_keys.setdefault(sub, set()).add((ses, run))
        projection_score, projection_pair_count = _adjacent_diff_ratio_sum(projection_values)
        behaviour_score, behaviour_pair_count = _adjacent_diff_ratio_sum(behaviour_values)

        rows.append(
            {
                "sub_tag": sub,
                "ses": ses,
                "run": run,
                "n_trials_paired_finite": int(np.count_nonzero(paired_finite)),
                "adjacent_diff_ratio_sum_projection": projection_score,
                "adjacent_diff_ratio_sum_behavior_col2": behaviour_score,
                "n_adjacent_pairs_projection": projection_pair_count,
                "n_adjacent_pairs_behavior_col2": behaviour_pair_count,
                "variability_preprocessing": "demean_paired_finite_run_values",
                "projection_source": projection_source,
                "projection_path": str(projection_path),
                "behaviour_path": str(_behaviour_path(behaviour_dir, sub, ses, run)),
            }
        )

    if not rows:
        raise ValueError("No projection/behaviour metric rows were created.")

    subject_rows = []
    for sub in sorted(subject_projection_values, key=_category_sort_key):
        projection_values = np.concatenate(subject_projection_values[sub])
        behaviour_values = np.concatenate(subject_behaviour_values[sub])
        correlation, n_values = _pearsonr(projection_values, behaviour_values)
        run_keys = subject_run_keys.get(sub, set())
        subject_rows.append(
            {
                "sub_tag": sub,
                "pearson_r_projection_behaviour": correlation,
                "n_trials_paired_finite": n_values,
                "n_sessions": len({ses for ses, _ in run_keys}),
                "n_runs": len(run_keys),
                "projection_source": projection_source,
                "correlation_input": "paired_finite_undifferenced_trial_values",
                "consecutive_trial_differencing": False,
            }
        )

    run_metric_df = pd.DataFrame(rows).sort_values(["sub_tag", "ses", "run"]).reset_index(drop=True)
    subject_correlation_df = pd.DataFrame(subject_rows)
    return run_metric_df, subject_correlation_df


def _mixedlm_projection_effect(paired_df: pd.DataFrame) -> dict[str, float]:
    row = {
        "lme_coef_projection_minus_behavior": np.nan,
        "lme_z_projection_minus_behavior": np.nan,
        "lme_p_two_sided": np.nan,
    }
    if paired_df["sub_tag"].nunique() < 2:
        return row

    model_df = paired_df.loc[:, ["sub_tag", "behavior_raw", "projection_raw"]].rename(columns={"sub_tag": "subject_id"})
    behavior_long = model_df.loc[:, ["subject_id", "behavior_raw"]].copy()
    behavior_long["signal"] = "Behaviour"
    behavior_long["value"] = behavior_long.pop("behavior_raw")
    projection_long = model_df.loc[:, ["subject_id", "projection_raw"]].copy()
    projection_long["signal"] = "Projection"
    projection_long["value"] = projection_long.pop("projection_raw")
    long_df = pd.concat([behavior_long, projection_long], axis=0, ignore_index=True)
    long_df["signal"] = pd.Categorical(long_df["signal"], categories=["Behaviour", "Projection"], ordered=True)

    fit = None
    for method in ("lbfgs", "powell", "bfgs", "cg", "nm"):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = smf.mixedlm("value ~ signal", data=long_df, groups=long_df["subject_id"], re_formula="1").fit(
                    reml=False, method=method, disp=False
                )
            break
        except Exception:
            fit = None
    if fit is not None:
        coef = "signal[T.Projection]"
        row["lme_coef_projection_minus_behavior"] = float(fit.params.get(coef, np.nan))
        row["lme_z_projection_minus_behavior"] = float(fit.tvalues.get(coef, np.nan))
        row["lme_p_two_sided"] = float(fit.pvalues.get(coef, np.nan))
    return row


def _expanded_limits(values: np.ndarray, pad_fraction: float = 0.08, force_zero: bool = False) -> tuple[float, float]:
    finite_values = np.asarray(values, dtype=np.float64)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return (-1.0, 1.0) if force_zero else (0.0, 1.0)

    low = float(np.min(finite_values))
    high = float(np.max(finite_values))
    if force_zero:
        low = min(low, 0.0)
        high = max(high, 0.0)

    if high <= low:
        pad = 1.0 if high == 0.0 else abs(high) * pad_fraction
    else:
        pad = (high - low) * pad_fraction
    return low - pad, high + pad


def _mean_ci(values: np.ndarray, confidence: float = 0.95) -> tuple[float, float, float]:
    finite_values = np.asarray(values, dtype=np.float64)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return np.nan, np.nan, np.nan

    mean = float(np.mean(finite_values))
    if finite_values.size < 2:
        return mean, np.nan, np.nan

    sem = float(stats.sem(finite_values))
    if not np.isfinite(sem):
        return mean, mean, mean
    half_width = float(stats.t.ppf(0.5 + confidence / 2.0, finite_values.size - 1) * sem)
    return mean, mean - half_width, mean + half_width


def _summarize_subject_correlations(subject_df: pd.DataFrame) -> pd.DataFrame:
    correlations = subject_df["pearson_r_projection_behaviour"].to_numpy(dtype=np.float64)
    correlations = correlations[np.isfinite(correlations)]
    if correlations.size == 0:
        raise ValueError("No finite subject-level projection/behaviour correlations were available.")

    arithmetic_mean, arithmetic_ci_low, arithmetic_ci_high = _mean_ci(correlations)
    clipped_correlations = np.clip(
        correlations,
        np.nextafter(-1.0, 0.0),
        np.nextafter(1.0, 0.0),
    )
    fisher_z = np.arctanh(clipped_correlations)
    fisher_mean_z, fisher_ci_low_z, fisher_ci_high_z = _mean_ci(fisher_z)
    fisher_mean_r = float(np.tanh(fisher_mean_z))
    fisher_ci_low_r = float(np.tanh(fisher_ci_low_z))
    fisher_ci_high_r = float(np.tanh(fisher_ci_high_z))

    t_statistic = np.nan
    p_value = np.nan
    if fisher_z.size > 1:
        test = stats.ttest_1samp(fisher_z, popmean=0.0)
        t_statistic = float(test.statistic)
        p_value = float(test.pvalue)

    return pd.DataFrame(
        [
            {
                "n_subjects_total": int(subject_df.shape[0]),
                "n_subjects_finite": int(correlations.size),
                "arithmetic_mean_pearson_r": arithmetic_mean,
                "arithmetic_mean_ci95_low": arithmetic_ci_low,
                "arithmetic_mean_ci95_high": arithmetic_ci_high,
                "median_pearson_r": float(np.median(correlations)),
                "sd_pearson_r": float(np.std(correlations, ddof=1)) if correlations.size > 1 else np.nan,
                "fisher_z_mean_backtransformed_r": fisher_mean_r,
                "fisher_z_ci95_low_r": fisher_ci_low_r,
                "fisher_z_ci95_high_r": fisher_ci_high_r,
                "fisher_z_one_sample_t": t_statistic,
                "fisher_z_one_sample_p_two_sided": p_value,
                "arithmetic_ci_method": "two-sided t interval across subject Pearson r values",
                "fisher_z_ci_method": "two-sided t interval across Fisher-z subject correlations",
                "pooling_scope": "all paired trials pooled across available runs and sessions within subject",
                "correlation_input": "paired_finite_undifferenced_trial_values",
                "consecutive_trial_differencing": False,
            }
        ]
    )


def _save_subject_correlation_histogram(
    subject_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    out_dir: Path,
    figure_stem: str = DEFAULT_CORRELATION_FIGURE_STEM,
) -> tuple[Path, Path]:
    correlations = subject_df["pearson_r_projection_behaviour"].to_numpy(dtype=np.float64)
    correlations = correlations[np.isfinite(correlations)]
    if correlations.size == 0:
        raise ValueError("No finite subject-level correlations were available for the histogram.")

    summary = summary_df.iloc[0]
    arithmetic_mean = float(summary["arithmetic_mean_pearson_r"])
    ci_low = float(summary["arithmetic_mean_ci95_low"])
    ci_high = float(summary["arithmetic_mean_ci95_high"])
    n_bins = max(5, min(10, int(np.ceil(np.sqrt(correlations.size)))))

    limit_values = np.concatenate([correlations, [0.0, ci_low, ci_high]])
    x_low, x_high = _expanded_limits(limit_values, pad_fraction=0.14)
    x_low = max(-1.0, x_low)
    x_high = min(1.0, x_high)

    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [PAPER_FONT_FAMILY, "Arial", "DejaVu Sans"],
            "font.size": AXIS_TICK_FONT_SIZE,
            "axes.labelsize": AXIS_TICK_FONT_SIZE,
            "axes.titlesize": TITLE_FONT_SIZE,
            "xtick.labelsize": AXIS_TICK_FONT_SIZE,
            "ytick.labelsize": AXIS_TICK_FONT_SIZE,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    ):
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        ax.hist(
            correlations,
            bins=n_bins,
            color="#4C78A8",
            edgecolor="white",
            linewidth=1.0,
            alpha=0.88,
        )
        ax.axvspan(ci_low, ci_high, color="#D55E00", alpha=0.16, label="95% CI of arithmetic mean")
        ax.axvline(arithmetic_mean, color="#D55E00", linewidth=2.0, label=f"Arithmetic mean r = {arithmetic_mean:.3f}")
        ax.axvline(0.0, color="0.25", linestyle=(0, (4, 3)), linewidth=1.2, label="r = 0")
        ax.set_xlim(x_low, x_high)
        ax.set_xlabel("Pearson r (projected signal, behaviour)")
        ax.set_ylabel("Number of subjects")
        ax.set_title("Subject-wise projection–behaviour correlations\nUndifferenced paired trial-level values")
        ax.grid(axis="y", linewidth=0.5, alpha=0.20)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, fontsize=10, loc="upper left")
        ax.text(
            0.98,
            0.96,
            (
                f"n = {correlations.size}\n"
                f"Mean 95% CI [{ci_low:.3f}, {ci_high:.3f}]"
            ),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10.5,
        )
        fig.tight_layout()
        paths = _save_pdf_and_png(fig, out_dir / f"{figure_stem}.pdf", dpi=300)
        plt.close(fig)
        return paths


def _format_p_value(p_value: float) -> str:
    if not np.isfinite(p_value):
        return "p = n/a"
    if p_value < 0.001:
        return "p < 0.001"
    return f"p = {p_value:.3f}"


def _p_value_stars(p_value: float) -> str:
    if not np.isfinite(p_value):
        return "n.s."
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "n.s."


def _significance_label(p_value: float) -> str:
    return f"{_p_value_stars(p_value)} ({_format_p_value(p_value)})"


def _draw_mean_ci(ax: plt.Axes, x: float, values: np.ndarray, color: str, markersize: float = 5.2) -> tuple[float, float, float]:
    mean, ci_low, ci_high = _mean_ci(values)
    yerr = None
    if np.isfinite(ci_low) and np.isfinite(ci_high):
        yerr = np.array([[mean - ci_low], [ci_high - mean]], dtype=np.float64)
    ax.errorbar(
        [x],
        [mean],
        yerr=yerr,
        fmt=MEAN_MARKER,
        markersize=markersize,
        markerfacecolor="white",
        markeredgecolor=color,
        markeredgewidth=1.15,
        ecolor=color,
        elinewidth=1.15,
        capsize=3.2,
        capthick=1.0,
        zorder=5,
    )
    return mean, ci_low, ci_high


def _draw_boxplot(ax: plt.Axes, values: list[np.ndarray], positions: list[float], colors: list[str], width: float) -> None:
    finite_values = [np.asarray(value, dtype=np.float64)[np.isfinite(value)] for value in values]
    box = ax.boxplot(
        finite_values,
        positions=positions,
        widths=width,
        patch_artist=True,
        showfliers=False,
        whis=1.5,
        zorder=2,
    )
    for patch, color in zip(box["boxes"], colors):
        patch.set(facecolor=color, edgecolor=color, alpha=0.18, linewidth=1.0)
    for median in box["medians"]:
        median.set(color="0.10", linewidth=1.2)
    for whisker in box["whiskers"]:
        whisker.set(color="0.30", linewidth=0.9)
    for cap in box["caps"]:
        cap.set(color="0.30", linewidth=0.9)


def _subject_level_pairs(paired_df: pd.DataFrame) -> pd.DataFrame:
    subject_df = (
        paired_df.groupby("sub_tag", as_index=False)
        .agg(
            behavior_raw=("behavior_raw", "mean"),
            projection_raw=("projection_raw", "mean"),
            n_runs=("sub_tag", "size"),
        )
    )
    subject_df["_sort_key"] = subject_df["sub_tag"].astype(str).map(_category_sort_key)
    subject_df = subject_df.sort_values("_sort_key").drop(columns="_sort_key").reset_index(drop=True)
    subject_df["projection_minus_behavior"] = subject_df["projection_raw"] - subject_df["behavior_raw"]
    subject_df["behaviour_minus_projection"] = subject_df["behavior_raw"] - subject_df["projection_raw"]
    return subject_df


def _plot_paired_estimation(
    ax: plt.Axes,
    subject_df: pd.DataFrame,
    y_limits: tuple[float, float],
) -> None:
    plot_df = subject_df.sort_values("behaviour_minus_projection").reset_index(drop=True)
    behavior_values = plot_df["behavior_raw"].to_numpy(dtype=np.float64)
    projection_values = plot_df["projection_raw"].to_numpy(dtype=np.float64)
    jitter = np.linspace(-0.045, 0.045, behavior_values.size) if behavior_values.size > 1 else np.array([0.0])
    x_behavior = jitter
    x_projection = 1.0 + jitter
    _draw_boxplot(
        ax,
        [behavior_values, projection_values],
        [0.0, 1.0],
        [BEHAVIOUR_COLOR, PROJECTION_COLOR],
        width=0.30,
    )
    for x0, x1, y0, y1 in zip(x_behavior, x_projection, behavior_values, projection_values):
        ax.plot([x0, x1], [y0, y1], color="0.55", linewidth=0.55, alpha=0.28, zorder=1)
    ax.scatter(
        x_behavior,
        behavior_values,
        s=SUBJECT_MARKER_SIZE,
        facecolors=BEHAVIOUR_COLOR,
        edgecolors="white",
        linewidths=0.25,
        alpha=0.72,
        zorder=3,
    )
    ax.scatter(
        x_projection,
        projection_values,
        s=SUBJECT_MARKER_SIZE,
        facecolors=PROJECTION_COLOR,
        edgecolors="white",
        linewidths=0.25,
        alpha=0.72,
        zorder=3,
    )

    y_low, y_high = y_limits

    ax.set_xlim(-0.38, 1.38)
    ax.set_ylim((y_low, y_high))
    ax.set_xticks([0.0, 1.0])
    ax.set_xticklabels(["Behaviour", "Projection"])
    ax.tick_params(axis="y", labelleft=True)
    ax.set_ylabel(VARIABILITY_AXIS_LABEL)
    ax.grid(axis="y", linestyle="-", linewidth=0.45, alpha=0.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("0.25")
    ax.spines["bottom"].set_color("0.25")


def _plot_behaviour_minus_projection(ax: plt.Axes, subject_df: pd.DataFrame) -> None:
    reductions = subject_df["behaviour_minus_projection"].to_numpy(dtype=np.float64)
    rng = np.random.default_rng(141)
    x = rng.uniform(-0.055, 0.055, size=reductions.size)
    _draw_boxplot(ax, [reductions], [0.0], [DIFFERENCE_COLOR], width=0.24)
    ax.scatter(
        x,
        reductions,
        s=SUBJECT_MARKER_SIZE,
        facecolors=DIFFERENCE_COLOR,
        alpha=0.74,
        edgecolors="white",
        linewidths=0.25,
        zorder=3,
    )
    y_low, y_high = _expanded_limits(reductions, force_zero=True)
    y_high += (y_high - y_low) * 0.16

    ax.axhline(0.0, color="0.30", linestyle=(0, (4, 2)), linewidth=0.9, zorder=0)
    ax.set_xlim(-0.30, 0.30)
    ax.set_ylim((y_low, y_high))
    ax.set_xticks([0.0])
    ax.set_xticklabels(["Difference"])
    ax.tick_params(axis="y", labelleft=True)
    ax.set_ylabel("Behaviour - Projection\nvariability")
    ax.grid(axis="y", linestyle="-", linewidth=0.45, alpha=0.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("0.25")
    ax.spines["bottom"].set_color("0.25")


def _bold_figure_text(fig: plt.Figure) -> None:
    fig.canvas.draw()
    for text in fig.findobj(match=Text):
        text.set_fontweight("bold")


def _save_pdf_and_png(fig: plt.Figure, pdf_path: Path, dpi: int) -> tuple[Path, Path]:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    png_path = pdf_path.with_suffix(".png")
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    return pdf_path, png_path


def _add_panel_labels(axes: np.ndarray | list[plt.Axes], labels: tuple[str, ...] = ("A", "B")) -> None:
    for ax, label in zip(np.ravel(axes), labels):
        ax.text(
            -0.08,
            1.04,
            label,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=24,
            fontweight="bold",
            color="black",
            clip_on=False,
        )


def _subject_pairs_and_y_limits(metric_df: pd.DataFrame) -> tuple[pd.DataFrame, tuple[float, float]]:
    projection_col = "adjacent_diff_ratio_sum_projection"
    behavior_col = "adjacent_diff_ratio_sum_behavior_col2"
    paired_df = metric_df.loc[np.isfinite(metric_df[projection_col]) & np.isfinite(metric_df[behavior_col])].copy()
    if paired_df.empty:
        raise ValueError("No finite projection/behaviour variability pairs available for plotting.")

    paired_df["projection_raw"] = paired_df[projection_col].to_numpy(dtype=np.float64)
    paired_df["behavior_raw"] = paired_df[behavior_col].to_numpy(dtype=np.float64)
    subject_df = _subject_level_pairs(paired_df)
    finite_values = np.concatenate(
        [subject_df["behavior_raw"].to_numpy(dtype=np.float64), subject_df["projection_raw"].to_numpy(dtype=np.float64)]
    )
    y_limits = _expanded_limits(finite_values)
    return subject_df, y_limits


def _save_behavior_projection_figure(
    metric_df: pd.DataFrame,
    out_dir: Path,
    figure_stem: str = DEFAULT_FIGURE_STEM,
) -> tuple[Path, Path]:
    subject_df, y_limits = _subject_pairs_and_y_limits(metric_df)
    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [PAPER_FONT_FAMILY, "Arial", "DejaVu Sans"],
            "font.size": AXIS_TICK_FONT_SIZE,
            "axes.titlesize": TITLE_FONT_SIZE,
            "figure.titlesize": TITLE_FONT_SIZE,
            "axes.labelsize": AXIS_TICK_FONT_SIZE,
            "xtick.labelsize": AXIS_TICK_FONT_SIZE,
            "ytick.labelsize": AXIS_TICK_FONT_SIZE,
            "legend.fontsize": CELL_VALUE_FONT_SIZE,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    ):
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(8.2, 4.65),
            gridspec_kw={"width_ratios": [1.45, 1.0]},
        )
        _plot_paired_estimation(axes[0], subject_df, y_limits)
        _plot_behaviour_minus_projection(axes[1], subject_df)
        _add_panel_labels(axes)
        _bold_figure_text(fig)
        fig.subplots_adjust(left=0.105, right=0.985, bottom=0.18, top=0.965, wspace=0.25)
        paths = _save_pdf_and_png(fig, out_dir / f"{figure_stem}.pdf", dpi=300)
        plt.close(fig)
        return paths


def _save_session_first_subplot_comparison(metric_df: pd.DataFrame, out_dir: Path) -> tuple[Path, Path]:
    session_specs = [
        (2, "medication_on", "Medication on"),
        (1, "medication_off", "Medication off"),
    ]
    figure_stem = "projection_behavior_subject_panel_first_subplots_session2_medication_on_session1_medication_off"

    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [PAPER_FONT_FAMILY, "Arial", "DejaVu Sans"],
            "font.size": AXIS_TICK_FONT_SIZE,
            "axes.titlesize": TITLE_FONT_SIZE,
            "figure.titlesize": TITLE_FONT_SIZE,
            "axes.labelsize": AXIS_TICK_FONT_SIZE,
            "xtick.labelsize": AXIS_TICK_FONT_SIZE,
            "ytick.labelsize": AXIS_TICK_FONT_SIZE,
            "legend.fontsize": CELL_VALUE_FONT_SIZE,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    ):
        fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.25))
        session_panels: list[tuple[pd.DataFrame, str]] = []
        shared_values = []
        for session, _, title in session_specs:
            session_df = metric_df.loc[metric_df["ses"] == session].copy()
            if session_df.empty:
                raise ValueError(f"No run metrics available for session {session}.")
            subject_df, _ = _subject_pairs_and_y_limits(session_df)
            session_panels.append((subject_df, title))
            shared_values.extend(
                [
                    subject_df["behavior_raw"].to_numpy(dtype=np.float64),
                    subject_df["projection_raw"].to_numpy(dtype=np.float64),
                ]
            )

        shared_y_limits = _expanded_limits(np.concatenate(shared_values))
        for ax, (subject_df, title) in zip(axes, session_panels):
            _plot_paired_estimation(ax, subject_df, shared_y_limits)
            ax.set_title(title, pad=7, fontsize=TITLE_FONT_SIZE - 2)

        axes[1].set_ylabel("")
        _bold_figure_text(fig)
        fig.subplots_adjust(left=0.085, right=0.99, bottom=0.20, top=0.84, wspace=0.08)
        paths = _save_pdf_and_png(fig, out_dir / f"{figure_stem}.pdf", dpi=300)
        plt.close(fig)
        return paths


def main() -> None:
    args = _parse_args()
    weights = _load_weights(args.weight_map)
    metric_df, subject_correlation_df = _build_run_metric_table(
        data_dir=args.data_dir,
        behaviour_dir=args.behaviour_dir,
        weights=weights,
        projection_source=args.projection_source,
        behaviour_column=args.behaviour_column,
        bold_trial_reducer=args.bold_trial_reducer,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.out_dir / "projection_behavior_run_metrics.csv"
    metric_df.to_csv(metrics_path, index=False)
    subject_correlations_path = args.out_dir / "projection_behavior_subject_correlations.csv"
    subject_correlation_df.to_csv(subject_correlations_path, index=False)
    correlation_summary_df = _summarize_subject_correlations(subject_correlation_df)
    correlation_summary_df["projection_source"] = args.projection_source
    correlation_summary_df["bold_trial_reducer"] = args.bold_trial_reducer
    correlation_summary_df["behaviour_column_zero_based"] = args.behaviour_column
    correlation_summary_df["behaviour_dir"] = str(args.behaviour_dir)
    correlation_summary_path = args.out_dir / "projection_behavior_subject_correlation_summary.csv"
    correlation_summary_df.to_csv(correlation_summary_path, index=False)
    correlation_pdf_path, correlation_png_path = _save_subject_correlation_histogram(
        subject_correlation_df,
        correlation_summary_df,
        args.out_dir,
    )
    pdf_path, png_path = _save_behavior_projection_figure(metric_df, args.out_dir)

    print(f"Saved run metrics to {metrics_path}")
    print(f"Saved subject correlations to {subject_correlations_path}")
    print(f"Saved subject correlation summary to {correlation_summary_path}")
    print(f"Saved subject correlation histogram to {correlation_png_path}")
    print(f"Saved subject correlation histogram PDF to {correlation_pdf_path}")
    print(f"Saved paired variability figure to {png_path}")
    print(f"Saved paired variability PDF to {pdf_path}")

    if args.save_session_figures:
        for session, medication_slug in SESSION_FIGURE_SPECS.items():
            session_df = metric_df.loc[metric_df["ses"] == session].copy()
            if session_df.empty:
                warnings.warn(f"No run metrics available for session {session}; skipping session figure.", stacklevel=2)
                continue

            session_metrics_path = args.out_dir / f"projection_behavior_run_metrics_session{session}_{medication_slug}.csv"
            session_df.to_csv(session_metrics_path, index=False)
            session_stem = f"projection_behavior_subject_panel_session{session}_{medication_slug}"
            session_pdf_path, session_png_path = _save_behavior_projection_figure(
                session_df,
                args.out_dir,
                figure_stem=session_stem,
            )
            print(f"Saved session {session} run metrics to {session_metrics_path}")
            print(f"Saved session {session} paired variability figure to {session_png_path}")
            print(f"Saved session {session} paired variability PDF to {session_pdf_path}")

        comparison_pdf_path, comparison_png_path = _save_session_first_subplot_comparison(metric_df, args.out_dir)
        print(f"Saved session first-subplot comparison figure to {comparison_png_path}")
        print(f"Saved session first-subplot comparison PDF to {comparison_pdf_path}")


if __name__ == "__main__":
    main()
