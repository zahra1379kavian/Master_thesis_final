#!/usr/bin/env python3
"""Test whether any feature of the vigour projection tracks MDS-UPDRS III bradykinesia.

The projected signal is the one behind ``figures/gvs_projection_features_vs_sham``:
the HTML-mask weighted projection of ``data/active_bold_group.npy``, all trials of a
subject pooled across the nine GVS/sham conditions.  Every candidate feature is computed
per run so that run-1 vs run-2 agreement gives a test-retest reliability for the feature
itself; a null association on an unreliable feature would be uninformative, so the
reliability column is what makes the negative result interpretable.

Reported per feature: test-retest reliability, Spearman and Pearson correlation with the
state-matched bradykinesia subscore, a Fisher-z confidence interval, and a BH-FDR q-value
across the whole family.  The script also reports the effect sizes the sample can and
cannot rule out.
"""

from __future__ import annotations

import argparse
import os
import re
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-gvs-updrs")
warnings.filterwarnings("ignore", message="Unable to import Axes3D.*")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import spagetti_plot as sp
from analyze_rt_updrs_associations import bh_fdr, load_updrs_bradykinesia


ROOT = Path(__file__).resolve().parent
DEFAULT_WORKBOOK = ROOT / "data" / "subjects_info" / "AllDressed_PD_Participant_Study_Visit_Info.xlsx"
DEFAULT_METADATA = ROOT / "data" / "GVS_projection_BOLD" / "gvs_projection_trial_metadata.tsv"
DEFAULT_OUT_DIR = ROOT / "figures" / "gvs_projection_features_vs_sham"
OUT_STEM = "projection_features_updrs_iii_bradykinesia_null"
CACHE_NAME = "projection_full_trials_cache.npy"
RELIABLE_THRESHOLD = 0.5
MIN_TRIALS_PER_RUN = 20

# family, name, human-readable definition
FEATURE_DEFINITIONS: list[tuple[str, str, str]] = [
    ("Amplitude / level", "level_mean", "Mean of the projected signal over all trials and time points"),
    ("Amplitude / level", "level_auc", "Mean per-trial area under the curve"),
    ("Amplitude / level", "signal_sd_overall", "SD of the concatenated within-run projected time series"),
    ("Amplitude / level", "signal_mad", "Median absolute deviation of the concatenated time series"),
    ("Within-trial shape", "wt_peak_to_peak", "Per-trial max minus min, averaged over trials"),
    ("Within-trial shape", "wt_temporal_sd", "SD across the 9 time points of a trial, averaged over trials"),
    ("Within-trial shape", "wt_slope", "Least-squares slope across the 9 time points, averaged over trials"),
    ("Within-trial shape", "wt_early_late_change", "Last 3 minus first 3 time points, averaged over trials"),
    ("Within-trial shape", "wt_abs_baseline_response", "Max absolute deviation from the trial's first sample"),
    ("Within-trial shape", "wt_baseline_to_peak", "Max rise above the trial's first sample, averaged"),
    ("Within-trial shape", "wt_baseline_to_trough", "Max fall below the trial's first sample, averaged"),
    ("Evoked waveform", "ev_peak_to_peak", "Peak-to-peak of the trial-averaged waveform"),
    ("Evoked waveform", "ev_temporal_sd", "SD over time of the trial-averaged waveform"),
    ("Evoked waveform", "ev_slope", "Least-squares slope of the trial-averaged waveform"),
    ("Evoked waveform", "ev_early_late_change", "Late minus early samples of the trial-averaged waveform"),
    ("Evoked waveform", "ev_peak_latency", "Time point of the largest absolute baseline-referenced deflection"),
    ("Trial-to-trial variability", "tt_sd", "SD across trials of the per-trial mean level"),
    ("Trial-to-trial variability", "tt_mssd", "Consecutive-trial variability (root mean squared successive difference / 2)"),
    ("Trial-to-trial variability", "tt_mad", "Median absolute deviation across trials of the per-trial mean level"),
    ("Trial-to-trial variability", "tt_sd_peak_to_peak", "SD across trials of per-trial peak-to-peak"),
    ("Trial-to-trial variability", "tt_sd_temporal_sd", "SD across trials of per-trial temporal SD"),
    ("Trial-to-trial variability", "tt_cv", "Trial-to-trial SD divided by the absolute mean level"),
    ("Temporal persistence", "ac1_trial", "Lag-1 autocorrelation of the per-trial mean-level series"),
    ("Temporal persistence", "ac2_trial", "Lag-2 autocorrelation of the per-trial mean-level series"),
    ("Temporal persistence", "mssd_over_sd", "Consecutive-trial variability relative to overall trial SD"),
    ("Temporal persistence", "ac_tau_signal", "Lag at which the time-series autocorrelation drops below 1/e"),
    ("Session dynamics", "drift_slope", "Least-squares slope of the per-trial mean level against trial number"),
    ("Session dynamics", "var_late_minus_early", "Detrended trial-level SD in the second minus the first half of a run"),
    ("Session dynamics", "ptp_late_minus_early", "Per-trial peak-to-peak in the second minus the first half of a run"),
    ("Session dynamics", "within_block_slope", "Slope of the per-trial mean level against position inside a GVS block"),
    ("Session dynamics", "absdev_drift_slope", "Slope of the per-trial absolute deviation from the run mean against trial number"),
    ("Session dynamics", "absdev_block_slope", "Slope of the per-trial absolute deviation from the run mean against block position"),
    ("Session dynamics", "absdev_late_minus_early", "Per-trial absolute deviation from the run mean, second minus first half of the run"),
    ("Reproducibility", "splithalf_waveform_r", "Correlation between odd-trial and even-trial average waveforms"),
    ("Reproducibility", "template_r", "Mean correlation of each trial's waveform with the trial-averaged template"),
    ("Reproducibility", "pc1_variance_explained", "Variance explained by the first principal component of the trial waveforms"),
    ("Spectral / complexity", "spectral_slope", "Log-log slope of the power spectrum of the concatenated time series"),
    ("Spectral / complexity", "spectral_centroid", "Power-weighted mean frequency of the concatenated time series"),
    ("Spectral / complexity", "spectral_entropy", "Normalised entropy of the power spectrum"),
    ("Spectral / complexity", "relative_low_power", "Fraction of spectral power below 0.1 cycles/sample"),
    ("Spectral / complexity", "relative_high_power", "Fraction of spectral power above 0.25 cycles/sample"),
    ("Spectral / complexity", "dfa_alpha", "Detrended fluctuation analysis scaling exponent"),
    ("Spectral / complexity", "sample_entropy", "Sample entropy (m=2, r=0.2 SD) of the concatenated time series"),
    ("Spectral / complexity", "signal_roughness", "Successive-difference variability of the time series relative to its SD"),
    ("Distribution", "trial_skewness", "Skewness across trials of the per-trial mean level"),
    ("Distribution", "trial_kurtosis", "Excess kurtosis across trials of the per-trial mean level"),
    ("Distribution", "frac_extreme_trials", "Fraction of trials beyond 2 SD of the detrended trial series"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--active-bold-group", type=Path, default=ROOT / sp.DEFAULT_ACTIVE_BOLD_GROUP)
    parser.add_argument("--active-flat-indices", type=Path, default=ROOT / sp.DEFAULT_ACTIVE_FLAT_INDICES)
    parser.add_argument("--projection-weight-map", type=Path, default=ROOT / sp.DEFAULT_WEIGHT_MAP)
    parser.add_argument("--projection-html-mask", type=Path, default=ROOT / sp.DEFAULT_WEIGHT_HTML)
    parser.add_argument("--trial-metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--clinical-workbook", type=Path, default=DEFAULT_WORKBOOK)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--refresh-cache", action="store_true")
    return parser.parse_args()


def canonical_projection_subject(value: object) -> str:
    match = re.search(r"(?:PS)?[_-]?PD\s*0*(\d+)", str(value), flags=re.IGNORECASE)
    if match is None:
        raise ValueError(f"Could not parse subject identifier: {value!r}")
    return f"sub-pd{int(match.group(1)):03d}"


def load_projection(args: argparse.Namespace) -> np.ndarray:
    cache = args.out_dir / CACHE_NAME
    if cache.is_file() and not args.refresh_cache:
        return np.load(cache)
    projection, metadata = sp._weighted_html_projection_from_active_bold(
        args.active_bold_group, args.active_flat_indices,
        args.projection_weight_map, args.projection_html_mask, 512)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache, projection)
    print(f"Recomputed projection: {metadata['selected_active_voxels']} voxels -> {projection.shape}")
    return projection


def _slope(values: np.ndarray) -> float:
    t = np.arange(values.size, dtype=float)
    centred = t - t.mean()
    return float(((values - values.mean()) * centred).sum() / (centred**2).sum())


def _dfa(x: np.ndarray) -> float:
    y = np.cumsum(x - x.mean())
    scales, fluctuations = [], []
    for scale in (8, 12, 16, 24, 32, 48, 64, 96):
        if y.size < 4 * scale:
            continue
        n_segments = y.size // scale
        segments = y[: n_segments * scale].reshape(n_segments, scale)
        design = np.vstack([np.arange(scale, dtype=float), np.ones(scale)]).T
        coefficients, *_ = np.linalg.lstsq(design, segments.T, rcond=None)
        residuals = segments.T - design @ coefficients
        fluctuations.append(np.sqrt((residuals**2).mean()))
        scales.append(scale)
    if len(fluctuations) < 3:
        return np.nan
    return float(np.polyfit(np.log(scales), np.log(fluctuations), 1)[0])


def _sample_entropy(x: np.ndarray, m: int = 2, r: float = 0.2, max_length: int = 600) -> float:
    x = (x - x.mean()) / x.std(ddof=1)
    x = x[:max_length]

    def count(dimension: int) -> int:
        embedded = np.lib.stride_tricks.sliding_window_view(x, dimension)
        distance = np.abs(embedded[:, None, :] - embedded[None, :, :]).max(-1)
        np.fill_diagonal(distance, np.inf)
        return int((distance <= r).sum())

    return float(-np.log((count(m + 1) + 1e-12) / (count(m) + 1e-12)))


def compute_features(trials: np.ndarray, block_position: np.ndarray) -> dict[str, float]:
    """All candidate features for one subject-run. ``trials`` is (n_trials, n_timepoints)."""
    n_trials, n_time = trials.shape
    mean_level = trials.mean(axis=1)
    peak_to_peak = trials.max(axis=1) - trials.min(axis=1)
    temporal_sd = trials.std(axis=1, ddof=1)
    baseline_centred = trials - trials[:, [0]]
    evoked = trials.mean(axis=0)
    evoked_centred = evoked - evoked.mean()
    series = trials.ravel()
    trial_index = np.arange(n_trials, dtype=float)
    detrended = mean_level - np.polyval(np.polyfit(trial_index, mean_level, 1), trial_index)
    half = n_trials // 2
    absolute_deviation = np.abs(mean_level - mean_level.mean())

    spectrum = np.abs(np.fft.rfft(series - series.mean())) ** 2
    frequency = np.fft.rfftfreq(series.size, d=1.0)
    positive = frequency > 0
    spectrum, frequency = spectrum[positive], frequency[positive]
    normalised = spectrum / spectrum.sum()

    autocorrelation = np.correlate(series - series.mean(), series - series.mean(), "full")[series.size - 1:]
    autocorrelation /= autocorrelation[0]
    below = np.flatnonzero(autocorrelation < 1 / np.e)

    odd, even = trials[0::2].mean(axis=0), trials[1::2].mean(axis=0)
    waveform_centred = trials - trials.mean(axis=1, keepdims=True)
    singular = np.linalg.svd(trials - trials.mean(axis=0, keepdims=True), compute_uv=False)

    return {
        "level_mean": float(mean_level.mean()),
        "level_auc": float(np.trapezoid(trials, dx=1.0, axis=1).mean()),
        "signal_sd_overall": float(series.std(ddof=1)),
        "signal_mad": float(np.median(np.abs(series - np.median(series)))),
        "wt_peak_to_peak": float(peak_to_peak.mean()),
        "wt_temporal_sd": float(temporal_sd.mean()),
        "wt_slope": float(np.mean([_slope(row) for row in trials])),
        "wt_early_late_change": float((trials[:, -3:].mean(axis=1) - trials[:, :3].mean(axis=1)).mean()),
        "wt_abs_baseline_response": float(np.abs(baseline_centred).max(axis=1).mean()),
        "wt_baseline_to_peak": float(baseline_centred.max(axis=1).mean()),
        "wt_baseline_to_trough": float(baseline_centred.min(axis=1).mean()),
        "ev_peak_to_peak": float(evoked.max() - evoked.min()),
        "ev_temporal_sd": float(evoked.std(ddof=1)),
        "ev_slope": _slope(evoked),
        "ev_early_late_change": float(evoked[-3:].mean() - evoked[:3].mean()),
        "ev_peak_latency": float(np.argmax(np.abs(evoked - evoked[0]))),
        "tt_sd": float(mean_level.std(ddof=1)),
        "tt_mssd": float(np.sqrt(np.mean(np.diff(mean_level) ** 2) / 2)),
        "tt_mad": float(np.median(np.abs(mean_level - np.median(mean_level)))),
        "tt_sd_peak_to_peak": float(peak_to_peak.std(ddof=1)),
        "tt_sd_temporal_sd": float(temporal_sd.std(ddof=1)),
        "tt_cv": float(mean_level.std(ddof=1) / (abs(mean_level.mean()) + 1e-12)),
        "ac1_trial": float(np.corrcoef(mean_level[:-1], mean_level[1:])[0, 1]),
        "ac2_trial": float(np.corrcoef(mean_level[:-2], mean_level[2:])[0, 1]),
        "mssd_over_sd": float(np.sqrt(np.mean(np.diff(mean_level) ** 2) / 2) / (mean_level.std(ddof=1) + 1e-12)),
        "ac_tau_signal": float(below[0]) if below.size else float(series.size),
        "drift_slope": _slope(mean_level),
        "var_late_minus_early": float(detrended[half:].std(ddof=1) - detrended[:half].std(ddof=1)),
        "ptp_late_minus_early": float(peak_to_peak[half:].mean() - peak_to_peak[:half].mean()),
        "within_block_slope": float(np.polyfit(block_position, mean_level, 1)[0]),
        "absdev_drift_slope": float(np.polyfit(trial_index, absolute_deviation, 1)[0]),
        "absdev_block_slope": float(np.polyfit(block_position, absolute_deviation, 1)[0]),
        "absdev_late_minus_early": float(absolute_deviation[half:].mean() - absolute_deviation[:half].mean()),
        "splithalf_waveform_r": float(np.corrcoef(odd - odd.mean(), even - even.mean())[0, 1]),
        "template_r": float(np.mean((waveform_centred @ evoked_centred)
                                    / (np.linalg.norm(waveform_centred, axis=1)
                                       * np.linalg.norm(evoked_centred) + 1e-12))),
        "pc1_variance_explained": float(singular[0] ** 2 / (singular**2).sum()),
        "spectral_slope": float(np.polyfit(np.log(frequency), np.log(spectrum + 1e-30), 1)[0]),
        "spectral_centroid": float((frequency * normalised).sum()),
        "spectral_entropy": float(-np.sum(normalised * np.log(normalised + 1e-30)) / np.log(normalised.size)),
        "relative_low_power": float(normalised[frequency < 0.1].sum()),
        "relative_high_power": float(normalised[frequency >= 0.25].sum()),
        "dfa_alpha": _dfa(series),
        "sample_entropy": _sample_entropy(series),
        "signal_roughness": float(np.sqrt(np.mean(np.diff(series) ** 2) / 2) / (series.std(ddof=1) + 1e-12)),
        "trial_skewness": float(stats.skew(mean_level)),
        "trial_kurtosis": float(stats.kurtosis(mean_level)),
        "frac_extreme_trials": float(np.mean(np.abs(detrended) > 2 * detrended.std(ddof=1))),
    }


def run_level_features(projection: np.ndarray, metadata: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (subject, medication, run), group in metadata.groupby(["subject", "medication", "run"]):
        group = group.sort_values("run_trial_index")
        if len(group) < MIN_TRIALS_PER_RUN:
            continue
        trials = projection[group["projected_trial_index"].to_numpy(dtype=np.int64)]
        features = compute_features(trials, group["trial_in_block"].to_numpy(dtype=float))
        records.append({"subject": subject, "medication": medication, "run": int(run),
                        "n_trials": len(group), **features})
    return pd.DataFrame(records)


def fisher_interval(r: float, n: int, spearman: bool) -> tuple[float, float]:
    if not np.isfinite(r) or n < 5 or abs(r) >= 1:
        return np.nan, np.nan
    z = np.arctanh(r)
    se = (1.06 if spearman else 1.0) / np.sqrt(n - 3)
    return float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))


def detectable_effect(n: int, power: float = 0.8, alpha: float = 0.05) -> float:
    se = 1.0 / np.sqrt(n - 3)
    return float(np.tanh(se * (stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power))))


def achieved_power(rho: float, n: int, alpha: float = 0.05) -> float:
    se = 1.0 / np.sqrt(n - 3)
    critical = stats.norm.ppf(1 - alpha / 2)
    shift = np.arctanh(rho) / se
    return float(stats.norm.cdf(shift - critical) + stats.norm.cdf(-shift - critical))


def analyze(run_features: pd.DataFrame, updrs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    names = [name for _, name, _ in FEATURE_DEFINITIONS]
    definitions = {name: (family, text) for family, name, text in FEATURE_DEFINITIONS}
    clinical = updrs.copy()
    clinical["subject"] = clinical["subject"].map(canonical_projection_subject)

    rows, subject_frames = [], []
    for medication in ("OFF", "ON"):
        data = run_features.loc[run_features["medication"].eq(medication)]
        wide = data.pivot(index="subject", columns="run")
        subjects = data.groupby("subject", as_index=False)[names].mean().merge(
            clinical.loc[clinical["medication"].eq(medication),
                         ["subject", "updrs_iii_bradykinesia_subscore"]],
            on="subject", how="left").dropna(subset=["updrs_iii_bradykinesia_subscore"])
        subjects.insert(1, "medication", medication)
        subject_frames.append(subjects)
        y = subjects["updrs_iii_bradykinesia_subscore"].to_numpy(dtype=float)

        for name in names:
            x = subjects[name].to_numpy(dtype=float)
            family, text = definitions[name]
            record = {"medication": medication, "family": family, "feature": name,
                      "definition": text, "n_subjects": int(x.size)}
            first, second = wide[(name, 1)].to_numpy(dtype=float), wide[(name, 2)].to_numpy(dtype=float)
            usable = np.isfinite(first) & np.isfinite(second)
            record["retest_r"] = (float(np.corrcoef(first[usable], second[usable])[0, 1])
                                  if usable.sum() > 3 and np.std(first[usable]) > 0 else np.nan)
            if np.all(np.isfinite(x)) and np.std(x) > 0:
                spearman, pearson = stats.spearmanr(x, y), stats.pearsonr(x, y)
                record["spearman_rho"] = float(spearman.statistic)
                record["spearman_p_value"] = float(spearman.pvalue)
                record["pearson_r"] = float(pearson.statistic)
                record["pearson_p_value"] = float(pearson.pvalue)
                low, high = fisher_interval(float(spearman.statistic), int(x.size), spearman=True)
                record["spearman_ci95_low"], record["spearman_ci95_high"] = low, high
            rows.append(record)

    results = pd.DataFrame(rows)
    for medication in ("OFF", "ON"):
        mask = results["medication"].eq(medication) & results["spearman_p_value"].notna()
        results.loc[mask, "spearman_fdr_q_value"] = bh_fdr(results.loc[mask, "spearman_p_value"])
    return results, pd.concat(subject_frames, ignore_index=True)


def plot(results: pd.DataFrame, output_stem: Path) -> None:
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"],
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    off = results.loc[results["medication"].eq("OFF") & results["spearman_rho"].notna()].copy()
    off = off.sort_values(["family", "spearman_rho"]).reset_index(drop=True)
    n = int(off["n_subjects"].iloc[0])

    fig = plt.figure(figsize=(16.5, 9.6), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=[1.35, 1.0], height_ratios=[1.0, 1.0])
    ax_forest = fig.add_subplot(grid[:, 0])
    ax_reliability = fig.add_subplot(grid[0, 1])
    ax_power = fig.add_subplot(grid[1, 1])

    families = list(dict.fromkeys(off["family"]))
    palette = dict(zip(families, plt.cm.tab10(np.linspace(0, 1, 10))[:len(families)]))
    positions = np.arange(len(off))
    for position, row in zip(positions, off.itertuples()):
        ax_forest.plot([row.spearman_ci95_low, row.spearman_ci95_high], [position, position],
                       color=palette[row.family], linewidth=1.9, alpha=0.75, solid_capstyle="round")
        ax_forest.scatter(row.spearman_rho, position, s=34, color=palette[row.family],
                          edgecolor="white", linewidth=0.6, zorder=3)
    ax_forest.axvline(0, color="#333333", linewidth=1.2)
    critical = detectable_effect(n, power=0.5)
    for sign in (-1, 1):
        ax_forest.axvline(sign * critical, color="#B03A2E", linestyle="--", linewidth=1.1)
    ax_forest.set_yticks(positions)
    ax_forest.set_yticklabels(off["feature"], fontsize=8.4)
    ax_forest.set_ylim(-1, len(off))
    ax_forest.set_xlim(-1, 1)
    ax_forest.set_xlabel("Spearman $\\rho$ with bradykinesia subscore (OFF), 95% CI", fontsize=12.5)
    ax_forest.set_title(f"No projection feature tracks bradykinesia (n = {n})", fontsize=13.5, fontweight="bold")
    ax_forest.tick_params(axis="x", labelsize=11)
    ax_forest.grid(axis="x", alpha=0.2)
    ax_forest.spines[["top", "right"]].set_visible(False)
    handles = [plt.Line2D([], [], color=palette[f], marker="o", linestyle="-", markersize=5, label=f)
               for f in families]
    ax_forest.legend(handles=handles, fontsize=8.6, loc="lower right", frameon=True, framealpha=0.92)
    ax_forest.text(critical + 0.03, len(off) - 1.2, "p = 0.05", color="#B03A2E", fontsize=9.5, va="top")

    usable = off["retest_r"].notna()
    ax_reliability.scatter(off.loc[usable, "retest_r"], off.loc[usable, "spearman_rho"].abs(),
                           s=52, c=[palette[f] for f in off.loc[usable, "family"]],
                           edgecolor="white", linewidth=0.7)
    ax_reliability.axvline(RELIABLE_THRESHOLD, color="#333333", linestyle=":", linewidth=1.3)
    ax_reliability.axhline(critical, color="#B03A2E", linestyle="--", linewidth=1.1)
    ax_reliability.set_xlabel("Test-retest reliability of the feature (run 1 vs run 2)", fontsize=12)
    ax_reliability.set_ylabel("|Spearman $\\rho$| with bradykinesia", fontsize=12)
    ax_reliability.set_title("Well-measured features are the flattest", fontsize=13, fontweight="bold")
    ax_reliability.tick_params(axis="both", labelsize=11)
    ax_reliability.grid(alpha=0.2)
    ax_reliability.spines[["top", "right"]].set_visible(False)
    reliable = off.loc[usable & off["retest_r"].ge(RELIABLE_THRESHOLD)]
    ax_reliability.text(0.03, 0.96,
                        f"reliable features (r $\\geq$ {RELIABLE_THRESHOLD}): {len(reliable)}\n"
                        f"largest |$\\rho$| among them: {reliable['spearman_rho'].abs().max():.2f}",
                        transform=ax_reliability.transAxes, va="top", fontsize=10.5,
                        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white",
                              "edgecolor": "#BBBBBB", "alpha": 0.9})

    grid_rho = np.linspace(0.05, 0.95, 200)
    ax_power.plot(grid_rho, [achieved_power(r, n) for r in grid_rho], color="#245A86", linewidth=2.4)
    ax_power.axhline(0.8, color="#B03A2E", linestyle="--", linewidth=1.2)
    detectable = detectable_effect(n, power=0.8)
    ax_power.axvline(detectable, color="#B03A2E", linestyle="--", linewidth=1.2)
    ax_power.fill_between(grid_rho, 0, [achieved_power(r, n) for r in grid_rho],
                          where=grid_rho >= detectable, color="#4C78A8", alpha=0.16)
    ax_power.set_xlabel("True correlation $\\rho$", fontsize=12)
    ax_power.set_ylabel("Power to detect it", fontsize=12)
    ax_power.set_title(f"What this sample could have seen (n = {n})", fontsize=13, fontweight="bold")
    ax_power.set_ylim(0, 1.02)
    ax_power.set_xlim(0, 1)
    ax_power.tick_params(axis="both", labelsize=11)
    ax_power.grid(alpha=0.2)
    ax_power.spines[["top", "right"]].set_visible(False)
    ax_power.text(0.03, 0.96,
                  f"80% power at |$\\rho$| $\\geq$ {detectable:.2f}\n"
                  f"power at $\\rho$ = 0.5: {achieved_power(0.5, n):.0%}\n"
                  f"power at $\\rho$ = 0.3: {achieved_power(0.3, n):.0%}",
                  transform=ax_power.transAxes, va="top", fontsize=10.5,
                  bbox={"boxstyle": "round,pad=0.35", "facecolor": "white",
                        "edgecolor": "#BBBBBB", "alpha": 0.9})

    for ax, label in zip((ax_forest, ax_reliability, ax_power), ("A", "B", "C")):
        ax.text(-0.09, 1.02, label, transform=ax.transAxes, ha="left", va="bottom",
                fontsize=21, fontweight="bold", clip_on=False)

    fig.savefig(output_stem.with_suffix(".png"), dpi=210)
    fig.savefig(output_stem.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(args.trial_metadata, sep="\t")
    projection = load_projection(args)
    run_features = run_level_features(projection, metadata)
    results, subjects = analyze(run_features, load_updrs_bradykinesia(args.clinical_workbook))

    stem = args.out_dir / OUT_STEM
    results.to_csv(stem.with_name(f"{OUT_STEM}_statistics.csv"), index=False)
    subjects.to_csv(stem.with_name(f"{OUT_STEM}_subject_data.csv"), index=False)
    run_features.to_csv(stem.with_name(f"{OUT_STEM}_run_data.csv"), index=False)
    plot(results, stem)

    pd.set_option("display.width", 250)
    for medication in ("OFF", "ON"):
        block = results.loc[results["medication"].eq(medication) & results["spearman_rho"].notna()]
        n = int(block["n_subjects"].iloc[0])
        reliable = block.loc[block["retest_r"].ge(RELIABLE_THRESHOLD)]
        print(f"\n=== {medication} (n = {n}) ===")
        print(f"features tested            : {len(block)}")
        print(f"reliable (retest r >= {RELIABLE_THRESHOLD}) : {len(reliable)}")
        print(f"uncorrected p < 0.05       : {int((block['spearman_p_value'] < 0.05).sum())}"
              f"  (expected by chance: {0.05 * len(block):.1f})")
        print(f"  of those, reliable       : {int((reliable['spearman_p_value'] < 0.05).sum())}")
        print(f"surviving BH-FDR q < 0.05  : {int((block['spearman_fdr_q_value'] < 0.05).sum())}")
        print(f"largest |rho| overall      : {block['spearman_rho'].abs().max():.3f}")
        print(f"largest |rho| among reliable: {reliable['spearman_rho'].abs().max():.3f}")
        print(f"80% power at |rho| >=      : {detectable_effect(n):.2f}"
              f"   (power at rho=0.5: {achieved_power(0.5, n):.0%})")
        print("\nstrongest five (uncorrected):")
        print(block.reindex(block["spearman_p_value"].sort_values().index)
              [["family", "feature", "retest_r", "spearman_rho", "spearman_p_value",
                "spearman_fdr_q_value"]].head(5).to_string(index=False))
    print(f"\nSaved outputs with stem {stem}")


if __name__ == "__main__":
    main()
