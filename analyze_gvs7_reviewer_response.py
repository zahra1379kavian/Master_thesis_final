#!/usr/bin/env python3
"""Reviewer-facing GVS7 vigour-versus-task interaction analysis.

The vigour projection is a weighted voxel sum whereas the task-map signal is a
voxel mean.  Raw differences between those signals are therefore scale
dependent.  This analysis puts both signals on a comparable scale before
feature extraction: each trial is baseline-centred, then divided by the
standard deviation of all baseline-centred sham samples from the matched
subject/session/run/network.

The primary neural hypothesis is the medication-OFF GVS7 interaction

    (GVS7 - sham)[vigour] - (GVS7 - sham)[task map]

for four predefined features.  A mixed-effects model tests the interaction at
the run level, and an exact subject-level sign-flip test is reported as a
distribution-free sensitivity analysis.  GVS7 is treated as the planned
waveform because it was selected by the independent behavioural result.  A
32-test analysis spanning every waveform is also saved as a supplementary
multiplicity sensitivity analysis.

Finally, matched subject-level GVS7 behavioural effects are related to the
vigour, task-map, and direct-interaction neural effects.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import matplotlib

warnings.filterwarnings("ignore", message="Unable to import Axes3D.*")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.formula.api import mixedlm

from spagetti_plot import (
    FEATURE_LABELS,
    GVS_TEST_FEATURE_NAMES,
    _compute_trial_features,
    _exact_sign_flip_pvalue,
    _fdr_bh,
    _reduce_task_map_bold_trials,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_VIGOUR_TRIALS = (
    ROOT / "figures" / "gvs_projection_features_vs_sham"
    / "projection_full_trials_cache.npy"
)
DEFAULT_TASK_TRIALS = ROOT / "data" / "Task_map_BOLD_trials.npy"
DEFAULT_METADATA = (
    ROOT / "data" / "GVS_projection_BOLD" / "gvs_projection_trial_metadata.tsv"
)
DEFAULT_MANIFEST = ROOT / "data" / "concat_manifest_group.tsv"
DEFAULT_BEHAVIOUR = (
    ROOT / "figures" / "projected_RT" / "revised_behaviour_from_mat"
    / "gvs_rt_lme_metrics_results" / "gvs_rt_lme_trial_table.csv"
)
DEFAULT_OUT_DIR = ROOT / "figures" / "gvs7_reviewer_response"

SHAM = "gvs-01"
GVS7 = "gvs-08"
OFF_SESSION = 1
OFF_MEDICATION = "OFF"
NETWORKS = ["task", "vigour"]
FEATURES = list(GVS_TEST_FEATURE_NAMES)
MIXEDLM_METHODS = ("powell", "nm", "lbfgs", "bfgs", "cg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vigour-trials", type=Path, default=DEFAULT_VIGOUR_TRIALS)
    parser.add_argument("--task-trials", type=Path, default=DEFAULT_TASK_TRIALS)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--behaviour", type=Path, default=DEFAULT_BEHAVIOUR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--task-chunk-size", type=int, default=512)
    return parser.parse_args()


def mean_ci(values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan, np.nan
    mean = float(np.mean(values))
    if values.size == 1:
        return mean, np.nan, np.nan
    half_width = float(stats.t.ppf(0.975, values.size - 1) * stats.sem(values))
    return mean, mean - half_width, mean + half_width


def cohen_dz(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return np.nan
    sd = float(np.std(values, ddof=1))
    return float(np.mean(values) / sd) if sd > 0 else np.nan


def load_task_mean_trials(
    task_path: Path,
    manifest_path: Path,
    cache_path: Path,
    chunk_size: int,
) -> np.ndarray:
    if cache_path.exists():
        cached = np.load(cache_path, mmap_mode="r")
        source = np.load(task_path, mmap_mode="r")
        expected = (source.shape[1], source.shape[2])
        if cached.shape == expected:
            return np.asarray(cached, dtype=np.float64)

    reduced = _reduce_task_map_bold_trials(
        task_path,
        manifest_path,
        reducer="mean",
        chunk_size=chunk_size,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, reduced)
    return reduced


def validate_inputs(metadata: pd.DataFrame, trial_arrays: dict[str, np.ndarray]) -> None:
    required = {
        "projected_trial_index",
        "subject",
        "session",
        "medication",
        "run",
        "gvs_code",
    }
    missing = sorted(required - set(metadata.columns))
    if missing:
        raise ValueError(f"Metadata is missing columns: {', '.join(missing)}")
    expected_rows = len(metadata)
    expected_time = None
    for network, trials in trial_arrays.items():
        if trials.ndim != 2 or trials.shape[0] != expected_rows:
            raise ValueError(
                f"{network} trials have shape {trials.shape}; expected "
                f"({expected_rows}, time)"
            )
        if expected_time is None:
            expected_time = trials.shape[1]
        elif trials.shape[1] != expected_time:
            raise ValueError("Vigour and task trials have different time dimensions")
    indices = metadata["projected_trial_index"].to_numpy(dtype=np.int64)
    if not np.array_equal(indices, np.arange(expected_rows, dtype=np.int64)):
        raise ValueError("Metadata rows are not in projected-trial order")


def sham_standardize_trials(
    trials: np.ndarray,
    metadata: pd.DataFrame,
    network: str,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Baseline-centre trials and divide by matched run-level sham SD."""
    centered = trials - trials[:, [0]]
    standardized = np.full_like(centered, np.nan, dtype=np.float64)
    scale_rows: list[dict[str, Any]] = []
    keys = ["subject", "session", "medication", "run"]

    for key, group in metadata.groupby(keys, sort=True):
        run_indices = group.index.to_numpy(dtype=np.int64)
        sham_indices = group.loc[group["gvs_code"].eq(SHAM)].index.to_numpy(dtype=np.int64)
        if sham_indices.size == 0:
            raise ValueError(f"No sham trials for {network} run {key}")
        sham_values = centered[sham_indices].ravel()
        sham_values = sham_values[np.isfinite(sham_values)]
        scale = float(np.std(sham_values, ddof=1)) if sham_values.size > 1 else np.nan
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"Invalid sham scale for {network} run {key}: {scale}")
        standardized[run_indices] = centered[run_indices] / scale
        scale_rows.append(
            {
                "subject": key[0],
                "session": int(key[1]),
                "medication": str(key[2]),
                "run": int(key[3]),
                "network": network,
                "sham_response_sd": scale,
                "n_sham_trials": int(sham_indices.size),
                "n_sham_samples": int(sham_values.size),
            }
        )

    return standardized, pd.DataFrame(scale_rows)


def make_standardized_run_features(
    trial_arrays: dict[str, np.ndarray],
    metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_frames = []
    scale_frames = []
    keys = ["subject", "session", "medication", "run", "gvs_code"]

    for network, trials in trial_arrays.items():
        standardized, scales = sham_standardize_trials(trials, metadata, network)
        features = _compute_trial_features(standardized)
        trial_frame = pd.concat(
            [metadata.reset_index(drop=True), features.reset_index(drop=True)],
            axis=1,
        )
        run_frame = trial_frame.groupby(keys, as_index=False)[FEATURES].mean()
        counts = trial_frame.groupby(keys, as_index=False).size().rename(columns={"size": "n_trials"})
        run_frame = run_frame.merge(counts, on=keys, how="left", validate="one_to_one")
        run_frame["network"] = network
        run_frames.append(run_frame)
        scale_frames.append(scales)

    runs = pd.concat(run_frames, ignore_index=True)
    runs["network"] = pd.Categorical(runs["network"], categories=NETWORKS, ordered=True)
    return (
        runs.sort_values(["session", "subject", "run", "gvs_code", "network"]).reset_index(drop=True),
        pd.concat(scale_frames, ignore_index=True).sort_values(
            ["session", "subject", "run", "network"]
        ).reset_index(drop=True),
    )


def paired_network_values(
    run_features: pd.DataFrame,
    session: int,
    active_gvs: str,
    feature: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = run_features.loc[
        run_features["session"].astype(int).eq(int(session))
        & run_features["gvs_code"].isin([SHAM, active_gvs]),
        ["subject", "session", "medication", "run", "network", "gvs_code", feature],
    ].copy()
    wide = selected.pivot(
        index=["subject", "session", "medication", "run"],
        columns=["network", "gvs_code"],
        values=feature,
    )
    required = [(network, code) for network in NETWORKS for code in (SHAM, active_gvs)]
    missing = [column for column in required if column not in wide.columns]
    if missing:
        raise ValueError(f"Missing paired columns for {feature}: {missing}")
    wide = wide.dropna(subset=required)

    run_values = wide.index.to_frame(index=False)
    run_values["active_gvs"] = active_gvs
    run_values["feature"] = feature
    for network in NETWORKS:
        run_values[f"{network}_sham"] = wide[(network, SHAM)].to_numpy(dtype=float)
        run_values[f"{network}_active"] = wide[(network, active_gvs)].to_numpy(dtype=float)
        run_values[f"{network}_delta"] = (
            run_values[f"{network}_active"] - run_values[f"{network}_sham"]
        )
    run_values["interaction_vigour_minus_task"] = (
        run_values["vigour_delta"] - run_values["task_delta"]
    )

    subject_values = (
        run_values.groupby(["subject", "session", "medication", "active_gvs", "feature"], as_index=False)
        .agg(
            vigour_delta=("vigour_delta", "mean"),
            task_delta=("task_delta", "mean"),
            interaction_vigour_minus_task=("interaction_vigour_minus_task", "mean"),
            n_runs=("interaction_vigour_minus_task", "size"),
        )
        .sort_values("subject")
        .reset_index(drop=True)
    )
    return run_values, subject_values


def fit_interaction_mixedlm(
    run_features: pd.DataFrame,
    session: int,
    active_gvs: str,
    feature: str,
) -> dict[str, Any]:
    data = run_features.loc[
        run_features["session"].astype(int).eq(int(session))
        & run_features["gvs_code"].isin([SHAM, active_gvs]),
        ["subject", "run", "network", "gvs_code", feature],
    ].dropna().copy()
    data = data.rename(columns={feature: "value", "gvs_code": "stimulation"})
    data["network"] = pd.Categorical(data["network"], categories=NETWORKS, ordered=True)
    data["stimulation"] = pd.Categorical(
        data["stimulation"], categories=[SHAM, active_gvs], ordered=True
    )
    data["subject_run"] = data["subject"].astype(str) + "_run" + data["run"].astype(str)
    complete_cells = (
        data.groupby("subject_run", observed=True)
        .size()
        .loc[lambda values: values.eq(4)]
        .index
    )
    data = data.loc[data["subject_run"].isin(complete_cells)].copy()
    if data.empty:
        raise RuntimeError(f"No complete run-level network/stimulation cells for {feature}")
    formula = (
        'value ~ C(network, Treatment(reference="task")) * '
        f'C(stimulation, Treatment(reference="{SHAM}")) + C(run)'
    )

    last_error: Exception | None = None
    for method in MIXEDLM_METHODS:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                model = mixedlm(
                    formula,
                    data=data,
                    groups=data["subject"],
                    re_formula="1",
                    vc_formula={"subject_run": "0 + C(subject_run)"},
                )
                result = model.fit(reml=False, method=method, maxiter=1000, disp=False)
            except Exception as exc:
                last_error = exc
                continue
        if not np.isfinite(result.llf):
            continue
        interaction_terms = [
            term
            for term in result.params.index
            if ":" in str(term) and "network" in str(term) and "stimulation" in str(term)
        ]
        if len(interaction_terms) != 1:
            raise RuntimeError(f"Could not identify one interaction term: {interaction_terms}")
        term = interaction_terms[0]
        estimate = float(result.params[term])
        se = float(result.bse[term])
        z_value = float(result.tvalues[term])
        return {
            "mixedlm_formula": formula,
            "mixedlm_interaction_term": str(term),
            "mixedlm_interaction_estimate": estimate,
            "mixedlm_interaction_se": se,
            "mixedlm_interaction_ci95_low": estimate - 1.96 * se,
            "mixedlm_interaction_ci95_high": estimate + 1.96 * se,
            "mixedlm_interaction_z": z_value,
            "mixedlm_interaction_p": float(result.pvalues[term]),
            "mixedlm_n_rows": int(len(data)),
            "mixedlm_n_subjects": int(data["subject"].nunique()),
            "mixedlm_optimizer": method,
            "mixedlm_converged": bool(getattr(result, "converged", False)),
            "mixedlm_warnings": " | ".join(dict.fromkeys(str(item.message) for item in caught)),
            "mixedlm_subject_variance": float(result.cov_re.iloc[0, 0]),
            "mixedlm_residual_variance": float(result.scale),
        }
    raise RuntimeError(f"MixedLM failed for {feature}: {last_error}")


def contrast_summary(values: np.ndarray, prefix: str) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    mean, low, high = mean_ci(values)
    p_value, exact = _exact_sign_flip_pvalue(values)
    return {
        f"{prefix}_mean": mean,
        f"{prefix}_ci95_low": low,
        f"{prefix}_ci95_high": high,
        f"{prefix}_cohen_dz": cohen_dz(values),
        f"{prefix}_p_signflip": p_value,
        f"{prefix}_p_signflip_exact": bool(exact),
    }


def primary_gvs7_analysis(
    run_features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    run_tables = []
    subject_tables = []
    rows = []
    for feature in FEATURES:
        run_values, subject_values = paired_network_values(
            run_features, OFF_SESSION, GVS7, feature
        )
        model = fit_interaction_mixedlm(run_features, OFF_SESSION, GVS7, feature)
        row = {
            "session": OFF_SESSION,
            "medication": OFF_MEDICATION,
            "active_gvs": GVS7,
            "feature": feature,
            "label": FEATURE_LABELS[feature],
            "n_subjects": int(len(subject_values)),
            "n_run_pairs": int(subject_values["n_runs"].sum()),
            **contrast_summary(subject_values["vigour_delta"].to_numpy(), "vigour"),
            **contrast_summary(subject_values["task_delta"].to_numpy(), "task"),
            **contrast_summary(
                subject_values["interaction_vigour_minus_task"].to_numpy(),
                "interaction",
            ),
            **model,
        }
        rows.append(row)
        run_tables.append(run_values)
        subject_tables.append(subject_values)

    result = pd.DataFrame(rows)
    result["interaction_q_signflip_fdr_4_features"] = _fdr_bh(
        result["interaction_p_signflip"].to_numpy(dtype=float)
    )
    result["vigour_q_signflip_fdr_4_features"] = _fdr_bh(
        result["vigour_p_signflip"].to_numpy(dtype=float)
    )
    result["task_q_signflip_fdr_4_features"] = _fdr_bh(
        result["task_p_signflip"].to_numpy(dtype=float)
    )
    result["mixedlm_interaction_q_fdr_4_features"] = _fdr_bh(
        result["mixedlm_interaction_p"].to_numpy(dtype=float)
    )
    return (
        pd.concat(run_tables, ignore_index=True),
        pd.concat(subject_tables, ignore_index=True),
        result.sort_values("interaction_p_signflip").reset_index(drop=True),
    )


def all_waveform_sensitivity(run_features: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for session in sorted(run_features["session"].astype(int).unique()):
        active_codes = sorted(set(run_features.loc[run_features["session"].eq(session), "gvs_code"]) - {SHAM})
        for active_gvs in active_codes:
            for feature in FEATURES:
                _, subject_values = paired_network_values(
                    run_features, session, active_gvs, feature
                )
                interaction = subject_values["interaction_vigour_minus_task"].to_numpy(dtype=float)
                p_value, exact = _exact_sign_flip_pvalue(interaction)
                mean, low, high = mean_ci(interaction)
                rows.append(
                    {
                        "session": int(session),
                        "medication": str(subject_values["medication"].iloc[0]),
                        "active_gvs": active_gvs,
                        "feature": feature,
                        "label": FEATURE_LABELS[feature],
                        "n_subjects": int(len(subject_values)),
                        "interaction_mean": mean,
                        "interaction_ci95_low": low,
                        "interaction_ci95_high": high,
                        "interaction_cohen_dz": cohen_dz(interaction),
                        "interaction_p_signflip": p_value,
                        "interaction_p_signflip_exact": bool(exact),
                    }
                )
    output = pd.DataFrame(rows)
    output["interaction_q_fdr_32_within_session"] = np.nan
    for _, indices in output.groupby("session").groups.items():
        output.loc[indices, "interaction_q_fdr_32_within_session"] = _fdr_bh(
            output.loc[indices, "interaction_p_signflip"].to_numpy(dtype=float)
        )
    return output.sort_values(["session", "interaction_p_signflip"]).reset_index(drop=True)


def behaviour_subject_effects(
    behaviour_path: Path,
    neural_run_values: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    behaviour = pd.read_csv(behaviour_path)
    required = {"subject", "session", "run", "condition_code", "inverse_rt_z"}
    missing = sorted(required - set(behaviour.columns))
    if missing:
        raise ValueError(f"Behaviour table is missing columns: {', '.join(missing)}")
    behaviour = behaviour.loc[
        behaviour["session"].astype(int).eq(OFF_SESSION)
        & behaviour["condition_code"].isin([SHAM, GVS7])
    ].copy()
    run_means = (
        behaviour.groupby(["subject", "run", "condition_code"], as_index=False)["inverse_rt_z"]
        .mean()
        .pivot(index=["subject", "run"], columns="condition_code", values="inverse_rt_z")
        .dropna(subset=[SHAM, GVS7])
    )
    run_means["behaviour_delta_faster"] = run_means[GVS7] - run_means[SHAM]

    early_runs = neural_run_values.loc[
        neural_run_values["feature"].eq("early_late_change")
    ].set_index(["subject", "run"])
    matched = early_runs[
        ["vigour_delta", "task_delta", "interaction_vigour_minus_task"]
    ].join(run_means[["behaviour_delta_faster"]], how="inner")
    matched = matched.reset_index()
    subjects = matched.groupby("subject", as_index=False).agg(
        behaviour_delta_faster=("behaviour_delta_faster", "mean"),
        vigour_delta=("vigour_delta", "mean"),
        task_delta=("task_delta", "mean"),
        interaction_vigour_minus_task=("interaction_vigour_minus_task", "mean"),
        n_runs=("behaviour_delta_faster", "size"),
    )
    subjects["vigour_concordant_change"] = -subjects["vigour_delta"]
    subjects["task_concordant_change"] = -subjects["task_delta"]
    subjects["interaction_concordant_change"] = -subjects["interaction_vigour_minus_task"]

    behaviour_values = subjects["behaviour_delta_faster"].to_numpy(dtype=float)
    behaviour_summary = contrast_summary(behaviour_values, "behaviour")
    behaviour_summary.update(
        {
            "behaviour_n_subjects": int(len(subjects)),
            "behaviour_n_run_pairs": int(subjects["n_runs"].sum()),
            "behaviour_direction": "positive values indicate faster GVS7 responses than sham",
        }
    )
    return subjects.sort_values("subject").reset_index(drop=True), behaviour_summary


def correlation_rows(subjects: pd.DataFrame) -> pd.DataFrame:
    x = subjects["behaviour_delta_faster"].to_numpy(dtype=float)
    outcomes = {
        "vigour": "vigour_concordant_change",
        "task": "task_concordant_change",
        "vigour_minus_task": "interaction_concordant_change",
    }
    rows = []
    for label, column in outcomes.items():
        y = subjects[column].to_numpy(dtype=float)
        pearson = stats.pearsonr(x, y)
        pearson_ci = pearson.confidence_interval(confidence_level=0.95)
        spearman = stats.spearmanr(x, y)
        slope, intercept, slope_r, slope_p, slope_se = stats.linregress(x, y)
        rows.append(
            {
                "outcome": label,
                "n_subjects": int(len(x)),
                "pearson_r": float(pearson.statistic),
                "pearson_ci95_low": float(pearson_ci.low),
                "pearson_ci95_high": float(pearson_ci.high),
                "pearson_p": float(pearson.pvalue),
                "spearman_rho": float(spearman.statistic),
                "spearman_p": float(spearman.pvalue),
                "ols_slope": float(slope),
                "ols_intercept": float(intercept),
                "ols_slope_se": float(slope_se),
                "ols_slope_p": float(slope_p),
            }
        )

    vigour = subjects["vigour_concordant_change"].to_numpy(dtype=float)
    task = subjects["task_concordant_change"].to_numpy(dtype=float)
    observed = float(np.corrcoef(x, vigour)[0, 1] - np.corrcoef(x, task)[0, 1])
    n = len(x)
    signs = np.where(
        ((np.arange(1 << n, dtype=np.uint64)[:, None] >> np.arange(n, dtype=np.uint64)[None, :]) & 1) == 1,
        1.0,
        -1.0,
    )
    midpoint = (vigour + task) / 2.0
    half_difference = (vigour - task) / 2.0
    permuted_vigour = midpoint[None, :] + signs * half_difference[None, :]
    permuted_task = midpoint[None, :] - signs * half_difference[None, :]
    x_centered = x - np.mean(x)

    def row_correlations(values: np.ndarray) -> np.ndarray:
        centered = values - np.mean(values, axis=1, keepdims=True)
        numerator = centered @ x_centered
        denominator = np.sqrt(np.sum(centered**2, axis=1) * np.sum(x_centered**2))
        return np.divide(
            numerator,
            denominator,
            out=np.full(numerator.shape, np.nan),
            where=denominator > 0,
        )

    null_difference = row_correlations(permuted_vigour) - row_correlations(permuted_task)
    comparison_p = float(np.nanmean(np.abs(null_difference) >= abs(observed) - 1e-15))
    rows.append(
        {
            "outcome": "vigour_vs_task_correlation_difference",
            "n_subjects": int(n),
            "pearson_r": observed,
            "pearson_ci95_low": np.nan,
            "pearson_ci95_high": np.nan,
            "pearson_p": comparison_p,
            "spearman_rho": np.nan,
            "spearman_p": np.nan,
            "ols_slope": np.nan,
            "ols_intercept": np.nan,
            "ols_slope_se": np.nan,
            "ols_slope_p": np.nan,
        }
    )
    return pd.DataFrame(rows)


def plot_reviewer_figure(
    subjects: pd.DataFrame,
    stats_table: pd.DataFrame,
    correlation_table: pd.DataFrame,
    output_base: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"],
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#e8e8e8",
            "grid.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    colors = {"vigour": "#2c7fb8", "task": "#d98532"}
    early = stats_table.loc[stats_table["feature"].eq("early_late_change")].iloc[0]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2), constrained_layout=True)

    ax = axes[0, 0]
    behaviour = subjects["behaviour_delta_faster"].to_numpy(dtype=float)
    jitter = np.random.default_rng(0).uniform(-0.06, 0.06, len(behaviour))
    ax.scatter(jitter, behaviour, color="#333333", alpha=0.75, s=30, zorder=3)
    b_mean, b_low, b_high = mean_ci(behaviour)
    ax.errorbar(0, b_mean, yerr=[[b_mean - b_low], [b_high - b_mean]], fmt="D", color="#7f3c8d", lw=2, capsize=5)
    ax.axhline(0, color="black", lw=0.9)
    ax.set_xlim(-0.35, 0.35)
    ax.set_xticks([0])
    ax.set_xticklabels(["GVS7 − sham"])
    ax.set_ylabel("Inverse-RT change (within-session z)")
    ax.set_title("A. Individual behavioural effect")
    ax.text(0.03, 0.96, f"mean={b_mean:.3f}\nexact p={early['behaviour_p_signflip']:.4f}", transform=ax.transAxes, va="top")

    ax = axes[0, 1]
    x_positions = {"task": 0.0, "vigour": 1.0}
    for row in subjects.itertuples(index=False):
        ax.plot(
            [x_positions["task"], x_positions["vigour"]],
            [-row.task_delta, -row.vigour_delta],
            color="#a0a0a0",
            alpha=0.45,
            lw=0.9,
            zorder=1,
        )
    for network, column in [("task", "task_concordant_change"), ("vigour", "vigour_concordant_change")]:
        values = subjects[column].to_numpy(dtype=float)
        xpos = x_positions[network]
        jitter = np.random.default_rng(1 if network == "task" else 2).uniform(-0.06, 0.06, len(values))
        ax.scatter(np.full(len(values), xpos) + jitter, values, color=colors[network], alpha=0.8, s=28, zorder=3)
        mean, low, high = mean_ci(values)
        ax.errorbar(xpos, mean, yerr=[[mean - low], [high - mean]], fmt="D", color=colors[network], lw=2, capsize=5, zorder=4)
    ax.axhline(0, color="black", lw=0.9)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Task map", "Vigour"])
    ax.set_ylabel("Neural GVS7 effect in sham-SD units\n(positive = stronger late-to-early decrease)")
    ax.set_title("B. Scale-comparable network effects")

    ax = axes[1, 0]
    interaction = subjects["interaction_concordant_change"].to_numpy(dtype=float)
    jitter = np.random.default_rng(3).uniform(-0.06, 0.06, len(interaction))
    ax.scatter(jitter, interaction, color="#244f73", alpha=0.78, s=30, zorder=3)
    mean, low, high = mean_ci(interaction)
    ax.errorbar(0, mean, yerr=[[mean - low], [high - mean]], fmt="D", color="#244f73", lw=2, capsize=5)
    ax.axhline(0, color="black", lw=0.9)
    ax.set_xlim(-0.35, 0.35)
    ax.set_xticks([0])
    ax.set_xticklabels(["Vigour − task map"])
    ax.set_ylabel("Direct GVS-effect difference\n(sham-SD units)")
    ax.set_title("C. Network × stimulation interaction")
    ax.text(
        0.03,
        0.96,
        f"Mixed model p={early['mixedlm_interaction_p']:.3f}\n"
        f"exact p={early['interaction_p_signflip']:.3f}",
        transform=ax.transAxes,
        va="top",
    )

    ax = axes[1, 1]
    x = subjects["behaviour_delta_faster"].to_numpy(dtype=float)
    annotation_y = 0.97
    for network, column in [("vigour", "vigour_concordant_change"), ("task", "task_concordant_change")]:
        y = subjects[column].to_numpy(dtype=float)
        ax.scatter(x, y, color=colors[network], alpha=0.72, s=32, label=network.capitalize())
        slope, intercept = np.polyfit(x, y, 1)
        x_line = np.linspace(np.min(x), np.max(x), 100)
        ax.plot(x_line, intercept + slope * x_line, color=colors[network], lw=2)
        correlation = correlation_table.loc[correlation_table["outcome"].eq(network)].iloc[0]
        ax.text(
            0.03,
            annotation_y,
            f"{network.capitalize()}: r={correlation['pearson_r']:.2f}, p={correlation['pearson_p']:.3f}",
            color=colors[network],
            transform=ax.transAxes,
            va="top",
        )
        annotation_y -= 0.08
    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("GVS7 behavioural improvement\n(inverse-RT z: active − sham)")
    ax.set_ylabel("Neural GVS7 effect (concordant direction)")
    ax.set_title("D. Across-patient brain–behaviour relation")
    ax.legend(frameon=False, loc="lower right")

    fig.suptitle("Direct GVS7 network comparison and brain–behaviour association", fontsize=15)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=300)
    fig.savefig(output_base.with_suffix(".pdf"))
    plt.close(fig)


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_report(
    path: Path,
    primary: pd.DataFrame,
    all_waveforms: pd.DataFrame,
    correlations: pd.DataFrame,
) -> None:
    early = primary.loc[primary["feature"].eq("early_late_change")].iloc[0]
    vigour_corr = correlations.loc[correlations["outcome"].eq("vigour")].iloc[0]
    task_corr = correlations.loc[correlations["outcome"].eq("task")].iloc[0]
    correlation_difference = correlations.loc[
        correlations["outcome"].eq("vigour_vs_task_correlation_difference")
    ].iloc[0]
    global_row = all_waveforms.loc[
        all_waveforms["session"].eq(OFF_SESSION)
        & all_waveforms["active_gvs"].eq(GVS7)
        & all_waveforms["feature"].eq("early_late_change")
    ].iloc[0]

    lines = [
        "# GVS7 reviewer-response analysis",
        "",
        "## Analysis definition",
        "",
        "Both neural signals were baseline-centred trial by trial and divided by the standard deviation of baseline-centred sham samples from the matched subject/session/run/network before the four predefined features were extracted. Positive values in the figure's concordant direction mean that GVS7 produced a stronger late-to-early decrease than sham.",
        "",
        "The primary waveform was GVS7 in medication OFF because GVS7 was selected by the independent behavioural analysis. Interaction q-values therefore control FDR across the four predefined GVS7 features. A sensitivity correction across all 8 waveforms x 4 features within session is also reported.",
        "",
        "## Behavioural GVS7 effect",
        "",
        f"Across the {int(early['behaviour_n_subjects'])} patients with matched neural data, the mean GVS7-minus-sham inverse-RT change was {early['behaviour_mean']:+.3f} within-session SD units (95% CI [{early['behaviour_ci95_low']:+.3f}, {early['behaviour_ci95_high']:+.3f}], exact sign-flip p={early['behaviour_p_signflip']:.4f}, dz={early['behaviour_cohen_dz']:.3f}). Positive values indicate faster responses.",
        "",
        "## Direct standardized neural comparison",
        "",
        f"For the late-minus-early feature, the standardized GVS7-minus-sham effect was {early['vigour_mean']:+.3f} sham-SD units in the vigour projection and {early['task_mean']:+.3f} sham-SD units in the task-map signal. The direct vigour-minus-task interaction was {early['interaction_mean']:+.3f} (95% CI [{early['interaction_ci95_low']:+.3f}, {early['interaction_ci95_high']:+.3f}], dz={early['interaction_cohen_dz']:.3f}). The run-level mixed-effects interaction was p={early['mixedlm_interaction_p']:.4f}; the exact subject-level sensitivity test was p={early['interaction_p_signflip']:.4f}, q={early['interaction_q_signflip_fdr_4_features']:.4f} across four features. Under the broader 32-test session family, q={global_row['interaction_q_fdr_32_within_session']:.4f}.",
        "",
        "## Individual brain-behaviour association",
        "",
        f"Behavioural improvement was not significantly related to the vigour-network effect (Pearson r={vigour_corr['pearson_r']:+.3f}, 95% CI [{vigour_corr['pearson_ci95_low']:+.3f}, {vigour_corr['pearson_ci95_high']:+.3f}], p={vigour_corr['pearson_p']:.4f}; Spearman rho={vigour_corr['spearman_rho']:+.3f}, p={vigour_corr['spearman_p']:.4f}). The corresponding task-map association was Pearson r={task_corr['pearson_r']:+.3f}, p={task_corr['pearson_p']:.4f}. The vigour and task-map correlations did not differ in the within-subject network-label permutation test (difference in r={correlation_difference['pearson_r']:+.3f}, p={correlation_difference['pearson_p']:.4f}).",
        "",
        "## Reviewer-facing conclusion",
        "",
        "The analysis directly addresses the requested interaction and patient-level association. It supports a reliable OFF-medication behavioural GVS7 effect and a group-level vigour-projection response. Whether it supports greater vigour-network specificity depends on the standardized interaction above; a nonsignificant interaction should be reported as no detected network difference, not as evidence that the two networks are equivalent. The individual-level result should be reported as null and exploratory.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_rebuttal_and_manuscript_text(
    path: Path,
    primary: pd.DataFrame,
    correlations: pd.DataFrame,
) -> None:
    early = primary.loc[primary["feature"].eq("early_late_change")].iloc[0]
    vigour_corr = correlations.loc[correlations["outcome"].eq("vigour")].iloc[0]
    task_corr = correlations.loc[correlations["outcome"].eq("task")].iloc[0]
    corr_difference = correlations.loc[
        correlations["outcome"].eq("vigour_vs_task_correlation_difference")
    ].iloc[0]
    lines = [
        "# Suggested reviewer response and manuscript revisions",
        "",
        "## Response to Reviewer 1",
        "",
        "> We thank the reviewer for identifying that our original comparison relied on different significance outcomes in the two maps rather than a direct test. We have now placed the vigour-projection and task-activation-map signals on a common, within-run scale. For each participant, session, run, and network definition, trial time courses were baseline-centred and divided by the standard deviation of the matched sham response before extracting the four predefined temporal features. We then fitted a direct network-by-stimulation mixed-effects model for the OFF-medication GVS7-versus-sham comparison, with run included as a fixed effect and participant/run dependence represented by random effects. GVS7 was the planned waveform because it was selected by the independent behavioural analysis.",
        "",
        f"> The matched-patient behavioural effect remained reliable: GVS7 increased inverse RT by {early['behaviour_mean']:+.3f} within-session SD units relative to sham (95% CI [{early['behaviour_ci95_low']:+.3f}, {early['behaviour_ci95_high']:+.3f}], exact p={early['behaviour_p_signflip']:.4f}, n={int(early['behaviour_n_subjects'])}). On the standardized neural scale, the late-minus-early GVS7 effect was {early['vigour_mean']:+.3f} sham-SD units in the vigour projection and {early['task_mean']:+.3f} sham-SD units in the task-map signal. The direct network-by-stimulation interaction was not significant (mixed-model estimate={early['mixedlm_interaction_estimate']:+.3f}, 95% CI [{early['mixedlm_interaction_ci95_low']:+.3f}, {early['mixedlm_interaction_ci95_high']:+.3f}], p={early['mixedlm_interaction_p']:.4f}; exact subject-level sensitivity p={early['interaction_p_signflip']:.4f}, four-feature FDR q={early['interaction_q_signflip_fdr_4_features']:.4f}). We therefore removed the claim that the vigour network captured the GVS7 waveform more specifically than the task-activation map.",
        "",
        f"> We also added the requested individual-patient analysis. The GVS7 behavioural improvement was not significantly associated with the vigour-projection change (Pearson r={vigour_corr['pearson_r']:+.3f}, 95% CI [{vigour_corr['pearson_ci95_low']:+.3f}, {vigour_corr['pearson_ci95_high']:+.3f}], p={vigour_corr['pearson_p']:.4f}; Spearman rho={vigour_corr['spearman_rho']:+.3f}, p={vigour_corr['spearman_p']:.4f}). The task-map association was also nonsignificant (Pearson r={task_corr['pearson_r']:+.3f}, p={task_corr['pearson_p']:.4f}), and the two correlations did not differ (within-participant network-label permutation p={corr_difference['pearson_p']:.4f}). We now describe the behavioural and neural findings as co-occurring group-level effects rather than evidence of patient-level coupling.",
        "",
        "## Response to Reviewer 2",
        "",
        "> We agree and have replaced the comparison of separate significance decisions with an explicit network-by-stimulation interaction. As described above, the standardized GVS7 interaction was not significant, so we removed statements that the vigour projection was more specific or that the task-activation map did not respond. We now report that both neural summaries showed a group-level GVS7-related late-to-early change, with no detected difference between them. We made the same inferential distinction throughout the manuscript: claims of differences between network definitions are retained only where a direct interaction supports them; otherwise the comparisons are labelled descriptive or the specificity language has been removed.",
        "",
        "## Proposed Methods paragraph",
        "",
        "To compare GVS effects directly between the vigour projection and task-activation-map signal, we first placed the two neural summaries on a comparable within-run scale. Trial time courses were baseline-centred, and each network signal was divided by the standard deviation of all baseline-centred sham samples from the matched participant, medication session, and run. The four predefined temporal features were then extracted from these standardized signals. The primary direct comparison concerned GVS7 in the OFF-medication session because GVS7 was selected independently by the behavioural analysis. For each feature, we fitted a mixed-effects model containing network definition, stimulation condition (GVS7 or sham), their interaction, and run, with repeated participant/run observations modelled as random effects. The network-by-stimulation interaction tested whether the GVS7-minus-sham effect differed between the vigour and task-map signals. As a distribution-free sensitivity analysis, run-level difference-of-differences values were averaged within participant and tested with an exact two-sided sign-flip test. FDR was controlled across the four predefined features. We additionally correlated participant-level GVS7-minus-sham inverse-RT changes with the corresponding neural changes using Pearson and Spearman correlations.",
        "",
        "## Proposed Results replacement for the specificity paragraph",
        "",
        f"GVS7 produced faster OFF-medication responses than sham in the {int(early['behaviour_n_subjects'])} participants with matched neural data (mean inverse-RT change={early['behaviour_mean']:+.3f} within-session SD units, 95% CI [{early['behaviour_ci95_low']:+.3f}, {early['behaviour_ci95_high']:+.3f}], exact p={early['behaviour_p_signflip']:.4f}). After sham-SD standardization of both neural signals, GVS7 reduced the late relative to the early response in both the vigour projection (mean GVS7-minus-sham={early['vigour_mean']:+.3f}, exact p={early['vigour_p_signflip']:.4f}, four-feature q={early['vigour_q_signflip_fdr_4_features']:.4f}) and the task-activation-map signal (mean={early['task_mean']:+.3f}, exact p={early['task_p_signflip']:.4f}, four-feature q={early['task_q_signflip_fdr_4_features']:.4f}). Crucially, the direct network-by-stimulation interaction was not significant (mixed-model estimate={early['mixedlm_interaction_estimate']:+.3f}, 95% CI [{early['mixedlm_interaction_ci95_low']:+.3f}, {early['mixedlm_interaction_ci95_high']:+.3f}], p={early['mixedlm_interaction_p']:.4f}; exact participant-level sensitivity p={early['interaction_p_signflip']:.4f}). Thus, the standardized analysis did not show that GVS7 affected the vigour projection more strongly than the task-activation-map signal.",
        "",
        f"Across participants, behavioural improvement under GVS7 was not significantly related to the vigour-projection change (Pearson r={vigour_corr['pearson_r']:+.3f}, 95% CI [{vigour_corr['pearson_ci95_low']:+.3f}, {vigour_corr['pearson_ci95_high']:+.3f}], p={vigour_corr['pearson_p']:.4f}; Spearman rho={vigour_corr['spearman_rho']:+.3f}, p={vigour_corr['spearman_p']:.4f}). The corresponding task-map association was also nonsignificant (Pearson r={task_corr['pearson_r']:+.3f}, p={task_corr['pearson_p']:.4f}), with no detected difference between the two correlations (permutation p={corr_difference['pearson_p']:.4f}). The behavioural and neural changes therefore co-occurred at the group level but were not demonstrably coupled across individual participants.",
        "",
        "## Proposed Discussion replacement",
        "",
        "The exploratory GVS results identify a reliable OFF-medication behavioural effect of GVS7 and corresponding group-level changes in neural response dynamics. However, the direct standardized comparison did not show that the GVS7 effect differed between the vigour projection and the broader task-activation-map signal. Moreover, participants with larger behavioural improvements did not reliably show larger neural changes. These findings indicate that GVS7 influenced behaviour and task-related neural dynamics at the group level, but they do not establish preferential modulation of the vigour network or patient-level brain-behaviour coupling. This narrower interpretation does not affect the primary identification and validation of the vigour-related projection, which was defined without medication or GVS labels.",
        "",
        "## Short replacements for Abstract and Conclusion",
        "",
        "**Abstract:** In exploratory analyses, GVS7 was associated with faster OFF-medication responses and with group-level changes in neural response dynamics. Direct standardized comparisons did not establish that this response was specific to the vigour projection relative to the task-activation map, and individual behavioural and neural effects were not significantly associated.",
        "",
        "**Conclusion:** GVS7 produced an OFF-medication behavioural effect accompanied by group-level changes in both neural summaries, without evidence of differential vigour-network sensitivity or individual-level brain-behaviour coupling.",
        "",
        "## Important manuscript-wide consistency revision",
        "",
        "The medication-connectivity statement that the effect was 'clearer' or 'more specific' in the vigour network should also be removed. The existing direct network x medication x FC-type analysis gives p=0.627, so the defensible conclusion is that the local-versus-distributed medication effect was detected across the network definitions, without evidence that its magnitude differed between them. GVS edge-set overlap comparisons should be described as descriptive unless a direct common-edge network interaction is added.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    task_cache = args.out_dir / "task_map_voxel_mean_trials_cache.npy"

    metadata = pd.read_csv(args.metadata, sep="\t").reset_index(drop=True)
    vigour_trials = np.asarray(np.load(args.vigour_trials, mmap_mode="r"), dtype=np.float64)
    task_trials = load_task_mean_trials(
        args.task_trials, args.manifest, task_cache, args.task_chunk_size
    )
    trial_arrays = {"vigour": vigour_trials, "task": task_trials}
    validate_inputs(metadata, trial_arrays)

    run_features, scales = make_standardized_run_features(trial_arrays, metadata)
    primary_runs, primary_subjects, primary_stats = primary_gvs7_analysis(run_features)
    all_stats = all_waveform_sensitivity(run_features)
    behaviour_subjects, behaviour_stats = behaviour_subject_effects(
        args.behaviour, primary_runs
    )
    correlations = correlation_rows(behaviour_subjects)
    for key, value in behaviour_stats.items():
        primary_stats[key] = value

    run_features.to_csv(args.out_dir / "standardized_run_features.csv", index=False)
    scales.to_csv(args.out_dir / "sham_scaling_values.csv", index=False)
    primary_runs.to_csv(args.out_dir / "gvs7_standardized_run_values.csv", index=False)
    primary_subjects.to_csv(args.out_dir / "gvs7_standardized_subject_values.csv", index=False)
    primary_stats.to_csv(args.out_dir / "gvs7_standardized_interaction_stats.csv", index=False)
    all_stats.to_csv(args.out_dir / "all_waveform_standardized_interaction_stats.csv", index=False)
    behaviour_subjects.to_csv(args.out_dir / "gvs7_brain_behaviour_subject_values.csv", index=False)
    correlations.to_csv(args.out_dir / "gvs7_brain_behaviour_correlations.csv", index=False)

    plot_reviewer_figure(
        behaviour_subjects,
        primary_stats,
        correlations,
        args.out_dir / "gvs7_reviewer_response",
    )
    write_report(
        args.out_dir / "gvs7_reviewer_response_report.md",
        primary_stats,
        all_stats,
        correlations,
    )
    write_rebuttal_and_manuscript_text(
        args.out_dir / "rebuttal_and_manuscript_text.md",
        primary_stats,
        correlations,
    )
    results = {
        "normalization": "trial baseline-centering followed by matched subject/session/run/network sham-response SD scaling",
        "primary_stats": primary_stats.to_dict(orient="records"),
        "brain_behaviour": correlations.to_dict(orient="records"),
        "outputs": {
            "figure_png": str(args.out_dir / "gvs7_reviewer_response.png"),
            "figure_pdf": str(args.out_dir / "gvs7_reviewer_response.pdf"),
            "report": str(args.out_dir / "gvs7_reviewer_response_report.md"),
            "rebuttal_text": str(args.out_dir / "rebuttal_and_manuscript_text.md"),
        },
    }
    (args.out_dir / "gvs7_reviewer_response_results.json").write_text(
        json.dumps(jsonable(results), indent=2) + "\n", encoding="utf-8"
    )

    print(primary_stats.to_string(index=False, float_format=lambda value: f"{value:.6g}"))
    print()
    print(correlations.to_string(index=False, float_format=lambda value: f"{value:.6g}"))
    print(f"\nSaved reviewer-response outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
