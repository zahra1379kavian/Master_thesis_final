#!/usr/bin/env python3
"""Evaluate low-vs-high reward effects on inverse reaction time."""

from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path

import matplotlib

warnings.filterwarnings("ignore", message="Unable to import Axes3D.*")
matplotlib.use("Agg")
matplotlib.rcParams.update(
    {
        "font.family": "Liberation Sans",
        "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"],
    }
)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.io import loadmat

try:
    import statsmodels.formula.api as smf
except ImportError:  # pragma: no cover - optional dependency
    smf = None


ROOT = Path(__file__).resolve().parent
DEFAULT_BEHAVIOUR_ROOT = Path(
    "/mnt/TeamShare/Data_Masterfile/H20-00572_All-Dressed/"
    "PRECISIONSTIM_PD_Data_Results/Behaviour"
)
DEFAULT_OUT_DIR = ROOT / "figures" / "reward_effects"
DEFAULT_LOW_REWARD_CODES = (0, 1)
DEFAULT_HIGH_REWARD_CODES = (5,)
DEFAULT_RT_COLUMN_ONE_BASED = 2
CONSOLIDATED_RE = re.compile(r"^(?P<subject>PSPD\d+)_(?P<medication>ON|OFF)_consolidated_behavdata\.mat$")
GVS_LABELS = ["Sham"] + [f"GVS{i}" for i in range(2, 10)]
SESSION_BY_MEDICATION = {"OFF": 1, "ON": 2}
SUBJECT_MEDICATION_SESSION_OVERRIDES = {
    (17, "OFF"): 2,
    (17, "ON"): 1,
}
DEFAULT_EXCLUDED_SUBJECT_SESSIONS = ((17, 1),)
SUBJECT_FIGURE_COLOR = "#F28B82"
MEDICATION_FIGURE_COLOR = "#8FD19E"
GVS_FIGURE_COLOR = "#8EC5F6"
PAIRED_FIGURE_SIZE = (4.2, 4.4)
PAIRED_COLUMN_X = np.array([0.0, 0.42])
PAIRED_COLUMN_X_LIMITS = (-0.08, 0.5)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load PRECISIONSTIM behaviour .mat files and compare low- vs "
            "high-reward inverse reaction time trials."
        )
    )
    parser.add_argument("--behaviour-root", type=Path, default=DEFAULT_BEHAVIOUR_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--low-reward-codes", type=float, nargs="+", default=list(DEFAULT_LOW_REWARD_CODES))
    parser.add_argument("--high-reward-codes", type=float, nargs="+", default=list(DEFAULT_HIGH_REWARD_CODES))
    parser.add_argument(
        "--rt-column",
        type=int,
        default=DEFAULT_RT_COLUMN_ONE_BASED,
        help="One-based behaviour metric column containing inverse RT (1/RT). Default: 2.",
    )
    parser.add_argument("--min-rt", type=float, default=0.0, help="Minimum inverse RT value to include.")
    parser.add_argument("--max-rt", type=float, default=None, help="Maximum inverse RT value to include.")
    return parser.parse_args()


def _load_reward(path: Path) -> np.ndarray:
    mat = loadmat(path, squeeze_me=False, struct_as_record=False)
    if "res" not in mat:
        raise RuntimeError(f"{path} does not contain variable 'res'")
    res = mat["res"][0, 0]
    if not hasattr(res, "reward"):
        raise RuntimeError(f"{path} variable 'res' does not contain field 'reward'")
    reward = np.asarray(res.reward, dtype=np.float64)
    if reward.ndim != 2:
        raise RuntimeError(f"{path} res.reward must be 2D, got shape {reward.shape}")
    return reward


def _load_behav_metrics(path: Path) -> list[np.ndarray]:
    mat = loadmat(path, squeeze_me=False, struct_as_record=False)
    if "behav_metrics" not in mat:
        raise RuntimeError(f"{path} does not contain variable 'behav_metrics'")
    cells = np.asarray(mat["behav_metrics"], dtype=object).ravel()
    metrics = [np.asarray(cell, dtype=np.float64) for cell in cells]
    if not metrics:
        raise RuntimeError(f"{path} behav_metrics is empty")
    return metrics


def _reward_level(code: float, low_codes: set[float], high_codes: set[float]) -> str | None:
    if not np.isfinite(code):
        return None
    rounded = float(int(round(code)))
    if rounded in low_codes:
        return "low"
    if rounded in high_codes:
        return "high"
    return None


def _subject_number(subject: str) -> int:
    match = re.search(r"\d+", str(subject))
    if match is None:
        raise RuntimeError(f"Could not parse subject number from {subject!r}")
    return int(match.group(0))


def _session_from_medication(medication: str) -> int:
    key = str(medication).upper()
    if key not in SESSION_BY_MEDICATION:
        raise RuntimeError(f"Could not map medication label {medication!r} to a session number")
    return SESSION_BY_MEDICATION[key]


def _session_from_subject_medication(subject: str, medication: str) -> int:
    key = (_subject_number(subject), str(medication).upper())
    if key in SUBJECT_MEDICATION_SESSION_OVERRIDES:
        return SUBJECT_MEDICATION_SESSION_OVERRIDES[key]
    return _session_from_medication(medication)


def _apply_subject_session_exclusions(
    pairs: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    exclusions = {(int(subject), int(session)) for subject, session in DEFAULT_EXCLUDED_SUBJECT_SESSIONS}
    included = []
    excluded = []
    for pair in pairs:
        pair = dict(pair)
        pair["session"] = _session_from_subject_medication(str(pair["subject"]), str(pair["medication"]))
        key = (_subject_number(str(pair["subject"])), int(pair["session"]))
        if key in exclusions:
            excluded.append(pair)
        else:
            included.append(pair)
    if not included:
        raise RuntimeError("No behaviour files remain after subject/session exclusions")
    return included, excluded


def _discover_pairs(behaviour_root: Path) -> list[dict[str, object]]:
    reward_dir = behaviour_root / "Consolidated_behav_data"
    metrics_dir = behaviour_root / "Behaviour_metrics_revised"
    if not reward_dir.exists():
        raise FileNotFoundError(f"Missing reward directory: {reward_dir}")
    if not metrics_dir.exists():
        raise FileNotFoundError(f"Missing behaviour metrics directory: {metrics_dir}")

    pairs = []
    for reward_path in sorted(reward_dir.glob("*_consolidated_behavdata.mat")):
        match = CONSOLIDATED_RE.match(reward_path.name)
        if match is None:
            warnings.warn(f"Skipping unrecognized reward filename: {reward_path.name}", stacklevel=2)
            continue
        stem = f"{match.group('subject')}_{match.group('medication')}"
        metrics_path = metrics_dir / f"{stem}_behav_metrics.mat"
        if not metrics_path.exists():
            raise FileNotFoundError(f"Missing paired behaviour metrics file for {reward_path.name}: {metrics_path}")
        pairs.append(
            {
                "subject": match.group("subject"),
                "medication": match.group("medication"),
                "reward_path": reward_path,
                "metrics_path": metrics_path,
            }
        )
    if not pairs:
        raise RuntimeError(f"No paired behaviour .mat files found under {behaviour_root}")
    return pairs


def _build_trial_table(
    pairs: list[dict[str, object]],
    rt_column_index: int,
    low_codes: set[float],
    high_codes: set[float],
    min_rt: float | None,
    max_rt: float | None,
) -> pd.DataFrame:
    rows = []
    for pair in pairs:
        reward = _load_reward(Path(pair["reward_path"]))
        metrics = _load_behav_metrics(Path(pair["metrics_path"]))
        if len(metrics) < reward.shape[1]:
            raise RuntimeError(
                f"{pair['metrics_path']} has {len(metrics)} GVS cells but reward has {reward.shape[1]} columns"
            )
        for gvs_index in range(reward.shape[1]):
            gvs_number = gvs_index + 1
            metric = metrics[gvs_index]
            if metric.ndim != 2:
                raise RuntimeError(f"{pair['metrics_path']} GVS{gvs_number} metric must be 2D, got {metric.shape}")
            if metric.shape[1] <= rt_column_index:
                raise RuntimeError(
                    f"{pair['metrics_path']} GVS{gvs_number} has {metric.shape[1]} columns; "
                    f"cannot read RT column {rt_column_index + 1}"
                )
            if metric.shape[0] != reward.shape[0]:
                raise RuntimeError(
                    f"{pair['subject']} {pair['medication']} GVS{gvs_number}: "
                    f"reward rows {reward.shape[0]} != behaviour rows {metric.shape[0]}"
                )
            inv_rt = metric[:, rt_column_index]
            for row_index, (reward_code, inv_rt_value) in enumerate(zip(reward[:, gvs_index], inv_rt), start=1):
                level = _reward_level(float(reward_code), low_codes=low_codes, high_codes=high_codes)
                if level is None or not np.isfinite(inv_rt_value) or inv_rt_value <= 0:
                    continue
                inv_rt_value = float(inv_rt_value)
                if min_rt is not None and inv_rt_value < min_rt:
                    continue
                if max_rt is not None and inv_rt_value > max_rt:
                    continue
                rows.append(
                    {
                        "subject": pair["subject"],
                        "medication": pair["medication"],
                        "session": int(
                            pair.get(
                                "session",
                                _session_from_subject_medication(str(pair["subject"]), str(pair["medication"])),
                            )
                        ),
                        "gvs": gvs_number,
                        "gvs_label": GVS_LABELS[gvs_index] if gvs_index < len(GVS_LABELS) else f"GVS{gvs_number}",
                        "run": 1 if row_index <= 10 else 2,
                        "trial_in_gvs": row_index,
                        "trial_in_run": ((row_index - 1) % 10) + 1,
                        "reward_code": float(reward_code),
                        "reward_level": level,
                        "reward_high": 1 if level == "high" else 0,
                        "inv_rt": inv_rt_value,
                        "reward_file": Path(pair["reward_path"]).name,
                        "behav_metrics_file": Path(pair["metrics_path"]).name,
                    }
                )
    if not rows:
        raise RuntimeError("No finite low/high reward inverse RT trials were found")
    return pd.DataFrame(rows).sort_values(["subject", "medication", "gvs", "trial_in_gvs"]).reset_index(drop=True)


def _paired_summary(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    grouped = (
        df.groupby(group_cols + ["reward_level"], dropna=False)
        .agg(mean_inv_rt=("inv_rt", "mean"), median_inv_rt=("inv_rt", "median"), n_trials=("inv_rt", "size"))
        .reset_index()
    )
    mean_pivot = grouped.pivot(index=group_cols, columns="reward_level", values="mean_inv_rt")
    median_pivot = grouped.pivot(index=group_cols, columns="reward_level", values="median_inv_rt")
    n_pivot = grouped.pivot(index=group_cols, columns="reward_level", values="n_trials")
    summary = pd.DataFrame(index=mean_pivot.index)
    summary["mean_inv_rt_low"] = mean_pivot.get("low")
    summary["mean_inv_rt_high"] = mean_pivot.get("high")
    summary["median_inv_rt_low"] = median_pivot.get("low")
    summary["median_inv_rt_high"] = median_pivot.get("high")
    summary["n_low"] = n_pivot.get("low")
    summary["n_high"] = n_pivot.get("high")
    summary = summary.reset_index()
    summary = summary.dropna(subset=["mean_inv_rt_low", "mean_inv_rt_high"]).copy()
    summary["delta_inv_rt_high_minus_low"] = summary["mean_inv_rt_high"] - summary["mean_inv_rt_low"]
    summary["percent_change_high_vs_low"] = 100.0 * summary["delta_inv_rt_high_minus_low"] / summary["mean_inv_rt_low"]
    return summary


def _one_sample_stats(values: np.ndarray, analysis: str, unit: str) -> dict[str, float | str | int]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    row: dict[str, float | str | int] = {
        "analysis": analysis,
        "unit": unit,
        "contrast": "high_minus_low",
        "n": int(values.size),
        "mean_delta_inv_rt": float(np.mean(values)) if values.size else float("nan"),
        "median_delta_inv_rt": float(np.median(values)) if values.size else float("nan"),
        "sd_delta_inv_rt": float(np.std(values, ddof=1)) if values.size > 1 else float("nan"),
        "sem_delta_inv_rt": float(stats.sem(values)) if values.size > 1 else float("nan"),
        "ci95_low": float("nan"),
        "ci95_high": float("nan"),
        "cohen_dz": float("nan"),
        "t_statistic": float("nan"),
        "p_ttest_two_sided": float("nan"),
        "wilcoxon_statistic": float("nan"),
        "p_wilcoxon_two_sided": float("nan"),
    }
    if values.size > 1:
        sd = float(np.std(values, ddof=1))
        sem = float(stats.sem(values))
        ci_low, ci_high = stats.t.interval(0.95, values.size - 1, loc=float(np.mean(values)), scale=sem)
        t_result = stats.ttest_1samp(values, 0.0)
        row.update(
            {
                "ci95_low": float(ci_low),
                "ci95_high": float(ci_high),
                "cohen_dz": float(float(np.mean(values)) / sd) if sd > 0 else float("nan"),
                "t_statistic": float(t_result.statistic),
                "p_ttest_two_sided": float(t_result.pvalue),
            }
        )
        if np.any(values != 0):
            try:
                wilcoxon = stats.wilcoxon(values, alternative="two-sided")
                row["wilcoxon_statistic"] = float(wilcoxon.statistic)
                row["p_wilcoxon_two_sided"] = float(wilcoxon.pvalue)
            except ValueError:
                pass
    return row


def _paired_difference_stats(summary: pd.DataFrame) -> pd.DataFrame:
    rows = [
        _one_sample_stats(
            summary["subject"]["delta_inv_rt_high_minus_low"].to_numpy(dtype=np.float64),
            "primary_subject_collapsed_across_medication_and_gvs",
            "subject",
        ),
        _one_sample_stats(
            summary["subject_medication"]["delta_inv_rt_high_minus_low"].to_numpy(dtype=np.float64),
            "secondary_session_collapsed_across_gvs",
            "subject_medication_session",
        ),
        _one_sample_stats(
            summary["subject_medication_gvs"]["delta_inv_rt_high_minus_low"].to_numpy(dtype=np.float64),
            "exploratory_session_gvs",
            "subject_medication_gvs",
        ),
    ]
    for medication, group in summary["subject_medication"].groupby("medication", sort=True):
        rows.append(
            _one_sample_stats(
                group["delta_inv_rt_high_minus_low"].to_numpy(dtype=np.float64),
                f"secondary_{medication.lower()}_sessions",
                f"subject_{medication.lower()}_session",
            )
        )
    for gvs, group in summary["subject_gvs"].groupby("gvs", sort=True):
        label = str(group["gvs_label"].iloc[0])
        rows.append(
            _one_sample_stats(
                group["delta_inv_rt_high_minus_low"].to_numpy(dtype=np.float64),
                f"exploratory_{label.lower().replace('/', '_')}",
                "subject_gvs",
            )
        )
    medication_pivot = summary["subject_medication"].pivot(
        index="subject", columns="medication", values="delta_inv_rt_high_minus_low"
    )
    if {"OFF", "ON"}.issubset(set(medication_pivot.columns)):
        rows.append(
            _one_sample_stats(
                (medication_pivot["ON"] - medication_pivot["OFF"]).to_numpy(dtype=np.float64),
                "reward_delta_medication_interaction_on_minus_off",
                "subject",
            )
        )
        rows[-1]["contrast"] = "(high_minus_low_ON) - (high_minus_low_OFF)"
    return pd.DataFrame(rows)


def _regression_tables(trials: pd.DataFrame) -> pd.DataFrame:
    if smf is None:
        return pd.DataFrame([{"model": "trial_fixed_effects", "status": "skipped", "reason": "statsmodels is unavailable"}])
    model_data = trials.copy()
    model_data["medication"] = pd.Categorical(model_data["medication"], categories=["OFF", "ON"])
    formulas = {
        "reward_main_subject_fixed_effects": "inv_rt ~ reward_high + C(subject) + C(medication) + C(gvs) + C(run)",
        "reward_by_medication_subject_fixed_effects": (
            "inv_rt ~ reward_high * C(medication) + C(subject) + C(gvs) + C(run)"
        ),
    }
    rows = []
    for model_name, formula in formulas.items():
        try:
            fit = smf.ols(formula, data=model_data).fit(
                cov_type="cluster",
                cov_kwds={"groups": model_data["subject"]},
            )
        except Exception as exc:  # pragma: no cover - depends on statsmodels/patsy
            rows.append({"model": model_name, "status": "failed", "reason": str(exc)})
            continue
        conf_int = fit.conf_int()
        for term, coef in fit.params.items():
            rows.append(
                {
                    "model": model_name,
                    "status": "fit",
                    "term": term,
                    "coefficient": float(coef),
                    "std_error": float(fit.bse[term]) if term in fit.bse else float("nan"),
                    "z_value": float(fit.tvalues[term]) if term in fit.tvalues else float("nan"),
                    "p_value": float(fit.pvalues[term]) if term in fit.pvalues else float("nan"),
                    "ci95_low": float(conf_int.loc[term, 0]) if term in conf_int.index else float("nan"),
                    "ci95_high": float(conf_int.loc[term, 1]) if term in conf_int.index else float("nan"),
                    "n_observations": int(fit.nobs),
                    "n_subjects": int(model_data["subject"].nunique()),
                    "covariance": "cluster_robust_by_subject",
                    "formula": formula,
                }
            )
    return pd.DataFrame(rows)


def _format_p_value(value: float) -> str:
    if not np.isfinite(value):
        return "p = n/a"
    if value < 0.001:
        return "p < 0.001"
    return f"p = {value:.3f}"


def _format_t_test(row: pd.Series, label: str) -> str:
    n = int(row["n"])
    df = n - 1
    return (
        f"{label}: mean={row['mean_delta_inv_rt']:.4f}\n"
        f"t({df})={row['t_statistic']:.2f}, {_format_p_value(float(row['p_ttest_two_sided']))}"
    )


def _add_stat_annotation(ax: plt.Axes, text: str, y: float = 0.98, va: str = "top", fontsize: int = 9) -> None:
    ax.text(
        0.98,
        y,
        text,
        transform=ax.transAxes,
        ha="right",
        va=va,
        fontsize=fontsize,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.0},
    )


def _gvs_stat_annotation(stats_df: pd.DataFrame) -> str:
    rows = stats_df.loc[
        stats_df["analysis"].astype(str).str.startswith("exploratory_")
        & stats_df["unit"].astype(str).eq("subject_gvs")
    ].copy()
    rows = rows.loc[np.isfinite(rows["p_ttest_two_sided"].to_numpy(dtype=float))]
    if rows.empty:
        return "GVS tests: p = n/a"
    row = rows.sort_values("p_ttest_two_sided", ascending=True).iloc[0]
    label = str(row["analysis"]).replace("exploratory_", "").replace("_", "/")
    label = "Sham" if label.lower() == "sham" else label.upper()
    return f"min p: {label}\nt({int(row['n']) - 1})={row['t_statistic']:.2f}, {_format_p_value(float(row['p_ttest_two_sided']))}"


def _save_paired_subject_figure(subject_summary: pd.DataFrame, stats_df: pd.DataFrame, out_dir: Path) -> list[Path]:
    fig, ax = plt.subplots(figsize=PAIRED_FIGURE_SIZE)
    x = PAIRED_COLUMN_X
    for _, row in subject_summary.iterrows():
        ax.plot(
            x,
            [row["mean_inv_rt_low"], row["mean_inv_rt_high"]],
            color=SUBJECT_FIGURE_COLOR,
            linewidth=1.1,
            alpha=0.24,
            zorder=1,
        )
        ax.scatter(
            x,
            [row["mean_inv_rt_low"], row["mean_inv_rt_high"]],
            color=SUBJECT_FIGURE_COLOR,
            s=18,
            alpha=0.35,
            zorder=2,
        )

    means = subject_summary[["mean_inv_rt_low", "mean_inv_rt_high"]].mean().to_numpy(dtype=float)
    sems = subject_summary[["mean_inv_rt_low", "mean_inv_rt_high"]].sem().to_numpy(dtype=float)
    ax.errorbar(
        x,
        means,
        yerr=sems,
        color=SUBJECT_FIGURE_COLOR,
        marker="o",
        markerfacecolor=SUBJECT_FIGURE_COLOR,
        markeredgecolor=SUBJECT_FIGURE_COLOR,
        markeredgewidth=1.2,
        markersize=7,
        linewidth=2.2,
        capsize=4,
        zorder=4,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(["Low reward", "High reward"])
    ax.set_xlim(*PAIRED_COLUMN_X_LIMITS)
    ax.set_ylabel("Inverse reaction time (1/s)")
    primary = stats_df.loc[stats_df["analysis"] == "primary_subject_collapsed_across_medication_and_gvs"].iloc[0]
    _add_stat_annotation(ax, _format_t_test(primary, "High-low"))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    fig.tight_layout()
    return _save_figure(fig, out_dir / "reward_rt_subject_paired")


def _save_medication_delta_figure(subject_medication: pd.DataFrame, stats_df: pd.DataFrame, out_dir: Path) -> list[Path]:
    fig, ax = plt.subplots(figsize=PAIRED_FIGURE_SIZE)
    pivot = subject_medication.pivot(index="subject", columns="medication", values="delta_inv_rt_high_minus_low")
    paired_pivot = pivot.dropna(subset=["OFF", "ON"])
    x = PAIRED_COLUMN_X
    for _, row in paired_pivot.iterrows():
        ax.plot(x, [row["OFF"], row["ON"]], color=MEDICATION_FIGURE_COLOR, linewidth=1.1, alpha=0.24, zorder=1)
        ax.scatter(x, [row["OFF"], row["ON"]], color=MEDICATION_FIGURE_COLOR, s=18, alpha=0.35, zorder=2)
    means = paired_pivot[["OFF", "ON"]].mean().to_numpy(dtype=float)
    sems = paired_pivot[["OFF", "ON"]].sem().to_numpy(dtype=float)
    ax.errorbar(
        x,
        means,
        yerr=sems,
        color=MEDICATION_FIGURE_COLOR,
        marker="o",
        markerfacecolor=MEDICATION_FIGURE_COLOR,
        markeredgecolor=MEDICATION_FIGURE_COLOR,
        markeredgewidth=1.2,
        markersize=7,
        linewidth=2.2,
        capsize=4,
        zorder=4,
    )
    ax.axhline(0, color=MEDICATION_FIGURE_COLOR, linewidth=1.0, linestyle="--", alpha=0.65)
    ax.set_xticks(x)
    ax.set_xticklabels(["OFF medication", "ON medication"])
    ax.set_xlim(*PAIRED_COLUMN_X_LIMITS)
    ax.set_ylabel("high-low(1/s)")
    interaction = stats_df.loc[stats_df["analysis"] == "reward_delta_medication_interaction_on_minus_off"]
    if not interaction.empty:
        _add_stat_annotation(ax, _format_t_test(interaction.iloc[0], "ON-OFF"), y=0.52, va="center")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    fig.tight_layout()
    return _save_figure(fig, out_dir / "reward_rt_medication_delta")


def _save_gvs_delta_figure(subject_gvs: pd.DataFrame, stats_df: pd.DataFrame, out_dir: Path) -> list[Path]:
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    summary = (
        subject_gvs.groupby(["gvs", "gvs_label"], sort=True)["delta_inv_rt_high_minus_low"]
        .agg(["mean", "count", "std"])
        .reset_index()
    )
    summary["sem"] = summary["std"] / np.sqrt(summary["count"])
    x = summary["gvs"].to_numpy(dtype=float)
    ax.errorbar(
        x,
        summary["mean"].to_numpy(dtype=float),
        yerr=summary["sem"].to_numpy(dtype=float),
        color=GVS_FIGURE_COLOR,
        marker="o",
        markerfacecolor=GVS_FIGURE_COLOR,
        markeredgecolor=GVS_FIGURE_COLOR,
        markeredgewidth=1.1,
        markersize=6,
        linewidth=1.8,
        capsize=4,
    )
    ax.axhline(0, color=GVS_FIGURE_COLOR, linewidth=1.0, linestyle="--", alpha=0.65)
    ax.set_xticks(x)
    ax.set_xticklabels(summary["gvs_label"].tolist(), rotation=35, ha="right")
    ax.set_ylabel("high-low(1/s)")
    _add_stat_annotation(ax, _gvs_stat_annotation(stats_df), fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    fig.tight_layout()
    return _save_figure(fig, out_dir / "reward_rt_gvs_delta")


def _save_figure(fig: plt.Figure, stem: Path) -> list[Path]:
    png_path = stem.with_suffix(".png")
    pdf_path = stem.with_suffix(".pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return [png_path, pdf_path]


def _write_report(
    out_dir: Path,
    trials: pd.DataFrame,
    summary: dict[str, pd.DataFrame],
    stats_df: pd.DataFrame,
    regression_df: pd.DataFrame,
    low_codes: set[float],
    high_codes: set[float],
    excluded_codes: list[float],
    excluded_subject_sessions: list[dict[str, object]],
) -> None:
    primary = stats_df.loc[stats_df["analysis"] == "primary_subject_collapsed_across_medication_and_gvs"].iloc[0]
    regression_reward = regression_df.loc[
        (regression_df.get("model", "") == "reward_main_subject_fixed_effects")
        & (regression_df.get("term", "") == "reward_high")
    ]
    regression_line = "Trial-level fixed-effect regression was not available or did not fit."
    if not regression_reward.empty:
        row = regression_reward.iloc[0]
        regression_line = (
            f"Trial-level fixed-effect regression reward_high beta = {row['coefficient']:.4f} 1/s "
            f"(95% CI {row['ci95_low']:.4f}, {row['ci95_high']:.4f}; p = {row['p_value']:.3g}), "
            "adjusted for subject, medication, GVS, and run with subject-clustered standard errors."
        )
    excluded_subject_session_text = ", ".join(
        f"{row['subject']} session {row['session']} ({row['medication']})" for row in excluded_subject_sessions
    )
    if not excluded_subject_session_text:
        excluded_subject_session_text = "none"

    report = [
        "# Reward Effects on Inverse Reaction Time",
        "",
        f"- Low reward codes: {sorted(low_codes)}",
        f"- High reward codes: {sorted(high_codes)}",
        f"- Excluded reward codes: {excluded_codes}",
        f"- Excluded subject/session entries: {excluded_subject_session_text}",
        f"- Included finite inverse RT trials: {len(trials)}",
        f"- Subjects: {summary['subject']['subject'].nunique()}",
        "",
        "## Primary Paired Result",
        "",
        (
            f"Subject-level inverse RT high - low reward delta = {primary['mean_delta_inv_rt']:.4f} 1/s "
            f"(95% CI {primary['ci95_low']:.4f}, {primary['ci95_high']:.4f}; "
            f"paired t p = {primary['p_ttest_two_sided']:.3g}; "
            f"Wilcoxon p = {primary['p_wilcoxon_two_sided']:.3g}; n = {int(primary['n'])})."
        ),
        "",
        regression_line,
        "",
        "Positive high-minus-low inverse RT values mean high-reward trials were faster.",
    ]
    (out_dir / "reward_rt_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    if args.rt_column < 1:
        raise ValueError("--rt-column is one-based and must be >= 1")
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stale_mixedlm = out_dir / "reward_rt_mixedlm_statistics.csv"
    if stale_mixedlm.exists():
        stale_mixedlm.unlink()

    low_codes = {float(int(round(code))) for code in args.low_reward_codes}
    high_codes = {float(int(round(code))) for code in args.high_reward_codes}
    if low_codes & high_codes:
        raise ValueError(f"Low/high reward code sets overlap: {sorted(low_codes & high_codes)}")

    pairs, excluded_pairs = _apply_subject_session_exclusions(_discover_pairs(args.behaviour_root))
    trials = _build_trial_table(
        pairs,
        rt_column_index=args.rt_column - 1,
        low_codes=low_codes,
        high_codes=high_codes,
        min_rt=args.min_rt,
        max_rt=args.max_rt,
    )
    observed_codes = sorted(float(code) for code in trials["reward_code"].unique())
    all_reward_codes = []
    for pair in pairs:
        reward = _load_reward(Path(pair["reward_path"]))
        all_reward_codes.extend(float(code) for code in np.ravel(reward[np.isfinite(reward)]))
    excluded_codes = sorted(set(float(int(round(code))) for code in all_reward_codes) - low_codes - high_codes)

    summary = {
        "subject": _paired_summary(trials, ["subject"]),
        "subject_medication": _paired_summary(trials, ["subject", "medication"]),
        "subject_gvs": _paired_summary(trials, ["subject", "gvs", "gvs_label"]),
        "subject_medication_gvs": _paired_summary(trials, ["subject", "medication", "gvs", "gvs_label"]),
    }
    stats_df = _paired_difference_stats(summary)
    regression_df = _regression_tables(trials)

    trials.to_csv(out_dir / "reward_rt_trials.csv", index=False)
    for name, df in summary.items():
        df.to_csv(out_dir / f"reward_rt_{name}_summary.csv", index=False)
    stats_df.to_csv(out_dir / "reward_rt_paired_statistics.csv", index=False)
    regression_df.to_csv(out_dir / "reward_rt_regression_statistics.csv", index=False)

    figures = []
    figures.extend(_save_paired_subject_figure(summary["subject"], stats_df, out_dir))
    figures.extend(_save_medication_delta_figure(summary["subject_medication"], stats_df, out_dir))
    figures.extend(_save_gvs_delta_figure(summary["subject_gvs"], stats_df, out_dir))

    excluded_subject_sessions = [
        {
            "subject": str(pair["subject"]),
            "session": int(pair["session"]),
            "medication": str(pair["medication"]),
        }
        for pair in excluded_pairs
    ]
    _write_report(
        out_dir,
        trials,
        summary,
        stats_df,
        regression_df,
        low_codes,
        high_codes,
        excluded_codes,
        excluded_subject_sessions,
    )

    metadata = {
        "behaviour_root": str(args.behaviour_root),
        "out_dir": str(out_dir),
        "n_subjects": int(summary["subject"]["subject"].nunique()),
        "n_subject_medication_sessions": int(summary["subject_medication"].shape[0]),
        "n_trial_rows": int(trials.shape[0]),
        "low_reward_codes": sorted(low_codes),
        "high_reward_codes": sorted(high_codes),
        "observed_included_reward_codes": observed_codes,
        "excluded_reward_codes": excluded_codes,
        "excluded_subject_sessions": excluded_subject_sessions,
        "analysis_metric": "inverse_rt",
        "rt_column_one_based": int(args.rt_column),
        "inverse_rt_definition": "inverse RT = behav_metrics{gvs}[:, rt_column]",
        "output_files": sorted(str(path) for path in out_dir.iterdir() if path.is_file()),
        "figures": [str(path) for path in figures],
    }
    (out_dir / "reward_rt_summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    primary = stats_df.loc[stats_df["analysis"] == "primary_subject_collapsed_across_medication_and_gvs"].iloc[0]
    print(f"Wrote reward effect outputs to {out_dir}")
    print(
        "Primary subject-level high-low inverse RT delta: "
        f"{primary['mean_delta_inv_rt']:.4f} 1/s, p={primary['p_ttest_two_sided']:.3g}, n={int(primary['n'])}"
    )


if __name__ == "__main__":
    main()
