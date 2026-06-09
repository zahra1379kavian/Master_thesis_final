#!/usr/bin/env python3
"""Behavioural RT summaries by GVS condition using true run block order."""

from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path

import matplotlib

warnings.filterwarnings("ignore", message="Unable to import Axes3D.*")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

try:
    import statsmodels.formula.api as smf
except ImportError:  # pragma: no cover - optional dependency
    smf = None


ROOT = Path(__file__).resolve().parent
DEFAULT_BEHAVIOUR_DIR = Path(
    "/mnt/TeamShare/Data_Masterfile/H20-00572_All-Dressed/"
    "Zahra-Thesis-Data/fmri_opt_group/behaviour"
)
DEFAULT_INVENTORY = ROOT / "figures" / "GVS_effects" / "common" / "run_condition_inventory.csv"
DEFAULT_OUT_DIR = ROOT / "figures" / "GVS_effects" / "behaviour"
DEFAULT_BEHAVIOUR_COLUMN_ONE_BASED = 2
SESSION_STATES = {1: "OFF", 2: "ON"}
REST_AFTER_STIM_BLOCKS = (3, 6)
REST_TRIALS = 20
SHAM_CODE = "gvs-01"
BEHAVIOUR_RE = re.compile(r"^PSPD(?P<digits>\d+)_ses_(?P<session>\d+)_run_(?P<run>\d+)\.npy$")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract behaviour RT by GVS/sham blocks from run_condition_inventory.csv "
            "and test active GVS conditions against sham."
        )
    )
    parser.add_argument("--behaviour-dir", type=Path, default=DEFAULT_BEHAVIOUR_DIR)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--behaviour-column",
        type=int,
        default=DEFAULT_BEHAVIOUR_COLUMN_ONE_BASED,
        help="One-based column for 2D behaviour arrays. Default 2 matches existing medication scripts.",
    )
    parser.add_argument(
        "--input-is-inverse-rt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Treat behaviour values as inverse RT and convert to seconds with 1/value.",
    )
    parser.add_argument("--min-rt", type=float, default=0.0)
    parser.add_argument("--max-rt", type=float, default=None)
    return parser.parse_args()


def _subject_digits(subject: str) -> str:
    match = re.search(r"(\d+)$", str(subject))
    if match is None:
        raise ValueError(f"Could not parse subject digits from {subject!r}")
    return match.group(1).zfill(3)


def _behaviour_path(behaviour_dir: Path, subject: str, session: int, run: int) -> Path:
    return behaviour_dir / f"PSPD{_subject_digits(subject)}_ses_{int(session)}_run_{int(run)}.npy"


def _load_behaviour_metric(path: Path, column_index: int) -> tuple[np.ndarray, str]:
    values = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)
    if values.ndim == 1:
        return values, "rt_vector"
    if values.ndim == 2:
        if values.shape[1] <= column_index:
            raise RuntimeError(f"{path} has {values.shape[1]} columns; cannot read column {column_index + 1}")
        return values[:, column_index], f"matrix_column_{column_index + 1}"
    raise RuntimeError(f"{path} must be 1D or 2D, got shape {values.shape}")


def _rt_seconds(values: np.ndarray, input_is_inverse_rt: bool) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    out = np.full(values.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(values)
    if input_is_inverse_rt:
        keep = finite & (values > 0)
        out[keep] = 1.0 / values[keep]
    else:
        out[finite] = values[finite]
    return out


def _consecutive_rt_variability(values: np.ndarray) -> tuple[float, int]:
    values = np.asarray(values, dtype=np.float64)
    if values.size < 2:
        return float("nan"), 0
    previous = values[:-1]
    following = values[1:]
    denominator = previous**2 + following**2
    valid = np.isfinite(previous) & np.isfinite(following) & np.isfinite(denominator) & (denominator > 0)
    if not np.any(valid):
        return float("nan"), 0
    score = np.sum(((previous[valid] - following[valid]) ** 2) / denominator[valid])
    return float(score), int(np.count_nonzero(valid))


def _rest_gap_count(block_position: int) -> int:
    return sum(1 for block in REST_AFTER_STIM_BLOCKS if int(block_position) > int(block))


def _add_rest_aware_timing(inventory: pd.DataFrame) -> pd.DataFrame:
    out = inventory.copy()
    out["block_position"] = (out["trial_start"].astype(int) // 10) + 1
    out["rest_gap_count_before_block"] = out["block_position"].map(_rest_gap_count).astype(int)
    out["rest_aware_trial_start"] = out["trial_start"].astype(int) + out["rest_gap_count_before_block"] * REST_TRIALS
    out["rest_aware_trial_stop"] = out["rest_aware_trial_start"] + out["n_trials"].astype(int)
    out["follows_rest"] = out["block_position"].isin([block + 1 for block in REST_AFTER_STIM_BLOCKS])
    out["precedes_rest"] = out["block_position"].isin(REST_AFTER_STIM_BLOCKS)
    return out


def _load_inventory(path: Path) -> pd.DataFrame:
    inventory = pd.read_csv(path)
    required = {
        "subject",
        "session",
        "medication",
        "run",
        "condition_code",
        "condition_label",
        "trial_start",
        "trial_stop",
        "n_trials",
        "run_n_trials",
    }
    missing = sorted(required - set(inventory.columns))
    if missing:
        raise RuntimeError(f"{path} is missing required columns: {', '.join(missing)}")
    inventory = _add_rest_aware_timing(inventory)
    return inventory.sort_values(["subject", "session", "run", "trial_start"]).reset_index(drop=True)


def _build_trial_table(
    inventory: pd.DataFrame,
    behaviour_dir: Path,
    behaviour_column_index: int,
    input_is_inverse_rt: bool,
    min_rt: float | None,
    max_rt: float | None,
) -> tuple[pd.DataFrame, list[str]]:
    rows: list[dict[str, object]] = []
    missing: list[str] = []
    cache: dict[tuple[str, int, int], tuple[np.ndarray, str, Path]] = {}

    for row in inventory.itertuples(index=False):
        subject = str(row.subject)
        session = int(row.session)
        run = int(row.run)
        key = (subject, session, run)
        if key not in cache:
            path = _behaviour_path(behaviour_dir, subject, session, run)
            if not path.exists():
                missing.append(f"{subject} ses-{session} run-{run}: {path}")
                continue
            metric, source_kind = _load_behaviour_metric(path, behaviour_column_index)
            cache[key] = (metric, source_kind, path)
        metric, source_kind, path = cache[key]

        start = int(row.trial_start)
        stop = int(row.trial_stop)
        stop = min(stop, metric.shape[0])
        if start >= stop:
            continue

        block_metric = metric[start:stop]
        block_rt = _rt_seconds(block_metric, input_is_inverse_rt=input_is_inverse_rt)
        for offset, (input_value, rt_s) in enumerate(zip(block_metric, block_rt), start=1):
            keep = np.isfinite(rt_s)
            if keep and min_rt is not None:
                keep = rt_s >= float(min_rt)
            if keep and max_rt is not None:
                keep = rt_s <= float(max_rt)
            if not keep:
                continue
            rows.append(
                {
                    "subject": subject,
                    "session": session,
                    "medication": str(row.medication),
                    "run": run,
                    "condition_code": str(row.condition_code),
                    "condition_label": str(row.condition_label),
                    "active_gvs": str(row.condition_code) != SHAM_CODE,
                    "block_position": int(row.block_position),
                    "trial_in_condition": int(offset),
                    "task_trial_index_zero_based": int(start + offset - 1),
                    "rest_aware_trial_index_zero_based": int(row.rest_aware_trial_start + offset - 1),
                    "follows_rest": bool(row.follows_rest),
                    "precedes_rest": bool(row.precedes_rest),
                    "input_metric_value": float(input_value),
                    "rt_s": float(rt_s),
                    "rt_ms": float(rt_s * 1000.0),
                    "behaviour_source_kind": source_kind,
                    "behaviour_path": str(path),
                }
            )

    if missing:
        missing_lines = "\n".join(f"- {item}" for item in missing)
        raise FileNotFoundError(f"Missing behaviour files:\n{missing_lines}")
    if not rows:
        raise RuntimeError("No finite GVS behaviour trials were extracted.")
    return pd.DataFrame(rows).sort_values(
        ["subject", "session", "run", "task_trial_index_zero_based"]
    ).reset_index(drop=True), missing


def _block_summary(trials: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = [
        "subject",
        "session",
        "medication",
        "run",
        "condition_code",
        "condition_label",
        "active_gvs",
        "block_position",
        "follows_rest",
        "precedes_rest",
    ]
    for key, group in trials.groupby(group_cols, sort=True):
        record = dict(zip(group_cols, key))
        values = group.sort_values("task_trial_index_zero_based")["rt_s"].to_numpy(dtype=np.float64)
        variability, n_pairs = _consecutive_rt_variability(values)
        record.update(
            {
                "n_rt": int(values.size),
                "mean_rt_s": float(np.mean(values)),
                "mean_rt_ms": float(np.mean(values) * 1000.0),
                "median_rt_ms": float(np.median(values) * 1000.0),
                "sd_rt_ms": float(np.std(values, ddof=1) * 1000.0) if values.size > 1 else float("nan"),
                "mean_inverse_rt": float(np.mean(1.0 / values)),
                "rt_variability": variability,
                "n_adjacent_pairs": n_pairs,
            }
        )
        rows.append(record)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def _aggregate_condition(blocks: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    return (
        blocks.groupby(group_cols, dropna=False, sort=True)
        .agg(
            n_blocks=("mean_rt_ms", "size"),
            n_rt=("n_rt", "sum"),
            mean_rt_ms=("mean_rt_ms", "mean"),
            median_rt_ms=("median_rt_ms", "mean"),
            sd_rt_ms=("mean_rt_ms", "std"),
            mean_inverse_rt=("mean_inverse_rt", "mean"),
            rt_variability=("rt_variability", "mean"),
            n_adjacent_pairs=("n_adjacent_pairs", "sum"),
        )
        .reset_index()
    )


def _one_sample_stats(values: np.ndarray, analysis: str, metric: str, unit: str, contrast: str) -> dict[str, object]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    row: dict[str, object] = {
        "analysis": analysis,
        "metric": metric,
        "unit": unit,
        "contrast": contrast,
        "n": int(values.size),
        "mean_delta": float(np.mean(values)) if values.size else float("nan"),
        "median_delta": float(np.median(values)) if values.size else float("nan"),
        "sd_delta": float(np.std(values, ddof=1)) if values.size > 1 else float("nan"),
        "sem_delta": float(stats.sem(values)) if values.size > 1 else float("nan"),
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


def _active_minus_sham(summary: pd.DataFrame, index_cols: list[str], metric: str) -> pd.DataFrame:
    sham = summary.loc[summary["condition_code"].eq(SHAM_CODE), index_cols + [metric]].rename(columns={metric: "sham"})
    active = summary.loc[~summary["condition_code"].eq(SHAM_CODE)].copy()
    merged = active.merge(sham, on=index_cols, how="inner")
    merged[f"delta_{metric}_active_minus_sham"] = merged[metric] - merged["sham"]
    return merged


def _paired_stats(
    subject_condition: pd.DataFrame,
    session_condition: pd.DataFrame,
    block_condition: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = ["mean_rt_ms", "mean_inverse_rt", "rt_variability"]
    rows = []
    deltas = []
    for metric in metrics:
        subject_delta = _active_minus_sham(subject_condition, ["subject"], metric)
        subject_any = (
            subject_delta.groupby("subject", as_index=False)[f"delta_{metric}_active_minus_sham"].mean()
        )
        rows.append(
            _one_sample_stats(
                subject_any[f"delta_{metric}_active_minus_sham"].to_numpy(dtype=np.float64),
                "subject_any_active_vs_sham",
                metric,
                "subject",
                "any_active_minus_sham",
            )
        )
        subject_any["analysis"] = "subject_any_active_vs_sham"
        subject_any["metric"] = metric
        subject_any["condition_label"] = "ANY_GVS"
        subject_any["condition_code"] = "ANY_GVS"
        deltas.append(subject_any)

        session_delta = _active_minus_sham(session_condition, ["subject", "session", "medication"], metric)
        session_any = (
            session_delta.groupby(["subject", "session", "medication"], as_index=False)[
                f"delta_{metric}_active_minus_sham"
            ].mean()
        )
        rows.append(
            _one_sample_stats(
                session_any[f"delta_{metric}_active_minus_sham"].to_numpy(dtype=np.float64),
                "session_any_active_vs_sham",
                metric,
                "subject_session",
                "any_active_minus_sham",
            )
        )
        session_any["analysis"] = "session_any_active_vs_sham"
        session_any["metric"] = metric
        session_any["condition_label"] = "ANY_GVS"
        session_any["condition_code"] = "ANY_GVS"
        deltas.append(session_any)

        for condition, group in subject_delta.groupby("condition_label", sort=True):
            condition_code = str(group["condition_code"].iloc[0])
            values = group[f"delta_{metric}_active_minus_sham"].to_numpy(dtype=np.float64)
            rows.append(
                _one_sample_stats(
                    values,
                    f"subject_{condition}_vs_sham",
                    metric,
                    "subject",
                    f"{condition}_minus_sham",
                )
            )
            copy = group.loc[:, ["subject", "condition_code", "condition_label", f"delta_{metric}_active_minus_sham"]].copy()
            copy["analysis"] = "subject_condition_vs_sham"
            copy["metric"] = metric
            deltas.append(copy)

        for medication, group in session_delta.groupby("medication", sort=True):
            medication_any = (
                group.groupby(["subject", "session", "medication"], as_index=False)[
                    f"delta_{metric}_active_minus_sham"
                ].mean()
            )
            rows.append(
                _one_sample_stats(
                    medication_any[f"delta_{metric}_active_minus_sham"].to_numpy(dtype=np.float64),
                    f"{medication}_session_any_active_vs_sham",
                    metric,
                    "subject_session",
                    "any_active_minus_sham",
                )
            )

        block_delta = _active_minus_sham(block_condition, ["subject", "session", "medication", "run"], metric)
        block_any = (
            block_delta.groupby(["subject", "session", "medication", "run"], as_index=False)[
                f"delta_{metric}_active_minus_sham"
            ].mean()
        )
        rows.append(
            _one_sample_stats(
                block_any[f"delta_{metric}_active_minus_sham"].to_numpy(dtype=np.float64),
                "run_any_active_vs_sham",
                metric,
                "subject_session_run",
                "any_active_minus_sham",
            )
        )
    stats_df = pd.DataFrame(rows)
    stats_df = _add_condition_fdr(stats_df)
    return stats_df, pd.concat(deltas, ignore_index=True, sort=False)


def _bh_fdr(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=np.float64)
    adjusted = np.full(p_values.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(p_values)
    if not np.any(finite):
        return adjusted
    finite_p = p_values[finite]
    order = np.argsort(finite_p)
    ranked = finite_p[order]
    n = ranked.size
    ranked_adjusted = ranked * n / np.arange(1, n + 1)
    ranked_adjusted = np.minimum.accumulate(ranked_adjusted[::-1])[::-1]
    ranked_adjusted = np.clip(ranked_adjusted, 0.0, 1.0)
    restored = np.empty_like(ranked_adjusted)
    restored[order] = ranked_adjusted
    adjusted[finite] = restored
    return adjusted


def _add_condition_fdr(stats_df: pd.DataFrame) -> pd.DataFrame:
    out = stats_df.copy()
    out["p_ttest_fdr_bh_condition_family"] = np.nan
    out["p_wilcoxon_fdr_bh_condition_family"] = np.nan
    condition_mask = out["analysis"].astype(str).str.startswith("subject_GVS")
    for metric, index in out.loc[condition_mask].groupby("metric", sort=False).groups.items():
        idx = list(index)
        out.loc[idx, "p_ttest_fdr_bh_condition_family"] = _bh_fdr(
            out.loc[idx, "p_ttest_two_sided"].to_numpy(dtype=np.float64)
        )
        out.loc[idx, "p_wilcoxon_fdr_bh_condition_family"] = _bh_fdr(
            out.loc[idx, "p_wilcoxon_two_sided"].to_numpy(dtype=np.float64)
        )
    return out


def _regression_tables(blocks: pd.DataFrame) -> pd.DataFrame:
    if smf is None:
        return pd.DataFrame([{"model": "block_fixed_effects", "status": "skipped", "reason": "statsmodels unavailable"}])
    rows = []
    model_data = blocks.copy()
    model_data["condition_label"] = pd.Categorical(
        model_data["condition_label"],
        categories=["sham"] + sorted([label for label in model_data["condition_label"].unique() if label != "sham"]),
    )
    model_data["medication"] = pd.Categorical(model_data["medication"], categories=["OFF", "ON"])
    formulas = {
        "mean_rt_ms_condition_fixed_effects": "mean_rt_ms ~ C(condition_label) + C(medication) + C(run)",
        "rt_variability_condition_fixed_effects": "rt_variability ~ C(condition_label) + C(medication) + C(run)",
    }
    for model_name, formula in formulas.items():
        data = model_data.replace([np.inf, -np.inf], np.nan).dropna(subset=formula.split("~")[0].strip())
        try:
            fit = smf.ols(formula, data=data).fit(cov_type="cluster", cov_kwds={"groups": data["subject"]})
        except Exception as exc:  # pragma: no cover - depends on statsmodels/patsy
            rows.append({"model": model_name, "status": "failed", "reason": str(exc)})
            continue
        conf = fit.conf_int()
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
                    "ci95_low": float(conf.loc[term, 0]) if term in conf.index else float("nan"),
                    "ci95_high": float(conf.loc[term, 1]) if term in conf.index else float("nan"),
                    "n_observations": int(fit.nobs),
                    "n_subjects": int(data["subject"].nunique()),
                    "covariance": "cluster_robust_by_subject",
                    "formula": formula,
                }
            )
    return pd.DataFrame(rows)


def _condition_rows(stats_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    rows = stats_df.loc[
        stats_df["metric"].eq(metric) & stats_df["analysis"].astype(str).str.startswith("subject_GVS")
    ].copy()
    if rows.empty:
        return rows
    rows["condition_label"] = rows["contrast"].astype(str).str.replace("_minus_sham", "", regex=False)
    rows["_condition_number"] = rows["condition_label"].str.extract(r"(\d+)").astype(float)
    return rows.sort_values(["_condition_number", "condition_label"]).drop(columns="_condition_number")


def _metric_axis_label(metric: str) -> str:
    if metric == "mean_rt_ms":
        return "Active GVS - sham\nMean RT difference (ms)"
    if metric == "rt_variability":
        return "Active GVS - sham\nRT variability difference"
    return f"Active GVS - sham\n{metric}"


def _metric_panel_title(metric: str) -> str:
    if metric == "mean_rt_ms":
        return "Mean RT"
    if metric == "rt_variability":
        return "Consecutive-trial RT variability"
    return metric


def _metric_color(metric: str) -> str:
    if metric == "mean_rt_ms":
        return "#4C78A8"
    if metric == "rt_variability":
        return "#009E73"
    return "#4C78A8"


def _min_condition_fdr_text(rows: pd.DataFrame) -> str:
    q_values = rows["p_ttest_fdr_bh_condition_family"].to_numpy(dtype=np.float64)
    q_values = q_values[np.isfinite(q_values)]
    if q_values.size == 0:
        return "FDR q = n/a"
    return f"min FDR q = {float(np.min(q_values)):.3f}"


def _draw_condition_delta_panel(ax: plt.Axes, rows: pd.DataFrame, metric: str, panel_label: str | None = None) -> None:
    x = np.arange(rows.shape[0])
    means = rows["mean_delta"].to_numpy(dtype=np.float64)
    color = _metric_color(metric)
    yerr = np.vstack(
        [
            means - rows["ci95_low"].to_numpy(dtype=np.float64),
            rows["ci95_high"].to_numpy(dtype=np.float64) - means,
        ]
    )
    ax.errorbar(
        x,
        means,
        yerr=yerr,
        fmt="o",
        linestyle="none",
        color=color,
        markerfacecolor=color,
        markeredgecolor=color,
        markersize=5.8,
        elinewidth=1.25,
        capsize=3.0,
        capthick=1.0,
    )
    ax.axhline(0, color="0.35", linestyle="--", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(rows["condition_label"].tolist(), rotation=35, ha="right")
    ax.set_ylabel(_metric_axis_label(metric))
    ax.set_title(_metric_panel_title(metric), fontsize=11)
    if panel_label is not None:
        ax.text(-0.12, 1.04, panel_label, transform=ax.transAxes, fontsize=12, fontweight="bold")
    ax.text(
        0.98,
        0.96,
        _min_condition_fdr_text(rows),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.5},
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.8)


def _save_delta_plot(stats_df: pd.DataFrame, metric: str, out_dir: Path) -> list[str]:
    rows = _condition_rows(stats_df, metric)
    if rows.empty:
        return []
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    _draw_condition_delta_panel(ax, rows, metric)
    fig.tight_layout()
    stem = out_dir / f"gvs_behaviour_{metric}_condition_delta"
    png = stem.with_suffix(".png")
    pdf = stem.with_suffix(".pdf")
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return [str(png), str(pdf)]


def _save_combined_delta_plot(stats_df: pd.DataFrame, out_dir: Path) -> list[str]:
    metric_rows = [("mean_rt_ms", _condition_rows(stats_df, "mean_rt_ms")), ("rt_variability", _condition_rows(stats_df, "rt_variability"))]
    if any(rows.empty for _, rows in metric_rows):
        return []
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2), sharex=False)
    for ax, (metric, rows), label in zip(axes, metric_rows, ["A", "B"]):
        _draw_condition_delta_panel(ax, rows, metric, panel_label=label)
    fig.text(
        0.5,
        0.01,
        "Error bars show 95% CI across subjects. FDR q values are Benjamini-Hochberg corrected across the 8 waveform tests within each panel.",
        ha="center",
        va="bottom",
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
    stem = out_dir / "gvs_behaviour_condition_delta_combined"
    png = stem.with_suffix(".png")
    pdf = stem.with_suffix(".pdf")
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return [str(png), str(pdf)]


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def _write_report(out_dir: Path, summary: dict[str, object], stats_df: pd.DataFrame) -> None:
    primary = stats_df.loc[
        stats_df["analysis"].eq("subject_any_active_vs_sham") & stats_df["metric"].eq("mean_rt_ms")
    ].iloc[0]
    variability = stats_df.loc[
        stats_df["analysis"].eq("subject_any_active_vs_sham") & stats_df["metric"].eq("rt_variability")
    ].iloc[0]
    report = [
        "# GVS Behaviour Effects",
        "",
        (
            f"Extracted {summary['n_trials']} finite behaviour trials from {summary['n_runs']} runs, "
            f"{summary['n_sessions']} sessions, and {summary['n_subjects']} subjects."
        ),
        "",
        (
            "Block timing uses `run_condition_inventory.csv`; each GVS/sham block contributes up to 10 task trials. "
            "Two 20-trial rest gaps after stimulation blocks 3 and 6 are encoded in the rest-aware timing columns, "
            "and RT variability is computed within stimulation blocks only."
        ),
        "",
        "## Primary Any-Active GVS Result",
        "",
        (
            f"Subject-level any-active GVS minus sham mean RT delta = {primary['mean_delta']:.2f} ms "
            f"(95% CI {primary['ci95_low']:.2f}, {primary['ci95_high']:.2f}; "
            f"paired t p = {primary['p_ttest_two_sided']:.3g}; n = {int(primary['n'])})."
        ),
        "",
        (
            f"Subject-level any-active GVS minus sham RT-variability delta = {variability['mean_delta']:.3f} "
            f"(95% CI {variability['ci95_low']:.3f}, {variability['ci95_high']:.3f}; "
            f"paired t p = {variability['p_ttest_two_sided']:.3g}; n = {int(variability['n'])})."
        ),
        "",
        "## Figure Caption",
        "",
        (
            "Behavioural effects of GVS condition relative to sham. Subject-level paired differences are shown "
            "for each active GVS waveform relative to sham. Error bars indicate 95% confidence intervals across "
            "subjects. Panel A shows mean RT differences in milliseconds; positive values indicate slower "
            "responses during active GVS than sham. Panel B shows differences in within-block consecutive-trial "
            "RT variability. FDR q values are Benjamini-Hochberg corrected across the 8 waveform tests within "
            "each panel."
        ),
    ]
    (out_dir / "gvs_behaviour_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    if args.behaviour_column < 1:
        raise ValueError("--behaviour-column is one-based and must be >= 1")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    inventory = _load_inventory(args.inventory)
    trials, _missing = _build_trial_table(
        inventory,
        args.behaviour_dir,
        behaviour_column_index=args.behaviour_column - 1,
        input_is_inverse_rt=bool(args.input_is_inverse_rt),
        min_rt=args.min_rt,
        max_rt=args.max_rt,
    )
    blocks = _block_summary(trials)
    block_condition = _aggregate_condition(
        blocks,
        ["subject", "session", "medication", "run", "condition_code", "condition_label", "active_gvs"],
    )
    session_condition = _aggregate_condition(
        blocks,
        ["subject", "session", "medication", "condition_code", "condition_label", "active_gvs"],
    )
    subject_condition = _aggregate_condition(
        blocks,
        ["subject", "condition_code", "condition_label", "active_gvs"],
    )
    stats_df, delta_df = _paired_stats(subject_condition, session_condition, block_condition)
    regression_df = _regression_tables(blocks)

    trials.to_csv(args.out_dir / "gvs_behaviour_trials.csv", index=False)
    blocks.to_csv(args.out_dir / "gvs_behaviour_block_summary.csv", index=False)
    block_condition.to_csv(args.out_dir / "gvs_behaviour_run_condition_summary.csv", index=False)
    session_condition.to_csv(args.out_dir / "gvs_behaviour_session_condition_summary.csv", index=False)
    subject_condition.to_csv(args.out_dir / "gvs_behaviour_subject_condition_summary.csv", index=False)
    stats_df.to_csv(args.out_dir / "gvs_behaviour_paired_statistics.csv", index=False)
    delta_df.to_csv(args.out_dir / "gvs_behaviour_active_minus_sham_deltas.csv", index=False)
    regression_df.to_csv(args.out_dir / "gvs_behaviour_regression_statistics.csv", index=False)

    figure_paths = []
    figure_paths.extend(_save_delta_plot(stats_df, "mean_rt_ms", args.out_dir))
    figure_paths.extend(_save_delta_plot(stats_df, "rt_variability", args.out_dir))
    figure_paths.extend(_save_combined_delta_plot(stats_df, args.out_dir))

    summary = {
        "behaviour_dir": str(args.behaviour_dir),
        "inventory": str(args.inventory),
        "out_dir": str(args.out_dir),
        "input_is_inverse_rt": bool(args.input_is_inverse_rt),
        "behaviour_column_one_based": int(args.behaviour_column),
        "min_rt": args.min_rt,
        "max_rt": args.max_rt,
        "rest_gaps": {
            "rest_after_stim_blocks": list(REST_AFTER_STIM_BLOCKS),
            "rest_trials_each": REST_TRIALS,
            "note": "Rest-aware timing columns add 20 trials after block 3 and another 20 after block 6.",
        },
        "n_subjects": int(trials["subject"].nunique()),
        "n_sessions": int(trials[["subject", "session"]].drop_duplicates().shape[0]),
        "n_runs": int(trials[["subject", "session", "run"]].drop_duplicates().shape[0]),
        "n_trials": int(trials.shape[0]),
        "n_blocks": int(blocks.shape[0]),
        "output_files": [
            str(args.out_dir / "gvs_behaviour_trials.csv"),
            str(args.out_dir / "gvs_behaviour_block_summary.csv"),
            str(args.out_dir / "gvs_behaviour_run_condition_summary.csv"),
            str(args.out_dir / "gvs_behaviour_session_condition_summary.csv"),
            str(args.out_dir / "gvs_behaviour_subject_condition_summary.csv"),
            str(args.out_dir / "gvs_behaviour_paired_statistics.csv"),
            str(args.out_dir / "gvs_behaviour_active_minus_sham_deltas.csv"),
            str(args.out_dir / "gvs_behaviour_regression_statistics.csv"),
            str(args.out_dir / "gvs_behaviour_report.md"),
            *figure_paths,
        ],
    }
    _write_report(args.out_dir, summary, stats_df)
    (args.out_dir / "gvs_behaviour_summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Saved GVS behaviour outputs to {args.out_dir}")
    print(f"Subjects: {summary['n_subjects']}; sessions: {summary['n_sessions']}; runs: {summary['n_runs']}; trials: {summary['n_trials']}")


if __name__ == "__main__":
    main()
