#!/usr/bin/env python3
"""Run 14 pooled GVS analyses on the p90 vigour-network beta table.

The goal is to compare active GVS conditions against sham while pooling across
the weighted vigour-network ROIs.  Outputs are written beside this script.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm
from statsmodels.stats.multitest import fdrcorrection

try:
    import statsmodels.formula.api as smf

    HAS_STATSMODELS = True
except Exception:
    HAS_STATSMODELS = False

try:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False


ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "figures" / "GVS_effects" / "network_analyses"

TRIAL_TABLE = ROOT / "Claude_results" / "group_analyses" / "analysis3_lme" / "per_trial_roi_betas.csv"
ROI_DEF = ROOT / "Claude_results" / "group_analyses" / "analysis3_lme" / "per_trial_roi_betas_roi_definition.csv"
BEHAVIOUR_SESSION = ROOT / "figures" / "med_effects" / "tmp" / "behaviour_rt_session_summary.csv"
BEHAVIOUR_PAIRED = ROOT / "figures" / "med_effects" / "tmp" / "behaviour_topology_subject_values.csv"
RUN_INVENTORY = ROOT / "figures" / "GVS_effects" / "common" / "run_condition_inventory.csv"
WEIGHT_MAP = ROOT / "data" / "voxel_weights_task1_bold0.6_beta0.6_smooth1.25_gamma1.5.nii.gz"

META_COLS = {
    "subject",
    "session",
    "medication",
    "run",
    "condition_code",
    "condition_label",
    "trial_in_condition",
}
SHAM_CODE = "gvs-01"
ACTIVE_LABELS = [f"GVS{i}" for i in range(1, 9)]
RNG_SEED = 20260602
N_PERMUTATIONS = 5000
ALPHA = 0.05


@dataclass
class PreparedData:
    trial: pd.DataFrame
    block: pd.DataFrame
    block_delta: pd.DataFrame
    roi_block_delta: pd.DataFrame
    subject_network_any: pd.DataFrame
    subject_roi_any: pd.DataFrame
    subject_roi_condition: pd.DataFrame
    roi_cols: list[str]
    roi_weights: pd.Series


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_fig(fig: plt.Figure, path: Path, dpi: int = 220) -> None:
    ensure_dir(path.parent)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, default=_json_default))


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if pd.isna(value):
        return None
    return str(value)


def clean_label(value: str) -> str:
    return str(value).replace("\n", " ")


def weighted_row_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    numerator = np.where(finite, values, 0.0) @ weights
    denominator = finite.astype(float) @ weights
    out = np.full(values.shape[0], np.nan, dtype=float)
    valid = denominator > 0
    out[valid] = numerator[valid] / denominator[valid]
    return out


def cohen_d(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return float("nan")
    sd = float(np.std(values, ddof=1))
    return float(np.mean(values) / sd) if sd > 0 else float("nan")


def sign_flip_pvalue(
    values: np.ndarray,
    *,
    alternative: str = "two-sided",
    n_permutations: int = N_PERMUTATIONS,
    rng: np.random.Generator | None = None,
) -> float:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    n = int(vals.size)
    if n == 0:
        return float("nan")
    observed = float(np.mean(vals))
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)
    exact_count = 2**n
    if exact_count <= n_permutations:
        signs = np.array(np.meshgrid(*[[-1.0, 1.0]] * n)).T.reshape(-1, n)
    else:
        signs = rng.choice([-1.0, 1.0], size=(n_permutations, n))
    null = (signs * vals[None, :]).mean(axis=1)
    if alternative == "greater":
        return float((1 + np.count_nonzero(null >= observed)) / (null.size + 1))
    if alternative == "less":
        return float((1 + np.count_nonzero(null <= observed)) / (null.size + 1))
    return float((1 + np.count_nonzero(np.abs(null) >= abs(observed))) / (null.size + 1))


def one_sample_summary(
    values: np.ndarray,
    *,
    label: str,
    alternative: str = "two-sided",
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    n = int(vals.size)
    out: dict[str, Any] = {
        "label": label,
        "n": n,
        "mean": float(np.mean(vals)) if n else float("nan"),
        "sd": float(np.std(vals, ddof=1)) if n > 1 else float("nan"),
        "sem": float(stats.sem(vals)) if n > 1 else float("nan"),
        "cohens_d": cohen_d(vals),
        "t_stat": float("nan"),
        "p_ttest": float("nan"),
        "p_signflip": float("nan"),
        "ci95_lo": float("nan"),
        "ci95_hi": float("nan"),
    }
    if n > 1:
        test = stats.ttest_1samp(vals, 0.0, alternative=alternative)
        out["t_stat"] = float(test.statistic)
        out["p_ttest"] = float(test.pvalue)
        out["p_signflip"] = sign_flip_pvalue(vals, alternative=alternative, rng=rng)
        ci = stats.t.interval(0.95, n - 1, loc=float(np.mean(vals)), scale=float(stats.sem(vals)))
        out["ci95_lo"] = float(ci[0])
        out["ci95_hi"] = float(ci[1])
    return out


def add_fdr(df: pd.DataFrame, p_col: str, q_col: str = "q_fdr") -> pd.DataFrame:
    out = df.copy()
    out[q_col] = np.nan
    out["sig_fdr"] = False
    finite = np.isfinite(pd.to_numeric(out[p_col], errors="coerce"))
    if finite.any():
        rejected, q = fdrcorrection(out.loc[finite, p_col].astype(float).to_numpy(), alpha=ALPHA)
        out.loc[finite, q_col] = q
        out.loc[finite, "sig_fdr"] = rejected
    return out


def add_groupwise_fdr(
    df: pd.DataFrame,
    p_col: str,
    group_cols: list[str],
    q_col: str = "q_fdr",
    sig_col: str = "sig_fdr",
) -> pd.DataFrame:
    out = df.copy()
    out[q_col] = np.nan
    out[sig_col] = False
    if out.empty:
        return out
    for _, idx in out.groupby(group_cols, dropna=False).groups.items():
        pvals = pd.to_numeric(out.loc[idx, p_col], errors="coerce")
        finite = np.isfinite(pvals)
        if finite.any():
            rejected, q = fdrcorrection(pvals.loc[finite].to_numpy(dtype=float), alpha=ALPHA)
            valid_idx = pvals.loc[finite].index
            out.loc[valid_idx, q_col] = q
            out.loc[valid_idx, sig_col] = rejected
    return out


def matrix_signflip_pvalues(
    matrix: np.ndarray,
    means: np.ndarray,
    *,
    alternative: str = "two-sided",
    rng: np.random.Generator | None = None,
    n_permutations: int = N_PERMUTATIONS,
    chunk_size: int = 512,
) -> np.ndarray:
    arr = np.asarray(matrix, dtype=float)
    pvals = np.full(arr.shape[1], np.nan, dtype=float)
    if arr.size == 0:
        return pvals
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)

    finite = np.isfinite(arr)
    mask_groups: dict[tuple[bool, ...], list[int]] = {}
    for col_idx in range(arr.shape[1]):
        mask = tuple(bool(x) for x in finite[:, col_idx])
        if any(mask):
            mask_groups.setdefault(mask, []).append(col_idx)

    for mask_tuple, cols in mask_groups.items():
        row_mask = np.array(mask_tuple, dtype=bool)
        n = int(row_mask.sum())
        if n == 0:
            continue
        exact_count = 2**n
        if exact_count <= n_permutations:
            signs = np.array(np.meshgrid(*[[-1.0, 1.0]] * n)).T.reshape(-1, n)
        else:
            signs = rng.choice([-1.0, 1.0], size=(n_permutations, n))
        for start in range(0, len(cols), chunk_size):
            chunk = cols[start : start + chunk_size]
            values = arr[np.ix_(row_mask, chunk)]
            null = signs @ values / n
            observed = means[chunk]
            if alternative == "greater":
                counts = np.count_nonzero(null >= observed[None, :], axis=0)
            elif alternative == "less":
                counts = np.count_nonzero(null <= observed[None, :], axis=0)
            else:
                counts = np.count_nonzero(np.abs(null) >= np.abs(observed[None, :]), axis=0)
            pvals[chunk] = (1 + counts) / (signs.shape[0] + 1)
    return pvals


def one_sample_matrix_summary(
    matrix: pd.DataFrame,
    *,
    label: str,
    alternative: str = "two-sided",
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    arr = matrix.to_numpy(dtype=float)
    n_cols = arr.shape[1]
    finite = np.isfinite(arr)
    n = np.sum(finite, axis=0)
    means = np.full(n_cols, np.nan, dtype=float)
    sds = np.full(n_cols, np.nan, dtype=float)
    valid_mean = n > 0
    valid_sd = n > 1
    if valid_mean.any():
        sums = np.where(finite[:, valid_mean], arr[:, valid_mean], 0.0).sum(axis=0)
        means[valid_mean] = sums / n[valid_mean]
    if valid_sd.any():
        centered = np.where(finite[:, valid_sd], arr[:, valid_sd] - means[valid_sd][None, :], 0.0)
        sds[valid_sd] = np.sqrt(np.sum(centered**2, axis=0) / (n[valid_sd] - 1))

    sem = np.full(n_cols, np.nan, dtype=float)
    np.divide(sds, np.sqrt(n), out=sem, where=valid_sd)
    t_stat = np.full(n_cols, np.nan, dtype=float)
    np.divide(means, sem, out=t_stat, where=np.isfinite(sem) & (sem > 0))
    cohens = np.full(n_cols, np.nan, dtype=float)
    np.divide(means, sds, out=cohens, where=np.isfinite(sds) & (sds > 0))

    p_ttest = np.full(n_cols, np.nan, dtype=float)
    valid_t = np.isfinite(t_stat) & valid_sd
    if valid_t.any():
        dof = n[valid_t] - 1
        if alternative == "greater":
            p_ttest[valid_t] = stats.t.sf(t_stat[valid_t], dof)
        elif alternative == "less":
            p_ttest[valid_t] = stats.t.cdf(t_stat[valid_t], dof)
        else:
            p_ttest[valid_t] = 2.0 * stats.t.sf(np.abs(t_stat[valid_t]), dof)

    ci95_lo = np.full(n_cols, np.nan, dtype=float)
    ci95_hi = np.full(n_cols, np.nan, dtype=float)
    if valid_sd.any():
        tcrit = stats.t.ppf(0.975, n[valid_sd] - 1)
        ci95_lo[valid_sd] = means[valid_sd] - tcrit * sem[valid_sd]
        ci95_hi[valid_sd] = means[valid_sd] + tcrit * sem[valid_sd]

    p_signflip = matrix_signflip_pvalues(arr, means, alternative=alternative, rng=rng)
    out = pd.DataFrame(
        {
            "label": label,
            "n": n,
            "mean": means,
            "sd": sds,
            "sem": sem,
            "cohens_d": cohens,
            "t_stat": t_stat,
            "p_ttest": p_ttest,
            "p_signflip": p_signflip,
            "ci95_lo": ci95_lo,
            "ci95_hi": ci95_hi,
        },
        index=matrix.columns,
    )
    return out.reset_index()


def subject_sort_key(subject: str) -> tuple[int, str]:
    text = str(subject)
    digits = "".join(ch for ch in text if ch.isdigit())
    return (int(digits), text) if digits else (10**9, text)


def load_and_prepare() -> PreparedData:
    trial = pd.read_csv(TRIAL_TABLE)
    roi_def = pd.read_csv(ROI_DEF)
    roi_cols = [str(r) for r in roi_def["roi_label"].tolist() if str(r) in trial.columns]
    if not roi_cols:
        raise RuntimeError(f"No ROI columns from {ROI_DEF} were found in {TRIAL_TABLE}.")

    weight_lookup = roi_def.set_index("roi_label")["roi_weight_sum"].astype(float)
    roi_weights = weight_lookup.reindex(roi_cols).fillna(weight_lookup.mean())
    roi_weights = roi_weights / roi_weights.sum()

    trial = trial.copy()
    trial["condition_label"] = trial["condition_label"].map(clean_label)
    trial["medication"] = trial["medication"].astype(str).str.upper()
    trial["active"] = trial["condition_code"].astype(str).ne(SHAM_CODE).astype(int)
    trial["med_on"] = trial["medication"].eq("ON").astype(int)
    trial["run_id"] = (
        trial["subject"].astype(str)
        + "_ses-"
        + trial["session"].astype(str)
        + "_run-"
        + trial["run"].astype(str)
    )

    values = trial[roi_cols].to_numpy(dtype=float)
    weights = roi_weights.to_numpy(dtype=float)
    trial["network_score"] = weighted_row_mean(values, weights)
    trial["roi_dispersion"] = np.nanstd(values, axis=1, ddof=0)
    trial = trial.loc[np.isfinite(trial["network_score"])].reset_index(drop=True)

    group_cols = ["subject", "session", "medication", "run", "run_id", "condition_code", "condition_label", "active"]
    block = (
        trial.groupby(group_cols, dropna=False)
        .agg(
            network_score=("network_score", "mean"),
            network_var=("network_score", "var"),
            roi_dispersion=("roi_dispersion", "mean"),
            n_trials=("network_score", "count"),
        )
        .reset_index()
    )
    block["network_var"] = block["network_var"].fillna(0.0)
    block["med_on"] = block["medication"].eq("ON").astype(int)

    sham = block.loc[block["condition_code"].eq(SHAM_CODE)].rename(
        columns={
            "network_score": "sham_network_score",
            "network_var": "sham_network_var",
            "roi_dispersion": "sham_roi_dispersion",
            "n_trials": "sham_n_trials",
        }
    )
    sham = sham[
        [
            "subject",
            "session",
            "run",
            "run_id",
            "sham_network_score",
            "sham_network_var",
            "sham_roi_dispersion",
            "sham_n_trials",
        ]
    ]
    block_delta = block.loc[block["active"].eq(1)].merge(sham, on=["subject", "session", "run", "run_id"], how="inner")
    block_delta["network_delta"] = block_delta["network_score"] - block_delta["sham_network_score"]
    block_delta["network_var_delta"] = block_delta["network_var"] - block_delta["sham_network_var"]
    block_delta["roi_dispersion_delta"] = block_delta["roi_dispersion"] - block_delta["sham_roi_dispersion"]

    roi_block = trial.groupby(group_cols, dropna=False)[roi_cols].mean().reset_index()
    roi_long = roi_block.melt(
        id_vars=group_cols,
        value_vars=roi_cols,
        var_name="roi_label",
        value_name="roi_mean",
    )
    roi_sham = roi_long.loc[roi_long["condition_code"].eq(SHAM_CODE)].rename(columns={"roi_mean": "sham_roi_mean"})
    roi_sham = roi_sham[["subject", "session", "run", "run_id", "roi_label", "sham_roi_mean"]]
    roi_block_delta = (
        roi_long.loc[roi_long["active"].eq(1)]
        .merge(roi_sham, on=["subject", "session", "run", "run_id", "roi_label"], how="inner")
        .copy()
    )
    roi_block_delta["roi_delta"] = roi_block_delta["roi_mean"] - roi_block_delta["sham_roi_mean"]
    roi_block_delta["med_on"] = roi_block_delta["medication"].eq("ON").astype(int)

    subject_network_any = (
        block_delta.groupby(["subject", "session", "medication", "med_on"], dropna=False)
        .agg(
            network_delta=("network_delta", "mean"),
            network_var_delta=("network_var_delta", "mean"),
            roi_dispersion_delta=("roi_dispersion_delta", "mean"),
            n_active_blocks=("network_delta", "count"),
        )
        .reset_index()
    )

    subject_roi_any = (
        roi_block_delta.groupby(["subject", "session", "medication", "med_on", "roi_label"], dropna=False)
        .agg(roi_delta=("roi_delta", "mean"), n_active_blocks=("roi_delta", "count"))
        .reset_index()
    )

    subject_roi_condition = (
        roi_block_delta.groupby(
            ["subject", "session", "medication", "med_on", "condition_label", "roi_label"],
            dropna=False,
        )
        .agg(roi_delta=("roi_delta", "mean"), n_runs=("roi_delta", "count"))
        .reset_index()
    )

    return PreparedData(
        trial=trial,
        block=block,
        block_delta=block_delta,
        roi_block_delta=roi_block_delta,
        subject_network_any=subject_network_any,
        subject_roi_any=subject_roi_any,
        subject_roi_condition=subject_roi_condition,
        roi_cols=roi_cols,
        roi_weights=roi_weights,
    )


def fit_mixedlm(formula: str, data: pd.DataFrame, group_col: str, vc_formula: dict[str, str] | None = None) -> pd.DataFrame:
    if not HAS_STATSMODELS:
        return pd.DataFrame([{"status": "skipped", "reason": "statsmodels is not available"}])
    fit_data = data.copy()
    try:
        model = smf.mixedlm(formula, fit_data, groups=fit_data[group_col], vc_formula=vc_formula)
        result = model.fit(reml=False, method="lbfgs", maxiter=400, disp=False)
    except Exception as first_error:
        try:
            model = smf.mixedlm(formula, fit_data, groups=fit_data[group_col])
            result = model.fit(reml=False, method="lbfgs", maxiter=400, disp=False)
        except Exception as second_error:
            return pd.DataFrame(
                [
                    {
                        "status": "failed",
                        "first_error": str(first_error),
                        "fallback_error": str(second_error),
                    }
                ]
            )
    rows = []
    for name in result.params.index:
        rows.append(
            {
                "status": "ok",
                "term": str(name),
                "coef": float(result.params[name]),
                "se": float(result.bse.get(name, np.nan)),
                "z_stat": float(result.tvalues.get(name, np.nan)),
                "p_value": float(result.pvalues.get(name, np.nan)),
                "converged": bool(getattr(result, "converged", False)),
                "aic": float(getattr(result, "aic", np.nan)),
            }
        )
    return pd.DataFrame(rows)


def plot_subject_deltas(df: pd.DataFrame, value_col: str, title: str, path: Path) -> None:
    meds = ["OFF", "ON"]
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    rng = np.random.default_rng(RNG_SEED)
    for x, med in enumerate(meds):
        vals = df.loc[df["medication"].eq(med), value_col].dropna().to_numpy(dtype=float)
        if vals.size == 0:
            continue
        jitter = rng.normal(0, 0.035, vals.size)
        ax.scatter(np.full(vals.size, x) + jitter, vals, s=34, alpha=0.78, edgecolor="white", linewidth=0.5)
        mean = float(np.mean(vals))
        ci = stats.t.interval(0.95, vals.size - 1, loc=mean, scale=float(stats.sem(vals))) if vals.size > 1 else (mean, mean)
        ax.errorbar([x], [mean], yerr=[[mean - ci[0]], [ci[1] - mean]], color="black", capsize=5, marker="s")
    ax.axhline(0.0, color="#555555", linewidth=1.0, linestyle="--")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(meds)
    ax.set_ylabel(value_col.replace("_", " "))
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    save_fig(fig, path)


def analysis_01_collapsed_projection(data: PreparedData, rng: np.random.Generator) -> dict[str, Any]:
    out = ensure_dir(OUT_DIR / "01_collapsed_network_projection")
    model_df = fit_mixedlm(
        "network_score ~ active + med_on + active:med_on",
        data.trial,
        "subject",
        vc_formula={"run": "0 + C(run_id)"},
    )
    model_df.to_csv(out / "mixedlm_trial_network_score.csv", index=False)

    means = (
        data.trial.groupby(["medication", "active"], dropna=False)
        .agg(mean_network_score=("network_score", "mean"), sd=("network_score", "std"), n=("network_score", "count"))
        .reset_index()
    )
    means.to_csv(out / "active_vs_sham_trial_means.csv", index=False)

    stats_rows = [
        one_sample_summary(g["network_delta"].to_numpy(), label=med, rng=rng)
        for med, g in data.subject_network_any.groupby("medication", sort=False)
    ]
    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(out / "subject_any_gvs_network_delta_stats.csv", index=False)
    data.subject_network_any.to_csv(out / "subject_any_gvs_network_delta_values.csv", index=False)
    plot_subject_deltas(
        data.subject_network_any,
        "network_delta",
        "Collapsed network projection: active GVS - sham",
        out / "subject_any_gvs_network_delta.png",
    )

    best_p = float(np.nanmin(stats_df["p_signflip"])) if not stats_df.empty else float("nan")
    return {
        "analysis_id": "01",
        "analysis": "Collapsed network projection",
        "primary_metric": "subject mean active-GVS network delta",
        "best_p": best_p,
        "effect": float(stats_df.loc[stats_df["p_signflip"].idxmin(), "mean"]) if stats_df["p_signflip"].notna().any() else np.nan,
        "result_file": str(out / "subject_any_gvs_network_delta_stats.csv"),
    }


def analysis_02_block_mixedlm(data: PreparedData, rng: np.random.Generator) -> dict[str, Any]:
    out = ensure_dir(OUT_DIR / "02_block_level_mixedlm")
    model_df = fit_mixedlm(
        "network_delta ~ med_on",
        data.block_delta,
        "subject",
        vc_formula={"condition": "0 + C(condition_label)", "run": "0 + C(run_id)"},
    )
    model_df.to_csv(out / "mixedlm_block_network_delta.csv", index=False)
    condition_stats = []
    for (med, condition), g in data.block_delta.groupby(["medication", "condition_label"], sort=False):
        row = one_sample_summary(g["network_delta"].to_numpy(), label=f"{med}_{condition}", rng=rng)
        row.update({"medication": med, "condition_label": condition})
        condition_stats.append(row)
    condition_df = add_fdr(pd.DataFrame(condition_stats), "p_signflip") if condition_stats else pd.DataFrame()
    condition_df.to_csv(out / "block_network_delta_condition_stats.csv", index=False)
    data.block_delta.to_csv(out / "block_network_delta_values.csv", index=False)

    pivot = condition_df.pivot(index="medication", columns="condition_label", values="mean").reindex(index=["OFF", "ON"], columns=ACTIVE_LABELS)
    fig, ax = plt.subplots(figsize=(8.8, 3.2))
    im = ax.imshow(pivot.to_numpy(dtype=float), cmap="coolwarm", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Block-level network delta by GVS condition")
    fig.colorbar(im, ax=ax, label="active - sham")
    save_fig(fig, out / "block_network_delta_condition_heatmap.png")

    fixed = model_df.loc[~model_df.get("term", pd.Series(dtype=str)).astype(str).str.contains(" Var", regex=False)].copy()
    pvals = pd.to_numeric(fixed.get("p_value", pd.Series(dtype=float)), errors="coerce")
    best_p = float(np.nanmin(pvals)) if pvals.notna().any() else float("nan")
    return {
        "analysis_id": "02",
        "analysis": "Block-level mixed-effects model",
        "primary_metric": "block network delta",
        "best_p": best_p,
        "effect": float(condition_df["mean"].mean()) if not condition_df.empty else np.nan,
        "result_file": str(out / "mixedlm_block_network_delta.csv"),
    }


def analysis_03_subject_any_gvs(data: PreparedData, rng: np.random.Generator) -> dict[str, Any]:
    out = ensure_dir(OUT_DIR / "03_subject_any_gvs_vs_sham")
    rows = []
    for med, g in data.subject_network_any.groupby("medication", sort=False):
        row = one_sample_summary(g["network_delta"].to_numpy(), label=med, rng=rng)
        row["medication"] = med
        rows.append(row)
    pooled = one_sample_summary(data.subject_network_any["network_delta"].to_numpy(), label="pooled_OFF_ON", rng=rng)
    pooled["medication"] = "POOLED"
    rows.append(pooled)
    stats_df = pd.DataFrame(rows)
    stats_df.to_csv(out / "subject_any_gvs_vs_sham_stats.csv", index=False)
    data.subject_network_any.to_csv(out / "subject_any_gvs_vs_sham_values.csv", index=False)
    plot_subject_deltas(data.subject_network_any, "network_delta", "Any active GVS vs sham, all ROIs", out / "subject_any_gvs_vs_sham.png")
    best = stats_df.loc[stats_df["p_signflip"].idxmin()]
    return {
        "analysis_id": "03",
        "analysis": "Subject-level any-GVS vs sham summary",
        "primary_metric": "subject-session mean active-GVS network delta",
        "best_p": float(best["p_signflip"]),
        "effect": float(best["mean"]),
        "result_file": str(out / "subject_any_gvs_vs_sham_stats.csv"),
    }


def vector_norm_permutation(matrix: np.ndarray, rng: np.random.Generator, n_permutations: int = N_PERMUTATIONS) -> dict[str, float]:
    arr = np.asarray(matrix, dtype=float)
    arr = arr[np.all(np.isfinite(arr), axis=1)]
    n, p = arr.shape if arr.ndim == 2 else (0, 0)
    if n < 2 or p < 1:
        return {"n_subjects": n, "n_features": p, "norm": np.nan, "p_value": np.nan}
    observed = float(np.linalg.norm(np.mean(arr, axis=0)))
    signs = rng.choice([-1.0, 1.0], size=(n_permutations, n, 1))
    null = np.linalg.norm(np.mean(signs * arr[None, :, :], axis=1), axis=1)
    p_value = float((1 + np.count_nonzero(null >= observed)) / (n_permutations + 1))
    return {"n_subjects": n, "n_features": p, "norm": observed, "p_value": p_value}


def subject_roi_matrix(subject_roi: pd.DataFrame, med: str, roi_cols: list[str]) -> pd.DataFrame:
    subset = subject_roi.loc[subject_roi["medication"].eq(med)].copy()
    pivot = subset.pivot(index="subject", columns="roi_label", values="roi_delta").reindex(columns=roi_cols)
    return pivot.sort_index(key=lambda idx: [subject_sort_key(x) for x in idx])


def analysis_04_multivariate_network(data: PreparedData, rng: np.random.Generator) -> dict[str, Any]:
    out = ensure_dir(OUT_DIR / "04_multivariate_network_perturbation")
    rows = []
    top_rows = []
    for med in ["OFF", "ON"]:
        pivot = subject_roi_matrix(data.subject_roi_any, med, data.roi_cols)
        res = vector_norm_permutation(pivot.to_numpy(dtype=float), rng)
        res.update({"medication": med})
        rows.append(res)
        effects = pivot.mean(axis=0).rename("mean_roi_delta").reset_index().rename(columns={"roi_label": "roi_label"})
        effects["medication"] = med
        effects["abs_mean_roi_delta"] = effects["mean_roi_delta"].abs()
        top_rows.append(effects.sort_values("abs_mean_roi_delta", ascending=False).head(12))
        pivot.to_csv(out / f"{med.lower()}_subject_roi_any_gvs_delta_matrix.csv")
    stats_df = add_fdr(pd.DataFrame(rows), "p_value")
    stats_df.to_csv(out / "multivariate_vector_norm_permutation_stats.csv", index=False)
    top_df = pd.concat(top_rows, ignore_index=True)
    top_df.to_csv(out / "top_roi_vector_contributors.csv", index=False)
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for idx, med in enumerate(["OFF", "ON"]):
        subset = top_df.loc[top_df["medication"].eq(med)].head(8)
        y = np.arange(len(subset)) + idx * 0.38
        ax.barh(y, subset["mean_roi_delta"], height=0.34, label=med)
        ax.set_yticks(y)
        ax.set_yticklabels(subset["roi_label"])
    ax.axvline(0, color="#555555", linewidth=1)
    ax.set_xlabel("Mean ROI delta")
    ax.set_title("Largest multivariate ROI contributors")
    ax.legend()
    save_fig(fig, out / "top_roi_vector_contributors.png")
    best = stats_df.loc[stats_df["p_value"].idxmin()]
    return {
        "analysis_id": "04",
        "analysis": "Multivariate network perturbation permutation",
        "primary_metric": "norm of group mean 38-ROI delta vector",
        "best_p": float(best["p_value"]),
        "effect": float(best["norm"]),
        "result_file": str(out / "multivariate_vector_norm_permutation_stats.csv"),
    }


def analysis_05_medication_interaction(data: PreparedData, rng: np.random.Generator) -> dict[str, Any]:
    out = ensure_dir(OUT_DIR / "05_medication_interaction")
    wide = data.subject_network_any.pivot(index="subject", columns="medication", values="network_delta")
    wide = wide.dropna(subset=["OFF", "ON"])
    wide["on_minus_off_network_delta"] = wide["ON"] - wide["OFF"]
    stats_df = pd.DataFrame(
        [one_sample_summary(wide["on_minus_off_network_delta"].to_numpy(), label="ON_minus_OFF", rng=rng)]
    )
    stats_df.to_csv(out / "network_delta_medication_interaction_stats.csv", index=False)
    wide.to_csv(out / "network_delta_medication_interaction_values.csv")

    roi_interaction_rows = []
    off = data.subject_roi_any.loc[data.subject_roi_any["medication"].eq("OFF")]
    on = data.subject_roi_any.loc[data.subject_roi_any["medication"].eq("ON")]
    merged = on.merge(off, on=["subject", "roi_label"], suffixes=("_on", "_off"))
    merged["roi_interaction_delta"] = merged["roi_delta_on"] - merged["roi_delta_off"]
    pivot = merged.pivot(index="subject", columns="roi_label", values="roi_interaction_delta").reindex(columns=data.roi_cols)
    vector_res = vector_norm_permutation(pivot.to_numpy(dtype=float), rng)
    vector_res["comparison"] = "ON_minus_OFF_38ROI_vector"
    roi_interaction_rows.append(vector_res)
    pd.DataFrame(roi_interaction_rows).to_csv(out / "roi_vector_medication_interaction_permutation.csv", index=False)
    merged.to_csv(out / "roi_medication_interaction_values.csv", index=False)

    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    vals = wide["on_minus_off_network_delta"].to_numpy(dtype=float)
    ax.scatter(np.zeros(vals.size), vals, s=36, alpha=0.8)
    ax.errorbar([0], [np.mean(vals)], yerr=[[np.mean(vals) - stats_df.loc[0, "ci95_lo"]], [stats_df.loc[0, "ci95_hi"] - np.mean(vals)]], color="black", capsize=5)
    ax.axhline(0, color="#555555", linestyle="--")
    ax.set_xticks([0])
    ax.set_xticklabels(["ON - OFF"])
    ax.set_ylabel("Network GVS delta interaction")
    ax.set_title("Medication interaction")
    save_fig(fig, out / "network_delta_medication_interaction.png")
    return {
        "analysis_id": "05",
        "analysis": "Medication interaction",
        "primary_metric": "(active GVS - sham) ON minus OFF",
        "best_p": float(stats_df.loc[0, "p_signflip"]),
        "effect": float(stats_df.loc[0, "mean"]),
        "result_file": str(out / "network_delta_medication_interaction_stats.csv"),
    }


def resample_row(row: np.ndarray, target_trials: int = 10) -> np.ndarray:
    row = np.asarray(row, dtype=float)
    out = np.full(target_trials, np.nan, dtype=float)
    finite = np.isfinite(row)
    if not finite.any():
        return out
    vals = row[finite]
    if vals.size == 1:
        out[:] = vals[0]
        return out
    x_src = np.linspace(0, 1, row.size)[finite]
    x_dst = np.linspace(0, 1, target_trials)
    out[:] = np.interp(x_dst, x_src, vals)
    return out


def zscore_rows(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=float)
    out = np.full_like(arr, np.nan)
    for idx, row in enumerate(arr):
        finite = np.isfinite(row)
        if not finite.any():
            continue
        vals = row[finite]
        sd = float(np.std(vals, ddof=0))
        out[idx, finite] = 0.0 if sd == 0 else (vals - float(np.mean(vals))) / sd
    return out


def matrix_similarity_metrics(target_trials_by_roi: np.ndarray, sham_trials_by_roi: np.ndarray) -> dict[str, float]:
    target = np.asarray(target_trials_by_roi, dtype=float).T
    sham = np.asarray(sham_trials_by_roi, dtype=float).T
    target_r = np.vstack([resample_row(row) for row in target])
    sham_r = np.vstack([resample_row(row) for row in sham])
    target_z = zscore_rows(target_r)
    sham_z = zscore_rows(sham_r)
    valid = np.isfinite(target_z) & np.isfinite(sham_z)
    x = target_z[valid]
    y = sham_z[valid]
    if x.size < 3:
        pearson = np.nan
        rmse = np.nan
    else:
        pearson = float(np.corrcoef(x, y)[0, 1])
        rmse = float(np.sqrt(np.mean((x - y) ** 2)))
    raw_valid = np.isfinite(target_r) & np.isfinite(sham_r)
    xr = target_r[raw_valid]
    yr = sham_r[raw_valid]
    denom = float(np.linalg.norm(xr) * np.linalg.norm(yr))
    cosine = float(np.dot(xr, yr) / denom) if denom > 0 else np.nan
    return {
        "flat_pearson_r": pearson,
        "flat_cosine_similarity": cosine,
        "zscore_rmse": rmse,
        "cosine_perturbation": float(1.0 - cosine) if np.isfinite(cosine) else np.nan,
    }


def analysis_06_similarity_to_sham(data: PreparedData, rng: np.random.Generator) -> dict[str, Any]:
    out = ensure_dir(OUT_DIR / "06_similarity_to_sham")
    rows = []
    for key, run_df in data.trial.groupby(["subject", "session", "medication", "run", "run_id"], sort=False):
        subject, session, medication, run, run_id = key
        sham = run_df.loc[run_df["condition_code"].eq(SHAM_CODE), data.roi_cols].to_numpy(dtype=float)
        if sham.shape[0] < 3:
            continue
        for condition, cond_df in run_df.loc[run_df["condition_code"].ne(SHAM_CODE)].groupby("condition_label", sort=False):
            target = cond_df[data.roi_cols].to_numpy(dtype=float)
            if target.shape[0] < 3:
                continue
            metrics = matrix_similarity_metrics(target, sham)
            metrics.update(
                {
                    "subject": subject,
                    "session": int(session),
                    "medication": medication,
                    "run": int(run),
                    "run_id": run_id,
                    "condition_label": condition,
                }
            )
            rows.append(metrics)
    sim = pd.DataFrame(rows)
    sim.to_csv(out / "active_gvs_to_sham_similarity_by_block.csv", index=False)
    subject_sim = (
        sim.groupby(["subject", "session", "medication"], dropna=False)
        .agg(
            mean_cosine_similarity=("flat_cosine_similarity", "mean"),
            mean_cosine_perturbation=("cosine_perturbation", "mean"),
            mean_zscore_rmse=("zscore_rmse", "mean"),
            n_blocks=("condition_label", "count"),
        )
        .reset_index()
    )
    subject_sim.to_csv(out / "subject_similarity_to_sham_summary.csv", index=False)
    stats_df = pd.DataFrame(
        [
            one_sample_summary(g["mean_cosine_perturbation"].to_numpy(), label=med, alternative="greater", rng=rng)
            | {"medication": med}
            for med, g in subject_sim.groupby("medication", sort=False)
        ]
    )
    stats_df.to_csv(out / "similarity_perturbation_stats.csv", index=False)
    plot_subject_deltas(subject_sim.rename(columns={"mean_cosine_perturbation": "network_delta"}), "network_delta", "Similarity perturbation: 1 - cosine(active, sham)", out / "similarity_perturbation.png")
    best = stats_df.loc[stats_df["p_signflip"].idxmin()]
    return {
        "analysis_id": "06",
        "analysis": "Similarity-to-sham matrix analysis",
        "primary_metric": "descriptive 1 - cosine similarity between active and sham ROI-trial matrices",
        "best_p": np.nan,
        "effect": float(best["mean"]),
        "result_file": str(out / "similarity_perturbation_stats.csv"),
    }


def analysis_07_variability_dispersion(data: PreparedData, rng: np.random.Generator) -> dict[str, Any]:
    out = ensure_dir(OUT_DIR / "07_network_variability_dispersion")
    metrics = ["network_var_delta", "roi_dispersion_delta"]
    rows = []
    subject_vals = (
        data.block_delta.groupby(["subject", "session", "medication"], dropna=False)[metrics]
        .mean()
        .reset_index()
    )
    for metric in metrics:
        for med, g in subject_vals.groupby("medication", sort=False):
            row = one_sample_summary(g[metric].to_numpy(), label=f"{med}_{metric}", rng=rng)
            row.update({"medication": med, "metric": metric})
            rows.append(row)
    stats_df = add_fdr(pd.DataFrame(rows), "p_signflip")
    stats_df.to_csv(out / "variability_dispersion_stats.csv", index=False)
    subject_vals.to_csv(out / "variability_dispersion_subject_values.csv", index=False)
    for metric in metrics:
        plot_subject_deltas(subject_vals.rename(columns={metric: "network_delta"}), "network_delta", metric, out / f"{metric}.png")
    best = stats_df.loc[stats_df["p_signflip"].idxmin()]
    return {
        "analysis_id": "07",
        "analysis": "Network variability and ROI dispersion",
        "primary_metric": "active minus sham variance/dispersion",
        "best_p": float(best["p_signflip"]),
        "effect": float(best["mean"]),
        "result_file": str(out / "variability_dispersion_stats.csv"),
    }


def corr_edge_vector(values: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2:
        return np.array([], dtype=float), {"mean_abs_corr": np.nan, "mean_corr": np.nan}
    n_rois = arr.shape[1]
    edge_idx = np.triu_indices(n_rois, k=1)
    edges = np.full(edge_idx[0].size, np.nan, dtype=float)
    finite_cols = np.sum(np.isfinite(arr), axis=0) >= 4
    if arr.shape[0] < 4 or finite_cols.sum() < 2:
        return edges, {"mean_abs_corr": np.nan, "mean_corr": np.nan}
    work = arr[:, finite_cols].copy()
    col_means = np.nanmean(work, axis=0)
    inds = np.where(~np.isfinite(work))
    work[inds] = np.take(col_means, inds[1])
    corr = np.corrcoef(work, rowvar=False)
    full_corr = np.full((n_rois, n_rois), np.nan, dtype=float)
    valid_idx = np.flatnonzero(finite_cols)
    full_corr[np.ix_(valid_idx, valid_idx)] = corr
    edges = full_corr[edge_idx]
    finite_edges = edges[np.isfinite(edges)]
    if finite_edges.size == 0:
        return edges, {"mean_abs_corr": np.nan, "mean_corr": np.nan}
    return edges, {"mean_abs_corr": float(np.mean(np.abs(finite_edges))), "mean_corr": float(np.mean(finite_edges))}


def roi_edge_index(roi_cols: list[str]) -> pd.DataFrame:
    edge_idx = np.triu_indices(len(roi_cols), k=1)
    rows = []
    for edge_id, (i, j) in enumerate(zip(edge_idx[0], edge_idx[1])):
        roi_i = roi_cols[int(i)]
        roi_j = roi_cols[int(j)]
        rows.append({"edge_id": edge_id, "roi_i": roi_i, "roi_j": roi_j, "edge_label": f"{roi_i}--{roi_j}"})
    return pd.DataFrame(rows)


def plot_edge_tstat_matrix(stats_df: pd.DataFrame, roi_cols: list[str], title: str, path: Path) -> None:
    matrix = pd.DataFrame(np.nan, index=roi_cols, columns=roi_cols, dtype=float)
    sig = pd.DataFrame(False, index=roi_cols, columns=roi_cols, dtype=bool)
    for row in stats_df.itertuples(index=False):
        matrix.loc[row.roi_i, row.roi_j] = float(row.t_stat)
        matrix.loc[row.roi_j, row.roi_i] = float(row.t_stat)
        is_sig = bool(getattr(row, "sig_fdr", False))
        sig.loc[row.roi_i, row.roi_j] = is_sig
        sig.loc[row.roi_j, row.roi_i] = is_sig
    vmax = float(np.nanpercentile(np.abs(matrix.to_numpy(dtype=float)), 98)) if np.isfinite(matrix.to_numpy(dtype=float)).any() else 1.0
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0
    fig, ax = plt.subplots(figsize=(9.4, 8.0))
    im = ax.imshow(matrix.to_numpy(dtype=float), cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(roi_cols)))
    ax.set_xticklabels(roi_cols, rotation=90, fontsize=5.5)
    ax.set_yticks(range(len(roi_cols)))
    ax.set_yticklabels(roi_cols, fontsize=5.5)
    yy, xx = np.where(sig.to_numpy(dtype=bool))
    if xx.size:
        ax.scatter(xx, yy, s=9, facecolors="none", edgecolors="black", linewidths=0.55)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="edge t-stat")
    save_fig(fig, path)


def analysis_08_connectivity_coactivation(data: PreparedData, rng: np.random.Generator) -> dict[str, Any]:
    out = ensure_dir(OUT_DIR / "08_connectivity_coactivation")
    edge_index = roi_edge_index(data.roi_cols)
    rows = []
    edge_lookup: dict[tuple[Any, ...], np.ndarray] = {}
    for key, block_df in data.trial.groupby(["subject", "session", "medication", "run", "run_id", "condition_label", "condition_code"], sort=False):
        edges, metrics = corr_edge_vector(block_df[data.roi_cols].to_numpy(dtype=float))
        record = dict(zip(["subject", "session", "medication", "run", "run_id", "condition_label", "condition_code"], key))
        record.update(metrics)
        rows.append(record)
        subject, session, medication, run, run_id, _condition_label, condition_code = key
        edge_lookup[(subject, session, medication, run, run_id, condition_code)] = edges
    conn = pd.DataFrame(rows)
    conn.to_csv(out / "block_connectivity_metrics.csv", index=False)
    active = conn.loc[conn["condition_code"].ne(SHAM_CODE)].copy()
    sham = conn.loc[conn["condition_code"].eq(SHAM_CODE)].rename(
        columns={"mean_abs_corr": "sham_mean_abs_corr", "mean_corr": "sham_mean_corr"}
    )
    sham = sham[["subject", "session", "run", "run_id", "sham_mean_abs_corr", "sham_mean_corr"]]
    delta = active.merge(sham, on=["subject", "session", "run", "run_id"], how="inner")
    delta["mean_abs_corr_delta"] = delta["mean_abs_corr"] - delta["sham_mean_abs_corr"]
    delta["mean_corr_delta"] = delta["mean_corr"] - delta["sham_mean_corr"]
    edge_distances = []
    edge_delta_frames = []
    for row in delta.itertuples(index=False):
        active_key = (row.subject, row.session, row.medication, row.run, row.run_id, row.condition_code)
        sham_key = (row.subject, row.session, row.medication, row.run, row.run_id, SHAM_CODE)
        a = edge_lookup.get(active_key, np.array([]))
        s = edge_lookup.get(sham_key, np.array([]))
        n = min(a.size, s.size)
        edge_distances.append(float(np.mean(np.abs(a[:n] - s[:n]))) if n else np.nan)
        if n:
            edge_frame = edge_index.iloc[:n].copy()
            edge_frame["subject"] = row.subject
            edge_frame["session"] = row.session
            edge_frame["medication"] = row.medication
            edge_frame["run"] = row.run
            edge_frame["run_id"] = row.run_id
            edge_frame["condition_label"] = row.condition_label
            edge_frame["condition_code"] = row.condition_code
            edge_frame["active_corr"] = a[:n]
            edge_frame["sham_corr"] = s[:n]
            edge_frame["corr_delta"] = a[:n] - s[:n]
            edge_delta_frames.append(edge_frame)
    delta["edge_mean_abs_distance_to_sham"] = edge_distances
    delta.to_csv(out / "active_minus_sham_connectivity_deltas.csv", index=False)
    subject = delta.groupby(["subject", "session", "medication"], dropna=False)[
        ["mean_abs_corr_delta", "mean_corr_delta", "edge_mean_abs_distance_to_sham"]
    ].mean().reset_index()
    subject.to_csv(out / "subject_connectivity_delta_summary.csv", index=False)
    rows = []
    for metric in ["mean_abs_corr_delta", "mean_corr_delta", "edge_mean_abs_distance_to_sham"]:
        alt = "greater" if metric == "edge_mean_abs_distance_to_sham" else "two-sided"
        for med, g in subject.groupby("medication", sort=False):
            row = one_sample_summary(g[metric].to_numpy(), label=f"{med}_{metric}", alternative=alt, rng=rng)
            row.update({"medication": med, "metric": metric})
            rows.append(row)
    stats_df = add_fdr(pd.DataFrame(rows), "p_signflip")
    stats_df.to_csv(out / "connectivity_coactivation_stats.csv", index=False)

    edge_delta = pd.concat(edge_delta_frames, ignore_index=True) if edge_delta_frames else pd.DataFrame()
    if not edge_delta.empty:
        edge_cols = [
            "subject",
            "session",
            "medication",
            "run",
            "run_id",
            "condition_label",
            "condition_code",
            "edge_id",
            "roi_i",
            "roi_j",
            "edge_label",
            "active_corr",
            "sham_corr",
            "corr_delta",
        ]
        edge_delta = edge_delta[edge_cols]
    edge_delta.to_csv(out / "block_edge_connectivity_deltas.csv", index=False)

    subject_edge_any = (
        edge_delta.groupby(["subject", "session", "medication", "edge_id", "roi_i", "roi_j", "edge_label"], dropna=False)
        .agg(corr_delta=("corr_delta", "mean"), n_active_blocks=("corr_delta", "count"))
        .reset_index()
        if not edge_delta.empty
        else pd.DataFrame()
    )
    subject_edge_any.to_csv(out / "subject_edge_any_gvs_delta_summary.csv", index=False)

    subject_edge_condition = (
        edge_delta.groupby(
            ["subject", "session", "medication", "condition_label", "condition_code", "edge_id", "roi_i", "roi_j", "edge_label"],
            dropna=False,
        )
        .agg(corr_delta=("corr_delta", "mean"), n_runs=("corr_delta", "count"))
        .reset_index()
        if not edge_delta.empty
        else pd.DataFrame()
    )
    subject_edge_condition.to_csv(out / "subject_edge_by_gvs_delta_summary.csv", index=False)

    edge_stat_cols = ["edge_id", "roi_i", "roi_j", "edge_label"]
    any_edge_rows = []
    if not subject_edge_any.empty:
        for med, g in subject_edge_any.groupby("medication", sort=False):
            matrix = g.pivot_table(index=["subject", "session"], columns=edge_stat_cols, values="corr_delta", aggfunc="mean")
            edge_stats = one_sample_matrix_summary(matrix, label=f"{med}_ANY_GVS_edge_delta", rng=rng)
            edge_stats["medication"] = med
            edge_stats["gvs_group"] = "ANY_GVS"
            any_edge_rows.append(edge_stats)
        pooled_matrix = subject_edge_any.pivot_table(
            index=["subject", "session", "medication"],
            columns=edge_stat_cols,
            values="corr_delta",
            aggfunc="mean",
        )
        pooled_stats = one_sample_matrix_summary(pooled_matrix, label="POOLED_ANY_GVS_edge_delta", rng=rng)
        pooled_stats["medication"] = "POOLED"
        pooled_stats["gvs_group"] = "ANY_GVS"
        any_edge_rows.append(pooled_stats)
    any_edge_stats = pd.concat(any_edge_rows, ignore_index=True) if any_edge_rows else pd.DataFrame()
    if not any_edge_stats.empty:
        any_edge_stats["abs_mean"] = any_edge_stats["mean"].abs()
        any_edge_stats["sig_uncorrected"] = any_edge_stats["p_signflip"].lt(ALPHA)
        any_edge_stats = add_groupwise_fdr(any_edge_stats, "p_signflip", ["medication", "gvs_group"])
    any_edge_stats.to_csv(out / "edge_any_gvs_vs_sham_stats.csv", index=False)
    if not any_edge_stats.empty:
        top_any = any_edge_stats.sort_values(
            ["sig_fdr", "q_fdr", "p_signflip", "abs_mean"],
            ascending=[False, True, True, False],
        ).head(100)
        top_any.to_csv(out / "top_edge_any_gvs_vs_sham.csv", index=False)
        for med in ["OFF", "ON", "POOLED"]:
            med_stats = any_edge_stats.loc[any_edge_stats["medication"].eq(med)]
            if not med_stats.empty:
                plot_edge_tstat_matrix(
                    med_stats,
                    data.roi_cols,
                    f"{med}: any active GVS edge deltas",
                    out / f"{med.lower()}_any_gvs_edge_tstat_heatmap.png",
                )

    condition_edge_rows = []
    if not subject_edge_condition.empty:
        for (med, condition_label, condition_code), g in subject_edge_condition.groupby(
            ["medication", "condition_label", "condition_code"], sort=False
        ):
            matrix = g.pivot_table(index=["subject", "session"], columns=edge_stat_cols, values="corr_delta", aggfunc="mean")
            edge_stats = one_sample_matrix_summary(matrix, label=f"{med}_{condition_label}_edge_delta", rng=rng)
            edge_stats["medication"] = med
            edge_stats["condition_label"] = condition_label
            edge_stats["condition_code"] = condition_code
            condition_edge_rows.append(edge_stats)
        for (condition_label, condition_code), g in subject_edge_condition.groupby(["condition_label", "condition_code"], sort=False):
            matrix = g.pivot_table(
                index=["subject", "session", "medication"],
                columns=edge_stat_cols,
                values="corr_delta",
                aggfunc="mean",
            )
            edge_stats = one_sample_matrix_summary(matrix, label=f"POOLED_{condition_label}_edge_delta", rng=rng)
            edge_stats["medication"] = "POOLED"
            edge_stats["condition_label"] = condition_label
            edge_stats["condition_code"] = condition_code
            condition_edge_rows.append(edge_stats)
    condition_edge_stats = pd.concat(condition_edge_rows, ignore_index=True) if condition_edge_rows else pd.DataFrame()
    if not condition_edge_stats.empty:
        condition_edge_stats["abs_mean"] = condition_edge_stats["mean"].abs()
        condition_edge_stats["sig_uncorrected"] = condition_edge_stats["p_signflip"].lt(ALPHA)
        condition_edge_stats = add_groupwise_fdr(condition_edge_stats, "p_signflip", ["medication", "condition_label", "condition_code"])
    condition_edge_stats.to_csv(out / "edge_by_gvs_vs_sham_stats.csv", index=False)
    if not condition_edge_stats.empty:
        top_condition = condition_edge_stats.sort_values(
            ["sig_fdr", "q_fdr", "p_signflip", "abs_mean"],
            ascending=[False, True, True, False],
        ).head(150)
        top_condition.to_csv(out / "top_edge_by_gvs_vs_sham.csv", index=False)

    run_edge_any = (
        edge_delta.groupby(["subject", "session", "medication", "run", "edge_id", "roi_i", "roi_j", "edge_label"], dropna=False)
        .agg(corr_delta=("corr_delta", "mean"), n_active_conditions=("corr_delta", "count"))
        .reset_index()
        if not edge_delta.empty
        else pd.DataFrame()
    )
    run_edge_any.to_csv(out / "subject_run_edge_any_gvs_delta_summary.csv", index=False)
    run_any_rows = []
    if not run_edge_any.empty:
        for (med, run), g in run_edge_any.groupby(["medication", "run"], sort=False):
            matrix = g.pivot_table(index=["subject", "session"], columns=edge_stat_cols, values="corr_delta", aggfunc="mean")
            edge_stats = one_sample_matrix_summary(matrix, label=f"{med}_run{run}_ANY_GVS_edge_delta", rng=rng)
            edge_stats["medication"] = med
            edge_stats["run"] = run
            edge_stats["gvs_group"] = "ANY_GVS"
            run_any_rows.append(edge_stats)
        for run, g in run_edge_any.groupby("run", sort=False):
            matrix = g.pivot_table(index=["subject", "session", "medication"], columns=edge_stat_cols, values="corr_delta", aggfunc="mean")
            edge_stats = one_sample_matrix_summary(matrix, label=f"POOLED_run{run}_ANY_GVS_edge_delta", rng=rng)
            edge_stats["medication"] = "POOLED"
            edge_stats["run"] = run
            edge_stats["gvs_group"] = "ANY_GVS"
            run_any_rows.append(edge_stats)
    run_any_stats = pd.concat(run_any_rows, ignore_index=True) if run_any_rows else pd.DataFrame()
    if not run_any_stats.empty:
        run_any_stats["abs_mean"] = run_any_stats["mean"].abs()
        run_any_stats["sig_uncorrected"] = run_any_stats["p_signflip"].lt(ALPHA)
        run_any_stats = add_groupwise_fdr(run_any_stats, "p_signflip", ["medication", "run", "gvs_group"])
    run_any_stats.to_csv(out / "edge_any_gvs_by_run_vs_sham_stats.csv", index=False)
    if not run_any_stats.empty:
        run_any_stats.sort_values(
            ["sig_fdr", "q_fdr", "p_signflip", "abs_mean"],
            ascending=[False, True, True, False],
        ).head(150).to_csv(out / "top_edge_any_gvs_by_run_vs_sham.csv", index=False)

    run_edge_condition = (
        edge_delta.groupby(
            ["subject", "session", "medication", "run", "condition_label", "condition_code", "edge_id", "roi_i", "roi_j", "edge_label"],
            dropna=False,
        )
        .agg(corr_delta=("corr_delta", "mean"), n_blocks=("corr_delta", "count"))
        .reset_index()
        if not edge_delta.empty
        else pd.DataFrame()
    )
    run_edge_condition.to_csv(out / "subject_run_edge_by_gvs_delta_summary.csv", index=False)
    run_condition_rows = []
    if not run_edge_condition.empty:
        for (med, run, condition_label, condition_code), g in run_edge_condition.groupby(
            ["medication", "run", "condition_label", "condition_code"], sort=False
        ):
            matrix = g.pivot_table(index=["subject", "session"], columns=edge_stat_cols, values="corr_delta", aggfunc="mean")
            edge_stats = one_sample_matrix_summary(matrix, label=f"{med}_run{run}_{condition_label}_edge_delta", rng=rng)
            edge_stats["medication"] = med
            edge_stats["run"] = run
            edge_stats["condition_label"] = condition_label
            edge_stats["condition_code"] = condition_code
            run_condition_rows.append(edge_stats)
        for (run, condition_label, condition_code), g in run_edge_condition.groupby(["run", "condition_label", "condition_code"], sort=False):
            matrix = g.pivot_table(index=["subject", "session", "medication"], columns=edge_stat_cols, values="corr_delta", aggfunc="mean")
            edge_stats = one_sample_matrix_summary(matrix, label=f"POOLED_run{run}_{condition_label}_edge_delta", rng=rng)
            edge_stats["medication"] = "POOLED"
            edge_stats["run"] = run
            edge_stats["condition_label"] = condition_label
            edge_stats["condition_code"] = condition_code
            run_condition_rows.append(edge_stats)
    run_condition_stats = pd.concat(run_condition_rows, ignore_index=True) if run_condition_rows else pd.DataFrame()
    if not run_condition_stats.empty:
        run_condition_stats["abs_mean"] = run_condition_stats["mean"].abs()
        run_condition_stats["sig_uncorrected"] = run_condition_stats["p_signflip"].lt(ALPHA)
        run_condition_stats = add_groupwise_fdr(
            run_condition_stats,
            "p_signflip",
            ["medication", "run", "condition_label", "condition_code"],
        )
    run_condition_stats.to_csv(out / "edge_by_gvs_by_run_vs_sham_stats.csv", index=False)
    if not run_condition_stats.empty:
        run_condition_stats.sort_values(
            ["sig_fdr", "q_fdr", "p_signflip", "abs_mean"],
            ascending=[False, True, True, False],
        ).head(150).to_csv(out / "top_edge_by_gvs_by_run_vs_sham.csv", index=False)

    both_session_any = (
        subject_edge_any.groupby(["subject", "edge_id", "roi_i", "roi_j", "edge_label"], dropna=False)
        .agg(corr_delta=("corr_delta", "mean"), n_sessions=("corr_delta", "count"))
        .reset_index()
        if not subject_edge_any.empty
        else pd.DataFrame()
    )
    both_session_any.to_csv(out / "subject_edge_both_sessions_any_gvs_delta_summary.csv", index=False)
    both_any_stats = pd.DataFrame()
    if not both_session_any.empty:
        matrix = both_session_any.pivot_table(index="subject", columns=edge_stat_cols, values="corr_delta", aggfunc="mean")
        both_any_stats = one_sample_matrix_summary(matrix, label="BOTH_SESSIONS_ANY_GVS_edge_delta", rng=rng)
        both_any_stats["session_pool"] = "BOTH_SESSIONS"
        both_any_stats["gvs_group"] = "ANY_GVS"
        both_any_stats["abs_mean"] = both_any_stats["mean"].abs()
        both_any_stats["sig_uncorrected"] = both_any_stats["p_signflip"].lt(ALPHA)
        both_any_stats = add_groupwise_fdr(both_any_stats, "p_signflip", ["session_pool", "gvs_group"])
    both_any_stats.to_csv(out / "edge_both_sessions_any_gvs_vs_sham_stats.csv", index=False)
    if not both_any_stats.empty:
        both_any_stats.sort_values(
            ["sig_fdr", "q_fdr", "p_signflip", "abs_mean"],
            ascending=[False, True, True, False],
        ).head(100).to_csv(out / "top_edge_both_sessions_any_gvs_vs_sham.csv", index=False)

    both_session_condition = (
        subject_edge_condition.groupby(["subject", "condition_label", "condition_code", "edge_id", "roi_i", "roi_j", "edge_label"], dropna=False)
        .agg(corr_delta=("corr_delta", "mean"), n_sessions=("corr_delta", "count"))
        .reset_index()
        if not subject_edge_condition.empty
        else pd.DataFrame()
    )
    both_session_condition.to_csv(out / "subject_edge_both_sessions_by_gvs_delta_summary.csv", index=False)
    both_condition_rows = []
    if not both_session_condition.empty:
        for (condition_label, condition_code), g in both_session_condition.groupby(["condition_label", "condition_code"], sort=False):
            matrix = g.pivot_table(index="subject", columns=edge_stat_cols, values="corr_delta", aggfunc="mean")
            edge_stats = one_sample_matrix_summary(matrix, label=f"BOTH_SESSIONS_{condition_label}_edge_delta", rng=rng)
            edge_stats["session_pool"] = "BOTH_SESSIONS"
            edge_stats["condition_label"] = condition_label
            edge_stats["condition_code"] = condition_code
            both_condition_rows.append(edge_stats)
    both_condition_stats = pd.concat(both_condition_rows, ignore_index=True) if both_condition_rows else pd.DataFrame()
    if not both_condition_stats.empty:
        both_condition_stats["abs_mean"] = both_condition_stats["mean"].abs()
        both_condition_stats["sig_uncorrected"] = both_condition_stats["p_signflip"].lt(ALPHA)
        both_condition_stats = add_groupwise_fdr(both_condition_stats, "p_signflip", ["session_pool", "condition_label", "condition_code"])
    both_condition_stats.to_csv(out / "edge_both_sessions_by_gvs_vs_sham_stats.csv", index=False)
    if not both_condition_stats.empty:
        both_condition_stats.sort_values(
            ["sig_fdr", "q_fdr", "p_signflip", "abs_mean"],
            ascending=[False, True, True, False],
        ).head(150).to_csv(out / "top_edge_both_sessions_by_gvs_vs_sham.csv", index=False)

    all_subject_rows = []
    if not edge_delta.empty:
        matrix = edge_delta.pivot_table(
            index=["subject", "session", "medication", "run", "condition_label", "condition_code"],
            columns=edge_stat_cols,
            values="corr_delta",
            aggfunc="mean",
        )
        all_any_stats = one_sample_matrix_summary(matrix, label="ALL_SUBJECTS_BLOCK_POOL_ANY_GVS_edge_delta", rng=rng)
        all_any_stats["pool"] = "ALL_SUBJECTS_BLOCKS"
        all_any_stats["condition_label"] = "ANY_GVS"
        all_any_stats["condition_code"] = "ANY_GVS"
        all_any_stats["analysis_unit"] = "subject_session_medication_run_condition"
        all_subject_rows.append(all_any_stats)
        for (condition_label, condition_code), g in edge_delta.groupby(["condition_label", "condition_code"], sort=False):
            matrix = g.pivot_table(
                index=["subject", "session", "medication", "run"],
                columns=edge_stat_cols,
                values="corr_delta",
                aggfunc="mean",
            )
            edge_stats = one_sample_matrix_summary(matrix, label=f"ALL_SUBJECTS_BLOCK_POOL_{condition_label}_edge_delta", rng=rng)
            edge_stats["pool"] = "ALL_SUBJECTS_BLOCKS"
            edge_stats["condition_label"] = condition_label
            edge_stats["condition_code"] = condition_code
            edge_stats["analysis_unit"] = "subject_session_medication_run"
            all_subject_rows.append(edge_stats)
    all_subject_stats = pd.concat(all_subject_rows, ignore_index=True) if all_subject_rows else pd.DataFrame()
    if not all_subject_stats.empty:
        all_subject_stats["abs_mean"] = all_subject_stats["mean"].abs()
        all_subject_stats["sig_uncorrected"] = all_subject_stats["p_signflip"].lt(ALPHA)
        all_subject_stats = add_groupwise_fdr(all_subject_stats, "p_signflip", ["pool", "condition_label", "condition_code"])
    all_subject_stats.to_csv(out / "edge_all_subjects_block_pool_vs_sham_stats.csv", index=False)
    if not all_subject_stats.empty:
        all_subject_stats.sort_values(
            ["sig_fdr", "q_fdr", "p_signflip", "abs_mean"],
            ascending=[False, True, True, False],
        ).head(150).to_csv(out / "top_edge_all_subjects_block_pool_vs_sham.csv", index=False)

    inferential = any_edge_stats if not any_edge_stats.empty else stats_df.loc[~stats_df["metric"].eq("edge_mean_abs_distance_to_sham")].copy()
    best_p_col = "p_signflip"
    best = inferential.loc[inferential[best_p_col].idxmin()] if not inferential.empty else stats_df.loc[stats_df["p_signflip"].idxmin()]
    return {
        "analysis_id": "08",
        "analysis": "Edge-wise connectivity/coactivation graph analysis",
        "primary_metric": "edge-wise active minus sham ROI-ROI correlation delta",
        "best_p": float(best[best_p_col]),
        "effect": float(best["mean"]),
        "result_file": str(out / "edge_any_gvs_vs_sham_stats.csv"),
    }


def circuit_for_roi(roi: str) -> str:
    base = str(roi).rsplit("_", 1)[0]
    if base in {"Precentral", "Postcentral", "Supp_Motor_Area", "Paracentral_Lobule"}:
        return "motor_sensorimotor"
    if base in {"Caudate", "Putamen", "Thalamus"}:
        return "basal_ganglia_thalamus"
    if base in {"Orbitofrontal", "Cingulate", "Amygdala", "Hippocampus", "ParaHippocampal", "Olfactory"}:
        return "reward_limbic"
    if base in {"Cerebellum"}:
        return "cerebellar"
    return "cortical_association"


def analysis_09_circuit_level(data: PreparedData, rng: np.random.Generator) -> dict[str, Any]:
    out = ensure_dir(OUT_DIR / "09_circuit_level_aggregation")
    trial = data.trial.copy()
    circuit_map = {roi: circuit_for_roi(roi) for roi in data.roi_cols}
    circuit_rows = pd.DataFrame({"roi_label": data.roi_cols, "circuit": [circuit_map[r] for r in data.roi_cols]})
    circuit_rows.to_csv(out / "roi_to_circuit_map.csv", index=False)
    for circuit, rois in circuit_rows.groupby("circuit")["roi_label"]:
        cols = [r for r in rois if r in data.roi_cols]
        weights = data.roi_weights.reindex(cols).to_numpy(dtype=float)
        weights = weights / weights.sum()
        trial[circuit] = weighted_row_mean(trial[cols].to_numpy(dtype=float), weights)
    circuits = sorted(circuit_rows["circuit"].unique())
    group_cols = ["subject", "session", "medication", "run", "run_id", "condition_code", "condition_label", "active"]
    block = trial.groupby(group_cols, dropna=False)[circuits].mean().reset_index()
    long = block.melt(id_vars=group_cols, value_vars=circuits, var_name="circuit", value_name="circuit_score")
    sham = long.loc[long["condition_code"].eq(SHAM_CODE)].rename(columns={"circuit_score": "sham_circuit_score"})
    sham = sham[["subject", "session", "run", "run_id", "circuit", "sham_circuit_score"]]
    delta = long.loc[long["active"].eq(1)].merge(sham, on=["subject", "session", "run", "run_id", "circuit"], how="inner")
    delta["circuit_delta"] = delta["circuit_score"] - delta["sham_circuit_score"]
    delta.to_csv(out / "circuit_block_deltas.csv", index=False)
    subject = delta.groupby(["subject", "session", "medication", "circuit"], dropna=False)["circuit_delta"].mean().reset_index()
    subject.to_csv(out / "circuit_subject_any_gvs_deltas.csv", index=False)
    rows = []
    for (med, circuit), g in subject.groupby(["medication", "circuit"], sort=False):
        row = one_sample_summary(g["circuit_delta"].to_numpy(), label=f"{med}_{circuit}", rng=rng)
        row.update({"medication": med, "circuit": circuit})
        rows.append(row)
    stats_df = add_fdr(pd.DataFrame(rows), "p_signflip")
    stats_df.to_csv(out / "circuit_level_stats.csv", index=False)
    pivot = stats_df.pivot(index="circuit", columns="medication", values="mean").reindex(columns=["OFF", "ON"])
    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    im = ax.imshow(pivot.to_numpy(dtype=float), cmap="coolwarm", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    fig.colorbar(im, ax=ax, label="active - sham")
    ax.set_title("Circuit-level active GVS effect")
    save_fig(fig, out / "circuit_level_heatmap.png")
    best = stats_df.loc[stats_df["p_signflip"].idxmin()]
    return {
        "analysis_id": "09",
        "analysis": "Circuit-level aggregation",
        "primary_metric": "active minus sham circuit score",
        "best_p": float(best["p_signflip"]),
        "effect": float(best["mean"]),
        "result_file": str(out / "circuit_level_stats.csv"),
    }


def maxt_permutation(cell_matrix: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    arr = cell_matrix.to_numpy(dtype=float)
    n_per_cell = np.sum(np.isfinite(arr), axis=0)
    means = np.nanmean(arr, axis=0)
    sds = np.nanstd(arr, axis=0, ddof=1)
    obs_t = means / (sds / np.sqrt(n_per_cell))
    obs_t[~np.isfinite(obs_t)] = np.nan
    max_null = []
    signs = rng.choice([-1.0, 1.0], size=(N_PERMUTATIONS, arr.shape[0], 1))
    signed = signs * arr[None, :, :]
    null_means = np.nanmean(signed, axis=1)
    null_sds = np.nanstd(signed, axis=1, ddof=1)
    null_t = null_means / (null_sds / np.sqrt(n_per_cell[None, :]))
    null_t[~np.isfinite(null_t)] = np.nan
    max_null = np.nanmax(np.abs(null_t), axis=1)
    p_maxt = np.array(
        [
            (1 + np.count_nonzero(max_null >= abs(t))) / (max_null.size + 1) if np.isfinite(t) else np.nan
            for t in obs_t
        ]
    )
    out = pd.DataFrame({"mean_delta": means, "t_stat": obs_t, "p_maxt": p_maxt, "n_subjects": n_per_cell}, index=cell_matrix.columns)
    return out.reset_index()


def analysis_10_maxt_heatmap(data: PreparedData, rng: np.random.Generator) -> dict[str, Any]:
    out = ensure_dir(OUT_DIR / "10_permutation_maxt_heatmap")
    rows = []
    for med in ["OFF", "ON"]:
        subset = data.subject_roi_condition.loc[data.subject_roi_condition["medication"].eq(med)].copy()
        matrix = subset.pivot_table(index="subject", columns=["condition_label", "roi_label"], values="roi_delta", aggfunc="mean")
        matrix = matrix.reindex(columns=pd.MultiIndex.from_product([ACTIVE_LABELS, data.roi_cols], names=["condition_label", "roi_label"]))
        res = maxt_permutation(matrix, rng)
        res["medication"] = med
        rows.append(res)
        heat = res.pivot(index="roi_label", columns="condition_label", values="t_stat").reindex(index=data.roi_cols, columns=ACTIVE_LABELS)
        fig, ax = plt.subplots(figsize=(8.8, 9.8))
        vmax = np.nanpercentile(np.abs(heat.to_numpy(dtype=float)), 98)
        im = ax.imshow(heat.to_numpy(dtype=float), cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(ACTIVE_LABELS)))
        ax.set_xticklabels(ACTIVE_LABELS)
        ax.set_yticks(range(len(data.roi_cols)))
        ax.set_yticklabels(data.roi_cols, fontsize=7)
        ax.set_title(f"{med}: max-T controlled ROI x GVS t-statistics")
        fig.colorbar(im, ax=ax, label="t-stat")
        save_fig(fig, out / f"{med.lower()}_maxt_roi_gvs_heatmap.png")
    stats_df = pd.concat(rows, ignore_index=True)
    stats_df["sig_maxt"] = stats_df["p_maxt"].lt(ALPHA)
    stats_df.to_csv(out / "maxt_roi_gvs_results.csv", index=False)
    best = stats_df.loc[stats_df["p_maxt"].idxmin()]
    return {
        "analysis_id": "10",
        "analysis": "Permutation max-T ROI x GVS heatmap",
        "primary_metric": "max-T corrected ROI x GVS cell",
        "best_p": float(best["p_maxt"]),
        "effect": float(best["mean_delta"]),
        "result_file": str(out / "maxt_roi_gvs_results.csv"),
    }


def analysis_11_empirical_bayes(data: PreparedData, rng: np.random.Generator) -> dict[str, Any]:
    del rng
    out = ensure_dir(OUT_DIR / "11_bayesian_hierarchical_shrinkage")
    rows = []
    for (med, condition, roi), g in data.subject_roi_condition.groupby(["medication", "condition_label", "roi_label"], sort=False):
        vals = g["roi_delta"].dropna().to_numpy(dtype=float)
        if vals.size < 2:
            continue
        rows.append(
            {
                "medication": med,
                "condition_label": condition,
                "roi_label": roi,
                "n": vals.size,
                "observed_mean": float(np.mean(vals)),
                "observed_se": float(stats.sem(vals)),
            }
        )
    obs = pd.DataFrame(rows)
    eb_rows = []
    for med, group in obs.groupby("medication", sort=False):
        y = group["observed_mean"].to_numpy(dtype=float)
        se2 = group["observed_se"].to_numpy(dtype=float) ** 2
        finite = np.isfinite(y) & np.isfinite(se2) & (se2 > 0)
        mu = float(np.average(y[finite], weights=1 / se2[finite])) if finite.any() else float(np.nanmean(y))
        tau2 = max(0.0, float(np.nanvar(y, ddof=1) - np.nanmean(se2)))
        for row in group.itertuples(index=False):
            se2_i = float(row.observed_se) ** 2
            if tau2 <= 0 or se2_i <= 0 or not np.isfinite(se2_i):
                post_mean = mu
                post_sd = float(np.sqrt(se2_i)) if se2_i > 0 else np.nan
            else:
                post_var = 1.0 / (1.0 / se2_i + 1.0 / tau2)
                post_mean = post_var * (float(row.observed_mean) / se2_i + mu / tau2)
                post_sd = float(np.sqrt(post_var))
            prob_gt0 = float(norm.sf(0.0, loc=post_mean, scale=post_sd)) if post_sd > 0 else np.nan
            eb_rows.append(
                {
                    **row._asdict(),
                    "global_mu": mu,
                    "tau": float(np.sqrt(tau2)),
                    "posterior_mean": float(post_mean),
                    "posterior_sd": float(post_sd),
                    "posterior_prob_gt0": prob_gt0,
                    "posterior_prob_direction": max(prob_gt0, 1.0 - prob_gt0) if np.isfinite(prob_gt0) else np.nan,
                    "posterior_direction": "positive" if prob_gt0 >= 0.5 else "negative",
                }
            )
    eb = pd.DataFrame(eb_rows)
    eb["credible_direction_95"] = eb["posterior_prob_direction"].ge(0.95)
    eb.to_csv(out / "empirical_bayes_shrinkage_results.csv", index=False)
    top = eb.sort_values("posterior_prob_direction", ascending=False).head(20)
    top.to_csv(out / "top_posterior_direction_cells.csv", index=False)
    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    labels = top["medication"] + " " + top["condition_label"] + " " + top["roi_label"]
    signed_prob = np.where(top["posterior_direction"].eq("positive"), top["posterior_prob_direction"], -top["posterior_prob_direction"])
    ax.barh(np.arange(len(top)), signed_prob)
    ax.set_yticks(np.arange(len(top)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.axvline(0, color="#555555", linewidth=1)
    ax.set_xlabel("Posterior directional probability (signed)")
    ax.set_title("Empirical-Bayes shrinkage: strongest directional cells")
    save_fig(fig, out / "top_posterior_direction_cells.png")
    best = eb.loc[eb["posterior_prob_direction"].idxmax()]
    return {
        "analysis_id": "11",
        "analysis": "Bayesian hierarchical shrinkage approximation",
        "primary_metric": "posterior directional probability after shrinkage",
        "best_p": float(1.0 - best["posterior_prob_direction"]),
        "effect": float(best["posterior_mean"]),
        "result_file": str(out / "empirical_bayes_shrinkage_results.csv"),
    }


def run_decoder(df: pd.DataFrame, roi_cols: list[str], label: str, out: Path) -> dict[str, Any]:
    if not HAS_SKLEARN:
        return {"label": label, "status": "skipped", "reason": "sklearn unavailable"}
    sub = df.dropna(subset=["active"]).copy()
    subjects = sub["subject"].astype(str)
    n_splits = min(5, subjects.nunique())
    if n_splits < 2 or sub["active"].nunique() < 2:
        return {"label": label, "status": "skipped", "reason": "not enough groups/classes"}
    pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")),
        ]
    )
    gkf = GroupKFold(n_splits=n_splits)
    y = sub["active"].astype(int).to_numpy()
    X = sub[roi_cols].to_numpy(dtype=float)
    pred = np.full(y.size, np.nan)
    prob = np.full(y.size, np.nan)
    fold_rows = []
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=subjects), start=1):
        pipe.fit(X[train_idx], y[train_idx])
        fold_pred = pipe.predict(X[test_idx])
        fold_prob = pipe.predict_proba(X[test_idx])[:, 1]
        pred[test_idx] = fold_pred
        prob[test_idx] = fold_prob
        fold_rows.append(
            {
                "label": label,
                "fold": fold,
                "n_test": int(test_idx.size),
                "accuracy": float(accuracy_score(y[test_idx], fold_pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y[test_idx], fold_pred)),
                "auc": float(roc_auc_score(y[test_idx], fold_prob)) if len(np.unique(y[test_idx])) == 2 else np.nan,
            }
        )
    pd.DataFrame(fold_rows).to_csv(out / f"{label}_decoder_folds.csv", index=False)
    predictions = sub[["subject", "session", "medication", "run", "condition_label", "active"]].copy()
    predictions["predicted_active"] = pred
    predictions["prob_active"] = prob
    predictions.to_csv(out / f"{label}_decoder_predictions.csv", index=False)
    return {
        "label": label,
        "status": "ok",
        "n": int(y.size),
        "n_subjects": int(subjects.nunique()),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "auc": float(roc_auc_score(y, prob)),
    }


def analysis_12_predictive_decoding(data: PreparedData, rng: np.random.Generator) -> dict[str, Any]:
    del rng
    out = ensure_dir(OUT_DIR / "12_predictive_decoding")
    rows = [run_decoder(data.trial, data.roi_cols, "all_sessions", out)]
    for med in ["OFF", "ON"]:
        rows.append(run_decoder(data.trial.loc[data.trial["medication"].eq(med)], data.roi_cols, med.lower(), out))
    stats_df = pd.DataFrame(rows)
    stats_df.to_csv(out / "active_vs_sham_decoder_summary.csv", index=False)
    plot_df = stats_df.loc[stats_df["status"].eq("ok")]
    if not plot_df.empty:
        fig, ax = plt.subplots(figsize=(5.4, 3.6))
        ax.bar(plot_df["label"], plot_df["auc"], color="#4B7AA8")
        ax.axhline(0.5, color="#555555", linestyle="--")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Group-CV AUC")
        ax.set_title("Active GVS vs sham decoding")
        save_fig(fig, out / "active_vs_sham_decoder_auc.png")
    best = plot_df.sort_values("auc", ascending=False).head(1)
    auc = float(best["auc"].iloc[0]) if not best.empty else np.nan
    return {
        "analysis_id": "12",
        "analysis": "Predictive decoding active GVS vs sham",
        "primary_metric": "subject-group cross-validated AUC",
        "best_p": np.nan,
        "effect": auc,
        "result_file": str(out / "active_vs_sham_decoder_summary.csv"),
    }


def correlation_row(x: pd.Series, y: pd.Series, label: str) -> dict[str, Any]:
    valid = np.isfinite(x.astype(float)) & np.isfinite(y.astype(float))
    xv = x[valid].astype(float).to_numpy()
    yv = y[valid].astype(float).to_numpy()
    if xv.size < 4:
        return {"label": label, "n": int(xv.size), "r": np.nan, "p_value": np.nan}
    r, p = stats.pearsonr(xv, yv)
    rho, p_s = stats.spearmanr(xv, yv)
    return {"label": label, "n": int(xv.size), "pearson_r": float(r), "p_value": float(p), "spearman_rho": float(rho), "spearman_p": float(p_s)}


def analysis_13_behaviour_linked(data: PreparedData, rng: np.random.Generator) -> dict[str, Any]:
    del rng
    out = ensure_dir(OUT_DIR / "13_behaviour_linked_network")
    rows = []
    merged = pd.DataFrame()
    if BEHAVIOUR_SESSION.exists():
        beh = pd.read_csv(BEHAVIOUR_SESSION)
        merged = data.subject_network_any.merge(beh, on=["subject", "session"], how="inner")
        merged.to_csv(out / "network_delta_with_session_behaviour.csv", index=False)
        for med, g in merged.groupby("medication", sort=False):
            rows.append(correlation_row(g["network_delta"], g["mean_rt"], f"{med}_network_delta_vs_mean_rt") | {"medication": med, "behaviour_metric": "mean_rt"})
            rows.append(correlation_row(g["network_delta"], g["rt_variability"], f"{med}_network_delta_vs_rt_variability") | {"medication": med, "behaviour_metric": "rt_variability"})
    if BEHAVIOUR_PAIRED.exists():
        paired = pd.read_csv(BEHAVIOUR_PAIRED)
        network_wide = data.subject_network_any.pivot(index="subject", columns="medication", values="network_delta").reset_index()
        if {"OFF", "ON"}.issubset(network_wide.columns):
            network_wide["network_delta_on_minus_off"] = network_wide["ON"] - network_wide["OFF"]
            paired_merged = network_wide.merge(paired, on="subject", how="inner")
            paired_merged.to_csv(out / "network_interaction_with_behaviour_change.csv", index=False)
            for metric in ["mean_rt_improvement_off_minus_on", "rt_variability_improvement_off_minus_on"]:
                if metric in paired_merged.columns:
                    rows.append(correlation_row(paired_merged["network_delta_on_minus_off"], paired_merged[metric], f"network_interaction_vs_{metric}") | {"medication": "ON_minus_OFF", "behaviour_metric": metric})
    stats_df = add_fdr(pd.DataFrame(rows), "p_value") if rows else pd.DataFrame([{"status": "skipped", "reason": "No behaviour summary matched subjects/sessions"}])
    stats_df.to_csv(out / "behaviour_linked_correlation_stats.csv", index=False)
    if not merged.empty:
        fig, ax = plt.subplots(figsize=(5.2, 4.0))
        for med, g in merged.groupby("medication", sort=False):
            ax.scatter(g["network_delta"], g["mean_rt"], label=med, s=36, alpha=0.8)
        ax.set_xlabel("Network active GVS - sham")
        ax.set_ylabel("Mean RT")
        ax.set_title("Network GVS effect vs behavior")
        ax.legend()
        save_fig(fig, out / "network_delta_vs_mean_rt.png")
    pvals = pd.to_numeric(stats_df.get("p_value", pd.Series(dtype=float)), errors="coerce")
    best_idx = pvals.idxmin() if pvals.notna().any() else None
    return {
        "analysis_id": "13",
        "analysis": "Behaviour-linked network analysis",
        "primary_metric": "correlation of network GVS delta with RT summaries",
        "best_p": float(pvals.loc[best_idx]) if best_idx is not None else np.nan,
        "effect": float(stats_df.loc[best_idx, "pearson_r"]) if best_idx is not None and "pearson_r" in stats_df else np.nan,
        "result_file": str(out / "behaviour_linked_correlation_stats.csv"),
    }


def project_beta_with_threshold(beta_path: Path, weights: np.ndarray, percentile: float) -> np.ndarray:
    positive = np.isfinite(weights) & (weights > 0)
    threshold = float(np.percentile(weights[positive], percentile))
    mask = positive & (weights >= threshold)
    selected_weights = weights[mask].astype(np.float64)
    beta = np.load(beta_path, mmap_mode="r")
    flat = beta.reshape(-1, beta.shape[-1])
    selected = np.asarray(flat[mask.ravel(), :], dtype=np.float64)
    finite = np.isfinite(selected)
    numerator = np.nansum(selected * selected_weights.ravel()[:, None], axis=0)
    denominator = np.sum(finite * selected_weights.ravel()[:, None], axis=0)
    out = np.full(beta.shape[-1], np.nan, dtype=float)
    valid = denominator > 0
    out[valid] = numerator[valid] / denominator[valid]
    return out


def project_beta_for_percentiles(beta_path: Path, weights: np.ndarray, percentiles: list[float]) -> dict[float, np.ndarray]:
    """Project one run once through the largest mask, then reuse subsets."""
    positive = np.isfinite(weights) & (weights > 0)
    thresholds = {float(p): float(np.percentile(weights[positive], float(p))) for p in percentiles}
    min_percentile = float(min(percentiles))
    base_mask = positive & (weights >= thresholds[min_percentile])
    base_indices = np.flatnonzero(base_mask.ravel())
    weight_flat = weights.ravel().astype(np.float64)
    base_weights = weight_flat[base_indices]

    beta = np.load(beta_path, mmap_mode="r")
    flat = beta.reshape(-1, beta.shape[-1])
    selected = np.asarray(flat[base_indices, :], dtype=np.float64)
    finite = np.isfinite(selected)

    out: dict[float, np.ndarray] = {}
    for percentile in percentiles:
        keep = base_weights >= thresholds[float(percentile)]
        selected_weights = base_weights[keep]
        selected_beta = selected[keep, :]
        selected_finite = finite[keep, :]
        numerator = np.nansum(selected_beta * selected_weights[:, None], axis=0)
        denominator = np.sum(selected_finite * selected_weights[:, None], axis=0)
        projected = np.full(beta.shape[-1], np.nan, dtype=float)
        valid = denominator > 0
        projected[valid] = numerator[valid] / denominator[valid]
        out[float(percentile)] = projected
    return out


def analysis_14_threshold_sensitivity(data: PreparedData, rng: np.random.Generator) -> dict[str, Any]:
    del data
    out = ensure_dir(OUT_DIR / "14_threshold_sensitivity")
    if not RUN_INVENTORY.exists() or not WEIGHT_MAP.exists():
        skipped = pd.DataFrame([{"status": "skipped", "reason": "run inventory or weight map missing"}])
        skipped.to_csv(out / "threshold_sensitivity_stats.csv", index=False)
        return {"analysis_id": "14", "analysis": "Threshold sensitivity", "primary_metric": "p85/p90/p95 network projection", "best_p": np.nan, "effect": np.nan, "result_file": str(out / "threshold_sensitivity_stats.csv")}
    inventory = pd.read_csv(RUN_INVENTORY)
    weights = nib.load(str(WEIGHT_MAP)).get_fdata(dtype=np.float32)
    rows = []
    percentiles = [85.0, 90.0, 95.0]
    grouped_inventory = list(inventory.groupby("source_beta_path", sort=False))
    for run_index, (source_beta_path, run_rows) in enumerate(grouped_inventory, start=1):
        beta_path = Path(str(source_beta_path))
        if not beta_path.exists():
            continue
        print(f"  threshold projection {run_index}/{len(grouped_inventory)}: {beta_path.name}", flush=True)
        projected_by_percentile = project_beta_for_percentiles(beta_path, weights, percentiles)
        for row in run_rows.itertuples(index=False):
            start = int(row.trial_start)
            stop = int(row.trial_stop)
            for percentile, projected in projected_by_percentile.items():
                vals = projected[start:stop]
                finite = vals[np.isfinite(vals)]
                rows.append(
                    {
                        "percentile": percentile,
                        "subject": row.subject,
                        "session": int(row.session),
                        "medication": row.medication,
                        "run": int(row.run),
                        "condition_code": row.condition_code,
                        "condition_label": clean_label(row.condition_label),
                        "mean_projection": float(np.mean(finite)) if finite.size else np.nan,
                        "n_trials": int(finite.size),
                    }
                )
    proj = pd.DataFrame(rows)
    proj.to_csv(out / "threshold_projected_condition_means.csv", index=False)
    if proj.empty:
        skipped = pd.DataFrame([{"status": "skipped", "reason": "No beta paths from inventory were available"}])
        skipped.to_csv(out / "threshold_sensitivity_stats.csv", index=False)
        return {"analysis_id": "14", "analysis": "Threshold sensitivity", "primary_metric": "p85/p90/p95 network projection", "best_p": np.nan, "effect": np.nan, "result_file": str(out / "threshold_sensitivity_stats.csv")}
    subject_condition = (
        proj.groupby(["percentile", "subject", "session", "medication", "condition_label"], dropna=False)["mean_projection"]
        .mean()
        .reset_index()
    )
    sham = subject_condition.loc[subject_condition["condition_label"].eq("sham")].rename(columns={"mean_projection": "sham_projection"})
    sham = sham[["percentile", "subject", "session", "sham_projection"]]
    active = subject_condition.loc[subject_condition["condition_label"].ne("sham")].merge(sham, on=["percentile", "subject", "session"], how="inner")
    active["projection_delta"] = active["mean_projection"] - active["sham_projection"]
    subject_any = (
        active.groupby(["percentile", "subject", "session", "medication"], dropna=False)["projection_delta"]
        .mean()
        .reset_index()
    )
    subject_any.to_csv(out / "threshold_subject_any_gvs_projection_delta.csv", index=False)
    stats_rows = []
    for (percentile, med), g in subject_any.groupby(["percentile", "medication"], sort=False):
        row = one_sample_summary(g["projection_delta"].to_numpy(), label=f"p{percentile:g}_{med}", rng=rng)
        row.update({"percentile": percentile, "medication": med})
        stats_rows.append(row)
    stats_df = add_fdr(pd.DataFrame(stats_rows), "p_signflip")
    stats_df.to_csv(out / "threshold_sensitivity_stats.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for med, g in stats_df.groupby("medication", sort=False):
        ax.errorbar(g["percentile"], g["mean"], yerr=1.96 * g["sem"], marker="o", label=med, capsize=4)
    ax.axhline(0, color="#555555", linestyle="--")
    ax.set_xlabel("Weight percentile threshold")
    ax.set_ylabel("Any-GVS projection delta")
    ax.set_title("Threshold sensitivity")
    ax.legend()
    save_fig(fig, out / "threshold_sensitivity.png")
    best = stats_df.loc[stats_df["p_signflip"].idxmin()]
    return {
        "analysis_id": "14",
        "analysis": "Threshold sensitivity",
        "primary_metric": "p85/p90/p95 projected network active GVS delta",
        "best_p": float(best["p_signflip"]),
        "effect": float(best["mean"]),
        "result_file": str(out / "threshold_sensitivity_stats.csv"),
    }


def write_method_summary(summary: pd.DataFrame) -> None:
    summary = summary.copy()
    summary["best_p_numeric"] = pd.to_numeric(summary["best_p"], errors="coerce")
    summary["rank"] = summary["best_p_numeric"].rank(method="min", na_option="bottom").astype(int)
    summary = summary.sort_values(["rank", "analysis_id"])
    summary.to_csv(OUT_DIR / "all_14_methods_summary.csv", index=False)

    promising = summary.loc[summary["best_p_numeric"].notna()].head(5)
    lines = [
        "# GVS Vigour-Network Analysis Summary",
        "",
        f"Generated by `{Path(__file__).name}`.",
        "",
        "## Top Methods By Primary Evidence",
        "",
    ]
    for row in promising.itertuples(index=False):
        lines.append(
            f"- {row.analysis_id}. {row.analysis}: best p={row.best_p_numeric:.4g}, "
            f"effect={row.effect:.6g}; metric={row.primary_metric}"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Mixed-model p-values are approximate Wald p-values.",
            "- Permutation/sign-flip p-values preserve subject-level pairing.",
            "- The similarity perturbation and edge-distance analyses are directional/descriptive; interpret them as effect-size evidence unless paired with a specific null contrast.",
            "- Threshold sensitivity reprojects beta volumes through p85, p90, and p95 voxel-weight masks when the source beta paths are available.",
            "",
        ]
    )
    (OUT_DIR / "PROMISING_METHODS.md").write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-threshold-sensitivity", action="store_true", help="Skip analysis 14, which reprojects beta volumes.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(OUT_DIR)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    rng = np.random.default_rng(RNG_SEED)
    data = load_and_prepare()
    write_json(
        OUT_DIR / "input_manifest.json",
        {
            "trial_table": TRIAL_TABLE,
            "roi_definition": ROI_DEF,
            "behaviour_session": BEHAVIOUR_SESSION,
            "behaviour_paired": BEHAVIOUR_PAIRED,
            "run_inventory": RUN_INVENTORY,
            "weight_map": WEIGHT_MAP,
            "n_trial_rows": int(data.trial.shape[0]),
            "n_subjects": int(data.trial["subject"].nunique()),
            "n_rois": int(len(data.roi_cols)),
            "roi_cols": data.roi_cols,
            "n_permutations": N_PERMUTATIONS,
            "rng_seed": RNG_SEED,
        },
    )
    data.trial[["subject", "session", "medication", "run", "condition_label", "active", "network_score", "roi_dispersion"]].to_csv(
        OUT_DIR / "prepared_trial_network_scores.csv", index=False
    )
    data.block.to_csv(OUT_DIR / "prepared_block_network_scores.csv", index=False)
    data.block_delta.to_csv(OUT_DIR / "prepared_block_network_deltas.csv", index=False)
    data.subject_network_any.to_csv(OUT_DIR / "prepared_subject_network_any_gvs_deltas.csv", index=False)

    runners = [
        analysis_01_collapsed_projection,
        analysis_02_block_mixedlm,
        analysis_03_subject_any_gvs,
        analysis_04_multivariate_network,
        analysis_05_medication_interaction,
        analysis_06_similarity_to_sham,
        analysis_07_variability_dispersion,
        analysis_08_connectivity_coactivation,
        analysis_09_circuit_level,
        analysis_10_maxt_heatmap,
        analysis_11_empirical_bayes,
        analysis_12_predictive_decoding,
        analysis_13_behaviour_linked,
    ]
    summary_rows = []
    for runner in runners:
        print(f"Running {runner.__name__}", flush=True)
        summary_rows.append(runner(data, rng))
    if not args.skip_threshold_sensitivity:
        print("Running analysis_14_threshold_sensitivity", flush=True)
        summary_rows.append(analysis_14_threshold_sensitivity(data, rng))
    else:
        summary_rows.append(
            {
                "analysis_id": "14",
                "analysis": "Threshold sensitivity",
                "primary_metric": "p85/p90/p95 network projection",
                "best_p": np.nan,
                "effect": np.nan,
                "result_file": "skipped by --skip-threshold-sensitivity",
            }
        )
    write_method_summary(pd.DataFrame(summary_rows))
    print(f"Saved outputs under {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
