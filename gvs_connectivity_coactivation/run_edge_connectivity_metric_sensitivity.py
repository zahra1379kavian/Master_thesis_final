#!/usr/bin/env python3
"""Edge-wise FC sensitivity across linear and nonlinear connectivity metrics."""

from __future__ import annotations

import importlib.util
import sys
import warnings
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BASE_SCRIPT = HERE / "run_all_gvs_network_analyses.py"
OUT = ROOT / "figures" / "GVS_effects" / "08_connectivity_coactivation" / "metric_sensitivity"


def load_base_module():
    spec = importlib.util.spec_from_file_location("gvs_network_analyses", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = load_base_module()

try:
    from sklearn.covariance import LedoitWolf

    HAS_LEDOIT_WOLF = True
except Exception:
    HAS_LEDOIT_WOLF = False


def prepare_complete_matrix(values: np.ndarray, min_obs: int = 4) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < min_obs:
        return np.empty((0, 0), dtype=float), np.array([], dtype=int)
    finite_cols = np.sum(np.isfinite(arr), axis=0) >= min_obs
    if finite_cols.sum() < 2:
        return np.empty((0, 0), dtype=float), np.array([], dtype=int)
    work = arr[:, finite_cols].copy()
    col_means = np.nanmean(work, axis=0)
    inds = np.where(~np.isfinite(work))
    work[inds] = np.take(col_means, inds[1])
    return work, np.flatnonzero(finite_cols)


def fill_full_edges(corr: np.ndarray, valid_idx: np.ndarray, n_rois: int) -> np.ndarray:
    full = np.full((n_rois, n_rois), np.nan, dtype=float)
    full[np.ix_(valid_idx, valid_idx)] = corr
    return full[np.triu_indices(n_rois, k=1)]


def pearson_edges(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    n_rois = arr.shape[1] if arr.ndim == 2 else 0
    edges = np.full(n_rois * (n_rois - 1) // 2, np.nan, dtype=float)
    work, valid_idx = prepare_complete_matrix(arr)
    if work.size == 0:
        return edges
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(work, rowvar=False)
    return fill_full_edges(corr, valid_idx, n_rois)


def spearman_edges(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    n_rois = arr.shape[1] if arr.ndim == 2 else 0
    edges = np.full(n_rois * (n_rois - 1) // 2, np.nan, dtype=float)
    work, valid_idx = prepare_complete_matrix(arr)
    if work.size == 0:
        return edges
    ranked = np.apply_along_axis(stats.rankdata, 0, work)
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(ranked, rowvar=False)
    return fill_full_edges(corr, valid_idx, n_rois)


def partial_corr_edges(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    n_rois = arr.shape[1] if arr.ndim == 2 else 0
    edges = np.full(n_rois * (n_rois - 1) // 2, np.nan, dtype=float)
    work, valid_idx = prepare_complete_matrix(arr)
    if work.size == 0:
        return edges

    work = work - work.mean(axis=0, keepdims=True)
    sd = work.std(axis=0, ddof=0, keepdims=True)
    sd[sd == 0] = 1.0
    work = work / sd
    if HAS_LEDOIT_WOLF:
        cov = LedoitWolf().fit(work).covariance_
    else:
        cov = np.cov(work, rowvar=False)
        cov = cov + np.eye(cov.shape[0]) * 1e-6
    precision = np.linalg.pinv(cov)
    diag = np.diag(precision)
    denom = np.sqrt(np.outer(diag, diag))
    with np.errstate(invalid="ignore", divide="ignore"):
        partial = -precision / denom
    np.fill_diagonal(partial, 1.0)
    partial = np.clip(partial, -1.0, 1.0)
    return fill_full_edges(partial, valid_idx, n_rois)


def covariance_edges(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    n_rois = arr.shape[1] if arr.ndim == 2 else 0
    edges = np.full(n_rois * (n_rois - 1) // 2, np.nan, dtype=float)
    work, valid_idx = prepare_complete_matrix(arr)
    if work.size == 0:
        return edges
    cov = np.cov(work, rowvar=False)
    return fill_full_edges(cov, valid_idx, n_rois)


def precision_edges(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    n_rois = arr.shape[1] if arr.ndim == 2 else 0
    edges = np.full(n_rois * (n_rois - 1) // 2, np.nan, dtype=float)
    work, valid_idx = prepare_complete_matrix(arr)
    if work.size == 0:
        return edges
    if HAS_LEDOIT_WOLF:
        cov = LedoitWolf().fit(work).covariance_
    else:
        cov = np.cov(work, rowvar=False)
        cov = cov + np.eye(cov.shape[0]) * 1e-6
    precision = -np.linalg.pinv(cov)
    np.fill_diagonal(precision, np.nan)
    return fill_full_edges(precision, valid_idx, n_rois)


def quantile_codes(x: np.ndarray, n_bins: int = 3) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size == 0 or np.nanstd(x) == 0:
        return np.zeros(x.size, dtype=int)
    bins = min(n_bins, max(2, int(np.unique(x).size)))
    order = np.argsort(x, kind="mergesort")
    codes = np.empty(x.size, dtype=int)
    codes[order] = np.floor(np.arange(x.size) * bins / x.size).astype(int)
    return np.minimum(codes, bins - 1)


def mutual_information_pair(x: np.ndarray, y: np.ndarray, *, normalized: bool = False) -> float:
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite].astype(float)
    y = y[finite].astype(float)
    if x.size < 4:
        return float("nan")
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    x_code = quantile_codes(x)
    y_code = quantile_codes(y)
    joint = np.zeros((int(x_code.max()) + 1, int(y_code.max()) + 1), dtype=float)
    np.add.at(joint, (x_code, y_code), 1.0)
    joint = joint / joint.sum()
    px = joint.sum(axis=1)
    py = joint.sum(axis=0)
    denom = px[:, None] * py[None, :]
    nz = joint > 0
    mi = float(np.sum(joint[nz] * np.log(joint[nz] / denom[nz])))
    if not normalized:
        return mi
    hx = -float(np.sum(px[px > 0] * np.log(px[px > 0])))
    hy = -float(np.sum(py[py > 0] * np.log(py[py > 0])))
    norm = np.sqrt(hx * hy)
    return float(mi / norm) if norm > 0 else 0.0


def mutual_information_edges(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    n_rois = arr.shape[1] if arr.ndim == 2 else 0
    edges = np.full(n_rois * (n_rois - 1) // 2, np.nan, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 4 or n_rois < 2:
        return edges
    edge_idx = np.triu_indices(n_rois, k=1)
    for edge_id, (i, j) in enumerate(zip(edge_idx[0], edge_idx[1])):
        edges[edge_id] = mutual_information_pair(arr[:, int(i)], arr[:, int(j)])
    return edges


def normalized_mutual_information_edges(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    n_rois = arr.shape[1] if arr.ndim == 2 else 0
    edges = np.full(n_rois * (n_rois - 1) // 2, np.nan, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 4 or n_rois < 2:
        return edges
    edge_idx = np.triu_indices(n_rois, k=1)
    for edge_id, (i, j) in enumerate(zip(edge_idx[0], edge_idx[1])):
        edges[edge_id] = mutual_information_pair(arr[:, int(i)], arr[:, int(j)], normalized=True)
    return edges


def rbf_kernel_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    diffs = x[:, None] - x[None, :]
    d2 = diffs * diffs
    positive = d2[d2 > 0]
    sigma2 = float(np.median(positive)) if positive.size else 1.0
    if sigma2 <= 0 or not np.isfinite(sigma2):
        sigma2 = 1.0
    return np.exp(-d2 / (2.0 * sigma2))


def centered_kernel(kernel: np.ndarray) -> np.ndarray:
    return kernel - kernel.mean(axis=0, keepdims=True) - kernel.mean(axis=1, keepdims=True) + kernel.mean()


def rbf_hsic_pair(x: np.ndarray, y: np.ndarray) -> float:
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite].astype(float)
    y = y[finite].astype(float)
    if x.size < 4:
        return float("nan")
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    kx = centered_kernel(rbf_kernel_1d(x))
    ky = centered_kernel(rbf_kernel_1d(y))
    hsic = float(np.mean(kx * ky))
    norm = float(np.sqrt(np.mean(kx * kx) * np.mean(ky * ky)))
    return float(hsic / norm) if norm > 0 else 0.0


def rbf_hsic_edges(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    n_rois = arr.shape[1] if arr.ndim == 2 else 0
    edges = np.full(n_rois * (n_rois - 1) // 2, np.nan, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 4 or n_rois < 2:
        return edges
    edge_idx = np.triu_indices(n_rois, k=1)
    for edge_id, (i, j) in enumerate(zip(edge_idx[0], edge_idx[1])):
        edges[edge_id] = rbf_hsic_pair(arr[:, int(i)], arr[:, int(j)])
    return edges


def distance_correlation_pair(x: np.ndarray, y: np.ndarray) -> float:
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite].astype(float)
    y = y[finite].astype(float)
    if x.size < 4:
        return float("nan")
    ax = np.abs(x[:, None] - x[None, :])
    ay = np.abs(y[:, None] - y[None, :])
    ax = ax - ax.mean(axis=0, keepdims=True) - ax.mean(axis=1, keepdims=True) + ax.mean()
    ay = ay - ay.mean(axis=0, keepdims=True) - ay.mean(axis=1, keepdims=True) + ay.mean()
    dcov2 = float(np.mean(ax * ay))
    dvarx = float(np.mean(ax * ax))
    dvary = float(np.mean(ay * ay))
    denom = np.sqrt(dvarx * dvary)
    if denom <= 0 or not np.isfinite(denom):
        return 0.0
    return float(np.sqrt(max(dcov2 / denom, 0.0)))


def distance_corr_edges(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    n_rois = arr.shape[1] if arr.ndim == 2 else 0
    edges = np.full(n_rois * (n_rois - 1) // 2, np.nan, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 4 or n_rois < 2:
        return edges
    edge_idx = np.triu_indices(n_rois, k=1)
    for edge_id, (i, j) in enumerate(zip(edge_idx[0], edge_idx[1])):
        edges[edge_id] = distance_correlation_pair(arr[:, int(i)], arr[:, int(j)])
    return edges


MetricFn = Callable[[np.ndarray], np.ndarray]
METRICS: list[tuple[str, str, MetricFn]] = [
    ("pearson_r", "linear_correlation", pearson_edges),
    ("covariance", "linear_covariance", covariance_edges),
    ("spearman_rho", "rank_monotonic", spearman_edges),
    ("partial_corr_ledoitwolf", "linear_conditional", partial_corr_edges),
    ("precision_ledoitwolf", "linear_conditional_precision", precision_edges),
    ("mutual_info_quantile", "nonlinear_information", mutual_information_edges),
    ("normalized_mutual_info_quantile", "nonlinear_information", normalized_mutual_information_edges),
    ("rbf_hsic", "nonlinear_kernel_dependence", rbf_hsic_edges),
    ("distance_corr", "nonlinear_dependence", distance_corr_edges),
]


def block_edge_deltas(data, metric_name: str, metric_fn: MetricFn, edge_index: pd.DataFrame) -> pd.DataFrame:
    edge_lookup: dict[tuple[object, ...], np.ndarray] = {}
    blocks = []
    group_cols = ["subject", "session", "medication", "run", "run_id", "condition_label", "condition_code"]
    for key, block_df in data.trial.groupby(group_cols, sort=False):
        edges = metric_fn(block_df[data.roi_cols].to_numpy(dtype=float))
        record = dict(zip(group_cols, key))
        blocks.append(record)
        subject, session, medication, run, run_id, _condition_label, condition_code = key
        edge_lookup[(subject, session, medication, run, run_id, condition_code)] = edges

    active = pd.DataFrame(blocks).loc[lambda df: df["condition_code"].ne(base.SHAM_CODE)]
    frames = []
    for row in active.itertuples(index=False):
        active_key = (row.subject, row.session, row.medication, row.run, row.run_id, row.condition_code)
        sham_key = (row.subject, row.session, row.medication, row.run, row.run_id, base.SHAM_CODE)
        active_edges = edge_lookup.get(active_key, np.array([]))
        sham_edges = edge_lookup.get(sham_key, np.array([]))
        n = min(active_edges.size, sham_edges.size)
        if n == 0:
            continue
        frame = edge_index.iloc[:n].copy()
        frame["metric"] = metric_name
        frame["subject"] = row.subject
        frame["session"] = row.session
        frame["medication"] = row.medication
        frame["run"] = row.run
        frame["run_id"] = row.run_id
        frame["condition_label"] = row.condition_label
        frame["condition_code"] = row.condition_code
        frame["active_value"] = active_edges[:n]
        frame["sham_value"] = sham_edges[:n]
        frame["edge_delta"] = active_edges[:n] - sham_edges[:n]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def append_stats(
    rows: list[pd.DataFrame],
    matrix: pd.DataFrame,
    *,
    metric: str,
    metric_family: str,
    analysis_view: str,
    label: str,
    fdr_scope: str,
    rng: np.random.Generator,
    **metadata,
) -> None:
    if matrix.empty:
        return
    stats_df = base.one_sample_matrix_summary(matrix, label=label, rng=rng)
    stats_df["metric"] = metric
    stats_df["metric_family"] = metric_family
    stats_df["analysis_view"] = analysis_view
    stats_df["fdr_scope"] = fdr_scope
    for key, value in metadata.items():
        stats_df[key] = value
    rows.append(stats_df)


def metric_stats(edge_delta: pd.DataFrame, metric: str, metric_family: str, rng: np.random.Generator) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    edge_cols = ["edge_id", "roi_i", "roi_j", "edge_label"]
    if edge_delta.empty:
        return pd.DataFrame()

    subject_any = (
        edge_delta.groupby(["subject", "session", "medication", *edge_cols], dropna=False)
        .agg(edge_delta=("edge_delta", "mean"), n_active_blocks=("edge_delta", "count"))
        .reset_index()
    )
    for med, group in subject_any.groupby("medication", sort=False):
        matrix = group.pivot_table(index=["subject", "session"], columns=edge_cols, values="edge_delta", aggfunc="mean")
        append_stats(
            rows,
            matrix,
            metric=metric,
            metric_family=metric_family,
            analysis_view="any_gvs_by_medication",
            label=f"{metric}_{med}_ANY_GVS",
            fdr_scope=f"medication={med};gvs=ANY_GVS",
            rng=rng,
            medication=med,
            gvs_group="ANY_GVS",
        )
    matrix = subject_any.pivot_table(index=["subject", "session", "medication"], columns=edge_cols, values="edge_delta", aggfunc="mean")
    append_stats(
        rows,
        matrix,
        metric=metric,
        metric_family=metric_family,
        analysis_view="any_gvs_by_medication",
        label=f"{metric}_POOLED_ANY_GVS",
        fdr_scope="medication=POOLED;gvs=ANY_GVS",
        rng=rng,
        medication="POOLED",
        gvs_group="ANY_GVS",
    )

    subject_condition = (
        edge_delta.groupby(["subject", "session", "medication", "condition_label", "condition_code", *edge_cols], dropna=False)
        .agg(edge_delta=("edge_delta", "mean"), n_runs=("edge_delta", "count"))
        .reset_index()
    )
    for (med, condition_label, condition_code), group in subject_condition.groupby(
        ["medication", "condition_label", "condition_code"], sort=False
    ):
        matrix = group.pivot_table(index=["subject", "session"], columns=edge_cols, values="edge_delta", aggfunc="mean")
        append_stats(
            rows,
            matrix,
            metric=metric,
            metric_family=metric_family,
            analysis_view="by_gvs_by_medication",
            label=f"{metric}_{med}_{condition_label}",
            fdr_scope=f"medication={med};condition={condition_label}",
            rng=rng,
            medication=med,
            condition_label=condition_label,
            condition_code=condition_code,
        )
    for (condition_label, condition_code), group in subject_condition.groupby(["condition_label", "condition_code"], sort=False):
        matrix = group.pivot_table(index=["subject", "session", "medication"], columns=edge_cols, values="edge_delta", aggfunc="mean")
        append_stats(
            rows,
            matrix,
            metric=metric,
            metric_family=metric_family,
            analysis_view="by_gvs_by_medication",
            label=f"{metric}_POOLED_{condition_label}",
            fdr_scope=f"medication=POOLED;condition={condition_label}",
            rng=rng,
            medication="POOLED",
            condition_label=condition_label,
            condition_code=condition_code,
        )

    run_any = (
        edge_delta.groupby(["subject", "session", "medication", "run", *edge_cols], dropna=False)
        .agg(edge_delta=("edge_delta", "mean"), n_active_conditions=("edge_delta", "count"))
        .reset_index()
    )
    for (med, run), group in run_any.groupby(["medication", "run"], sort=False):
        matrix = group.pivot_table(index=["subject", "session"], columns=edge_cols, values="edge_delta", aggfunc="mean")
        append_stats(
            rows,
            matrix,
            metric=metric,
            metric_family=metric_family,
            analysis_view="any_gvs_by_run",
            label=f"{metric}_{med}_run{run}_ANY_GVS",
            fdr_scope=f"medication={med};run={run};gvs=ANY_GVS",
            rng=rng,
            medication=med,
            run=run,
            gvs_group="ANY_GVS",
        )
    for run, group in run_any.groupby("run", sort=False):
        matrix = group.pivot_table(index=["subject", "session", "medication"], columns=edge_cols, values="edge_delta", aggfunc="mean")
        append_stats(
            rows,
            matrix,
            metric=metric,
            metric_family=metric_family,
            analysis_view="any_gvs_by_run",
            label=f"{metric}_POOLED_run{run}_ANY_GVS",
            fdr_scope=f"medication=POOLED;run={run};gvs=ANY_GVS",
            rng=rng,
            medication="POOLED",
            run=run,
            gvs_group="ANY_GVS",
        )

    run_condition = (
        edge_delta.groupby(["subject", "session", "medication", "run", "condition_label", "condition_code", *edge_cols], dropna=False)
        .agg(edge_delta=("edge_delta", "mean"), n_blocks=("edge_delta", "count"))
        .reset_index()
    )
    for (med, run, condition_label, condition_code), group in run_condition.groupby(
        ["medication", "run", "condition_label", "condition_code"], sort=False
    ):
        matrix = group.pivot_table(index=["subject", "session"], columns=edge_cols, values="edge_delta", aggfunc="mean")
        append_stats(
            rows,
            matrix,
            metric=metric,
            metric_family=metric_family,
            analysis_view="by_gvs_by_run",
            label=f"{metric}_{med}_run{run}_{condition_label}",
            fdr_scope=f"medication={med};run={run};condition={condition_label}",
            rng=rng,
            medication=med,
            run=run,
            condition_label=condition_label,
            condition_code=condition_code,
        )
    for (run, condition_label, condition_code), group in run_condition.groupby(["run", "condition_label", "condition_code"], sort=False):
        matrix = group.pivot_table(index=["subject", "session", "medication"], columns=edge_cols, values="edge_delta", aggfunc="mean")
        append_stats(
            rows,
            matrix,
            metric=metric,
            metric_family=metric_family,
            analysis_view="by_gvs_by_run",
            label=f"{metric}_POOLED_run{run}_{condition_label}",
            fdr_scope=f"medication=POOLED;run={run};condition={condition_label}",
            rng=rng,
            medication="POOLED",
            run=run,
            condition_label=condition_label,
            condition_code=condition_code,
        )

    both_any = (
        subject_any.groupby(["subject", *edge_cols], dropna=False)
        .agg(edge_delta=("edge_delta", "mean"), n_sessions=("edge_delta", "count"))
        .reset_index()
    )
    matrix = both_any.pivot_table(index="subject", columns=edge_cols, values="edge_delta", aggfunc="mean")
    append_stats(
        rows,
        matrix,
        metric=metric,
        metric_family=metric_family,
        analysis_view="any_gvs_both_sessions",
        label=f"{metric}_BOTH_SESSIONS_ANY_GVS",
        fdr_scope="session_pool=BOTH_SESSIONS;gvs=ANY_GVS",
        rng=rng,
        session_pool="BOTH_SESSIONS",
        gvs_group="ANY_GVS",
    )

    both_condition = (
        subject_condition.groupby(["subject", "condition_label", "condition_code", *edge_cols], dropna=False)
        .agg(edge_delta=("edge_delta", "mean"), n_sessions=("edge_delta", "count"))
        .reset_index()
    )
    for (condition_label, condition_code), group in both_condition.groupby(["condition_label", "condition_code"], sort=False):
        matrix = group.pivot_table(index="subject", columns=edge_cols, values="edge_delta", aggfunc="mean")
        append_stats(
            rows,
            matrix,
            metric=metric,
            metric_family=metric_family,
            analysis_view="by_gvs_both_sessions",
            label=f"{metric}_BOTH_SESSIONS_{condition_label}",
            fdr_scope=f"session_pool=BOTH_SESSIONS;condition={condition_label}",
            rng=rng,
            session_pool="BOTH_SESSIONS",
            condition_label=condition_label,
            condition_code=condition_code,
        )

    matrix = edge_delta.pivot_table(
        index=["subject", "session", "medication", "run", "condition_label", "condition_code"],
        columns=edge_cols,
        values="edge_delta",
        aggfunc="mean",
    )
    append_stats(
        rows,
        matrix,
        metric=metric,
        metric_family=metric_family,
        analysis_view="all_subjects_block_pool",
        label=f"{metric}_ALL_SUBJECTS_BLOCK_POOL_ANY_GVS",
        fdr_scope="pool=ALL_SUBJECTS_BLOCKS;gvs=ANY_GVS",
        rng=rng,
        pool="ALL_SUBJECTS_BLOCKS",
        condition_label="ANY_GVS",
        condition_code="ANY_GVS",
        analysis_unit="subject_session_medication_run_condition",
    )
    for (condition_label, condition_code), group in edge_delta.groupby(["condition_label", "condition_code"], sort=False):
        matrix = group.pivot_table(index=["subject", "session", "medication", "run"], columns=edge_cols, values="edge_delta", aggfunc="mean")
        append_stats(
            rows,
            matrix,
            metric=metric,
            metric_family=metric_family,
            analysis_view="all_subjects_block_pool",
            label=f"{metric}_ALL_SUBJECTS_BLOCK_POOL_{condition_label}",
            fdr_scope=f"pool=ALL_SUBJECTS_BLOCKS;condition={condition_label}",
            rng=rng,
            pool="ALL_SUBJECTS_BLOCKS",
            condition_label=condition_label,
            condition_code=condition_code,
            analysis_unit="subject_session_medication_run",
        )

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def summarize_stats(stats_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["metric", "metric_family", "analysis_view", "fdr_scope"]
    for key, group in stats_df.groupby(group_cols, dropna=False, sort=False):
        metric, metric_family, analysis_view, fdr_scope = key
        q = pd.to_numeric(group["q_fdr"], errors="coerce")
        p = pd.to_numeric(group["p_signflip"], errors="coerce")
        best_idx = q.idxmin() if q.notna().any() else p.idxmin()
        best = group.loc[best_idx]
        rows.append(
            {
                "metric": metric,
                "metric_family": metric_family,
                "analysis_view": analysis_view,
                "fdr_scope": fdr_scope,
                "n_edges_tested": int(group["edge_id"].nunique()),
                "n_sig_fdr": int(group["sig_fdr"].sum()),
                "min_p_signflip": float(p.min()) if p.notna().any() else np.nan,
                "min_q_fdr": float(q.min()) if q.notna().any() else np.nan,
                "best_edge": best["edge_label"],
                "best_mean_delta": float(best["mean"]),
                "best_p_signflip": float(best["p_signflip"]),
                "best_q_fdr": float(best["q_fdr"]),
                "best_sig_fdr": bool(best["sig_fdr"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["n_sig_fdr", "min_q_fdr", "metric", "analysis_view"], ascending=[False, True, True, True])


def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(base.RNG_SEED)
    data = base.load_and_prepare()
    edge_index = base.roi_edge_index(data.roi_cols)

    all_stats = []
    for metric, metric_family, metric_fn in METRICS:
        print(f"Running {metric}", flush=True)
        edge_delta = block_edge_deltas(data, metric, metric_fn, edge_index)
        stats_df = metric_stats(edge_delta, metric, metric_family, rng)
        all_stats.append(stats_df)

    stats_df = pd.concat(all_stats, ignore_index=True)
    stats_df["abs_mean"] = stats_df["mean"].abs()
    stats_df["sig_uncorrected"] = stats_df["p_signflip"].lt(base.ALPHA)
    stats_df = base.add_groupwise_fdr(stats_df, "p_signflip", ["metric", "analysis_view", "fdr_scope"])
    front = [
        "metric",
        "metric_family",
        "analysis_view",
        "fdr_scope",
        "edge_id",
        "roi_i",
        "roi_j",
        "edge_label",
        "label",
        "n",
        "mean",
        "t_stat",
        "p_signflip",
        "q_fdr",
        "sig_fdr",
    ]
    stats_df = stats_df[front + [c for c in stats_df.columns if c not in front]]
    stats_df.to_csv(OUT / "edge_connectivity_metric_sensitivity_stats.csv", index=False)
    stats_df.loc[stats_df["sig_fdr"]].to_csv(OUT / "fdr_significant_edge_connectivity_metric_sensitivity.csv", index=False)

    summary = summarize_stats(stats_df)
    summary.to_csv(OUT / "edge_connectivity_metric_sensitivity_summary.csv", index=False)

    top = (
        stats_df.sort_values(["sig_fdr", "q_fdr", "p_signflip", "abs_mean"], ascending=[False, True, True, False])
        .groupby(["metric", "analysis_view", "fdr_scope"], dropna=False)
        .head(10)
        .reset_index(drop=True)
    )
    top.to_csv(OUT / "top_edge_connectivity_metric_sensitivity.csv", index=False)
    print(summary.head(30).to_string(index=False), flush=True)
    print(f"Saved outputs under {OUT}", flush=True)


if __name__ == "__main__":
    main()
