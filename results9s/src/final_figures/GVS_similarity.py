#!/usr/bin/env python3
"""Subject-level GVS beta-matrix similarity analysis.

This analysis works directly on ROI-by-trial beta matrices rather than ROI
connectivity matrices. Each condition-specific matrix is derived from the saved
`results/connectivity/GVS_effects/data/by_gvs/.../selected_beta_trials_gvs-XX.npy`
files using the existing selected-network ROI membership.

Primary similarity metric:
    Flattened Pearson correlation after:
    1. averaging selected voxels into ROI rows,
    2. resampling each ROI row to a fixed trial grid,
    3. z-scoring each ROI row.

Additional metrics:
    - flattened cosine similarity on the resampled raw beta matrices
    - Frobenius norm of the difference between row-z-scored matrices
    - RMSE on the row-z-scored matrices
    - eigenvalue-profile correlation
    - eigenvalue-profile L2 distance

Because the ROI beta matrices are rectangular, the spectral metrics are derived
from the eigenvalues of the ROI Gram matrix `X @ X.T / n_trials` after the same
row-wise z-scoring used by the primary metric.

Comparisons requested:
    1. OFF condition vs OFF sham, within subject
    2. ON condition vs ON sham, within subject
    3. OFF condition, including OFF sham, vs ON sham, within subject
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import textwrap
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection

_HERE = Path(__file__).resolve().parent
_GROUP_ANALYSIS_DIR = _HERE.parent
_REPO_ROOT = _GROUP_ANALYSIS_DIR.parent
if str(_GROUP_ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(_GROUP_ANALYSIS_DIR))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from group_analysis.main.gvs_effects_analysis import (  # noqa: E402
    ACTIVE_CONDITION_CODES,
    DEFAULT_BY_GVS_DIR,
    DEFAULT_ROI_IMG,
    DEFAULT_ROI_SUMMARY,
    DEFAULT_SELECTED_VOXELS_PATH,
    SHAM_CONDITION_CODE,
    build_roi_membership,
    ensure_dir,
    medication_from_session,
)


DEFAULT_OUTPUT_ROOT = _REPO_ROOT / "results" / "connectivity" / "GVS_effects" / "gvs_similarity"
DEFAULT_TARGET_TRIALS = 20
DEFAULT_N_PERMUTATIONS = 2000
DEFAULT_RANDOM_SEED = 42
DEFAULT_WASSERSTEIN_PERMUTATION_BINS = 128
MIN_ROI_VOXELS = 5
CONDITION_FILE_RE = re.compile(r"^selected_beta_trials_(gvs-\d+)\.npy$")
PRIMARY_METRIC = "flat_pearson_r"
SECONDARY_METRICS = (
    "flat_cosine_similarity",
    "frobenius_norm_diff",
    "zscore_rmse",
    "eigenvalue_profile_correlation",
    "eigenvalue_profile_l2_distance",
)
ALL_METRICS = (PRIMARY_METRIC, *SECONDARY_METRICS)
ALL_CONDITION_CODES = [SHAM_CONDITION_CODE, *ACTIVE_CONDITION_CODES]
HEATMAP_CMAP = "jet"
METRIC_SPECS: dict[str, dict[str, Any]] = {
    "flat_pearson_r": {
        "label": "Flattened Pearson r",
        "vmin": -1.0,
        "vmax": 1.0,
        "higher_is_more_similar": True,
    },
    "flat_cosine_similarity": {
        "label": "Flattened cosine similarity",
        "vmin": -1.0,
        "vmax": 1.0,
        "higher_is_more_similar": True,
    },
    "frobenius_norm_diff": {
        "label": "Frobenius norm difference",
        "vmin": 0.0,
        "vmax": None,
        "higher_is_more_similar": False,
    },
    "zscore_rmse": {
        "label": "Z-scored RMSE",
        "vmin": 0.0,
        "vmax": None,
        "higher_is_more_similar": False,
    },
    "eigenvalue_profile_correlation": {
        "label": "Eigenvalue-profile correlation",
        "vmin": -1.0,
        "vmax": 1.0,
        "higher_is_more_similar": True,
    },
    "eigenvalue_profile_l2_distance": {
        "label": "Eigenvalue-profile L2 distance",
        "vmin": 0.0,
        "vmax": None,
        "higher_is_more_similar": False,
    },
}
DEFAULT_TRIAL_ANALYSIS_KEY = "cosine_similarity"
TRIAL_ANALYSIS_SPECS: dict[str, dict[str, Any]] = {
    "cosine_similarity": {
        "folder_name": "trial_distance_to_on_sham",
        "plot_filename_suffix": "trial_distance_to_on_sham_boxplots.png",
        "display_name": "normalized dot-product similarity",
        "title_template": "{subject}: OFF trial similarity to ON sham trials (normalized dot product)",
        "pairwise_metric": "cosine_similarity",
        "closest_reducer": "max",
        "include_legacy_distance_columns": True,
        "summary_metrics": (
            {
                "column": "mean_cosine_similarity_to_on_sham",
                "label": "Mean cosine similarity to ON sham",
                "plot_ylabel": "Mean similarity to ON sham",
                "reducer": "mean",
                "alternative": "less",
            },
            {
                "column": "max_cosine_similarity_to_on_sham",
                "label": "Max cosine similarity to ON sham",
                "plot_ylabel": "Max similarity to ON sham",
                "reducer": "max",
                "alternative": "less",
            },
        ),
    },
    "correlation_similarity": {
        "folder_name": "trial_correlation_to_on_sham",
        "plot_filename_suffix": "trial_correlation_to_on_sham_boxplots.png",
        "display_name": "trial-wise Pearson correlation",
        "title_template": "{subject}: OFF trial correlation similarity to ON sham trials",
        "pairwise_metric": "correlation_similarity",
        "closest_reducer": "max",
        "summary_metrics": (
            {
                "column": "mean_correlation_similarity_to_on_sham",
                "label": "Mean trial correlation to ON sham",
                "plot_ylabel": "Mean similarity to ON sham",
                "reducer": "mean",
                "alternative": "less",
            },
            {
                "column": "max_correlation_similarity_to_on_sham",
                "label": "Max trial correlation to ON sham",
                "plot_ylabel": "Max similarity to ON sham",
                "reducer": "max",
                "alternative": "less",
            },
        ),
    },
    "frobenius_distance": {
        "folder_name": "trial_frobenius_distance_to_on_sham",
        "plot_filename_suffix": "trial_frobenius_distance_to_on_sham_boxplots.png",
        "display_name": "trial-wise Frobenius distance",
        "title_template": "{subject}: OFF trial Frobenius distance to ON sham trials",
        "pairwise_metric": "frobenius_distance",
        "closest_reducer": "min",
        "summary_metrics": (
            {
                "column": "mean_frobenius_distance_to_on_sham",
                "label": "Mean trial Frobenius distance to ON sham",
                "plot_ylabel": "Mean distance to ON sham",
                "reducer": "mean",
                "alternative": "greater",
            },
            {
                "column": "min_frobenius_distance_to_on_sham",
                "label": "Min trial Frobenius distance to ON sham",
                "plot_ylabel": "Min distance to ON sham",
                "reducer": "min",
                "alternative": "greater",
            },
        ),
    },
}


def _trial_summary_metric_labels(analysis_spec: dict[str, Any]) -> dict[str, str]:
    return {
        str(metric_spec["column"]): str(metric_spec["label"])
        for metric_spec in analysis_spec["summary_metrics"]
    }


def _trial_summary_columns(analysis_spec: dict[str, Any]) -> list[str]:
    columns = [str(metric_spec["column"]) for metric_spec in analysis_spec["summary_metrics"]]
    if bool(analysis_spec.get("include_legacy_distance_columns")):
        columns.extend(["mean_cosine_distance_to_on_sham", "min_cosine_distance_to_on_sham"])
    return columns


SHAM_REFERENCE_TRIAL_ANALYSIS_SPEC: dict[str, Any] = {
    "display_name": "normalized dot-product similarity to sham reference",
    "title_template": "{subject}: {target_medication} trial similarity to {reference_group_label}",
    "x_label_template": "{target_medication} condition",
    "pairwise_metric": "cosine_similarity",
    "closest_reducer": "max",
    "summary_metrics": (
        {
            "column": "mean_cosine_similarity_to_reference",
            "label": "Mean trial similarity to sham reference",
            "plot_ylabel": "Mean similarity to sham reference",
            "reducer": "mean",
            "alternative": "less",
        },
        {
            "column": "max_cosine_similarity_to_reference",
            "label": "Max trial similarity to sham reference",
            "plot_ylabel": "Max similarity to sham reference",
            "reducer": "max",
            "alternative": "less",
        },
    ),
}
FLATTENED_BETA_DISTRIBUTION_METRIC = "flattened_voxel_trial_beta_distribution"
FLATTENED_BETA_DISTRIBUTION_LABEL = "Flattened voxel-by-trial beta distribution"
SHAM_ROI_MEDICATION_TTEST_ANALYSIS = "sham_off_vs_on_roi_ttest"
SHAM_ROI_MEDICATION_TTEST_LABEL = "Sham OFF vs sham ON ROI trial-vector Welch t-test"
ROI_REFERENCE_DELTA_SPECS: tuple[dict[str, Any], ...] = (
    {
        "comparison_kind": "off_condition_minus_sham_on_roi_delta",
        "target_session": 1,
        "reference_session": 2,
        "title": "Group mean ROI beta delta: OFF condition - sham ON",
        "colorbar_label": "Mean beta delta (OFF condition - sham ON)",
        "file_prefix": "off_condition_minus_sham_on_roi_mean_delta",
    },
    {
        "comparison_kind": "off_condition_minus_sham_off_roi_delta",
        "target_session": 1,
        "reference_session": 1,
        "title": "Group mean ROI beta delta: OFF condition - sham OFF",
        "colorbar_label": "Mean beta delta (OFF condition - sham OFF)",
        "file_prefix": "off_condition_minus_sham_off_roi_mean_delta",
    },
    {
        "comparison_kind": "on_condition_minus_sham_on_roi_delta",
        "target_session": 2,
        "reference_session": 2,
        "title": "Group mean ROI beta delta: ON condition - sham ON",
        "colorbar_label": "Mean beta delta (ON condition - sham ON)",
        "file_prefix": "on_condition_minus_sham_on_roi_mean_delta",
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build subject/session ROI beta matrices from GVS beta splits and "
            "measure sham-reference similarity within and across medication states."
        )
    )
    parser.add_argument("--by-gvs-dir", type=Path, default=DEFAULT_BY_GVS_DIR)
    parser.add_argument("--selected-voxels-path", type=Path, default=DEFAULT_SELECTED_VOXELS_PATH)
    parser.add_argument("--roi-img", type=Path, default=DEFAULT_ROI_IMG)
    parser.add_argument("--roi-summary", type=Path, default=DEFAULT_ROI_SUMMARY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--min-roi-voxels", type=int, default=MIN_ROI_VOXELS)
    parser.add_argument(
        "--target-trials",
        type=int,
        default=DEFAULT_TARGET_TRIALS,
        help="Fixed trial-grid width used when resampling ROI rows before comparison.",
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=DEFAULT_N_PERMUTATIONS,
        help="Monte Carlo permutations used for each subject-level sham comparison.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Random seed used for subject-level permutation tests.",
    )
    return parser.parse_args()


def _condition_display_name(code: str) -> str:
    if code == SHAM_CONDITION_CODE:
        return "sham"
    match = re.match(r"^gvs-(\d+)$", code)
    if match is None:
        return code
    condition_number = int(match.group(1))
    if condition_number >= 2:
        return f"GVS{condition_number - 1}"
    return code


def _reference_group_label(session: int, condition_code: str) -> str:
    return f"{medication_from_session(session)} {_condition_display_name(condition_code)}"


def _compute_roi_beta_matrix(beta: np.ndarray, roi_members: list[np.ndarray]) -> np.ndarray:
    beta_array = np.asarray(beta, dtype=np.float64)
    roi_beta = np.full((len(roi_members), beta_array.shape[1]), np.nan, dtype=np.float64)
    if beta_array.shape[1] == 0:
        return roi_beta
    for roi_index, members in enumerate(roi_members):
        roi_beta[roi_index] = np.nanmean(beta_array[members, :], axis=0)
    return roi_beta


def _resample_row(values: np.ndarray, target_trials: int) -> np.ndarray:
    row = np.asarray(values, dtype=np.float64).ravel()
    out = np.full(int(target_trials), np.nan, dtype=np.float64)
    if row.size == 0:
        return out
    finite = np.isfinite(row)
    if not np.any(finite):
        return out
    valid_values = row[finite]
    if valid_values.size == 1:
        out[:] = float(valid_values[0])
        return out
    x_src = np.linspace(0.0, 1.0, row.size, dtype=np.float64)[finite]
    x_dst = np.linspace(0.0, 1.0, int(target_trials), dtype=np.float64)
    out[:] = np.interp(x_dst, x_src, valid_values)
    return out


def _resample_matrix(matrix: np.ndarray, target_trials: int) -> np.ndarray:
    array = np.asarray(matrix, dtype=np.float64)
    return np.vstack([_resample_row(row, target_trials=target_trials) for row in array])


def _zscore_rows(matrix: np.ndarray) -> np.ndarray:
    array = np.asarray(matrix, dtype=np.float64)
    out = np.full_like(array, np.nan, dtype=np.float64)
    for row_index, row in enumerate(array):
        finite = np.isfinite(row)
        if not np.any(finite):
            continue
        values = row[finite]
        mean_value = float(np.mean(values))
        std_value = float(np.std(values, ddof=0))
        if not np.isfinite(std_value) or np.isclose(std_value, 0.0):
            out[row_index, finite] = 0.0
            continue
        out[row_index, finite] = (values - mean_value) / std_value
    return out


def _pearson_similarity(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3 or y.size < 3:
        return float("nan")
    x_std = float(np.std(x, ddof=0))
    y_std = float(np.std(y, ddof=0))
    if np.isclose(x_std, 0.0) or np.isclose(y_std, 0.0):
        return float("nan")
    value = float(np.corrcoef(x, y)[0, 1])
    return value if np.isfinite(value) else float("nan")


def _cosine_similarity(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or y.size == 0:
        return float("nan")
    x_norm = float(np.linalg.norm(x))
    y_norm = float(np.linalg.norm(y))
    if np.isclose(x_norm, 0.0) or np.isclose(y_norm, 0.0):
        return float("nan")
    value = float(np.dot(x, y) / (x_norm * y_norm))
    return value if np.isfinite(value) else float("nan")


def _row_gram_eigenvalues(matrix: np.ndarray) -> np.ndarray:
    array = np.asarray(matrix, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got {array.shape}.")
    finite = np.where(np.isfinite(array), array, 0.0)
    scale = float(max(1, finite.shape[1]))
    gram = (finite @ finite.T) / scale
    eigvals = np.linalg.eigvalsh(gram)
    eigvals = np.sort(np.asarray(eigvals, dtype=np.float64))[::-1]
    eigvals[~np.isfinite(eigvals)] = np.nan
    return eigvals


def compute_matrix_similarity(
    target_matrix: np.ndarray,
    reference_matrix: np.ndarray,
    target_trials: int,
) -> dict[str, float | int]:
    target_resampled = _resample_matrix(target_matrix, target_trials=target_trials)
    reference_resampled = _resample_matrix(reference_matrix, target_trials=target_trials)
    target_norm = _zscore_rows(target_resampled)
    reference_norm = _zscore_rows(reference_resampled)

    valid = np.isfinite(target_norm) & np.isfinite(reference_norm)
    x = target_norm[valid]
    y = reference_norm[valid]
    raw_valid = np.isfinite(target_resampled) & np.isfinite(reference_resampled)
    x_raw = target_resampled[raw_valid]
    y_raw = reference_resampled[raw_valid]
    target_eig = _row_gram_eigenvalues(target_norm)
    reference_eig = _row_gram_eigenvalues(reference_norm)
    eig_valid = np.isfinite(target_eig) & np.isfinite(reference_eig)
    eig_x = target_eig[eig_valid]
    eig_y = reference_eig[eig_valid]
    if x.size == 0 or y.size == 0:
        return {
            "n_overlap_values": 0,
            "flat_pearson_r": float("nan"),
            "flat_cosine_similarity": float("nan"),
            "frobenius_norm_diff": float("nan"),
            "zscore_rmse": float("nan"),
            "eigenvalue_profile_correlation": float("nan"),
            "eigenvalue_profile_l2_distance": float("nan"),
        }

    diff = x - y
    return {
        "n_overlap_values": int(x.size),
        "flat_pearson_r": _pearson_similarity(x, y),
        "flat_cosine_similarity": _cosine_similarity(x_raw, y_raw),
        "frobenius_norm_diff": float(np.linalg.norm(target_norm - reference_norm, ord="fro")),
        "zscore_rmse": float(np.sqrt(np.mean(diff**2))),
        "eigenvalue_profile_correlation": _pearson_similarity(eig_x, eig_y),
        "eigenvalue_profile_l2_distance": float(np.linalg.norm(eig_x - eig_y, ord=2))
        if eig_x.size and eig_y.size
        else float("nan"),
    }


def _permute_two_group_columns(
    target_matrix: np.ndarray,
    sham_matrix: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    target_array = np.asarray(target_matrix, dtype=np.float64)
    sham_array = np.asarray(sham_matrix, dtype=np.float64)
    if target_array.ndim != 2 or sham_array.ndim != 2:
        raise ValueError(
            f"Expected 2D matrices for permutation, got {target_array.shape} and {sham_array.shape}."
        )
    if target_array.shape[0] != sham_array.shape[0]:
        raise ValueError(
            "Target and sham matrices must have the same number of ROI rows "
            f"for permutation, got {target_array.shape[0]} and {sham_array.shape[0]}."
        )

    pooled = np.concatenate([target_array, sham_array], axis=1)
    n_target = int(target_array.shape[1])
    n_total = int(pooled.shape[1])
    if n_target <= 0 or n_target >= n_total:
        return pooled[:, :n_target].copy(), pooled[:, n_target:].copy()

    target_mask = np.zeros(n_total, dtype=bool)
    target_mask[rng.choice(n_total, size=n_target, replace=False)] = True
    return pooled[:, target_mask], pooled[:, ~target_mask]


def _change_score_from_metric(
    metric_name: str,
    observed_value: float,
    baseline_value: float | None = None,
) -> float:
    if not np.isfinite(observed_value):
        return float("nan")
    higher_is_more_similar = bool(METRIC_SPECS[metric_name]["higher_is_more_similar"])
    if baseline_value is None:
        return float(-observed_value) if higher_is_more_similar else float(observed_value)
    if not np.isfinite(baseline_value):
        return float("nan")
    return (
        float(baseline_value - observed_value)
        if higher_is_more_similar
        else float(observed_value - baseline_value)
    )


def _empirical_upper_tail_pvalue(null_values: np.ndarray, observed_value: float) -> float:
    finite_null = np.asarray(null_values, dtype=np.float64)
    finite_null = finite_null[np.isfinite(finite_null)]
    if finite_null.size == 0 or not np.isfinite(observed_value):
        return float("nan")
    return float((1 + np.count_nonzero(finite_null >= float(observed_value))) / (finite_null.size + 1))


def _wasserstein_distance_from_counts(
    counts_a: np.ndarray,
    counts_b: np.ndarray,
    support: np.ndarray,
) -> np.ndarray | float:
    weights_a = np.asarray(counts_a, dtype=np.float64)
    weights_b = np.asarray(counts_b, dtype=np.float64)
    support_array = np.asarray(support, dtype=np.float64)
    if weights_a.shape[-1] != support_array.size or weights_b.shape[-1] != support_array.size:
        raise ValueError(
            "Histogram count arrays must align with the support size, got "
            f"{weights_a.shape}, {weights_b.shape}, and support length {support_array.size}."
        )

    original_ndim = weights_a.ndim
    weights_a_2d = np.atleast_2d(weights_a)
    weights_b_2d = np.atleast_2d(weights_b)
    if support_array.size <= 1:
        distances = np.zeros(weights_a_2d.shape[0], dtype=np.float64)
    else:
        weights_a_sum = np.sum(weights_a_2d, axis=1, keepdims=True)
        weights_b_sum = np.sum(weights_b_2d, axis=1, keepdims=True)
        a_cdf = np.cumsum(weights_a_2d / weights_a_sum, axis=1)
        b_cdf = np.cumsum(weights_b_2d / weights_b_sum, axis=1)
        distances = np.sum(
            np.abs(a_cdf[:, :-1] - b_cdf[:, :-1]) * np.diff(support_array)[None, :],
            axis=1,
        )
    if original_ndim == 1 and weights_b.ndim == 1:
        return float(distances[0])
    return distances


def _wasserstein_permutation_test(
    target_values: np.ndarray,
    reference_values: np.ndarray,
    *,
    n_permutations: int,
    rng: np.random.Generator,
    n_bins: int = DEFAULT_WASSERSTEIN_PERMUTATION_BINS,
) -> dict[str, float | int]:
    finite_target = np.asarray(target_values, dtype=np.float64)
    finite_target = finite_target[np.isfinite(finite_target)]
    finite_reference = np.asarray(reference_values, dtype=np.float64)
    finite_reference = finite_reference[np.isfinite(finite_reference)]
    observed_distance = float("nan")
    if finite_target.size >= 1 and finite_reference.size >= 1:
        observed_distance = float(stats.wasserstein_distance(finite_target, finite_reference))
    if finite_target.size == 0 or finite_reference.size == 0:
        return {
            "wasserstein_distance": observed_distance,
            "wasserstein_p_value": float("nan"),
            "n_permutations_requested": int(n_permutations),
            "n_permutations_effective": 0,
            "wasserstein_permutation_bins": 0,
        }

    pooled = np.concatenate([finite_target, finite_reference], axis=0)
    pooled_min = float(np.min(pooled))
    pooled_max = float(np.max(pooled))
    if not np.isfinite(pooled_min) or not np.isfinite(pooled_max):
        return {
            "wasserstein_distance": observed_distance,
            "wasserstein_p_value": float("nan"),
            "n_permutations_requested": int(n_permutations),
            "n_permutations_effective": 0,
            "wasserstein_permutation_bins": 0,
        }
    if pooled_min == pooled_max:
        return {
            "wasserstein_distance": 0.0,
            "wasserstein_p_value": 1.0,
            "n_permutations_requested": int(n_permutations),
            "n_permutations_effective": int(n_permutations),
            "wasserstein_permutation_bins": 1,
        }

    n_bins_effective = int(max(2, min(int(n_bins), pooled.size)))
    bin_edges = np.linspace(pooled_min, pooled_max, n_bins_effective + 1, dtype=np.float64)
    pooled_counts = np.histogram(pooled, bins=bin_edges)[0].astype(np.int64, copy=False)
    nonzero_mask = pooled_counts > 0
    pooled_counts = pooled_counts[nonzero_mask]
    support = (0.5 * (bin_edges[:-1] + bin_edges[1:]))[nonzero_mask]
    if pooled_counts.size <= 1:
        return {
            "wasserstein_distance": observed_distance,
            "wasserstein_p_value": 1.0,
            "n_permutations_requested": int(n_permutations),
            "n_permutations_effective": int(n_permutations),
            "wasserstein_permutation_bins": int(pooled_counts.size),
        }

    perm_target_counts = np.asarray(
        rng.multivariate_hypergeometric(
            pooled_counts,
            int(finite_target.size),
            size=int(n_permutations),
        ),
        dtype=np.int64,
    )
    perm_reference_counts = pooled_counts[None, :] - perm_target_counts
    perm_distances = np.asarray(
        _wasserstein_distance_from_counts(perm_target_counts, perm_reference_counts, support),
        dtype=np.float64,
    )
    return {
        "wasserstein_distance": observed_distance,
        "wasserstein_p_value": _empirical_upper_tail_pvalue(perm_distances, observed_distance),
        "n_permutations_requested": int(n_permutations),
        "n_permutations_effective": int(perm_distances.size),
        "wasserstein_permutation_bins": int(pooled_counts.size),
    }


def _add_groupwise_fdr(
    df: pd.DataFrame,
    *,
    p_value_column: str,
    group_columns: list[str],
) -> pd.DataFrame:
    if df.empty:
        out = df.copy()
        out["q_value_fdr"] = np.array([], dtype=np.float64)
        out["significant_fdr"] = np.array([], dtype=bool)
        return out

    sort_candidates = [
        *group_columns,
        "subject",
        "target_medication",
        "target_condition_code",
        "condition_code",
        "metric_name",
        "summary_metric",
    ]
    sort_columns = [column for column in dict.fromkeys(sort_candidates) if column in df.columns]
    out = df.sort_values(sort_columns).reset_index(drop=True)
    q_values = np.full(out.shape[0], np.nan, dtype=np.float64)
    significant = np.zeros(out.shape[0], dtype=bool)
    for _, group_df in out.groupby(group_columns, dropna=False, observed=False, sort=False):
        idx = group_df.index.to_numpy(dtype=np.int64)
        p_values = group_df[p_value_column].to_numpy(dtype=np.float64)
        finite_mask = np.isfinite(p_values)
        if not np.any(finite_mask):
            continue
        sig_valid, q_valid = fdrcorrection(p_values[finite_mask], alpha=0.05)
        q_values[idx[finite_mask]] = q_valid
        significant[idx[finite_mask]] = sig_valid
    out["q_value_fdr"] = q_values
    out["significant_fdr"] = significant
    return out


def compute_within_medication_permutation_stats(
    matrices: dict[tuple[str, int, str], np.ndarray],
    target_trials: int,
    n_permutations: int,
    random_seed: int,
) -> pd.DataFrame:
    if not matrices:
        return pd.DataFrame()
    if int(n_permutations) <= 0:
        raise ValueError(f"--n-permutations must be positive, got {n_permutations}.")

    rng = np.random.default_rng(int(random_seed))
    rows: list[dict[str, Any]] = []
    subjects = sorted({subject for subject, _, _ in matrices})
    for subject in subjects:
        for session in (1, 2):
            sham_matrix = matrices.get((subject, session, SHAM_CONDITION_CODE))
            if sham_matrix is None:
                continue
            for condition_code in ACTIVE_CONDITION_CODES:
                target_matrix = matrices.get((subject, session, condition_code))
                if target_matrix is None:
                    continue

                observed_metrics = compute_matrix_similarity(
                    target_matrix=target_matrix,
                    reference_matrix=sham_matrix,
                    target_trials=target_trials,
                )
                null_metrics: dict[str, list[float]] = {metric_name: [] for metric_name in ALL_METRICS}
                for _ in range(int(n_permutations)):
                    perm_target, perm_sham = _permute_two_group_columns(
                        target_matrix=target_matrix,
                        sham_matrix=sham_matrix,
                        rng=rng,
                    )
                    perm_metrics = compute_matrix_similarity(
                        target_matrix=perm_target,
                        reference_matrix=perm_sham,
                        target_trials=target_trials,
                    )
                    for metric_name in ALL_METRICS:
                        null_metrics[metric_name].append(float(perm_metrics[metric_name]))

                for metric_name in ALL_METRICS:
                    observed_value = float(observed_metrics[metric_name])
                    null_metric_values = np.asarray(null_metrics[metric_name], dtype=np.float64)
                    null_metric_values = null_metric_values[np.isfinite(null_metric_values)]
                    observed_change_score = _change_score_from_metric(metric_name, observed_value)
                    null_change_scores = np.asarray(
                        [_change_score_from_metric(metric_name, value) for value in null_metric_values],
                        dtype=np.float64,
                    )
                    null_change_scores = null_change_scores[np.isfinite(null_change_scores)]

                    rows.append(
                        {
                            "comparison_kind": "within_med_sham_reference",
                            "subject": subject,
                            "target_session": int(session),
                            "target_medication": medication_from_session(session),
                            "target_condition_code": condition_code,
                            "target_condition_label": _condition_display_name(condition_code),
                            "reference_session": int(session),
                            "reference_medication": medication_from_session(session),
                            "reference_condition_code": SHAM_CONDITION_CODE,
                            "reference_condition_label": _condition_display_name(SHAM_CONDITION_CODE),
                            "metric_name": metric_name,
                            "metric_label": str(METRIC_SPECS[metric_name]["label"]),
                            "higher_is_more_similar": bool(METRIC_SPECS[metric_name]["higher_is_more_similar"]),
                            "observed_metric_value": observed_value,
                            "observed_change_score": observed_change_score,
                            "null_mean_metric": (
                                float(np.mean(null_metric_values)) if null_metric_values.size else float("nan")
                            ),
                            "null_std_metric": (
                                float(np.std(null_metric_values, ddof=1))
                                if null_metric_values.size >= 2
                                else float("nan")
                            ),
                            "null_mean_change_score": (
                                float(np.mean(null_change_scores)) if null_change_scores.size else float("nan")
                            ),
                            "null_std_change_score": (
                                float(np.std(null_change_scores, ddof=1))
                                if null_change_scores.size >= 2
                                else float("nan")
                            ),
                            "empirical_p_value_change_one_sided": _empirical_upper_tail_pvalue(
                                null_change_scores,
                                observed_change_score,
                            ),
                            "n_permutations_requested": int(n_permutations),
                            "n_permutations_effective": int(null_change_scores.size),
                            "n_trials_target": int(target_matrix.shape[1]),
                            "n_trials_reference": int(sham_matrix.shape[1]),
                        }
                    )

    if not rows:
        return pd.DataFrame()

    stats_df = pd.DataFrame(rows)
    return _add_groupwise_fdr(
        stats_df,
        p_value_column="empirical_p_value_change_one_sided",
        group_columns=["comparison_kind", "subject", "target_medication", "metric_name"],
    )


def compute_off_to_on_permutation_stats(
    matrices: dict[tuple[str, int, str], np.ndarray],
    target_trials: int,
    n_permutations: int,
    random_seed: int,
) -> pd.DataFrame:
    if not matrices:
        return pd.DataFrame()
    if int(n_permutations) <= 0:
        raise ValueError(f"--n-permutations must be positive, got {n_permutations}.")

    rng = np.random.default_rng(int(random_seed) + 1)
    rows: list[dict[str, Any]] = []
    subjects = sorted({subject for subject, _, _ in matrices})
    for subject in subjects:
        off_sham = matrices.get((subject, 1, SHAM_CONDITION_CODE))
        on_sham = matrices.get((subject, 2, SHAM_CONDITION_CODE))
        if off_sham is None or on_sham is None:
            continue

        sham_reference_metrics = compute_matrix_similarity(
            target_matrix=off_sham,
            reference_matrix=on_sham,
            target_trials=target_trials,
        )
        for condition_code in ACTIVE_CONDITION_CODES:
            target_matrix = matrices.get((subject, 1, condition_code))
            if target_matrix is None:
                continue

            observed_metrics = compute_matrix_similarity(
                target_matrix=target_matrix,
                reference_matrix=on_sham,
                target_trials=target_trials,
            )
            null_target_metrics: dict[str, list[float]] = {metric_name: [] for metric_name in ALL_METRICS}
            null_sham_metrics: dict[str, list[float]] = {metric_name: [] for metric_name in ALL_METRICS}
            for _ in range(int(n_permutations)):
                perm_target, perm_off_sham = _permute_two_group_columns(
                    target_matrix=target_matrix,
                    sham_matrix=off_sham,
                    rng=rng,
                )
                perm_target_metrics = compute_matrix_similarity(
                    target_matrix=perm_target,
                    reference_matrix=on_sham,
                    target_trials=target_trials,
                )
                perm_sham_metrics = compute_matrix_similarity(
                    target_matrix=perm_off_sham,
                    reference_matrix=on_sham,
                    target_trials=target_trials,
                )
                for metric_name in ALL_METRICS:
                    null_target_metrics[metric_name].append(float(perm_target_metrics[metric_name]))
                    null_sham_metrics[metric_name].append(float(perm_sham_metrics[metric_name]))

            for metric_name in ALL_METRICS:
                observed_value = float(observed_metrics[metric_name])
                sham_baseline_value = float(sham_reference_metrics[metric_name])
                null_target_values = np.asarray(null_target_metrics[metric_name], dtype=np.float64)
                null_sham_values = np.asarray(null_sham_metrics[metric_name], dtype=np.float64)
                finite_mask = np.isfinite(null_target_values) & np.isfinite(null_sham_values)
                null_target_values = null_target_values[finite_mask]
                null_sham_values = null_sham_values[finite_mask]
                observed_effect_score = _change_score_from_metric(
                    metric_name,
                    observed_value,
                    baseline_value=sham_baseline_value,
                )
                null_effect_scores = np.asarray(
                    [
                        _change_score_from_metric(
                            metric_name,
                            null_target_values[index],
                            baseline_value=null_sham_values[index],
                        )
                        for index in range(null_target_values.size)
                    ],
                    dtype=np.float64,
                )
                null_effect_scores = null_effect_scores[np.isfinite(null_effect_scores)]

                rows.append(
                    {
                        "comparison_kind": "off_to_on_sham_reference",
                        "subject": subject,
                        "target_session": 1,
                        "target_medication": "OFF",
                        "target_condition_code": condition_code,
                        "target_condition_label": _condition_display_name(condition_code),
                        "reference_session": 2,
                        "reference_medication": "ON",
                        "reference_condition_code": SHAM_CONDITION_CODE,
                        "reference_condition_label": _condition_display_name(SHAM_CONDITION_CODE),
                        "baseline_condition_code": SHAM_CONDITION_CODE,
                        "baseline_condition_label": _condition_display_name(SHAM_CONDITION_CODE),
                        "metric_name": metric_name,
                        "metric_label": str(METRIC_SPECS[metric_name]["label"]),
                        "higher_is_more_similar": bool(METRIC_SPECS[metric_name]["higher_is_more_similar"]),
                        "observed_metric_value": observed_value,
                        "baseline_sham_metric_value": sham_baseline_value,
                        "observed_effect_score": observed_effect_score,
                        "null_mean_target_metric": (
                            float(np.mean(null_target_values)) if null_target_values.size else float("nan")
                        ),
                        "null_mean_baseline_metric": (
                            float(np.mean(null_sham_values)) if null_sham_values.size else float("nan")
                        ),
                        "null_mean_effect_score": (
                            float(np.mean(null_effect_scores)) if null_effect_scores.size else float("nan")
                        ),
                        "null_std_effect_score": (
                            float(np.std(null_effect_scores, ddof=1))
                            if null_effect_scores.size >= 2
                            else float("nan")
                        ),
                        "empirical_p_value_effect_one_sided": _empirical_upper_tail_pvalue(
                            null_effect_scores,
                            observed_effect_score,
                        ),
                        "n_permutations_requested": int(n_permutations),
                        "n_permutations_effective": int(null_effect_scores.size),
                        "n_trials_target": int(target_matrix.shape[1]),
                        "n_trials_off_sham": int(off_sham.shape[1]),
                        "n_trials_on_sham": int(on_sham.shape[1]),
                    }
                )

    if not rows:
        return pd.DataFrame()

    stats_df = pd.DataFrame(rows)
    return _add_groupwise_fdr(
        stats_df,
        p_value_column="empirical_p_value_effect_one_sided",
        group_columns=["comparison_kind", "subject", "metric_name"],
    )


def load_roi_beta_matrices(
    by_gvs_dir: Path,
    roi_members: list[np.ndarray],
    matrix_dir: Path,
) -> tuple[dict[tuple[str, int, str], np.ndarray], pd.DataFrame]:
    matrices: dict[tuple[str, int, str], np.ndarray] = {}
    inventory_rows: list[dict[str, Any]] = []

    for subject_dir in sorted(path for path in by_gvs_dir.iterdir() if path.is_dir()):
        subject = subject_dir.name
        for session_dir in sorted(path for path in subject_dir.iterdir() if path.is_dir()):
            session = int(session_dir.name.split("-")[-1])
            medication = medication_from_session(session)
            output_session_dir = ensure_dir(matrix_dir / subject / session_dir.name)
            for beta_path in sorted(session_dir.glob("selected_beta_trials_gvs-*.npy")):
                match = CONDITION_FILE_RE.match(beta_path.name)
                if match is None:
                    continue
                condition_code = match.group(1)
                beta = np.asarray(np.load(beta_path), dtype=np.float64)
                roi_beta = _compute_roi_beta_matrix(beta=beta, roi_members=roi_members)
                matrices[(subject, session, condition_code)] = roi_beta

                saved_matrix_path = output_session_dir / f"{condition_code}_roi_beta_matrix.npy"
                np.save(saved_matrix_path, roi_beta.astype(np.float32, copy=False))
                inventory_rows.append(
                    {
                        "subject": subject,
                        "session": int(session),
                        "medication": medication,
                        "condition_code": condition_code,
                        "condition_label": _condition_display_name(condition_code),
                        "n_rois": int(roi_beta.shape[0]),
                        "n_trials": int(roi_beta.shape[1]),
                        "matrix_path": str(saved_matrix_path),
                        "source_beta_path": str(beta_path),
                    }
                )

    inventory_df = pd.DataFrame(inventory_rows).sort_values(
        ["subject", "session", "condition_code"]
    ).reset_index(drop=True)
    return matrices, inventory_df


def load_beta_value_distributions(
    by_gvs_dir: Path,
) -> dict[tuple[str, int, str], np.ndarray]:
    distributions: dict[tuple[str, int, str], np.ndarray] = {}
    for subject_dir in sorted(path for path in by_gvs_dir.iterdir() if path.is_dir()):
        subject = subject_dir.name
        for session_dir in sorted(path for path in subject_dir.iterdir() if path.is_dir()):
            session = int(session_dir.name.split("-")[-1])
            for beta_path in sorted(session_dir.glob("selected_beta_trials_gvs-*.npy")):
                match = CONDITION_FILE_RE.match(beta_path.name)
                if match is None:
                    continue
                condition_code = match.group(1)
                beta = np.asarray(np.load(beta_path), dtype=np.float64)
                distributions[(subject, session, condition_code)] = beta[np.isfinite(beta)].ravel()
    return distributions


def _distribution_summary(values: np.ndarray) -> dict[str, float | int]:
    finite_values = np.asarray(values, dtype=np.float64)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return {
            "n_values": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "q25": float("nan"),
            "q75": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    return {
        "n_values": int(finite_values.size),
        "mean": float(np.mean(finite_values)),
        "median": float(np.median(finite_values)),
        "std": float(np.std(finite_values, ddof=1)) if finite_values.size >= 2 else float("nan"),
        "q25": float(np.quantile(finite_values, 0.25)),
        "q75": float(np.quantile(finite_values, 0.75)),
        "min": float(np.min(finite_values)),
        "max": float(np.max(finite_values)),
    }


def _distribution_summary_row(
    *,
    subject: str,
    target_session: int,
    target_condition_code: str,
    target_values: np.ndarray,
    reference_session: int,
    reference_condition_code: str,
    reference_values: np.ndarray,
    comparison_kind: str,
) -> dict[str, Any]:
    target_summary = _distribution_summary(target_values)
    reference_summary = _distribution_summary(reference_values)
    row: dict[str, Any] = {
        "comparison_kind": comparison_kind,
        "subject": subject,
        "target_session": int(target_session),
        "target_medication": medication_from_session(target_session),
        "target_condition_code": target_condition_code,
        "target_condition_label": _condition_display_name(target_condition_code),
        "reference_session": int(reference_session),
        "reference_medication": medication_from_session(reference_session),
        "reference_condition_code": reference_condition_code,
        "reference_condition_label": _condition_display_name(reference_condition_code),
        "reference_group_label": _reference_group_label(reference_session, reference_condition_code),
    }
    for key, value in target_summary.items():
        row[f"target_{key}"] = value
    for key, value in reference_summary.items():
        row[f"reference_{key}"] = value
    return row


def build_within_medication_distribution_summary(
    distributions: dict[tuple[str, int, str], np.ndarray],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    subjects = sorted({subject for subject, _, _ in distributions})
    for subject in subjects:
        for session in (1, 2):
            sham_values = distributions.get((subject, session, SHAM_CONDITION_CODE))
            if sham_values is None or sham_values.size == 0:
                continue
            for condition_code in ALL_CONDITION_CODES:
                target_values = distributions.get((subject, session, condition_code))
                if target_values is None or target_values.size == 0:
                    continue
                rows.append(
                    _distribution_summary_row(
                        subject=subject,
                        target_session=session,
                        target_condition_code=condition_code,
                        target_values=target_values,
                        reference_session=session,
                        reference_condition_code=SHAM_CONDITION_CODE,
                        reference_values=sham_values,
                        comparison_kind="within_med_sham_reference",
                    )
                )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["target_medication", "subject", "target_condition_code"]
    ).reset_index(drop=True)


def build_off_to_on_distribution_summary(
    distributions: dict[tuple[str, int, str], np.ndarray],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    subjects = sorted({subject for subject, _, _ in distributions})
    for subject in subjects:
        on_sham_values = distributions.get((subject, 2, SHAM_CONDITION_CODE))
        if on_sham_values is None or on_sham_values.size == 0:
            continue
        for condition_code in ALL_CONDITION_CODES:
            target_values = distributions.get((subject, 1, condition_code))
            if target_values is None or target_values.size == 0:
                continue
            rows.append(
                _distribution_summary_row(
                    subject=subject,
                    target_session=1,
                    target_condition_code=condition_code,
                    target_values=target_values,
                    reference_session=2,
                    reference_condition_code=SHAM_CONDITION_CODE,
                    reference_values=on_sham_values,
                    comparison_kind="off_to_on_sham_reference",
                )
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["subject", "target_condition_code"]).reset_index(drop=True)


def _distribution_test_row(
    *,
    subject: str,
    target_session: int,
    target_condition_code: str,
    target_values: np.ndarray,
    reference_session: int,
    reference_condition_code: str,
    reference_values: np.ndarray,
    comparison_kind: str,
    n_permutations: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    finite_target = np.asarray(target_values, dtype=np.float64)
    finite_target = finite_target[np.isfinite(finite_target)]
    finite_reference = np.asarray(reference_values, dtype=np.float64)
    finite_reference = finite_reference[np.isfinite(finite_reference)]
    wasserstein_result = _wasserstein_permutation_test(
        finite_target,
        finite_reference,
        n_permutations=n_permutations,
        rng=rng,
    )

    return {
        "comparison_kind": comparison_kind,
        "subject": subject,
        "target_session": int(target_session),
        "target_medication": medication_from_session(target_session),
        "target_condition_code": target_condition_code,
        "target_condition_label": _condition_display_name(target_condition_code),
        "reference_session": int(reference_session),
        "reference_medication": medication_from_session(reference_session),
        "reference_condition_code": reference_condition_code,
        "reference_condition_label": _condition_display_name(reference_condition_code),
        "reference_group_label": _reference_group_label(reference_session, reference_condition_code),
        "summary_metric": FLATTENED_BETA_DISTRIBUTION_METRIC,
        "summary_metric_label": FLATTENED_BETA_DISTRIBUTION_LABEL,
        "n_values_target": int(finite_target.size),
        "n_values_reference": int(finite_reference.size),
        "target_mean": float(np.mean(finite_target)) if finite_target.size else float("nan"),
        "target_median": float(np.median(finite_target)) if finite_target.size else float("nan"),
        "reference_mean": float(np.mean(finite_reference)) if finite_reference.size else float("nan"),
        "reference_median": float(np.median(finite_reference)) if finite_reference.size else float("nan"),
        **wasserstein_result,
    }


def compute_within_medication_distribution_stats(
    distributions: dict[tuple[str, int, str], np.ndarray],
    *,
    n_permutations: int,
    random_seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(int(random_seed))
    subjects = sorted({subject for subject, _, _ in distributions})
    for subject in subjects:
        for session in (1, 2):
            sham_values = distributions.get((subject, session, SHAM_CONDITION_CODE))
            if sham_values is None or sham_values.size == 0:
                continue
            for condition_code in ACTIVE_CONDITION_CODES:
                target_values = distributions.get((subject, session, condition_code))
                if target_values is None or target_values.size == 0:
                    continue
                rows.append(
                    _distribution_test_row(
                        subject=subject,
                        target_session=session,
                        target_condition_code=condition_code,
                        target_values=target_values,
                        reference_session=session,
                        reference_condition_code=SHAM_CONDITION_CODE,
                        reference_values=sham_values,
                        comparison_kind="within_med_sham_reference",
                        n_permutations=n_permutations,
                        rng=rng,
                    )
                )
    if not rows:
        return pd.DataFrame()
    stats_df = pd.DataFrame(rows)
    return _add_groupwise_fdr(
        stats_df,
        p_value_column="wasserstein_p_value",
        group_columns=["comparison_kind", "subject", "target_medication", "summary_metric"],
    )


def compute_off_to_on_distribution_stats(
    distributions: dict[tuple[str, int, str], np.ndarray],
    *,
    n_permutations: int,
    random_seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(int(random_seed) + 1)
    subjects = sorted({subject for subject, _, _ in distributions})
    for subject in subjects:
        on_sham_values = distributions.get((subject, 2, SHAM_CONDITION_CODE))
        if on_sham_values is None or on_sham_values.size == 0:
            continue
        for condition_code in ALL_CONDITION_CODES:
            target_values = distributions.get((subject, 1, condition_code))
            if target_values is None or target_values.size == 0:
                continue
            rows.append(
                _distribution_test_row(
                    subject=subject,
                    target_session=1,
                    target_condition_code=condition_code,
                    target_values=target_values,
                    reference_session=2,
                    reference_condition_code=SHAM_CONDITION_CODE,
                    reference_values=on_sham_values,
                    comparison_kind="off_to_on_sham_reference",
                    n_permutations=n_permutations,
                    rng=rng,
                )
            )
    if not rows:
        return pd.DataFrame()
    stats_df = pd.DataFrame(rows)
    return _add_groupwise_fdr(
        stats_df,
        p_value_column="wasserstein_p_value",
        group_columns=["comparison_kind", "subject", "summary_metric"],
    )


def compute_sham_off_to_on_roi_ttest_stats(
    matrices: dict[tuple[str, int, str], np.ndarray],
    roi_labels: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    subjects = sorted({subject for subject, _, _ in matrices})
    for subject in subjects:
        off_sham_matrix = matrices.get((subject, 1, SHAM_CONDITION_CODE))
        on_sham_matrix = matrices.get((subject, 2, SHAM_CONDITION_CODE))
        if off_sham_matrix is None or on_sham_matrix is None:
            continue

        n_roi_rows = int(min(off_sham_matrix.shape[0], on_sham_matrix.shape[0], len(roi_labels)))
        for roi_index in range(n_roi_rows):
            off_values = np.asarray(off_sham_matrix[roi_index], dtype=np.float64)
            off_values = off_values[np.isfinite(off_values)]
            on_values = np.asarray(on_sham_matrix[roi_index], dtype=np.float64)
            on_values = on_values[np.isfinite(on_values)]

            t_stat = float("nan")
            p_value = float("nan")
            if off_values.size >= 2 and on_values.size >= 2:
                test_result = stats.ttest_ind(
                    on_values,
                    off_values,
                    equal_var=False,
                    nan_policy="omit",
                )
                t_stat = float(test_result.statistic) if np.isfinite(test_result.statistic) else float("nan")
                p_value = float(test_result.pvalue) if np.isfinite(test_result.pvalue) else float("nan")

            rows.append(
                {
                    "comparison_kind": "off_to_on_sham_reference",
                    "analysis_kind": SHAM_ROI_MEDICATION_TTEST_ANALYSIS,
                    "analysis_label": SHAM_ROI_MEDICATION_TTEST_LABEL,
                    "subject": subject,
                    "off_session": 1,
                    "on_session": 2,
                    "off_medication": "OFF",
                    "on_medication": "ON",
                    "condition_code": SHAM_CONDITION_CODE,
                    "condition_label": _condition_display_name(SHAM_CONDITION_CODE),
                    "roi_index": int(roi_index + 1),
                    "roi_label": str(roi_labels[roi_index]),
                    "n_trials_off": int(off_values.size),
                    "n_trials_on": int(on_values.size),
                    "mean_off": float(np.mean(off_values)) if off_values.size else float("nan"),
                    "mean_on": float(np.mean(on_values)) if on_values.size else float("nan"),
                    "median_off": float(np.median(off_values)) if off_values.size else float("nan"),
                    "median_on": float(np.median(on_values)) if on_values.size else float("nan"),
                    "mean_delta_on_minus_off": (
                        float(np.mean(on_values) - np.mean(off_values))
                        if off_values.size and on_values.size
                        else float("nan")
                    ),
                    "median_delta_on_minus_off": (
                        float(np.median(on_values) - np.median(off_values))
                        if off_values.size and on_values.size
                        else float("nan")
                    ),
                    "t_stat_welch": t_stat,
                    "p_value_two_sided": p_value,
                }
            )

    if not rows:
        return pd.DataFrame()

    stats_df = pd.DataFrame(rows).sort_values(["subject", "roi_index"]).reset_index(drop=True)
    return _add_groupwise_fdr(
        stats_df,
        p_value_column="p_value_two_sided",
        group_columns=["comparison_kind", "analysis_kind", "subject"],
    )


def _finite_mean(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def build_condition_roi_reference_delta_rows(
    matrices: dict[tuple[str, int, str], np.ndarray],
    roi_labels: list[str],
    *,
    target_session: int,
    reference_session: int,
    comparison_kind: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    subjects = sorted({subject for subject, _, _ in matrices})
    for subject in subjects:
        reference_matrix = matrices.get((subject, reference_session, SHAM_CONDITION_CODE))
        if reference_matrix is None:
            continue
        for condition_code in ALL_CONDITION_CODES:
            target_matrix = matrices.get((subject, target_session, condition_code))
            if target_matrix is None:
                continue
            n_roi_rows = int(min(target_matrix.shape[0], reference_matrix.shape[0], len(roi_labels)))
            for roi_index in range(n_roi_rows):
                target_mean = _finite_mean(target_matrix[roi_index])
                reference_mean = _finite_mean(reference_matrix[roi_index])
                rows.append(
                    {
                        "comparison_kind": comparison_kind,
                        "subject": subject,
                        "target_session": int(target_session),
                        "target_medication": medication_from_session(target_session),
                        "target_condition_code": condition_code,
                        "target_condition_label": _condition_display_name(condition_code),
                        "reference_session": int(reference_session),
                        "reference_medication": medication_from_session(reference_session),
                        "reference_condition_code": SHAM_CONDITION_CODE,
                        "reference_condition_label": _condition_display_name(SHAM_CONDITION_CODE),
                        "roi_index": int(roi_index + 1),
                        "roi_label": str(roi_labels[roi_index]),
                        "target_mean_beta": target_mean,
                        "reference_mean_beta": reference_mean,
                        "mean_delta_target_minus_reference": (
                            float(target_mean - reference_mean)
                            if np.isfinite(reference_mean) and np.isfinite(target_mean)
                            else float("nan")
                        ),
                    }
                )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["comparison_kind", "subject", "target_condition_code", "roi_index"]
    ).reset_index(drop=True)


def compute_condition_roi_reference_ttest_stats(
    matrices: dict[tuple[str, int, str], np.ndarray],
    roi_labels: list[str],
    *,
    target_session: int,
    reference_session: int,
    comparison_kind: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    subjects = sorted({subject for subject, _, _ in matrices})
    for subject in subjects:
        reference_matrix = matrices.get((subject, reference_session, SHAM_CONDITION_CODE))
        if reference_matrix is None:
            continue
        for condition_code in ALL_CONDITION_CODES:
            target_matrix = matrices.get((subject, target_session, condition_code))
            if target_matrix is None:
                continue
            n_roi_rows = int(min(target_matrix.shape[0], reference_matrix.shape[0], len(roi_labels)))
            for roi_index in range(n_roi_rows):
                target_values = np.asarray(target_matrix[roi_index], dtype=np.float64)
                target_values = target_values[np.isfinite(target_values)]
                reference_values = np.asarray(reference_matrix[roi_index], dtype=np.float64)
                reference_values = reference_values[np.isfinite(reference_values)]

                t_stat = float("nan")
                p_value = float("nan")
                if target_values.size >= 2 and reference_values.size >= 2:
                    test_result = stats.ttest_ind(
                        target_values,
                        reference_values,
                        equal_var=False,
                        nan_policy="omit",
                    )
                    t_stat = float(test_result.statistic) if np.isfinite(test_result.statistic) else float("nan")
                    p_value = float(test_result.pvalue) if np.isfinite(test_result.pvalue) else float("nan")

                rows.append(
                    {
                        "comparison_kind": comparison_kind,
                        "subject": subject,
                        "target_session": int(target_session),
                        "target_medication": medication_from_session(target_session),
                        "target_condition_code": condition_code,
                        "target_condition_label": _condition_display_name(condition_code),
                        "reference_session": int(reference_session),
                        "reference_medication": medication_from_session(reference_session),
                        "reference_condition_code": SHAM_CONDITION_CODE,
                        "reference_condition_label": _condition_display_name(SHAM_CONDITION_CODE),
                        "roi_index": int(roi_index + 1),
                        "roi_label": str(roi_labels[roi_index]),
                        "n_trials_target": int(target_values.size),
                        "n_trials_reference": int(reference_values.size),
                        "mean_target": float(np.mean(target_values)) if target_values.size else float("nan"),
                        "mean_reference": float(np.mean(reference_values)) if reference_values.size else float("nan"),
                        "mean_delta_target_minus_reference": (
                            float(np.mean(target_values) - np.mean(reference_values))
                            if target_values.size and reference_values.size
                            else float("nan")
                        ),
                        "t_stat_welch": t_stat,
                        "p_value_two_sided": p_value,
                    }
                )
    if not rows:
        return pd.DataFrame()

    stats_df = pd.DataFrame(rows).sort_values(
        ["comparison_kind", "subject", "target_condition_code", "roi_index"]
    ).reset_index(drop=True)
    return _add_groupwise_fdr(
        stats_df,
        p_value_column="p_value_two_sided",
        group_columns=["comparison_kind", "subject"],
    )


def _comparison_row(
    *,
    subject: str,
    target_session: int,
    target_condition_code: str,
    target_matrix: np.ndarray,
    reference_session: int,
    reference_condition_code: str,
    reference_matrix: np.ndarray,
    comparison_kind: str,
    target_trials: int,
) -> dict[str, Any]:
    similarity = compute_matrix_similarity(
        target_matrix=target_matrix,
        reference_matrix=reference_matrix,
        target_trials=target_trials,
    )
    return {
        "subject": subject,
        "target_session": int(target_session),
        "target_medication": medication_from_session(target_session),
        "target_condition_code": target_condition_code,
        "target_condition_label": _condition_display_name(target_condition_code),
        "reference_session": int(reference_session),
        "reference_medication": medication_from_session(reference_session),
        "reference_condition_code": reference_condition_code,
        "reference_condition_label": _condition_display_name(reference_condition_code),
        "comparison_kind": comparison_kind,
        "n_rois": int(target_matrix.shape[0]),
        "n_trials_target": int(target_matrix.shape[1]),
        "n_trials_reference": int(reference_matrix.shape[1]),
        "target_trials_resampled": int(target_trials),
        **similarity,
    }


def build_within_medication_rows(
    matrices: dict[tuple[str, int, str], np.ndarray],
    target_trials: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    subjects = sorted({subject for subject, _, _ in matrices})
    for subject in subjects:
        for session in (1, 2):
            sham_matrix = matrices.get((subject, session, SHAM_CONDITION_CODE))
            if sham_matrix is None:
                continue
            for condition_code in ACTIVE_CONDITION_CODES:
                target_matrix = matrices.get((subject, session, condition_code))
                if target_matrix is None:
                    continue
                rows.append(
                    _comparison_row(
                        subject=subject,
                        target_session=session,
                        target_condition_code=condition_code,
                        target_matrix=target_matrix,
                        reference_session=session,
                        reference_condition_code=SHAM_CONDITION_CODE,
                        reference_matrix=sham_matrix,
                        comparison_kind="within_med_sham_reference",
                        target_trials=target_trials,
                    )
                )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["target_medication", "subject", "target_condition_code"]
    ).reset_index(drop=True)


def build_off_to_on_sham_rows(
    matrices: dict[tuple[str, int, str], np.ndarray],
    target_trials: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    subjects = sorted({subject for subject, _, _ in matrices})
    for subject in subjects:
        on_sham = matrices.get((subject, 2, SHAM_CONDITION_CODE))
        if on_sham is None:
            continue
        for condition_code in ALL_CONDITION_CODES:
            off_matrix = matrices.get((subject, 1, condition_code))
            if off_matrix is None:
                continue
            rows.append(
                _comparison_row(
                    subject=subject,
                    target_session=1,
                    target_condition_code=condition_code,
                    target_matrix=off_matrix,
                    reference_session=2,
                    reference_condition_code=SHAM_CONDITION_CODE,
                    reference_matrix=on_sham,
                    comparison_kind="off_to_on_sham_reference",
                    target_trials=target_trials,
                )
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["subject", "target_condition_code"]).reset_index(drop=True)


def _trial_vectors(matrix: np.ndarray) -> np.ndarray:
    array = np.asarray(matrix, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"Expected 2D ROI beta matrix, got {array.shape}.")
    return array.T


def _l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    array = np.asarray(matrix, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"Expected 2D trial matrix, got {array.shape}.")
    filled = np.where(np.isfinite(array), array, 0.0)
    norms = np.linalg.norm(filled, axis=1, keepdims=True)
    norms = np.where(np.isclose(norms, 0.0), 1.0, norms)
    return filled / norms


def _pairwise_cosine_similarity_matrix(
    target_trials: np.ndarray,
    reference_trials: np.ndarray,
) -> np.ndarray:
    return _l2_normalize_rows(target_trials) @ _l2_normalize_rows(reference_trials).T


def _pairwise_correlation_similarity_matrix(
    target_trials: np.ndarray,
    reference_trials: np.ndarray,
) -> np.ndarray:
    target_array = np.asarray(target_trials, dtype=np.float64)
    reference_array = np.asarray(reference_trials, dtype=np.float64)
    if target_array.ndim != 2 or reference_array.ndim != 2:
        raise ValueError(
            "Expected 2D trial matrices for trial-wise correlation, "
            f"got {target_array.shape} and {reference_array.shape}."
        )
    result = np.full((target_array.shape[0], reference_array.shape[0]), np.nan, dtype=np.float64)
    for row_index, target_row in enumerate(target_array):
        for col_index, reference_row in enumerate(reference_array):
            valid = np.isfinite(target_row) & np.isfinite(reference_row)
            if np.count_nonzero(valid) < 3:
                continue
            result[row_index, col_index] = _pearson_similarity(target_row[valid], reference_row[valid])
    return result


def _pairwise_frobenius_distance_matrix(
    target_trials: np.ndarray,
    reference_trials: np.ndarray,
) -> np.ndarray:
    target_array = np.asarray(target_trials, dtype=np.float64)
    reference_array = np.asarray(reference_trials, dtype=np.float64)
    if target_array.ndim != 2 or reference_array.ndim != 2:
        raise ValueError(
            "Expected 2D trial matrices for trial-wise Frobenius distance, "
            f"got {target_array.shape} and {reference_array.shape}."
        )
    result = np.full((target_array.shape[0], reference_array.shape[0]), np.nan, dtype=np.float64)
    for row_index, target_row in enumerate(target_array):
        for col_index, reference_row in enumerate(reference_array):
            valid = np.isfinite(target_row) & np.isfinite(reference_row)
            if not np.any(valid):
                continue
            result[row_index, col_index] = float(np.linalg.norm(target_row[valid] - reference_row[valid], ord=2))
    return result


def _compute_trial_pairwise_matrix(
    target_trials: np.ndarray,
    reference_trials: np.ndarray,
    *,
    pairwise_metric: str,
) -> np.ndarray:
    if pairwise_metric == "cosine_similarity":
        return _pairwise_cosine_similarity_matrix(target_trials, reference_trials)
    if pairwise_metric == "correlation_similarity":
        return _pairwise_correlation_similarity_matrix(target_trials, reference_trials)
    if pairwise_metric == "frobenius_distance":
        return _pairwise_frobenius_distance_matrix(target_trials, reference_trials)
    raise KeyError(f"Unknown trial pairwise metric {pairwise_metric!r}.")


def _reduce_trial_metric(values: np.ndarray, reducer: str) -> float:
    finite_values = np.asarray(values, dtype=np.float64)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return float("nan")
    if reducer == "mean":
        return float(np.mean(finite_values))
    if reducer == "max":
        return float(np.max(finite_values))
    if reducer == "min":
        return float(np.min(finite_values))
    raise ValueError(f"Unsupported trial metric reducer {reducer!r}.")


def _argextreme_trial_metric(values: np.ndarray, reducer: str) -> int | None:
    finite_mask = np.isfinite(values)
    if not np.any(finite_mask):
        return None
    finite_indices = np.flatnonzero(finite_mask)
    finite_values = np.asarray(values, dtype=np.float64)[finite_mask]
    if reducer == "max":
        offset = int(np.argmax(finite_values))
    elif reducer == "min":
        offset = int(np.argmin(finite_values))
    else:
        raise ValueError(f"Unsupported closest-trial reducer {reducer!r}.")
    return int(finite_indices[offset])


def _build_trial_metric_rows_for_pair(
    *,
    subject: str,
    target_session: int,
    target_condition_code: str,
    target_matrix: np.ndarray,
    reference_session: int,
    reference_condition_code: str,
    reference_matrix: np.ndarray,
    comparison_kind: str,
    analysis_spec: dict[str, Any],
    same_trial_pool: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    target_trials = _trial_vectors(target_matrix)
    reference_trials = _trial_vectors(reference_matrix)
    if target_trials.shape[0] == 0 or reference_trials.shape[0] == 0:
        return rows

    pairwise_metric = str(analysis_spec["pairwise_metric"])
    closest_reducer = str(analysis_spec["closest_reducer"])
    summary_metrics = list(analysis_spec["summary_metrics"])
    include_legacy_distance_columns = bool(analysis_spec.get("include_legacy_distance_columns"))
    pairwise_matrix = _compute_trial_pairwise_matrix(
        target_trials,
        reference_trials,
        pairwise_metric=pairwise_metric,
    )
    if same_trial_pool and pairwise_matrix.shape[0] == pairwise_matrix.shape[1]:
        pairwise_matrix = np.asarray(pairwise_matrix, dtype=np.float64).copy()
        np.fill_diagonal(pairwise_matrix, np.nan)

    reference_group_label = _reference_group_label(reference_session, reference_condition_code)
    for trial_index in range(target_trials.shape[0]):
        trial_values = np.asarray(pairwise_matrix[trial_index], dtype=np.float64)
        row: dict[str, Any] = {
            "comparison_kind": comparison_kind,
            "subject": subject,
            "target_session": int(target_session),
            "target_medication": medication_from_session(target_session),
            "target_condition_code": target_condition_code,
            "target_condition_label": _condition_display_name(target_condition_code),
            "reference_session": int(reference_session),
            "reference_medication": medication_from_session(reference_session),
            "reference_condition_code": reference_condition_code,
            "reference_condition_label": _condition_display_name(reference_condition_code),
            "reference_group_label": reference_group_label,
            "target_trial_index": int(trial_index + 1),
            "n_rois": int(target_matrix.shape[0]),
            "n_reference_trials": int(reference_trials.shape[0]),
        }
        for metric_spec in summary_metrics:
            row[str(metric_spec["column"])] = _reduce_trial_metric(
                trial_values,
                str(metric_spec["reducer"]),
            )
        closest_index = _argextreme_trial_metric(trial_values, closest_reducer)
        row["closest_reference_trial_index"] = (
            int(closest_index + 1) if closest_index is not None else float("nan")
        )
        if include_legacy_distance_columns:
            trial_distance = 1.0 - trial_values
            row["mean_cosine_distance_to_on_sham"] = _reduce_trial_metric(trial_distance, "mean")
            row["min_cosine_distance_to_on_sham"] = _reduce_trial_metric(trial_distance, "min")
        rows.append(row)
    return rows


def build_within_medication_trial_metric_rows(
    matrices: dict[tuple[str, int, str], np.ndarray],
    *,
    analysis_spec: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    subjects = sorted({subject for subject, _, _ in matrices})
    for subject in subjects:
        for session in (1, 2):
            sham_matrix = matrices.get((subject, session, SHAM_CONDITION_CODE))
            if sham_matrix is None:
                continue
            for condition_code in ALL_CONDITION_CODES:
                target_matrix = matrices.get((subject, session, condition_code))
                if target_matrix is None:
                    continue
                rows.extend(
                    _build_trial_metric_rows_for_pair(
                        subject=subject,
                        target_session=session,
                        target_condition_code=condition_code,
                        target_matrix=target_matrix,
                        reference_session=session,
                        reference_condition_code=SHAM_CONDITION_CODE,
                        reference_matrix=sham_matrix,
                        comparison_kind="within_med_sham_reference",
                        analysis_spec=analysis_spec,
                        same_trial_pool=condition_code == SHAM_CONDITION_CODE,
                    )
                )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["target_medication", "subject", "target_condition_code", "target_trial_index"]
    ).reset_index(drop=True)


def build_off_to_on_trial_metric_rows(
    matrices: dict[tuple[str, int, str], np.ndarray],
    *,
    analysis_spec: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    subjects = sorted({subject for subject, _, _ in matrices})
    for subject in subjects:
        on_sham_matrix = matrices.get((subject, 2, SHAM_CONDITION_CODE))
        if on_sham_matrix is None:
            continue
        for condition_code in ALL_CONDITION_CODES:
            off_matrix = matrices.get((subject, 1, condition_code))
            if off_matrix is None:
                continue
            rows.extend(
                _build_trial_metric_rows_for_pair(
                    subject=subject,
                    target_session=1,
                    target_condition_code=condition_code,
                    target_matrix=off_matrix,
                    reference_session=2,
                    reference_condition_code=SHAM_CONDITION_CODE,
                    reference_matrix=on_sham_matrix,
                    comparison_kind="off_to_on_sham_reference",
                    analysis_spec=analysis_spec,
                )
            )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["subject", "target_condition_code", "target_trial_index"]
    ).reset_index(drop=True)


def build_trial_metric_to_on_sham_rows(
    matrices: dict[tuple[str, int, str], np.ndarray],
    *,
    analysis_spec: dict[str, Any],
) -> pd.DataFrame:
    return build_off_to_on_trial_metric_rows(
        matrices,
        analysis_spec=analysis_spec,
    )


def build_trial_distance_to_on_sham_rows(
    matrices: dict[tuple[str, int, str], np.ndarray],
) -> pd.DataFrame:
    return build_trial_metric_to_on_sham_rows(
        matrices,
        analysis_spec=TRIAL_ANALYSIS_SPECS[DEFAULT_TRIAL_ANALYSIS_KEY],
    )


def _format_value(value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    return f"{value:.3f}"


def _render_heatmap(
    pivot_df: pd.DataFrame,
    title: str,
    metric_label: str,
    out_path: Path,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    if pivot_df.empty:
        return
    data = pivot_df.to_numpy(dtype=np.float64)
    masked = np.ma.masked_invalid(data)
    fig_w = max(6.0, 1.15 * pivot_df.shape[1] + 2.0)
    fig_h = max(4.0, 0.45 * pivot_df.shape[0] + 1.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    cmap = plt.get_cmap(HEATMAP_CMAP).copy()
    cmap.set_bad(color="#d9d9d9")
    im = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(pivot_df.shape[1]))
    ax.set_xticklabels(pivot_df.columns.tolist(), rotation=45, ha="right")
    ax.set_yticks(np.arange(pivot_df.shape[0]))
    ax.set_yticklabels(pivot_df.index.tolist())
    ax.set_title(title)
    for row_idx in range(pivot_df.shape[0]):
        for col_idx in range(pivot_df.shape[1]):
            value = data[row_idx, col_idx]
            if not np.isfinite(value):
                continue
            text_color = "white" if (vmin is not None and vmax is not None and value < (vmin + vmax) / 2.0) else "black"
            ax.text(col_idx, row_idx, f"{value:.2f}", ha="center", va="center", fontsize=8, color=text_color)
    cbar = fig.colorbar(im, ax=ax, shrink=0.92)
    cbar.set_label(metric_label)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _write_metric_heatmap_bundle(
    df: pd.DataFrame,
    tables_dir: Path,
    plots_dir: Path,
    *,
    metric_name: str,
    group_name: str,
    title: str,
    condition_codes: list[str],
) -> None:
    if df.empty:
        return
    if metric_name not in METRIC_SPECS:
        raise KeyError(f"Unknown metric {metric_name!r}.")
    ordered_labels = [_condition_display_name(code) for code in condition_codes]
    subset = df.copy()
    subset["target_condition_label"] = pd.Categorical(
        subset["target_condition_label"],
        categories=ordered_labels,
        ordered=True,
    )
    pivot_df = subset.pivot(index="subject", columns="target_condition_label", values=metric_name)
    pivot_df = pivot_df.reindex(columns=ordered_labels)
    pivot_df = pivot_df.sort_index()
    pivot_df.to_csv(tables_dir / f"{group_name}_{metric_name}_wide.csv")
    metric_spec = METRIC_SPECS[metric_name]
    _render_heatmap(
        pivot_df=pivot_df,
        title=title,
        metric_label=str(metric_spec["label"]),
        out_path=plots_dir / f"{group_name}_{metric_name}_heatmap.png",
        vmin=metric_spec["vmin"],
        vmax=metric_spec["vmax"],
    )


def _write_metric_summary(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        pd.DataFrame().to_csv(out_path, index=False)
        return
    grouped = (
        df.groupby(
            ["comparison_kind", "target_medication", "target_condition_code", "target_condition_label"],
            dropna=False,
            observed=False,
        )[[PRIMARY_METRIC, *SECONDARY_METRICS]]
        .agg(["mean", "std", "count"])
    )
    grouped.columns = ["_".join(part for part in col if part) for col in grouped.columns]
    grouped = grouped.reset_index().sort_values(
        ["comparison_kind", "target_medication", "target_condition_code"]
    )
    grouped.to_csv(out_path, index=False)


def _write_trial_metric_summary(
    df: pd.DataFrame,
    out_path: Path,
    *,
    metric_columns: list[str],
) -> None:
    if df.empty:
        pd.DataFrame().to_csv(out_path, index=False)
        return
    available_columns = [column for column in metric_columns if column in df.columns]
    if not available_columns:
        pd.DataFrame().to_csv(out_path, index=False)
        return
    group_columns = [
        column
        for column in [
            "comparison_kind",
            "subject",
            "target_medication",
            "reference_group_label",
            "target_condition_code",
            "target_condition_label",
        ]
        if column in df.columns
    ]
    grouped = (
        df.groupby(
            group_columns,
            dropna=False,
            observed=False,
        )[
            available_columns
        ]
        .agg(["mean", "median", "std", "count"])
    )
    grouped.columns = ["_".join(part for part in col if part) for col in grouped.columns]
    sort_columns = [column for column in ["comparison_kind", "target_medication", "subject", "target_condition_code"] if column in group_columns]
    grouped = grouped.reset_index().sort_values(sort_columns)
    grouped.to_csv(out_path, index=False)


def _write_trial_distance_summary(df: pd.DataFrame, out_path: Path) -> None:
    _write_trial_metric_summary(
        df,
        out_path,
        metric_columns=_trial_summary_columns(TRIAL_ANALYSIS_SPECS[DEFAULT_TRIAL_ANALYSIS_KEY]),
    )


def compute_trial_gvs_vs_sham_stats(
    df: pd.DataFrame,
    *,
    analysis_spec: dict[str, Any],
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    context_columns = [
        column
        for column in [
            "comparison_kind",
            "subject",
            "target_session",
            "target_medication",
            "reference_session",
            "reference_medication",
            "reference_condition_code",
            "reference_condition_label",
            "reference_group_label",
        ]
        if column in df.columns
    ]
    rows: list[dict[str, Any]] = []
    for group_key, group_df in df.groupby(context_columns, dropna=False, observed=False, sort=True):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        context_row = dict(zip(context_columns, group_key, strict=True))
        subject = str(context_row["subject"])
        for metric_spec in analysis_spec["summary_metrics"]:
            metric_name = str(metric_spec["column"])
            metric_label = str(metric_spec["label"])
            alternative = str(metric_spec["alternative"])
            sham_values = (
                group_df.loc[group_df["target_condition_code"] == SHAM_CONDITION_CODE, metric_name]
                .dropna()
                .to_numpy(dtype=np.float64)
            )
            metric_rows: list[dict[str, Any]] = []
            for condition_code in ACTIVE_CONDITION_CODES:
                condition_label = _condition_display_name(condition_code)
                condition_values = (
                    group_df.loc[group_df["target_condition_code"] == condition_code, metric_name]
                    .dropna()
                    .to_numpy(dtype=np.float64)
                )

                p_value = float("nan")
                statistic = float("nan")
                if sham_values.size >= 1 and condition_values.size >= 1:
                    try:
                        test_result = stats.mannwhitneyu(
                            condition_values,
                            sham_values,
                            alternative=alternative,
                            method="auto",
                        )
                        statistic = float(test_result.statistic)
                        p_value = float(test_result.pvalue)
                    except ValueError:
                        p_value = float("nan")
                        statistic = float("nan")

                metric_rows.append(
                    {
                        **context_row,
                        "subject": subject,
                        "summary_metric": metric_name,
                        "summary_metric_label": metric_label,
                        "condition_code": condition_code,
                        "condition_label": condition_label,
                        "sham_condition_code": SHAM_CONDITION_CODE,
                        "sham_condition_label": _condition_display_name(SHAM_CONDITION_CODE),
                        "alternative_hypothesis": alternative,
                        "n_trials_condition": int(condition_values.size),
                        "n_trials_sham": int(sham_values.size),
                        "mean_condition": float(np.mean(condition_values)) if condition_values.size else float("nan"),
                        "median_condition": float(np.median(condition_values)) if condition_values.size else float("nan"),
                        "mean_sham": float(np.mean(sham_values)) if sham_values.size else float("nan"),
                        "median_sham": float(np.median(sham_values)) if sham_values.size else float("nan"),
                        "mean_delta_vs_sham": (
                            float(np.mean(condition_values) - np.mean(sham_values))
                            if condition_values.size and sham_values.size
                            else float("nan")
                        ),
                        "median_delta_vs_sham": (
                            float(np.median(condition_values) - np.median(sham_values))
                            if condition_values.size and sham_values.size
                            else float("nan")
                        ),
                        "mannwhitney_u_statistic": statistic,
                        "p_value": p_value,
                    }
                )

            metric_df = pd.DataFrame(metric_rows)
            q_values = np.full(metric_df.shape[0], np.nan, dtype=np.float64)
            sig = np.zeros(metric_df.shape[0], dtype=bool)
            finite_mask = np.isfinite(metric_df["p_value"].to_numpy(dtype=np.float64))
            if np.any(finite_mask):
                sig_valid, q_valid = fdrcorrection(metric_df.loc[finite_mask, "p_value"], alpha=0.05)
                q_values[finite_mask] = q_valid
                sig[finite_mask] = sig_valid
            metric_df["q_value_fdr"] = q_values
            metric_df["significant_fdr"] = sig
            rows.extend(metric_df.to_dict(orient="records"))

    if not rows:
        return pd.DataFrame()
    sort_columns = [column for column in ["comparison_kind", "target_medication", "subject", "summary_metric", "condition_code"] if column in pd.DataFrame(rows).columns]
    return pd.DataFrame(rows).sort_values(sort_columns).reset_index(drop=True)


def compute_trial_distance_gvs_vs_sham_stats(df: pd.DataFrame) -> pd.DataFrame:
    return compute_trial_gvs_vs_sham_stats(
        df,
        analysis_spec=TRIAL_ANALYSIS_SPECS[DEFAULT_TRIAL_ANALYSIS_KEY],
    )


def _write_significant_condition_summary(stats_df: pd.DataFrame, out_path: Path) -> None:
    if stats_df.empty:
        pd.DataFrame().to_csv(out_path, index=False)
        return

    has_medication = "target_medication" in stats_df.columns
    metric_name_column = "metric_name" if "metric_name" in stats_df.columns else "summary_metric"
    metric_label_column = "metric_label" if "metric_label" in stats_df.columns else "summary_metric_label"
    condition_code_column = "target_condition_code" if "target_condition_code" in stats_df.columns else "condition_code"
    condition_label_column = "target_condition_label" if "target_condition_label" in stats_df.columns else "condition_label"
    group_cols = ["comparison_kind", "subject"]
    if has_medication:
        group_cols.append("target_medication")
    group_cols.extend([metric_name_column, metric_label_column])

    rows: list[dict[str, Any]] = []
    for group_key, group_df in stats_df.groupby(group_cols, dropna=False, observed=False, sort=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        row = dict(zip(group_cols, group_key, strict=True))
        significant_conditions = (
            group_df.loc[group_df["significant_fdr"], condition_label_column].astype(str).tolist()
        )
        row["n_conditions_tested"] = int(group_df[condition_code_column].nunique())
        row["n_significant_conditions_fdr"] = int(len(significant_conditions))
        row["significant_conditions_fdr"] = ", ".join(significant_conditions) if significant_conditions else "None"
        rows.append(row)

    pd.DataFrame(rows).to_csv(out_path, index=False)


def _clear_directory_files(dir_path: Path) -> None:
    for child in dir_path.iterdir():
        if child.is_file():
            child.unlink()


def _clear_directory_tree(dir_path: Path) -> None:
    for child in dir_path.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        elif child.is_file():
            child.unlink()


def _write_distribution_summary(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        pd.DataFrame().to_csv(out_path, index=False)
        return
    df.to_csv(out_path, index=False)


def _write_significant_roi_summary(stats_df: pd.DataFrame, out_path: Path) -> None:
    if stats_df.empty:
        pd.DataFrame().to_csv(out_path, index=False)
        return

    rows: list[dict[str, Any]] = []
    for subject, subject_df in stats_df.groupby("subject", dropna=False, observed=False, sort=True):
        significant_rois = subject_df.loc[subject_df["significant_fdr"], "roi_label"].astype(str).tolist()
        rows.append(
            {
                "comparison_kind": "off_to_on_sham_reference",
                "analysis_kind": SHAM_ROI_MEDICATION_TTEST_ANALYSIS,
                "subject": str(subject),
                "n_rois_tested": int(subject_df["roi_label"].nunique()),
                "n_significant_rois_fdr": int(len(significant_rois)),
                "significant_rois_fdr": ", ".join(significant_rois) if significant_rois else "None",
            }
        )

    pd.DataFrame(rows).to_csv(out_path, index=False)


def _format_significant_roi_cell(roi_labels: list[str], *, wrap_width: int = 22) -> str:
    cleaned_labels = [_strip_relative_suffix(label) for label in roi_labels if str(label).strip()]
    cleaned_labels = [
        label
        for label in cleaned_labels
        if label.strip().casefold() != "unassigned active voxels"
    ]
    cleaned_labels = list(dict.fromkeys(cleaned_labels))
    if not cleaned_labels:
        return "-"
    return "\n".join(
        textwrap.fill(label, width=wrap_width, break_long_words=False, break_on_hyphens=False)
        for label in cleaned_labels
    )


def _write_significant_roi_condition_table_png(
    stats_df: pd.DataFrame,
    out_path: Path,
    *,
    title: str,
    condition_labels: list[str],
) -> None:
    if stats_df.empty:
        fig, ax = plt.subplots(figsize=(10.0, 3.0))
        ax.axis("off")
        ax.text(0.5, 0.5, "No rows available.", ha="center", va="center", fontsize=12)
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        return

    subjects = sorted(stats_df["subject"].astype(str).unique().tolist())
    active_condition_labels = [str(label) for label in condition_labels if str(label) != "sham"]
    if not active_condition_labels:
        return

    cell_text: list[list[str]] = []
    row_line_counts: list[int] = []
    for subject in subjects:
        row_values: list[str] = []
        row_max_lines = 1
        for condition_label in active_condition_labels:
            cell_df = stats_df.loc[
                (stats_df["subject"].astype(str) == subject)
                & (stats_df["target_condition_label"].astype(str) == condition_label)
                & (stats_df["significant_fdr"].fillna(False).astype(bool))
            ]
            cell_label = _format_significant_roi_cell(cell_df["roi_label"].astype(str).tolist())
            row_values.append(cell_label)
            row_max_lines = max(row_max_lines, cell_label.count("\n") + 1)
        cell_text.append(row_values)
        row_line_counts.append(row_max_lines)

    fig_w = max(9.5, 1.6 * len(active_condition_labels) + 2.2)
    fig_h = max(4.0, 0.42 * sum(row_line_counts) + 1.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        rowLabels=subjects,
        colLabels=active_condition_labels,
        cellLoc="left",
        colWidths=[0.10] * len(active_condition_labels),
        bbox=[0.0, 0.0, 1.0, 0.92],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(0.92, max(1.7, 0.46 * max(row_line_counts)))

    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("black")
        cell.set_linewidth(0.6)
        if row_idx == 0:
            cell.set_facecolor("#e6e6e6")
            cell.set_text_props(weight="bold", ha="center", va="center")
        elif col_idx == -1:
            cell.set_facecolor("#f2f2f2")
            cell.set_text_props(weight="bold", ha="right", va="center")
        else:
            cell.set_facecolor("white")
            cell.set_text_props(ha="left", va="center")

    ax.set_title(title, pad=12)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.98])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _write_significant_roi_condition_panel_png(
    stats_df: pd.DataFrame,
    out_path: Path,
    *,
    title: str,
    condition_labels: list[str],
    roi_labels: list[str],
) -> None:
    if stats_df.empty:
        fig, ax = plt.subplots(figsize=(10.0, 3.0))
        ax.axis("off")
        ax.text(0.5, 0.5, "No rows available.", ha="center", va="center", fontsize=12)
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        return

    subjects = sorted(stats_df["subject"].astype(str).unique().tolist())
    active_condition_labels = [str(label) for label in condition_labels if str(label) != "sham"]
    ordered_roi_labels = [str(label) for label in roi_labels]
    if not active_condition_labels or not ordered_roi_labels or not subjects:
        return

    subset = stats_df.copy()
    subset["target_condition_label"] = pd.Categorical(
        subset["target_condition_label"],
        categories=active_condition_labels,
        ordered=True,
    )
    subset["roi_label"] = pd.Categorical(
        subset["roi_label"],
        categories=ordered_roi_labels,
        ordered=True,
    )
    subset = subset.loc[subset["target_condition_label"].notna()].copy()
    subset = subset.sort_values(["subject", "target_condition_label", "roi_label"]).reset_index(drop=True)

    n_subjects = len(subjects)
    n_cols = min(4, max(1, n_subjects))
    n_rows = int(np.ceil(n_subjects / n_cols))
    fig_w = max(15.0, 5.2 * n_cols + 1.2)
    fig_h = max(8.0, 3.4 * n_rows + 1.6)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), constrained_layout=True)
    axes = np.atleast_2d(axes)
    lowest_filled_row_by_col = {
        col_idx: (n_subjects - 1 - col_idx) // n_cols
        for col_idx in range(n_cols)
        if col_idx < n_subjects
    }

    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad(color="#e5e5e5")
    roi_display_labels = [_strip_relative_suffix(label) for label in ordered_roi_labels]

    for panel_idx, subject in enumerate(subjects):
        row_idx, col_idx = divmod(panel_idx, n_cols)
        ax = axes[row_idx, col_idx]
        subject_subset = subset.loc[subset["subject"].astype(str) == subject].copy()

        value_pivot = subject_subset.pivot(
            index="target_condition_label",
            columns="roi_label",
            values="mean_delta_target_minus_reference",
        )
        value_pivot = value_pivot.reindex(index=active_condition_labels, columns=ordered_roi_labels)

        sig_pivot = subject_subset.pivot(
            index="target_condition_label",
            columns="roi_label",
            values="significant_fdr",
        )
        sig_pivot = sig_pivot.reindex(index=active_condition_labels, columns=ordered_roi_labels)

        data = value_pivot.to_numpy(dtype=np.float64)
        sig_mask = sig_pivot.fillna(False).to_numpy(dtype=bool)
        display_data = np.where(sig_mask, data, np.nan)
        masked = np.ma.masked_invalid(display_data)
        subject_values = display_data[np.isfinite(display_data)]
        if subject_values.size == 0:
            subject_values = data[np.isfinite(data)]
        subject_abs_max = float(np.max(np.abs(subject_values))) if subject_values.size else 1e-6
        subject_abs_max = max(subject_abs_max, 1e-6)
        norm = mcolors.TwoSlopeNorm(vmin=-subject_abs_max, vcenter=0.0, vmax=subject_abs_max)
        image = ax.imshow(masked, aspect="auto", cmap=cmap, norm=norm)

        ax.set_title(str(subject), fontsize=10, pad=8)
        ax.set_xticks(np.arange(len(ordered_roi_labels)))
        if row_idx == lowest_filled_row_by_col.get(col_idx, n_rows - 1):
            ax.set_xticklabels(roi_display_labels, rotation=45, ha="right", fontsize=8)
        else:
            ax.set_xticklabels([])
        ax.set_yticks(np.arange(len(active_condition_labels)))
        if col_idx == 0:
            ax.set_yticklabels(active_condition_labels, fontsize=8)
        else:
            ax.set_yticklabels([])

        ax.set_xticks(np.arange(len(ordered_roi_labels) + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(active_condition_labels) + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=0.5)
        ax.tick_params(which="minor", bottom=False, left=False)
        cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
        if col_idx == n_cols - 1:
            cbar.set_label("Mean beta delta", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    for empty_idx in range(n_subjects, n_rows * n_cols):
        row_idx, col_idx = divmod(empty_idx, n_cols)
        axes[row_idx, col_idx].axis("off")

    fig.text(0.5, 0.02, "ROI", ha="center", va="center")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _format_heatmap_value(value: float, *, scientific_below: float = 1e-3) -> str:
    if not np.isfinite(value):
        return ""
    if abs(value) < scientific_below and value != 0.0:
        return f"{value:.1e}"
    return f"{value:.3f}"


def _render_annotated_pivot_heatmap(
    pivot_df: pd.DataFrame,
    *,
    title: str,
    colorbar_label: str,
    out_path: Path,
    cmap_name: str,
    vmin: float | None = None,
    vmax: float | None = None,
    norm: mcolors.Normalize | None = None,
    sig_pivot: pd.DataFrame | None = None,
    scientific_below: float = 1e-3,
    value_formatter: Callable[[float], str] | None = None,
) -> None:
    data = pivot_df.to_numpy(dtype=np.float64)
    masked = np.ma.masked_invalid(data)
    fig_w = max(7.0, 1.1 * pivot_df.shape[1] + 2.4)
    fig_h = max(4.8, 0.42 * pivot_df.shape[0] + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad(color="#d9d9d9")
    if norm is None:
        im = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    else:
        im = ax.imshow(masked, aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(np.arange(pivot_df.shape[1]))
    ax.set_xticklabels(pivot_df.columns.tolist(), rotation=45, ha="right")
    ax.set_yticks(np.arange(pivot_df.shape[0]))
    ax.set_yticklabels(pivot_df.index.tolist())
    ax.set_title(title)
    if value_formatter is None:
        value_formatter = lambda value: _format_heatmap_value(value, scientific_below=scientific_below)
    effective_vmax = vmax
    if effective_vmax is None and np.isfinite(data).any():
        effective_vmax = float(np.nanmax(data))
    for row_idx in range(pivot_df.shape[0]):
        for col_idx in range(pivot_df.shape[1]):
            value = data[row_idx, col_idx]
            if not np.isfinite(value):
                continue
            is_sig = False
            if sig_pivot is not None and row_idx < sig_pivot.shape[0] and col_idx < sig_pivot.shape[1]:
                sig_value = sig_pivot.iloc[row_idx, col_idx]
                is_sig = bool(sig_value) if pd.notna(sig_value) else False
            label = value_formatter(value) + ("*" if is_sig else "")
            rgba = im.cmap(im.norm(value))
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            text_color = "white" if luminance < 0.5 else "black"
            ax.text(col_idx, row_idx, label, ha="center", va="center", fontsize=8, color=text_color)
    cbar = fig.colorbar(im, ax=ax, shrink=0.92)
    cbar.set_label(colorbar_label)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _write_distribution_heatmap(
    stats_df: pd.DataFrame,
    tables_dir: Path,
    plots_dir: Path,
    *,
    group_name: str,
    title: str,
    medication: str | None = None,
    condition_codes: list[str] | tuple[str, ...] = ACTIVE_CONDITION_CODES,
) -> None:
    if stats_df.empty:
        return
    subset = stats_df.copy()
    if medication is not None and "target_medication" in subset.columns:
        subset = subset.loc[subset["target_medication"] == medication].reset_index(drop=True)
    if subset.empty:
        return

    ordered_labels = [_condition_display_name(code) for code in condition_codes]
    subset["target_condition_label"] = pd.Categorical(
        subset["target_condition_label"],
        categories=ordered_labels,
        ordered=True,
    )
    wasserstein_pivot = subset.pivot(index="subject", columns="target_condition_label", values="wasserstein_distance")
    wasserstein_pivot = wasserstein_pivot.reindex(columns=ordered_labels).sort_index()
    wasserstein_pivot.to_csv(tables_dir / f"{group_name}_wasserstein_distance_wide.csv")

    p_pivot = subset.pivot(index="subject", columns="target_condition_label", values="wasserstein_p_value")
    p_pivot = p_pivot.reindex(columns=ordered_labels).sort_index()
    p_pivot.to_csv(tables_dir / f"{group_name}_wasserstein_p_value_wide.csv")

    q_pivot = subset.pivot(index="subject", columns="target_condition_label", values="q_value_fdr")
    q_pivot = q_pivot.reindex(columns=ordered_labels).sort_index()
    q_pivot.to_csv(tables_dir / f"{group_name}_wasserstein_q_value_fdr_wide.csv")

    sig_pivot = subset.pivot(index="subject", columns="target_condition_label", values="significant_fdr")
    sig_pivot = sig_pivot.reindex(columns=ordered_labels).sort_index()

    wasserstein_data = wasserstein_pivot.to_numpy(dtype=np.float64)
    wasserstein_vmax = float(np.nanmax(wasserstein_data)) if np.isfinite(wasserstein_data).any() else None
    _render_annotated_pivot_heatmap(
        wasserstein_pivot,
        title=title,
        colorbar_label="Wasserstein distance",
        out_path=plots_dir / f"{group_name}_wasserstein_distance_heatmap.png",
        cmap_name=HEATMAP_CMAP,
        vmin=0.0,
        vmax=wasserstein_vmax,
        sig_pivot=sig_pivot,
        scientific_below=1e-3,
    )
    _render_annotated_pivot_heatmap(
        p_pivot,
        title=title.replace("Wasserstein distance", "Wasserstein permutation p-value"),
        colorbar_label="Permutation p-value",
        out_path=plots_dir / f"{group_name}_wasserstein_p_value_heatmap.png",
        cmap_name=f"{HEATMAP_CMAP}_r",
        vmin=0.0,
        vmax=0.05,
        sig_pivot=sig_pivot,
        scientific_below=1e-4,
    )
    _render_annotated_pivot_heatmap(
        q_pivot,
        title=title.replace("Wasserstein distance", "Wasserstein FDR q-value"),
        colorbar_label="FDR q-value",
        out_path=plots_dir / f"{group_name}_wasserstein_q_value_fdr_heatmap.png",
        cmap_name=f"{HEATMAP_CMAP}_r",
        vmin=0.0,
        vmax=0.05,
        sig_pivot=sig_pivot,
        scientific_below=1e-4,
    )


def _write_subject_focused_distribution_plots(
    stats_df: pd.DataFrame,
    tables_dir: Path,
    plots_dir: Path,
    *,
    group_name: str,
    title_prefix: str,
    condition_codes: list[str] | tuple[str, ...] = ACTIVE_CONDITION_CODES,
    ytick_label_transform: Callable[[str], str] | None = None,
    color_overrides: dict[str, Any] | None = None,
) -> None:
    if stats_df.empty:
        return

    ordered_labels = [_condition_display_name(code) for code in condition_codes]
    subset = stats_df.copy()
    subset["target_condition_label"] = pd.Categorical(
        subset["target_condition_label"],
        categories=ordered_labels,
        ordered=True,
    )
    subset = subset.sort_values(["subject", "target_condition_label"]).reset_index(drop=True)
    if subset.empty:
        return

    sig_pivot = subset.pivot(index="subject", columns="target_condition_label", values="significant_fdr")
    sig_pivot = sig_pivot.reindex(columns=ordered_labels).sort_index()

    rank_df = subset.sort_values(
        ["subject", "wasserstein_distance", "target_condition_label"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    rank_df["wasserstein_rank_within_subject"] = (
        rank_df.groupby("subject", dropna=False, observed=False)["wasserstein_distance"]
        .rank(method="first", ascending=False)
        .astype(np.float64)
    )
    rank_pivot = rank_df.pivot(
        index="subject",
        columns="target_condition_label",
        values="wasserstein_rank_within_subject",
    )
    rank_pivot = rank_pivot.reindex(columns=ordered_labels).sort_index()
    rank_pivot.to_csv(tables_dir / f"{group_name}_wasserstein_rank_wide.csv")
    _render_annotated_pivot_heatmap(
        rank_pivot,
        title=f"{title_prefix}: within-subject rank of Wasserstein distance",
        colorbar_label="Rank (1 = most different from sham)",
        out_path=plots_dir / f"{group_name}_wasserstein_rank_heatmap.png",
        cmap_name="viridis_r",
        vmin=1.0,
        vmax=float(len(ordered_labels)),
        sig_pivot=sig_pivot,
        scientific_below=0.0,
        value_formatter=lambda value: f"{int(round(value))}",
    )

    subjects = rank_pivot.index.astype(str).tolist()
    n_subjects = len(subjects)
    n_cols = min(3, max(1, n_subjects))
    n_rows = (n_subjects + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.8 * n_cols, 3.0 * n_rows + 0.6),
        sharex=False,
        squeeze=False,
    )
    active_labels = [label for label in ordered_labels if label != "sham"]
    condition_colors = plt.get_cmap(HEATMAP_CMAP)(np.linspace(0.12, 0.88, len(active_labels)))
    color_lookup = dict(zip(active_labels, condition_colors, strict=True))
    if "sham" in ordered_labels:
        color_lookup["sham"] = "#000000"
    axes_flat = axes.ravel()
    for ax_idx, subject in enumerate(subjects):
        ax = axes_flat[ax_idx]
        subject_df = subset.loc[subset["subject"] == subject].copy()
        subject_df = subject_df.sort_values(
            ["wasserstein_distance", "target_condition_label"],
            ascending=[False, True],
        ).reset_index(drop=True)
        y_pos = np.arange(subject_df.shape[0])
        colors = [
            color_overrides.get(str(label), color_lookup[str(label)])
            if color_overrides is not None
            else color_lookup[str(label)]
            for label in subject_df["target_condition_label"]
        ]
        ytick_labels = [
            ytick_label_transform(str(label)) if ytick_label_transform is not None else str(label)
            for label in subject_df["target_condition_label"]
        ]
        bars = ax.barh(
            y_pos,
            subject_df["wasserstein_distance"].to_numpy(dtype=np.float64),
            color=colors,
            edgecolor="black",
            linewidth=0.6,
        )
        ax.set_yticks(y_pos)
        ax.set_yticklabels(ytick_labels)
        ax.invert_yaxis()
        ax.set_title(subject, fontsize=10)
        ax.set_xlabel("Wasserstein distance")
        ax.grid(axis="x", alpha=0.25)

        finite_values = subject_df["wasserstein_distance"].to_numpy(dtype=np.float64)
        finite_values = finite_values[np.isfinite(finite_values)]
        x_max = float(np.max(finite_values)) if finite_values.size else 1.0
        x_pad = max(0.02, 0.12 * x_max)
        ax.set_xlim(0.0, x_max + x_pad)

        for bar, row in zip(bars, subject_df.itertuples(index=False), strict=True):
            value = float(row.wasserstein_distance)
            label = f"{value:.3f}" + ("*" if bool(row.significant_fdr) else "")
            ax.text(
                value + 0.015 * max(x_max, 1e-6),
                bar.get_y() + bar.get_height() / 2.0,
                label,
                va="center",
                ha="left",
                fontsize=8,
            )

    for ax in axes_flat[n_subjects:]:
        ax.axis("off")

    fig.suptitle(f"{title_prefix}: per-subject Wasserstein ranking", fontsize=12)
    fig.text(0.5, 0.01, "* FDR < 0.05", ha="center", va="bottom", fontsize=9)
    fig.tight_layout(rect=[0.0, 0.03, 1.0, 0.96])
    fig.savefig(plots_dir / f"{group_name}_wasserstein_subject_barplots.png", dpi=200)
    plt.close(fig)


def _shift_within_medication_barplot_label(label: str) -> str:
    match = re.fullmatch(r"GVS(\d+)", str(label))
    if match is None:
        return str(label)
    condition_number = int(match.group(1))
    if condition_number <= 1:
        return "sham"
    return f"GVS{condition_number - 1}"


def _strip_relative_suffix(label: str) -> str:
    cleaned = str(label).replace(" (relative)", "")
    cleaned = cleaned.replace(" (Control & monitoring)", "")
    return cleaned


def _write_sham_roi_medication_ttest_outputs(
    stats_df: pd.DataFrame,
    tables_dir: Path,
    plots_dir: Path,
    *,
    roi_labels: list[str],
) -> None:
    if stats_df.empty:
        return

    ordered_labels = [str(label) for label in roi_labels]
    subset = stats_df.copy()
    subset["roi_label"] = pd.Categorical(subset["roi_label"], categories=ordered_labels, ordered=True)
    subset = subset.sort_values(["subject", "roi_label"]).reset_index(drop=True)

    delta_pivot = subset.pivot(index="subject", columns="roi_label", values="mean_delta_on_minus_off")
    delta_pivot = delta_pivot.reindex(columns=ordered_labels).sort_index()
    delta_pivot.to_csv(tables_dir / "sham_off_vs_on_roi_mean_delta_wide.csv")

    t_pivot = subset.pivot(index="subject", columns="roi_label", values="t_stat_welch")
    t_pivot = t_pivot.reindex(columns=ordered_labels).sort_index()
    t_pivot.to_csv(tables_dir / "sham_off_vs_on_roi_t_stat_wide.csv")

    p_pivot = subset.pivot(index="subject", columns="roi_label", values="p_value_two_sided")
    p_pivot = p_pivot.reindex(columns=ordered_labels).sort_index()
    p_pivot.to_csv(tables_dir / "sham_off_vs_on_roi_p_value_wide.csv")

    q_pivot = subset.pivot(index="subject", columns="roi_label", values="q_value_fdr")
    q_pivot = q_pivot.reindex(columns=ordered_labels).sort_index()
    q_pivot.to_csv(tables_dir / "sham_off_vs_on_roi_q_value_fdr_wide.csv")

    sig_pivot = subset.pivot(index="subject", columns="roi_label", values="significant_fdr")
    sig_pivot = sig_pivot.reindex(columns=ordered_labels).sort_index()
    sig_pivot.to_csv(tables_dir / "sham_off_vs_on_roi_significant_fdr_wide.csv")

    delta_values = delta_pivot.to_numpy(dtype=np.float64)
    delta_abs_max = float(np.nanmax(np.abs(delta_values))) if np.isfinite(delta_values).any() else 0.0
    delta_abs_max = max(delta_abs_max, 1e-6)
    _render_annotated_pivot_heatmap(
        delta_pivot,
        title="Sham ON - sham OFF ROI beta delta",
        colorbar_label="Mean beta delta (ON - OFF)",
        out_path=plots_dir / "sham_off_vs_on_roi_mean_delta_heatmap.png",
        cmap_name="coolwarm",
        vmin=-delta_abs_max,
        vmax=delta_abs_max,
        sig_pivot=sig_pivot,
        scientific_below=1e-3,
    )

    t_values = t_pivot.to_numpy(dtype=np.float64)
    t_abs_max = float(np.nanmax(np.abs(t_values))) if np.isfinite(t_values).any() else 0.0
    t_abs_max = max(t_abs_max, 1e-6)
    _render_annotated_pivot_heatmap(
        t_pivot,
        title="Sham ON vs sham OFF ROI Welch t-statistic",
        colorbar_label="Welch t-statistic",
        out_path=plots_dir / "sham_off_vs_on_roi_t_stat_heatmap.png",
        cmap_name="coolwarm",
        vmin=-t_abs_max,
        vmax=t_abs_max,
        sig_pivot=sig_pivot,
        scientific_below=1e-3,
    )

    _render_annotated_pivot_heatmap(
        p_pivot,
        title="Sham ON vs sham OFF ROI Welch p-value",
        colorbar_label="Two-sided p-value",
        out_path=plots_dir / "sham_off_vs_on_roi_p_value_heatmap.png",
        cmap_name=f"{HEATMAP_CMAP}_r",
        vmin=0.0,
        vmax=0.05,
        sig_pivot=sig_pivot,
        scientific_below=1e-4,
    )

    _render_annotated_pivot_heatmap(
        q_pivot,
        title="Sham ON vs sham OFF ROI Welch FDR q-value",
        colorbar_label="FDR q-value",
        out_path=plots_dir / "sham_off_vs_on_roi_q_value_fdr_heatmap.png",
        cmap_name=f"{HEATMAP_CMAP}_r",
        vmin=0.0,
        vmax=0.05,
        sig_pivot=sig_pivot,
        scientific_below=1e-4,
    )

    subject_plots_dir = ensure_dir(plots_dir / "sham_off_vs_on_roi_by_subject")
    _clear_directory_files(subject_plots_dir)
    for subject in delta_pivot.index.astype(str).tolist():
        subject_delta = delta_pivot.loc[[subject]]
        subject_sig = sig_pivot.loc[[subject]]
        _render_annotated_pivot_heatmap(
            subject_delta,
            title=f"{subject}: sham ON - sham OFF ROI beta delta",
            colorbar_label="Mean beta delta (ON - OFF)",
            out_path=subject_plots_dir / f"{subject}_sham_off_vs_on_roi_mean_delta_heatmap.png",
            cmap_name="coolwarm",
            vmin=-delta_abs_max,
            vmax=delta_abs_max,
            sig_pivot=subject_sig,
            scientific_below=1e-3,
        )


def _write_condition_roi_reference_delta_outputs(
    matrices: dict[tuple[str, int, str], np.ndarray],
    tables_dir: Path,
    plots_dir: Path,
    *,
    roi_labels: list[str],
) -> None:
    ordered_condition_labels = [_condition_display_name(code) for code in ALL_CONDITION_CODES]
    ordered_roi_labels = [str(label) for label in roi_labels]
    combined_rows: list[pd.DataFrame] = []
    for spec in ROI_REFERENCE_DELTA_SPECS:
        subject_df = build_condition_roi_reference_delta_rows(
            matrices,
            roi_labels,
            target_session=int(spec["target_session"]),
            reference_session=int(spec["reference_session"]),
            comparison_kind=str(spec["comparison_kind"]),
        )
        if subject_df.empty:
            continue
        file_prefix = str(spec["file_prefix"])
        combined_rows.append(subject_df)
        subject_df.to_csv(tables_dir / f"{file_prefix}_by_subject_long.csv", index=False)
        stats_df = compute_condition_roi_reference_ttest_stats(
            matrices,
            roi_labels,
            target_session=int(spec["target_session"]),
            reference_session=int(spec["reference_session"]),
            comparison_kind=str(spec["comparison_kind"]),
        )
        stats_df.to_csv(tables_dir / f"{file_prefix}_ttest_stats_by_subject_long.csv", index=False)
        _write_significant_roi_condition_table_png(
            stats_df,
            plots_dir / f"{file_prefix}_significant_rois_by_subject_table.png",
            title=f"Significant ROIs by subject and GVS: {str(spec['title'])} (FDR < 0.05)",
            condition_labels=ordered_condition_labels,
        )
        _write_significant_roi_condition_panel_png(
            stats_df,
            plots_dir / f"{file_prefix}_significant_rois_by_subject_panel.png",
            title=f"Significant ROIs by subject and GVS: {str(spec['title'])} (FDR < 0.05)",
            condition_labels=ordered_condition_labels,
            roi_labels=ordered_roi_labels,
        )
        delta_values_all = subject_df["mean_delta_target_minus_reference"].to_numpy(dtype=np.float64)
        delta_values_all = delta_values_all[np.isfinite(delta_values_all)]
        delta_abs_max = float(np.max(np.abs(delta_values_all))) if delta_values_all.size else 0.0
        delta_abs_max = max(delta_abs_max, 1e-6)

        summary_df = (
            subject_df.groupby(
                ["target_condition_code", "target_condition_label", "roi_index", "roi_label"],
                dropna=False,
                observed=False,
            )["mean_delta_target_minus_reference"]
            .agg(["mean", "std", "count"])
            .reset_index()
            .rename(
                columns={
                    "mean": "group_mean_delta_target_minus_reference",
                    "std": "group_std_delta_target_minus_reference",
                    "count": "n_subjects",
                }
            )
            .sort_values(["target_condition_code", "roi_index"])
            .reset_index(drop=True)
        )
        summary_df.to_csv(tables_dir / f"{file_prefix}_summary_long.csv", index=False)

        ordered_summary = summary_df.copy()
        ordered_summary["target_condition_label"] = pd.Categorical(
            ordered_summary["target_condition_label"],
            categories=ordered_condition_labels,
            ordered=True,
        )
        ordered_summary["roi_label"] = pd.Categorical(
            ordered_summary["roi_label"],
            categories=ordered_roi_labels,
            ordered=True,
        )
        ordered_summary = ordered_summary.sort_values(["target_condition_label", "roi_label"]).reset_index(drop=True)

        delta_pivot = ordered_summary.pivot(
            index="target_condition_label",
            columns="roi_label",
            values="group_mean_delta_target_minus_reference",
        )
        delta_pivot = delta_pivot.reindex(index=ordered_condition_labels, columns=ordered_roi_labels)
        delta_pivot.to_csv(tables_dir / f"{file_prefix}_wide.csv")

        count_pivot = ordered_summary.pivot(
            index="target_condition_label",
            columns="roi_label",
            values="n_subjects",
        )
        count_pivot = count_pivot.reindex(index=ordered_condition_labels, columns=ordered_roi_labels)
        count_pivot.to_csv(tables_dir / f"{file_prefix}_n_subjects_wide.csv")

        _render_annotated_pivot_heatmap(
            delta_pivot,
            title=str(spec["title"]),
            colorbar_label=str(spec["colorbar_label"]),
            out_path=plots_dir / f"{file_prefix}_heatmap.png",
            cmap_name="coolwarm",
            vmin=-delta_abs_max,
            vmax=delta_abs_max,
            scientific_below=1e-3,
        )

        subject_plots_dir = ensure_dir(plots_dir / f"{file_prefix}_by_subject")
        _clear_directory_files(subject_plots_dir)
        is_off_minus_sham_off = str(spec["comparison_kind"]) == "off_condition_minus_sham_off_roi_delta"
        subject_cmap_name = "bwr" if is_off_minus_sham_off else "coolwarm"
        subject_roi_labels = (
            [_strip_relative_suffix(label) for label in ordered_roi_labels]
            if is_off_minus_sham_off
            else ordered_roi_labels
        )
        for subject in sorted(subject_df["subject"].astype(str).unique().tolist()):
            subject_subset = subject_df.loc[subject_df["subject"] == subject].copy()
            subject_stats_subset = stats_df.loc[stats_df["subject"] == subject].copy()
            subject_subset["target_condition_label"] = pd.Categorical(
                subject_subset["target_condition_label"],
                categories=ordered_condition_labels,
                ordered=True,
            )
            subject_subset["roi_label"] = pd.Categorical(
                subject_subset["roi_label"],
                categories=ordered_roi_labels,
                ordered=True,
            )
            subject_subset = subject_subset.sort_values(["target_condition_label", "roi_label"]).reset_index(drop=True)
            subject_pivot = subject_subset.pivot(
                index="target_condition_label",
                columns="roi_label",
                values="mean_delta_target_minus_reference",
            )
            subject_pivot = subject_pivot.reindex(index=ordered_condition_labels, columns=ordered_roi_labels)
            subject_pivot.columns = subject_roi_labels
            subject_values = subject_pivot.to_numpy(dtype=np.float64)
            if np.isfinite(subject_values).any():
                subject_vmin = float(np.nanmin(subject_values))
                subject_vmax = float(np.nanmax(subject_values))
                if np.isclose(subject_vmin, subject_vmax):
                    pad = max(abs(subject_vmin), 1e-6) * 0.05
                    if np.isclose(pad, 0.0):
                        pad = 1e-6
                    subject_vmin -= pad
                    subject_vmax += pad
            else:
                subject_vmin = -1e-6
                subject_vmax = 1e-6
            subject_norm = None
            if is_off_minus_sham_off and subject_vmin < 0.0 < subject_vmax:
                subject_norm = mcolors.TwoSlopeNorm(vmin=subject_vmin, vcenter=0.0, vmax=subject_vmax)
            sig_pivot = None
            if not subject_stats_subset.empty:
                subject_stats_subset["target_condition_label"] = pd.Categorical(
                    subject_stats_subset["target_condition_label"],
                    categories=ordered_condition_labels,
                    ordered=True,
                )
                subject_stats_subset["roi_label"] = pd.Categorical(
                    subject_stats_subset["roi_label"],
                    categories=ordered_roi_labels,
                    ordered=True,
                )
                subject_stats_subset = subject_stats_subset.sort_values(
                    ["target_condition_label", "roi_label"]
                ).reset_index(drop=True)
                sig_pivot = subject_stats_subset.pivot(
                    index="target_condition_label",
                    columns="roi_label",
                    values="significant_fdr",
                )
                sig_pivot = sig_pivot.reindex(index=ordered_condition_labels, columns=ordered_roi_labels)
                sig_pivot.columns = subject_roi_labels
            subject_title = f"{subject}: {str(spec['title'])}"
            if is_off_minus_sham_off:
                subject_title = f"{subject}, Off condition-sham off"
            _render_annotated_pivot_heatmap(
                subject_pivot,
                title=subject_title,
                colorbar_label=str(spec["colorbar_label"]),
                out_path=subject_plots_dir / f"{subject}_{file_prefix}_heatmap.png",
                cmap_name=subject_cmap_name,
                vmin=subject_vmin,
                vmax=subject_vmax,
                norm=subject_norm,
                sig_pivot=sig_pivot,
                scientific_below=1e-3,
            )

    if combined_rows:
        pd.concat(combined_rows, ignore_index=True).to_csv(
            tables_dir / "condition_roi_reference_delta_by_subject_long.csv",
            index=False,
        )


def _plot_trial_metric_boxplots(
    subject_df: pd.DataFrame,
    subject_stats_df: pd.DataFrame,
    out_path: Path,
    *,
    analysis_spec: dict[str, Any],
) -> None:
    if subject_df.empty:
        return
    format_fields = {
        "subject": str(subject_df["subject"].iloc[0]),
        "target_medication": str(subject_df["target_medication"].iloc[0]) if "target_medication" in subject_df.columns else "",
        "reference_group_label": str(subject_df["reference_group_label"].iloc[0]) if "reference_group_label" in subject_df.columns else "reference",
    }
    ordered_codes = [code for code in ALL_CONDITION_CODES if code in set(subject_df["target_condition_code"])]
    ordered_labels = [_condition_display_name(code) for code in ordered_codes]
    panels: list[tuple[str, str, list[np.ndarray]]] = []
    for metric_spec in analysis_spec["summary_metrics"]:
        metric_name = str(metric_spec["column"])
        values_list: list[np.ndarray] = []
        for code in ordered_codes:
            condition_rows = subject_df.loc[subject_df["target_condition_code"] == code]
            values = condition_rows[metric_name].to_numpy(dtype=np.float64)
            values_list.append(values[np.isfinite(values)])
        panels.append((metric_name, str(metric_spec["plot_ylabel"]), values_list))

    colors = plt.get_cmap("jet")(np.linspace(0.1, 0.9, len(ordered_codes)))
    fig, axes = plt.subplots(1, len(panels), figsize=(max(10.0, 1.2 * len(ordered_codes) + 4.0), 4.8), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, (metric_name, ylabel, values_list) in zip(axes, panels, strict=True):
        box = ax.boxplot(values_list, patch_artist=True, tick_labels=ordered_labels, widths=0.65, showfliers=False)
        for patch, color in zip(box["boxes"], colors, strict=True):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        for median in box["medians"]:
            median.set_color("black")
            median.set_linewidth(1.4)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(str(analysis_spec.get("x_label_template", "OFF condition")).format(**format_fields))
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=0.25)
        flat_values = np.concatenate([values for values in values_list if values.size], axis=0) if any(
            values.size for values in values_list
        ) else np.array([], dtype=np.float64)
        if flat_values.size:
            y_min = float(np.nanmin(flat_values))
            y_max = float(np.nanmax(flat_values))
            y_range = max(0.05, y_max - y_min)
            y_text = y_max + 0.06 * y_range
            ax.set_ylim(bottom=y_min - 0.04 * y_range, top=y_max + 0.18 * y_range)
        else:
            y_text = 1.0
        for pos, code in enumerate(ordered_codes, start=1):
            if code == SHAM_CONDITION_CODE:
                continue
            stat_row = subject_stats_df.loc[
                (subject_stats_df["summary_metric"] == metric_name)
                & (subject_stats_df["condition_code"] == code)
            ]
            if stat_row.empty:
                continue
            p_value = float(stat_row["p_value"].iloc[0])
            if bool(stat_row["significant_fdr"].iloc[0]):
                text = "**"
            elif np.isfinite(p_value) and p_value < 0.05:
                text = "*"
            else:
                text = ""
            if text:
                ax.text(pos, y_text, text, ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.text(1, y_text, "ref", ha="center", va="bottom", fontsize=8)
    fig.suptitle(str(analysis_spec["title_template"]).format(**format_fields))
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_trial_distance_boxplots(
    subject_df: pd.DataFrame,
    subject_stats_df: pd.DataFrame,
    out_path: Path,
) -> None:
    _plot_trial_metric_boxplots(
        subject_df,
        subject_stats_df,
        out_path,
        analysis_spec=TRIAL_ANALYSIS_SPECS[DEFAULT_TRIAL_ANALYSIS_KEY],
    )


def write_report(
    out_dir: Path,
    roi_df: pd.DataFrame,
    inventory_df: pd.DataFrame,
    within_distribution_df: pd.DataFrame,
    within_distribution_stats_df: pd.DataFrame,
    off_to_on_distribution_df: pd.DataFrame,
    off_to_on_distribution_stats_df: pd.DataFrame,
    sham_off_to_on_roi_ttest_df: pd.DataFrame,
    *,
    n_permutations: int,
) -> None:
    lines: list[str] = []
    lines.append("# GVS Similarity Analysis")
    lines.append("")
    lines.append("## Setup")
    lines.append(f"- ROI beta matrices were built from saved `by_gvs` beta splits using {int(roi_df.shape[0])} ROI rows.")
    lines.append("- The `within_med_sham_reference` and `off_to_on_sham_reference` outputs use flattened voxel-by-trial beta distributions from the original selected-beta matrices.")
    lines.append(
        "- The `roi_condition_reference_deltas` outputs summarize ROI beta deltas as condition minus sham reference for OFF vs sham ON, OFF vs sham OFF, and ON vs sham ON."
    )
    lines.append("- Each comparison uses all finite beta values from the condition matrix, pooled across selected voxels and trials.")
    lines.append("")
    lines.append("## Coverage")
    lines.append(f"- Subject/session/condition beta matrices: {int(inventory_df.shape[0])}")
    lines.append(f"- Unique subjects with at least one beta matrix: {int(inventory_df['subject'].nunique()) if not inventory_df.empty else 0}")
    lines.append(f"- Within-medication sham-reference condition distributions: {int(within_distribution_df.shape[0])}")
    lines.append(f"- OFF-vs-ON sham-reference condition distributions: {int(off_to_on_distribution_df.shape[0])}")
    lines.append("")

    def _append_summary_block(title: str, df: pd.DataFrame) -> None:
        lines.append(f"## {title}")
        if df.empty:
            lines.append("- No rows available.")
            lines.append("")
            return
        summary = (
            df.groupby("target_condition_label", dropna=False, observed=False)[["target_median", "target_q25", "target_q75"]]
            .mean()
            .reset_index()
            .sort_values("target_median", ascending=False)
        )
        for row in summary.itertuples(index=False):
            lines.append(
                f"- {row.target_condition_label}: mean subject median beta={_format_value(row.target_median)}, "
                f"mean subject IQR=[{_format_value(row.target_q25)}, {_format_value(row.target_q75)}]"
            )
        lines.append("")

    _append_summary_block(
        "Within-Medication OFF vs OFF Sham Distribution",
        within_distribution_df.loc[within_distribution_df["target_medication"] == "OFF"].reset_index(drop=True)
        if not within_distribution_df.empty
        else pd.DataFrame(),
    )
    _append_summary_block(
        "Within-Medication ON vs ON Sham Distribution",
        within_distribution_df.loc[within_distribution_df["target_medication"] == "ON"].reset_index(drop=True)
        if not within_distribution_df.empty
        else pd.DataFrame(),
    )
    lines.append("## Within-Medication Distribution Tests")
    if within_distribution_stats_df.empty:
        lines.append("- No rows available.")
        lines.append("")
    else:
        lines.append(
            "- Main effect-size heatmap: Wasserstein distance between each active-condition beta distribution and the same-session sham beta distribution."
        )
        lines.append(
            f"- Inference: one-sided permutation p-values on Wasserstein distance ({int(n_permutations)} permutations), "
            "with FDR correction across the 8 active GVS conditions within each subject and medication state."
        )
        for medication in ("OFF", "ON"):
            medication_df = within_distribution_stats_df.loc[
                within_distribution_stats_df["target_medication"] == medication
            ].reset_index(drop=True)
            if medication_df.empty:
                lines.append(f"- {medication}: None")
                continue
            parts = []
            for subject in sorted(medication_df["subject"].astype(str).unique().tolist()):
                subject_df = medication_df.loc[medication_df["subject"] == subject].reset_index(drop=True)
                condition_label_column = (
                    "condition_label" if "condition_label" in subject_df.columns else "target_condition_label"
                )
                significant_conditions = (
                    subject_df.loc[subject_df["significant_fdr"], condition_label_column].astype(str).tolist()
                )
                parts.append(f"{subject}: {', '.join(significant_conditions) if significant_conditions else 'None'}")
            lines.append(f"- {medication}: " + " | ".join(parts))
        lines.append("")

    _append_summary_block("OFF Condition vs ON Sham Distribution", off_to_on_distribution_df)
    lines.append("## OFF-to-ON Sham Distribution Tests")
    if off_to_on_distribution_stats_df.empty:
        lines.append("- No rows available.")
        lines.append("")
    else:
        lines.append(
            "- Main effect-size heatmap: Wasserstein distance between each OFF active-condition beta distribution and the ON sham beta distribution."
        )
        lines.append(
            f"- Inference: one-sided permutation p-values on Wasserstein distance ({int(n_permutations)} permutations), "
            "with FDR correction across all 9 OFF conditions within each subject, including OFF sham vs ON sham."
        )
        lines.append(
            "- Subject-focused visualization: a faceted per-subject Wasserstein bar plot and a within-subject rank heatmap are saved alongside the heatmaps."
        )
        parts = []
        for subject in sorted(off_to_on_distribution_stats_df["subject"].astype(str).unique().tolist()):
            subject_df = off_to_on_distribution_stats_df.loc[
                off_to_on_distribution_stats_df["subject"] == subject
            ].reset_index(drop=True)
            condition_label_column = (
                "condition_label" if "condition_label" in subject_df.columns else "target_condition_label"
            )
            significant_conditions = (
                subject_df.loc[subject_df["significant_fdr"], condition_label_column].astype(str).tolist()
            )
            parts.append(f"{subject}: {', '.join(significant_conditions) if significant_conditions else 'None'}")
        lines.append("- " + " | ".join(parts))
        lines.append("")

    lines.append("## Sham OFF vs ON ROI Trial-Vector Tests")
    if sham_off_to_on_roi_ttest_df.empty:
        lines.append("- No rows available.")
        lines.append("")
    else:
        lines.append(
            "- New ROI-level medication-effect analysis: within each subject, the sham OFF ROI trial vector was compared with the sham ON ROI trial vector using a two-sided Welch t-test."
        )
        lines.append(
            "- Multiple-comparison correction: FDR across ROI rows within each subject."
        )
        parts = []
        for subject in sorted(sham_off_to_on_roi_ttest_df["subject"].astype(str).unique().tolist()):
            subject_df = sham_off_to_on_roi_ttest_df.loc[
                sham_off_to_on_roi_ttest_df["subject"] == subject
            ].reset_index(drop=True)
            significant_rois = subject_df.loc[subject_df["significant_fdr"], "roi_label"].astype(str).tolist()
            parts.append(f"{subject}: {', '.join(significant_rois) if significant_rois else 'None'}")
        lines.append("- " + " | ".join(parts))
        lines.append("")

    lines.append("## Additional Trial-Level Outputs")
    lines.append(
        "- `trial_correlation_to_on_sham`: same OFF-vs-ON sham workflow using trial-wise Pearson correlation, summarized with mean and max similarity."
    )
    lines.append(
        "- `trial_frobenius_distance_to_on_sham`: same OFF-vs-ON sham workflow using trial-wise Frobenius distance, summarized with mean and min distance."
    )
    lines.append("")

    (out_dir / "report.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    by_gvs_dir = args.by_gvs_dir.expanduser().resolve()
    selected_voxels_path = args.selected_voxels_path.expanduser().resolve()
    roi_img_path = args.roi_img.expanduser().resolve()
    roi_summary_path = args.roi_summary.expanduser().resolve()
    out_dir = ensure_dir(args.out_dir.expanduser().resolve())
    if int(args.target_trials) <= 0:
        raise ValueError(f"--target-trials must be positive, got {args.target_trials}.")
    if int(args.n_permutations) <= 0:
        raise ValueError(f"--n-permutations must be positive, got {args.n_permutations}.")

    roi_members, roi_labels, _ = build_roi_membership(
        roi_img_path=roi_img_path,
        roi_summary_path=roi_summary_path,
        selected_voxels_path=selected_voxels_path,
        min_roi_voxels=int(args.min_roi_voxels),
    )
    if not roi_members:
        raise ValueError("No ROI members passed the requested threshold.")

    common_dir = ensure_dir(out_dir / "common")
    within_dir = ensure_dir(out_dir / "within_med_sham_reference")
    off_to_on_dir = ensure_dir(out_dir / "off_to_on_sham_reference")
    roi_reference_dir = ensure_dir(out_dir / "roi_condition_reference_deltas")
    matrix_dir = ensure_dir(common_dir / "matrices")
    roi_df = pd.DataFrame(
        {
            "roi_index": np.arange(1, len(roi_labels) + 1, dtype=np.int64),
            "roi_label": roi_labels,
            "n_selected_voxels": [int(members.size) for members in roi_members],
        }
    )
    roi_df.to_csv(common_dir / "roi_nodes.csv", index=False)

    matrices, inventory_df = load_roi_beta_matrices(
        by_gvs_dir=by_gvs_dir,
        roi_members=roi_members,
        matrix_dir=matrix_dir,
    )
    inventory_df.to_csv(common_dir / "matrix_inventory.csv", index=False)
    beta_distributions = load_beta_value_distributions(by_gvs_dir=by_gvs_dir)

    within_df = build_within_medication_distribution_summary(beta_distributions)
    within_stats_df = compute_within_medication_distribution_stats(
        beta_distributions,
        n_permutations=int(args.n_permutations),
        random_seed=int(args.random_seed),
    )
    within_tables_dir = ensure_dir(within_dir / "tables")
    within_plots_dir = ensure_dir(within_dir / "plots")
    _clear_directory_files(within_tables_dir)
    _clear_directory_files(within_plots_dir)
    _write_distribution_summary(
        within_df,
        within_tables_dir / "condition_distribution_summary.csv",
    )
    within_stats_df.to_csv(within_tables_dir / "gvs_vs_sham_distribution_stats.csv", index=False)
    _write_significant_condition_summary(
        within_stats_df,
        within_tables_dir / "significant_conditions_by_subject.csv",
    )
    _write_distribution_heatmap(
        within_stats_df,
        within_tables_dir,
        within_plots_dir,
        group_name="off_vs_off_sham",
        title="OFF: Wasserstein distance for flattened beta distribution vs OFF sham",
        medication="OFF",
    )
    _write_subject_focused_distribution_plots(
        within_stats_df.loc[within_stats_df["target_medication"] == "OFF"].reset_index(drop=True),
        within_tables_dir,
        within_plots_dir,
        group_name="off_vs_off_sham",
        title_prefix="OFF vs OFF sham",
        ytick_label_transform=_shift_within_medication_barplot_label,
        color_overrides={"GVS1": "#000000"},
    )
    _write_distribution_heatmap(
        within_stats_df,
        within_tables_dir,
        within_plots_dir,
        group_name="on_vs_on_sham",
        title="ON: Wasserstein distance for flattened beta distribution vs ON sham",
        medication="ON",
    )
    _write_subject_focused_distribution_plots(
        within_stats_df.loc[within_stats_df["target_medication"] == "ON"].reset_index(drop=True),
        within_tables_dir,
        within_plots_dir,
        group_name="on_vs_on_sham",
        title_prefix="ON vs ON sham",
        ytick_label_transform=_shift_within_medication_barplot_label,
        color_overrides={"GVS1": "#000000"},
    )

    off_to_on_df = build_off_to_on_distribution_summary(beta_distributions)
    off_to_on_stats_df = compute_off_to_on_distribution_stats(
        beta_distributions,
        n_permutations=int(args.n_permutations),
        random_seed=int(args.random_seed),
    )
    off_to_on_tables_dir = ensure_dir(off_to_on_dir / "tables")
    off_to_on_plots_dir = ensure_dir(off_to_on_dir / "plots")
    _clear_directory_files(off_to_on_tables_dir)
    _clear_directory_files(off_to_on_plots_dir)
    _write_distribution_summary(
        off_to_on_df,
        off_to_on_tables_dir / "condition_distribution_summary.csv",
    )
    off_to_on_stats_df.to_csv(off_to_on_tables_dir / "gvs_vs_sham_distribution_stats.csv", index=False)
    _write_significant_condition_summary(
        off_to_on_stats_df,
        off_to_on_tables_dir / "significant_conditions_by_subject.csv",
    )
    _write_distribution_heatmap(
        off_to_on_stats_df,
        off_to_on_tables_dir,
        off_to_on_plots_dir,
        group_name="off_condition_vs_on_sham",
        title="OFF: Wasserstein distance for flattened beta distribution vs ON sham",
        condition_codes=ALL_CONDITION_CODES,
    )
    _write_subject_focused_distribution_plots(
        off_to_on_stats_df,
        off_to_on_tables_dir,
        off_to_on_plots_dir,
        group_name="off_condition_vs_on_sham",
        title_prefix="OFF vs ON sham",
        condition_codes=ALL_CONDITION_CODES,
    )
    sham_off_to_on_roi_ttest_df = compute_sham_off_to_on_roi_ttest_stats(
        matrices,
        roi_labels=roi_labels,
    )
    sham_off_to_on_roi_ttest_df.to_csv(
        off_to_on_tables_dir / "sham_off_vs_on_roi_ttest_stats.csv",
        index=False,
    )
    _write_significant_roi_summary(
        sham_off_to_on_roi_ttest_df,
        off_to_on_tables_dir / "sham_off_vs_on_roi_significant_summary.csv",
    )
    _write_sham_roi_medication_ttest_outputs(
        sham_off_to_on_roi_ttest_df,
        off_to_on_tables_dir,
        off_to_on_plots_dir,
        roi_labels=roi_labels,
    )

    roi_reference_tables_dir = ensure_dir(roi_reference_dir / "tables")
    roi_reference_plots_dir = ensure_dir(roi_reference_dir / "plots")
    _clear_directory_tree(roi_reference_tables_dir)
    _clear_directory_tree(roi_reference_plots_dir)
    _write_condition_roi_reference_delta_outputs(
        matrices,
        roi_reference_tables_dir,
        roi_reference_plots_dir,
        roi_labels=roi_labels,
    )

    trial_analysis_results: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for analysis_key, analysis_spec in TRIAL_ANALYSIS_SPECS.items():
        trial_dir = ensure_dir(out_dir / str(analysis_spec["folder_name"]))
        trial_tables_dir = ensure_dir(trial_dir / "tables")
        trial_plots_dir = ensure_dir(trial_dir / "plots")
        trial_df = build_trial_metric_to_on_sham_rows(matrices=matrices, analysis_spec=analysis_spec)
        trial_df.to_csv(trial_tables_dir / "trial_level_similarity.csv", index=False)
        _write_trial_metric_summary(
            trial_df,
            trial_tables_dir / "subject_condition_summary.csv",
            metric_columns=_trial_summary_columns(analysis_spec),
        )
        trial_stats_df = compute_trial_gvs_vs_sham_stats(trial_df, analysis_spec=analysis_spec)
        trial_stats_df.to_csv(trial_tables_dir / "gvs_vs_sham_within_subject_trial_stats.csv", index=False)
        for subject, subject_df in trial_df.groupby("subject", sort=True):
            _plot_trial_metric_boxplots(
                subject_df=subject_df.reset_index(drop=True),
                subject_stats_df=trial_stats_df.loc[trial_stats_df["subject"] == subject].reset_index(drop=True),
                out_path=trial_plots_dir / f"{subject}_{analysis_spec['plot_filename_suffix']}",
                analysis_spec=analysis_spec,
            )
        trial_analysis_results[analysis_key] = (trial_df, trial_stats_df)

    trial_distance_df, trial_distance_stats_df = trial_analysis_results.get(
        DEFAULT_TRIAL_ANALYSIS_KEY,
        (pd.DataFrame(), pd.DataFrame()),
    )

    manifest = {
        "by_gvs_dir": str(by_gvs_dir),
        "selected_voxels_path": str(selected_voxels_path),
        "roi_img": str(roi_img_path),
        "roi_summary": str(roi_summary_path),
        "out_dir": str(out_dir),
        "n_rois": int(len(roi_members)),
        "target_trials": int(args.target_trials),
        "n_permutations": int(args.n_permutations),
        "random_seed": int(args.random_seed),
        "primary_metric": PRIMARY_METRIC,
        "secondary_metrics": list(SECONDARY_METRICS),
        "sham_reference_distribution_analysis": {
            "display_name": FLATTENED_BETA_DISTRIBUTION_LABEL,
            "source": "selected_beta_trials_gvs-XX.npy",
            "value_pool": "all finite beta values across selected voxels and trials",
            "main_effect_size": "Wasserstein distance",
            "primary_test": "one-sided permutation test on Wasserstein distance",
            "wasserstein_permutation_bins": int(DEFAULT_WASSERSTEIN_PERMUTATION_BINS),
            "reported_values": ["wasserstein_distance", "wasserstein_p_value", "q_value_fdr"],
        },
        "sham_roi_medication_ttest": {
            "display_name": SHAM_ROI_MEDICATION_TTEST_LABEL,
            "source": "ROI beta matrices derived from selected_beta_trials_gvs-01.npy",
            "comparison": "subject-wise sham OFF ROI row vs sham ON ROI row",
            "trial_handling": "raw per-session trial vectors (no resampling)",
            "primary_test": "two-sided Welch t-test",
            "multiple_comparison_correction": "FDR across ROI rows within each subject",
            "reported_values": [
                "mean_delta_on_minus_off",
                "t_stat_welch",
                "p_value_two_sided",
                "q_value_fdr",
                "significant_fdr",
            ],
        },
        "condition_roi_reference_deltas": {
            "folder_name": "roi_condition_reference_deltas",
            "source": "ROI beta matrices derived from selected_beta_trials_gvs-XX.npy",
            "summary_level": "subject-average ROI mean beta, then group mean across subjects",
            "reported_value": "target condition mean beta minus reference sham mean beta",
            "cellwise_test": "two-sided Welch t-test on ROI trial vectors",
            "multiple_comparison_correction": "FDR across ROI-by-condition cells within each subject and comparison",
            "comparisons": [str(spec["comparison_kind"]) for spec in ROI_REFERENCE_DELTA_SPECS],
        },
        "trial_level_analyses": [
            {
                "key": analysis_key,
                "folder_name": str(analysis_spec["folder_name"]),
                "display_name": str(analysis_spec["display_name"]),
                "pairwise_metric": str(analysis_spec["pairwise_metric"]),
                "summary_metrics": _trial_summary_columns(analysis_spec),
            }
            for analysis_key, analysis_spec in TRIAL_ANALYSIS_SPECS.items()
        ],
    }
    (out_dir / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_report(
        out_dir=out_dir,
        roi_df=roi_df,
        inventory_df=inventory_df,
        within_distribution_df=within_df,
        within_distribution_stats_df=within_stats_df,
        off_to_on_distribution_df=off_to_on_df,
        off_to_on_distribution_stats_df=off_to_on_stats_df,
        sham_off_to_on_roi_ttest_df=sham_off_to_on_roi_ttest_df,
        n_permutations=int(args.n_permutations),
    )


if __name__ == "__main__":
    main()
