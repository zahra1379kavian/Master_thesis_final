#!/usr/bin/env python3
"""Build GVS sham-reference ROI burden heatmaps for the new beta dataset."""

from __future__ import annotations

import argparse
import json
import re
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore", message="Unable to import Axes3D.*")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import ticker
import nibabel as nib
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection

from med_effects import (
    DEFAULT_AAL_VERSION,
    DEFAULT_ATLAS_CACHE_DIR,
    DEFAULT_MIN_REPORT_VOXELS,
    _analysis_roi_setup,
    _build_weighted_rois,
    _default_region_table_for,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHT_MAP = ROOT / "data" / "voxel_weights_task1_bold0.6_beta0.6_smooth1.25_gamma1.5.nii.gz"
DEFAULT_ROI_FIGURE = (
    ROOT / "figures" / "voxel_weights_task1_bold0.6_beta0.6_smooth1.25_gamma1.5_threshold_robustness_atlas_regions.png"
)
DEFAULT_GVS_ORDER = ROOT / "data" / "gvs_order_by_subject_session_run.tsv"
DEFAULT_BETA_ROOT = Path(
    "/mnt/TeamShare/Data_Masterfile/H20-00572_All-Dressed/Zahra-Thesis-Data/fmri_opt_group/results_beta_preprocessed"
)
DEFAULT_OUT_DIR = ROOT / "figures" / "GVS_effects" / "gvs_similarity_hemi"
DEFAULT_PROJECTED_SIGNAL_OUT_DIR = ROOT / "figures" / "GVS_effects"
DEFAULT_MOTOR_REWARD_ROI_PROFILE_OUT_DIR = ROOT / "figures" / "GVS_effects"
DEFAULT_LME_TRIAL_CSV = ROOT / "Claude_results" / "group_analyses" / "analysis3_lme" / "per_trial_roi_betas.csv"
DEFAULT_ROI_PERCENTILE = 90.0
DEFAULT_PROJECTED_SIGNAL_WEIGHT_PERCENTILE = 90.0
DEFAULT_TRIALS_PER_CONDITION = 10
DEFAULT_ALPHA = 0.05
PROJECTED_SIGNAL_MED_COLORS = {"OFF": "#2F6FAE", "ON": "#D87924"}
PROJECTED_SIGNAL_STIM_SHORT = {
    1: "Sham",
    2: "Pink\nnoise",
    3: "DC+1",
    4: "DC-1",
    5: "Delta",
    6: "Theta",
    7: "Alpha",
    8: "Beta",
    9: "Gamma",
}
ROI_PROFILE_STIM_SHORT = {
    "GVS1": "Pink\nnoise",
    "GVS2": "DC+1",
    "GVS3": "DC-1",
    "GVS4": "Delta",
    "GVS5": "Theta",
    "GVS6": "Alpha",
    "GVS7": "Beta",
    "GVS8": "Gamma",
}
ROI_PROFILE_ACTIVE_LABELS = tuple(ROI_PROFILE_STIM_SHORT)
MOTOR_REWARD_ROI_SELECTION: tuple[tuple[str, str], ...] = (
    ("Precentral_L", "motor_sensorimotor"),
    ("Precentral_R", "motor_sensorimotor"),
    ("Supp_Motor_Area_L", "motor_sensorimotor"),
    ("Supp_Motor_Area_R", "motor_sensorimotor"),
    ("Postcentral_L", "motor_sensorimotor"),
    ("Postcentral_R", "motor_sensorimotor"),
    ("Paracentral_Lobule_L", "motor_sensorimotor"),
    ("Paracentral_Lobule_R", "motor_sensorimotor"),
    ("Caudate_L", "basal_ganglia_thalamus"),
    ("Caudate_R", "basal_ganglia_thalamus"),
    ("Putamen_L", "basal_ganglia_thalamus"),
    ("Putamen_R", "basal_ganglia_thalamus"),
    ("Thalamus_L", "basal_ganglia_thalamus"),
    ("Thalamus_R", "basal_ganglia_thalamus"),
    ("Orbitofrontal_L", "reward_limbic"),
    ("Orbitofrontal_R", "reward_limbic"),
    ("Cingulate_L", "reward_limbic"),
    ("Cingulate_R", "reward_limbic"),
    ("Amygdala_L", "reward_limbic"),
    ("Amygdala_R", "reward_limbic"),
    ("Hippocampus_L", "reward_limbic"),
    ("Hippocampus_R", "reward_limbic"),
)
ROI_CELL_COLORS = (
    "#D7E9F7",
    "#F8E3D0",
    "#DDECCB",
    "#E6DDF3",
    "#F7D9E5",
    "#D8EFE8",
    "#FFF0B8",
    "#E2E2E2",
    "#D7ECF2",
    "#F1D9C7",
    "#DFE4FA",
    "#EADBCB",
)

BETA_FILE_RE = re.compile(r"^cleaned_beta_volume_(sub-pd\d+)_ses-(\d+)_run-(\d+)\.npy$")
SUBJECT_RE = re.compile(r"^sub-pd(\d+)$", re.IGNORECASE)
SESSION_RE = re.compile(r"^GVS(\d+)$", re.IGNORECASE)
SHAM_CONDITION_CODE = "gvs-01"
CONDITION_CODES = tuple(f"gvs-{index:02d}" for index in range(1, 10))
ROI_REFERENCE_SPECS: tuple[dict[str, Any], ...] = (
    {
        "comparison_kind": "off_condition_minus_sham_off_roi_delta",
        "target_session": 1,
        "reference_session": 1,
        "file_prefix": "off_condition_minus_sham_off_roi_mean_delta",
    },
    {
        "comparison_kind": "on_condition_minus_sham_on_roi_delta",
        "target_session": 2,
        "reference_session": 2,
        "file_prefix": "on_condition_minus_sham_on_roi_mean_delta",
    },
)


@dataclass(frozen=True)
class RunSpec:
    subject: str
    session: int
    run: int
    beta_path: Path
    stim_order: tuple[int, ...]
    n_trials: int


@dataclass(frozen=True)
class ROIArray:
    name: str
    flat_indices: np.ndarray
    weights: np.ndarray
    n_voxels: int


def _coerce_user_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    match = re.match(r"^(?P<drive>[A-Za-z]):[\\/](?P<rest>.*)$", text)
    if match is not None:
        rest = [part for part in re.split(r"[\\/]+", match.group("rest")) if part]
        if match.group("drive").upper() == "M":
            return Path("/mnt/TeamShare").joinpath(*rest)
        return Path("/mnt").joinpath(match.group("drive").upper(), *rest)
    return Path(text).expanduser()


def _resolve_path(path: str | Path | None) -> Path | None:
    coerced = _coerce_user_path(path)
    if coerced is None:
        return None
    if coerced.is_absolute():
        return coerced
    return (ROOT / coerced).resolve()


def _subject_sort_key(subject: str) -> tuple[int, str]:
    match = SUBJECT_RE.match(str(subject))
    return (int(match.group(1)), str(subject)) if match else (10**9, str(subject))


def _session_sort_key(label: str) -> tuple[int, str]:
    match = SESSION_RE.match(str(label))
    return (int(match.group(1)), str(label)) if match else (10**9, str(label))


def _condition_display_name(code: str) -> str:
    if code == SHAM_CONDITION_CODE:
        return "sham"
    match = re.match(r"^gvs-(\d+)$", str(code))
    if match is None:
        return str(code)
    condition_number = int(match.group(1))
    return f"GVS{condition_number - 1}" if condition_number >= 2 else str(code)


def _medication_from_session(session: int) -> str:
    return "OFF" if int(session) == 1 else "ON" if int(session) == 2 else f"ses-{session}"


def _normalize_subject_id(subject_id: str) -> str:
    text = str(subject_id).strip()
    if text.startswith("sub-pd"):
        return text
    match = re.search(r"(\d+)$", text)
    if match is None:
        raise ValueError(f"Could not parse subject id from {subject_id!r}.")
    return f"sub-pd{match.group(1).zfill(3)}"


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.fillna(False).astype(str).str.casefold().isin({"true", "1", "yes"})


def _iter_condition_slices(
    n_trials: int,
    stim_order: tuple[int, ...],
    trials_per_condition: int,
) -> list[tuple[str, int, int]]:
    slices: list[tuple[str, int, int]] = []
    for block_index, stim_number in enumerate(stim_order):
        start = int(block_index * trials_per_condition)
        if start >= int(n_trials):
            break
        stop = min(start + int(trials_per_condition), int(n_trials))
        slices.append((f"gvs-{int(stim_number):02d}", start, stop))
    expected_trials = len(stim_order) * int(trials_per_condition)
    if int(n_trials) > expected_trials:
        warnings.warn(
            f"Run has {n_trials} trials but only {expected_trials} are described by the GVS order; "
            "extra trailing trials are ignored.",
            RuntimeWarning,
            stacklevel=2,
        )
    return slices


def _load_gvs_order(path: Path) -> dict[tuple[str, int, int], tuple[int, ...]]:
    order_df = pd.read_csv(path, sep="\t")
    required = {"subject_id", "session_index", "run", "true_stim_order", "status"}
    missing = sorted(required - set(order_df.columns))
    if missing:
        raise RuntimeError(f"{path} is missing required columns: {', '.join(missing)}")

    rows = order_df.loc[order_df["status"].astype(str).str.casefold().eq("ok")].copy()
    order_map: dict[tuple[str, int, int], tuple[int, ...]] = {}
    for row in rows.itertuples(index=False):
        subject = _normalize_subject_id(getattr(row, "subject_id"))
        session = int(getattr(row, "session_index"))
        run = int(getattr(row, "run"))
        order = tuple(int(item.strip()) for item in str(getattr(row, "true_stim_order")).split(",") if item.strip())
        if not order:
            continue
        order_map[(subject, session, run)] = order
    if not order_map:
        raise RuntimeError(f"No usable GVS order rows found in {path}.")
    return order_map


def _discover_run_specs(
    beta_root: Path,
    gvs_order: dict[tuple[str, int, int], tuple[int, ...]],
    subjects: list[str] | None,
) -> list[RunSpec]:
    keep_subjects = {_normalize_subject_id(subject) for subject in subjects} if subjects else None
    specs: list[RunSpec] = []
    missing_order: list[str] = []

    for beta_path in sorted(beta_root.glob("sub-pd*/cleaned_beta_volume_sub-pd*_ses-*_run-*.npy")):
        match = BETA_FILE_RE.match(beta_path.name)
        if match is None:
            continue
        subject = str(match.group(1))
        session = int(match.group(2))
        run = int(match.group(3))
        if keep_subjects is not None and subject not in keep_subjects:
            continue
        key = (subject, session, run)
        stim_order = gvs_order.get(key)
        if stim_order is None:
            missing_order.append(f"{subject} ses-{session} run-{run}")
            continue
        beta = np.load(beta_path, mmap_mode="r")
        if beta.ndim != 4:
            raise RuntimeError(f"{beta_path} must be a 4D beta volume, got shape {beta.shape}.")
        specs.append(
            RunSpec(
                subject=subject,
                session=session,
                run=run,
                beta_path=beta_path,
                stim_order=stim_order,
                n_trials=int(beta.shape[-1]),
            )
        )

    if missing_order:
        raise RuntimeError("Missing GVS order rows for beta runs: " + "; ".join(missing_order))
    if not specs:
        raise RuntimeError(f"No cleaned beta volumes found under {beta_root}.")
    return sorted(specs, key=lambda spec: (_subject_sort_key(spec.subject), spec.session, spec.run))


def _build_roi_arrays(rois: list[Any], volume_shape: tuple[int, int, int]) -> list[ROIArray]:
    roi_arrays: list[ROIArray] = []
    for roi in rois:
        flat_indices = np.flatnonzero(np.asarray(roi.mask, dtype=bool).ravel()).astype(np.int64, copy=False)
        weights = np.asarray(roi.weights, dtype=np.float64)
        if flat_indices.size != weights.size:
            raise RuntimeError(
                f"ROI {roi.name} has {flat_indices.size} voxels but {weights.size} weights on grid {volume_shape}."
            )
        roi_arrays.append(
            ROIArray(
                name=str(roi.name),
                flat_indices=flat_indices,
                weights=weights,
                n_voxels=int(flat_indices.size),
            )
        )
    return roi_arrays


def _weighted_roi_trial_matrix(
    beta_path: Path,
    reference_shape: tuple[int, int, int],
    rois: list[ROIArray],
) -> np.ndarray:
    beta = np.load(beta_path, mmap_mode="r")
    if beta.ndim != 4:
        raise RuntimeError(f"{beta_path} must be a 4D beta volume, got shape {beta.shape}.")
    if tuple(beta.shape[:3]) != tuple(reference_shape):
        raise RuntimeError(f"{beta_path} shape {beta.shape[:3]} differs from weight-map shape {reference_shape}.")

    n_trials = int(beta.shape[-1])
    flat_view = beta.reshape(-1, n_trials)
    roi_matrix = np.full((len(rois), n_trials), np.nan, dtype=np.float64)
    for roi_index, roi in enumerate(rois):
        roi_data = np.asarray(flat_view[roi.flat_indices, :], dtype=np.float64)
        finite = np.isfinite(roi_data)
        weights = roi.weights[:, None]
        weighted = np.where(finite, roi_data, 0.0) * weights
        denom = np.sum(np.where(finite, weights, 0.0), axis=0)
        valid = denom > 0
        roi_matrix[roi_index, valid] = np.sum(weighted[:, valid], axis=0) / denom[valid]
    return roi_matrix


def _load_condition_matrices(
    run_specs: list[RunSpec],
    rois: list[ROIArray],
    reference_shape: tuple[int, int, int],
    trials_per_condition: int,
) -> tuple[dict[tuple[str, int, str], np.ndarray], pd.DataFrame]:
    condition_parts: dict[tuple[str, int, str], list[np.ndarray]] = defaultdict(list)
    inventory_rows: list[dict[str, Any]] = []
    expected_trials = 9 * int(trials_per_condition)

    for run_index, spec in enumerate(run_specs, start=1):
        print(
            f"Loading ROI beta trials {run_index}/{len(run_specs)}: "
            f"{spec.subject} ses-{spec.session} run-{spec.run}",
            flush=True,
        )
        roi_matrix = _weighted_roi_trial_matrix(spec.beta_path, reference_shape, rois)
        for condition_code, start, stop in _iter_condition_slices(
            spec.n_trials,
            spec.stim_order,
            int(trials_per_condition),
        ):
            condition_parts[(spec.subject, spec.session, condition_code)].append(roi_matrix[:, start:stop])
            inventory_rows.append(
                {
                    "subject": spec.subject,
                    "session": int(spec.session),
                    "medication": _medication_from_session(spec.session),
                    "run": int(spec.run),
                    "condition_code": condition_code,
                    "condition_label": _condition_display_name(condition_code),
                    "trial_start": int(start),
                    "trial_stop": int(stop),
                    "n_trials": int(stop - start),
                    "run_n_trials": int(spec.n_trials),
                    "expected_run_trials": int(expected_trials),
                    "stim_order": ",".join(str(item) for item in spec.stim_order),
                    "source_beta_path": str(spec.beta_path),
                }
            )

    matrices = {
        key: np.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]
        for key, parts in condition_parts.items()
        if parts
    }
    inventory_df = pd.DataFrame(inventory_rows)
    if not inventory_df.empty:
        inventory_df = inventory_df.sort_values(["subject", "session", "run", "condition_code"]).reset_index(drop=True)
    return matrices, inventory_df


def _finite_values(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return array[np.isfinite(array)]


def _finite_mean(values: np.ndarray) -> float:
    finite = _finite_values(values)
    return float(np.mean(finite)) if finite.size else float("nan")


def _add_groupwise_fdr(
    stats_df: pd.DataFrame,
    *,
    p_value_column: str,
    group_columns: list[str],
    alpha: float,
) -> pd.DataFrame:
    out = stats_df.copy()
    out["q_value_fdr"] = np.nan
    out["significant_fdr"] = False
    if out.empty:
        return out

    for _, group_index in out.groupby(group_columns, dropna=False, sort=False).groups.items():
        p_values = out.loc[group_index, p_value_column].to_numpy(dtype=np.float64)
        finite = np.isfinite(p_values)
        if not np.any(finite):
            continue
        rejected, q_values = fdrcorrection(p_values[finite], alpha=float(alpha))
        finite_indices = np.asarray(group_index)[finite]
        out.loc[finite_indices, "q_value_fdr"] = q_values
        out.loc[finite_indices, "significant_fdr"] = rejected
    return out


def _stim_id_from_condition_code(condition_code: str) -> int:
    match = re.match(r"^gvs-(\d+)$", str(condition_code))
    if match is None:
        raise ValueError(f"Could not parse GVS condition code {condition_code!r}.")
    return int(match.group(1))


def _projected_signal_stim_label(stim_id: int) -> str:
    return PROJECTED_SIGNAL_STIM_SHORT.get(int(stim_id), str(stim_id))


def _projected_signal_weight_mask(
    weight_values: np.ndarray,
    weight_percentile: float | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    weight_flat = np.asarray(weight_values, dtype=np.float32).ravel()
    positive = np.isfinite(weight_flat) & (weight_flat > 0)
    if not np.any(positive):
        raise RuntimeError("Projected-signal analysis requires at least one positive finite weight.")

    threshold = float("nan")
    if weight_percentile is None:
        mask = positive
    else:
        threshold = float(np.percentile(weight_flat[positive], float(weight_percentile)))
        mask = positive & (weight_flat >= threshold)
    if not np.any(mask):
        raise RuntimeError("Projected-signal analysis selected zero weight-map voxels.")

    metadata = {
        "weight_percentile": weight_percentile,
        "weight_threshold": threshold,
        "n_positive_weight_voxels": int(np.sum(positive)),
        "n_projected_signal_voxels": int(np.sum(mask)),
    }
    return mask, weight_flat[mask], metadata


def _load_projected_signal_condition_means(
    run_specs: list[RunSpec],
    weight_values: np.ndarray,
    reference_shape: tuple[int, int, int],
    *,
    trials_per_condition: int,
    weight_percentile: float | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    mask, masked_weights, metadata = _projected_signal_weight_mask(weight_values, weight_percentile)
    rows: list[dict[str, Any]] = []

    for run_index, spec in enumerate(run_specs, start=1):
        print(
            f"Projecting beta trials {run_index}/{len(run_specs)}: "
            f"{spec.subject} ses-{spec.session} run-{spec.run}",
            flush=True,
        )
        beta = np.load(spec.beta_path, mmap_mode="r")
        if beta.ndim != 4:
            raise RuntimeError(f"{spec.beta_path} must be a 4D beta volume, got shape {beta.shape}.")
        if tuple(beta.shape[:3]) != tuple(reference_shape):
            raise RuntimeError(f"{spec.beta_path} shape {beta.shape[:3]} differs from weight-map shape {reference_shape}.")

        n_trials = int(beta.shape[-1])
        flat_view = beta.reshape(-1, n_trials)
        projected = np.nansum(masked_weights[:, None] * flat_view[mask, :], axis=0).astype(np.float32)

        for condition_code, start, stop in _iter_condition_slices(
            n_trials,
            spec.stim_order,
            int(trials_per_condition),
        ):
            stim_id = _stim_id_from_condition_code(condition_code)
            condition_values = projected[start:stop]
            finite = condition_values[np.isfinite(condition_values)]
            rows.append(
                {
                    "subject": spec.subject,
                    "session": int(spec.session),
                    "medication": _medication_from_session(spec.session),
                    "run": int(spec.run),
                    "stim_id": int(stim_id),
                    "stim_short": _projected_signal_stim_label(stim_id),
                    "n_proj": int(finite.size),
                    "mean_proj": float(np.mean(finite)) if finite.size else float("nan"),
                }
            )

    projected_df = pd.DataFrame(rows)
    if not projected_df.empty:
        projected_df = projected_df.sort_values(["subject", "session", "run", "stim_id"]).reset_index(drop=True)
    return projected_df, metadata


def _analyze_projected_signal(projected_df: pd.DataFrame, *, alpha: float) -> pd.DataFrame:
    if projected_df.empty:
        return pd.DataFrame()

    subject_condition_df = (
        projected_df.groupby(["subject", "session", "medication", "stim_id", "stim_short"], dropna=False)
        .agg(mean_proj=("mean_proj", "mean"))
        .reset_index()
    )
    sham = (
        subject_condition_df.loc[subject_condition_df["stim_id"].astype(int).eq(1)]
        .set_index(["subject", "session"])["mean_proj"]
        .rename("sham_proj")
    )
    subject_condition_df = subject_condition_df.join(sham, on=["subject", "session"])
    subject_condition_df["delta_proj"] = subject_condition_df["mean_proj"] - subject_condition_df["sham_proj"]

    rows: list[dict[str, Any]] = []
    active = subject_condition_df.loc[~subject_condition_df["stim_id"].astype(int).eq(1)].copy()
    for (stim_id, medication), group_df in active.groupby(["stim_id", "medication"], dropna=False, sort=True):
        deltas = group_df["delta_proj"].dropna().to_numpy(dtype=np.float64)
        n_subjects = int(deltas.size)
        if n_subjects < 3:
            continue

        test_result = stats.ttest_1samp(deltas, 0.0, nan_policy="omit")
        sem = float(stats.sem(deltas))
        ci_low, ci_high = stats.t.interval(
            0.95,
            df=n_subjects - 1,
            loc=float(np.mean(deltas)),
            scale=sem,
        )
        std = float(np.std(deltas, ddof=1))
        rows.append(
            {
                "stim_id": int(stim_id),
                "stim_short": _projected_signal_stim_label(int(stim_id)),
                "medication": str(medication),
                "n_subjects": n_subjects,
                "mean_delta_proj": float(np.mean(deltas)),
                "se_delta_proj": sem,
                "ci95_lo": float(ci_low),
                "ci95_hi": float(ci_high),
                "cohens_d": float(np.mean(deltas) / std) if std > 0 else float("nan"),
                "t_stat": float(test_result.statistic) if np.isfinite(test_result.statistic) else float("nan"),
                "p_value": float(test_result.pvalue) if np.isfinite(test_result.pvalue) else float("nan"),
            }
        )

    projected_stats = pd.DataFrame(rows)
    if projected_stats.empty:
        return projected_stats

    projected_stats["q_fdr"] = np.nan
    projected_stats["sig_fdr"] = False
    for medication, group_index in projected_stats.groupby("medication", dropna=False, sort=False).groups.items():
        p_values = projected_stats.loc[group_index, "p_value"].to_numpy(dtype=np.float64)
        finite = np.isfinite(p_values)
        if not np.any(finite):
            continue
        rejected, q_values = fdrcorrection(p_values[finite], alpha=float(alpha))
        finite_indices = np.asarray(group_index)[finite]
        projected_stats.loc[finite_indices, "q_fdr"] = q_values
        projected_stats.loc[finite_indices, "sig_fdr"] = rejected

    return projected_stats.sort_values(["medication", "stim_id"]).reset_index(drop=True)


def _add_projected_signal_sig_stars(ax: plt.Axes, x_value: float, y_value: float, p_value: float) -> None:
    if not np.isfinite(p_value):
        return
    if p_value < 0.001:
        label = "***"
    elif p_value < 0.01:
        label = "**"
    elif p_value < 0.05:
        label = "*"
    else:
        return
    ax.text(
        x_value,
        y_value,
        label,
        ha="center",
        va="bottom",
        fontsize=11,
        color="#c0392b",
        fontweight="bold",
    )


def _write_projected_signal_profile(
    projected_stats: pd.DataFrame,
    out_dir: Path,
) -> tuple[Path, Path]:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    for ax, medication in zip(np.atleast_1d(axes), ["OFF", "ON"], strict=True):
        subset = projected_stats.loc[projected_stats["medication"].astype(str).eq(medication)].sort_values("stim_id")
        if subset.empty:
            ax.text(0.5, 0.5, f"No {medication} projected-signal data", ha="center", va="center", fontsize=11)
            ax.axis("off")
            continue

        x_values = np.arange(len(subset))
        bar_colors = [
            PROJECTED_SIGNAL_MED_COLORS[medication] if float(row.p_value) < 0.05 else "#cccccc"
            for row in subset.itertuples(index=False)
        ]
        ax.bar(
            x_values,
            subset["mean_delta_proj"],
            yerr=subset["se_delta_proj"],
            color=bar_colors,
            edgecolor="k",
            linewidth=0.7,
            error_kw={"elinewidth": 1.2, "capsize": 4},
            alpha=0.85,
        )
        ax.axhline(0, color="k", linewidth=0.9)
        ax.set_xticks(x_values)
        ax.set_xticklabels(subset["stim_short"], fontsize=9)
        ax.set_xlabel("GVS condition", fontsize=11)
        ax.set_title(
            f"Medication {medication}",
            fontsize=12,
            fontweight="bold",
            color=PROJECTED_SIGNAL_MED_COLORS[medication],
        )
        ax.set_ylabel("Projected signal delta vs sham (a.u.)" if medication == "OFF" else "", fontsize=10)
        for x_value, row in zip(x_values, subset.itertuples(index=False), strict=True):
            y_value = max(float(row.mean_delta_proj) + float(row.se_delta_proj) + 0.005, 0.005)
            _add_projected_signal_sig_stars(ax, float(x_value), y_value, float(row.p_value))
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.grid(axis="y", alpha=0.3, linestyle="--")

    fig.suptitle(
        "Projected signal delta vs sham per GVS condition\n"
        "(P = <weight, beta> using top-10% weight voxels [bold_thr90])",
        fontsize=12,
    )
    fig.tight_layout()
    paths = _save_pdf_and_png(fig, out_dir / "B_projected_signal_profile.pdf", dpi=200)
    plt.close(fig)
    return paths


def _write_projected_signal_outputs(
    run_specs: list[RunSpec],
    weight_values: np.ndarray,
    reference_shape: tuple[int, int, int],
    out_dir: Path,
    *,
    trials_per_condition: int,
    weight_percentile: float | None,
    alpha: float,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    projected_df, mask_metadata = _load_projected_signal_condition_means(
        run_specs,
        weight_values,
        reference_shape,
        trials_per_condition=int(trials_per_condition),
        weight_percentile=weight_percentile,
    )
    if projected_df.empty:
        raise RuntimeError("Projected-signal analysis produced no condition rows.")

    projected_stats = _analyze_projected_signal(projected_df, alpha=float(alpha))
    if projected_stats.empty:
        raise RuntimeError("Projected-signal analysis produced no active-condition statistics.")

    raw_csv_path = out_dir / "B_projected_signal_raw.csv"
    stats_csv_path = out_dir / "B_projected_signal_results.csv"
    projected_df.to_csv(raw_csv_path, index=False)
    projected_stats.to_csv(stats_csv_path, index=False)
    pdf_path, png_path = _write_projected_signal_profile(projected_stats, out_dir)

    return {
        "projected_signal_raw_csv": raw_csv_path,
        "projected_signal_results_csv": stats_csv_path,
        "projected_signal_profile_pdf": pdf_path,
        "projected_signal_profile_png": png_path,
        "projected_signal_mask": mask_metadata,
    }


def _roi_profile_star_label(p_value: float) -> str:
    if not np.isfinite(p_value):
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def _compute_motor_reward_roi_profiles(
    trial_df: pd.DataFrame,
    rois: list[str],
    *,
    alpha: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for roi in rois:
        if roi not in trial_df.columns:
            continue

        mean_df = (
            trial_df.groupby(["subject", "session", "medication", "condition_label"])[roi]
            .mean()
            .reset_index(name="mean_beta")
        )
        sham = (
            mean_df.loc[mean_df["condition_label"].astype(str).eq("sham")]
            .set_index(["subject", "session"])["mean_beta"]
            .rename("sham_beta")
        )
        mean_df = mean_df.join(sham, on=["subject", "session"])
        mean_df["delta_beta"] = mean_df["mean_beta"] - mean_df["sham_beta"]
        active = mean_df.loc[mean_df["condition_label"].isin(ROI_PROFILE_ACTIVE_LABELS)].copy()

        for (condition, medication), cell_df in active.groupby(["condition_label", "medication"], sort=False):
            deltas = cell_df["delta_beta"].dropna().to_numpy(dtype=np.float64)
            if deltas.size < 3:
                continue
            test_result = stats.ttest_1samp(deltas, 0.0)
            p_value = float(test_result.pvalue)
            rows.append(
                {
                    "roi": roi,
                    "medication": str(medication),
                    "condition": str(condition),
                    "stim_short": ROI_PROFILE_STIM_SHORT[str(condition)].replace("\n", " "),
                    "n_subjects": int(deltas.size),
                    "mean_delta": float(np.mean(deltas)),
                    "se_delta": float(stats.sem(deltas)),
                    "t_stat": float(test_result.statistic),
                    "p_value": p_value,
                    "sig_uncorrected": bool(p_value < float(alpha)),
                }
            )

    results = pd.DataFrame(rows)
    if results.empty:
        return results

    results["q_fdr_within_roi_med"] = np.nan
    results["sig_fdr_within_roi_med"] = False
    for _, group_index in results.groupby(["roi", "medication"]).groups.items():
        p_values = results.loc[group_index, "p_value"].to_numpy(dtype=np.float64)
        rejected, q_values = fdrcorrection(p_values, alpha=float(alpha))
        results.loc[group_index, "q_fdr_within_roi_med"] = q_values
        results.loc[group_index, "sig_fdr_within_roi_med"] = rejected

    results["q_fdr_global_med"] = np.nan
    results["sig_fdr_global_med"] = False
    for _, group_index in results.groupby("medication").groups.items():
        p_values = results.loc[group_index, "p_value"].to_numpy(dtype=np.float64)
        rejected, q_values = fdrcorrection(p_values, alpha=float(alpha))
        results.loc[group_index, "q_fdr_global_med"] = q_values
        results.loc[group_index, "sig_fdr_global_med"] = rejected

    roi_order = {roi: index for index, roi in enumerate(rois)}
    condition_order = {condition: index for index, condition in enumerate(ROI_PROFILE_ACTIVE_LABELS)}
    return (
        results.assign(
            _roi_order=results["roi"].map(roi_order),
            _condition_order=results["condition"].map(condition_order),
        )
        .sort_values(["_roi_order", "medication", "_condition_order"])
        .drop(columns=["_roi_order", "_condition_order"])
        .reset_index(drop=True)
    )


def _write_motor_reward_roi_profile_plot(
    results: pd.DataFrame,
    rois: list[str],
    out_dir: Path,
    *,
    alpha: float,
) -> tuple[Path, Path]:
    significant_rois = set(results.loc[results["sig_uncorrected"], "roi"])
    present_rois = [roi for roi in rois if roi in significant_rois]
    if not present_rois:
        raise RuntimeError("No motor/reward ROIs had an uncorrected significant GVS condition.")

    roi_to_group = dict(MOTOR_REWARD_ROI_SELECTION)
    group_bg = {
        "motor_sensorimotor": "#eef5fb",
        "basal_ganglia_thalamus": "#fdf3e7",
        "reward_limbic": "#edf7ea",
    }
    group_display = {
        "motor_sensorimotor": "Sensorimotor",
        "basal_ganglia_thalamus": "Basal Ganglia\n& Thalamus",
        "reward_limbic": "Reward\n& Limbic",
    }
    stim_labels = [
        ROI_PROFILE_STIM_SHORT[c].replace("\n", " ") for c in ROI_PROFILE_ACTIVE_LABELS
    ]

    # Build ordered list of unique base ROI names (strip _L / _R suffix)
    present_base_names: list[str] = []
    seen_bases: set[str] = set()
    for roi in present_rois:
        base = re.sub(r"_[LR]$", "", roi)
        if base not in seen_bases:
            seen_bases.add(base)
            present_base_names.append(base)

    n_rows = len(present_base_names)
    # 4 columns: L-OFF, L-ON, R-OFF, R-ON
    COL_CONFIG: list[tuple[str, str]] = [("L", "OFF"), ("L", "ON"), ("R", "OFF"), ("R", "ON")]

    with plt.rc_context({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica Neue", "Helvetica", "DejaVu Sans"],
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
    }):
        fig, axes = plt.subplots(
            n_rows, 4,
            figsize=(10.0, n_rows * 1.3 + 1.8),
            sharey=False,
            squeeze=False,
        )
        fig.patch.set_facecolor("white")

        prev_group: str = ""
        for row_idx, base_name in enumerate(present_base_names):
            group = roi_to_group.get(f"{base_name}_L", roi_to_group.get(f"{base_name}_R", ""))
            bg = group_bg.get(group, "#f8f8f8")
            is_new_group = group != prev_group

            for col_idx, (hemi, medication) in enumerate(COL_CONFIG):
                roi = f"{base_name}_{hemi}"
                ax = axes[row_idx][col_idx]
                ax.set_facecolor(bg)

                subset = (
                    results
                    .loc[results["roi"].eq(roi) & results["medication"].eq(medication)]
                    .set_index("condition")
                    .reindex(ROI_PROFILE_ACTIVE_LABELS)
                    .reset_index()
                )
                x_values = np.arange(len(subset))
                bar_colors = [
                    PROJECTED_SIGNAL_MED_COLORS[medication] if bool(sig) else "#d0d0d0"
                    for sig in subset["sig_uncorrected"].fillna(False)
                ]
                ax.bar(
                    x_values, subset["mean_delta"],
                    yerr=subset["se_delta"],
                    color=bar_colors,
                    edgecolor="#333333", linewidth=0.35,
                    error_kw={"elinewidth": 0.8, "capsize": 2.0, "ecolor": "#444444"},
                    alpha=0.90, width=0.68,
                )
                ax.axhline(0, color="#555555", linewidth=0.6, zorder=0)
                ax.set_xticks(x_values)
                ax.set_xticklabels(stim_labels, rotation=35, ha="right", fontsize=7)

                if col_idx == 0:
                    ax.set_ylabel(base_name.replace("_", " "), fontsize=8.5, labelpad=3)

                finite_vals = pd.concat(
                    [subset["mean_delta"].abs(), subset["se_delta"].abs()]
                ).dropna()
                y_range = max(float(finite_vals.max()) if not finite_vals.empty else 0.01, 0.01)
                ax.set_ylim(-1.45 * y_range, 1.65 * y_range)
                star_off = 0.1 * y_range

                for x_val, row in subset.iterrows():
                    lbl = _roi_profile_star_label(float(row["p_value"])) if pd.notna(row["p_value"]) else ""
                    if not lbl:
                        continue
                    y_base = float(row["mean_delta"])
                    se = float(row["se_delta"]) if pd.notna(row["se_delta"]) else 0.0
                    sign = 1.0 if y_base >= 0 else -1.0
                    ax.text(
                        x_val, y_base + sign * (se + star_off), lbl,
                        ha="center", va="bottom" if sign > 0 else "top",
                        fontsize=9, color="#c0392b", fontweight="bold",
                    )

                ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.45, zorder=0)
                for spine in ("top", "right"):
                    ax.spines[spine].set_visible(False)
                for spine in ("bottom", "left"):
                    ax.spines[spine].set_linewidth(0.6)

                # Rotated group label at the first ROI of each anatomical group (last column)
                if col_idx == 3 and is_new_group and group in group_display:
                    ax.annotate(
                        group_display[group],
                        xy=(1.04, 0.5), xycoords="axes fraction",
                        fontsize=7.5, va="center", ha="left",
                        rotation=270, color="#555555", style="italic",
                        annotation_clip=False,
                    )

            prev_group = group

        # Shared y-axis label
        fig.text(
            0.01, 0.5, r"$\Delta$ Beta vs. sham",
            va="center", ha="center", rotation=90,
            fontsize=9, color="#333333",
        )
        fig.suptitle(
            "Motor / Reward-circuit ROI GVS Effects",
            fontsize=11, fontweight="bold", y=0.995,
        )
        fig.tight_layout(rect=(0.04, 0, 0.92, 0.90), h_pad=0.4, w_pad=0.6)

        # All column headers placed in figure coordinates (no set_title clash)
        import matplotlib.lines as mlines

        axes_top = axes[0][0].get_position().y1
        med_y = axes_top + 0.012     # "Med. OFF / ON" tier
        hemi_y = axes_top + 0.048   # "Left / Right Hemisphere" tier

        for col_idx, (hemi, med) in enumerate(COL_CONFIG):
            x_col = (axes[0][col_idx].get_position().x0 + axes[0][col_idx].get_position().x1) / 2
            fig.text(
                x_col, med_y,
                f"Med. {'OFF' if med == 'OFF' else 'ON'}",
                ha="center", va="bottom",
                fontsize=9, fontweight="bold",
                color=PROJECTED_SIGNAL_MED_COLORS[med],
            )

        for hemi_label, col_a, col_b in [("Left Hemisphere", 0, 1), ("Right Hemisphere", 2, 3)]:
            x_mid = (axes[0][col_a].get_position().x0 + axes[0][col_b].get_position().x1) / 2
            fig.text(
                x_mid, hemi_y, hemi_label,
                ha="center", va="bottom",
                fontsize=10, fontweight="bold", color="#333333",
            )

        # Vertical separator between L columns (0–1) and R columns (2–3)
        x_left = axes[0][1].get_position().x1
        x_right = axes[0][2].get_position().x0
        x_sep = (x_left + x_right) / 2
        fig.add_artist(mlines.Line2D(
            [x_sep, x_sep], [0.02, hemi_y + 0.02],
            transform=fig.transFigure,
            color="#aaaaaa", linewidth=0.9, linestyle="--",
        ))

        paths = _save_pdf_and_png(fig, out_dir / "C_motor_reward_roi_gvs_profiles.pdf", dpi=220)
        plt.close(fig)
        return paths


def _write_motor_reward_roi_profile_outputs(
    lme_trial_csv: Path,
    out_dir: Path,
    *,
    alpha: float,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rois = [roi for roi, _ in MOTOR_REWARD_ROI_SELECTION]
    trial_df = pd.read_csv(lme_trial_csv)
    results = _compute_motor_reward_roi_profiles(trial_df, rois, alpha=float(alpha))
    if results.empty:
        raise RuntimeError("Motor/reward ROI profile analysis produced no result rows.")

    results_csv_path = out_dir / "C_motor_reward_roi_gvs_profiles.csv"
    selection_csv_path = out_dir / "C_motor_reward_roi_selection.csv"
    results.to_csv(results_csv_path, index=False)
    pd.DataFrame(MOTOR_REWARD_ROI_SELECTION, columns=["roi", "circuit_group"]).to_csv(selection_csv_path, index=False)
    pdf_path, png_path = _write_motor_reward_roi_profile_plot(results, rois, out_dir, alpha=float(alpha))

    return {
        "motor_reward_roi_profile_results_csv": results_csv_path,
        "motor_reward_roi_selection_csv": selection_csv_path,
        "motor_reward_roi_profile_pdf": pdf_path,
        "motor_reward_roi_profile_png": png_path,
        "motor_reward_roi_profile_summary": {
            "selected_rois": int(len(rois)),
            "tests": int(len(results)),
            "uncorrected_significant_cells": int(results["sig_uncorrected"].sum()),
            "within_roi_med_fdr_significant_cells": int(results["sig_fdr_within_roi_med"].sum()),
        },
    }


def compute_condition_roi_reference_stats(
    matrices: dict[tuple[str, int, str], np.ndarray],
    roi_labels: list[str],
    *,
    target_session: int,
    reference_session: int,
    comparison_kind: str,
    alpha: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    subjects = sorted({subject for subject, _, _ in matrices}, key=_subject_sort_key)
    for subject in subjects:
        reference_matrix = matrices.get((subject, int(reference_session), SHAM_CONDITION_CODE))
        if reference_matrix is None:
            continue
        for condition_code in CONDITION_CODES:
            target_matrix = matrices.get((subject, int(target_session), condition_code))
            if target_matrix is None:
                continue
            n_roi_rows = int(min(target_matrix.shape[0], reference_matrix.shape[0], len(roi_labels)))
            for roi_index in range(n_roi_rows):
                target_values = _finite_values(target_matrix[roi_index])
                reference_values = _finite_values(reference_matrix[roi_index])
                target_mean = float(np.mean(target_values)) if target_values.size else float("nan")
                reference_mean = float(np.mean(reference_values)) if reference_values.size else float("nan")
                mean_delta = (
                    float(target_mean - reference_mean)
                    if np.isfinite(target_mean) and np.isfinite(reference_mean)
                    else float("nan")
                )

                t_stat = float("nan")
                p_value = float("nan")
                if condition_code == SHAM_CONDITION_CODE and int(target_session) == int(reference_session):
                    if target_values.size >= 2 and reference_values.size >= 2:
                        t_stat = 0.0
                        p_value = 1.0
                elif target_values.size >= 2 and reference_values.size >= 2:
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
                        "target_medication": _medication_from_session(target_session),
                        "target_condition_code": condition_code,
                        "target_condition_label": _condition_display_name(condition_code),
                        "reference_session": int(reference_session),
                        "reference_medication": _medication_from_session(reference_session),
                        "reference_condition_code": SHAM_CONDITION_CODE,
                        "reference_condition_label": _condition_display_name(SHAM_CONDITION_CODE),
                        "roi_index": int(roi_index + 1),
                        "roi_label": str(roi_labels[roi_index]),
                        "n_trials_target": int(target_values.size),
                        "n_trials_reference": int(reference_values.size),
                        "mean_target": target_mean,
                        "mean_reference": reference_mean,
                        "mean_delta_target_minus_reference": mean_delta,
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
        alpha=float(alpha),
    )


def _write_reference_tables(
    stats_df: pd.DataFrame,
    tables_dir: Path,
    file_prefix: str,
    roi_labels: list[str],
) -> None:
    if stats_df.empty:
        return
    stats_df.to_csv(tables_dir / f"{file_prefix}_ttest_stats_by_subject_long.csv", index=False)
    mean_columns = [
        "comparison_kind",
        "subject",
        "target_session",
        "target_medication",
        "target_condition_code",
        "target_condition_label",
        "reference_session",
        "reference_medication",
        "reference_condition_code",
        "reference_condition_label",
        "roi_index",
        "roi_label",
        "mean_target",
        "mean_reference",
        "mean_delta_target_minus_reference",
    ]
    stats_df.loc[:, mean_columns].to_csv(tables_dir / f"{file_prefix}_by_subject_long.csv", index=False)

    summary_df = (
        stats_df.groupby(
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

    ordered_condition_labels = [_condition_display_name(code) for code in CONDITION_CODES]
    ordered_roi_labels = [str(label) for label in roi_labels]
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
    delta_pivot = ordered_summary.pivot(
        index="target_condition_label",
        columns="roi_label",
        values="group_mean_delta_target_minus_reference",
    ).reindex(index=ordered_condition_labels, columns=ordered_roi_labels)
    delta_pivot.to_csv(tables_dir / f"{file_prefix}_wide.csv")

    count_pivot = ordered_summary.pivot(
        index="target_condition_label",
        columns="roi_label",
        values="n_subjects",
    ).reindex(index=ordered_condition_labels, columns=ordered_roi_labels)
    count_pivot.to_csv(tables_dir / f"{file_prefix}_n_subjects_wide.csv")


def _gvs_counts_from_mask(
    stats_df: pd.DataFrame,
    row_mask: pd.Series,
    subjects: list[str],
    sessions: list[str],
) -> np.ndarray:
    selected = stats_df.loc[row_mask].copy()
    if selected.empty:
        return np.zeros((len(subjects), len(sessions)), dtype=float)
    selected["subject"] = selected["subject"].astype(str)
    selected["target_condition_label"] = selected["target_condition_label"].astype(str)
    selected["roi_label"] = selected["roi_label"].astype(str)
    roi_sets = {(subject, session): set() for subject in subjects for session in sessions}
    for (subject, session), cell_df in selected.groupby(
        ["subject", "target_condition_label"],
        dropna=False,
        observed=False,
        sort=False,
    ):
        key = (str(subject), str(session))
        if key in roi_sets:
            roi_sets[key] = set(cell_df["roi_label"].tolist())
    return np.array([[len(roi_sets[(subject, session)]) for session in sessions] for subject in subjects], dtype=float)


def _gvs_counts(stats_df: pd.DataFrame, subjects: list[str], sessions: list[str]) -> np.ndarray:
    return _gvs_counts_from_mask(stats_df, _as_bool(stats_df["significant_fdr"]), subjects, sessions)


def _converged_raw_p_only_mask(stats_df: pd.DataFrame, *, alpha: float) -> pd.Series:
    required = {"p_value_two_sided", "significant_fdr", "model_converged"}
    if not required.issubset(stats_df.columns):
        return pd.Series(False, index=stats_df.index)
    p_values = pd.to_numeric(stats_df["p_value_two_sided"], errors="coerce")
    return p_values.lt(float(alpha)) & ~_as_bool(stats_df["significant_fdr"]) & _as_bool(stats_df["model_converged"])


def _draw_gvs_panel(
    ax: plt.Axes,
    counts: np.ndarray,
    subjects: list[str],
    sessions: list[str],
    cmap_name: str,
    vmax: int,
    show_ylabel: bool,
    annotate: bool = True,
) -> plt.AxesImage:
    image = ax.imshow(counts, aspect="auto", interpolation="nearest", cmap=cmap_name, vmin=0, vmax=max(1, vmax))
    if show_ylabel:
        ax.set_ylabel("Subject", fontsize=11)
    ax.set_xticks(np.arange(len(sessions)))
    ax.set_xticklabels(sessions, fontsize=10)
    ax.set_yticks(np.arange(len(subjects)))
    ax.set_yticklabels(subjects, fontsize=10)
    ax.set_xticks(np.arange(-0.5, len(sessions), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(subjects), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.1)
    ax.tick_params(which="minor", bottom=False, left=False)
    if not annotate:
        return image
    threshold = max(1.0, 0.62 * float(max(1, vmax)))
    for row_idx in range(len(subjects)):
        for col_idx in range(len(sessions)):
            value = int(counts[row_idx, col_idx])
            if value == 0:
                text, color = "-", "#7a7a7a"
            else:
                text, color = str(value), "white" if value >= threshold else "black"
            ax.text(col_idx, row_idx, text, ha="center", va="center", fontsize=10, color=color)
    return image


def _save_pdf_and_png(fig: plt.Figure, pdf_path: Path, dpi: int = 300) -> tuple[Path, Path]:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    png_path = pdf_path.with_suffix(".png")
    fig.savefig(pdf_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    return pdf_path, png_path


def _write_burden_heatmaps(
    stats_by_prefix: dict[str, pd.DataFrame],
    plots_dir: Path,
    *,
    include_converged_raw_p: bool = False,
    raw_p_alpha: float = DEFAULT_ALPHA,
) -> tuple[Path, Path]:
    specs = [
        ("off_condition_minus_sham_off_roi_mean_delta", "Blues"),
        ("on_condition_minus_sham_on_roi_mean_delta", "Oranges"),
    ]
    sessions = sorted(
        {
            str(label)
            for stats_df in stats_by_prefix.values()
            for label in stats_df.get("target_condition_label", pd.Series(dtype=str)).dropna().unique().tolist()
            if str(label).casefold() != "sham"
        },
        key=_session_sort_key,
    )
    if not sessions:
        raise RuntimeError("No active GVS condition labels were available for burden heatmaps.")

    plots_dir.mkdir(parents=True, exist_ok=True)
    prepared: list[tuple[list[str], np.ndarray, str, np.ndarray, np.ndarray | None]] = []
    max_count = 0
    for prefix, cmap_name in specs:
        stats_df = stats_by_prefix.get(prefix, pd.DataFrame())
        if stats_df.empty:
            continue
        subjects = sorted(stats_df["subject"].astype(str).unique().tolist(), key=_subject_sort_key)
        fdr_counts = _gvs_counts(stats_df, subjects, sessions)
        counts = fdr_counts
        raw_counts = None
        if include_converged_raw_p:
            raw_counts = _gvs_counts_from_mask(
                stats_df,
                _converged_raw_p_only_mask(stats_df, alpha=float(raw_p_alpha)),
                subjects,
                sessions,
            )
            counts = fdr_counts + raw_counts
        max_count = max(max_count, int(np.nanmax(counts)) if counts.size else 0)
        prepared.append((subjects, counts, cmap_name, fdr_counts, raw_counts))
        pd.DataFrame(counts.astype(int), index=subjects, columns=sessions).rename_axis("subject").to_csv(
            plots_dir / f"{prefix}_subject_session_roi_burden_heatmap.csv"
        )
        if raw_counts is not None:
            pd.DataFrame(fdr_counts.astype(int), index=subjects, columns=sessions).rename_axis("subject").to_csv(
                plots_dir / f"{prefix}_subject_session_roi_burden_heatmap_fdr.csv"
            )
            pd.DataFrame(raw_counts.astype(int), index=subjects, columns=sessions).rename_axis("subject").to_csv(
                plots_dir / f"{prefix}_subject_session_roi_burden_heatmap_converged_raw_p_only.csv"
            )

    if not prepared:
        raise RuntimeError("No non-empty stats tables were available for burden heatmaps.")

    fig_w = max(15.0, 1.7 * len(sessions) + 4.0)
    fig_h = max(7.4, max(0.48 * len(subjects) for subjects, _, _, _, _ in prepared) + 2.1)
    fig, axes = plt.subplots(1, len(prepared), figsize=(fig_w, fig_h))
    for idx, (ax, (subjects, counts, cmap_name, fdr_counts, raw_counts)) in enumerate(zip(np.atleast_1d(axes), prepared)):
        image = _draw_gvs_panel(
            ax,
            counts,
            subjects,
            sessions,
            cmap_name,
            max_count,
            show_ylabel=idx == 0,
            annotate=raw_counts is None,
        )
        if raw_counts is not None:
            threshold = max(1.0, 0.62 * float(max(1, max_count)))
            for row_idx in range(len(subjects)):
                for col_idx in range(len(sessions)):
                    raw_value = int(raw_counts[row_idx, col_idx])
                    fdr_value = int(fdr_counts[row_idx, col_idx])
                    total_value = int(counts[row_idx, col_idx])
                    if total_value == 0:
                        text, color = "-", "#7a7a7a"
                    elif fdr_value and raw_value:
                        text = f"{fdr_value}*+{raw_value}"
                        color = "white" if total_value >= threshold else "black"
                    elif fdr_value:
                        text = f"{fdr_value}*"
                        color = "white" if total_value >= threshold else "black"
                    else:
                        text = str(raw_value)
                        color = "white" if total_value >= threshold else "black"
                    ax.text(col_idx, row_idx, text, ha="center", va="center", fontsize=8.5, color=color)
        ax.set_xlabel("GVS session", fontsize=11)
        colorbar = fig.colorbar(image, ax=ax, shrink=0.84, pad=0.02)
        if idx != 0:
            label = "No. of shown ROIs" if include_converged_raw_p else "No. of significant ROIs"
            colorbar.set_label(label, fontsize=10)
        colorbar.ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        colorbar.ax.tick_params(labelsize=9)
    if include_converged_raw_p:
        fig.text(
            0.5,
            0.01,
            "* = FDR-significant ROI count; unstarred counts are model-converged raw p < 0.05, FDR not significant.",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    out_pdf = plots_dir / "off_on_condition_minus_sham_subject_session_roi_burden_heatmaps.pdf"
    paths = _save_pdf_and_png(fig, out_pdf, dpi=300)
    plt.close(fig)
    return paths


def _compact_roi_label(value: str) -> str:
    labels = [label.strip() for label in str(value).split(";") if label.strip()]
    return "\n".join(labels)


def _burden_cell_label(fdr_count: int, raw_only_count: int) -> str:
    if int(fdr_count) > 0 and int(raw_only_count) > 0:
        return f"{int(fdr_count)}*+{int(raw_only_count)}"
    if int(fdr_count) > 0:
        return f"{int(fdr_count)}*"
    if int(raw_only_count) > 0:
        return str(int(raw_only_count))
    return "-"


def _cell_label_fontsize(n_labels: int, max_label_chars: int = 0) -> float:
    if int(n_labels) <= 1:
        fontsize = 7.2
    elif int(n_labels) == 2:
        fontsize = 6.2
    elif int(n_labels) == 3:
        fontsize = 5.2
    else:
        fontsize = 4.5
    if int(max_label_chars) > 22:
        fontsize -= 0.6
    elif int(max_label_chars) > 16:
        fontsize -= 0.3
    return max(3.6, fontsize)


def _split_roi_labels(value: Any) -> list[str]:
    return [label.strip() for label in str(value).split(";") if label.strip()]


def _roi_color_key(label: str) -> str:
    clean = str(label).replace("*", "").strip()
    return re.sub(r"_[LR]$", "", clean)


def _roi_cell_color_map(nonzero_df: pd.DataFrame) -> dict[str, str]:
    labels: set[str] = set()
    for value in nonzero_df.get("shown_roi_labels", pd.Series(dtype=str)).dropna():
        labels.update(_roi_color_key(label) for label in _split_roi_labels(value))
    return {label: ROI_CELL_COLORS[index % len(ROI_CELL_COLORS)] for index, label in enumerate(sorted(labels))}


def _draw_labeled_roi_cell(
    ax: plt.Axes,
    x_value: int,
    y_value: int,
    labels: list[str],
    color_map: dict[str, str],
) -> None:
    if not labels:
        return
    cell_width = 0.96
    cell_height = 0.86
    band_height = cell_height / len(labels)
    fontsize = _cell_label_fontsize(len(labels), max(len(label) for label in labels))
    for label_index, label in enumerate(labels):
        y_min = y_value - cell_height / 2 + label_index * band_height
        ax.add_patch(
            plt.Rectangle(
                (x_value - cell_width / 2, y_min),
                cell_width,
                band_height,
                facecolor=color_map.get(_roi_color_key(label), "#E2E2E2"),
                edgecolor="white",
                linewidth=0.7,
                zorder=2,
            )
        )
        ax.text(
            x_value,
            y_min + band_height / 2,
            label,
            ha="center",
            va="center",
            fontsize=fontsize,
            color="#111111",
            fontweight="bold" if label.endswith("*") else "normal",
            linespacing=0.9,
            zorder=3,
        )
    ax.add_patch(
        plt.Rectangle(
            (x_value - cell_width / 2, y_value - cell_height / 2),
            cell_width,
            cell_height,
            facecolor="none",
            edgecolor="#7C7C7C",
            linewidth=0.55,
            zorder=4,
        )
    )


def _draw_labeled_burden_axis(
    ax: plt.Axes,
    subset: pd.DataFrame,
    sessions: list[str],
    subjects: list[str],
    x_lookup: dict[str, int],
    y_lookup: dict[str, int],
    color_map: dict[str, str],
    *,
    show_ylabel: bool,
) -> None:
    ax.set_xlim(-0.5, len(sessions) - 0.5)
    ax.set_ylim(len(subjects) - 0.5, -0.5)
    ax.set_xticks(np.arange(len(sessions)))
    ax.set_xticklabels(sessions, fontsize=9.5)
    ax.set_yticks(np.arange(len(subjects)))
    ax.set_yticklabels(subjects, fontsize=9.0)
    ax.set_xticks(np.arange(-0.5, len(sessions), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(subjects), 1), minor=True)
    ax.grid(which="minor", color="#D8D8D8", linewidth=0.7)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_xlabel("GVS condition", fontsize=10.0)
    ax.set_ylabel("Subject" if show_ylabel else "", fontsize=10.0)
    ax.set_facecolor("#FAFAFA")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    for row in subset.itertuples(index=False):
        x_value = x_lookup[str(row.target_condition_label)]
        y_value = y_lookup[str(row.subject)]
        _draw_labeled_roi_cell(
            ax,
            x_value,
            y_value,
            _split_roi_labels(getattr(row, "shown_roi_labels", "")),
            color_map,
        )


def _write_labeled_compact_burden_plot(
    nonzero_df: pd.DataFrame,
    plots_dir: Path,
    specs: dict[str, dict[str, Any]],
    *,
    combined_df: pd.DataFrame | None = None,
) -> tuple[list[Path], list[Path]]:
    sessions = sorted(nonzero_df["target_condition_label"].astype(str).unique().tolist(), key=_session_sort_key)
    subjects = sorted(nonzero_df["subject"].astype(str).unique().tolist(), key=_subject_sort_key)
    x_lookup = {label: index for index, label in enumerate(sessions)}
    y_lookup = {subject: index for index, subject in enumerate(subjects)}
    color_map = _roi_cell_color_map(nonzero_df)
    pdf_paths: list[Path] = []
    png_paths: list[Path] = []

    for prefix, _spec in specs.items():
        subset = nonzero_df.loc[nonzero_df["file_prefix"].eq(prefix)].copy()
        if subset.empty:
            continue

        fig_width = max(15.0, 1.85 * len(sessions) + 3.0)
        fig_height = max(9.0, 0.56 * len(subjects) + 1.6)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), facecolor="white")
        _draw_labeled_burden_axis(ax, subset, sessions, subjects, x_lookup, y_lookup, color_map, show_ylabel=True)

        fig.tight_layout()
        pdf_path, png_path = _save_pdf_and_png(
            fig,
            plots_dir / f"{prefix}_subject_session_roi_burden_compact.pdf",
            dpi=300,
        )
        plt.close(fig)
        pdf_paths.append(pdf_path)
        png_paths.append(png_path)

    combined_source = nonzero_df if combined_df is None else combined_df
    combined_subsets = [(prefix, combined_source.loc[combined_source["file_prefix"].eq(prefix)].copy()) for prefix in specs]
    combined_subsets = [(prefix, subset) for prefix, subset in combined_subsets if not subset.empty]
    if combined_subsets:
        combined_sessions = sorted(
            combined_source["target_condition_label"].astype(str).unique().tolist(),
            key=_session_sort_key,
        )
        combined_x_lookup = {label: index for index, label in enumerate(combined_sessions)}
        combined_color_map = _roi_cell_color_map(combined_source)
        max_subjects = max(
            len(subset["subject"].astype(str).unique().tolist())
            for _prefix, subset in combined_subsets
        )
        fig_width = max(23.0, 2.25 * len(combined_sessions) + 5.0)
        fig_height = max(7.2, 0.72 * max_subjects + 2.0)
        fig, axes = plt.subplots(1, len(combined_subsets), figsize=(fig_width, fig_height), facecolor="white")
        for index, (ax, (_prefix, subset)) in enumerate(zip(np.atleast_1d(axes), combined_subsets, strict=True)):
            panel_subjects = sorted(subset["subject"].astype(str).unique().tolist(), key=_subject_sort_key)
            panel_y_lookup = {subject: index for index, subject in enumerate(panel_subjects)}
            _draw_labeled_burden_axis(
                ax,
                subset,
                combined_sessions,
                panel_subjects,
                combined_x_lookup,
                panel_y_lookup,
                combined_color_map,
                show_ylabel=index == 0,
            )
        fig.tight_layout(w_pad=2.0)
        pdf_path, png_path = _save_pdf_and_png(
            fig,
            plots_dir / "off_on_condition_minus_sham_subject_session_roi_burden_compact.pdf",
            dpi=300,
        )
        plt.close(fig)
        pdf_paths.append(pdf_path)
        png_paths.append(png_path)
    return pdf_paths, png_paths


def _fdr_only_compact_view(nonzero_df: pd.DataFrame) -> pd.DataFrame:
    if nonzero_df.empty or "n_significant_rois" not in nonzero_df.columns:
        return nonzero_df.iloc[0:0].copy()
    fdr_df = nonzero_df.loc[nonzero_df["n_significant_rois"].astype(int) > 0].copy()
    if fdr_df.empty:
        return fdr_df
    fdr_df["shown_roi_labels"] = [
        "; ".join(f"{label}*" for label in _split_roi_labels(labels))
        for labels in fdr_df["significant_roi_labels"]
    ]
    return fdr_df


def _write_compact_burden_plot(
    stats_by_prefix: dict[str, pd.DataFrame],
    plots_dir: Path,
    *,
    include_converged_raw_p: bool = False,
    raw_p_alpha: float = DEFAULT_ALPHA,
) -> tuple[list[Path], list[Path], Path]:
    specs = {
        "off_condition_minus_sham_off_roi_mean_delta": {
            "label": "OFF - sham OFF",
            "color": "#2F6FAE",
            "marker": "o",
            "boxstyle": "round,pad=0.25,rounding_size=0.18",
            "x_offset": -0.18,
        },
        "on_condition_minus_sham_on_roi_mean_delta": {
            "label": "ON - sham ON",
            "color": "#D87924",
            "marker": "s",
            "boxstyle": "square,pad=0.25",
            "x_offset": 0.18,
        },
    }
    rows: list[dict[str, Any]] = []
    for prefix, spec in specs.items():
        stats_df = stats_by_prefix.get(prefix, pd.DataFrame())
        if stats_df.empty:
            continue
        active = stats_df.loc[stats_df["target_condition_label"].astype(str).str.casefold() != "sham"].copy()
        fdr = active.loc[_as_bool(active["significant_fdr"])].copy()
        raw_only = active.loc[_converged_raw_p_only_mask(active, alpha=float(raw_p_alpha))].copy()
        shown = pd.concat([fdr, raw_only], ignore_index=True) if include_converged_raw_p else fdr
        if shown.empty:
            continue
        for (subject, condition_label), _cell_df in shown.groupby(
            ["subject", "target_condition_label"],
            dropna=False,
            observed=False,
            sort=False,
        ):
            fdr_labels = sorted(
                fdr.loc[
                    fdr["subject"].astype(str).eq(str(subject))
                    & fdr["target_condition_label"].astype(str).eq(str(condition_label)),
                    "roi_label",
                ]
                .astype(str)
                .unique()
                .tolist()
            )
            raw_labels = sorted(
                raw_only.loc[
                    raw_only["subject"].astype(str).eq(str(subject))
                    & raw_only["target_condition_label"].astype(str).eq(str(condition_label)),
                    "roi_label",
                ]
                .astype(str)
                .unique()
                .tolist()
            )
            roi_labels = [f"{label}*" for label in fdr_labels]
            if include_converged_raw_p:
                roi_labels.extend(raw_labels)
            rows.append(
                {
                    "comparison": str(spec["label"]),
                    "file_prefix": prefix,
                    "subject": str(subject),
                    "target_condition_label": str(condition_label),
                    "n_significant_rois": int(len(fdr_labels)),
                    "n_converged_raw_p_only_rois": int(len(raw_labels)),
                    "significant_roi_labels": "; ".join(fdr_labels),
                    "converged_raw_p_only_roi_labels": "; ".join(raw_labels),
                    "shown_roi_labels": "; ".join(roi_labels),
                }
            )

    nonzero_df = pd.DataFrame(rows)
    csv_path = plots_dir / "off_on_condition_minus_sham_subject_session_roi_burden_compact_nonzero.csv"
    if not nonzero_df.empty:
        nonzero_df = nonzero_df.sort_values(
            ["subject", "target_condition_label", "comparison"],
            key=lambda col: col.map(_subject_sort_key) if col.name == "subject" else col.map(_session_sort_key)
            if col.name == "target_condition_label"
            else col,
        ).reset_index(drop=True)
    nonzero_df.to_csv(csv_path, index=False)

    if nonzero_df.empty:
        fig, ax = plt.subplots(figsize=(6.0, 2.8))
        ax.text(0.5, 0.5, "No FDR-significant ROI burden cells", ha="center", va="center", fontsize=11)
        ax.axis("off")
        pdf_path, png_path = _save_pdf_and_png(
            fig,
            plots_dir / "off_on_condition_minus_sham_subject_session_roi_burden_compact.pdf",
        )
        plt.close(fig)
        return ([pdf_path], [png_path], csv_path)

    combined_df = _fdr_only_compact_view(nonzero_df) if include_converged_raw_p else None
    pdf_paths, png_paths = _write_labeled_compact_burden_plot(nonzero_df, plots_dir, specs, combined_df=combined_df)
    return (pdf_paths, png_paths, csv_path)


def _missing_inputs(args: argparse.Namespace) -> list[str]:
    missing: list[str] = []
    if bool(args.motor_reward_roi_profiles_only):
        required_paths = [(args.lme_trial_csv, "per-trial ROI beta table")]
        for path, label in required_paths:
            if path is None or not Path(path).exists():
                missing.append(f"{path} ({label})")
        return missing

    required_paths = [
        (args.weight_map, "optimization-weight NIfTI"),
        (args.gvs_order, "GVS order table"),
    ]
    if not bool(args.projected_signal_only):
        required_paths.extend(
            [
                (args.roi_definition_figure, "ROI definition reference figure"),
                (args.roi_region_table, "machine-readable ROI region table"),
            ]
        )
        if not bool(args.skip_motor_reward_roi_profiles):
            required_paths.append((args.lme_trial_csv, "per-trial ROI beta table"))
    for path, label in required_paths:
        if path is None or not Path(path).exists():
            missing.append(f"{path} ({label})")
    if args.beta_root is None or not Path(args.beta_root).exists():
        missing.append(f"{args.beta_root} (cleaned beta root)")
    return missing


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute OFF/ON condition-minus-sham ROI burden heatmaps from "
            "cleaned beta volumes, the new voxel-weight map, and the AAL ROI table."
        )
    )
    parser.add_argument("--weight-map", type=_coerce_user_path, default=DEFAULT_WEIGHT_MAP)
    parser.add_argument("--beta-root", type=_coerce_user_path, default=DEFAULT_BETA_ROOT)
    parser.add_argument("--roi-definition-figure", type=_coerce_user_path, default=DEFAULT_ROI_FIGURE)
    parser.add_argument("--roi-region-table", type=_coerce_user_path, default=None)
    parser.add_argument("--gvs-order", type=_coerce_user_path, default=DEFAULT_GVS_ORDER)
    parser.add_argument("--out-dir", type=_coerce_user_path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--projected-signal-out-dir", type=_coerce_user_path, default=DEFAULT_PROJECTED_SIGNAL_OUT_DIR)
    parser.add_argument(
        "--motor-reward-roi-profile-out-dir",
        type=_coerce_user_path,
        default=DEFAULT_MOTOR_REWARD_ROI_PROFILE_OUT_DIR,
    )
    parser.add_argument("--lme-trial-csv", type=_coerce_user_path, default=DEFAULT_LME_TRIAL_CSV)
    parser.add_argument("--roi-percentile", type=float, default=DEFAULT_ROI_PERCENTILE)
    parser.add_argument(
        "--projected-signal-weight-percentile",
        type=float,
        default=DEFAULT_PROJECTED_SIGNAL_WEIGHT_PERCENTILE,
    )
    parser.add_argument("--min-report-voxels", type=int, default=DEFAULT_MIN_REPORT_VOXELS)
    parser.add_argument("--min-lateralized-voxels", type=int, default=1)
    parser.add_argument("--exclude-rois", nargs="*", default=())
    parser.add_argument("--no-split-hemispheres", action="store_true")
    parser.add_argument("--aal-version", default=DEFAULT_AAL_VERSION)
    parser.add_argument("--atlas-cache-dir", type=_coerce_user_path, default=DEFAULT_ATLAS_CACHE_DIR)
    parser.add_argument("--trials-per-condition", type=int, default=DEFAULT_TRIALS_PER_CONDITION)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--subjects", nargs="+", default=None)
    parser.add_argument("--check-inputs", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-projected-signal-profile", action="store_true")
    parser.add_argument("--projected-signal-only", action="store_true")
    parser.add_argument("--skip-motor-reward-roi-profiles", action="store_true")
    parser.add_argument("--motor-reward-roi-profiles-only", action="store_true")
    return parser


def _prepare_args(args: argparse.Namespace) -> argparse.Namespace:
    args.weight_map = _resolve_path(args.weight_map)
    args.beta_root = _resolve_path(args.beta_root)
    args.roi_definition_figure = _resolve_path(args.roi_definition_figure)
    args.gvs_order = _resolve_path(args.gvs_order)
    args.out_dir = _resolve_path(args.out_dir)
    args.projected_signal_out_dir = _resolve_path(args.projected_signal_out_dir)
    args.motor_reward_roi_profile_out_dir = _resolve_path(args.motor_reward_roi_profile_out_dir)
    args.lme_trial_csv = _resolve_path(args.lme_trial_csv)
    args.atlas_cache_dir = _resolve_path(args.atlas_cache_dir)
    args.roi_region_table = _resolve_path(args.roi_region_table)
    if args.roi_region_table is None:
        args.roi_region_table = _default_region_table_for(args.roi_definition_figure)
    args.split_hemispheres = not bool(args.no_split_hemispheres)
    if int(args.trials_per_condition) <= 0:
        raise ValueError("--trials-per-condition must be positive.")
    if not (0.0 < float(args.alpha) <= 1.0):
        raise ValueError("--alpha must be in (0, 1].")
    if not (0.0 <= float(args.projected_signal_weight_percentile) <= 100.0):
        raise ValueError("--projected-signal-weight-percentile must be between 0 and 100.")
    return args


def _print_dry_run(
    args: argparse.Namespace,
    rois: list[ROIArray],
    roi_threshold: float,
    run_specs: list[RunSpec],
) -> None:
    session_df = pd.DataFrame(
        [
            {
                "subject": spec.subject,
                "session": spec.session,
                "run": spec.run,
                "n_trials": spec.n_trials,
                "stim_order": ",".join(str(item) for item in spec.stim_order),
            }
            for spec in run_specs
        ]
    )
    print("Dry run only; no ROI beta matrices, stats tables, or heatmaps were written.")
    print(f"Weight map: {args.weight_map}")
    print(f"Beta root: {args.beta_root}")
    print(f"ROI region table: {args.roi_region_table}")
    print(f"ROI definition figure: {args.roi_definition_figure}")
    print(f"GVS order table: {args.gvs_order}")
    print(f"Output directory: {args.out_dir}")
    print(f"Projected-signal output directory: {args.projected_signal_out_dir}")
    print(f"Motor/reward ROI profile output directory: {args.motor_reward_roi_profile_out_dir}")
    print(f"LME trial table: {args.lme_trial_csv}")
    print(f"Projected-signal weight percentile: p{args.projected_signal_weight_percentile:g}")
    print(f"ROI percentile: p{args.roi_percentile:g}")
    print(f"Weight threshold: {roi_threshold:.8g}")
    print(f"Split hemispheres: {bool(args.split_hemispheres)}")
    print(f"ROI count: {len(rois)}")
    print("Top ROI masks: " + ", ".join(f"{roi.name} ({roi.n_voxels})" for roi in rois[:10]))
    print(f"Runs discovered: {len(run_specs)}")
    print("Runs by session:")
    print(session_df.groupby("session")["run"].count().to_string())
    print("Trial counts:")
    print(session_df["n_trials"].value_counts().sort_index().to_string())


def main() -> int:
    args = _prepare_args(_build_parser().parse_args())
    missing = _missing_inputs(args)
    if missing:
        print("Missing required inputs:")
        for item in missing:
            print(f"- {item}")
        return 1
    if args.check_inputs:
        print("All required inputs are present.")
        return 0

    if args.motor_reward_roi_profiles_only:
        if args.dry_run:
            print("Dry run only; no motor/reward ROI profile files were written.")
            print(f"LME trial table: {args.lme_trial_csv}")
            print(f"Motor/reward ROI profile output directory: {args.motor_reward_roi_profile_out_dir}")
            print(f"Selected ROIs: {len(MOTOR_REWARD_ROI_SELECTION)}")
            return 0
        motor_reward_roi_profile_outputs = _write_motor_reward_roi_profile_outputs(
            Path(args.lme_trial_csv),
            Path(args.motor_reward_roi_profile_out_dir),
            alpha=float(args.alpha),
        )
        for output_path in motor_reward_roi_profile_outputs.values():
            if isinstance(output_path, Path):
                print(f"Saved {output_path}")
        return 0

    weight_img = nib.load(str(args.weight_map))
    weight_values = np.asarray(weight_img.get_fdata(), dtype=np.float64)
    gvs_order = _load_gvs_order(args.gvs_order)
    run_specs = _discover_run_specs(args.beta_root, gvs_order, args.subjects)

    if args.projected_signal_only:
        if args.dry_run:
            print("Dry run only; no projected-signal files were written.")
            print(f"Weight map: {args.weight_map}")
            print(f"Beta root: {args.beta_root}")
            print(f"GVS order table: {args.gvs_order}")
            print(f"Projected-signal output directory: {args.projected_signal_out_dir}")
            print(f"Projected-signal weight percentile: p{args.projected_signal_weight_percentile:g}")
            print(f"Runs discovered: {len(run_specs)}")
            return 0
        projected_signal_outputs = _write_projected_signal_outputs(
            run_specs,
            weight_values,
            weight_img.shape[:3],
            Path(args.projected_signal_out_dir),
            trials_per_condition=int(args.trials_per_condition),
            weight_percentile=float(args.projected_signal_weight_percentile),
            alpha=float(args.alpha),
        )
        for output_path in projected_signal_outputs.values():
            if isinstance(output_path, Path):
                print(f"Saved {output_path}")
        return 0

    groups, roi_metadata, roi_names, min_roi_voxels = _analysis_roi_setup(args, weight_img)
    weighted_rois, roi_threshold = _build_weighted_rois(
        weight_values=weight_values,
        roi_names=roi_names,
        groups=groups,
        roi_percentile=float(args.roi_percentile),
        min_report_voxels=int(args.min_report_voxels),
        min_roi_voxels=int(min_roi_voxels),
    )
    roi_arrays = _build_roi_arrays(weighted_rois, weight_img.shape[:3])
    roi_labels = [roi.name for roi in roi_arrays]

    if args.dry_run:
        _print_dry_run(args, roi_arrays, roi_threshold, run_specs)
        return 0

    out_dir = Path(args.out_dir)
    common_dir = out_dir / "common"
    tables_dir = out_dir / "roi_condition_reference_deltas" / "tables" / "without_unassigned"
    plots_dir = out_dir / "roi_condition_reference_deltas" / "plots" / "without_unassigned"
    common_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    projected_signal_outputs: dict[str, Any] = {}
    if not args.skip_projected_signal_profile:
        projected_signal_outputs = _write_projected_signal_outputs(
            run_specs,
            weight_values,
            weight_img.shape[:3],
            Path(args.projected_signal_out_dir),
            trials_per_condition=int(args.trials_per_condition),
            weight_percentile=float(args.projected_signal_weight_percentile),
            alpha=float(args.alpha),
        )

    motor_reward_roi_profile_outputs: dict[str, Any] = {}
    if not args.skip_motor_reward_roi_profiles:
        motor_reward_roi_profile_outputs = _write_motor_reward_roi_profile_outputs(
            Path(args.lme_trial_csv),
            Path(args.motor_reward_roi_profile_out_dir),
            alpha=float(args.alpha),
        )

    roi_definition = pd.DataFrame(
        {
            "roi_index": np.arange(1, len(roi_arrays) + 1, dtype=np.int64),
            "roi_label": roi_labels,
            "n_weighted_voxels": [roi.n_voxels for roi in roi_arrays],
            "roi_percentile": float(args.roi_percentile),
            "weight_threshold": float(roi_threshold),
        }
    )
    roi_definition.to_csv(common_dir / "weighted_roi_definition.csv", index=False)

    matrices, inventory_df = _load_condition_matrices(
        run_specs,
        roi_arrays,
        weight_img.shape[:3],
        int(args.trials_per_condition),
    )
    inventory_df.to_csv(common_dir / "run_condition_inventory.csv", index=False)

    stats_by_prefix: dict[str, pd.DataFrame] = {}
    combined_stats: list[pd.DataFrame] = []
    for spec in ROI_REFERENCE_SPECS:
        stats_df = compute_condition_roi_reference_stats(
            matrices,
            roi_labels,
            target_session=int(spec["target_session"]),
            reference_session=int(spec["reference_session"]),
            comparison_kind=str(spec["comparison_kind"]),
            alpha=float(args.alpha),
        )
        prefix = str(spec["file_prefix"])
        stats_by_prefix[prefix] = stats_df
        if not stats_df.empty:
            combined_stats.append(stats_df)
            _write_reference_tables(stats_df, tables_dir, prefix, roi_labels)

    if combined_stats:
        pd.concat(combined_stats, ignore_index=True).to_csv(
            tables_dir / "condition_roi_reference_delta_ttest_stats_by_subject_long.csv",
            index=False,
        )

    pdf_path, png_path = _write_burden_heatmaps(stats_by_prefix, plots_dir)
    compact_pdf_paths, compact_png_paths, compact_csv_path = _write_compact_burden_plot(stats_by_prefix, plots_dir)

    manifest = {
        "inputs": {
            "weight_map": args.weight_map,
            "beta_root": args.beta_root,
            "roi_definition_figure": args.roi_definition_figure,
            "roi_region_table": args.roi_region_table,
            "gvs_order": args.gvs_order,
            "lme_trial_csv": args.lme_trial_csv,
        },
        "outputs": {
            "out_dir": out_dir,
            "weighted_roi_definition": common_dir / "weighted_roi_definition.csv",
            "run_condition_inventory": common_dir / "run_condition_inventory.csv",
            "tables_dir": tables_dir,
            "plots_dir": plots_dir,
            "burden_heatmap_png": png_path,
            "burden_heatmap_pdf": pdf_path,
            "compact_burden_png": compact_png_paths,
            "compact_burden_pdf": compact_pdf_paths,
            "compact_burden_nonzero_csv": compact_csv_path,
            "projected_signal": projected_signal_outputs,
            "motor_reward_roi_profiles": motor_reward_roi_profile_outputs,
        },
        "roi_definition": {
            "source": "new_weight_map_p90_aal_region_table",
            "split_hemispheres": bool(args.split_hemispheres),
            "roi_percentile": float(args.roi_percentile),
            "weight_threshold": float(roi_threshold),
            "min_report_voxels": int(args.min_report_voxels),
            "min_roi_voxels": int(min_roi_voxels),
            "n_rois": int(len(roi_arrays)),
            "roi_labels": roi_labels,
            "metadata": roi_metadata,
        },
        "condition_split": {
            "source": "data/gvs_order_by_subject_session_run.tsv true_stim_order",
            "trials_per_condition_block": int(args.trials_per_condition),
            "sham_condition_code": SHAM_CONDITION_CODE,
            "condition_codes": list(CONDITION_CODES),
        },
        "statistics": {
            "test": "subject-wise two-sided Welch t-test on weighted ROI beta trial vectors",
            "reference": "same-session sham gvs-01",
            "multiple_comparison_correction": "Benjamini-Hochberg FDR across ROI-by-condition cells within each subject and comparison",
            "alpha": float(args.alpha),
            "burden_value": "number of FDR-significant unique ROI labels per subject and active GVS condition",
        },
        "n_runs": int(len(run_specs)),
        "n_condition_matrices": int(len(matrices)),
    }
    (out_dir / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2, default=_json_default), encoding="utf-8")

    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")
    for compact_png_path in compact_png_paths:
        print(f"Saved {compact_png_path}")
    for compact_pdf_path in compact_pdf_paths:
        print(f"Saved {compact_pdf_path}")
    for output_path in projected_signal_outputs.values():
        if isinstance(output_path, Path):
            print(f"Saved {output_path}")
    for output_path in motor_reward_roi_profile_outputs.values():
        if isinstance(output_path, Path):
            print(f"Saved {output_path}")
    print(f"Saved {compact_csv_path}")
    print(f"Saved {common_dir / 'weighted_roi_definition.csv'}")
    print(f"Saved {common_dir / 'run_condition_inventory.csv'}")
    print(f"Saved {tables_dir}")
    print(f"Saved {out_dir / 'analysis_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
