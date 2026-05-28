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
DEFAULT_ROI_PERCENTILE = 90.0
DEFAULT_TRIALS_PER_CONDITION = 10
DEFAULT_ALPHA = 0.05

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


def _gvs_counts(stats_df: pd.DataFrame, subjects: list[str], sessions: list[str]) -> np.ndarray:
    significant = stats_df.loc[_as_bool(stats_df["significant_fdr"])].copy()
    if significant.empty:
        return np.zeros((len(subjects), len(sessions)), dtype=float)
    significant["subject"] = significant["subject"].astype(str)
    significant["target_condition_label"] = significant["target_condition_label"].astype(str)
    significant["roi_label"] = significant["roi_label"].astype(str)
    roi_sets = {(subject, session): set() for subject in subjects for session in sessions}
    for (subject, session), cell_df in significant.groupby(
        ["subject", "target_condition_label"],
        dropna=False,
        observed=False,
        sort=False,
    ):
        key = (str(subject), str(session))
        if key in roi_sets:
            roi_sets[key] = set(cell_df["roi_label"].tolist())
    return np.array([[len(roi_sets[(subject, session)]) for session in sessions] for subject in subjects], dtype=float)


def _draw_gvs_panel(
    ax: plt.Axes,
    counts: np.ndarray,
    subjects: list[str],
    sessions: list[str],
    cmap_name: str,
    vmax: int,
    show_ylabel: bool,
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


def _write_burden_heatmaps(stats_by_prefix: dict[str, pd.DataFrame], plots_dir: Path) -> tuple[Path, Path]:
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
    prepared: list[tuple[list[str], np.ndarray, str]] = []
    max_count = 0
    for prefix, cmap_name in specs:
        stats_df = stats_by_prefix.get(prefix, pd.DataFrame())
        if stats_df.empty:
            continue
        subjects = sorted(stats_df["subject"].astype(str).unique().tolist(), key=_subject_sort_key)
        counts = _gvs_counts(stats_df, subjects, sessions)
        max_count = max(max_count, int(np.nanmax(counts)) if counts.size else 0)
        prepared.append((subjects, counts, cmap_name))
        pd.DataFrame(counts.astype(int), index=subjects, columns=sessions).rename_axis("subject").to_csv(
            plots_dir / f"{prefix}_subject_session_roi_burden_heatmap.csv"
        )

    if not prepared:
        raise RuntimeError("No non-empty stats tables were available for burden heatmaps.")

    fig_w = max(15.0, 1.7 * len(sessions) + 4.0)
    fig_h = max(7.4, max(0.48 * len(subjects) for subjects, _, _ in prepared) + 2.1)
    fig, axes = plt.subplots(1, len(prepared), figsize=(fig_w, fig_h))
    for idx, (ax, (subjects, counts, cmap_name)) in enumerate(zip(np.atleast_1d(axes), prepared)):
        image = _draw_gvs_panel(ax, counts, subjects, sessions, cmap_name, max_count, show_ylabel=idx == 0)
        ax.set_xlabel("GVS session", fontsize=11)
        colorbar = fig.colorbar(image, ax=ax, shrink=0.84, pad=0.02)
        if idx != 0:
            colorbar.set_label("No. of significant ROIs", fontsize=10)
        colorbar.ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        colorbar.ax.tick_params(labelsize=9)
    fig.tight_layout()
    out_pdf = plots_dir / "off_on_condition_minus_sham_subject_session_roi_burden_heatmaps.pdf"
    paths = _save_pdf_and_png(fig, out_pdf, dpi=300)
    plt.close(fig)
    return paths


def _compact_roi_label(value: str) -> str:
    labels = [label.strip() for label in str(value).split(";") if label.strip()]
    return "\n".join(labels)


def _write_compact_burden_plot(stats_by_prefix: dict[str, pd.DataFrame], plots_dir: Path) -> tuple[Path, Path, Path]:
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
        significant = stats_df.loc[_as_bool(stats_df["significant_fdr"])].copy()
        significant = significant.loc[significant["target_condition_label"].astype(str).str.casefold() != "sham"]
        if significant.empty:
            continue
        for (subject, condition_label), cell_df in significant.groupby(
            ["subject", "target_condition_label"],
            dropna=False,
            observed=False,
            sort=False,
        ):
            roi_labels = sorted(cell_df["roi_label"].astype(str).unique().tolist())
            rows.append(
                {
                    "comparison": str(spec["label"]),
                    "file_prefix": prefix,
                    "subject": str(subject),
                    "target_condition_label": str(condition_label),
                    "n_significant_rois": int(len(roi_labels)),
                    "significant_roi_labels": "; ".join(roi_labels),
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
        paths = _save_pdf_and_png(fig, plots_dir / "off_on_condition_minus_sham_subject_session_roi_burden_compact.pdf")
        plt.close(fig)
        return (*paths, csv_path)

    sessions = sorted(nonzero_df["target_condition_label"].astype(str).unique().tolist(), key=_session_sort_key)
    subjects = sorted(nonzero_df["subject"].astype(str).unique().tolist(), key=_subject_sort_key)
    x_lookup = {label: index for index, label in enumerate(sessions)}
    y_lookup = {subject: index for index, subject in enumerate(subjects)}

    fig_width = max(9.2, 1.12 * len(sessions) + 3.0)
    fig_height = max(4.0, 0.62 * len(subjects) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), facecolor="white")
    ax.set_axisbelow(True)
    ax.grid(axis="both", color="#E1E1E1", linewidth=0.8)

    for prefix, spec in specs.items():
        subset = nonzero_df.loc[nonzero_df["file_prefix"].eq(prefix)]
        if subset.empty:
            continue
        x_values = np.asarray(
            [x_lookup[str(label)] + float(spec["x_offset"]) for label in subset["target_condition_label"]],
            dtype=np.float64,
        )
        y_values = np.asarray([y_lookup[str(subject)] for subject in subset["subject"]], dtype=np.float64)
        ax.scatter(
            [],
            [],
            s=150.0,
            marker=str(spec["marker"]),
            color=str(spec["color"]),
            edgecolor="#202020",
            linewidth=0.7,
            alpha=0.92,
            label=str(spec["label"]),
        )
        for x_value, y_value, roi_text in zip(
            x_values,
            y_values,
            subset["significant_roi_labels"].astype(str),
            strict=True,
        ):
            ax.text(
                x_value,
                y_value,
                _compact_roi_label(roi_text),
                ha="center",
                va="center",
                color="white",
                fontsize=7.2,
                fontweight="bold",
                linespacing=1.0,
                bbox={
                    "boxstyle": str(spec["boxstyle"]),
                    "facecolor": str(spec["color"]),
                    "edgecolor": "#202020",
                    "linewidth": 0.75,
                    "alpha": 0.94,
                },
                zorder=4,
            )

    ax.set_xticks(np.arange(len(sessions)))
    ax.set_xticklabels(sessions, fontsize=10)
    ax.set_yticks(np.arange(len(subjects)))
    ax.set_yticklabels(subjects, fontsize=10)
    ax.set_xlim(-0.55, len(sessions) - 0.45)
    ax.set_ylim(len(subjects) - 0.55, -0.55)
    ax.set_xlabel("GVS condition", fontsize=10.5)
    ax.set_ylabel("Subject with at least one significant ROI", fontsize=10.5)
    ax.set_title("FDR-significant ROI burden by subject and GVS condition", fontsize=12.5, pad=10)
    ax.legend(frameon=False, loc="upper right", fontsize=9.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    pdf_png = _save_pdf_and_png(
        fig,
        plots_dir / "off_on_condition_minus_sham_subject_session_roi_burden_compact.pdf",
        dpi=300,
    )
    plt.close(fig)
    return (*pdf_png, csv_path)


def _missing_inputs(args: argparse.Namespace) -> list[str]:
    missing: list[str] = []
    for path, label in (
        (args.weight_map, "optimization-weight NIfTI"),
        (args.roi_definition_figure, "ROI definition reference figure"),
        (args.roi_region_table, "machine-readable ROI region table"),
        (args.gvs_order, "GVS order table"),
    ):
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
    parser.add_argument("--roi-percentile", type=float, default=DEFAULT_ROI_PERCENTILE)
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
    return parser


def _prepare_args(args: argparse.Namespace) -> argparse.Namespace:
    args.weight_map = _resolve_path(args.weight_map)
    args.beta_root = _resolve_path(args.beta_root)
    args.roi_definition_figure = _resolve_path(args.roi_definition_figure)
    args.gvs_order = _resolve_path(args.gvs_order)
    args.out_dir = _resolve_path(args.out_dir)
    args.atlas_cache_dir = _resolve_path(args.atlas_cache_dir)
    args.roi_region_table = _resolve_path(args.roi_region_table)
    if args.roi_region_table is None:
        args.roi_region_table = _default_region_table_for(args.roi_definition_figure)
    args.split_hemispheres = not bool(args.no_split_hemispheres)
    if int(args.trials_per_condition) <= 0:
        raise ValueError("--trials-per-condition must be positive.")
    if not (0.0 < float(args.alpha) <= 1.0):
        raise ValueError("--alpha must be in (0, 1].")
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

    weight_img = nib.load(str(args.weight_map))
    weight_values = np.asarray(weight_img.get_fdata(), dtype=np.float64)
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

    gvs_order = _load_gvs_order(args.gvs_order)
    run_specs = _discover_run_specs(args.beta_root, gvs_order, args.subjects)
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
    compact_pdf_path, compact_png_path, compact_csv_path = _write_compact_burden_plot(stats_by_prefix, plots_dir)

    manifest = {
        "inputs": {
            "weight_map": args.weight_map,
            "beta_root": args.beta_root,
            "roi_definition_figure": args.roi_definition_figure,
            "roi_region_table": args.roi_region_table,
            "gvs_order": args.gvs_order,
        },
        "outputs": {
            "out_dir": out_dir,
            "weighted_roi_definition": common_dir / "weighted_roi_definition.csv",
            "run_condition_inventory": common_dir / "run_condition_inventory.csv",
            "tables_dir": tables_dir,
            "plots_dir": plots_dir,
            "burden_heatmap_png": png_path,
            "burden_heatmap_pdf": pdf_path,
            "compact_burden_png": compact_png_path,
            "compact_burden_pdf": compact_pdf_path,
            "compact_burden_nonzero_csv": compact_csv_path,
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
    print(f"Saved {compact_png_path}")
    print(f"Saved {compact_pdf_path}")
    print(f"Saved {compact_csv_path}")
    print(f"Saved {common_dir / 'weighted_roi_definition.csv'}")
    print(f"Saved {common_dir / 'run_condition_inventory.csv'}")
    print(f"Saved {tables_dir}")
    print(f"Saved {out_dir / 'analysis_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
