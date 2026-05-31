#!/usr/bin/env python3
"""Ablation-study summaries and publication plots.

This script reads the existing ablation maps and SLURM logs, then writes
fold-level metric tables, spatial-overlap summaries, ROI summaries, and
publication-oriented figures under ``figures/ablation``.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import re
from io import BytesIO
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import datasets, image
from scipy import ndimage
from scipy.stats import spearmanr, wilcoxon

from threshold_robustness_voxel_network import (
    DEFAULT_AAL_VERSION,
    UNASSIGNED_ROI,
    _build_roi_groups,
    _display_region_name,
)


DEFAULT_MAIN_MAP = Path("data/voxel_weights_task1_bold0.6_beta0.6_smooth1.25_gamma1.5.nii.gz")
DEFAULT_FULL_MODEL_HTML = Path("data/voxel_weights_task1_bold0.6_beta0.6_smooth1.25_gamma1.5_bold_thr90.html")
DEFAULT_TASK_ONLY_MAP = Path("data/z_valu_standard_glm.nii.gz")
DEFAULT_TASK_ONLY_Z_THRESHOLD = 3.5
DEFAULT_ABLATION_DIR = Path("data/ablation")
DEFAULT_OUT_DIR = Path("figures/ablation")
DEFAULT_LOGS = (
    Path("data/ablation/slurm-11460024.out"),
    Path("data/ablation/slurm-11445550.out"),
)
DEFAULT_ATLAS_CACHE_DIR = Path("/home/zkavian/nilearn_data")
REFERENCE_PERCENTILE = 90.0
EPS = 1e-8
HARVARD_OXFORD_SUBCORTICAL_ATLAS = "sub-maxprob-thr25-2mm"
BRAINSTEM_ATLAS_LABELS = ("Brain-Stem",)
BRAINSTEM_AAL_ROIS = ("VTA", "SN_pc", "SN_pr", "Red_N", "LC", "Raphe")
BRAINSTEM_MNI_BOUNDS_MM = {
    "x_abs": 20.0,
    "y_min": -50.0,
    "y_max": -4.0,
    "z_min": -36.0,
    "z_max": 10.0,
}
MOTOR_CONTOUR_ROIS = ("Precentral", "Supp_Motor_Area", "Paracentral_Lobule", "Postcentral")
MIXED_LOG_FINAL_CANDIDATE_KEYS = {
    (0.0, 0.0, 0.0, 0.0, 1.5),
    (1.0, 0.0, 0.0, 0.0, 1.5),
}
LOG_CANDIDATE_ALLOWLISTS = {
    "slurm-11445550.out": MIXED_LOG_FINAL_CANDIDATE_KEYS,
}
STALE_UNRELATED_OUTPUTS = (
    "ablation_balanced_score_broad.png",
    "ablation_balanced_score_broad.pdf",
    "ablation_broad_leave_one_out_summary.png",
    "ablation_broad_leave_one_out_summary.pdf",
    "ablation_cv_tradeoff_broad.png",
    "ablation_cv_tradeoff_broad.pdf",
)

COLORS = {
    "Full model": "#1f77b4",
    "Full ablation map": "#1f77b4",
    "No task": "#d95f02",
    "No BOLD stability": "#1b9e77",
    "No beta stability": "#7570b3",
    "No smoothness": "#e7298a",
    "Task-only reference": "#666666",
    "BOLD-only": "#66a61e",
    "Beta-only": "#e6ab02",
    "Smooth-only": "#a6761d",
    "No objective penalties": "#8c8c8c",
}


@dataclass(frozen=True)
class Candidate:
    task: float
    bold: float
    beta: float
    smooth: float
    gamma: float

    @property
    def key(self) -> tuple[float, float, float, float, float]:
        return (self.task, self.bold, self.beta, self.smooth, self.gamma)


@dataclass
class MapSpec:
    map_id: str
    label: str
    path: Path | None
    candidate: Candidate | None
    source: str
    threshold_percentile: float | None = None


def _fmt_float(value: float) -> str:
    return f"{value:g}"


def _candidate_id(candidate: Candidate) -> str:
    return (
        f"task{_fmt_float(candidate.task)}_bold{_fmt_float(candidate.bold)}_"
        f"beta{_fmt_float(candidate.beta)}_smooth{_fmt_float(candidate.smooth)}_"
        f"gamma{_fmt_float(candidate.gamma)}"
    )


def _candidate_from_match(match: re.Match[str]) -> Candidate:
    return Candidate(*(float(match.group(name)) for name in ("task", "bold", "beta", "smooth", "gamma")))


def _candidate_label(candidate: Candidate, group: str | None = None) -> str:
    t, b, be, s, _ = candidate.key
    if group == "final_0p6":
        if (t, b, be, s) == (1.0, 0.6, 0.6, 1.25):
            return "Full model"
        if (t, b, be, s) == (0.0, 0.6, 0.6, 1.25):
            return "No task"
        if (t, b, be, s) == (1.0, 0.0, 0.6, 1.25):
            return "No BOLD stability"
        if (t, b, be, s) == (1.0, 0.6, 0.0, 1.25):
            return "No beta stability"
        if (t, b, be, s) == (0.0, 0.6, 0.0, 0.0):
            return "BOLD-only"
        if (t, b, be, s) == (0.0, 0.0, 0.6, 0.0):
            return "Beta-only"
        if (t, b, be, s) == (0.0, 0.0, 0.0, 1.25):
            return "Smooth-only"
        if (t, b, be, s) == (0.0, 0.0, 0.0, 0.0):
            return "No objective penalties"
        if (t, b, be, s) == (1.0, 0.0, 0.0, 0.0):
            return "Task-only reference"
    if (t, b, be, s) == (1.0, 0.0, 0.0, 0.0):
        return "Task-only reference"
    return f"task={t:g}, bold={b:g}, beta={be:g}, smooth={s:g}"


def _analysis_group(candidate: Candidate, source_log: str) -> str:
    if source_log == "slurm-11460024.out":
        return "final_0p6"
    if candidate.bold in {0.0, 0.6} and candidate.beta in {0.0, 0.6} and candidate.smooth in {0.0, 1.25}:
        return "final_0p6"
    return "other"


def parse_slurm_logs(log_paths: tuple[Path, ...] | list[Path]) -> pd.DataFrame:
    start_re = re.compile(
        r"^Fold (?P<fold>\d+): optimization with task=(?P<task>-?\d+(?:\.\d+)?), "
        r"bold=(?P<bold>-?\d+(?:\.\d+)?), beta=(?P<beta>-?\d+(?:\.\d+)?), "
        r"smooth=(?P<smooth>-?\d+(?:\.\d+)?), gamma=(?P<gamma>-?\d+(?:\.\d+)?)"
    )
    loss_terms_re = re.compile(
        r"Loss terms -> C_task: (?P<task>[-+0-9.eE]+), C_bold: (?P<bold>[-+0-9.eE]+), "
        r"C_beta: (?P<beta>[-+0-9.eE]+), C_smooth: (?P<smooth>[-+0-9.eE]+)"
    )
    test_term_re = re.compile(r"^\[test\] (?P<term>task|bold|beta|smooth)_penalty: (?P<value>[-+0-9.eE]+)")
    train_loss_re = re.compile(r"Total loss \(train objective\): (?P<value>[-+0-9.eE]+)")
    test_loss_re = re.compile(r"Total loss \(test objective\):\s+(?P<value>[-+0-9.eE]+)")
    gamma_ratio_re = re.compile(r"Gamma-penalty ratio -> train: (?P<train>[-+0-9.eE]+), test: (?P<test>[-+0-9.eE]+)")
    train_corr_re = re.compile(r"Train metrics -> corr: (?P<value>[-+0-9.eE]+)")
    test_corr_re = re.compile(r"Test metrics\s+-> corr: (?P<value>[-+0-9.eE]+)")

    rows: list[dict[str, object]] = []
    current: dict[str, object] | None = None

    def flush() -> None:
        nonlocal current
        if current is None:
            return
        required = {"fold", "train_corr", "test_corr", "train_total_loss", "test_total_loss"}
        if required.issubset(current):
            candidate = Candidate(
                float(current["task"]),
                float(current["bold"]),
                float(current["beta"]),
                float(current["smooth"]),
                float(current["gamma"]),
            )
            current["candidate_id"] = _candidate_id(candidate)
            current["analysis_group"] = _analysis_group(candidate, str(current["source_log"]))
            current["candidate_label"] = _candidate_label(candidate, str(current["analysis_group"]))
            current["abs_train_corr"] = abs(float(current["train_corr"]))
            current["abs_test_corr"] = abs(float(current["test_corr"]))
            current["corr_generalization_gap"] = abs(float(current["abs_train_corr"]) - float(current["abs_test_corr"]))
            current["relative_loss_gap"] = (
                (float(current["test_total_loss"]) - float(current["train_total_loss"]))
                / (abs(float(current["train_total_loss"])) + EPS)
            )
            rows.append(current)
        current = None

    for log_path in log_paths:
        if not log_path.exists():
            continue
        for raw_line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw_line.strip()
            start = start_re.search(line)
            if start:
                flush()
                candidate = _candidate_from_match(start)
                allowlist = LOG_CANDIDATE_ALLOWLISTS.get(log_path.name)
                if allowlist is not None and candidate.key not in allowlist:
                    current = None
                    continue
                current = {
                    "source_log": log_path.name,
                    "fold": int(start.group("fold")),
                    "task": candidate.task,
                    "bold": candidate.bold,
                    "beta": candidate.beta,
                    "smooth": candidate.smooth,
                    "gamma": candidate.gamma,
                }
                continue
            if current is None:
                continue
            if match := loss_terms_re.search(line):
                for term in ("task", "bold", "beta", "smooth"):
                    current[f"train_{term}_term"] = float(match.group(term))
                continue
            if match := test_term_re.search(line):
                current[f"test_{match.group('term')}_term"] = float(match.group("value"))
                continue
            if match := train_loss_re.search(line):
                current["train_total_loss"] = float(match.group("value"))
                continue
            if match := test_loss_re.search(line):
                current["test_total_loss"] = float(match.group("value"))
                continue
            if match := gamma_ratio_re.search(line):
                current["train_gamma_ratio"] = float(match.group("train"))
                current["test_gamma_ratio"] = float(match.group("test"))
                continue
            if match := train_corr_re.search(line):
                current["train_corr"] = float(match.group("value"))
                continue
            if match := test_corr_re.search(line):
                current["test_corr"] = float(match.group("value"))
                continue
        flush()

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["analysis_group", "candidate_id", "fold"]).reset_index(drop=True)


def add_balanced_scores(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if metrics.empty:
        return metrics, pd.DataFrame()
    metrics = metrics.copy()
    metrics["score_corr_term"] = -metrics["abs_test_corr"]
    metrics["score_corr_gap_term"] = metrics["corr_generalization_gap"]
    metrics["score_loss_gap_term"] = metrics["relative_loss_gap"]
    weight_rows = []

    for group, group_df in metrics.groupby("analysis_group", sort=False):
        means = group_df.groupby("candidate_id", as_index=False)[
            ["score_corr_term", "score_corr_gap_term", "score_loss_gap_term"]
        ].mean()
        weights: dict[str, float] = {}
        for term in ("score_corr_term", "score_corr_gap_term", "score_loss_gap_term"):
            value_range = float(means[term].max() - means[term].min())
            weights[term] = 1.0 / value_range if np.isfinite(value_range) and value_range > 0 else 1.0
        selector = metrics["analysis_group"].eq(group)
        metrics.loc[selector, "balanced_score"] = (
            weights["score_corr_term"] * metrics.loc[selector, "score_corr_term"]
            + weights["score_corr_gap_term"] * metrics.loc[selector, "score_corr_gap_term"]
            + weights["score_loss_gap_term"] * metrics.loc[selector, "score_loss_gap_term"]
        )
        weight_rows.append(
            {
                "analysis_group": group,
                "corr_weight": weights["score_corr_term"],
                "corr_gap_weight": weights["score_corr_gap_term"],
                "loss_gap_weight": weights["score_loss_gap_term"],
                "definition": "inverse range of candidate-mean score components within analysis_group",
            }
        )
    return metrics, pd.DataFrame(weight_rows)


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    rows = []
    metric_cols = [
        "abs_train_corr",
        "abs_test_corr",
        "corr_generalization_gap",
        "train_total_loss",
        "test_total_loss",
        "relative_loss_gap",
        "balanced_score",
    ]
    for (group, candidate_id), sub in metrics.groupby(["analysis_group", "candidate_id"], sort=False):
        row = {
            "analysis_group": group,
            "candidate_id": candidate_id,
            "candidate_label": sub["candidate_label"].iloc[0],
            "task": float(sub["task"].iloc[0]),
            "bold": float(sub["bold"].iloc[0]),
            "beta": float(sub["beta"].iloc[0]),
            "smooth": float(sub["smooth"].iloc[0]),
            "gamma": float(sub["gamma"].iloc[0]),
            "fold_count": int(sub["fold"].nunique()),
        }
        for col in metric_cols:
            values = pd.to_numeric(sub[col], errors="coerce").dropna().to_numpy(dtype=float)
            row[f"{col}_mean"] = float(np.mean(values)) if values.size else np.nan
            row[f"{col}_sem"] = float(np.std(values, ddof=1) / math.sqrt(values.size)) if values.size > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def compare_scores_to_full(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if metrics.empty or "balanced_score" not in metrics:
        return pd.DataFrame()
    for group, group_df in metrics.groupby("analysis_group", sort=False):
        full_candidates = group_df[group_df["candidate_label"].eq("Full model")]["candidate_id"].unique()
        if len(full_candidates) != 1:
            continue
        full = group_df[group_df["candidate_id"].eq(full_candidates[0])][["fold", "balanced_score"]].rename(
            columns={"balanced_score": "full_score"}
        )
        for candidate_id, sub in group_df.groupby("candidate_id", sort=False):
            if candidate_id == full_candidates[0]:
                continue
            paired = full.merge(sub[["fold", "balanced_score"]], on="fold", how="inner")
            if paired.shape[0] < 2:
                p_value = np.nan
                median_delta = np.nan
            else:
                delta = paired["balanced_score"] - paired["full_score"]
                median_delta = float(np.median(delta))
                try:
                    p_value = float(wilcoxon(delta).pvalue)
                except ValueError:
                    p_value = np.nan
            rows.append(
                {
                    "analysis_group": group,
                    "candidate_id": candidate_id,
                    "candidate_label": sub["candidate_label"].iloc[0],
                    "n_paired_folds": int(paired.shape[0]),
                    "median_score_minus_full": median_delta,
                    "wilcoxon_p": p_value,
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["bonferroni_p"] = np.nan
    for group, idx in out.groupby("analysis_group").groups.items():
        p = out.loc[idx, "wilcoxon_p"].to_numpy(dtype=float)
        finite = np.isfinite(p)
        corrected = np.full_like(p, np.nan)
        corrected[finite] = np.minimum(p[finite] * int(np.count_nonzero(finite)), 1.0)
        out.loc[idx, "bonferroni_p"] = corrected
    return out


def _load_data(path: Path) -> tuple[nib.Nifti1Image, np.ndarray]:
    img = nib.load(str(path))
    return img, np.asarray(img.get_fdata(), dtype=float)


def _mask_from_map(data: np.ndarray, threshold_percentile: float | None = None) -> tuple[np.ndarray, float | None]:
    finite_nonzero = np.isfinite(data) & (data != 0)
    if threshold_percentile is None:
        return finite_nonzero, None
    values = data[finite_nonzero]
    if values.size == 0:
        return np.zeros(data.shape, dtype=bool), np.nan
    threshold = float(np.percentile(values, threshold_percentile))
    return finite_nonzero & (data >= threshold), threshold


def _weighted_center(mask: np.ndarray, values: np.ndarray, affine: np.ndarray) -> np.ndarray:
    coords = np.column_stack(np.nonzero(mask))
    if coords.size == 0:
        return np.array([np.nan, np.nan, np.nan], dtype=float)
    weights = np.abs(values[mask])
    if not np.any(np.isfinite(weights)) or float(np.nansum(weights)) <= 0:
        weights = np.ones(coords.shape[0], dtype=float)
    center_vox = np.average(coords, axis=0, weights=weights)
    return np.asarray(nib.affines.apply_affine(affine, center_vox), dtype=float)


def _component_stats(mask: np.ndarray) -> tuple[int, int, float]:
    labels, n_components = ndimage.label(mask)
    sizes = np.bincount(labels.ravel())
    largest = int(sizes[1:].max()) if sizes.size > 1 else 0
    total = int(np.count_nonzero(mask))
    return int(n_components), largest, float(largest / total) if total else np.nan


def _discover_map_specs(main_map: Path, ablation_dir: Path) -> list[MapSpec]:
    specs = [
        MapSpec(
            map_id="main_result_p90",
            label="Full model",
            path=main_map,
            candidate=Candidate(1.0, 0.6, 0.6, 1.25, 1.5),
            source="main_result_top90_nonzero_weights",
            threshold_percentile=REFERENCE_PERCENTILE,
        )
    ]
    pattern = re.compile(
        r"task(?P<task>\d+(?:\.\d+)?)_bold(?P<bold>\d+(?:\.\d+)?)_"
        r"beta(?P<beta>\d+(?:\.\d+)?)_smooth(?P<smooth>\d+(?:\.\d+)?)_"
        r"gamma(?P<gamma>\d+(?:\.\d+)?)_bold_thr90\.nii\.gz$"
    )
    for path in sorted(ablation_dir.glob("voxel_weights_mean_foldavg_sub9_ses1_*.nii.gz")):
        match = pattern.search(path.name)
        if not match:
            continue
        candidate = _candidate_from_match(match)
        group = "final_0p6" if (
            candidate.bold in {0.0, 0.6}
            and candidate.beta in {0.0, 0.6}
            and candidate.smooth in {0.0, 1.25}
        ) else "other"
        label = _candidate_label(candidate, group)
        if candidate.key == (1.0, 0.6, 0.6, 1.25, 1.5):
            label = "Full ablation map"
        specs.append(
            MapSpec(
                map_id=_candidate_id(candidate),
                label=label,
                path=path,
                candidate=candidate,
                source="cluster_filtered_ablation_thr90",
            )
        )
    return specs


def summarize_maps(map_specs: list[MapSpec], out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    loaded: dict[str, dict[str, object]] = {}
    reference_img: nib.Nifti1Image | None = None
    reference_mask: np.ndarray | None = None
    reference_values: np.ndarray | None = None

    for spec in map_specs:
        if spec.path is None or not spec.path.exists():
            continue
        img, data = _load_data(spec.path)
        if reference_img is None:
            reference_img = img
        if img.shape[:3] != reference_img.shape[:3] or not np.allclose(img.affine, reference_img.affine):
            raise RuntimeError(f"{spec.path} is not on the same grid as {map_specs[0].path}.")
        mask, threshold = _mask_from_map(data, spec.threshold_percentile)
        loaded[spec.map_id] = {
            "spec": spec,
            "img": img,
            "data": data,
            "mask": mask,
            "threshold": threshold,
            "center_mm": _weighted_center(mask, data, img.affine),
        }
        if spec.map_id == "main_result_p90":
            reference_mask = mask
            reference_values = data
            mask_img = nib.Nifti1Image(mask.astype(np.uint8), img.affine, img.header)
            nib.save(mask_img, str(out_dir / "main_result_p90_mask.nii.gz"))

    if reference_img is None or reference_mask is None or reference_values is None:
        raise RuntimeError("Main result map is required for spatial comparison.")
    reference_center = loaded["main_result_p90"]["center_mm"]

    map_rows = []
    overlap_rows = []
    for map_id, payload in loaded.items():
        spec: MapSpec = payload["spec"]  # type: ignore[assignment]
        data = payload["data"]  # type: ignore[assignment]
        mask = payload["mask"]  # type: ignore[assignment]
        center = payload["center_mm"]  # type: ignore[assignment]
        n_components, largest_component, largest_fraction = _component_stats(mask)
        values = data[mask]
        row = {
            "map_id": map_id,
            "label": spec.label,
            "source": spec.source,
            "path": str(spec.path) if spec.path else "",
            "threshold_percentile": spec.threshold_percentile,
            "threshold_value": payload["threshold"],
            "n_voxels": int(np.count_nonzero(mask)),
            "n_components": n_components,
            "largest_component_voxels": largest_component,
            "largest_component_fraction": largest_fraction,
            "mean_value": float(np.mean(values)) if values.size else np.nan,
            "mean_abs_value": float(np.mean(np.abs(values))) if values.size else np.nan,
            "x_mm": float(center[0]),
            "y_mm": float(center[1]),
            "z_mm": float(center[2]),
        }
        if spec.candidate is not None:
            row.update(
                {
                    "candidate_id": _candidate_id(spec.candidate),
                    "task": spec.candidate.task,
                    "bold": spec.candidate.bold,
                    "beta": spec.candidate.beta,
                    "smooth": spec.candidate.smooth,
                    "gamma": spec.candidate.gamma,
                }
            )
        map_rows.append(row)

        intersection = int(np.count_nonzero(reference_mask & mask))
        union = int(np.count_nonzero(reference_mask | mask))
        n_ref = int(np.count_nonzero(reference_mask))
        n_map = int(np.count_nonzero(mask))
        union_mask = reference_mask | mask
        if np.count_nonzero(union_mask) > 2:
            rho = float(
                spearmanr(
                    np.abs(reference_values[union_mask]),
                    np.abs(data[union_mask]),
                    nan_policy="omit",
                ).statistic
            )
        else:
            rho = np.nan
        overlap_rows.append(
            {
                "reference_map_id": "main_result_p90",
                "reference_label": "Full model",
                "map_id": map_id,
                "label": spec.label,
                "shared_voxels": intersection,
                "reference_voxels": n_ref,
                "map_voxels": n_map,
                "dice": float(2 * intersection / (n_ref + n_map)) if n_ref + n_map else np.nan,
                "jaccard": float(intersection / union) if union else np.nan,
                "overlap_coefficient": float(intersection / min(n_ref, n_map)) if min(n_ref, n_map) else np.nan,
                "center_of_mass_distance_mm": float(np.linalg.norm(np.asarray(center) - np.asarray(reference_center))),
                "union_abs_spearman": rho,
            }
        )

    groups, metadata = _build_roi_groups(reference_img, DEFAULT_AAL_VERSION, DEFAULT_ATLAS_CACHE_DIR)
    region_rows = []
    analysis_mask = np.zeros(reference_img.shape[:3], dtype=bool)
    for payload in loaded.values():
        analysis_mask |= payload["mask"]  # type: ignore[operator]
    region_sizes = {group.name: int(np.count_nonzero(group.mask & analysis_mask)) for group in groups}

    for map_id, payload in loaded.items():
        spec: MapSpec = payload["spec"]  # type: ignore[assignment]
        data = payload["data"]  # type: ignore[assignment]
        mask = payload["mask"]  # type: ignore[assignment]
        selected = np.column_stack(np.nonzero(mask)).astype(np.int32, copy=False)
        if selected.size == 0:
            continue
        x, y, z = selected.T
        assigned = np.zeros(selected.shape[0], dtype=np.int16)
        group_names = [UNASSIGNED_ROI] + [group.name for group in groups]
        for group_id, group in enumerate(groups, start=1):
            hit = group.mask[x, y, z] & (assigned == 0)
            assigned[hit] = group_id
        coords_mm = nib.affines.apply_affine(reference_img.affine, selected)
        values = data[x, y, z]
        total = int(selected.shape[0])
        for group_id in np.unique(assigned):
            positions = np.flatnonzero(assigned == group_id)
            if positions.size == 0:
                continue
            roi_name = group_names[int(group_id)]
            atlas_region_voxels = region_sizes.get(roi_name, np.nan)
            roi_values = values[positions]
            roi_coords = coords_mm[positions]
            region_rows.append(
                {
                    "map_id": map_id,
                    "label": spec.label,
                    "roi_name": roi_name,
                    "n_voxels": int(positions.size),
                    "percent_of_map": float(positions.size / total) if total else np.nan,
                    "atlas_region_voxels_in_analysis": atlas_region_voxels,
                    "percent_of_analysis_roi": (
                        float(positions.size / atlas_region_voxels)
                        if roi_name != UNASSIGNED_ROI and atlas_region_voxels
                        else np.nan
                    ),
                    "mean_value": float(np.mean(roi_values)),
                    "mean_abs_value": float(np.mean(np.abs(roi_values))),
                    "x_mm": float(np.mean(roi_coords[:, 0])),
                    "y_mm": float(np.mean(roi_coords[:, 1])),
                    "z_mm": float(np.mean(roi_coords[:, 2])),
                }
            )

    metadata["analysis_mask_n_voxels"] = int(np.count_nonzero(analysis_mask))
    return pd.DataFrame(map_rows), pd.DataFrame(overlap_rows), pd.DataFrame(region_rows), metadata


def _ordered_candidates(summary: pd.DataFrame, group: str, primary_only: bool = False) -> list[str]:
    rows = summary[summary["analysis_group"].eq(group)].copy()
    preferred = [
        "Full model",
        "No task",
        "No BOLD stability",
        "No beta stability",
        "No smoothness",
        "Task-only reference",
        "BOLD-only",
        "Beta-only",
        "Smooth-only",
        "No objective penalties",
    ]
    if primary_only:
        preferred = preferred[:6]
    labels = []
    for label in preferred:
        if label in set(rows["candidate_label"]):
            labels.append(label)
    labels.extend([label for label in rows["candidate_label"] if label not in labels])
    return labels


def _sem(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.std(values, ddof=1) / math.sqrt(values.size)) if values.size > 1 else np.nan


def plot_balanced_score(metrics: pd.DataFrame, group: str, out_base: Path) -> None:
    labels = _ordered_candidates(summarize_metrics(metrics), group)
    if not labels:
        return
    data = [
        metrics[(metrics["analysis_group"].eq(group)) & (metrics["candidate_label"].eq(label))]["balanced_score"].to_numpy(dtype=float)
        for label in labels
    ]
    fig, ax = plt.subplots(figsize=(max(8.0, 0.75 * len(labels)), 4.8), facecolor="white")
    positions = np.arange(len(labels))
    bp = ax.boxplot(data, positions=positions, widths=0.55, patch_artist=True, showfliers=False)
    for patch, label in zip(bp["boxes"], labels):
        patch.set_facecolor(COLORS.get(label, "#9ecae1"))
        patch.set_alpha(0.65)
        patch.set_edgecolor("#333333")
    for median in bp["medians"]:
        median.set_color("#111111")
        median.set_linewidth(1.6)
    rng = np.random.default_rng(3)
    for i, (label, values) in enumerate(zip(labels, data)):
        jitter = rng.normal(0, 0.035, size=len(values))
        ax.scatter(np.full(len(values), i) + jitter, values, s=22, color="#222222", alpha=0.55, linewidths=0)
    ax.axhline(0, color="#777777", lw=0.8, alpha=0.5)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Balanced ablation score (lower is better)")
    ax.set_title("Cross-fold balanced score", fontsize=12, weight="bold")
    ax.grid(axis="y", color="#dddddd", linewidth=0.8, alpha=0.7)
    fig.tight_layout()
    fig.savefig(f"{out_base}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{out_base}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_tradeoff(summary: pd.DataFrame, group: str, out_base: Path, title: str) -> None:
    sub = summary[summary["analysis_group"].eq(group)].copy()
    if sub.empty:
        return
    labels = _ordered_candidates(summary, group)
    sub["order"] = sub["candidate_label"].map({label: i for i, label in enumerate(labels)})
    sub = sub.sort_values("order")
    fig, ax = plt.subplots(figsize=(7.0, 5.2), facecolor="white")
    for row in sub.itertuples(index=False):
        color = COLORS.get(row.candidate_label, "#9ecae1")
        ax.errorbar(
            row.relative_loss_gap_mean,
            row.abs_test_corr_mean,
            xerr=row.relative_loss_gap_sem,
            yerr=row.abs_test_corr_sem,
            fmt="o",
            ms=8,
            color=color,
            ecolor=color,
            capsize=2.5,
            alpha=0.9,
        )
        ax.text(
            row.relative_loss_gap_mean,
            row.abs_test_corr_mean + 0.0025,
            row.candidate_label,
            fontsize=8,
            ha="center",
            va="bottom",
        )
    ax.set_xlabel("Relative train-test loss gap")
    ax.set_ylabel("Held-out |RT correlation|")
    ax.set_title(title, fontsize=12, weight="bold")
    ax.grid(color="#dddddd", linewidth=0.8, alpha=0.7)
    fig.tight_layout()
    fig.savefig(f"{out_base}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{out_base}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_spatial_similarity(overlap_df: pd.DataFrame, out_base: Path) -> None:
    keep = [
        "Full ablation map",
        "No task",
        "No BOLD stability",
        "No beta stability",
        "Task-only reference",
        "BOLD-only",
        "Beta-only",
        "Smooth-only",
        "No objective penalties",
    ]
    sub = overlap_df[overlap_df["label"].isin(keep) & ~overlap_df["map_id"].eq("main_result_p90")].copy()
    if sub.empty:
        return
    main = sub[sub["label"].eq("Full ablation map")]
    others = sub[~sub["label"].eq("Full ablation map")].sort_values("dice", ascending=False)
    sub = pd.concat([main, others], ignore_index=True)
    display_labels = {
        "Full ablation map": "Main model",
        "No task": "Task penalty removed",
        "No BOLD stability": "BOLD stability removed",
        "No beta stability": "Beta stability removed",
        "Task-only reference": "Task penalty only",
        "BOLD-only": "BOLD stability only",
        "Beta-only": "Beta stability only",
        "Smooth-only": "Smoothness penalty only",
        "No objective penalties": "Corr-only",
    }

    def metric_colors(values: pd.Series, cmap_name: str) -> list[tuple[float, float, float, float]]:
        finite_values = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
        finite_mask = np.isfinite(finite_values)
        levels = np.full(finite_values.shape, 0.55)
        if finite_mask.any():
            low = float(np.nanmin(finite_values[finite_mask]))
            high = float(np.nanmax(finite_values[finite_mask]))
            if high > low:
                levels[finite_mask] = 0.25 + 0.65 * (finite_values[finite_mask] - low) / (high - low)
            else:
                levels[finite_mask] = 0.65
        cmap = plt.get_cmap(cmap_name)
        return [cmap(float(level)) for level in levels]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), facecolor="white", sharey=True)
    y = np.arange(sub.shape[0])
    axes[0].barh(y, sub["dice"], color=metric_colors(sub["dice"], "Blues"), alpha=0.9)
    axes[0].set_xlabel("Dice with full reference map")
    axes[0].set_xlim(0, 1)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(sub["label"].map(display_labels))
    axes[1].barh(
        y,
        sub["center_of_mass_distance_mm"],
        color=metric_colors(sub["center_of_mass_distance_mm"], "Reds"),
        alpha=0.9,
    )
    axes[1].set_xlabel("Center-of-mass distance (mm)")
    axes[1].set_yticks(y)
    axes[1].tick_params(labelleft=False)
    for ax in axes:
        ax.grid(axis="x", color="#dddddd", linewidth=0.8, alpha=0.7)
    axes[0].invert_yaxis()
    fig.tight_layout()
    fig.savefig(f"{out_base}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{out_base}.pdf", bbox_inches="tight")
    plt.close(fig)


def _roi_matrix(region_df: pd.DataFrame, labels: list[str], value_col: str = "percent_of_map") -> pd.DataFrame:
    rows = region_df[region_df["label"].isin(labels) & ~region_df["roi_name"].eq(UNASSIGNED_ROI)].copy()
    if rows.empty:
        return pd.DataFrame()
    wide = rows.pivot_table(index="roi_name", columns="label", values=value_col, aggfunc="sum", fill_value=0.0)
    return wide.reindex(columns=[label for label in labels if label in wide.columns])


def plot_roi_heatmaps(region_df: pd.DataFrame, out_base: Path) -> None:
    labels = ["Full model", "Full ablation map", "No task", "No BOLD stability", "No beta stability", "Task-only reference"]
    wide = _roi_matrix(region_df, labels)
    if wide.empty:
        return
    top = wide.max(axis=1).sort_values(ascending=False).head(22).index
    wide = wide.loc[top]
    fig_height = max(6.0, 0.30 * len(wide) + 2.0)
    fig, ax = plt.subplots(figsize=(9.5, fig_height), facecolor="white")
    values = wide.to_numpy(dtype=float) * 100.0
    im = ax.imshow(values, aspect="auto", cmap="YlGnBu", vmin=0)
    ax.set_xticks(np.arange(len(wide.columns)))
    ax.set_xticklabels(wide.columns, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(wide.index)))
    ax.set_yticklabels([_display_region_name(name) for name in wide.index], fontsize=8)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            if value >= 2.0:
                ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=6.3, color="black")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("% of selected map")
    ax.set_title("Regional composition of ablation maps", fontsize=12, weight="bold")
    fig.tight_layout()
    fig.savefig(f"{out_base}_percent.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{out_base}_percent.pdf", bbox_inches="tight")
    plt.close(fig)

    if "Full model" in wide.columns:
        delta = wide.drop(columns=["Full model"], errors="ignore").subtract(wide["Full model"], axis=0) * 100.0
        if not delta.empty:
            vmax = max(5.0, float(np.nanmax(np.abs(delta.to_numpy(dtype=float)))))
            fig, ax = plt.subplots(figsize=(8.6, fig_height), facecolor="white")
            im = ax.imshow(delta.to_numpy(dtype=float), aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            ax.set_xticks(np.arange(len(delta.columns)))
            ax.set_xticklabels(delta.columns, rotation=30, ha="right")
            ax.set_yticks(np.arange(len(delta.index)))
            ax.set_yticklabels([_display_region_name(name) for name in delta.index], fontsize=8)
            for i in range(delta.shape[0]):
                for j in range(delta.shape[1]):
                    value = float(delta.iloc[i, j])
                    if abs(value) >= 2.0:
                        ax.text(j, i, f"{value:+.1f}", ha="center", va="center", fontsize=6.3, color="black")
            cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
            cbar.set_label("Change vs main full p90 (percentage points)")
            ax.set_title("Regional shifts induced by ablations", fontsize=12, weight="bold")
            fig.tight_layout()
            fig.savefig(f"{out_base}_delta.png", dpi=300, bbox_inches="tight")
            fig.savefig(f"{out_base}_delta.pdf", bbox_inches="tight")
            plt.close(fig)


def plot_publication_summary(metrics: pd.DataFrame, summary: pd.DataFrame, overlap_df: pd.DataFrame, region_df: pd.DataFrame, out_base: Path) -> None:
    group = "final_0p6"
    # Primary leave-one-out conditions + task-only floor reference
    primary_labels = ["Full model", "No task", "No BOLD stability", "No beta stability"]
    if "No smoothness" in set(summary["candidate_label"]):
        primary_labels.insert(4, "No smoothness")
    reference_labels = ["Task-only reference"]
    all_ab_labels = primary_labels + reference_labels
    if not set(primary_labels[:4]).issubset(set(summary["candidate_label"])):
        return

    fig = plt.figure(figsize=(13.2, 10.0), facecolor="white")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15], hspace=0.35, wspace=0.30)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    sub = summary[summary["analysis_group"].eq(group) & summary["candidate_label"].isin(all_ab_labels)].copy()
    sub["order"] = sub["candidate_label"].map({label: i for i, label in enumerate(all_ab_labels)})
    sub = sub.sort_values("order")
    x = np.arange(sub.shape[0])
    colors = [COLORS.get(label, "#9ecae1") for label in sub["candidate_label"]]
    hatches = ["///" if label in reference_labels else "" for label in sub["candidate_label"]]
    # draw separator between leave-one-out and reference baseline
    sep_x = len(primary_labels) - 0.5

    def _draw_panels(ax: plt.Axes, mean_col: str, sem_col: str, fold_col: str, ylabel: str, title: str) -> None:
        bars = ax.bar(x, sub[mean_col], yerr=sub[sem_col], capsize=3, color=colors, alpha=0.82)
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)
            bar.set_edgecolor("#555555" if hatch else "#333333")
        for i, row in enumerate(sub.itertuples(index=False)):
            fold_vals = metrics[
                metrics["analysis_group"].eq(group) & metrics["candidate_label"].eq(row.candidate_label)
            ][fold_col].to_numpy(dtype=float)
            ax.scatter(np.full(fold_vals.size, i), fold_vals, s=16, color="#222222", alpha=0.45, zorder=3)
        if len(primary_labels) < len(all_ab_labels):
            ax.axvline(sep_x, color="#aaaaaa", lw=1.0, linestyle="--", alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(sub["candidate_label"], rotation=25, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontsize=11, weight="bold")
        ax.grid(axis="y", color="#dddddd", alpha=0.7)

    _draw_panels(ax_a, "abs_test_corr_mean", "abs_test_corr_sem", "abs_test_corr",
                 "Held-out |RT correlation|", "A. Behavioural generalization")
    _draw_panels(ax_b, "relative_loss_gap_mean", "relative_loss_gap_sem", "relative_loss_gap",
                 "Relative train-test loss gap", "B. Cross-fold stability")

    spatial_labels = ["Full ablation map", "No task", "No BOLD stability", "No beta stability", "Task-only reference"]
    spatial = overlap_df[overlap_df["label"].isin(spatial_labels) & ~overlap_df["map_id"].eq("main_result_p90")].copy()
    spatial["order"] = spatial["label"].map({label: i for i, label in enumerate(spatial_labels)})
    spatial = spatial.sort_values("order")
    y = np.arange(spatial.shape[0])
    ax_c.barh(y, spatial["dice"], color=[COLORS.get(label, "#9ecae1") for label in spatial["label"]], alpha=0.85)
    ax_c.set_yticks(y)
    ax_c.set_yticklabels(spatial["label"])
    ax_c.set_xlim(0, 1)
    ax_c.invert_yaxis()
    ax_c.set_xlabel("Dice with main full p90")
    ax_c.set_title("C. Spatial similarity", loc="left", fontsize=11, weight="bold")
    ax_c.grid(axis="x", color="#dddddd", alpha=0.7)
    for i, value in enumerate(spatial["dice"]):
        ax_c.text(value + 0.015, i, f"{value:.2f}", va="center", fontsize=8)

    roi_labels = ["Full ablation map", "No task", "No BOLD stability", "No beta stability", "Task-only reference"]
    wide = _roi_matrix(region_df, ["Full model"] + roi_labels)
    if not wide.empty and "Full model" in wide.columns:
        delta = wide[roi_labels].subtract(wide["Full model"], axis=0) * 100.0
        top_idx = delta.abs().max(axis=1).sort_values(ascending=False).head(14).index
        delta = delta.loc[top_idx]
        vmax = max(4.0, float(np.nanmax(np.abs(delta.to_numpy(dtype=float)))))
        im = ax_d.imshow(delta.to_numpy(dtype=float), aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax_d.set_xticks(np.arange(delta.shape[1]))
        ax_d.set_xticklabels(delta.columns, rotation=30, ha="right", fontsize=8)
        ax_d.set_yticks(np.arange(delta.shape[0]))
        ax_d.set_yticklabels([_display_region_name(name) for name in delta.index], fontsize=8)
        cbar = fig.colorbar(im, ax=ax_d, fraction=0.046, pad=0.02)
        cbar.set_label("Percentage-point change vs main full p90", fontsize=8)
    ax_d.set_title("D. Regional redistribution", loc="left", fontsize=11, weight="bold")

    fig.savefig(f"{out_base}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{out_base}.pdf", bbox_inches="tight")
    plt.close(fig)


def _html_sprite_volumes(html_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required to read embedded PNG overlays from HTML maps.") from exc

    html = html_path.read_text(encoding="utf-8", errors="replace")
    encoded_images = re.findall(r'src="data:image/png;base64,([^"]+)"', html)
    if len(encoded_images) < 3:
        raise RuntimeError(f"Could not find the overlay PNG in {html_path}.")
    cfg_match = re.search(r"brainsprite\((\{.*?\})\);", html, flags=re.S)
    if cfg_match is None:
        raise RuntimeError(f"Could not find the brainsprite config in {html_path}.")

    cfg = json.loads(cfg_match.group(1))
    nx, ny, nz = [int(cfg["nbSlice"][axis]) for axis in "XYZ"]

    def sprite_to_volume(sprite: np.ndarray) -> np.ndarray:
        tiles_per_row = sprite.shape[1] // ny
        if tiles_per_row <= 0:
            raise RuntimeError(f"Unexpected sprite dimensions in {html_path}.")
        volume = np.zeros((nx, ny, nz, sprite.shape[2]), dtype=sprite.dtype)
        for x_idx in range(nx):
            tile_col = x_idx % tiles_per_row
            tile_row = x_idx // tiles_per_row
            tile = sprite[tile_row * nz : (tile_row + 1) * nz, tile_col * ny : (tile_col + 1) * ny]
            volume[x_idx] = tile[::-1].transpose(1, 0, 2)
        return volume

    background_sprite = np.asarray(Image.open(BytesIO(base64.b64decode(encoded_images[0]))).convert("RGBA"))
    overlay_sprite = np.asarray(Image.open(BytesIO(base64.b64decode(encoded_images[2]))).convert("RGBA"))
    background = sprite_to_volume(background_sprite)[..., :3].mean(axis=-1).astype(float)
    mask = sprite_to_volume(overlay_sprite)[..., 3] > 0
    return background, mask, np.asarray(cfg["affine"], dtype=float)


def _html_overlay_mask(html_path: Path, reference_img: nib.Nifti1Image) -> np.ndarray:
    _, mask, html_affine = _html_sprite_volumes(html_path)
    if mask.shape != reference_img.shape[:3]:
        raise RuntimeError(
            f"{html_path} overlay shape {mask.shape} does not match reference shape {reference_img.shape[:3]}."
        )
    for axis in range(3):
        html_step = float(html_affine[axis, axis])
        ref_step = float(reference_img.affine[axis, axis])
        if html_step != 0.0 and ref_step != 0.0 and np.sign(html_step) != np.sign(ref_step):
            mask = np.flip(mask, axis=axis)
    return mask


def _resampled_label_mask(
    label_img: nib.Nifti1Image,
    label_values: list[int],
    reference_img: nib.Nifti1Image,
) -> np.ndarray:
    if label_img.shape[:3] == reference_img.shape[:3] and np.allclose(label_img.affine, reference_img.affine):
        data = np.rint(label_img.get_fdata()).astype(np.int32, copy=False)
    else:
        resampled = image.resample_to_img(
            label_img,
            reference_img,
            interpolation="nearest",
            force_resample=True,
            copy_header=True,
        )
        data = np.rint(resampled.get_fdata()).astype(np.int32, copy=False)
    return np.isin(data, label_values)


def _atlas_roi_mask(reference_img: nib.Nifti1Image, roi_names: tuple[str, ...]) -> np.ndarray:
    groups, _ = _build_roi_groups(reference_img, DEFAULT_AAL_VERSION, DEFAULT_ATLAS_CACHE_DIR)
    mask = np.zeros(reference_img.shape[:3], dtype=bool)
    requested = set(roi_names)
    for group in groups:
        if group.name in requested:
            mask |= group.mask
    return mask


def _mni_box_mask(reference_img: nib.Nifti1Image, bounds: dict[str, float]) -> np.ndarray:
    coords_ijk = np.column_stack(np.nonzero(np.ones(reference_img.shape[:3], dtype=bool)))
    coords_mm = nib.affines.apply_affine(reference_img.affine, coords_ijk)
    in_box = (
        (np.abs(coords_mm[:, 0]) <= bounds["x_abs"])
        & (coords_mm[:, 1] >= bounds["y_min"])
        & (coords_mm[:, 1] <= bounds["y_max"])
        & (coords_mm[:, 2] >= bounds["z_min"])
        & (coords_mm[:, 2] <= bounds["z_max"])
    )
    mask = np.zeros(reference_img.shape[:3], dtype=bool)
    mask[tuple(coords_ijk[in_box].T)] = True
    return mask


def _brainstem_mask(reference_img: nib.Nifti1Image) -> np.ndarray:
    atlas = datasets.fetch_atlas_harvard_oxford(
        HARVARD_OXFORD_SUBCORTICAL_ATLAS,
        data_dir=str(DEFAULT_ATLAS_CACHE_DIR),
        verbose=0,
    )
    atlas_img = atlas.maps if isinstance(atlas.maps, nib.Nifti1Image) else nib.load(atlas.maps)
    label_values = [idx for idx, label in enumerate(atlas.labels) if str(label) in BRAINSTEM_ATLAS_LABELS]
    mask = _resampled_label_mask(atlas_img, label_values, reference_img)
    mask |= _atlas_roi_mask(reference_img, BRAINSTEM_AAL_ROIS)
    mask |= _mni_box_mask(reference_img, BRAINSTEM_MNI_BOUNDS_MM)
    return mask


def _cut_indices_for_mask(mask: np.ndarray, axis: int, n_cuts: int = 6, min_gap: int = 5) -> list[int]:
    counts = mask.sum(axis=tuple(dim for dim in range(3) if dim != axis))
    selected: list[int] = []
    for idx in np.argsort(counts)[::-1]:
        if counts[idx] <= 0:
            break
        idx = int(idx)
        if all(abs(idx - previous) >= min_gap for previous in selected):
            selected.append(idx)
        if len(selected) == n_cuts:
            break
    if len(selected) < n_cuts:
        occupied = np.flatnonzero(counts > 0)
        for idx in np.linspace(int(occupied.min()), int(occupied.max()), n_cuts, dtype=int):
            if idx not in selected:
                selected.append(int(idx))
            if len(selected) == n_cuts:
                break
    return sorted(selected)


def _plane_slice(volume: np.ndarray, axis: int, index: int) -> np.ndarray:
    if axis == 0:
        return volume[index].T[::-1]
    if axis == 1:
        return volume[:, index, :].T[::-1]
    return volume[:, :, index].T[::-1]


def _pad_slice(values: np.ndarray, pad_y: int = 3, pad_x: int = 2) -> np.ndarray:
    return np.pad(values, ((pad_y, pad_y), (pad_x, pad_x)), mode="constant")


def _crop_slices(background: np.ndarray, masks: list[np.ndarray]) -> tuple[np.ndarray, list[np.ndarray]]:
    crop_mask = ndimage.binary_fill_holes(background > 0)
    for mask in masks:
        crop_mask |= mask
    y, x = np.where(crop_mask)
    if y.size == 0:
        return _pad_slice(background), [_pad_slice(mask) for mask in masks]
    y0, y1 = max(int(y.min()) - 4, 0), min(int(y.max()) + 5, background.shape[0])
    x0, x1 = max(int(x.min()) - 4, 0), min(int(x.max()) + 5, background.shape[1])
    return (
        _pad_slice(background[y0:y1, x0:x1]),
        [_pad_slice(mask[y0:y1, x0:x1]) for mask in masks],
    )


def _anatomy_rgba(background: np.ndarray, vmax: float) -> np.ndarray:
    brain = ndimage.binary_fill_holes(background > 0)
    rgba = plt.cm.gray(np.clip(background / vmax, 0, 1))
    rgba[~brain, 3] = 0
    return rgba


def _coord_mm(affine: np.ndarray, axis: int, index: int) -> float:
    return float(affine[axis, axis] * (index + 1) + affine[axis, 3])


def _add_mask_overlay(
    ax: plt.Axes,
    mask: np.ndarray,
    color: str,
    linewidth: float,
    fill_alpha: float,
) -> None:
    if np.any(mask):
        values = mask.astype(float)
        if fill_alpha > 0:
            ax.contourf(values, levels=[0.5, 1.5], colors=[color], alpha=fill_alpha, antialiased=True)
        ax.contour(values, levels=[0.5], colors=color, linewidths=linewidth)


def plot_full_vs_task_only_anatomy(
    reference_map: Path,
    full_html: Path,
    task_only_map: Path,
    out_base: Path,
    task_z_threshold: float = DEFAULT_TASK_ONLY_Z_THRESHOLD,
) -> pd.DataFrame:
    if not full_html.exists() or not task_only_map.exists():
        return pd.DataFrame()

    reference_img, _ = _load_data(reference_map)
    full_bg, full_mask, html_affine = _html_sprite_volumes(full_html)
    task_img, task_data = _load_data(task_only_map)
    if full_mask.shape != reference_img.shape[:3]:
        raise RuntimeError(
            f"{full_html} overlay shape {full_mask.shape} does not match reference shape {reference_img.shape[:3]}."
        )
    if task_img.shape[:3] != full_mask.shape:
        raise RuntimeError(f"{task_only_map} shape {task_img.shape[:3]} differs from {full_html}.")

    task_mask = np.isfinite(task_data) & (task_data >= task_z_threshold)
    for axis in range(3):
        task_step = float(task_img.affine[axis, axis])
        html_step = float(html_affine[axis, axis])
        if task_step != 0.0 and html_step != 0.0 and np.sign(task_step) != np.sign(html_step):
            task_mask = np.flip(task_mask, axis=axis)

    display_img = nib.Nifti1Image(np.zeros(full_mask.shape, dtype=np.uint8), html_affine)
    brainstem_mask = _brainstem_mask(display_img)
    motor_mask = _atlas_roi_mask(display_img, MOTOR_CONTOUR_ROIS)

    raw_full_mask = full_mask.copy()
    raw_task_mask = task_mask.copy()
    raw_shared_mask = raw_full_mask & raw_task_mask
    full_mask &= ~brainstem_mask
    task_mask &= ~brainstem_mask
    union_mask = full_mask | task_mask
    if not np.any(union_mask):
        return pd.DataFrame()

    shared_mask = full_mask & task_mask
    motor_shared_mask = shared_mask & motor_mask
    full_only_mask = full_mask & ~task_mask
    task_only_unique_mask = task_mask & ~full_mask

    n_full_raw = int(np.count_nonzero(raw_full_mask))
    n_task_raw = int(np.count_nonzero(raw_task_mask))
    n_shared_raw = int(np.count_nonzero(raw_shared_mask))
    n_full = int(np.count_nonzero(full_mask))
    n_task = int(np.count_nonzero(task_mask))
    n_shared = int(np.count_nonzero(shared_mask))
    n_union = int(np.count_nonzero(union_mask))
    n_brainstem_full = n_full_raw - n_full
    n_brainstem_task = n_task_raw - n_task
    n_brainstem_shared = n_shared_raw - n_shared
    n_motor_shared = int(np.count_nonzero(motor_shared_mask))
    mode_specs = [("x", 0), ("y", 1), ("z", 2)]
    cuts_by_mode = {mode: _cut_indices_for_mask(union_mask, axis, n_cuts=9, min_gap=4) for mode, axis in mode_specs}
    cut_summary = "|".join(
        f"{mode}:{';'.join(f'{_coord_mm(html_affine, axis, index):g}' for index in cuts_by_mode[mode])}"
        for mode, axis in mode_specs
    )
    summary = pd.DataFrame(
        [
            {
                "full_model_html": str(full_html),
                "task_only_map": str(task_only_map),
                "task_only_threshold": f"z >= {task_z_threshold:g}",
                "full_model_voxels": n_full,
                "task_only_voxels": n_task,
                "shared_voxels": n_shared,
                "raw_full_model_voxels": n_full_raw,
                "raw_task_only_voxels": n_task_raw,
                "raw_shared_voxels": n_shared_raw,
                "brainstem_suppressed_full_model_voxels": n_brainstem_full,
                "brainstem_suppressed_task_only_voxels": n_brainstem_task,
                "brainstem_suppressed_shared_voxels": n_brainstem_shared,
                "motor_overlap_voxels": n_motor_shared,
                "brainstem_mask_source": "Harvard-Oxford Brain-Stem + AAL3 brainstem nuclei + MNI brainstem box",
                "motor_overlap_rois": ";".join(MOTOR_CONTOUR_ROIS),
                "full_only_voxels": int(np.count_nonzero(full_only_mask)),
                "task_only_unique_voxels": int(np.count_nonzero(task_only_unique_mask)),
                "union_voxels": n_union,
                "dice": float(2 * n_shared / (n_full + n_task)),
                "jaccard": float(n_shared / n_union),
                "slice_cuts_mm": cut_summary,
            }
        ]
    )
    summary.to_csv(f"{out_base}_summary.csv", index=False)

    colors = {
        "full": "#0072b2",
        "task": "#009e73",
        "shared": "#cc79a7",
    }
    full_bg = np.nan_to_num(full_bg, nan=0.0)
    vmax = float(np.percentile(full_bg[full_bg > 0], 99.5)) if np.any(full_bg > 0) else 1.0

    n_cols = max(len(cuts) for cuts in cuts_by_mode.values())
    fig, axes = plt.subplots(3, n_cols, figsize=(11.4, 4.45), facecolor="white")
    for row, (mode, axis) in enumerate(mode_specs):
        cuts = cuts_by_mode[mode]
        for col in range(n_cols):
            ax = axes[row, col]
            if col >= len(cuts):
                ax.set_axis_off()
                continue
            index = cuts[col]
            bg_slice, mask_slices = _crop_slices(
                _plane_slice(full_bg, axis, index),
                [
                    _plane_slice(full_only_mask, axis, index),
                    _plane_slice(task_only_unique_mask, axis, index),
                    _plane_slice(shared_mask, axis, index),
                    _plane_slice(motor_shared_mask, axis, index),
                ],
            )
            full_only_slice, task_only_unique_slice, shared_slice, motor_shared_slice = mask_slices
            ax.imshow(_anatomy_rgba(bg_slice, vmax), interpolation="nearest")
            _add_mask_overlay(ax, full_only_slice, colors["full"], 0.55, 0.24)
            _add_mask_overlay(ax, task_only_unique_slice, colors["task"], 0.55, 0.22)
            _add_mask_overlay(ax, shared_slice, colors["shared"], 0.75, 0.42)
            _add_mask_overlay(ax, motor_shared_slice, colors["shared"], 1.8, 0.0)
            coord = _coord_mm(html_affine, axis, index)
            ax.text(0.02, -0.03, f"{mode}={coord:g}", transform=ax.transAxes, ha="left", va="top", fontsize=5.6)
            if mode in {"y", "z"}:
                ax.text(0.12, 1.02, "L", transform=ax.transAxes, ha="center", va="bottom", fontsize=5.6)
                ax.text(0.88, 1.02, "R", transform=ax.transAxes, ha="center", va="bottom", fontsize=5.6)
            ax.set_facecolor("none")
            ax.patch.set_alpha(0)
            ax.set_axis_off()

    handles = [
        Patch(facecolor=colors["full"], edgecolor=colors["full"], alpha=0.34, label=f"Full model only ({np.count_nonzero(full_only_mask):,})"),
        Patch(facecolor=colors["task"], edgecolor=colors["task"], alpha=0.34, label=f"Standard GLM only, z>={task_z_threshold:g} ({np.count_nonzero(task_only_unique_mask):,})"),
        Patch(facecolor=colors["shared"], edgecolor=colors["shared"], alpha=0.48, label=f"Overlap ({n_shared:,})"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=6.2, bbox_to_anchor=(0.5, 0.01))
    fig.subplots_adjust(left=0.01, right=0.995, top=0.995, bottom=0.12, wspace=0.02, hspace=0.18)
    fig.savefig(f"{out_base}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{out_base}.pdf", bbox_inches="tight")
    plt.close(fig)
    return summary


def plot_map_montage(map_specs: list[MapSpec], out_base: Path) -> None:
    labels = ["Full model", "Full ablation map", "No task", "No BOLD stability", "No beta stability", "Task-only reference"]
    specs = [spec for spec in map_specs if spec.label in labels and spec.path and spec.path.exists()]
    seen = set()
    unique_specs = []
    for spec in specs:
        if spec.label in seen:
            continue
        seen.add(spec.label)
        unique_specs.append(spec)
    unique_specs.sort(key=lambda spec: labels.index(spec.label) if spec.label in labels else len(labels))
    if not unique_specs:
        return

    reference_img, _ = _load_data(unique_specs[0].path)  # type: ignore[arg-type]
    template = datasets.load_mni152_template(resolution=2)
    bg_img = image.resample_to_img(template, reference_img, interpolation="continuous", force_resample=True, copy_header=True)
    bg = np.asarray(bg_img.get_fdata(), dtype=float)
    bg = np.nan_to_num(bg, nan=0.0)
    vmax = float(np.percentile(bg[bg > 0], 99.2)) if np.any(bg > 0) else 1.0
    cuts = [-24, -12, 0, 12, 24, 36]
    n_rows = len(unique_specs)
    n_cols = len(cuts)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10.8, 1.75 * n_rows), facecolor="white")
    if n_rows == 1:
        axes = np.asarray([axes])

    affine = reference_img.affine
    for row, spec in enumerate(unique_specs):
        img, data = _load_data(spec.path)  # type: ignore[arg-type]
        mask, _ = _mask_from_map(data, spec.threshold_percentile)
        if img.shape[:3] != reference_img.shape[:3] or not np.allclose(img.affine, affine):
            raise RuntimeError(f"{spec.path} differs from montage reference grid.")
        for col, z_mm in enumerate(cuts):
            ax = axes[row, col]
            k = int(round((z_mm - affine[2, 3]) / affine[2, 2]))
            k = min(max(k, 0), bg.shape[2] - 1)
            bg_slice = bg[:, :, k].T[::-1]
            mask_slice = mask[:, :, k].T[::-1]
            ax.imshow(bg_slice, cmap="gray", vmin=0, vmax=vmax, interpolation="nearest")
            overlay = np.ma.masked_where(~mask_slice, mask_slice)
            ax.imshow(overlay, cmap="autumn", vmin=0, vmax=1, alpha=0.78, interpolation="nearest")
            if row == 0:
                ax.set_title(f"z={z_mm:g}", fontsize=8, pad=2)
            ax.set_axis_off()
        axes[row, 0].text(
            -0.08,
            0.5,
            spec.label,
            transform=axes[row, 0].transAxes,
            ha="right",
            va="center",
            fontsize=9,
            weight="bold",
        )
    fig.subplots_adjust(left=0.18, right=0.99, top=0.96, bottom=0.02, wspace=0.02, hspace=0.06)
    fig.savefig(f"{out_base}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{out_base}.pdf", bbox_inches="tight")
    plt.close(fig)


def expected_missing_maps(ablation_dir: Path) -> pd.DataFrame:
    expected = [
        ("final_0p6", "No smoothness", Candidate(1.0, 0.6, 0.6, 0.0, 1.5)),
    ]
    rows = []
    for group, label, candidate in expected:
        expected_name = f"voxel_weights_mean_foldavg_sub9_ses1_{_candidate_id(candidate)}_bold_thr90.nii.gz"
        path = ablation_dir / expected_name
        rows.append(
            {
                "analysis_group": group,
                "candidate_label": label,
                "candidate_id": _candidate_id(candidate),
                "expected_path": str(path),
                "exists": path.exists(),
                "note": ""
                if path.exists()
                else "Final no-smooth map and metrics were not available in the retained final-parameter results.",
            }
        )
    return pd.DataFrame(rows)


def write_report(
    out_dir: Path,
    metric_summary: pd.DataFrame,
    overlap_df: pd.DataFrame,
    missing_maps: pd.DataFrame,
    balanced_weights: pd.DataFrame,
) -> None:
    lines = [
        "# Ablation Study Analysis",
        "",
        "## Inputs",
        "",
        f"- Main map: `{DEFAULT_MAIN_MAP}`; summarized as voxels above the {REFERENCE_PERCENTILE:g}th percentile of nonzero weights (top {100.0 - REFERENCE_PERCENTILE:g}%).",
        f"- Ablation maps: `{DEFAULT_ABLATION_DIR}`; existing `*_bold_thr90.nii.gz` maps were treated as already thresholded.",
        "- Correlation/generalization metrics: `data/ablation/slurm-11460024.out` and `data/ablation/slurm-11445550.out`.",
        "",
        "## Outputs",
        "",
        "- `ablation_fold_metrics.csv`: fold-level correlations, losses, term contributions, and balanced scores.",
        "- `ablation_metric_summary.csv`: candidate-level means and SEMs.",
        "- `ablation_spatial_overlap.csv`: Dice/Jaccard/center-of-mass distances relative to the main full p90 map.",
        "- `ablation_map_summary.csv`: selected-voxel counts and cluster summaries.",
        "- `ablation_roi_regions.csv`: AAL coarse ROI composition by map.",
        "- `ablation_publication_summary.{png,pdf}`: compact multi-panel publication figure.",
        "- `ablation_map_montage.{png,pdf}`: axial slice overview of available maps.",
        "- `ablation_full_vs_task_only_anatomy.{png,pdf}`: Figure-3-style sagittal/coronal/axial contour montage for the supplied full-model HTML map and standard-GLM z map.",
        "- `ablation_full_vs_task_only_anatomy_summary.csv`: voxel counts and overlap for the focused full-vs-task-only anatomy figure.",
        "",
        "## Notes",
        "",
        "- The final-weight SLURM log contains the `task=1, bold=0.6, beta=0.6, smooth=1.25, gamma=1.5` full model plus no-task/no-BOLD/no-beta and single-term baselines, but no final-weight no-smooth metrics.",
        "- `slurm-11445550.out` is mixed; only the parameter-independent task-only and no-objective baselines were retained. The unrelated `task=1, bold=1, beta=0.75, smooth=1.8` sweep was excluded.",
        "- Balanced scores were computed within the final-weight analysis group using inverse ranges of candidate-mean score components.",
        f"- The focused full-vs-task-only anatomy figure uses the selected-voxel overlay embedded in the full-model thresholded HTML map and `{DEFAULT_TASK_ONLY_MAP}` thresholded at z >= {DEFAULT_TASK_ONLY_Z_THRESHOLD:g}; brainstem contours are suppressed, filled blue overlays show full-model-only voxels, filled green overlays show standard-GLM-only voxels, and filled magenta overlays show overlap with stronger line weight over motor ROIs.",
        "",
        "## Balanced-Score Weights",
        "",
        balanced_weights.to_markdown(index=False) if not balanced_weights.empty else "No balanced-score weights were computed.",
        "",
        "## Missing Expected Maps",
        "",
        missing_maps.to_markdown(index=False) if not missing_maps.empty else "No missing expected maps.",
        "",
    ]

    if not metric_summary.empty:
        cols = [
            "analysis_group",
            "candidate_label",
            "abs_test_corr_mean",
            "corr_generalization_gap_mean",
            "relative_loss_gap_mean",
            "balanced_score_mean",
        ]
        lines.extend(["## Metric Summary", "", metric_summary[cols].to_markdown(index=False), ""])
    if not overlap_df.empty:
        cols = ["label", "map_voxels", "shared_voxels", "dice", "center_of_mass_distance_mm"]
        keep = overlap_df[~overlap_df["map_id"].eq("main_result_p90")].copy()
        lines.extend(["## Spatial Summary", "", keep[cols].to_markdown(index=False), ""])
    (out_dir / "ablation_analysis_report.md").write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-map", type=Path, default=DEFAULT_MAIN_MAP)
    parser.add_argument("--ablation-dir", type=Path, default=DEFAULT_ABLATION_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--logs", type=Path, nargs="*", default=list(DEFAULT_LOGS))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for stale_name in STALE_UNRELATED_OUTPUTS:
        stale_path = args.out_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()

    metrics = parse_slurm_logs(args.logs)
    metrics, balanced_weights = add_balanced_scores(metrics)
    metric_summary = summarize_metrics(metrics)
    score_tests = compare_scores_to_full(metrics)

    map_specs = _discover_map_specs(args.main_map, args.ablation_dir)
    map_summary, overlap_df, region_df, atlas_metadata = summarize_maps(map_specs, args.out_dir)
    missing_maps = expected_missing_maps(args.ablation_dir)

    metrics.to_csv(args.out_dir / "ablation_fold_metrics.csv", index=False)
    metric_summary.to_csv(args.out_dir / "ablation_metric_summary.csv", index=False)
    balanced_weights.to_csv(args.out_dir / "ablation_balanced_score_weights.csv", index=False)
    score_tests.to_csv(args.out_dir / "ablation_score_tests_vs_full.csv", index=False)
    map_summary.to_csv(args.out_dir / "ablation_map_summary.csv", index=False)
    overlap_df.to_csv(args.out_dir / "ablation_spatial_overlap.csv", index=False)
    region_df.to_csv(args.out_dir / "ablation_roi_regions.csv", index=False)
    missing_maps.to_csv(args.out_dir / "ablation_missing_expected_maps.csv", index=False)
    (args.out_dir / "ablation_atlas_metadata.json").write_text(json.dumps(atlas_metadata, indent=2), encoding="utf-8")

    plot_tradeoff(
        metric_summary,
        "final_0p6",
        args.out_dir / "ablation_cv_tradeoff_final",
        "Final-weight ablation tradeoff",
    )
    plot_balanced_score(metrics, "final_0p6", args.out_dir / "ablation_balanced_score_final")
    plot_spatial_similarity(overlap_df, args.out_dir / "ablation_spatial_similarity")
    plot_roi_heatmaps(region_df, args.out_dir / "ablation_roi_heatmap")
    plot_publication_summary(metrics, metric_summary, overlap_df, region_df, args.out_dir / "ablation_publication_summary")
    plot_full_vs_task_only_anatomy(
        args.main_map,
        DEFAULT_FULL_MODEL_HTML,
        DEFAULT_TASK_ONLY_MAP,
        args.out_dir / "ablation_full_vs_task_only_anatomy",
    )
    plot_map_montage(map_specs, args.out_dir / "ablation_map_montage")

    write_report(args.out_dir, metric_summary, overlap_df, missing_maps, balanced_weights)

    print(f"Parsed {len(metrics)} fold-level metric rows.")
    print(f"Summarized {len(map_summary)} maps.")
    print(f"Saved ablation outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
