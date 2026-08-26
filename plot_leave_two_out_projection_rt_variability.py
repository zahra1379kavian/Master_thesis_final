#!/usr/bin/env python3
"""Held-out projection-versus-RT variability for leave-two-subjects-out maps."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import stats

from projected_sig_vs_RT import (
    DEFAULT_BEHAVIOUR_DIR,
    DEFAULT_DATA_DIR,
    _build_run_metric_table,
    _save_behavior_projection_figure,
    _subject_level_pairs,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHT_DIR = ROOT / "data" / "Leave_two_out"
DEFAULT_FOLD_SUBJECTS = DEFAULT_WEIGHT_DIR / "fold_held_out_subjects.csv"
DEFAULT_OUT_DIR = ROOT / "figures" / "projected_RT" / "leave_two_out_top10"
WEIGHT_GLOB = "abs_voxel_weights_lgso9_fold*.nii.gz"
FOLD_RE = re.compile(r"fold(?P<fold>\d+)")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply each leave-two-subjects-out weight map only to its held-out "
            "subjects and compare projection variability with RT variability."
        )
    )
    parser.add_argument("--weight-dir", type=Path, default=DEFAULT_WEIGHT_DIR)
    parser.add_argument(
        "--fold-subjects",
        type=Path,
        default=DEFAULT_FOLD_SUBJECTS,
        help="CSV with fold and held_out_subject columns; no fold membership is inferred.",
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--behaviour-dir", type=Path, default=DEFAULT_BEHAVIOUR_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--weight-percentile",
        type=float,
        default=90.0,
        help="Keep weights at or above this percentile of finite nonzero weights (default: 90).",
    )
    parser.add_argument(
        "--projection-source",
        choices=("bold", "beta"),
        default="bold",
        help="Projection input; bold reproduces the source panel.",
    )
    parser.add_argument("--bold-trial-reducer", choices=("median", "mean"), default="median")
    parser.add_argument("--behaviour-column", type=int, default=1)
    return parser.parse_args()


def _fold_number(path: Path) -> int:
    match = FOLD_RE.search(path.name)
    if not match:
        raise ValueError(f"Could not parse fold number from {path.name}")
    return int(match.group("fold"))


def _discover_weight_maps(weight_dir: Path) -> list[Path]:
    paths = sorted(weight_dir.glob(WEIGHT_GLOB), key=_fold_number)
    if not paths:
        raise FileNotFoundError(f"No fold maps matching {WEIGHT_GLOB!r} under {weight_dir}")
    fold_numbers = [_fold_number(path) for path in paths]
    expected = list(range(1, len(paths) + 1))
    if fold_numbers != expected:
        raise ValueError(f"Expected consecutive fold maps {expected}, found {fold_numbers}")
    return paths


def _held_out_subjects(mapping_path: Path, n_folds: int) -> dict[int, list[str]]:
    mapping = pd.read_csv(mapping_path)
    required_columns = {"fold", "held_out_subject"}
    missing_columns = required_columns - set(mapping.columns)
    if missing_columns:
        raise ValueError(f"Fold mapping is missing columns {sorted(missing_columns)}: {mapping_path}")

    mapping = mapping.loc[:, ["fold", "held_out_subject"]].copy()
    mapping["fold"] = mapping["fold"].astype(int)
    mapping["held_out_subject"] = mapping["held_out_subject"].astype(str)
    expected_folds = set(range(1, n_folds + 1))
    found_folds = set(mapping["fold"])
    if found_folds != expected_folds:
        raise ValueError(f"Expected folds {sorted(expected_folds)}, found {sorted(found_folds)} in {mapping_path}")
    if mapping["held_out_subject"].duplicated().any():
        duplicates = sorted(mapping.loc[mapping["held_out_subject"].duplicated(False), "held_out_subject"].unique())
        raise ValueError(f"Held-out subjects must appear exactly once; duplicates: {duplicates}")
    if mapping.shape[0] != 2 * n_folds:
        raise ValueError(
            f"Expected {2 * n_folds} mapping rows for {n_folds} leave-two-out folds, "
            f"found {mapping.shape[0]} in {mapping_path}"
        )

    fold_subjects: dict[int, list[str]] = {}
    for fold, fold_df in mapping.groupby("fold", sort=True):
        subjects = fold_df["held_out_subject"].tolist()
        if len(subjects) != 2:
            raise ValueError(f"Fold {fold} has {len(subjects)} held-out subjects, expected 2: {subjects}")
        fold_subjects[int(fold)] = subjects
    return fold_subjects


def _threshold_weights(path: Path, percentile: float) -> tuple[np.ndarray, dict[str, float | int | str]]:
    image = nib.load(str(path))
    weights = image.get_fdata(dtype=np.float32)
    source_mask = np.isfinite(weights) & (weights != 0)
    source_values = weights[source_mask]
    if source_values.size == 0:
        raise ValueError(f"No finite nonzero weights in {path}")

    threshold = float(np.percentile(source_values, percentile))
    selected_mask = source_mask & (weights >= threshold)
    selected_weights = np.where(selected_mask, weights, 0.0).astype(np.float32)
    n_source = int(np.count_nonzero(source_mask))
    n_selected = int(np.count_nonzero(selected_mask))
    metadata: dict[str, float | int | str] = {
        "weight_map": str(path),
        "weight_percentile": float(percentile),
        "threshold_value": threshold,
        "n_nonzero_finite_weights": n_source,
        "n_selected_voxels": n_selected,
        "selected_fraction": n_selected / n_source,
        "selection_rule": f"finite nonzero weights >= p{percentile:g}",
    }
    return selected_weights, metadata


def _subject_metrics(run_metrics: pd.DataFrame) -> pd.DataFrame:
    projection_col = "adjacent_diff_ratio_sum_projection"
    behaviour_col = "adjacent_diff_ratio_sum_behavior_col2"
    paired = run_metrics.loc[
        np.isfinite(run_metrics[projection_col]) & np.isfinite(run_metrics[behaviour_col])
    ].copy()
    paired["projection_raw"] = paired[projection_col].astype(float)
    paired["behavior_raw"] = paired[behaviour_col].astype(float)
    return _subject_level_pairs(paired)


def _paired_summary(subject_metrics: pd.DataFrame, scope: str) -> dict[str, float | int | str]:
    behaviour = subject_metrics["behavior_raw"].to_numpy(dtype=float)
    projection = subject_metrics["projection_raw"].to_numpy(dtype=float)
    keep = np.isfinite(behaviour) & np.isfinite(projection)
    behaviour = behaviour[keep]
    projection = projection[keep]
    difference = behaviour - projection
    n_subjects = int(difference.size)
    if n_subjects < 2:
        raise ValueError(f"At least two finite subject pairs are required for {scope}")

    mean_difference = float(np.mean(difference))
    sem_difference = float(stats.sem(difference))
    critical_t = float(stats.t.ppf(0.975, df=n_subjects - 1))
    test = stats.ttest_rel(behaviour, projection)
    difference_sd = float(np.std(difference, ddof=1))
    return {
        "scope": scope,
        "n_subjects": n_subjects,
        "mean_behaviour_variability": float(np.mean(behaviour)),
        "mean_projection_variability": float(np.mean(projection)),
        "paired_effect": mean_difference,
        "paired_effect_direction": "behaviour_minus_projection",
        "ci95_low": mean_difference - critical_t * sem_difference,
        "ci95_high": mean_difference + critical_t * sem_difference,
        "ci_method": "two-sided t interval across subject-paired differences",
        "paired_t": float(test.statistic),
        "degrees_of_freedom": n_subjects - 1,
        "p_value_two_sided": float(test.pvalue),
        "cohens_dz": mean_difference / difference_sd if difference_sd > 0 else np.nan,
        "inference_unit": "held-out subject",
    }


def _write_report(path: Path, summary: dict[str, float | int | str], n_folds: int, percentile: float) -> None:
    effect = float(summary["paired_effect"])
    ci_low = float(summary["ci95_low"])
    ci_high = float(summary["ci95_high"])
    p_value = float(summary["p_value_two_sided"])
    p_text = "< 0.001" if p_value < 0.001 else f"= {p_value:.3f}"
    report = (
        "# Held-out projection variability versus RT variability\n\n"
        f"Across {int(summary['n_subjects'])} held-out subjects from {n_folds} leave-two-subjects-out "
        f"folds, mean RT variability was {float(summary['mean_behaviour_variability']):.3f} and mean "
        f"projection variability was {float(summary['mean_projection_variability']):.3f}. The paired "
        f"effect (RT minus projection variability) was {effect:.3f} (95% CI {ci_low:.3f} to "
        f"{ci_high:.3f}), paired t({int(summary['degrees_of_freedom'])}) = "
        f"{float(summary['paired_t']):.3f}, p {p_text}.\n\n"
        "## Analysis definition\n\n"
        f"- Each fold map was applied only to that fold's two held-out subjects.\n"
        f"- Selected voxels were the top {100.0 - percentile:g}% of finite nonzero fold-map weights "
        f"(weights at or above p{percentile:g}).\n"
        "- Consecutive-trial variability and run-to-subject aggregation match `projected_sig_vs_RT.py`.\n"
        "- The confidence interval and p-value use subject-level paired differences; folds are not "
        "treated as independent observations.\n"
    )
    path.write_text(report, encoding="ascii")


def main() -> None:
    args = _parse_args()
    if not 0.0 <= args.weight_percentile <= 100.0:
        raise ValueError("--weight-percentile must be between 0 and 100")

    weight_maps = _discover_weight_maps(args.weight_dir)
    fold_subjects = _held_out_subjects(args.fold_subjects, len(weight_maps))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_run_metrics = []
    all_subject_metrics = []
    fold_summaries = []
    mapping_rows = []
    weight_rows = []

    for weight_map in weight_maps:
        fold = _fold_number(weight_map)
        subjects = fold_subjects[fold]
        weights, weight_metadata = _threshold_weights(weight_map, args.weight_percentile)
        run_metrics, _ = _build_run_metric_table(
            data_dir=args.data_dir,
            behaviour_dir=args.behaviour_dir,
            weights=weights,
            projection_source=args.projection_source,
            behaviour_column=args.behaviour_column,
            bold_trial_reducer=args.bold_trial_reducer,
            include_subjects=set(subjects),
        )
        run_metrics.insert(0, "fold", fold)
        run_metrics.insert(1, "held_out", True)
        subject_metrics = _subject_metrics(run_metrics)
        subject_metrics.insert(0, "fold", fold)
        subject_metrics.insert(1, "held_out", True)
        fold_summary = _paired_summary(subject_metrics, scope=f"fold_{fold:02d}")
        fold_summary["fold"] = fold

        stem = f"fold{fold:02d}_projection_behavior_subject_panel"
        _save_behavior_projection_figure(run_metrics, args.out_dir, figure_stem=stem)
        run_metrics.to_csv(args.out_dir / f"fold{fold:02d}_run_metrics.csv", index=False)
        subject_metrics.to_csv(args.out_dir / f"fold{fold:02d}_subject_metrics.csv", index=False)

        all_run_metrics.append(run_metrics)
        all_subject_metrics.append(subject_metrics)
        fold_summaries.append(fold_summary)
        weight_rows.append({"fold": fold, **weight_metadata})
        mapping_rows.extend(
            {"fold": fold, "held_out_subject": subject, "weight_map": str(weight_map)}
            for subject in subjects
        )
        print(f"Fold {fold:02d}: held out {', '.join(subjects)}")

    pooled_run_metrics = pd.concat(all_run_metrics, ignore_index=True)
    pooled_subject_metrics = pd.concat(all_subject_metrics, ignore_index=True)
    if pooled_subject_metrics["sub_tag"].nunique() != pooled_subject_metrics.shape[0]:
        raise ValueError("A subject appears in more than one held-out fold")

    pooled_summary = _paired_summary(pooled_subject_metrics, scope="all_held_out_subjects")
    pooled_summary["n_folds"] = len(weight_maps)
    pooled_summary["weight_percentile"] = float(args.weight_percentile)
    pooled_summary["top_weight_fraction"] = 100.0 - float(args.weight_percentile)
    pooled_summary["projection_source"] = args.projection_source
    pooled_summary["bold_trial_reducer"] = args.bold_trial_reducer
    pooled_summary["behaviour_column_zero_based"] = args.behaviour_column
    pooled_summary["fold_assignment"] = f"explicit mapping from {args.fold_subjects}"

    _save_behavior_projection_figure(
        pooled_run_metrics,
        args.out_dir,
        figure_stem="held_out_projection_behavior_subject_panel",
    )
    pooled_run_metrics.to_csv(args.out_dir / "held_out_run_metrics.csv", index=False)
    pooled_subject_metrics.to_csv(args.out_dir / "held_out_subject_metrics.csv", index=False)
    pd.DataFrame(fold_summaries).to_csv(args.out_dir / "fold_paired_effects.csv", index=False)
    pd.DataFrame([pooled_summary]).to_csv(args.out_dir / "held_out_paired_effect.csv", index=False)
    pd.DataFrame(mapping_rows).to_csv(args.out_dir / "fold_held_out_subjects.csv", index=False)
    pd.DataFrame(weight_rows).to_csv(args.out_dir / "fold_weight_thresholds.csv", index=False)
    _write_report(
        args.out_dir / "held_out_report.md",
        pooled_summary,
        n_folds=len(weight_maps),
        percentile=args.weight_percentile,
    )

    print(f"Saved held-out results to {args.out_dir}")
    print(
        "Paired effect (behaviour - projection): "
        f"{float(pooled_summary['paired_effect']):.3f} "
        f"[95% CI {float(pooled_summary['ci95_low']):.3f}, "
        f"{float(pooled_summary['ci95_high']):.3f}], "
        f"p={float(pooled_summary['p_value_two_sided']):.6g}"
    )


if __name__ == "__main__":
    main()
