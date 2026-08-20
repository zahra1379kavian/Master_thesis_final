#!/usr/bin/env python3
"""Relate whole-signal projection means to state-matched bradykinesia."""

from __future__ import annotations

import argparse
import os
import re
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-gvs-updrs")
warnings.filterwarnings("ignore", message="Unable to import Axes3D.*")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from analyze_rt_updrs_associations import bh_fdr, load_updrs_bradykinesia


ROOT = Path(__file__).resolve().parent
DEFAULT_WORKBOOK = ROOT / "data" / "subjects_info" / "AllDressed_PD_Participant_Study_Visit_Info.xlsx"
TRIAL_FEATURES_NAME = "gvs_trial_signal_features.csv"
OUT_STEM = "projected_signal_mean_updrs_iii_bradykinesia"
EXPECTED_GVS_CONDITIONS = 9

PROJECTIONS = (
    {
        "key": "vigour_network",
        "title": "Vigour-network projection associations with bradykinesia",
        "directory": ROOT / "figures" / "gvs_projection_features_vs_sham",
    },
    {
        "key": "task_activation_map",
        "title": "Task-activation-map projection associations with bradykinesia",
        "directory": ROOT / "figures" / "gvs_task_map_bold_features_vs_sham",
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clinical-workbook", type=Path, default=DEFAULT_WORKBOOK)
    return parser.parse_args()


def canonical_projection_subject(value: object) -> str:
    match = re.search(r"(?:PS)?[_-]?PD\s*0*(\d+)", str(value), flags=re.IGNORECASE)
    if match is None:
        raise ValueError(f"Could not parse subject identifier: {value!r}")
    return f"sub-pd{int(match.group(1)):03d}"


def summarize_projection(directory: Path) -> pd.DataFrame:
    trial_features_path = directory / TRIAL_FEATURES_NAME
    if not trial_features_path.is_file():
        raise FileNotFoundError(f"Trial-level projection features not found: {trial_features_path}")
    trial_features = pd.read_csv(trial_features_path)
    required = {"subject", "session", "medication", "gvs_code", "mean_level"}
    missing = sorted(required - set(trial_features.columns))
    if missing:
        raise RuntimeError(f"{trial_features_path} is missing columns: {', '.join(missing)}")

    # Every trial has the same number of time points. Therefore, averaging its
    # mean_level values gives the grand mean of all projected-signal samples for
    # that subject/state, with active GVS and sham trials treated identically.
    subject_values = (
        trial_features.groupby(["subject", "session", "medication"], as_index=False)
        .agg(
            mean_projected_signal=("mean_level", "mean"),
            n_trials=("mean_level", "size"),
            n_gvs_conditions=("gvs_code", "nunique"),
        )
        .sort_values(["medication", "subject"])
    )
    incomplete = subject_values.loc[
        subject_values["n_gvs_conditions"].ne(EXPECTED_GVS_CONDITIONS),
        ["subject", "medication", "n_gvs_conditions"],
    ]
    if not incomplete.empty:
        raise RuntimeError(
            f"Expected all {EXPECTED_GVS_CONDITIONS} GVS/sham conditions per subject/state "
            f"for {directory}; found {incomplete.to_dict(orient='records')}"
        )
    return subject_values.reset_index(drop=True)


def match_updrs(projection: pd.DataFrame, updrs: pd.DataFrame) -> pd.DataFrame:
    clinical = updrs.copy()
    clinical["subject"] = clinical["subject"].map(canonical_projection_subject)
    matched = projection.merge(
        clinical,
        on=["subject", "medication"],
        how="left",
        validate="one_to_one",
    )
    matched = matched.dropna(
        subset=[
            "mean_projected_signal",
            "updrs_iii_bradykinesia_subscore",
        ]
    ).copy()
    if matched.groupby("medication")["subject"].nunique().min() < 4:
        raise RuntimeError("Fewer than four subjects have state-matched projection and UPDRS data")
    return matched.sort_values(["medication", "subject"]).reset_index(drop=True)


def analyze(matched: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for medication in ("OFF", "ON"):
        data = matched.loc[matched["medication"].eq(medication)]
        correlation = stats.spearmanr(
            data["updrs_iii_bradykinesia_subscore"].to_numpy(dtype=float),
            data["mean_projected_signal"].to_numpy(dtype=float),
        )
        rows.append(
            {
                "medication": medication,
                "n_subjects": len(data),
                "n_gvs_and_sham_conditions_per_subject": EXPECTED_GVS_CONDITIONS,
                "spearman_rho": float(correlation.statistic),
                "spearman_p_value": float(correlation.pvalue),
            }
        )
    results = pd.DataFrame(rows)
    results["spearman_fdr_q_value"] = bh_fdr(results["spearman_p_value"])
    return results


def plot_association(
    matched: pd.DataFrame,
    results: pd.DataFrame,
    title: str,
    output_stem: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0), constrained_layout=True, sharey=True)
    all_y = matched["mean_projected_signal"].to_numpy(dtype=float)
    y_min = min(0.0, float(np.min(all_y)))
    y_max = max(0.0, float(np.max(all_y)))
    y_span = y_max - y_min
    if y_span == 0.0:
        y_span = 1.0

    for ax, medication in zip(axes, ("OFF", "ON"), strict=True):
        data = matched.loc[matched["medication"].eq(medication)]
        x = data["updrs_iii_bradykinesia_subscore"].to_numpy(dtype=float)
        y = data["mean_projected_signal"].to_numpy(dtype=float)
        model = sm.OLS(y, sm.add_constant(x)).fit()
        grid = np.linspace(float(np.min(x)), float(np.max(x)), 200)
        prediction = model.get_prediction(sm.add_constant(grid)).summary_frame(alpha=0.05)
        ax.fill_between(
            grid,
            prediction["mean_ci_lower"].to_numpy(),
            prediction["mean_ci_upper"].to_numpy(),
            color="#4C78A8",
            alpha=0.18,
            linewidth=0,
        )
        ax.plot(grid, prediction["mean"], color="#245A86", linewidth=2.2)
        ax.scatter(
            x,
            y,
            s=82,
            color="#E07A5F",
            edgecolor="white",
            linewidth=0.9,
            zorder=3,
        )
        result = results.loc[results["medication"].eq(medication)].iloc[0]
        ax.text(
            0.04,
            0.96,
            (
                f"Spearman ρ = {result['spearman_rho']:.3f}, "
                f"p = {result['spearman_p_value']:.4g}\n"
                f"n = {int(result['n_subjects'])}"
            ),
            transform=ax.transAxes,
            va="top",
            fontsize=12,
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "white",
                "edgecolor": "#BBBBBB",
                "alpha": 0.9,
            },
        )
        ax.set_xlabel(
            f"MDS-UPDRS III bradykinesia subscore\n({medication} medication)",
            fontsize=13,
        )
        ax.set_ylabel("Mean projected signal", fontsize=13)
        ax.set_ylim(y_min - 0.08 * y_span, y_max + 0.14 * y_span)
        ax.tick_params(axis="both", labelsize=12)
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(title, fontsize=16, fontweight="bold")
    fig.savefig(output_stem.with_suffix(".png"), dpi=220)
    fig.savefig(output_stem.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    args = parse_args()
    updrs = load_updrs_bradykinesia(args.clinical_workbook)

    for spec in PROJECTIONS:
        directory = Path(spec["directory"])
        projection = summarize_projection(directory)
        matched = match_updrs(projection, updrs)
        results = analyze(matched)
        output_stem = directory / OUT_STEM
        matched.to_csv(output_stem.with_name(f"{OUT_STEM}_subject_data.csv"), index=False)
        results.to_csv(output_stem.with_name(f"{OUT_STEM}_spearman_correlations.csv"), index=False)
        plot_association(matched, results, str(spec["title"]), output_stem)
        print(f"\n{spec['key']}")
        print(results.to_string(index=False))
        print(f"Saved outputs with stem {output_stem}")


if __name__ == "__main__":
    main()
