#!/usr/bin/env python3
"""Relate reaction time to the state-matched MDS-UPDRS III bradykinesia subscore."""

from __future__ import annotations

import argparse
import os
import re
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-rt-updrs")
warnings.filterwarnings("ignore", message="Unable to import Axes3D.*")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from analyze_rt_clinical_associations import DEFAULT_BEHAVIOUR_DIR, load_behaviour


ROOT = Path(__file__).resolve().parent
DEFAULT_WORKBOOK = ROOT / "data/subjects_info/AllDressed_PD_Participant_Study_Visit_Info.xlsx"
DEFAULT_OUT_DIR = ROOT / "figures/clinical_rt_associations"
UPDRS_SHEET = "Sheet1"
UPDRS_COLUMN_RE = re.compile(r"^(PSPD\d+)_(OFF|ON)$", flags=re.IGNORECASE)
UPDRS_ITEM_CODE_RE = re.compile(r"\b(3\.(?:1[0-8]|[1-9])[A-E]?)\b", flags=re.IGNORECASE)
BRADYKINESIA_ITEM_CODES = (
    "3.4A",
    "3.4B",
    "3.5A",
    "3.5B",
    "3.6A",
    "3.6B",
    "3.7A",
    "3.7B",
    "3.8A",
    "3.8B",
    "3.14",
)
EXPECTED_BRADYKINESIA_SCORES = len(BRADYKINESIA_ITEM_CODES)
MAX_BRADYKINESIA_SUBSCORE = 4 * EXPECTED_BRADYKINESIA_SCORES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--behaviour-dir", type=Path, default=DEFAULT_BEHAVIOUR_DIR)
    parser.add_argument("--clinical-workbook", type=Path, default=DEFAULT_WORKBOOK)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def load_updrs_bradykinesia(workbook: Path) -> pd.DataFrame:
    """Calculate the state-specific bradykinesia subscore from items 3.4-3.8 and 3.14."""
    if not workbook.is_file():
        raise FileNotFoundError(f"Clinical workbook not found: {workbook}")
    source = pd.read_excel(workbook, sheet_name=UPDRS_SHEET, dtype=object)
    labels = source.iloc[:, 0].astype(str).str.strip()
    item_codes = labels.str.extract(UPDRS_ITEM_CODE_RE, expand=False).str.upper()
    bradykinesia_rows = item_codes.isin(BRADYKINESIA_ITEM_CODES)
    found_codes = set(item_codes.loc[bradykinesia_rows])
    expected_codes = set(BRADYKINESIA_ITEM_CODES)
    if found_codes != expected_codes or int(bradykinesia_rows.sum()) != EXPECTED_BRADYKINESIA_SCORES:
        raise RuntimeError(
            "Could not identify the expected MDS-UPDRS III bradykinesia components; "
            f"missing={sorted(expected_codes - found_codes)}, extra={sorted(found_codes - expected_codes)}"
        )

    rows: list[dict[str, object]] = []
    for column in source.columns[1:]:
        match = UPDRS_COLUMN_RE.fullmatch(str(column).strip())
        if match is None:
            continue
        item_scores = pd.to_numeric(source.loc[bradykinesia_rows, column], errors="coerce")
        n_items_present = int(item_scores.notna().sum())
        subscore = (
            float(item_scores.sum())
            if n_items_present == EXPECTED_BRADYKINESIA_SCORES
            else np.nan
        )
        if np.isfinite(subscore) and not 0 <= subscore <= MAX_BRADYKINESIA_SUBSCORE:
            raise RuntimeError(f"Out-of-range bradykinesia subscore for {column}: {subscore}")
        rows.append(
            {
                "subject": match.group(1).upper(),
                "medication": match.group(2).upper(),
                "updrs_iii_bradykinesia_subscore": subscore,
                "n_bradykinesia_items_present": n_items_present,
                "updrs_source_column": str(column),
            }
        )

    updrs = pd.DataFrame(rows)
    if updrs.empty:
        raise RuntimeError(
            f"No state-specific Part III bradykinesia scores found in {workbook} [{UPDRS_SHEET}]"
        )
    if updrs.duplicated(["subject", "medication"]).any():
        raise RuntimeError("Duplicate subject/medication Part III bradykinesia subscores")
    return updrs.sort_values(["subject", "medication"]).reset_index(drop=True)


def state_matched_data(behaviour_dir: Path, updrs: pd.DataFrame) -> pd.DataFrame:
    trials, _, _ = load_behaviour(behaviour_dir)
    reaction_time = (
        trials.groupby(["subject", "medication"], as_index=False)
        .agg(n_valid_trials=("rt_ms", "size"), median_rt_ms=("rt_ms", "median"))
    )
    matched = reaction_time.merge(
        updrs,
        on=["subject", "medication"],
        how="inner",
        validate="one_to_one",
    )
    matched = matched.dropna(subset=["updrs_iii_bradykinesia_subscore"])
    state_counts = matched.groupby("subject")["medication"].nunique()
    complete_subjects = state_counts[state_counts == 2].index
    matched = matched[matched["subject"].isin(complete_subjects)].copy()
    if matched["subject"].nunique() < 4:
        raise RuntimeError("Fewer than four participants have matched OFF and ON data")
    return matched.sort_values(["subject", "medication"]).reset_index(drop=True)


def bh_fdr(p_values: pd.Series) -> np.ndarray:
    p = p_values.to_numpy(dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.minimum(adjusted, 1.0)
    return output


def analyze(matched: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for state in ("OFF", "ON"):
        data = matched.loc[matched["medication"] == state]
        x = data["updrs_iii_bradykinesia_subscore"].to_numpy(dtype=float)
        y = data["median_rt_ms"].to_numpy(dtype=float)
        spearman = stats.spearmanr(x, y)
        rows.append(
            {
                "medication": state,
                "n_subjects": len(data),
                "spearman_rho": float(spearman.statistic),
                "spearman_p_value": float(spearman.pvalue),
            }
        )
    results = pd.DataFrame(rows)
    results["spearman_fdr_q_value"] = bh_fdr(results["spearman_p_value"])
    return results


def regression_plot(matched: pd.DataFrame, results: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0), constrained_layout=True)
    all_y = matched["median_rt_ms"].to_numpy(dtype=float)
    y_span = float(all_y.max() - all_y.min())
    for ax, state in zip(axes, ("OFF", "ON"), strict=True):
        data = matched.loc[matched["medication"] == state]
        x = data["updrs_iii_bradykinesia_subscore"].to_numpy(dtype=float)
        y = data["median_rt_ms"].to_numpy(dtype=float)
        model = sm.OLS(y, sm.add_constant(x)).fit()
        grid = np.linspace(x.min(), x.max(), 200)
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
        ax.scatter(x, y, s=82, color="#E07A5F", edgecolor="white", linewidth=0.9, zorder=3)
        result = results.loc[results["medication"] == state].iloc[0]
        ax.text(
            0.04,
            0.96,
            f"Spearman ρ = {result['spearman_rho']:.3f}, p = {result['spearman_p_value']:.4g}",
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
            f"MDS-UPDRS III bradykinesia subscore\n({state} medication)",
            fontsize=13,
        )
        ax.set_ylabel("Median reaction time (ms)", fontsize=13)
        ax.set_ylim(all_y.min() - 0.03 * y_span, all_y.max() + 0.12 * y_span)
        ax.tick_params(axis="both", labelsize=12)
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Reaction time associations with bradykinesia",
        fontsize=16,
        fontweight="bold",
    )
    fig.savefig(out_dir / "primary_rt_updrs_iii_bradykinesia_scatterplots.png", dpi=220)
    fig.savefig(out_dir / "primary_rt_updrs_iii_bradykinesia_scatterplots.pdf")
    plt.close(fig)


def write_report(args: argparse.Namespace, matched: pd.DataFrame, results: pd.DataFrame) -> None:
    incomplete = matched.loc[
        matched["n_bradykinesia_items_present"] < EXPECTED_BRADYKINESIA_SCORES,
        ["subject", "medication", "n_bradykinesia_items_present"],
    ]
    if incomplete.empty:
        item_audit_note = "- All matched source profiles contain all 11 bradykinesia component scores."
    else:
        details = "; ".join(
            f"{row.subject} {row.medication}: {int(row.n_bradykinesia_items_present)}/11 items"
            for row in incomplete.itertuples(index=False)
        )
        item_audit_note = "- Incomplete bradykinesia profile(s), excluded: " + details + "."
    report = [
        "# Reaction Time and MDS-UPDRS III Bradykinesia",
        "",
        "## Score and state choice",
        "",
        (
            "The only clinical predictor is the MDS-UPDRS III bradykinesia subscore, calculated as "
            "the sum of items 3.4A-B, 3.5A-B, 3.6A-B, 3.7A-B, 3.8A-B, and 3.14 (range 0-44). "
            "No other Part III items or the Part III total are used. OFF and ON subscores were not "
            "averaged: each was paired with median RT from the matching medication state."
        ),
        "",
        "This choice follows relevant precedents:",
        "",
        "- Published MDS-UPDRS analyses define bradykinesia as the sum of items 3.4-3.8 and 3.14: https://pmc.ncbi.nlm.nih.gov/articles/PMC12472460/",
        "- A reaction-time study examined bradykinesia separately from rigidity, tremor, and the UPDRS Part III sum: https://doi.org/10.1155/2014/848035",
        "- A medication/reaction-time study assessed MDS-UPDRS III and reaction time separately in OFF and ON sessions: https://doi.org/10.14814/phy2.16150",
        "",
        "## Results",
        "",
        results.to_markdown(index=False, floatfmt=".4g"),
        "",
        "## Data audit",
        "",
        f"- Behaviour source: `{args.behaviour_dir}`",
        f"- UPDRS source: `{args.clinical_workbook}` [`{UPDRS_SHEET}`]",
        "- `Sheet1` was used because it contains paired OFF/ON examinations; the separate ON-only `Sheet2` values were not mixed into this paired analysis.",
        f"- Participants with both state-matched observations: {matched['subject'].nunique()}",
        item_audit_note,
        "- Spearman p values are two-sided; q values control FDR across the two state-specific Spearman tests.",
        "- The fitted line and shaded 95% mean-confidence band are descriptive OLS visual guides; all reported correlation tests are Spearman rank correlations.",
        "",
        "These are associations in a small sample and do not establish causality.",
    ]
    (args.out_dir / "rt_updrs_iii_bradykinesia_association_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    updrs = load_updrs_bradykinesia(args.clinical_workbook)
    matched = state_matched_data(args.behaviour_dir, updrs)
    results = analyze(matched)
    matched.to_csv(
        args.out_dir / "updrs_iii_bradykinesia_state_matched_subject_data.csv", index=False
    )
    results.to_csv(
        args.out_dir / "updrs_iii_bradykinesia_state_matched_correlations.csv", index=False
    )
    regression_plot(matched, results, args.out_dir)
    write_report(args, matched, results)
    print(results.to_string(index=False))
    print(f"Saved MDS-UPDRS III bradykinesia association outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
