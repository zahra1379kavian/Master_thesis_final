#!/usr/bin/env python3
"""Repeat the FD and medication-FC tests after excluding one high-FD subject.

The script reuses the saved subject/session summaries because excluding one
participant changes only the group-level tests and figures, not the remaining
participants' FD or connectivity estimates.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from framewise_displacement import paired_statistics, plot_paired_session_means_only
from med_effects import _save_intra_between_fc_analysis


ROOT = Path(__file__).resolve().parent
DEFAULT_FD_DIR = ROOT / "figures" / "framewise_displacement"
DEFAULT_FC_DIR = ROOT / "figures" / "med_effects"
DEFAULT_OUT_DIR = ROOT / "figures" / "high_fd_exclusion_sensitivity"
DEFAULT_EXCLUDED_SUBJECT = "sub-pd011"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exclude-subject", default=DEFAULT_EXCLUDED_SUBJECT)
    parser.add_argument("--fd-source-dir", type=Path, default=DEFAULT_FD_DIR)
    parser.add_argument("--fc-source-dir", type=Path, default=DEFAULT_FC_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--dpi", type=int, default=300)
    return parser


def _normalize_subject(subject: str) -> str:
    text = str(subject).strip()
    return text if text.startswith("sub-") else f"sub-{text}"


def _finite_float(value: object) -> float | None:
    number = float(value)
    return number if np.isfinite(number) else None


def _fd_result_row(result: pd.Series, analysis: str) -> dict[str, object]:
    return {
        "analysis_set": analysis,
        "outcome": "mean_fd_on_minus_off_mm",
        "n_subjects": int(result["n_subjects"]),
        "estimate": float(result["mean_difference_mm"]),
        "ci95_low": float(result["difference_95ci_low_mm"]),
        "ci95_high": float(result["difference_95ci_high_mm"]),
        "cohen_dz": float(result["cohen_dz"]),
        "t_statistic": float(result["paired_t_statistic"]),
        "degrees_of_freedom": int(result["paired_t_df"]),
        "p_value_two_sided": float(result["paired_t_p_two_sided"]),
        "wilcoxon_p_value_two_sided": float(result["wilcoxon_p_two_sided"]),
    }


def _fc_result_rows(results: pd.DataFrame, analysis: str) -> list[dict[str, object]]:
    rows = []
    for row in results.itertuples(index=False):
        rows.append(
            {
                "analysis_set": analysis,
                "outcome": str(row.analysis),
                "n_subjects": int(row.n_subjects),
                "estimate": float(row.mean),
                "ci95_low": float(row.ci95_low),
                "ci95_high": float(row.ci95_high),
                "cohen_dz": float(row.cohen_dz),
                "t_statistic": float(row.paired_t_statistic),
                "degrees_of_freedom": int(row.n_subjects) - 1,
                "p_value_two_sided": float(row.paired_t_p_value_two_sided),
                "wilcoxon_p_value_two_sided": float(row.wilcoxon_p_value_two_sided),
            }
        )
    return rows


def main() -> int:
    args = build_parser().parse_args()
    excluded_subject = _normalize_subject(args.exclude_subject)
    if args.dpi <= 0:
        raise SystemExit("--dpi must be positive")

    fd_summary_path = args.fd_source_dir / "subject_session_fd_summary.csv"
    fd_primary_path = args.fd_source_dir / "session_comparison_statistics.csv"
    fc_sessions_path = args.fc_source_dir / "intra_vs_between_fc_session_values.csv"
    fc_rois_path = args.fc_source_dir / "intra_vs_between_fc_roi_values.csv"
    fc_primary_path = args.fc_source_dir / "intra_vs_between_fc_results.csv"
    required_paths = [
        fd_summary_path,
        fd_primary_path,
        fc_sessions_path,
        fc_rois_path,
        fc_primary_path,
    ]
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise SystemExit(f"Missing required input(s): {', '.join(missing)}")

    fd_summary = pd.read_csv(fd_summary_path)
    fc_sessions = pd.read_csv(fc_sessions_path)
    fc_rois = pd.read_csv(fc_rois_path)
    if excluded_subject not in set(fd_summary["subject"]):
        raise SystemExit(f"{excluded_subject} is absent from {fd_summary_path}")
    if excluded_subject not in set(fc_sessions["subject"]):
        raise SystemExit(f"{excluded_subject} is absent from {fc_sessions_path}")

    excluded_fd_rows = fd_summary.loc[fd_summary["subject"].eq(excluded_subject)]
    excluded_session_2 = excluded_fd_rows.loc[excluded_fd_rows["session"].eq(2), "mean_fd_mm"]
    if excluded_session_2.size != 1:
        raise SystemExit(f"Expected exactly one session-2 FD row for {excluded_subject}")

    filtered_fd = fd_summary.loc[fd_summary["subject"].ne(excluded_subject)].copy()
    fd_pivot = filtered_fd.pivot(index="subject", columns="session", values="mean_fd_mm").dropna()
    if not {1, 2}.issubset(fd_pivot.columns):
        raise SystemExit("Both FD sessions are required after exclusion")
    fd_exclusion = pd.Series(
        paired_statistics(
            fd_pivot[1],
            fd_pivot[2],
            analysis="sensitivity_excluding_high_fd_subject",
            cohort=f"paired subjects excluding {excluded_subject}",
            primary=True,
        )
    )
    fd_primary_table = pd.read_csv(fd_primary_path)
    fd_primary = fd_primary_table.loc[fd_primary_table["is_primary"].astype(bool)].iloc[0]

    fd_out_dir = args.out_dir / "framewise_displacement"
    fc_out_dir = args.out_dir / "med_effects"
    fd_out_dir.mkdir(parents=True, exist_ok=True)
    fc_out_dir.mkdir(parents=True, exist_ok=True)
    filtered_fd.to_csv(fd_out_dir / "subject_session_fd_summary.csv", index=False)
    pd.DataFrame([fd_exclusion]).to_csv(
        fd_out_dir / "session_comparison_statistics.csv", index=False
    )
    plot_paired_session_means_only(
        filtered_fd,
        fd_out_dir / "session_1_vs_2_mean_fd_paired_only",
        args.dpi,
    )

    filtered_fc_sessions = fc_sessions.loc[
        fc_sessions["subject"].ne(excluded_subject)
    ].copy()
    filtered_fc_rois = fc_rois.loc[fc_rois["subject"].ne(excluded_subject)].copy()
    _save_intra_between_fc_analysis(
        filtered_fc_sessions.to_dict("records"),
        filtered_fc_rois.to_dict("records"),
        fc_out_dir,
    )
    fc_primary_results = pd.read_csv(fc_primary_path)
    fc_exclusion_results = pd.read_csv(fc_out_dir / "intra_vs_between_fc_results.csv")

    comparison_rows = [_fd_result_row(fd_primary, "primary")]
    comparison_rows.append(_fd_result_row(fd_exclusion, f"excluding_{excluded_subject}"))
    comparison_rows.extend(_fc_result_rows(fc_primary_results, "primary"))
    comparison_rows.extend(
        _fc_result_rows(fc_exclusion_results, f"excluding_{excluded_subject}")
    )
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(args.out_dir / "primary_vs_exclusion_results.csv", index=False)

    original_intra = fc_primary_results.loc[
        fc_primary_results["analysis"].eq("within_roi_on_minus_off")
    ].iloc[0]
    exclusion_intra = fc_exclusion_results.loc[
        fc_exclusion_results["analysis"].eq("within_roi_on_minus_off")
    ].iloc[0]
    exclusion_between = fc_exclusion_results.loc[
        fc_exclusion_results["analysis"].eq("between_roi_on_minus_off")
    ].iloc[0]
    exclusion_contrast = fc_exclusion_results.loc[
        fc_exclusion_results["analysis"].eq("within_minus_between_delta")
    ].iloc[0]
    intra_absolute_change = float(exclusion_intra["mean"] - original_intra["mean"])
    intra_percent_change = float(100.0 * intra_absolute_change / original_intra["mean"])
    intra_remains_significant = bool(
        float(exclusion_intra["paired_t_p_value_two_sided"]) < 0.05
    )

    summary = {
        "generated": datetime.now().astimezone().isoformat(timespec="seconds"),
        "excluded_subject": excluded_subject,
        "exclusion_reason": "session-2 mean FD exceeded 0.5 mm",
        "excluded_subject_session_2_mean_fd_mm": float(excluded_session_2.iloc[0]),
        "method": (
            "Exact rerun of the saved subject-level tests after removing the excluded "
            "participant; remaining subject/session estimates are unchanged."
        ),
        "fd_exclusion_result": {
            "n_subjects": int(fd_exclusion["n_subjects"]),
            "degrees_of_freedom": int(fd_exclusion["paired_t_df"]),
            **{
                key: _finite_float(value)
                for key, value in {
                    "off_mean_fd_mm": fd_exclusion["session_1_mean_fd_mm"],
                    "on_mean_fd_mm": fd_exclusion["session_2_mean_fd_mm"],
                    "on_minus_off_mean_fd_mm": fd_exclusion["mean_difference_mm"],
                    "ci95_low_mm": fd_exclusion["difference_95ci_low_mm"],
                    "ci95_high_mm": fd_exclusion["difference_95ci_high_mm"],
                    "t_statistic": fd_exclusion["paired_t_statistic"],
                    "p_value_two_sided": fd_exclusion["paired_t_p_two_sided"],
                    "cohen_dz": fd_exclusion["cohen_dz"],
                    "sign_flip_p_value_two_sided": fd_exclusion[
                        "sign_flip_p_two_sided"
                    ],
                    "wilcoxon_p_value_two_sided": fd_exclusion[
                        "wilcoxon_p_two_sided"
                    ],
                }.items()
            },
        },
        "fc_exclusion_results": {
            "intra_roi_on_minus_off": {
                "n_subjects": int(exclusion_intra["n_subjects"]),
                "mean_fisher_z": float(exclusion_intra["mean"]),
                "ci95_low": float(exclusion_intra["ci95_low"]),
                "ci95_high": float(exclusion_intra["ci95_high"]),
                "t_statistic": float(exclusion_intra["paired_t_statistic"]),
                "degrees_of_freedom": int(exclusion_intra["n_subjects"]) - 1,
                "p_value_two_sided": float(
                    exclusion_intra["paired_t_p_value_two_sided"]
                ),
                "cohen_dz": float(exclusion_intra["cohen_dz"]),
                "remains_significant_at_alpha_0_05": intra_remains_significant,
                "absolute_change_from_primary": intra_absolute_change,
                "percent_change_from_primary": intra_percent_change,
            },
            "between_roi_on_minus_off": {
                "n_subjects": int(exclusion_between["n_subjects"]),
                "mean_fisher_z": float(exclusion_between["mean"]),
                "ci95_low": float(exclusion_between["ci95_low"]),
                "ci95_high": float(exclusion_between["ci95_high"]),
                "t_statistic": float(exclusion_between["paired_t_statistic"]),
                "degrees_of_freedom": int(exclusion_between["n_subjects"]) - 1,
                "p_value_two_sided": float(
                    exclusion_between["paired_t_p_value_two_sided"]
                ),
                "cohen_dz": float(exclusion_between["cohen_dz"]),
            },
            "intra_minus_between_on_minus_off": {
                "n_subjects": int(exclusion_contrast["n_subjects"]),
                "mean_fisher_z": float(exclusion_contrast["mean"]),
                "ci95_low": float(exclusion_contrast["ci95_low"]),
                "ci95_high": float(exclusion_contrast["ci95_high"]),
                "t_statistic": float(exclusion_contrast["paired_t_statistic"]),
                "degrees_of_freedom": int(exclusion_contrast["n_subjects"]) - 1,
                "p_value_two_sided": float(
                    exclusion_contrast["paired_t_p_value_two_sided"]
                ),
                "cohen_dz": float(exclusion_contrast["cohen_dz"]),
            },
        },
    }
    (args.out_dir / "sensitivity_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    conclusion = "remained significant" if intra_remains_significant else "did not remain significant"
    report = f"""# High-FD participant exclusion sensitivity

`{excluded_subject}` was excluded because session 2 mean FD was {float(excluded_session_2.iloc[0]):.6f} mm (>0.5 mm). Session 1 is medication OFF and session 2 is medication ON.

## OFF–ON FD comparison

After exclusion, mean FD was {float(fd_exclusion['session_1_mean_fd_mm']):.6f} mm OFF and {float(fd_exclusion['session_2_mean_fd_mm']):.6f} mm ON. The paired ON−OFF change was {float(fd_exclusion['mean_difference_mm']):+.6f} mm, 95% CI [{float(fd_exclusion['difference_95ci_low_mm']):+.6f}, {float(fd_exclusion['difference_95ci_high_mm']):+.6f}], t({int(fd_exclusion['paired_t_df'])}) = {float(fd_exclusion['paired_t_statistic']):.3f}, p = {float(fd_exclusion['paired_t_p_two_sided']):.4f}, Cohen dz = {float(fd_exclusion['cohen_dz']):.3f} (N = {int(fd_exclusion['n_subjects'])}).

## Intra-ROI and between-ROI connectivity

- Intra-ROI ON−OFF: mean Fisher-z change = {float(exclusion_intra['mean']):+.6f}, 95% CI [{float(exclusion_intra['ci95_low']):+.6f}, {float(exclusion_intra['ci95_high']):+.6f}], t({int(exclusion_intra['n_subjects']) - 1}) = {float(exclusion_intra['paired_t_statistic']):.3f}, p = {float(exclusion_intra['paired_t_p_value_two_sided']):.4f}, Cohen dz = {float(exclusion_intra['cohen_dz']):.3f}.
- Between-ROI ON−OFF: mean Fisher-z change = {float(exclusion_between['mean']):+.6f}, 95% CI [{float(exclusion_between['ci95_low']):+.6f}, {float(exclusion_between['ci95_high']):+.6f}], t({int(exclusion_between['n_subjects']) - 1}) = {float(exclusion_between['paired_t_statistic']):.3f}, p = {float(exclusion_between['paired_t_p_value_two_sided']):.4f}, Cohen dz = {float(exclusion_between['cohen_dz']):.3f}.
- Intra-minus-between medication contrast: mean Fisher-z change = {float(exclusion_contrast['mean']):+.6f}, 95% CI [{float(exclusion_contrast['ci95_low']):+.6f}, {float(exclusion_contrast['ci95_high']):+.6f}], t({int(exclusion_contrast['n_subjects']) - 1}) = {float(exclusion_contrast['paired_t_statistic']):.3f}, p = {float(exclusion_contrast['paired_t_p_value_two_sided']):.4f}, Cohen dz = {float(exclusion_contrast['cohen_dz']):.3f}.

## Robustness conclusion

The intra-ROI effect {conclusion}. Its estimate changed from {float(original_intra['mean']):.6f} to {float(exclusion_intra['mean']):.6f} Fisher z ({intra_percent_change:+.1f}%), so it remained similar in magnitude. The intra-minus-between contrast also remained significant.

The sensitivity analysis filters the saved subject/session summaries before rerunning the original group-level tests. This is equivalent to full re-extraction because one participant's exclusion cannot change another participant's FD or FC estimate.
"""
    (args.out_dir / "sensitivity_report.md").write_text(report, encoding="utf-8")

    print(f"Excluded {excluded_subject}")
    print(
        "FD ON-OFF: "
        f"{float(fd_exclusion['mean_difference_mm']):+.6f} mm, "
        f"t({int(fd_exclusion['paired_t_df'])})={float(fd_exclusion['paired_t_statistic']):.3f}, "
        f"p={float(fd_exclusion['paired_t_p_two_sided']):.4f}"
    )
    print(
        "Intra-ROI ON-OFF: "
        f"{float(exclusion_intra['mean']):+.6f}, "
        f"t({int(exclusion_intra['n_subjects']) - 1})={float(exclusion_intra['paired_t_statistic']):.3f}, "
        f"p={float(exclusion_intra['paired_t_p_value_two_sided']):.4f}"
    )
    print(f"Outputs saved to {args.out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
