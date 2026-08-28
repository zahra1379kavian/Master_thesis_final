#!/usr/bin/env python3
"""Fit medication random-intercept LME models to the two FC network analyses.

The script tests the aggregate intra-ROI and between-ROI values shown in the
target figures, every individual intra-ROI value, and every between-ROI edge.
OFF medication is the reference level, so the medication coefficient is ON
minus OFF.  Models use Fisher-z Pearson functional connectivity.
"""

from __future__ import annotations

import argparse
import json
import platform
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import statsmodels
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

import med_effects as M


ROOT = Path(__file__).resolve().parent
DEFAULT_VIGOUR_DIR = ROOT / "figures" / "med_effects"
DEFAULT_TASK_DIR = ROOT / "figures" / "med_effects_task_activation"
DEFAULT_OUT_DIR = ROOT / "figures" / "med_effects_roi_lme"
NETWORK_SPECS = (
    ("vigour", "weighted vigour network"),
    ("task_activation", "task-activation network"),
)
EXPECTED_METRIC = M.INTRA_BETWEEN_FC_METRIC_PEARSON


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vigour-dir", type=Path, default=DEFAULT_VIGOUR_DIR)
    parser.add_argument("--task-dir", type=Path, default=DEFAULT_TASK_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser


def _require_columns(table: pd.DataFrame, columns: set[str], path: Path) -> None:
    missing = sorted(columns - set(table.columns))
    if missing:
        raise RuntimeError(f"{path} is missing required columns: {', '.join(missing)}")


def _load_complete_sessions(source_dir: Path) -> tuple[pd.DataFrame, list[str], list[str]]:
    path = source_dir / "intra_vs_between_fc_session_values.csv"
    if not path.exists():
        raise FileNotFoundError(f"Session-value table does not exist: {path}")
    sessions = pd.read_csv(path)
    _require_columns(
        sessions,
        {
            "label",
            "subject",
            "session",
            "state",
            "connectivity_metric",
            "within_roi_mean_z",
            "between_roi_mean_z",
        },
        path,
    )
    sessions = sessions.copy()
    sessions["subject"] = sessions["subject"].astype(str)
    sessions["state"] = sessions["state"].astype(str).str.lower()
    metrics = set(sessions["connectivity_metric"].dropna().astype(str))
    if metrics != {EXPECTED_METRIC}:
        raise RuntimeError(
            f"{path} must contain only {EXPECTED_METRIC}; found {sorted(metrics)}"
        )
    if sessions.duplicated(["subject", "session"]).any():
        raise RuntimeError(f"{path} contains duplicate subject/session rows")
    unexpected_states = sorted(set(sessions["state"]) - {"off", "on"})
    if unexpected_states:
        raise RuntimeError(f"{path} contains unexpected medication states: {unexpected_states}")

    complete_subjects: list[str] = []
    excluded_subjects: list[str] = []
    for subject, values in sessions.groupby("subject", sort=True):
        state_counts = values["state"].value_counts().to_dict()
        target = complete_subjects if state_counts == {"off": 1, "on": 1} else excluded_subjects
        target.append(str(subject))
    if len(complete_subjects) < 2:
        raise RuntimeError(f"{path} has fewer than two complete OFF/ON subjects")

    sessions = sessions.loc[sessions["subject"].isin(complete_subjects)].copy()
    sessions["medication_on"] = sessions["state"].eq("on").astype(int)
    sessions = sessions.sort_values(["subject", "session"]).reset_index(drop=True)
    return sessions, complete_subjects, excluded_subjects


def _aggregate_values(sessions: pd.DataFrame, network: str) -> pd.DataFrame:
    common = ["label", "subject", "session", "state", "medication_on"]
    intra = sessions[common + ["within_roi_mean_z"]].rename(
        columns={"within_roi_mean_z": "fc_z"}
    )
    intra["fc_class"] = "intra_roi"
    intra["feature"] = "aggregate_intra_roi_mean"
    intra["roi_1"] = ""
    intra["roi_2"] = ""
    between = sessions[common + ["between_roi_mean_z"]].rename(
        columns={"between_roi_mean_z": "fc_z"}
    )
    between["fc_class"] = "between_roi"
    between["feature"] = "aggregate_between_roi_mean"
    between["roi_1"] = ""
    between["roi_2"] = ""
    values = pd.concat([intra, between], ignore_index=True)
    values.insert(0, "network", network)
    return values


def _intra_roi_values(
    source_dir: Path, sessions: pd.DataFrame, network: str
) -> pd.DataFrame:
    path = source_dir / "intra_vs_between_fc_roi_values.csv"
    if not path.exists():
        raise FileNotFoundError(f"Intra-ROI table does not exist: {path}")
    values = pd.read_csv(path)
    _require_columns(
        values,
        {
            "label",
            "subject",
            "session",
            "state",
            "connectivity_metric",
            "roi",
            "intra_roi_mean_z",
        },
        path,
    )
    metrics = set(values["connectivity_metric"].dropna().astype(str))
    if metrics != {EXPECTED_METRIC}:
        raise RuntimeError(
            f"{path} must contain only {EXPECTED_METRIC}; found {sorted(metrics)}"
        )
    if values.duplicated(["subject", "session", "roi"]).any():
        raise RuntimeError(f"{path} contains duplicate subject/session/ROI rows")

    session_keys = sessions[["label", "subject", "session", "state", "medication_on"]]
    values = values.merge(
        session_keys,
        on=["label", "subject", "session", "state"],
        how="inner",
        validate="many_to_one",
    )
    output = values[
        ["label", "subject", "session", "state", "medication_on", "roi", "intra_roi_mean_z"]
    ].rename(columns={"intra_roi_mean_z": "fc_z", "roi": "feature"})
    output["fc_class"] = "intra_roi"
    output["roi_1"] = output["feature"]
    output["roi_2"] = ""
    output.insert(0, "network", network)
    return output


def _between_roi_values(
    source_dir: Path, sessions: pd.DataFrame, network: str
) -> tuple[pd.DataFrame, float]:
    rows: list[dict[str, object]] = []
    maximum_aggregate_error = 0.0
    expected_rois: list[str] | None = None
    for session in sessions.itertuples(index=False):
        path = source_dir / "roi_timeseries" / f"{session.label}.csv"
        if not path.exists():
            raise FileNotFoundError(f"ROI time-series table does not exist: {path}")
        timeseries = pd.read_csv(path)
        rois = list(map(str, timeseries.columns))
        if expected_rois is None:
            expected_rois = rois
        elif rois != expected_rois:
            raise RuntimeError(f"ROI columns or order in {path} differ from the other sessions")
        cleaned = M._clean_timeseries(timeseries)
        correlation = (cleaned.T @ cleaned) / float(cleaned.shape[0] - 1)
        correlation = np.clip(correlation, -0.999999, 0.999999)
        left, right = np.triu_indices(len(rois), k=1)
        edge_z = np.arctanh(correlation[left, right])
        finite_mean = float(np.mean(edge_z[np.isfinite(edge_z)]))
        expected_mean = float(session.between_roi_mean_z)
        aggregate_error = abs(finite_mean - expected_mean)
        maximum_aggregate_error = max(maximum_aggregate_error, aggregate_error)
        if not np.isclose(finite_mean, expected_mean, rtol=0.0, atol=1e-10):
            raise RuntimeError(
                f"Reconstructed between-ROI mean for {session.label} differs from the saved "
                f"figure value ({finite_mean} versus {expected_mean})"
            )
        for left_index, right_index, value in zip(left, right, edge_z, strict=True):
            roi_1 = rois[int(left_index)]
            roi_2 = rois[int(right_index)]
            rows.append(
                {
                    "network": network,
                    "label": session.label,
                    "subject": session.subject,
                    "session": session.session,
                    "state": session.state,
                    "medication_on": session.medication_on,
                    "fc_class": "between_roi",
                    "feature": f"{roi_1}--{roi_2}",
                    "roi_1": roi_1,
                    "roi_2": roi_2,
                    "fc_z": float(value),
                }
            )
    return pd.DataFrame(rows), maximum_aggregate_error


def _complete_feature_values(values: pd.DataFrame) -> pd.DataFrame:
    finite = values.loc[np.isfinite(pd.to_numeric(values["fc_z"], errors="coerce"))].copy()
    complete_subjects = []
    for subject, subject_values in finite.groupby("subject", sort=True):
        state_counts = subject_values["state"].value_counts().to_dict()
        if state_counts == {"off": 1, "on": 1}:
            complete_subjects.append(str(subject))
    return finite.loc[finite["subject"].astype(str).isin(complete_subjects)].copy()


def _fit_lme(values: pd.DataFrame) -> dict[str, object]:
    fit_values = _complete_feature_values(values)
    n_subjects = int(fit_values["subject"].nunique())
    base = {
        "n_subjects": n_subjects,
        "n_observations": int(fit_values.shape[0]),
        "formula": "fc_z ~ medication_on + (1|subject)",
        "reference_medication": "OFF",
        "estimate_on_minus_off_z": np.nan,
        "std_error": np.nan,
        "z_statistic": np.nan,
        "p_value_two_sided": np.nan,
        "ci95_low": np.nan,
        "ci95_high": np.nan,
        "random_intercept_variance": np.nan,
        "residual_variance": np.nan,
        "icc": np.nan,
        "converged": False,
        "optimizer": "",
        "warnings": "",
        "status": "not_fitted",
    }
    if n_subjects < 2:
        base["status"] = "fewer_than_two_complete_subjects"
        return base
    if fit_values["fc_z"].nunique() < 2:
        base["status"] = "constant_outcome"
        return base

    failures: list[str] = []
    for optimizer in ("powell", "lbfgs", "cg"):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                model = smf.mixedlm(
                    "fc_z ~ medication_on",
                    fit_values,
                    groups=fit_values["subject"],
                    re_formula="1",
                )
                result = model.fit(
                    reml=True,
                    method=optimizer,
                    maxiter=5000,
                    disp=False,
                )
                estimate = float(result.fe_params["medication_on"])
                standard_error = float(result.bse_fe["medication_on"])
                p_value = float(result.pvalues["medication_on"])
                confidence_interval = result.conf_int().loc["medication_on"]
                random_variance = float(result.cov_re.iloc[0, 0])
                residual_variance = float(result.scale)
                variance_sum = random_variance + residual_variance
                base.update(
                    {
                        "estimate_on_minus_off_z": estimate,
                        "std_error": standard_error,
                        "z_statistic": estimate / standard_error,
                        "p_value_two_sided": p_value,
                        "ci95_low": float(confidence_interval.iloc[0]),
                        "ci95_high": float(confidence_interval.iloc[1]),
                        "random_intercept_variance": random_variance,
                        "residual_variance": residual_variance,
                        "icc": random_variance / variance_sum if variance_sum > 0 else np.nan,
                        "converged": bool(result.converged),
                        "optimizer": optimizer,
                        "warnings": " | ".join(sorted({str(item.message) for item in caught})),
                        "status": "ok" if bool(result.converged) else "not_converged",
                    }
                )
                if bool(result.converged) and np.isfinite(p_value):
                    return base
                failures.append(f"{optimizer}: did not converge")
            except Exception as exc:  # Preserve a row for features that cannot be estimated.
                failures.append(f"{optimizer}: {type(exc).__name__}: {exc}")
    base["status"] = "fit_failed"
    base["warnings"] = " | ".join(failures)
    return base


def _fit_feature_table(values: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouped = values.groupby(["network", "fc_class", "feature"], sort=True, dropna=False)
    for index, ((network, fc_class, feature), feature_values) in enumerate(grouped, start=1):
        first = feature_values.iloc[0]
        row = {
            "network": network,
            "fc_class": fc_class,
            "feature": feature,
            "roi_1": first["roi_1"],
            "roi_2": first["roi_2"],
        }
        row.update(_fit_lme(feature_values))
        rows.append(row)
        if index % 100 == 0:
            print(f"Fitted {index} feature models...", flush=True)
    return pd.DataFrame(rows)


def _bh_fdr(values: pd.Series) -> np.ndarray:
    p_values = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    output = np.full(p_values.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(p_values)
    if np.any(finite):
        output[finite] = multipletests(p_values[finite], method="fdr_bh")[1]
    return output


def _add_feature_fdr(results: pd.DataFrame) -> pd.DataFrame:
    results = results.copy()
    results["fdr_q_within_network_fc_class"] = np.nan
    results["fdr_q_within_network_all_features"] = np.nan
    for _, indices in results.groupby(["network", "fc_class"], sort=True).groups.items():
        results.loc[indices, "fdr_q_within_network_fc_class"] = _bh_fdr(
            results.loc[indices, "p_value_two_sided"]
        )
    for _, indices in results.groupby("network", sort=True).groups.items():
        results.loc[indices, "fdr_q_within_network_all_features"] = _bh_fdr(
            results.loc[indices, "p_value_two_sided"]
        )
    results["fdr_q_across_both_networks_all_features"] = _bh_fdr(
        results["p_value_two_sided"]
    )
    for column in (
        "fdr_q_within_network_fc_class",
        "fdr_q_within_network_all_features",
        "fdr_q_across_both_networks_all_features",
    ):
        results[f"significant_{column}"] = results[column].lt(0.05)
    return results


def _add_aggregate_fdr(results: pd.DataFrame) -> pd.DataFrame:
    results = results.copy()
    results["fdr_q_within_network_two_tests"] = np.nan
    for _, indices in results.groupby("network", sort=True).groups.items():
        results.loc[indices, "fdr_q_within_network_two_tests"] = _bh_fdr(
            results.loc[indices, "p_value_two_sided"]
        )
    results["fdr_q_across_four_tests"] = _bh_fdr(results["p_value_two_sided"])
    results["significant_fdr_within_network"] = results[
        "fdr_q_within_network_two_tests"
    ].lt(0.05)
    results["significant_fdr_across_four_tests"] = results["fdr_q_across_four_tests"].lt(
        0.05
    )
    return results


def _fit_interaction_model(
    values: pd.DataFrame, formula: str, analysis: str
) -> tuple[pd.DataFrame, dict[str, object]]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = smf.mixedlm(
            formula,
            values,
            groups=values["subject"],
            re_formula="1",
        )
        result = model.fit(
            reml=True,
            method="powell",
            maxiter=5000,
            disp=False,
        )
    confidence_intervals = result.conf_int()
    rows = []
    for term in result.fe_params.index:
        rows.append(
            {
                "analysis": analysis,
                "formula": formula + " + (1|subject)",
                "term": term,
                "estimate": float(result.fe_params[term]),
                "std_error": float(result.bse_fe[term]),
                "z_statistic": float(result.tvalues[term]),
                "p_value_two_sided": float(result.pvalues[term]),
                "ci95_low": float(confidence_intervals.loc[term, 0]),
                "ci95_high": float(confidence_intervals.loc[term, 1]),
                "n_subjects": int(values["subject"].nunique()),
                "n_observations": int(values.shape[0]),
                "converged": bool(result.converged),
            }
        )
    model_info = {
        "analysis": analysis,
        "formula": formula + " + (1|subject)",
        "n_subjects": int(values["subject"].nunique()),
        "n_observations": int(values.shape[0]),
        "converged": bool(result.converged),
        "optimizer": "powell",
        "random_intercept_variance": float(result.cov_re.iloc[0, 0]),
        "residual_variance": float(result.scale),
        "warnings": sorted({str(item.message) for item in caught}),
    }
    return pd.DataFrame(rows), model_info


def _fit_network_interactions(
    aggregate_values: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    values = aggregate_values.copy()
    values["task_activation_network"] = values["network"].eq("task_activation").astype(int)
    values["intra_roi"] = values["fc_class"].eq("intra_roi").astype(int)
    formula = "fc_z ~ medication_on * task_activation_network"
    separate_coefficient_frames = []
    model_info = []
    for fc_class in ("intra_roi", "between_roi"):
        model_values = values.loc[values["fc_class"].eq(fc_class)].copy()
        coefficients, info = _fit_interaction_model(
            model_values,
            formula,
            analysis=f"separate_{fc_class}_medication_by_network",
        )
        coefficients.insert(1, "fc_class", fc_class)
        separate_coefficient_frames.append(coefficients)
        model_info.append(info)
    separate_coefficients = pd.concat(separate_coefficient_frames, ignore_index=True)
    interaction_term = "medication_on:task_activation_network"
    separate_tests = separate_coefficients.loc[
        separate_coefficients["term"].eq(interaction_term)
    ].copy()
    separate_tests.insert(
        3,
        "contrast",
        "(ON-OFF task activation) - (ON-OFF vigour)",
    )
    separate_tests["fdr_q_across_two_fc_type_interactions"] = _bh_fdr(
        separate_tests["p_value_two_sided"]
    )
    separate_tests["significant_fdr"] = separate_tests[
        "fdr_q_across_two_fc_type_interactions"
    ].lt(0.05)

    full_formula = "fc_z ~ medication_on * task_activation_network * intra_roi"
    full_coefficients, full_info = _fit_interaction_model(
        values,
        full_formula,
        analysis="full_medication_by_network_by_fc_type",
    )
    full_coefficients.insert(1, "fc_class", "intra_and_between")
    full_three_way_term = "medication_on:task_activation_network:intra_roi"
    full_coefficients["primary_three_way_interaction"] = full_coefficients["term"].eq(
        full_three_way_term
    )
    full_coefficients["primary_interaction_fdr_q"] = np.where(
        full_coefficients["primary_three_way_interaction"],
        full_coefficients["p_value_two_sided"],
        np.nan,
    )
    model_info.append(full_info)
    return separate_coefficients, separate_tests, full_coefficients, model_info


def _write_method(path: Path) -> None:
    path.write_text(
        """# Medication LME analysis for intra-ROI and between-ROI FC

The two network definitions were analysed separately: the weighted vigour
network and the task-activation network. Only participants with exactly one
OFF and one ON session were retained, matching the complete-pair population in
the target figures.

All outcomes are Pearson functional-connectivity values on the Fisher-z scale.
Aggregate models use the session-level intra-ROI and between-ROI means plotted
in `intra_vs_between_fc_medication_change.png`. Feature-wise intra-ROI models
use each ROI's saved mean voxel-pair Fisher-z connectivity. Feature-wise
between-ROI models use each unique ROI-pair Fisher-z correlation reconstructed
from the saved ROI mean beta-series; reconstruction was checked against every
saved session-level between-ROI mean.

For each outcome, the random-intercept linear mixed-effects model was
`fc_z ~ medication_on + (1 | subject)`, where OFF=0 and ON=1. Therefore the
reported medication coefficient is ON minus OFF. Models were estimated by
REML with the Powell optimizer; L-BFGS and conjugate-gradient were used only as
fallbacks if necessary. Confidence intervals and two-sided p values use the
large-sample normal approximation reported by statsmodels MixedLM.

Benjamini-Hochberg FDR was applied to the aggregate results both within each
network (two tests: aggregate intra and aggregate between) and across all four
aggregate tests. For feature-wise results, the primary q value is corrected
within each network and FC class (all intra-ROI tests or all between-ROI edge
tests). Additional q values across all features within a network and across
both networks are also reported.

The task-activation Pallidum_L intra-ROI outcome cannot be estimated because
that mask contains only one selected voxel, so it has no within-ROI voxel pair.
It is retained in the feature table with a non-fitted status and is excluded
from FDR correction because its p value is missing.

Two aggregate interaction analyses were also fitted. First, separate intra-ROI
and between-ROI models used
`fc_z ~ medication_on * task_activation_network + (1 | subject)`. The two
medication-by-network interaction p values were corrected together with
Benjamini-Hochberg FDR. Second, the full model used
`fc_z ~ medication_on * task_activation_network * intra_roi + (1 | subject)`.
Its prespecified focal test is the three-way interaction, so that single test
does not require multiplicity correction (its q value equals its p value).
Reference levels are OFF medication, the vigour network, and between-ROI FC.
""",
        encoding="utf-8",
    )


def main() -> int:
    args = build_parser().parse_args()
    source_dirs = {
        "vigour": args.vigour_dir,
        "task_activation": args.task_dir,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)

    aggregate_frames = []
    feature_frames = []
    audit: dict[str, object] = {
        "model": "fc_z ~ medication_on + (1|subject)",
        "medication_reference": "OFF",
        "estimation": "REML",
        "primary_optimizer": "powell",
        "feature_fdr_primary_family": "within each network and FC class",
        "networks": {},
    }
    for network, description in NETWORK_SPECS:
        source_dir = source_dirs[network]
        sessions, complete_subjects, excluded_subjects = _load_complete_sessions(source_dir)
        aggregate_frames.append(_aggregate_values(sessions, network))
        intra_values = _intra_roi_values(source_dir, sessions, network)
        between_values, reconstruction_error = _between_roi_values(
            source_dir, sessions, network
        )
        feature_frames.extend([intra_values, between_values])
        audit["networks"][network] = {
            "description": description,
            "source_directory": str(source_dir),
            "n_complete_subjects": len(complete_subjects),
            "complete_subjects": complete_subjects,
            "excluded_incomplete_subjects": excluded_subjects,
            "n_intra_rois": int(intra_values["feature"].nunique()),
            "n_between_roi_edges": int(between_values["feature"].nunique()),
            "maximum_between_mean_reconstruction_absolute_error": reconstruction_error,
        }

    aggregate_values = pd.concat(aggregate_frames, ignore_index=True)
    feature_values = pd.concat(feature_frames, ignore_index=True)
    aggregate_results = _fit_feature_table(aggregate_values)
    aggregate_results = _add_aggregate_fdr(aggregate_results)
    feature_results = _fit_feature_table(feature_values)
    feature_results = _add_feature_fdr(feature_results)
    (
        separate_interaction_coefficients,
        separate_interaction_tests,
        full_interaction_coefficients,
        interaction_model_info,
    ) = _fit_network_interactions(aggregate_values)

    aggregate_results_path = args.out_dir / "medication_lme_aggregate_results.csv"
    feature_results_path = args.out_dir / "medication_lme_feature_results.csv"
    significant_path = args.out_dir / "medication_lme_fdr_significant_features.csv"
    feature_values_path = args.out_dir / "medication_lme_feature_session_values.csv"
    method_path = args.out_dir / "medication_lme_method.md"
    metadata_path = args.out_dir / "metadata.json"
    separate_coefficients_path = (
        args.out_dir / "medication_network_lme_separate_coefficients.csv"
    )
    separate_tests_path = args.out_dir / "medication_network_lme_separate_interaction_tests.csv"
    full_coefficients_path = args.out_dir / "medication_network_fc_type_lme_full_coefficients.csv"

    aggregate_results.to_csv(aggregate_results_path, index=False)
    feature_results.to_csv(feature_results_path, index=False)
    feature_results.loc[
        feature_results["significant_fdr_q_within_network_fc_class"]
    ].to_csv(significant_path, index=False)
    feature_values.to_csv(feature_values_path, index=False)
    separate_interaction_coefficients.to_csv(separate_coefficients_path, index=False)
    separate_interaction_tests.to_csv(separate_tests_path, index=False)
    full_interaction_coefficients.to_csv(full_coefficients_path, index=False)
    _write_method(method_path)
    audit.update(
        {
            "software": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scipy": scipy.__version__,
                "statsmodels": statsmodels.__version__,
            },
            "outputs": {
                "aggregate_results": str(aggregate_results_path),
                "feature_results": str(feature_results_path),
                "fdr_significant_features": str(significant_path),
                "feature_session_values": str(feature_values_path),
                "method": str(method_path),
                "separate_interaction_coefficients": str(separate_coefficients_path),
                "separate_interaction_tests": str(separate_tests_path),
                "full_interaction_coefficients": str(full_coefficients_path),
            },
            "interaction_models": interaction_model_info,
            "n_aggregate_models": int(aggregate_results.shape[0]),
            "n_feature_models": int(feature_results.shape[0]),
            "n_successful_feature_models": int(feature_results["status"].eq("ok").sum()),
            "n_primary_fdr_significant_features": int(
                feature_results["significant_fdr_q_within_network_fc_class"].sum()
            ),
        }
    )
    metadata_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    print("Aggregate LME results:")
    print(
        aggregate_results[
            [
                "network",
                "fc_class",
                "estimate_on_minus_off_z",
                "p_value_two_sided",
                "fdr_q_within_network_two_tests",
                "fdr_q_across_four_tests",
            ]
        ].to_string(index=False)
    )
    print(f"Saved {aggregate_results_path}")
    print(f"Saved {feature_results_path}")
    print(f"Saved {significant_path}")
    print(f"Saved {feature_values_path}")
    print(f"Saved {separate_coefficients_path}")
    print(f"Saved {separate_tests_path}")
    print(f"Saved {full_coefficients_path}")
    print(f"Saved {method_path}")
    print(f"Saved {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
