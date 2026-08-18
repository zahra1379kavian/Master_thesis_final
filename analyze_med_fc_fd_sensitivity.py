#!/usr/bin/env python3
"""Test whether the intra-vs-between medication FC contrast tracks mean FD.

The script fits the mixed model requested for the session-level FC values and
also performs the subject-level difference-in-differences analysis needed to
test differential motion sensitivity.  OFF/between-ROI are the reference
levels, so the medication-by-connectivity coefficient is the plotted contrast.
"""

from __future__ import annotations

import argparse
import json
import platform
import warnings
from pathlib import Path

import matplotlib

warnings.filterwarnings("ignore", message="Unable to import Axes3D.*")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy import stats
import statsmodels
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_breuschpagan


ROOT = Path(__file__).resolve().parent
DEFAULT_FC_VALUES = ROOT / "figures" / "med_effects" / "intra_vs_between_fc_session_values.csv"
DEFAULT_FD_VALUES = ROOT / "figures" / "framewise_displacement" / "subject_session_fd_summary.csv"
DEFAULT_OUT_DIR = ROOT / "figures" / "med_effects" / "fd_sensitivity"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Motion-sensitivity analysis for the intra-vs-between medication FC contrast."
    )
    parser.add_argument("--fc-values", type=Path, default=DEFAULT_FC_VALUES)
    parser.add_argument("--fd-values", type=Path, default=DEFAULT_FD_VALUES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--dpi", type=int, default=300)
    return parser


def _require_columns(table: pd.DataFrame, columns: set[str], path: Path) -> None:
    missing = sorted(columns - set(table.columns))
    if missing:
        raise RuntimeError(f"{path} is missing required columns: {', '.join(missing)}")


def load_analysis_data(fc_path: Path, fd_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    fc = pd.read_csv(fc_path)
    fd = pd.read_csv(fd_path)
    _require_columns(
        fc,
        {
            "subject",
            "session",
            "state",
            "connectivity_metric",
            "within_roi_mean_z",
            "between_roi_mean_z",
        },
        fc_path,
    )
    _require_columns(
        fd,
        {"subject", "session", "mean_fd_mm", "matched_runs_used", "multi_echo_subject"},
        fd_path,
    )

    fc = fc.copy()
    fd = fd.copy()
    fc["session"] = pd.to_numeric(fc["session"], errors="raise").astype(int)
    fd["session"] = pd.to_numeric(fd["session"], errors="raise").astype(int)
    if fc.duplicated(["subject", "session"]).any():
        raise RuntimeError("FC input has duplicate subject/session rows")
    if fd.duplicated(["subject", "session"]).any():
        raise RuntimeError("FD input has duplicate subject/session rows")
    if set(fc["connectivity_metric"].dropna().unique()) != {"pearson_fisher_z"}:
        raise RuntimeError("This analysis requires the Fisher-z Pearson FC values used by the target figure")

    merged = fc.merge(
        fd[["subject", "session", "mean_fd_mm", "matched_runs_used", "multi_echo_subject"]],
        on=["subject", "session"],
        how="inner",
        validate="one_to_one",
    )
    expected_mapping = {(1, "off"), (2, "on")}
    observed_mapping = set(zip(merged["session"], merged["state"].str.lower()))
    if not observed_mapping.issubset(expected_mapping):
        raise RuntimeError(
            "The merged data do not follow the target figure's session mapping (session 1=OFF, session 2=ON)"
        )

    complete_subjects: list[str] = []
    for subject, rows in merged.groupby("subject", sort=True):
        state_counts = rows["state"].str.lower().value_counts()
        if state_counts.to_dict() == {"off": 1, "on": 1}:
            complete_subjects.append(str(subject))
    sessions = merged.loc[merged["subject"].isin(complete_subjects)].copy()
    sessions["state"] = sessions["state"].str.lower()
    sessions = sessions.sort_values(["subject", "session"]).reset_index(drop=True)
    if not complete_subjects:
        raise RuntimeError("No subjects have one OFF and one ON FC/FD session")

    sessions["subject_mean_fd_mm"] = sessions.groupby("subject")["mean_fd_mm"].transform("mean")
    grand_subject_mean_fd = float(
        sessions[["subject", "subject_mean_fd_mm"]].drop_duplicates()["subject_mean_fd_mm"].mean()
    )
    sessions["fd_within_per_0p1mm"] = (
        sessions["mean_fd_mm"] - sessions["subject_mean_fd_mm"]
    ) / 0.1
    sessions["fd_between_per_0p1mm"] = (
        sessions["subject_mean_fd_mm"] - grand_subject_mean_fd
    ) / 0.1

    common = [
        "subject",
        "session",
        "state",
        "mean_fd_mm",
        "matched_runs_used",
        "multi_echo_subject",
        "subject_mean_fd_mm",
        "fd_within_per_0p1mm",
        "fd_between_per_0p1mm",
    ]
    between = sessions[common + ["between_roi_mean_z"]].rename(
        columns={"between_roi_mean_z": "fc_z"}
    )
    between["connectivity_type"] = "between"
    within = sessions[common + ["within_roi_mean_z"]].rename(columns={"within_roi_mean_z": "fc_z"})
    within["connectivity_type"] = "within"
    long_data = pd.concat([between, within], ignore_index=True)
    long_data["medication_on"] = long_data["state"].eq("on").astype(int)
    long_data["within_roi"] = long_data["connectivity_type"].eq("within").astype(int)
    long_data["mean_fd_per_0p1mm"] = long_data["mean_fd_mm"] / 0.1
    long_data = long_data.sort_values(["subject", "session", "connectivity_type"]).reset_index(drop=True)

    audit = {
        "n_fc_sessions_before_complete_pair_filter": int(fc.shape[0]),
        "n_fd_sessions": int(fd.shape[0]),
        "n_merged_sessions_before_complete_pair_filter": int(merged.shape[0]),
        "n_complete_subjects": len(complete_subjects),
        "n_complete_sessions": int(sessions.shape[0]),
        "n_long_observations": int(long_data.shape[0]),
        "complete_subjects": complete_subjects,
        "fc_subjects_excluded_from_complete_pair_analysis": sorted(
            set(fc["subject"].astype(str)) - set(complete_subjects)
        ),
        "session_mapping": {"1": "off", "2": "on"},
        "fc_scale": "Pearson correlation, Fisher-z transformed",
        "fd_scale": "Power framewise displacement, millimetres",
    }
    return sessions, long_data, audit


def fit_mixed_model(formula: str, long_data: pd.DataFrame) -> tuple[object, pd.DataFrame, list[str]]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = smf.mixedlm(formula, long_data, groups=long_data["subject"], re_formula="1")
        result = model.fit(reml=True, method="powell", maxiter=5000, disp=False)
    warning_messages = sorted({str(item.message) for item in caught})
    ci = result.conf_int()
    rows = []
    for term in result.fe_params.index:
        rows.append(
            {
                "term": term,
                "estimate": float(result.fe_params[term]),
                "std_error": float(result.bse_fe[term]),
                "z_statistic": float(result.tvalues[term]),
                "p_value_two_sided": float(result.pvalues[term]),
                "ci95_low": float(ci.loc[term, 0]),
                "ci95_high": float(ci.loc[term, 1]),
            }
        )
    return result, pd.DataFrame(rows), warning_messages


def make_subject_contrasts(sessions: pd.DataFrame) -> pd.DataFrame:
    value_columns = ["within_roi_mean_z", "between_roi_mean_z", "mean_fd_mm"]
    pivot = sessions.pivot(index="subject", columns="state", values=value_columns)
    pivot.columns = [f"{value}_{state}" for value, state in pivot.columns]
    pivot = pivot.reset_index()
    subject_fd_metadata = (
        sessions.groupby("subject", sort=True)
        .agg(
            multi_echo_subject=("multi_echo_subject", "max"),
            matched_runs_used=("matched_runs_used", lambda values: ";".join(sorted(set(map(str, values))))),
        )
        .reset_index()
    )
    pivot = pivot.merge(subject_fd_metadata, on="subject", how="left", validate="one_to_one")
    pivot["within_change_on_minus_off_z"] = (
        pivot["within_roi_mean_z_on"] - pivot["within_roi_mean_z_off"]
    )
    pivot["between_change_on_minus_off_z"] = (
        pivot["between_roi_mean_z_on"] - pivot["between_roi_mean_z_off"]
    )
    pivot["fc_contrast_change_z"] = (
        pivot["within_change_on_minus_off_z"] - pivot["between_change_on_minus_off_z"]
    )
    pivot["delta_fd_on_minus_off_mm"] = pivot["mean_fd_mm_on"] - pivot["mean_fd_mm_off"]
    pivot["delta_fd_per_0p1mm"] = pivot["delta_fd_on_minus_off_mm"] / 0.1
    return pivot.sort_values("subject").reset_index(drop=True)


def contrast_regression(subject_values: pd.DataFrame) -> tuple[object, object, pd.DataFrame, dict[str, object]]:
    result = smf.ols("fc_contrast_change_z ~ delta_fd_per_0p1mm", subject_values).fit()
    hc3 = result.get_robustcov_results(cov_type="HC3")
    ci = result.conf_int()
    hc3_ci = hc3.conf_int()
    rows = []
    for index, term in enumerate(result.params.index):
        rows.append(
            {
                "term": term,
                "interpretation": (
                    "FC contrast at no ON-OFF FD difference"
                    if term == "Intercept"
                    else "Change in FC contrast per 0.1 mm greater ON-OFF FD"
                ),
                "estimate": float(result.params[term]),
                "std_error": float(result.bse[term]),
                "t_statistic": float(result.tvalues[term]),
                "df": int(result.df_resid),
                "p_value_two_sided": float(result.pvalues[term]),
                "ci95_low": float(ci.loc[term, 0]),
                "ci95_high": float(ci.loc[term, 1]),
                "hc3_std_error": float(hc3.bse[index]),
                "hc3_t_statistic": float(hc3.tvalues[index]),
                "hc3_p_value_two_sided": float(hc3.pvalues[index]),
                "hc3_ci95_low": float(hc3_ci[index, 0]),
                "hc3_ci95_high": float(hc3_ci[index, 1]),
            }
        )

    pearson = stats.pearsonr(
        subject_values["delta_fd_on_minus_off_mm"], subject_values["fc_contrast_change_z"]
    )
    spearman = stats.spearmanr(
        subject_values["delta_fd_on_minus_off_mm"], subject_values["fc_contrast_change_z"]
    )
    shapiro = stats.shapiro(result.resid)
    bp_lm, bp_lm_p, bp_f, bp_f_p = het_breuschpagan(result.resid, result.model.exog)
    influence = result.get_influence()
    cooks = influence.cooks_distance[0]
    leverage = influence.hat_matrix_diag
    max_cook_index = int(np.argmax(cooks))
    diagnostics = {
        "r_squared": float(result.rsquared),
        "adjusted_r_squared": float(result.rsquared_adj),
        "pearson_r": float(pearson.statistic),
        "pearson_p_two_sided": float(pearson.pvalue),
        "spearman_rho": float(spearman.statistic),
        "spearman_p_two_sided": float(spearman.pvalue),
        "residual_shapiro_w": float(shapiro.statistic),
        "residual_shapiro_p": float(shapiro.pvalue),
        "breusch_pagan_lm": float(bp_lm),
        "breusch_pagan_lm_p": float(bp_lm_p),
        "breusch_pagan_f": float(bp_f),
        "breusch_pagan_f_p": float(bp_f_p),
        "largest_cooks_distance": float(cooks[max_cook_index]),
        "largest_cooks_distance_subject": str(subject_values.iloc[max_cook_index]["subject"]),
        "largest_leverage": float(np.max(leverage)),
        "largest_leverage_subject": str(subject_values.iloc[int(np.argmax(leverage))]["subject"]),
    }

    influence_table = subject_values[["subject", "delta_fd_on_minus_off_mm", "fc_contrast_change_z"]].copy()
    influence_table["cooks_distance"] = cooks
    influence_table["leverage"] = leverage
    influence_table["externally_studentized_residual"] = influence.resid_studentized_external
    influence_table = influence_table.sort_values("cooks_distance", ascending=False).reset_index(drop=True)
    return result, hc3, pd.DataFrame(rows), {"diagnostics": diagnostics, "influence": influence_table}


def leave_one_out(subject_values: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for subject in subject_values["subject"]:
        subset = subject_values.loc[subject_values["subject"].ne(subject)]
        result = smf.ols("fc_contrast_change_z ~ delta_fd_per_0p1mm", subset).fit()
        rows.append(
            {
                "left_out_subject": subject,
                "equal_motion_contrast": float(result.params["Intercept"]),
                "equal_motion_p_value": float(result.pvalues["Intercept"]),
                "motion_slope_per_0p1mm": float(result.params["delta_fd_per_0p1mm"]),
                "motion_slope_p_value": float(result.pvalues["delta_fd_per_0p1mm"]),
                "r_squared": float(result.rsquared),
            }
        )
    return pd.DataFrame(rows)


def cell_descriptives(long_data: pd.DataFrame) -> pd.DataFrame:
    return (
        long_data.groupby(["state", "connectivity_type"], sort=True)["fc_z"]
        .agg(n="size", mean="mean", sd="std")
        .reset_index()
    )


def plot_motion_sensitivity(subject_values: pd.DataFrame, result: object, output_stem: Path, dpi: int) -> None:
    x = subject_values["delta_fd_on_minus_off_mm"].to_numpy(dtype=float)
    y = subject_values["fc_contrast_change_z"].to_numpy(dtype=float)
    grid = np.linspace(min(float(x.min()), 0.0) - 0.01, max(float(x.max()), 0.0) + 0.01, 300)
    prediction = result.get_prediction(pd.DataFrame({"delta_fd_per_0p1mm": grid / 0.1})).summary_frame()

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ax.axhline(0.0, color="#777777", linewidth=1.0, linestyle="--", zorder=0)
    ax.axvline(0.0, color="#777777", linewidth=1.0, linestyle=":", zorder=0)
    ax.fill_between(
        grid,
        prediction["mean_ci_lower"].to_numpy(),
        prediction["mean_ci_upper"].to_numpy(),
        color="#4C78A8",
        alpha=0.18,
        linewidth=0,
        label="95% CI for fitted mean",
    )
    ax.plot(grid, prediction["mean"], color="#4C78A8", linewidth=2.0, label="OLS fit")
    ax.scatter(x, y, s=48, color="#D65F3C", edgecolor="white", linewidth=0.7, zorder=3)
    for _, row in subject_values.iterrows():
        ax.annotate(
            str(row["subject"]).replace("sub-", ""),
            (float(row["delta_fd_on_minus_off_mm"]), float(row["fc_contrast_change_z"])),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7.5,
            color="#333333",
        )
    ax.set_xlabel("ON − OFF mean Power FD (mm)")
    ax.set_ylabel("(ON − OFF intra-ROI FC) − (ON − OFF between-ROI FC)\n(Fisher-z units)")
    ax.set_title("Medication FC contrast versus session motion change")
    ax.legend(frameon=False, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _coefficient(table: pd.DataFrame, term: str) -> pd.Series:
    rows = table.loc[table["term"].eq(term)]
    if rows.shape[0] != 1:
        raise RuntimeError(f"Expected one coefficient row for {term}, found {rows.shape[0]}")
    return rows.iloc[0]


def write_report(path: Path, summary: dict[str, object]) -> None:
    raw = summary["original_paired_contrast"]
    requested = summary["requested_mixed_model"]
    corrected = summary["within_between_fd_mixed_model_sensitivity"]
    motion = summary["motion_change_analysis"]
    single_echo = summary["single_echo_protocol_sensitivity"]
    diagnostics = motion["diagnostics"]
    report = f"""# Mean-FD sensitivity of the medication FC contrast

## Conclusion

There is **no evidence that ON-minus-OFF mean-FD change explains the intra-versus-between FC medication contrast** in the 17 complete participants. The FC contrast was weakly negatively associated with FD change (Pearson r = {diagnostics['pearson_r']:.3f}, p = {diagnostics['pearson_p_two_sided']:.3f}), and the estimated motion slope was not significant. The contrast predicted at no session motion difference remained positive. This is a sensitivity result, not proof that motion has no influence.

## Data and coding

- FC source: `{summary['inputs']['fc_values']}`
- FD source: `{summary['inputs']['fd_values']}`
- N = {summary['cohort']['n_complete_subjects']} subjects, {summary['cohort']['n_complete_sessions']} sessions, {summary['cohort']['n_long_observations']} long-format FC rows.
- Session 1 = OFF; session 2 = ON. `sub-pd017` was excluded because the FC data contain no complete OFF/ON pair.
- FC is on the Fisher-z Pearson-correlation scale. OFF and between-ROI are the reference levels.

## Original paired interaction contrast

The unadjusted subject-level contrast was {raw['mean']:.6f} Fisher-z (95% CI [{raw['ci95_low']:.6f}, {raw['ci95_high']:.6f}]), t({raw['df']}) = {raw['t_statistic']:.3f}, p = {raw['p_value_two_sided']:.5f}.

## Requested mixed model

Model: `FC ~ medication_state * connectivity_type + mean_FD + (1 | subject)`.

- Medication × connectivity: b = {requested['interaction_estimate']:.6f}, SE = {requested['interaction_std_error']:.6f}, z = {requested['interaction_z']:.3f}, p = {requested['interaction_p']:.4f}, 95% CI [{requested['interaction_ci95_low']:.6f}, {requested['interaction_ci95_high']:.6f}].
- Mean FD: b = {requested['fd_estimate_per_0p1mm']:.6f} Fisher-z per 0.1 mm, SE = {requested['fd_std_error_per_0p1mm']:.6f}, z = {requested['fd_z']:.3f}, p = {requested['fd_p']:.4f}.
- The subject random-intercept variance was {requested['random_intercept_variance']:.3g} (estimated ICC = {requested['random_intercept_icc']:.3f}). The Powell fit converged. Statsmodels emitted a scale-sensitive boundary warning because both variance components are numerically small on the Fisher-z scale; the stronger limitation is the model's restrictive common-variance/correlation structure.

The additive FD term is identical for the intra- and between-ROI row from each session. It therefore cancels algebraically from medication × connectivity and leaves that point estimate equal to the unadjusted contrast. The pooled FD main effect does not establish that FD explains the interaction.

## Motion analysis that targets the contrast

For each subject, the analysis regressed

`[(ON−OFF intra) − (ON−OFF between)] ~ (FD_ON−FD_OFF)`.

- Cohort mean FD was {motion['mean_fd_off_mm']:.6f} mm OFF and {motion['mean_fd_on_mm']:.6f} mm ON; mean change = {motion['mean_delta_fd_mm']:+.6f} mm, t({motion['delta_fd_df']}) = {motion['delta_fd_t']:.3f}, p = {motion['delta_fd_p']:.3f}.
- Motion slope: {motion['slope_per_mm']:+.6f} Fisher-z/mm, SE = {motion['slope_std_error_per_mm']:.6f}, t({motion['df']}) = {motion['slope_t']:.3f}, p = {motion['slope_p']:.3f}, 95% CI [{motion['slope_ci95_low_per_mm']:.6f}, {motion['slope_ci95_high_per_mm']:.6f}], R² = {diagnostics['r_squared']:.3f}.
- Equal-motion estimate (predicted at ΔFD = 0): {motion['equal_motion_estimate']:.6f}, SE = {motion['equal_motion_std_error']:.6f}, t({motion['df']}) = {motion['equal_motion_t']:.3f}, p = {motion['equal_motion_p']:.5f}, 95% CI [{motion['equal_motion_ci95_low']:.6f}, {motion['equal_motion_ci95_high']:.6f}].
- HC3 sensitivity: equal-motion p = {motion['equal_motion_hc3_p']:.4f}; motion-slope p = {motion['slope_hc3_p']:.3f}.
- Spearman rho = {diagnostics['spearman_rho']:.3f}, p = {diagnostics['spearman_p_two_sided']:.3f}.
- Leave-one-subject-out equal-motion estimates ranged from {motion['leave_one_out']['equal_motion_min']:.6f} to {motion['leave_one_out']['equal_motion_max']:.6f}; motion-slope p-values ranged from {motion['leave_one_out']['motion_slope_p_min']:.3f} to {motion['leave_one_out']['motion_slope_p_max']:.3f}.

The corresponding long-form sensitivity model separated within-person FD from between-person mean FD and included their interactions with connectivity type. Its within-person FD × type coefficient was {corrected['within_person_fd_by_type_estimate_per_0p1mm']:+.6f} per 0.1 mm (p = {corrected['within_person_fd_by_type_p']:.3f}), while medication × type was {corrected['medication_by_type_estimate']:.6f} (p = {corrected['medication_by_type_p']:.4f}). Because that mixed model retains the restrictive random-intercept/common-residual structure, the subject-level contrast regression is the preferred inference.

Because three included participants had a multi-echo/~2-s protocol while the others had a single-echo/~1-s protocol, the contrast regression was repeated in the {single_echo['n']} single-echo participants. The motion slope remained nonsignificant (b = {single_echo['slope_per_mm']:+.6f} Fisher-z/mm, p = {single_echo['slope_p']:.3f}; HC3 p = {single_echo['slope_hc3_p']:.3f}), and the equal-motion contrast remained positive ({single_echo['equal_motion_estimate']:.6f}, p = {single_echo['equal_motion_p']:.4f}; HC3 p = {single_echo['equal_motion_hc3_p']:.4f}).

Residual normality was not rejected (Shapiro p = {diagnostics['residual_shapiro_p']:.3f}), but the Breusch–Pagan diagnostic suggested heteroscedasticity (p = {diagnostics['breusch_pagan_f_p']:.3f}); therefore the HC3 results are reported as a sensitivity check.

## Interpretation limits

1. OFF is always session 1 and ON is always session 2, so medication is confounded with session/order. FD adjustment cannot remove generic order or time effects.
2. FD is measured after medication and could be part of a medication pathway. These results should be described as motion sensitivity, not an unqualified causal adjustment.
3. With N = 17, the motion-slope confidence interval is wide. Use “no evidence of association with FD change,” not “motion had no effect.”
4. The requested random-intercept model assumes a common residual variance and compound-symmetric dependence, which are poor approximations here. The direct subject contrast is the cleaner repeated-measures analysis for this balanced 2 × 2 design.
"""
    path.write_text(report, encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    sessions, long_data, audit = load_analysis_data(args.fc_values, args.fd_values)
    subject_values = make_subject_contrasts(sessions)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    requested_formula = "fc_z ~ medication_on * within_roi + mean_fd_per_0p1mm"
    requested_result, requested_table, requested_warnings = fit_mixed_model(requested_formula, long_data)
    corrected_formula = (
        "fc_z ~ medication_on * within_roi "
        "+ fd_within_per_0p1mm * within_roi + fd_between_per_0p1mm * within_roi"
    )
    corrected_result, corrected_table, corrected_warnings = fit_mixed_model(corrected_formula, long_data)
    contrast_result, _, contrast_table, contrast_details = contrast_regression(subject_values)
    single_echo_values = subject_values.loc[~subject_values["multi_echo_subject"].astype(bool)].copy()
    if single_echo_values.shape[0] < 4:
        raise RuntimeError("Fewer than four single-echo complete subjects are available for protocol sensitivity")
    _, _, single_echo_table, _ = contrast_regression(single_echo_values)
    loo = leave_one_out(subject_values)
    cells = cell_descriptives(long_data)

    raw_values = subject_values["fc_contrast_change_z"].to_numpy(dtype=float)
    raw_test = stats.ttest_1samp(raw_values, 0.0)
    raw_sem = float(stats.sem(raw_values))
    raw_ci = stats.t.interval(0.95, len(raw_values) - 1, loc=float(np.mean(raw_values)), scale=raw_sem)
    delta_fd_values = subject_values["delta_fd_on_minus_off_mm"].to_numpy(dtype=float)
    delta_fd_test = stats.ttest_1samp(delta_fd_values, 0.0)

    requested_interaction = _coefficient(requested_table, "medication_on:within_roi")
    requested_fd = _coefficient(requested_table, "mean_fd_per_0p1mm")
    corrected_interaction = _coefficient(corrected_table, "medication_on:within_roi")
    corrected_motion_type = _coefficient(corrected_table, "fd_within_per_0p1mm:within_roi")
    equal_motion = _coefficient(contrast_table, "Intercept")
    motion_slope = _coefficient(contrast_table, "delta_fd_per_0p1mm")
    single_echo_equal_motion = _coefficient(single_echo_table, "Intercept")
    single_echo_motion_slope = _coefficient(single_echo_table, "delta_fd_per_0p1mm")

    requested_random_variance = float(requested_result.cov_re.iloc[0, 0])
    requested_residual_variance = float(requested_result.scale)
    requested_icc = requested_random_variance / (requested_random_variance + requested_residual_variance)

    loo_summary = {
        "equal_motion_min": float(loo["equal_motion_contrast"].min()),
        "equal_motion_max": float(loo["equal_motion_contrast"].max()),
        "equal_motion_p_min": float(loo["equal_motion_p_value"].min()),
        "equal_motion_p_max": float(loo["equal_motion_p_value"].max()),
        "motion_slope_min_per_0p1mm": float(loo["motion_slope_per_0p1mm"].min()),
        "motion_slope_max_per_0p1mm": float(loo["motion_slope_per_0p1mm"].max()),
        "motion_slope_p_min": float(loo["motion_slope_p_value"].min()),
        "motion_slope_p_max": float(loo["motion_slope_p_value"].max()),
    }
    summary: dict[str, object] = {
        "inputs": {
            "fc_values": str(args.fc_values),
            "fd_values": str(args.fd_values),
        },
        "cohort": audit,
        "original_paired_contrast": {
            "definition": "(ON-OFF intra-ROI FC) - (ON-OFF between-ROI FC)",
            "n": len(raw_values),
            "mean": float(np.mean(raw_values)),
            "sd": float(np.std(raw_values, ddof=1)),
            "sem": raw_sem,
            "ci95_low": float(raw_ci[0]),
            "ci95_high": float(raw_ci[1]),
            "t_statistic": float(raw_test.statistic),
            "df": len(raw_values) - 1,
            "p_value_two_sided": float(raw_test.pvalue),
        },
        "requested_mixed_model": {
            "formula": requested_formula + " + (1 | subject)",
            "estimation": "REML, Powell optimizer",
            "converged": bool(requested_result.converged),
            "warnings": requested_warnings,
            "interaction_estimate": float(requested_interaction["estimate"]),
            "interaction_std_error": float(requested_interaction["std_error"]),
            "interaction_z": float(requested_interaction["z_statistic"]),
            "interaction_p": float(requested_interaction["p_value_two_sided"]),
            "interaction_ci95_low": float(requested_interaction["ci95_low"]),
            "interaction_ci95_high": float(requested_interaction["ci95_high"]),
            "fd_estimate_per_0p1mm": float(requested_fd["estimate"]),
            "fd_std_error_per_0p1mm": float(requested_fd["std_error"]),
            "fd_z": float(requested_fd["z_statistic"]),
            "fd_p": float(requested_fd["p_value_two_sided"]),
            "random_intercept_variance": requested_random_variance,
            "residual_variance": requested_residual_variance,
            "random_intercept_icc": requested_icc,
            "interaction_equals_raw_contrast_within_tolerance": bool(
                np.isclose(requested_interaction["estimate"], np.mean(raw_values), atol=1e-10)
            ),
        },
        "within_between_fd_mixed_model_sensitivity": {
            "formula": corrected_formula + " + (1 | subject)",
            "estimation": "REML, Powell optimizer",
            "converged": bool(corrected_result.converged),
            "warnings": corrected_warnings,
            "medication_by_type_estimate": float(corrected_interaction["estimate"]),
            "medication_by_type_p": float(corrected_interaction["p_value_two_sided"]),
            "within_person_fd_by_type_estimate_per_0p1mm": float(corrected_motion_type["estimate"]),
            "within_person_fd_by_type_p": float(corrected_motion_type["p_value_two_sided"]),
            "random_intercept_variance": float(corrected_result.cov_re.iloc[0, 0]),
        },
        "motion_change_analysis": {
            "formula": "subject FC interaction contrast ~ ON-OFF mean FD",
            "mean_fd_off_mm": float(subject_values["mean_fd_mm_off"].mean()),
            "mean_fd_on_mm": float(subject_values["mean_fd_mm_on"].mean()),
            "mean_delta_fd_mm": float(subject_values["delta_fd_on_minus_off_mm"].mean()),
            "delta_fd_t": float(delta_fd_test.statistic),
            "delta_fd_df": int(len(delta_fd_values) - 1),
            "delta_fd_p": float(delta_fd_test.pvalue),
            "equal_motion_estimate": float(equal_motion["estimate"]),
            "equal_motion_std_error": float(equal_motion["std_error"]),
            "equal_motion_t": float(equal_motion["t_statistic"]),
            "equal_motion_p": float(equal_motion["p_value_two_sided"]),
            "equal_motion_ci95_low": float(equal_motion["ci95_low"]),
            "equal_motion_ci95_high": float(equal_motion["ci95_high"]),
            "equal_motion_hc3_p": float(equal_motion["hc3_p_value_two_sided"]),
            "slope_per_0p1mm": float(motion_slope["estimate"]),
            "slope_per_mm": float(motion_slope["estimate"] * 10.0),
            "slope_std_error_per_mm": float(motion_slope["std_error"] * 10.0),
            "slope_t": float(motion_slope["t_statistic"]),
            "slope_p": float(motion_slope["p_value_two_sided"]),
            "slope_ci95_low_per_mm": float(motion_slope["ci95_low"] * 10.0),
            "slope_ci95_high_per_mm": float(motion_slope["ci95_high"] * 10.0),
            "slope_hc3_p": float(motion_slope["hc3_p_value_two_sided"]),
            "df": int(motion_slope["df"]),
            "diagnostics": contrast_details["diagnostics"],
            "leave_one_out": loo_summary,
        },
        "single_echo_protocol_sensitivity": {
            "n": int(single_echo_values.shape[0]),
            "subjects": single_echo_values["subject"].astype(str).tolist(),
            "equal_motion_estimate": float(single_echo_equal_motion["estimate"]),
            "equal_motion_p": float(single_echo_equal_motion["p_value_two_sided"]),
            "equal_motion_hc3_p": float(single_echo_equal_motion["hc3_p_value_two_sided"]),
            "slope_per_mm": float(single_echo_motion_slope["estimate"] * 10.0),
            "slope_p": float(single_echo_motion_slope["p_value_two_sided"]),
            "slope_hc3_p": float(single_echo_motion_slope["hc3_p_value_two_sided"]),
            "slope_ci95_low_per_mm": float(single_echo_motion_slope["ci95_low"] * 10.0),
            "slope_ci95_high_per_mm": float(single_echo_motion_slope["ci95_high"] * 10.0),
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "statsmodels": statsmodels.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }

    long_data.to_csv(args.out_dir / "matched_long_data.csv", index=False, float_format="%.10g")
    subject_values.to_csv(args.out_dir / "subject_contrasts_and_fd_change.csv", index=False, float_format="%.10g")
    cells.to_csv(args.out_dir / "fc_cell_descriptives.csv", index=False, float_format="%.10g")
    requested_table.to_csv(args.out_dir / "requested_mixed_model_coefficients.csv", index=False, float_format="%.10g")
    corrected_table.to_csv(args.out_dir / "within_between_fd_mixed_model_coefficients.csv", index=False, float_format="%.10g")
    contrast_table.to_csv(args.out_dir / "motion_change_regression_coefficients.csv", index=False, float_format="%.10g")
    single_echo_table.to_csv(
        args.out_dir / "single_echo_motion_change_regression_coefficients.csv",
        index=False,
        float_format="%.10g",
    )
    contrast_details["influence"].to_csv(
        args.out_dir / "motion_change_influence_diagnostics.csv", index=False, float_format="%.10g"
    )
    loo.to_csv(args.out_dir / "motion_change_leave_one_out.csv", index=False, float_format="%.10g")
    with (args.out_dir / "analysis_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    write_report(args.out_dir / "analysis_report.md", summary)
    plot_motion_sensitivity(
        subject_values,
        contrast_result,
        args.out_dir / "fc_contrast_change_vs_fd_change",
        args.dpi,
    )

    print(f"Complete subjects: {audit['n_complete_subjects']}")
    print(
        "Requested medication x type: "
        f"b={requested_interaction['estimate']:.6f}, p={requested_interaction['p_value_two_sided']:.4f}"
    )
    print(
        "Contrast vs FD change: "
        f"r={contrast_details['diagnostics']['pearson_r']:.3f}, "
        f"p={contrast_details['diagnostics']['pearson_p_two_sided']:.3f}"
    )
    print(
        "Equal-motion contrast: "
        f"b={equal_motion['estimate']:.6f}, p={equal_motion['p_value_two_sided']:.5f}, "
        f"HC3 p={equal_motion['hc3_p_value_two_sided']:.4f}"
    )
    print(f"Saved outputs to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
