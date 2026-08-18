#!/usr/bin/env python3
"""Joint vigour/task-activation analysis of medication-related FC changes.

The two existing intra-vs-between analyses are combined in a fully
within-subject 2 (network) x 2 (medication) x 2 (FC type) design.  The two
planned tests are:

1. medication x FC type, averaged equally across the two networks; and
2. network x medication x FC type, which tests whether that contrast differs
   between the task-activation and vigour networks.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib

warnings.filterwarnings("ignore", message="Unable to import Axes3D.*")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.anova import AnovaRM


ROOT = Path(__file__).resolve().parent
DEFAULT_VIGOUR_VALUES = ROOT / "figures" / "med_effects" / "intra_vs_between_fc_session_values.csv"
DEFAULT_TASK_VALUES = (
    ROOT
    / "figures"
    / "med_effects_task_activation"
    / "intra_vs_between_fc_session_values.csv"
)
DEFAULT_OUT_DIR = ROOT / "figures" / "med_effects_network_interaction"

NETWORK_ORDER = ["vigour", "task_activation"]
MEDICATION_ORDER = ["off", "on"]
FC_TYPE_ORDER = ["intra", "between"]
INPUT_KEYS = ["label", "subject", "session", "state"]
LONG_SESSION_KEYS = ["label", "subject", "session", "medication"]
EXPECTED_METRIC = "pearson_fisher_z"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vigour-session-values", type=Path, default=DEFAULT_VIGOUR_VALUES)
    parser.add_argument("--task-session-values", type=Path, default=DEFAULT_TASK_VALUES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser


def _load_session_values(path: Path, network: str) -> tuple[pd.DataFrame, str]:
    if not path.exists():
        raise FileNotFoundError(f"Session-value table does not exist: {path}")

    values = pd.read_csv(path)
    required = set(INPUT_KEYS + ["connectivity_metric", "within_roi_mean_z", "between_roi_mean_z"])
    missing = sorted(required - set(values.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")

    values = values.copy()
    values["subject"] = values["subject"].astype(str)
    values["state"] = values["state"].astype(str).str.lower()
    unexpected_states = sorted(set(values["state"]) - set(MEDICATION_ORDER))
    if unexpected_states:
        raise ValueError(f"{path} contains unexpected medication states: {unexpected_states}")

    duplicates = values.duplicated(INPUT_KEYS, keep=False)
    if duplicates.any():
        duplicate_keys = values.loc[duplicates, INPUT_KEYS]
        raise ValueError(f"Duplicate session rows in {path}:\n{duplicate_keys.to_string(index=False)}")

    metrics = values["connectivity_metric"].dropna().astype(str).unique().tolist()
    if len(metrics) != 1:
        raise ValueError(f"Expected one connectivity metric in {path}; found {metrics}")
    metric = metrics[0]
    if metric != EXPECTED_METRIC:
        raise ValueError(
            f"This interaction analysis requires {EXPECTED_METRIC}; {path} contains {metric}"
        )

    within = values[INPUT_KEYS + ["within_roi_mean_z"]].rename(
        columns={"state": "medication", "within_roi_mean_z": "fc_z"}
    )
    within["fc_type"] = "intra"
    between = values[INPUT_KEYS + ["between_roi_mean_z"]].rename(
        columns={"state": "medication", "between_roi_mean_z": "fc_z"}
    )
    between["fc_type"] = "between"
    long_values = pd.concat([within, between], ignore_index=True)
    long_values["network"] = network
    long_values["fc_z"] = pd.to_numeric(long_values["fc_z"], errors="coerce")
    return long_values, metric


def _validate_session_alignment(vigour: pd.DataFrame, task: pd.DataFrame) -> None:
    vigour_keys = vigour.loc[
        vigour["fc_type"].eq("intra"), LONG_SESSION_KEYS
    ].drop_duplicates()
    task_keys = task.loc[
        task["fc_type"].eq("intra"), LONG_SESSION_KEYS
    ].drop_duplicates()
    aligned = vigour_keys.merge(task_keys, on=LONG_SESSION_KEYS, how="outer", indicator=True)
    unmatched = aligned.loc[aligned["_merge"].ne("both")]
    if not unmatched.empty:
        raise ValueError(
            "The vigour and task-activation inputs do not contain identical sessions:\n"
            + unmatched.to_string(index=False)
        )


def _complete_subject_data(long_values: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    long_values = long_values.copy()
    long_values["subject"] = long_values["subject"].astype(str)
    expected_cells = {
        (network, medication, fc_type)
        for network in NETWORK_ORDER
        for medication in MEDICATION_ORDER
        for fc_type in FC_TYPE_ORDER
    }
    complete_subjects = []
    excluded_subjects = []
    for subject, subject_values in long_values.groupby("subject", sort=True):
        observed_cells = set(
            subject_values[["network", "medication", "fc_type"]]
            .itertuples(index=False, name=None)
        )
        complete = (
            subject_values.shape[0] == len(expected_cells)
            and observed_cells == expected_cells
            and np.isfinite(subject_values["fc_z"].to_numpy(dtype=np.float64)).all()
        )
        (complete_subjects if complete else excluded_subjects).append(str(subject))

    if len(complete_subjects) < 2:
        raise RuntimeError("At least two subjects complete in both networks are required")

    complete_values = long_values.loc[long_values["subject"].isin(complete_subjects)].copy()
    complete_values["network"] = pd.Categorical(
        complete_values["network"], categories=NETWORK_ORDER, ordered=True
    )
    complete_values["medication"] = pd.Categorical(
        complete_values["medication"], categories=MEDICATION_ORDER, ordered=True
    )
    complete_values["fc_type"] = pd.Categorical(
        complete_values["fc_type"], categories=FC_TYPE_ORDER, ordered=True
    )
    complete_values = complete_values.sort_values(
        ["subject", "network", "medication", "fc_type"]
    ).reset_index(drop=True)
    return complete_values, excluded_subjects


def _run_repeated_measures_anova(long_values: pd.DataFrame) -> pd.DataFrame:
    result = AnovaRM(
        long_values,
        depvar="fc_z",
        subject="subject",
        within=["network", "medication", "fc_type"],
    ).fit()
    table = result.anova_table.reset_index(names="effect").rename(
        columns={
            "F Value": "f_value",
            "Num DF": "numerator_df",
            "Den DF": "denominator_df",
            "Pr > F": "p_value",
        }
    )
    descriptions = {
        "network": "Overall difference between vigour and task-activation network-definition pipelines",
        "medication": "Overall ON versus OFF difference",
        "fc_type": "Overall intra-ROI versus between-ROI difference",
        "network:medication": "Network-definition difference in overall medication change",
        "network:fc_type": "Network-definition difference in the intra-versus-between contrast",
        "medication:fc_type": "Medication x FC-type interaction, averaged equally across network definitions",
        "network:medication:fc_type": "Network-definition difference in the medication x FC-type interaction",
    }
    table.insert(1, "description", table["effect"].map(descriptions))
    return table


def _subject_contrasts(long_values: pd.DataFrame) -> pd.DataFrame:
    wide = long_values.pivot(
        index="subject", columns=["network", "medication", "fc_type"], values="fc_z"
    )
    output = pd.DataFrame(index=wide.index)
    for network in NETWORK_ORDER:
        intra_change = wide[(network, "on", "intra")] - wide[(network, "off", "intra")]
        between_change = wide[(network, "on", "between")] - wide[(network, "off", "between")]
        output[f"{network}_intra_on_minus_off_z"] = intra_change
        output[f"{network}_between_on_minus_off_z"] = between_change
        output[f"{network}_medication_by_fc_type_z"] = intra_change - between_change

    vigour_contrast = output["vigour_medication_by_fc_type_z"]
    task_contrast = output["task_activation_medication_by_fc_type_z"]
    output["average_network_medication_by_fc_type_z"] = (vigour_contrast + task_contrast) / 2.0
    output["task_activation_minus_vigour_medication_by_fc_type_z"] = (
        task_contrast - vigour_contrast
    )
    return output.reset_index()


def _one_sample_summary(values: pd.Series) -> dict[str, float | int]:
    finite = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size < 2:
        raise RuntimeError("At least two finite subject contrasts are required")

    mean_value = float(np.mean(finite))
    sd_value = float(np.std(finite, ddof=1))
    sem_value = float(stats.sem(finite))
    t_result = stats.ttest_1samp(finite, popmean=0.0)
    ci_low, ci_high = stats.t.interval(
        0.95, finite.size - 1, loc=mean_value, scale=sem_value
    )
    try:
        wilcoxon_result = stats.wilcoxon(finite, alternative="two-sided")
        wilcoxon_statistic = float(wilcoxon_result.statistic)
        wilcoxon_p = float(wilcoxon_result.pvalue)
    except ValueError:
        wilcoxon_statistic = float("nan")
        wilcoxon_p = float("nan")

    return {
        "n_subjects": int(finite.size),
        "mean": mean_value,
        "sd": sd_value,
        "sem": sem_value,
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "cohen_dz": float(mean_value / sd_value) if sd_value > 0.0 else float("nan"),
        "t_statistic": float(t_result.statistic),
        "degrees_of_freedom": int(finite.size - 1),
        "t_p_value_two_sided": float(t_result.pvalue),
        "equivalent_f_statistic": float(t_result.statistic**2),
        "wilcoxon_statistic": wilcoxon_statistic,
        "wilcoxon_p_value_two_sided": wilcoxon_p,
    }


def _planned_contrast_table(subject_values: pd.DataFrame) -> pd.DataFrame:
    tests = [
        (
            "vigour_simple_medication_by_fc_type",
            "vigour_medication_by_fc_type_z",
            "Vigour: (ON - OFF intra) - (ON - OFF between)",
            "simple_effect",
            False,
        ),
        (
            "task_activation_simple_medication_by_fc_type",
            "task_activation_medication_by_fc_type_z",
            "Task activation: (ON - OFF intra) - (ON - OFF between)",
            "simple_effect",
            False,
        ),
        (
            "medication_by_fc_type_average_networks",
            "average_network_medication_by_fc_type_z",
            "Mean of the vigour and task-activation medication x FC-type contrasts",
            "medication:fc_type",
            True,
        ),
        (
            "network_by_medication_by_fc_type",
            "task_activation_minus_vigour_medication_by_fc_type_z",
            "Task-activation minus vigour medication x FC-type contrast",
            "network:medication:fc_type",
            True,
        ),
    ]
    rows = []
    for analysis, column, description, anova_effect, planned in tests:
        row = {
            "analysis": analysis,
            "value_column": column,
            "description": description,
            "anova_effect": anova_effect,
            "planned_joint_test": planned,
        }
        row.update(_one_sample_summary(subject_values[column]))
        rows.append(row)
    return pd.DataFrame(rows)


def _anova_row(anova: pd.DataFrame, effect: str) -> pd.Series:
    rows = anova.loc[anova["effect"].eq(effect)]
    if rows.shape[0] != 1:
        raise RuntimeError(f"Expected one repeated-measures ANOVA row for {effect}")
    return rows.iloc[0]


def _contrast_row(contrasts: pd.DataFrame, analysis: str) -> pd.Series:
    rows = contrasts.loc[contrasts["analysis"].eq(analysis)]
    if rows.shape[0] != 1:
        raise RuntimeError(f"Expected one planned-contrast row for {analysis}")
    return rows.iloc[0]


def _validate_equivalent_tests(anova: pd.DataFrame, contrasts: pd.DataFrame) -> None:
    mappings = [
        ("medication:fc_type", "medication_by_fc_type_average_networks"),
        ("network:medication:fc_type", "network_by_medication_by_fc_type"),
    ]
    for effect, analysis in mappings:
        anova_values = _anova_row(anova, effect)
        contrast_values = _contrast_row(contrasts, analysis)
        if not np.isclose(
            float(anova_values["f_value"]),
            float(contrast_values["equivalent_f_statistic"]),
            rtol=1e-10,
            atol=1e-12,
        ) or not np.isclose(
            float(anova_values["p_value"]),
            float(contrast_values["t_p_value_two_sided"]),
            rtol=1e-10,
            atol=1e-12,
        ):
            raise RuntimeError(f"ANOVA and subject-contrast tests disagree for {effect}")


def _format_p(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    if value < 0.001:
        return "<0.001"
    if value < 0.01:
        return f"{value:.4f}"
    return f"{value:.3f}"


def _mean_ci(values: np.ndarray) -> tuple[float, float, float]:
    summary = _one_sample_summary(pd.Series(values))
    return float(summary["mean"]), float(summary["ci95_low"]), float(summary["ci95_high"])


def _plot_results(
    subject_values: pd.DataFrame,
    anova: pd.DataFrame,
    contrasts: pd.DataFrame,
    out_dir: Path,
) -> Path:
    colors = {"vigour": "#4C78A8", "task_activation": "#E45756"}
    labels = {"vigour": "Vigour", "task_activation": "Task activation"}
    offsets = {"vigour": -0.055, "task_activation": 0.055}
    rng = np.random.default_rng(0)

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.8), gridspec_kw={"wspace": 0.30})

    for network in NETWORK_ORDER:
        intra = subject_values[f"{network}_intra_on_minus_off_z"].to_numpy(dtype=float)
        between = subject_values[f"{network}_between_on_minus_off_z"].to_numpy(dtype=float)
        positions = np.array([0.0, 1.0]) + offsets[network]
        for intra_value, between_value in zip(intra, between):
            axes[0].plot(
                positions,
                [intra_value, between_value],
                color=colors[network],
                linewidth=0.6,
                alpha=0.20,
                zorder=1,
            )
        axes[0].scatter(
            rng.normal(positions[0], 0.012, intra.size),
            intra,
            s=18,
            color=colors[network],
            alpha=0.55,
            edgecolor="white",
            linewidth=0.3,
            zorder=2,
        )
        axes[0].scatter(
            rng.normal(positions[1], 0.012, between.size),
            between,
            s=18,
            color=colors[network],
            alpha=0.55,
            edgecolor="white",
            linewidth=0.3,
            zorder=2,
        )
        means = []
        for position, values in zip(positions, [intra, between]):
            mean_value, ci_low, ci_high = _mean_ci(values)
            means.append(mean_value)
            axes[0].errorbar(
                position,
                mean_value,
                yerr=[[mean_value - ci_low], [ci_high - mean_value]],
                fmt="o",
                color=colors[network],
                ecolor=colors[network],
                markersize=6,
                elinewidth=1.8,
                capsize=4,
                zorder=4,
                label=labels[network] if position == positions[0] else None,
            )
        axes[0].plot(positions, means, color=colors[network], linewidth=2.8, zorder=3)

    shared = _anova_row(anova, "medication:fc_type")
    three_way = _anova_row(anova, "network:medication:fc_type")
    axes[0].axhline(0.0, color="#777777", linestyle="--", linewidth=0.9, alpha=0.7)
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(["Intra-ROI", "Between-ROI"])
    axes[0].set_ylabel("FC change (ON - OFF; Fisher z)")
    axes[0].set_title("Medication change by FC type")
    axes[0].legend(frameon=False, loc="lower left")
    axes[0].text(
        0.03,
        0.97,
        "Medication × FC type: "
        f"F({float(shared['numerator_df']):.0f}, {float(shared['denominator_df']):.0f}) = "
        f"{float(shared['f_value']):.2f}, p = {_format_p(float(shared['p_value']))}\n"
        "Network-def. × med. × FC type: "
        f"F({float(three_way['numerator_df']):.0f}, {float(three_way['denominator_df']):.0f}) = "
        f"{float(three_way['f_value']):.2f}, p = {_format_p(float(three_way['p_value']))}",
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 2.0},
        zorder=5,
    )

    vigour = subject_values["vigour_medication_by_fc_type_z"].to_numpy(dtype=float)
    task = subject_values["task_activation_medication_by_fc_type_z"].to_numpy(dtype=float)
    for vigour_value, task_value in zip(vigour, task):
        axes[1].plot(
            [0, 1],
            [vigour_value, task_value],
            color="#aaaaaa",
            linewidth=0.7,
            alpha=0.45,
            zorder=1,
        )
    for x_value, network, values in [(0, "vigour", vigour), (1, "task_activation", task)]:
        axes[1].scatter(
            rng.normal(x_value, 0.025, values.size),
            values,
            s=24,
            color=colors[network],
            alpha=0.72,
            edgecolor="white",
            linewidth=0.35,
            zorder=2,
        )
        mean_value, ci_low, ci_high = _mean_ci(values)
        axes[1].errorbar(
            x_value,
            mean_value,
            yerr=[[mean_value - ci_low], [ci_high - mean_value]],
            fmt="o",
            color=colors[network],
            ecolor=colors[network],
            markersize=7,
            elinewidth=2.0,
            capsize=5,
            zorder=4,
        )
    axes[1].axhline(0.0, color="#777777", linestyle="--", linewidth=0.9, alpha=0.7)
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(["Vigour", "Task activation"])
    axes[1].set_ylabel("(ON - OFF intra) - (ON - OFF between)\n(Fisher z)")
    axes[1].set_title("Subject-level interaction contrasts")

    joint = _contrast_row(contrasts, "medication_by_fc_type_average_networks")
    difference = _contrast_row(contrasts, "network_by_medication_by_fc_type")
    axes[1].text(
        0.03,
        0.97,
        f"Joint mean = {float(joint['mean']):+.4f} "
        f"[{float(joint['ci95_low']):+.4f}, {float(joint['ci95_high']):+.4f}]\n"
        f"Task - vigour = {float(difference['mean']):+.4f} "
        f"[{float(difference['ci95_low']):+.4f}, {float(difference['ci95_high']):+.4f}]",
        transform=axes[1].transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 2.0},
        zorder=5,
    )

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=11)
        ax.xaxis.label.set_fontsize(12)
        ax.yaxis.label.set_fontsize(12)
        ax.title.set_fontsize(13)

    fig.suptitle("Joint vigour and task-activation network analysis", fontsize=15, y=0.995)
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.16, top=0.88, wspace=0.30)
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "intra_vs_between_fc_network_interaction.png"
    with plt.rc_context({"font.family": "Liberation Sans", "pdf.fonttype": 42, "ps.fonttype": 42}):
        fig.savefig(png_path, dpi=320, bbox_inches="tight", pad_inches=0.04)
        fig.savefig(png_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return png_path


def _session_mapping_sentence(long_values: pd.DataFrame) -> str:
    mappings = long_values[["session", "medication"]].drop_duplicates()
    states_per_session = mappings.groupby("session", dropna=False)["medication"].nunique()
    if states_per_session.le(1).all():
        ordered = mappings.sort_values("session", key=lambda values: values.astype(str))
        labels = [
            f"session {row.session}={str(row.medication).upper()}"
            for row in ordered.itertuples(index=False)
        ]
        return (
            "Medication labels are taken directly from the source tables; in these inputs, "
            + ", ".join(labels)
            + ". Medication and visit/order are therefore not separable unless supported by "
            "additional counterbalancing information."
        )
    return (
        "Medication labels are taken directly from the source tables and vary within at least "
        "one session number. Any visit/order effects should be evaluated from the study's "
        "counterbalancing information."
    )


def _write_method(path: Path, session_mapping_sentence: str) -> None:
    path.write_text(
        f"""# Joint vigour/task-activation intra-vs-between FC analysis

The analysis combines the session-level values underlying the vigour-network and task-activation-network intra-vs-between FC figures. Both outcomes are retained on the Fisher-z Pearson-correlation scale used for inference in the original analyses; no within-network standardization is applied.

Only subjects with exactly one OFF and one ON observation in both network definitions are included. The balanced long table therefore has eight observations per subject: 2 networks (vigour, task activation) x 2 medication states (OFF, ON) x 2 FC types (intra-ROI, between-ROI). A fully within-subject repeated-measures ANOVA tests `network * medication * FC_type` with subject as the repeated-measures unit. Because every factor has two levels, each interaction has one degree of freedom and does not require a sphericity correction.

Two planned subject-level contrasts make the key model terms explicit. For subject *s* and network *n*,

`D[s,n] = (intra_ON - intra_OFF) - (between_ON - between_OFF)`.

The medication x FC-type effect is tested using `(D[s,vigour] + D[s,task]) / 2`; this asks whether the original intra-vs-between medication effect persists when the two network definitions are considered jointly and equally weighted. The three-way network x medication x FC-type interaction is tested using `D[s,task] - D[s,vigour]`; this asks whether the effect differs between network definitions. Two-sided one-sample t tests and Wilcoxon signed-rank sensitivity tests are reported. The t-test squared is exactly the corresponding one-degree-of-freedom repeated-measures ANOVA F statistic.

The network factor compares two complete analysis pipelines, not otherwise identical ROI samples: the vigour analysis uses weighted vigour voxels and the task-activation analysis uses thresholded, unit-weighted task-map voxels. In the current task-activation source table, 35 ROIs contribute to between-ROI FC while 34 finite ROI summaries contribute to intra-ROI FC because Pallidum_L contains only one usable beta voxel. The same beta sessions contribute to both networks, so network observations are paired and are never treated as independent. A nonsignificant three-way interaction indicates no detected difference; it is not evidence of equivalence. Formal equivalence would require a prespecified smallest effect size of interest and an equivalence test.

{session_mapping_sentence}
""",
        encoding="utf-8",
    )


def _interpretation(joint: pd.Series, difference: pd.Series) -> str:
    joint_mean = float(joint["mean"])
    joint_p = float(joint["t_p_value_two_sided"])
    difference_mean = float(difference["mean"])
    difference_p = float(difference["t_p_value_two_sided"])
    direction = "positive" if joint_mean > 0.0 else "negative" if joint_mean < 0.0 else "zero"
    if joint_p < 0.05:
        joint_text = (
            f"The medication x FC-type interaction averaged across network definitions is "
            f"{direction} and statistically significant (mean={joint_mean:+.6g}, p={joint_p:.6g})."
        )
    else:
        joint_text = (
            "The medication x FC-type interaction averaged across network definitions is not "
            f"statistically significant (mean={joint_mean:+.6g}, p={joint_p:.6g})."
        )
    if difference_p < 0.05:
        difference_text = (
            "The three-way interaction detects a difference between network definitions "
            f"(task minus vigour={difference_mean:+.6g}, p={difference_p:.6g})."
        )
    else:
        difference_text = (
            "The three-way interaction does not detect a difference between network definitions "
            f"(task minus vigour={difference_mean:+.6g}, p={difference_p:.6g})."
        )
    return (
        f"{joint_text} {difference_text} A nonsignificant difference is not a formal "
        "equivalence conclusion."
    )


def _jsonable_record(row: pd.Series) -> dict[str, object]:
    record = {}
    for key, value in row.to_dict().items():
        if isinstance(value, (np.bool_, bool)):
            record[key] = bool(value)
        elif isinstance(value, (np.integer, int)):
            record[key] = int(value)
        elif isinstance(value, (np.floating, float)):
            record[key] = float(value)
        else:
            record[key] = value
    return record


def run_analysis(vigour_path: Path, task_path: Path, out_dir: Path) -> dict[str, Path]:
    vigour, vigour_metric = _load_session_values(vigour_path, "vigour")
    task, task_metric = _load_session_values(task_path, "task_activation")
    if vigour_metric != task_metric:
        raise ValueError(
            f"Connectivity metrics differ: vigour={vigour_metric}, task={task_metric}"
        )
    _validate_session_alignment(vigour, task)

    all_values = pd.concat([vigour, task], ignore_index=True)
    long_values, excluded_subjects = _complete_subject_data(all_values)
    anova = _run_repeated_measures_anova(long_values)
    subject_values = _subject_contrasts(long_values)
    contrasts = _planned_contrast_table(subject_values)
    _validate_equivalent_tests(anova, contrasts)

    correlation = stats.pearsonr(
        subject_values["vigour_medication_by_fc_type_z"].to_numpy(dtype=float),
        subject_values["task_activation_medication_by_fc_type_z"].to_numpy(dtype=float),
    )
    joint = _contrast_row(contrasts, "medication_by_fc_type_average_networks")
    difference = _contrast_row(contrasts, "network_by_medication_by_fc_type")
    shared_anova = _anova_row(anova, "medication:fc_type")
    three_way_anova = _anova_row(anova, "network:medication:fc_type")

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "long_values": out_dir / "intra_vs_between_fc_network_interaction_long.csv",
        "subject_contrasts": out_dir / "intra_vs_between_fc_network_interaction_subject_contrasts.csv",
        "anova": out_dir / "intra_vs_between_fc_network_interaction_anova.csv",
        "planned_contrasts": out_dir / "intra_vs_between_fc_network_interaction_planned_contrasts.csv",
        "results_json": out_dir / "intra_vs_between_fc_network_interaction_results.json",
        "method": out_dir / "intra_vs_between_fc_network_interaction_method.md",
    }
    long_values.to_csv(paths["long_values"], index=False)
    subject_values.to_csv(paths["subject_contrasts"], index=False)
    anova.to_csv(paths["anova"], index=False)
    contrasts.to_csv(paths["planned_contrasts"], index=False)
    _write_method(paths["method"], _session_mapping_sentence(all_values))

    summary = {
        "analysis": "2 x 2 x 2 within-subject repeated-measures ANOVA",
        "formula": "fc_z ~ network_definition * medication * fc_type, repeated within subject (network_definition is stored in the network column)",
        "connectivity_metric": vigour_metric,
        "vigour_input": str(vigour_path),
        "task_activation_input": str(task_path),
        "n_complete_subjects": int(subject_values.shape[0]),
        "n_long_observations": int(long_values.shape[0]),
        "excluded_incomplete_subjects": excluded_subjects,
        "primary_average_effect_across_network_definitions": _jsonable_record(shared_anova),
        "primary_average_contrast_across_network_definitions": _jsonable_record(joint),
        "network_heterogeneity_effect": _jsonable_record(three_way_anova),
        "network_heterogeneity_contrast": _jsonable_record(difference),
        "network_specific_contrasts": {
            "vigour": _jsonable_record(
                _contrast_row(contrasts, "vigour_simple_medication_by_fc_type")
            ),
            "task_activation": _jsonable_record(
                _contrast_row(contrasts, "task_activation_simple_medication_by_fc_type")
            ),
        },
        "exploratory_correlation_between_network_contrasts": {
            "pearson_r": float(correlation.statistic),
            "p_value_two_sided": float(correlation.pvalue),
            "note": "Association is not agreement; both contrasts reuse the same subjects and sessions.",
        },
        "interpretation": _interpretation(joint, difference),
    }
    paths["results_json"].write_text(json.dumps(summary, indent=2), encoding="utf-8")
    paths["figure"] = _plot_results(subject_values, anova, contrasts, out_dir)
    paths["figure_pdf"] = paths["figure"].with_suffix(".pdf")
    return paths


def main() -> None:
    args = build_parser().parse_args()
    paths = run_analysis(
        args.vigour_session_values.resolve(),
        args.task_session_values.resolve(),
        args.out_dir.resolve(),
    )
    results = json.loads(paths["results_json"].read_text(encoding="utf-8"))
    shared = results["primary_average_effect_across_network_definitions"]
    heterogeneity = results["network_heterogeneity_effect"]
    print(f"Complete subjects: {results['n_complete_subjects']}")
    print(
        "Medication x FC type: "
        f"F({shared['numerator_df']:.0f}, {shared['denominator_df']:.0f}) = "
        f"{shared['f_value']:.4f}, p = {shared['p_value']:.6g}"
    )
    print(
        "Network x medication x FC type: "
        f"F({heterogeneity['numerator_df']:.0f}, {heterogeneity['denominator_df']:.0f}) = "
        f"{heterogeneity['f_value']:.4f}, p = {heterogeneity['p_value']:.6g}"
    )
    for path in paths.values():
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
