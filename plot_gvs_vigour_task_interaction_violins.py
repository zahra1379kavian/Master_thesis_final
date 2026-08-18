#!/usr/bin/env python3
"""Plot the vigour-by-task-map interaction across all GVS conditions.

For each feature, run-level active-minus-sham deltas are calculated separately
for the vigour projection and task map. The interaction is their paired
difference, calculated on matched runs before averaging within subject. One
figure is produced per feature, with medication OFF and ON shown side by side.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib

warnings.filterwarnings("ignore", message="Unable to import Axes3D.*")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_gvs_task_feature_delta_violins import (
    FEATURES,
    FEATURE_LABELS,
    RUN_FEATURES_NAME,
    SESSIONS,
    SHAM_CODE,
    SIGNIFICANCE_Q_THRESHOLD,
    gvs_display_label,
    padded_limits,
)
from spagetti_plot import _exact_sign_flip_pvalue, _fdr_bh


ROOT = Path(__file__).resolve().parent
DEFAULT_VIGOUR_DIR = ROOT / "figures" / "gvs_projection_features_vs_sham"
DEFAULT_TASK_DIR = ROOT / "figures" / "gvs_task_map_bold_features_vs_sham"
DEFAULT_OUT_DIR = ROOT / "figures" / "gvs_vigour_task_feature_interaction"

PAIR_KEYS = ["subject", "session", "medication", "run"]
INTERACTION_COLUMN = "interaction_vigour_minus_task"
INTERACTION_Q_COLUMN = "interaction_q_perm_fdr_4_features"


def _paired_run_deltas(
    run_features: pd.DataFrame,
    active_gvs: str,
    session: int,
    medication: str,
) -> pd.DataFrame:
    required = set(PAIR_KEYS + ["gvs_code", *FEATURES])
    missing = sorted(required - set(run_features.columns))
    if missing:
        raise ValueError(f"Run-feature table is missing columns: {', '.join(missing)}")

    selected = run_features.loc[
        run_features["session"].astype(int).eq(int(session))
        & run_features["medication"].astype(str).str.upper().eq(medication.upper())
        & run_features["gvs_code"].isin([SHAM_CODE, active_gvs])
    ].copy()
    if selected.empty:
        raise ValueError(
            f"No run features found for session {session}, medication {medication}, "
            f"and {active_gvs}/{SHAM_CODE}"
        )

    duplicate = selected.duplicated(PAIR_KEYS + ["gvs_code"], keep=False)
    if duplicate.any():
        duplicate_keys = selected.loc[duplicate, PAIR_KEYS + ["gvs_code"]]
        raise ValueError(
            f"Duplicate run-condition rows found:\n{duplicate_keys.to_string(index=False)}"
        )

    sham = selected.loc[selected["gvs_code"].eq(SHAM_CODE)].set_index(PAIR_KEYS)
    active = selected.loc[selected["gvs_code"].eq(active_gvs)].set_index(PAIR_KEYS)
    paired = sham[FEATURES].join(
        active[FEATURES],
        how="inner",
        lsuffix="_sham",
        rsuffix="_active",
        validate="one_to_one",
    )
    if paired.empty:
        raise ValueError(f"No matched {active_gvs}-versus-{SHAM_CODE} runs were found")

    rows = []
    for feature in FEATURES:
        feature_rows = paired[[f"{feature}_sham", f"{feature}_active"]].rename(
            columns={
                f"{feature}_sham": "sham_value",
                f"{feature}_active": "active_value",
            }
        )
        feature_rows = feature_rows.replace([np.inf, -np.inf], np.nan).dropna()
        feature_rows = feature_rows.reset_index()
        feature_rows["delta_active_minus_sham"] = (
            feature_rows["active_value"] - feature_rows["sham_value"]
        )
        feature_rows["active_gvs"] = active_gvs
        feature_rows["sham_gvs"] = SHAM_CODE
        feature_rows["feature"] = feature
        feature_rows["label"] = FEATURE_LABELS[feature]
        rows.append(feature_rows)

    return pd.concat(rows, ignore_index=True)


def make_interaction_values(
    vigour_runs: pd.DataFrame,
    task_runs: pd.DataFrame,
    active_gvs: str,
    session: int,
    medication: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    vigour = _paired_run_deltas(vigour_runs, active_gvs, session, medication).rename(
        columns={
            "sham_value": "vigour_sham_value",
            "active_value": "vigour_active_value",
            "delta_active_minus_sham": "vigour_delta",
        }
    )
    task = _paired_run_deltas(task_runs, active_gvs, session, medication).rename(
        columns={
            "sham_value": "task_sham_value",
            "active_value": "task_active_value",
            "delta_active_minus_sham": "task_delta",
        }
    )
    merge_keys = PAIR_KEYS + ["active_gvs", "sham_gvs", "feature", "label"]
    run_values = vigour.merge(
        task,
        on=merge_keys,
        how="inner",
        validate="one_to_one",
    )
    if run_values.empty:
        raise ValueError("No common vigour/task active-versus-sham run pairs were found")
    run_values["interaction_vigour_minus_task"] = (
        run_values["vigour_delta"] - run_values["task_delta"]
    )
    run_values = run_values.sort_values(["feature", "subject", "run"]).reset_index(drop=True)

    subject_values = (
        run_values.groupby(
            ["subject", "active_gvs", "sham_gvs", "feature", "label"],
            as_index=False,
            sort=True,
        )
        .agg(
            vigour_sham_value=("vigour_sham_value", "mean"),
            vigour_active_value=("vigour_active_value", "mean"),
            vigour_delta=("vigour_delta", "mean"),
            task_sham_value=("task_sham_value", "mean"),
            task_active_value=("task_active_value", "mean"),
            task_delta=("task_delta", "mean"),
            interaction_vigour_minus_task=("interaction_vigour_minus_task", "mean"),
            n_runs=("interaction_vigour_minus_task", "size"),
        )
        .sort_values(["feature", "subject"])
        .reset_index(drop=True)
    )
    subject_values.insert(1, "session", int(session))
    subject_values.insert(2, "medication", medication.upper())
    expected_interaction = subject_values["vigour_delta"] - subject_values["task_delta"]
    if not np.allclose(
        subject_values["interaction_vigour_minus_task"],
        expected_interaction,
        rtol=1e-12,
        atol=1e-15,
    ):
        raise RuntimeError("Run-averaged interaction does not equal the paired subject contrast")
    return run_values, subject_values


def _cohen_dz(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return np.nan
    sd = float(np.std(values, ddof=1))
    return float(np.mean(values) / sd) if sd > 0.0 else np.nan


def interaction_stats(subject_values: pd.DataFrame) -> pd.DataFrame:
    """Test interactions and apply the original four-feature FDR per GVS."""
    rows = []

    group_columns = ["session", "medication", "active_gvs", "feature"]
    grouped_values = subject_values.groupby(group_columns, sort=True)
    for (
        session,
        medication,
        active_gvs,
        feature,
    ), feature_values in grouped_values:
        interaction = feature_values[INTERACTION_COLUMN].to_numpy(dtype=np.float64)
        interaction_p, interaction_exact = _exact_sign_flip_pvalue(interaction)

        rows.append(
            {
                "active_gvs": active_gvs,
                "gvs_label": gvs_display_label(active_gvs),
                "sham_gvs": SHAM_CODE,
                "session": int(session),
                "medication": str(medication).upper(),
                "feature": feature,
                "label": FEATURE_LABELS[feature],
                "n_subjects": int(len(feature_values)),
                "n_run_pairs": int(feature_values["n_runs"].sum()),
                "interaction_mean_vigour_minus_task": float(np.mean(interaction)),
                "interaction_median_vigour_minus_task": float(np.median(interaction)),
                "interaction_cohen_dz": _cohen_dz(interaction),
                "interaction_p_perm": interaction_p,
                "interaction_p_perm_exact": bool(interaction_exact),
            }
        )

    stats = pd.DataFrame(rows)
    stats[INTERACTION_Q_COLUMN] = np.nan
    fdr_groups = stats.groupby(
        ["session", "medication", "active_gvs"],
        sort=True,
    ).groups
    for _, indices in fdr_groups.items():
        if len(indices) != len(FEATURES):
            raise RuntimeError(
                f"Expected {len(FEATURES)} interaction feature tests per active "
                f"stimulation, found {len(indices)}"
            )
        stats.loc[indices, INTERACTION_Q_COLUMN] = _fdr_bh(
            stats.loc[indices, "interaction_p_perm"].to_numpy(dtype=np.float64)
        )
    return stats.sort_values(
        ["session", "medication", "feature", "active_gvs"]
    ).reset_index(drop=True)


def _active_gvs_codes(
    vigour_runs: pd.DataFrame,
    task_runs: pd.DataFrame,
    session: int,
    medication: str,
) -> list[str]:
    code_sets = []
    for run_values in (vigour_runs, task_runs):
        selected = run_values.loc[
            run_values["session"].astype(int).eq(int(session))
            & run_values["medication"].astype(str).str.upper().eq(medication.upper()),
            "gvs_code",
        ]
        codes = set(selected.dropna().astype(str))
        if SHAM_CODE not in codes:
            raise ValueError(f"Missing sham data for session {session} / {medication}")
        code_sets.append(codes - {SHAM_CODE})

    if code_sets[0] != code_sets[1]:
        raise ValueError(
            f"Vigour/task stimulation codes differ for session {session} / {medication}"
        )
    active_codes = sorted(code_sets[0])
    if len(active_codes) != 8:
        raise ValueError(
            f"Expected 8 active stimulation conditions for session {session} / "
            f"{medication}, found {len(active_codes)}"
        )
    return active_codes


def make_all_interaction_values(
    vigour_runs: pd.DataFrame,
    task_runs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_tables = []
    subject_tables = []
    for session, medication, _suffix, _title in SESSIONS:
        for active_gvs in _active_gvs_codes(
            vigour_runs,
            task_runs,
            session,
            medication,
        ):
            run_values, subject_values = make_interaction_values(
                vigour_runs,
                task_runs,
                active_gvs,
                session,
                medication,
            )
            run_tables.append(run_values)
            subject_tables.append(subject_values)

    run_values = pd.concat(run_tables, ignore_index=True).sort_values(
        ["session", "medication", "active_gvs", "feature", "subject", "run"]
    )
    subject_values = pd.concat(subject_tables, ignore_index=True).sort_values(
        ["session", "medication", "active_gvs", "feature", "subject"]
    )
    return run_values.reset_index(drop=True), subject_values.reset_index(drop=True)


def plot_feature_interaction(
    subject_values: pd.DataFrame,
    stats: pd.DataFrame,
    output_path: Path,
    feature: str,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"],
            "font.size": 12,
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(
        1,
        len(SESSIONS),
        figsize=(13.2, 4.6),
        constrained_layout=True,
    )
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.04, hspace=0.04)
    rng = np.random.default_rng(0)
    violin_color = "#8fb3c8"
    violin_edge_color = "#466f86"
    box_color = "#f6f6f2"
    significant_violin_color = "#d98c4a"
    significant_violin_edge_color = "#8f4d1f"
    significant_box_color = "#f4c08d"
    point_color = "#222222"
    feature_values = subject_values.loc[subject_values["feature"].eq(feature)]
    y_limits = padded_limits(
        [
            feature_values.loc[
                feature_values["session"].astype(int).eq(int(session))
                & feature_values["medication"]
                .astype(str)
                .str.upper()
                .eq(medication.upper()),
                INTERACTION_COLUMN,
            ].to_numpy(dtype=np.float64)
            for session, medication, _suffix, _title in SESSIONS
        ]
    )

    for ax_index, (ax, session_info) in enumerate(
        zip(np.atleast_1d(axes), SESSIONS)
    ):
        session, medication, _suffix, _title = session_info
        selected = subject_values.loc[
            subject_values["session"].astype(int).eq(int(session))
            & subject_values["medication"].astype(str).str.upper().eq(medication.upper())
            & subject_values["feature"].eq(feature)
        ]
        active_codes = sorted(selected["active_gvs"].unique())
        positions = np.arange(len(active_codes), dtype=float)
        groups = [
            selected.loc[selected["active_gvs"].eq(active_gvs), INTERACTION_COLUMN]
            .dropna()
            .to_numpy(dtype=np.float64)
            for active_gvs in active_codes
        ]
        if len(active_codes) != 8 or any(values.size == 0 for values in groups):
            raise ValueError(
                f"Incomplete interaction values for {FEATURE_LABELS[feature]}, "
                f"session {session} / {medication}"
            )

        stats_lookup = (
            stats.loc[
                stats["session"].astype(int).eq(int(session))
                & stats["medication"].astype(str).str.upper().eq(medication.upper())
                & stats["feature"].eq(feature)
            ]
            .set_index("active_gvs")[INTERACTION_Q_COLUMN]
            .to_dict()
        )
        significant_groups = [
            np.isfinite(float(stats_lookup.get(code, np.nan)))
            and float(stats_lookup.get(code, np.nan)) < SIGNIFICANCE_Q_THRESHOLD
            for code in active_codes
        ]

        violins = ax.violinplot(
            groups,
            positions=positions,
            widths=0.74,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        for body, is_significant in zip(violins["bodies"], significant_groups):
            body.set_facecolor(
                significant_violin_color if is_significant else violin_color
            )
            body.set_edgecolor(
                significant_violin_edge_color if is_significant else violin_edge_color
            )
            body.set_alpha(0.72)
            body.set_linewidth(1.0 if is_significant else 0.8)

        box = ax.boxplot(
            groups,
            positions=positions,
            widths=0.28,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#111111", "linewidth": 1.2},
            whiskerprops={"color": "#333333", "linewidth": 0.8},
            capprops={"color": "#333333", "linewidth": 0.8},
        )
        for patch, is_significant in zip(box["boxes"], significant_groups):
            patch.set_facecolor(significant_box_color if is_significant else box_color)
            patch.set_edgecolor(
                significant_violin_edge_color if is_significant else "#333333"
            )
            patch.set_linewidth(1.0 if is_significant else 0.8)

        for position, values in zip(positions, groups):
            jitter = rng.uniform(-0.12, 0.12, size=values.size)
            ax.scatter(
                np.full(values.size, position) + jitter,
                values,
                s=19,
                color=point_color,
                alpha=0.68,
                linewidths=0,
                zorder=3,
            )

        ax.set_ylim(*y_limits)
        ax.axhline(0.0, color="#222222", linewidth=0.8)
        n_subjects = int(selected["subject"].nunique())
        ax.set_title(
            f"Medication {medication.upper()}, session {session} (n={n_subjects})",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xticks(positions)
        ax.set_xticklabels(
            [gvs_display_label(code) for code in active_codes],
            rotation=0,
            fontsize=12,
            fontweight="bold",
        )
        ax.set_xlim(positions[0] - 0.5, positions[-1] + 0.5)
        ax.tick_params(axis="both", labelsize=12)
        if ax_index == 0:
            ax.set_ylabel(
                "Difference in GVS effects\n(Vigour − task map)",
                fontsize=13,
                fontweight="bold",
            )
        else:
            ax.set_ylabel("")
        ax.grid(axis="y", color="#e8e8e8", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
        for label in ax.get_yticklabels():
            label.set_fontweight("bold")

    fig.suptitle(
        FEATURE_LABELS[feature],
        fontsize=16,
        fontweight="bold",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create feature-specific vigour/task-map interaction violin plots for "
            "all GVS conditions and both medication/session cells."
        )
    )
    parser.add_argument("--vigour-dir", type=Path, default=DEFAULT_VIGOUR_DIR)
    parser.add_argument("--task-dir", type=Path, default=DEFAULT_TASK_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    vigour_runs = pd.read_csv(args.vigour_dir / RUN_FEATURES_NAME)
    task_runs = pd.read_csv(args.task_dir / RUN_FEATURES_NAME)
    run_values, subject_values = make_all_interaction_values(
        vigour_runs,
        task_runs,
    )
    stats = interaction_stats(subject_values)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    data_stem = "gvs_vigour_task_feature_interaction_all_stimulations"
    run_path = args.out_dir / f"{data_stem}_run_values.csv"
    subject_path = args.out_dir / f"{data_stem}_subject_values.csv"
    stats_path = args.out_dir / f"{data_stem}_stats.csv"
    run_output_columns = [
        *PAIR_KEYS,
        "active_gvs",
        "sham_gvs",
        "feature",
        "label",
        INTERACTION_COLUMN,
    ]
    subject_output_columns = [
        "subject",
        "session",
        "medication",
        "active_gvs",
        "sham_gvs",
        "feature",
        "label",
        INTERACTION_COLUMN,
        "n_runs",
    ]
    run_values[run_output_columns].to_csv(run_path, index=False)
    subject_values[subject_output_columns].to_csv(subject_path, index=False)
    stats.to_csv(stats_path, index=False)

    print(stats.to_string(index=False, float_format=lambda value: f"{value:.8g}"))
    for feature in FEATURES:
        figure_path = args.out_dir / (
            f"gvs_vigour_task_feature_interaction_{feature}_"
            "medication_off_on_violin_boxplot.png"
        )
        plot_feature_interaction(subject_values, stats, figure_path, feature)
        print(f"\nSaved {figure_path}")
    print(f"Saved {run_path}")
    print(f"Saved {subject_path}")
    print(f"Saved {stats_path}")


if __name__ == "__main__":
    main()
