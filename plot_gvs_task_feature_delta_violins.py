#!/usr/bin/env python3
"""Plot session-specific subject distributions for GVS feature deltas."""

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


ROOT = Path(__file__).resolve().parent
DEFAULT_IN_DIR = ROOT / "figures" / "gvs_task_map_bold_features_vs_sham"
RUN_FEATURES_NAME = "gvs_run_signal_features.csv"
SHAM_CODE = "gvs-01"
SIGNIFICANCE_Q_THRESHOLD = 0.06

FEATURES = [
    "early_late_change",
    "slope",
    "abs_baseline_response",
    "peak_to_peak",
]

FEATURE_LABELS = {
    "early_late_change": "Late minus early",
    "slope": "Linear slope",
    "abs_baseline_response": "Max abs. baseline response",
    "peak_to_peak": "Peak-to-peak amplitude",
}

SESSIONS = [
    (1, "OFF", "session1_off", "Medication OFF, session 1"),
    (2, "ON", "session2_on", "Medication ON, session 2"),
]

def gvs_display_label(gvs_code: str) -> str:
    code = str(gvs_code).strip().lower().replace("_", "-").replace(" ", "-")
    if code in {SHAM_CODE, "sham"}:
        return "sham"
    if not code.startswith("gvs-"):
        return str(gvs_code)
    try:
        index = int(code.split("-", 1)[1])
    except ValueError:
        return str(gvs_code)
    return f"gvs{index - 1}" if index > 1 else "sham"


def make_session_subject_pairs(
    run_features: pd.DataFrame,
    session: int,
    medication: str,
    sham_code: str = SHAM_CODE,
) -> pd.DataFrame:
    session_runs = run_features.loc[
        run_features["session"].astype(int).eq(int(session))
        & run_features["medication"].astype(str).str.upper().eq(medication.upper())
    ].copy()
    if session_runs.empty:
        raise ValueError(f"No run-level features found for session {session} / {medication}")

    index_cols = ["subject", "session", "medication", "run"]
    sham_runs = session_runs.loc[session_runs["gvs_code"].eq(sham_code)].set_index(index_cols)
    active_codes = sorted(code for code in session_runs["gvs_code"].unique() if code != sham_code)
    rows = []

    for active_code in active_codes:
        active_runs = session_runs.loc[session_runs["gvs_code"].eq(active_code)].set_index(index_cols)
        paired_runs = sham_runs.join(active_runs, how="inner", lsuffix="_sham", rsuffix="_active")
        if paired_runs.empty:
            continue

        for feature in FEATURES:
            per_run = pd.DataFrame(
                {
                    "subject": paired_runs.index.get_level_values("subject"),
                    "sham_value": paired_runs[f"{feature}_sham"].to_numpy(dtype=np.float64),
                    "active_value": paired_runs[f"{feature}_active"].to_numpy(dtype=np.float64),
                }
            )
            per_run["delta_active_minus_sham"] = per_run["active_value"] - per_run["sham_value"]
            per_subject = (
                per_run.replace([np.inf, -np.inf], np.nan)
                .dropna(subset=["sham_value", "active_value", "delta_active_minus_sham"])
                .groupby("subject", as_index=False)
                .agg(
                    sham_value=("sham_value", "mean"),
                    active_value=("active_value", "mean"),
                    delta_active_minus_sham=("delta_active_minus_sham", "mean"),
                    n_runs=("delta_active_minus_sham", "size"),
                )
            )
            per_subject.insert(0, "feature", feature)
            per_subject.insert(0, "label", FEATURE_LABELS[feature])
            per_subject.insert(0, "sham_gvs", sham_code)
            per_subject.insert(0, "active_gvs", active_code)
            rows.append(per_subject)

    if not rows:
        raise RuntimeError(f"No active-vs-sham subject pairs could be built against {sham_code}")
    return (
        pd.concat(rows, ignore_index=True)
        .sort_values(["active_gvs", "feature", "subject"])
        .reset_index(drop=True)
    )


def padded_limits(values: list[np.ndarray]) -> tuple[float, float]:
    finite = np.concatenate([np.asarray(v, dtype=float) for v in values])
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return -1.0, 1.0
    lo = min(0.0, float(np.nanmin(finite)))
    hi = max(0.0, float(np.nanmax(finite)))
    span = hi - lo
    if span == 0.0:
        span = max(abs(lo), abs(hi), 1.0)
    return lo - 0.08 * span, hi + 0.08 * span


def plot_session_violins(
    subject_pairs: pd.DataFrame,
    stats_df: pd.DataFrame,
    output_path: Path,
) -> None:
    active_codes = sorted(subject_pairs["active_gvs"].unique())
    x = np.arange(len(active_codes))
    rng = np.random.default_rng(0)

    stats_lookup = (
        stats_df.set_index(["active_gvs", "feature"])["q_perm_fdr"].to_dict()
        if not stats_df.empty
        else {}
    )

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

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 7.4), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.04, hspace=0.04)
    axes = axes.ravel()
    violin_color = "#8fb3c8"
    violin_edge_color = "#466f86"
    box_color = "#f6f6f2"
    significant_violin_color = "#d98c4a"
    significant_violin_edge_color = "#8f4d1f"
    significant_box_color = "#f4c08d"
    point_color = "#222222"

    for ax_index, (ax, feature) in enumerate(zip(axes, FEATURES)):
        row, col = divmod(ax_index, 2)
        groups = [
            subject_pairs.loc[
                subject_pairs["active_gvs"].eq(code) & subject_pairs["feature"].eq(feature),
                "delta_active_minus_sham",
            ]
            .dropna()
            .to_numpy(dtype=np.float64)
            for code in active_codes
        ]
        significant_groups = [
            np.isfinite(float(stats_lookup.get((code, feature), np.nan)))
            and float(stats_lookup.get((code, feature), np.nan)) < SIGNIFICANCE_Q_THRESHOLD
            for code in active_codes
        ]

        violins = ax.violinplot(
            groups,
            positions=x,
            widths=0.74,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        for body, is_significant in zip(violins["bodies"], significant_groups):
            body.set_facecolor(significant_violin_color if is_significant else violin_color)
            body.set_edgecolor(significant_violin_edge_color if is_significant else violin_edge_color)
            body.set_alpha(0.72)
            body.set_linewidth(1.0 if is_significant else 0.8)

        box = ax.boxplot(
            groups,
            positions=x,
            widths=0.28,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#111111", "linewidth": 1.2},
            whiskerprops={"color": "#333333", "linewidth": 0.8},
            capprops={"color": "#333333", "linewidth": 0.8},
        )
        for patch, is_significant in zip(box["boxes"], significant_groups):
            patch.set_facecolor(significant_box_color if is_significant else box_color)
            patch.set_edgecolor(significant_violin_edge_color if is_significant else "#333333")
            patch.set_linewidth(1.0 if is_significant else 0.8)

        for xpos, values in zip(x, groups):
            jitter = rng.uniform(-0.12, 0.12, size=values.size)
            ax.scatter(
                np.full(values.size, xpos) + jitter,
                values,
                s=19,
                color=point_color,
                alpha=0.68,
                linewidths=0,
                zorder=3,
            )

        y_min, y_max = padded_limits(groups)
        ax.set_ylim(y_min, y_max)
        ax.axhline(0.0, color="#222222", linewidth=0.8)
        ax.set_title(FEATURE_LABELS[feature], fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xlim(x[0] - 0.5, x[-1] + 0.5)
        ax.set_xticklabels(
            [gvs_display_label(code) for code in active_codes],
            rotation=0,
            fontsize=12,
            fontweight="bold",
        )
        ax.tick_params(axis="both", labelsize=12)
        for label in ax.get_yticklabels():
            label.set_fontweight("bold")
        if row == 0:
            ax.tick_params(axis="x", labelbottom=False)
        if col == 0:
            ax.set_ylabel("Active - sham feature value", fontsize=13, fontweight="bold")
        else:
            ax.set_ylabel("")
        ax.grid(axis="y", color="#e8e8e8", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create session-specific violin/box plots for GVS feature deltas."
    )
    parser.add_argument("--in-dir", type=Path, default=DEFAULT_IN_DIR)
    parser.add_argument("--write-subject-values", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_features = pd.read_csv(args.in_dir / RUN_FEATURES_NAME)

    for session, medication, suffix, _title in SESSIONS:
        subject_pairs = make_session_subject_pairs(run_features, session, medication)
        stats_path = args.in_dir / f"gvs_vs_sham_signal_feature_stats_{suffix}.csv"
        stats_df = pd.read_csv(stats_path) if stats_path.exists() else pd.DataFrame()
        output_path = args.in_dir / f"gvs_vs_sham_feature_delta_violin_boxplot_{suffix}.png"
        plot_session_violins(subject_pairs, stats_df, output_path)
        if args.write_subject_values:
            subject_pairs.to_csv(
                args.in_dir / f"gvs_vs_sham_subject_signal_feature_pairs_{suffix}.csv",
                index=False,
            )
        n_subjects = int(subject_pairs["subject"].nunique())
        print(f"Saved {output_path} ({n_subjects} subjects)")


if __name__ == "__main__":
    main()
