#!/usr/bin/env python3
"""Render two draft layouts for the projection-vs-behaviour main figure.

Reads the cached run-metrics CSV (no neuroimaging recompute) and writes:
  - draft_estimation.png  : single Gardner-Altman estimation panel (A+B merged)
  - draft_two_panel.png   : tightened two-panel version
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import projected_sig_vs_RT as P

CSV = P.DEFAULT_OUT_DIR / "projection_behavior_run_metrics.csv"
OUT = P.DEFAULT_OUT_DIR


def load_subject_df() -> pd.DataFrame:
    df = pd.read_csv(CSV)
    proj_col = "adjacent_diff_ratio_sum_projection"
    beh_col = "adjacent_diff_ratio_sum_behavior_col2"
    paired = df.loc[np.isfinite(df[proj_col]) & np.isfinite(df[beh_col])].copy()
    paired["projection_raw"] = paired[proj_col].to_numpy(float)
    paired["behavior_raw"] = paired[beh_col].to_numpy(float)
    return P._subject_level_pairs(paired)


# --------------------------------------------------------------------------
# Variant 1: single Gardner-Altman estimation plot
# --------------------------------------------------------------------------
def draft_estimation(subject_df: pd.DataFrame) -> Path:
    plot_df = subject_df.sort_values("behaviour_minus_projection").reset_index(drop=True)
    beh = plot_df["behavior_raw"].to_numpy(float)
    proj = plot_df["projection_raw"].to_numpy(float)
    reductions = subject_df["behaviour_minus_projection"].to_numpy(float)
    reductions = reductions[np.isfinite(reductions)]

    with plt.rc_context(
        {
            "font.size": 8.0,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 8.6,
            "ytick.labelsize": 8.6,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
        }
    ):
        fig, (ax, axd) = plt.subplots(
            1,
            2,
            figsize=(4.4, 3.2),
            gridspec_kw={"width_ratios": [1.0, 0.55], "wspace": 0.08},
        )

        # ---- paired slopegraph (left) ----
        jit = np.linspace(-0.04, 0.04, beh.size) if beh.size > 1 else np.array([0.0])
        xb, xp = jit, 1.0 + jit
        for x0, x1, y0, y1 in zip(xb, xp, beh, proj):
            ax.plot([x0, x1], [y0, y1], color="0.55", lw=0.55, alpha=0.30, zorder=1)
        ax.scatter(xb, beh, s=P.SUBJECT_MARKER_SIZE, facecolors=P.BEHAVIOUR_COLOR,
                   edgecolors="white", linewidths=0.25, alpha=0.75, zorder=3)
        ax.scatter(xp, proj, s=P.SUBJECT_MARKER_SIZE, facecolors=P.PROJECTION_COLOR,
                   edgecolors="white", linewidths=0.25, alpha=0.75, zorder=3)
        beh_mean = float(np.mean(beh))
        proj_mean = float(np.mean(proj))
        P._draw_mean_ci(ax, 0.0, beh, P.BEHAVIOUR_COLOR)
        P._draw_mean_ci(ax, 1.0, proj, P.PROJECTION_COLOR)

        y_lo, y_hi = P._expanded_limits(np.concatenate([beh, proj]))
        ax.set_xlim(-0.35, 1.35)
        ax.set_ylim(y_lo, y_hi)
        ax.set_xticks([0.0, 1.0])
        ax.set_xticklabels(["Behaviour", "Projection"])
        ax.set_ylabel(P.VARIABILITY_AXIS_LABEL)
        ax.grid(axis="y", ls="-", lw=0.45, alpha=0.18)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.spines["left"].set_color("0.25")
        ax.spines["bottom"].set_color("0.25")

        # ---- floating difference axis (right), aligned so 0 == behaviour mean ----
        mean, ci_lo, ci_hi = P._mean_ci(reductions)
        p = stats.ttest_1samp(reductions, 0.0).pvalue if reductions.size > 1 else np.nan
        # difference axis: value d sits at behaviour-mean - d on the shared y-scale
        # so projection mean (reduction = mean) lines up visually.
        axd.set_ylim(y_lo, y_hi)
        axd.set_xlim(-0.5, 1.0)
        # map reduction value -> shared-scale y
        def to_shared(d):
            return beh_mean - d
        # bootstrap distribution of the mean reduction
        rng = np.random.default_rng(7)
        boot = np.array([np.mean(rng.choice(reductions, reductions.size, replace=True))
                         for _ in range(5000)])
        violin_y = to_shared(boot)
        parts = axd.violinplot([violin_y], positions=[0.18], widths=0.5,
                               showextrema=False)
        for b in parts["bodies"]:
            b.set_facecolor(P.DIFFERENCE_COLOR)
            b.set_alpha(0.25)
            b.set_edgecolor("none")
        axd.plot([0.18], [to_shared(mean)], marker=P.MEAN_MARKER, ms=5.6,
                 mfc="white", mec="0.08", mew=1.2, zorder=5)
        axd.plot([0.18, 0.18], [to_shared(ci_lo), to_shared(ci_hi)],
                 color="0.08", lw=1.2, zorder=4)
        # reference line at zero reduction (== behaviour mean)
        axd.axhline(to_shared(0.0), color="0.30", ls=(0, (4, 2)), lw=0.9, zorder=0)

        # secondary right-side ticks in reduction units
        axd.spines["left"].set_visible(False)
        axd.spines["top"].set_visible(False)
        axd.spines["bottom"].set_visible(False)
        axd.spines["right"].set_position(("axes", 1.0))
        axd.spines["right"].set_color("0.25")
        axd.yaxis.set_label_position("right")
        axd.yaxis.tick_right()
        # build reduction ticks
        red_ticks = np.array([-10, 0, 10, 20, 30])
        red_ticks = red_ticks[(to_shared(red_ticks) >= y_lo) & (to_shared(red_ticks) <= y_hi)]
        axd.set_yticks(to_shared(red_ticks))
        axd.set_yticklabels([f"{t:g}" for t in red_ticks])
        axd.set_ylabel("Δ variability (Behaviour − Projection)")
        axd.set_xticks([0.18])
        axd.set_xticklabels(["Paired\nmean diff."])
        axd.tick_params(axis="x", length=0)

        axd.text(0.97, 0.02,
                 f"Δ = {mean:.1f}\n95% CI [{ci_lo:.1f}, {ci_hi:.1f}]\n{P._format_p_value(p)}",
                 transform=axd.transAxes, ha="right", va="bottom", fontsize=6.7,
                 linespacing=1.25)

        fig.subplots_adjust(left=0.13, right=0.85, bottom=0.13, top=0.93)
        out = OUT / "draft_estimation.png"
        fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
    return out


# --------------------------------------------------------------------------
# Variant 2: tightened two-panel
# --------------------------------------------------------------------------
def draft_two_panel(subject_df: pd.DataFrame) -> Path:
    finite = np.concatenate([subject_df["behavior_raw"].to_numpy(float),
                             subject_df["projection_raw"].to_numpy(float)])
    y_limits = P._expanded_limits(finite)
    with plt.rc_context(
        {
            "font.size": 8.0,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 8.6,
            "ytick.labelsize": 8.6,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
        }
    ):
        fig, axes = plt.subplots(
            1, 2, figsize=(5.0, 2.85),
            gridspec_kw={"width_ratios": [1.0, 1.0], "wspace": 0.45},
        )
        P._plot_paired_estimation(axes[0], subject_df, y_limits)

        # ---- panel B, tightened ----
        ax = axes[1]
        reductions = subject_df["behaviour_minus_projection"].to_numpy(float)
        finite_r = reductions[np.isfinite(reductions)]
        rng = np.random.default_rng(141)
        x = rng.uniform(-0.05, 0.05, size=reductions.size)
        ax.scatter(x, reductions, s=P.SUBJECT_MARKER_SIZE, facecolors=P.DIFFERENCE_COLOR,
                   alpha=0.75, edgecolors="white", linewidths=0.25, zorder=3)
        mean, ci_lo, ci_hi = P._draw_mean_ci(ax, 0.16, reductions, "0.08", markersize=5.6)
        p = stats.ttest_1samp(finite_r, 0.0).pvalue if finite_r.size > 1 else np.nan
        y_lo, y_hi = P._expanded_limits(reductions, force_zero=True)
        y_hi += (y_hi - y_lo) * 0.16
        ax.axhline(0.0, color="0.30", ls=(0, (4, 2)), lw=0.9, zorder=0)
        ax.set_xlim(-0.13, 0.27)  # tightened: was (-0.18, 0.32)
        ax.set_ylim(y_lo, y_hi)
        ax.set_xticks([0.0, 0.16])
        ax.set_xticklabels(["Subjects", "Mean"])
        ax.set_ylabel("Δ variability\n(Behaviour − Projection)")
        ax.grid(axis="y", ls="-", lw=0.45, alpha=0.18)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.spines["left"].set_color("0.25")
        ax.spines["bottom"].set_color("0.25")
        ax.text(0.98, 0.97,
                f"Δ = {mean:.1f}\n95% CI [{ci_lo:.1f}, {ci_hi:.1f}]\n{P._format_p_value(p)}",
                transform=ax.transAxes, ha="right", va="top", fontsize=6.7, linespacing=1.25)

        axes[0].text(0.01, 0.99, "A", transform=axes[0].transAxes, fontweight="bold",
                     fontsize=10.0, va="top")
        axes[1].text(0.01, 0.99, "B", transform=axes[1].transAxes, fontweight="bold",
                     fontsize=10.0, va="top")
        fig.subplots_adjust(left=0.11, right=0.97, bottom=0.13, top=0.92, wspace=0.45)
        out = OUT / "draft_two_panel.png"
        fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
    return out


if __name__ == "__main__":
    sdf = load_subject_df()
    print("n subjects:", len(sdf))
    print(draft_estimation(sdf))
    print(draft_two_panel(sdf))
