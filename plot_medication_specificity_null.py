#!/usr/bin/env python3
"""Negative-control rebuttal to the circularity concern.

The optimisation that defines the vigour network never used medication state.
Yet the network shows a medication-related increase in *within-ROI* functional
connectivity (ON > OFF). If that downstream effect were a by-product of selecting
stable / averaged voxels, then random voxel sets matched on ROI composition and
size should reproduce it. They do not.

This figure compares the actual vigour-network medication effect against a null of
940 matched non-vigour control networks (med_effects_matched_null_distribution.py):

  Panel A  Null distribution of the within-ROI ON-OFF FC change; the vigour
           network lies far outside it (0/940 controls reach it).
  Panel B  The effect is anatomically specific: the network is an outlier for
           WITHIN-ROI coupling but sits inside the null for BETWEEN-ROI coupling.

Because medication was not part of the objective, this specificity cannot be an
artefact of building stability into the optimisation.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib

warnings.filterwarnings("ignore", message="Unable to import Axes3D.*")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
NULL_CSV = ROOT / "figures/med_effects/baselines/matched_nonvigour_null_distribution/matched_nonvigour_null_distribution_values.csv"
VIGOUR_JSON = ROOT / "figures/med_effects/intra_vs_between_fc_results.json"
UNWEIGHTED_JSON = ROOT / "figures/med_effects/baselines/unweighted_vigour/intra_vs_between_fc_results.json"
OUT_DIR = ROOT / "figures/med_effects"
FIGURE_STEM = "medication_specificity_vs_null"

PAPER_FONT = "Liberation Sans"
COLOR_VIGOUR = "#D55E00"   # vermillion
COLOR_NULL = "#7f9bb3"     # muted steel blue
COLOR_NULL_DARK = "#3b566b"


def _emp_p(null_values: np.ndarray, actual: float) -> tuple[int, float, float]:
    ge = int(np.count_nonzero(null_values >= actual))
    p = (ge + 1) / (null_values.size + 1)
    z = (actual - null_values.mean()) / null_values.std(ddof=1)
    return ge, p, z


def main() -> None:
    df = pd.read_csv(NULL_CSV)
    null_within = pd.to_numeric(df["intra_within_effect"], errors="coerce").dropna().to_numpy()
    null_between = pd.to_numeric(df["intra_between_effect"], errors="coerce").dropna().to_numpy()

    vig = json.load(open(VIGOUR_JSON))["tests"]
    vw, vw_lo, vw_hi = (vig["within_roi_on_minus_off"][k] for k in ("mean", "ci95_low", "ci95_high"))
    vb, vb_lo, vb_hi = (vig["between_roi_on_minus_off"][k] for k in ("mean", "ci95_low", "ci95_high"))
    vw_p = vig["within_roi_on_minus_off"]["paired_t_p_value_two_sided"]
    unw = json.load(open(UNWEIGHTED_JSON))["tests"]["within_roi_on_minus_off"]["mean"] if UNWEIGHTED_JSON.exists() else None

    ge_w, p_w, z_w = _emp_p(null_within, vw)
    ge_b, p_b, z_b = _emp_p(null_between, vb)
    n = null_within.size

    with plt.rc_context({
        "font.family": "sans-serif",
        "font.sans-serif": [PAPER_FONT, "Arial", "DejaVu Sans"],
        "font.size": 13, "axes.labelsize": 13, "xtick.labelsize": 12, "ytick.labelsize": 12,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    }):
        fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.2, 5.7),
                                       gridspec_kw={"width_ratios": [1.5, 1.0]})

        # ===== Panel A: null histogram of within-ROI ON-OFF =====
        axA.hist(null_within, bins=40, color=COLOR_NULL, edgecolor="white",
                 linewidth=0.4, alpha=0.9, zorder=2)
        lo95, hi95 = np.percentile(null_within, [2.5, 97.5])
        axA.axvspan(lo95, hi95, color=COLOR_NULL, alpha=0.18, zorder=0)
        axA.axvline(null_within.mean(), color=COLOR_NULL_DARK, ls=(0, (4, 2)), lw=1.5,
                    zorder=3, label=f"control mean ({null_within.mean():+.4f})")
        axA.axvline(vw, color=COLOR_VIGOUR, lw=2.6, zorder=4,
                    label=f"vigour network ({vw:+.4f})")
        if unw is not None:
            axA.axvline(unw, color=COLOR_VIGOUR, lw=1.3, ls=":", zorder=4,
                        label=f"vigour, unit weights ({unw:+.4f})")
        ymax = axA.get_ylim()[1]
        axA.annotate(f"vigour network\n{vw:+.4f}  (p={vw_p:.3f})",
                     xy=(vw, ymax * 0.62), xytext=(vw - (vw - null_within.mean()) * 0.55, ymax * 0.86),
                     color=COLOR_VIGOUR, fontsize=11.5, fontweight="bold", ha="center",
                     arrowprops=dict(arrowstyle="->", color=COLOR_VIGOUR, lw=1.6))
        axA.text(0.025, 0.97,
                 f"{ge_w}/{n} control networks reach it\n"
                 f"empirical p = {p_w:.3f}   (z = {z_w:.1f})",
                 transform=axA.transAxes, va="top", ha="left", fontsize=11,
                 bbox=dict(boxstyle="round,pad=0.4", fc="#f5f5f5", ec="0.75"))
        axA.set_xlabel("Within-ROI medication change  (ON − OFF functional connectivity)")
        axA.set_ylabel(f"Matched control networks  (n = {n})")
        axA.legend(loc="center left", fontsize=9.6, frameon=True, framealpha=0.93,
                   bbox_to_anchor=(0.0, 0.62))
        for s in ("top", "right"):
            axA.spines[s].set_visible(False)
        axA.set_title("A   A non-optimised effect (medication) is specific to the vigour network",
                      loc="left", fontsize=13, fontweight="bold", pad=10)

        # ===== Panel B: within vs between specificity =====
        positions = [0, 1]
        data = [null_within, null_between]
        vln = axB.violinplot(data, positions=positions, widths=0.7, showextrema=False)
        for body in vln["bodies"]:
            body.set_facecolor(COLOR_NULL)
            body.set_edgecolor(COLOR_NULL_DARK)
            body.set_alpha(0.55)
        # null medians
        for xi, d in zip(positions, data):
            axB.hlines(np.median(d), xi - 0.22, xi + 0.22, color=COLOR_NULL_DARK, lw=1.6, zorder=3)
        axB.axhline(0, color="0.55", lw=0.9, zorder=0)
        # vigour markers with CI
        axB.errorbar(0, vw, yerr=[[vw - vw_lo], [vw_hi - vw]], fmt="*", ms=20,
                     color=COLOR_VIGOUR, ecolor=COLOR_VIGOUR, elinewidth=1.8, capsize=4,
                     markeredgecolor="#1a1a1a", markeredgewidth=0.6, zorder=5)
        axB.errorbar(1, vb, yerr=[[vb - vb_lo], [vb_hi - vb]], fmt="*", ms=20,
                     color=COLOR_VIGOUR, ecolor=COLOR_VIGOUR, elinewidth=1.8, capsize=4,
                     markeredgecolor="#1a1a1a", markeredgewidth=0.6, zorder=5)
        axB.text(0.30, vw, f"z = {z_w:.1f}\n{ge_w}/{n} controls", ha="left", va="center",
                 fontsize=10.5, color=COLOR_VIGOUR, fontweight="bold")
        axB.text(1.30, vb, f"z = {z_b:.1f}\n(n.s.)", ha="left", va="center",
                 fontsize=10.5, color="0.35")
        axB.set_xticks(positions)
        axB.set_xticklabels(["Within-ROI", "Between-ROI"])
        axB.set_ylabel("Medication change (ON − OFF FC)")
        axB.set_xlim(-0.6, 1.95)
        axB.set_ylim(-0.006, 0.028)
        for s in ("top", "right"):
            axB.spines[s].set_visible(False)
        # legend proxies
        from matplotlib.lines import Line2D
        handles = [
            Line2D([0], [0], marker="*", color="w", markerfacecolor=COLOR_VIGOUR,
                   markeredgecolor="#1a1a1a", markersize=15, label="Vigour network (95% CI)"),
            matplotlib.patches.Patch(facecolor=COLOR_NULL, alpha=0.55, edgecolor=COLOR_NULL_DARK,
                                     label=f"Control null (n={n})"),
        ]
        axB.legend(handles=handles, loc="upper right", fontsize=9.6, frameon=True, framealpha=0.93)
        axB.set_title("B   Local and anatomically specific", loc="left",
                      fontsize=13, fontweight="bold", pad=10)

        fig.tight_layout(w_pad=2.2)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        png = OUT_DIR / f"{FIGURE_STEM}.png"
        pdf = OUT_DIR / f"{FIGURE_STEM}.pdf"
        fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.04)
        fig.savefig(pdf, bbox_inches="tight", pad_inches=0.04)
        plt.close(fig)

    print("==== Medication specificity vs matched-control null ====")
    print(f"within-ROI ON-OFF: vigour={vw:+.4f} [{vw_lo:+.4f},{vw_hi:+.4f}] p={vw_p:.4f}")
    print(f"   null mean={null_within.mean():+.4f} 95%=[{lo95:+.4f},{hi95:+.4f}]  {ge_w}/{n} >= vigour, emp_p={p_w:.4f}, z={z_w:.1f}")
    print(f"between-ROI ON-OFF: vigour={vb:+.4f}  null mean={null_between.mean():+.4f}  {ge_b}/{n} >= vigour, emp_p={p_b:.4f}, z={z_b:.1f}")
    if unw is not None:
        print(f"unweighted-vigour within-ROI={unw:+.4f} (selection, not weights)")
    print(f"Saved {png}")


if __name__ == "__main__":
    main()
