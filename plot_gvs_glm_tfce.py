#!/usr/bin/env python3
"""Glass-brain overview of the whole-brain GVS GLM+TFCE results.

Rows = parametric contrasts (gvs_main, carrier, envelope).
Cols = OFF, ON, ON-OFF interaction.
Each panel shows the unthresholded second-level t-map (to reveal sub-threshold
spatial trends), titled with the whole-brain min FWE-corrected p (TFCE).
A green outline appears only where FWE p < 0.05 (none, in practice).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import plotting

ROOT = Path(__file__).resolve().parent
D = ROOT / "figures" / "GVS_effects" / "glm_tfce"

CONTRASTS = ["gvs_main", "carrier", "envelope"]
ANALYSES = ["OFF", "ON", "interaction_ON_minus_OFF"]
ROW_TITLE = {"gvs_main": "GVS - sham", "carrier": "125 - 75 Hz", "envelope": "envelope trend"}
COL_TITLE = {"OFF": "OFF", "ON": "ON", "interaction_ON_minus_OFF": "ON - OFF"}

summary = pd.read_csv(D / "tfce_summary.csv")


def min_fwe_p(contrast: str, analysis: str) -> float:
    row = summary[(summary.contrast == contrast) & (summary.analysis == analysis)]
    return float(row["min_fwe_p"].iloc[0]) if len(row) else float("nan")


fig, axes = plt.subplots(len(CONTRASTS), len(ANALYSES), figsize=(15, 9))
for i, contrast in enumerate(CONTRASTS):
    # shared color scale per row from the t-maps
    tmaps = {a: D / f"{contrast}__{a}_t.nii.gz" for a in ANALYSES}
    vmax = 0.0
    for p in tmaps.values():
        if p.exists():
            vmax = max(vmax, float(np.nanmax(np.abs(nib.load(str(p)).get_fdata()))))
    vmax = max(vmax, 1.0)
    for j, analysis in enumerate(ANALYSES):
        ax = axes[i, j]
        tpath = tmaps[analysis]
        if not tpath.exists():
            ax.axis("off")
            continue
        p_fwe = min_fwe_p(contrast, analysis)
        disp = plotting.plot_glass_brain(
            str(tpath), display_mode="z", axes=ax, colorbar=(j == len(ANALYSES) - 1),
            plot_abs=False, vmax=vmax, cmap="cold_hot",
        )
        # overlay FWE-significant voxels if any
        logp = D / f"{contrast}__{analysis}_logp_fwe_tfce.nii.gz"
        if logp.exists():
            data = nib.load(str(logp)).get_fdata()
            if np.nanmax(data) >= -np.log10(0.05):
                disp.add_contours(str(logp), levels=[-np.log10(0.05)], colors="lime", linewidths=1.5)
        sig = "FWE n.s." if not (p_fwe < 0.05) else "FWE<.05"
        ax.set_title(f"{ROW_TITLE[contrast]} | {COL_TITLE[analysis]}\nmin FWE p={p_fwe:.3f} ({sig})",
                     fontsize=10)

fig.suptitle("Whole-brain GVS effects (voxelwise GLM + TFCE permutation, FWE-corrected)\n"
             "unthresholded t-maps shown; nothing survives FWE<0.05",
             fontsize=13, y=1.005)
fig.tight_layout()
out = D / "gvs_glm_tfce_overview.png"
fig.savefig(str(out), dpi=170, bbox_inches="tight")
print(f"wrote {out}")
