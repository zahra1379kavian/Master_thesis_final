#!/usr/bin/env python3
"""Figure for the searchlight decoding results: AUC is at chance everywhere.

Top row: distribution of group-mean searchlight AUC across voxels (stim, carrier),
with the chance line at 0.5 -> shows a narrow blob on chance.
Bottom row: glass-brain second-level t-maps (AUC-0.5) for OFF, titled with the
whole-brain min FWE p (TFCE). Nothing survives.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import plotting

ROOT = Path(__file__).resolve().parent
D = ROOT / "figures" / "GVS_effects" / "searchlight_decoding"
TARGETS = ["stim", "carrier"]
TITLE = {"stim": "sham vs GVS", "carrier": "75 vs 125 Hz"}


def group_mean_auc_map(target: str):
    files = sorted((D / f"acc_maps_{target}").glob("*_auc.nii.gz"))
    ref = nib.load(str(files[0]))
    stack = np.stack([nib.load(str(f)).get_fdata() for f in files], 0)
    with np.errstate(invalid="ignore"):
        m = np.where((stack != 0).any(0), np.nanmean(np.where(stack == 0, np.nan, stack), 0), np.nan)
    return nib.Nifti1Image(np.nan_to_num(m).astype(np.float32), ref.affine), m


fig = plt.figure(figsize=(13, 8))
summ = {t: pd.read_csv(D / f"tfce_summary_{t}.csv") for t in TARGETS}

for j, t in enumerate(TARGETS):
    auc_img, m = group_mean_auc_map(t)
    vals = m[np.isfinite(m) & (m != 0)]

    axh = fig.add_subplot(2, 2, j + 1)
    axh.hist(vals, bins=80, color="#4477aa", alpha=0.85)
    axh.axvline(0.5, color="crimson", lw=2, label="chance (0.5)")
    axh.axvline(float(np.mean(vals)), color="k", lw=1.2, ls="--", label=f"mean={np.mean(vals):.3f}")
    axh.set_title(f"{TITLE[t]}: searchlight AUC across {vals.size:,} voxels", fontsize=11)
    axh.set_xlabel("group-mean decoding AUC"); axh.set_ylabel("voxels"); axh.legend(fontsize=9)
    axh.set_xlim(0.42, 0.58)

    ax = fig.add_subplot(2, 2, j + 3)
    tpath = D / f"{t}__OFF_t.nii.gz"
    p = float(summ[t].query("analysis=='OFF'")["min_fwe_p"].iloc[0])
    plotting.plot_glass_brain(str(tpath), display_mode="z", axes=ax, colorbar=True,
                              plot_abs=False, cmap="cold_hot")
    ax.set_title(f"{TITLE[t]} | OFF: 2nd-level t(AUC-0.5)\nmin FWE p={p:.3f} (n.s., 0 sig voxels)", fontsize=10)

fig.suptitle("Searchlight MVPA: GVS is not decodable from local BOLD patterns\n"
             "per-voxel AUC sits on chance; nothing survives TFCE FWE<0.05",
             fontsize=13, y=1.0)
fig.tight_layout()
out = D / "searchlight_decoding_overview.png"
fig.savefig(str(out), dpi=170, bbox_inches="tight")
print(f"wrote {out}")
