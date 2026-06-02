#!/usr/bin/env python3
"""ROI-free searchlight MVPA: can GVS be decoded from local BOLD patterns?

Sensitive multivariate complement to the whole-brain univariate test. A spherical
searchlight scans every voxel; within each sphere a linear SVM is cross-validated
(leave-one-run-out) to decode a target from single-trial betas. Per subject x
session this yields a whole-brain decoding-accuracy map (chance = 0.5, ROC AUC).

Group inference: per-subject (AUC - 0.5) maps are tested with sign-flip TFCE
permutation (FWE-corrected) via nilearn non_parametric_inference -- within OFF,
within ON, and paired ON-OFF.

Targets:
  * stim    : sham (0) vs any active GVS (1)   -> "is stimulation detectable at all"
  * carrier : 75 Hz (0) vs 125 Hz (1), active trials only (balanced, parametric)

Stim-order convention: stim value v in {1..9}; v=1 = sham; v>=2 = GVS(v-1).
Grid (Master_paper_new.pdf 135-141): GVS1-4 = 75 Hz carrier, GVS5-8 = 125 Hz;
envelope 0/10/16/20 Hz within each carrier.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from GVS_effect import (
    DEFAULT_BETA_ROOT,
    DEFAULT_GVS_ORDER,
    DEFAULT_TRIALS_PER_CONDITION,
    _discover_run_specs,
    _iter_condition_slices,
    _load_gvs_order,
    _medication_from_session,
    _stim_id_from_condition_code,
)

ROOT = Path(__file__).resolve().parent
DEFAULT_REFERENCE_NIFTI = ROOT / "data" / "voxel_weights_task1_bold0.6_beta0.6_smooth1.25_gamma1.5.nii.gz"
DEFAULT_OUT_DIR = ROOT / "figures" / "GVS_effects" / "searchlight_decoding"

CARRIER_HZ = {2: 75, 3: 75, 4: 75, 5: 75, 6: 125, 7: 125, 8: 125, 9: 125}
SHAM_STIM = 1
CHANCE = 0.5


def _load_session_trials(session_runs, n_vox, trials_per_condition):
    """Stack single-trial betas for one subject-session -> (X[n_trials, n_vox], stim, run)."""
    X_parts, stim_parts, run_parts = [], [], []
    for spec in session_runs:
        beta = np.load(spec.beta_path, mmap_mode="r")
        flat = np.asarray(beta).reshape(n_vox, beta.shape[-1]).T  # (n_trials, n_vox)
        for code, start, stop in _iter_condition_slices(spec.n_trials, spec.stim_order, trials_per_condition):
            stim = _stim_id_from_condition_code(code)
            block = flat[start:stop]
            X_parts.append(block)
            stim_parts.append(np.full(block.shape[0], stim, dtype=np.int16))
            run_parts.append(np.full(block.shape[0], spec.run, dtype=np.int16))
    X = np.concatenate(X_parts, axis=0).astype(np.float32)
    stim = np.concatenate(stim_parts)
    run = np.concatenate(run_parts)
    return X, stim, run


def _labels_for_target(stim: np.ndarray, target: str):
    if target == "stim":
        y = (stim != SHAM_STIM).astype(np.int8)
        keep = np.ones(stim.shape, dtype=bool)
    elif target == "carrier":
        keep = stim != SHAM_STIM
        y = np.full(stim.shape, -1, dtype=np.int8)
        y[keep] = np.array([1 if CARRIER_HZ.get(int(s)) == 125 else 0 for s in stim[keep]], dtype=np.int8)
    else:
        raise ValueError(target)
    return y, keep


def _zscore_fillnan(X: np.ndarray) -> np.ndarray:
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd[~np.isfinite(sd) | (sd == 0)] = 1.0
    mu[~np.isfinite(mu)] = 0.0
    return np.nan_to_num((X - mu) / sd, nan=0.0).astype(np.float32)


def _to_img(flat, mask_flat, shape, affine, fill=0.0):
    vol = np.full(int(np.prod(shape)), fill, dtype=np.float32)
    vol[mask_flat] = flat[mask_flat] if flat.shape == mask_flat.shape else flat
    return nib.Nifti1Image(vol.reshape(shape), affine)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--beta-root", type=Path, default=DEFAULT_BETA_ROOT)
    ap.add_argument("--gvs-order", type=Path, default=DEFAULT_GVS_ORDER)
    ap.add_argument("--reference-nifti", type=Path, default=DEFAULT_REFERENCE_NIFTI)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--trials-per-condition", type=int, default=DEFAULT_TRIALS_PER_CONDITION)
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--target", choices=["stim", "carrier"], default="stim")
    ap.add_argument("--radius-mm", type=float, default=6.0)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--time-one", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ref = nib.load(str(args.reference_nifti))
    shape, affine = ref.shape[:3], ref.affine
    n_vox = int(np.prod(shape))

    gvs_order = _load_gvs_order(args.gvs_order)
    run_specs = _discover_run_specs(args.beta_root, gvs_order, args.subjects)
    sessions: dict[tuple[str, int], list] = {}
    for spec in run_specs:
        sessions.setdefault((spec.subject, int(spec.session)), []).append(spec)
    session_keys = sorted(sessions, key=lambda k: (k[0], k[1]))

    from nilearn.decoding import SearchLight
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.svm import LinearSVC

    estimator = LinearSVC(C=args.C, class_weight="balanced", dual=False, max_iter=5000)
    acc_dir = args.out_dir / f"acc_maps_{args.target}"
    acc_dir.mkdir(parents=True, exist_ok=True)

    acc_maps: dict[tuple[str, int], np.ndarray] = {}
    rows = []
    for idx, key in enumerate(session_keys, start=1):
        subject, session = key
        runs = sorted(sessions[key], key=lambda s: s.run)
        if len({s.run for s in runs}) < 2:
            print(f"skip {subject} ses-{session}: <2 runs"); continue
        cache_path = acc_dir / f"{subject}_ses-{session}_{args.target}_auc.nii.gz"
        if cache_path.exists() and not args.time_one:
            acc_maps[key] = nib.load(str(cache_path)).get_fdata().ravel()
            print(f"[{idx}/{len(session_keys)}] cached {subject} ses-{session}"); continue

        X, stim, run = _load_session_trials(runs, n_vox, args.trials_per_condition)
        y, keep = _labels_for_target(stim, args.target)
        X, y, run = X[keep], y[keep], run[keep]

        finite_frac = np.isfinite(X).mean(axis=0)
        nonconst = np.nanstd(X, axis=0) > 0
        mask_flat = (finite_frac > 0.5) & nonconst
        Z = _zscore_fillnan(X)
        img4d = nib.Nifti1Image(Z.T.reshape(shape + (Z.shape[0],)), affine)
        mask_img = nib.Nifti1Image(mask_flat.reshape(shape).astype(np.uint8), affine)

        t0 = time.time()
        sl = SearchLight(mask_img=mask_img, process_mask_img=mask_img, radius=args.radius_mm,
                         estimator=estimator, scoring="roc_auc", cv=LeaveOneGroupOut(),
                         n_jobs=args.n_jobs, verbose=0)
        sl.fit(img4d, y, groups=run)
        dt = time.time() - t0
        acc = np.asarray(sl.scores_, dtype=np.float32).ravel()
        acc_maps[key] = acc
        _to_img(acc, mask_flat, shape, affine).to_filename(str(cache_path))
        n1, n0 = int((y == 1).sum()), int((y == 0).sum())
        rows.append({"subject": subject, "session": session, "n_class0": n0, "n_class1": n1,
                     "mask_voxels": int(mask_flat.sum()), "mean_auc": float(acc[mask_flat].mean()),
                     "max_auc": float(acc[mask_flat].max()), "seconds": round(dt, 1)})
        print(f"[{idx}/{len(session_keys)}] {subject} ses-{session}: n=({n0},{n1}) "
              f"meanAUC={acc[mask_flat].mean():.3f} maxAUC={acc[mask_flat].max():.3f} {dt:.0f}s", flush=True)
        if args.time_one:
            print(f"\nONE-SESSION: {dt:.1f}s -> full ~= {dt*len(session_keys)/60:.0f} min for {len(session_keys)} sessions")
            return

    pd.DataFrame(rows).to_csv(args.out_dir / f"acc_summary_{args.target}.csv", index=False)

    from nilearn.glm.second_level import non_parametric_inference

    keys = sorted(acc_maps)
    finite_all = np.ones(n_vox, dtype=bool)
    for k in keys:
        finite_all &= np.isfinite(acc_maps[k]) & (acc_maps[k] != 0)
    mask_img = nib.Nifti1Image(finite_all.reshape(shape).astype(np.uint8), affine)
    print(f"group mask voxels: {int(finite_all.sum())}")

    off = {s: acc_maps[(s, ses)] for (s, ses) in keys if _medication_from_session(ses) == "OFF"}
    on = {s: acc_maps[(s, ses)] for (s, ses) in keys if _medication_from_session(ses) == "ON"}
    analyses = {
        "OFF": [(s, off[s]) for s in sorted(off)],
        "ON": [(s, on[s]) for s in sorted(on)],
        "interaction_ON_minus_OFF": [(s, on[s] - off[s] + CHANCE) for s in sorted(set(off) & set(on))],
    }

    summary_rows = []
    for label, items in analyses.items():
        if len(items) < 5:
            print(f"skip {label}: {len(items)} subjects"); continue
        maps = [m for _, m in items]
        imgs = [_to_img(m - CHANCE, finite_all, shape, affine) for m in maps]
        design = pd.DataFrame({"intercept": np.ones(len(imgs))})
        print(f"{args.target}/{label}: n={len(imgs)}, TFCE perms={args.n_perm}", flush=True)
        out = non_parametric_inference(
            imgs, design_matrix=design, second_level_contrast="intercept", mask=mask_img,
            n_perm=args.n_perm, tfce=True, two_sided_test=label.startswith("interaction"),
            n_jobs=args.n_jobs, verbose=0)
        logp = out["logp_max_tfce"]
        stem = f"{args.target}__{label}"
        logp.to_filename(str(args.out_dir / f"{stem}_logp_fwe_tfce.nii.gz"))
        out["t"].to_filename(str(args.out_dir / f"{stem}_t.nii.gz"))
        data = logp.get_fdata()
        n_sig = int((data >= -np.log10(0.05)).sum())
        min_p = float(10 ** (-np.nanmax(data))) if np.isfinite(np.nanmax(data)) else float("nan")
        grp_auc = float(np.mean([m[finite_all].mean() for m in maps])) if not label.startswith("inter") else float("nan")
        summary_rows.append({"target": args.target, "analysis": label, "n_subjects": len(imgs),
                             "group_mean_auc": grp_auc, "n_sig_voxels_fwe05": n_sig, "min_fwe_p": min_p})
        print(f"  -> mean AUC={grp_auc:.3f}, sig voxels(FWE<.05)={n_sig}, min FWE p={min_p:.4g}", flush=True)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.out_dir / f"tfce_summary_{args.target}.csv", index=False)
    print("\n=== Searchlight decoding TFCE summary ===")
    print(summary.to_string(index=False) if not summary.empty else "(none)")


if __name__ == "__main__":
    main()
