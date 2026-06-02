#!/usr/bin/env python3
"""Whole-brain, ROI-free GVS analysis via voxelwise GLM + TFCE permutation inference.

This replaces the ROI burden / projected-signal analyses with a parametric,
whole-brain approach that exploits the 2 (carrier: 75/125 Hz) x 4 (envelope:
0/10/16/20 Hz) factorial structure of the eight active GVS conditions
(Master_paper_new.pdf, lines 135-141).

For each subject x medication state we form voxelwise first-level contrast maps
from condition-mean betas:

  * gvs_main : mean(8 active) - sham        -> "any stimulation effect"
  * carrier  : mean(125 Hz) - mean(75 Hz)   -> carrier-frequency effect
  * envelope : linear trend over 0/10/16/20  -> envelope-frequency dose response

Group inference is threshold-free cluster enhancement (TFCE) with sign-flip
permutation (FWE-controlled), via nilearn.glm.second_level.non_parametric_inference.
One-sample tests are run within OFF and within ON; a paired (ON-OFF) test gives
the medication interaction.

Condition->trial mapping, GVS order parsing, and run discovery are reused from
GVS_effect.py so this stays consistent with the rest of the pipeline.

Stim-order convention (codebase): stim value v in {1..9}; v=1 = sham; v>=2 = GVS(v-1).
Paper grid (GVS1..8):  carrier = 75 Hz for GVS1-4, 125 Hz for GVS5-8;
                       envelope = [0,10,16,20] Hz cycling within each carrier.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
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
DEFAULT_OUT_DIR = ROOT / "figures" / "GVS_effects" / "glm_tfce"
DEFAULT_CACHE = DEFAULT_OUT_DIR / "condition_means_cache.npz"

# Parametric grid keyed by stim value (2..9 -> GVS1..8).
CARRIER_HZ = {2: 75, 3: 75, 4: 75, 5: 75, 6: 125, 7: 125, 8: 125, 9: 125}
ENVELOPE_HZ = {2: 0, 3: 10, 4: 16, 5: 20, 6: 0, 7: 10, 8: 16, 9: 20}
SHAM_STIM = 1
ACTIVE_STIMS = tuple(range(2, 10))


def _build_condition_means(
    beta_root: Path,
    gvs_order_path: Path,
    trials_per_condition: int,
    reference_shape: tuple[int, int, int],
    subjects: list[str] | None,
) -> tuple[dict[tuple[str, int], dict[int, np.ndarray]], int]:
    """Return {(subject, session): {stim_value: mean_beta_flat}} and n_voxels."""
    gvs_order = _load_gvs_order(gvs_order_path)
    run_specs = _discover_run_specs(beta_root, gvs_order, subjects)

    n_vox = int(np.prod(reference_shape))
    # Accumulate finite sums / counts per (subject, session, stim).
    sums: dict[tuple[str, int, int], np.ndarray] = {}
    counts: dict[tuple[str, int, int], np.ndarray] = {}

    for idx, spec in enumerate(run_specs, start=1):
        print(f"[{idx}/{len(run_specs)}] {spec.subject} ses-{spec.session} run-{spec.run}", flush=True)
        beta = np.load(spec.beta_path, mmap_mode="r")
        if tuple(beta.shape[:3]) != tuple(reference_shape):
            raise RuntimeError(f"{spec.beta_path} grid {beta.shape[:3]} != reference {reference_shape}")
        flat = np.asarray(beta).reshape(n_vox, beta.shape[-1])
        for condition_code, start, stop in _iter_condition_slices(
            spec.n_trials, spec.stim_order, int(trials_per_condition)
        ):
            stim = _stim_id_from_condition_code(condition_code)
            block = flat[:, start:stop]
            finite = np.isfinite(block)
            key = (spec.subject, int(spec.session), int(stim))
            blk_sum = np.where(finite, block, 0.0).sum(axis=1)
            blk_cnt = finite.sum(axis=1).astype(np.float64)
            if key in sums:
                sums[key] += blk_sum
                counts[key] += blk_cnt
            else:
                sums[key] = blk_sum
                counts[key] = blk_cnt

    means: dict[tuple[str, int], dict[int, np.ndarray]] = defaultdict(dict)
    for (subject, session, stim), s in sums.items():
        c = counts[(subject, session, stim)]
        with np.errstate(invalid="ignore", divide="ignore"):
            m = np.where(c > 0, s / c, np.nan).astype(np.float32)
        means[(subject, session)][stim] = m
    return means, n_vox


def _contrast_map(cond: dict[int, np.ndarray], kind: str) -> np.ndarray | None:
    """First-level contrast from one subject-session's condition means (flat voxels)."""
    if SHAM_STIM not in cond or any(s not in cond for s in ACTIVE_STIMS):
        return None
    active = np.stack([cond[s] for s in ACTIVE_STIMS], axis=0)  # (8, n_vox)
    if kind == "gvs_main":
        return np.nanmean(active, axis=0) - cond[SHAM_STIM]
    if kind == "carrier":
        hi = np.nanmean(np.stack([cond[s] for s in ACTIVE_STIMS if CARRIER_HZ[s] == 125]), axis=0)
        lo = np.nanmean(np.stack([cond[s] for s in ACTIVE_STIMS if CARRIER_HZ[s] == 75]), axis=0)
        return hi - lo
    if kind == "envelope":
        env = np.array([ENVELOPE_HZ[s] for s in ACTIVE_STIMS], dtype=np.float64)
        w = env - env.mean()
        w = w / np.sum(w**2)  # slope normalisation (scale-irrelevant for the t-stat)
        return np.tensordot(w.astype(np.float32), active, axes=(0, 0))
    raise ValueError(kind)


def _to_img(flat: np.ndarray, mask_flat: np.ndarray, shape, affine) -> nib.Nifti1Image:
    vol = np.zeros(int(np.prod(shape)), dtype=np.float32)
    vol[mask_flat] = np.nan_to_num(flat[mask_flat], nan=0.0)
    return nib.Nifti1Image(vol.reshape(shape), affine)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--beta-root", type=Path, default=DEFAULT_BETA_ROOT)
    ap.add_argument("--gvs-order", type=Path, default=DEFAULT_GVS_ORDER)
    ap.add_argument("--reference-nifti", type=Path, default=DEFAULT_REFERENCE_NIFTI)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--trials-per-condition", type=int, default=DEFAULT_TRIALS_PER_CONDITION)
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--contrasts", nargs="*", default=["gvs_main", "carrier", "envelope"])
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--quick", action="store_true", help="50 perms + intersection mask, smoke test")
    ap.add_argument("--rebuild-cache", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ref = nib.load(str(args.reference_nifti))
    shape, affine = ref.shape[:3], ref.affine
    n_perm = 50 if args.quick else args.n_perm

    # ---- condition means (cached) ----
    if args.cache.exists() and not args.rebuild_cache:
        print(f"Loading cached condition means: {args.cache}")
        npz = np.load(args.cache, allow_pickle=True)
        means = npz["means"].item()
        n_vox = int(npz["n_vox"])
    else:
        means, n_vox = _build_condition_means(
            args.beta_root, args.gvs_order, args.trials_per_condition, shape, args.subjects
        )
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.cache, means=np.array(means, dtype=object), n_vox=n_vox)
        print(f"Cached condition means -> {args.cache}")

    cells = sorted(means.keys(), key=lambda k: (k[0], k[1]))
    print(f"Subject-sessions: {len(cells)}  |  voxels(full): {n_vox}")

    # ---- analysis mask: finite in every condition of every cell, and not constant-zero ----
    all_means = [m for cond in means.values() for m in cond.values()]
    finite_all = np.ones(n_vox, dtype=bool)
    max_abs = np.zeros(n_vox, dtype=np.float32)
    for m in all_means:
        finite_all &= np.isfinite(m)
        max_abs = np.maximum(max_abs, np.abs(np.nan_to_num(m, nan=0.0)))
    mask_flat = finite_all & (max_abs > 0)
    print(f"Analysis mask voxels: {int(mask_flat.sum())}")
    mask_img = nib.Nifti1Image(mask_flat.reshape(shape).astype(np.uint8), affine)
    mask_img.to_filename(str(args.out_dir / "analysis_mask.nii.gz"))

    # ---- first-level contrast maps per cell ----
    from nilearn.glm.second_level import non_parametric_inference

    summary_rows = []
    for contrast in args.contrasts:
        # build per-cell maps
        per_cell = {c: _contrast_map(means[c], contrast) for c in cells}
        per_cell = {c: v for c, v in per_cell.items() if v is not None}

        groups = {
            "OFF": [(s, ses) for (s, ses) in per_cell if _medication_from_session(ses) == "OFF"],
            "ON": [(s, ses) for (s, ses) in per_cell if _medication_from_session(ses) == "ON"],
        }
        # paired ON-OFF (subjects with both)
        off_subj = {s for (s, ses) in groups["OFF"]}
        on_subj = {s for (s, ses) in groups["ON"]}
        paired_subj = sorted(off_subj & on_subj)

        analyses: dict[str, list[np.ndarray]] = {}
        for med in ("OFF", "ON"):
            analyses[med] = [per_cell[c] for c in groups[med]]
        if paired_subj:
            off_map = {s: per_cell[(s, ses)] for (s, ses) in groups["OFF"]}
            on_map = {s: per_cell[(s, ses)] for (s, ses) in groups["ON"]}
            analyses["interaction_ON_minus_OFF"] = [on_map[s] - off_map[s] for s in paired_subj]

        for label, flat_maps in analyses.items():
            if len(flat_maps) < 5:
                print(f"  skip {contrast}/{label}: only {len(flat_maps)} subjects")
                continue
            imgs = [_to_img(fm, mask_flat, shape, affine) for fm in flat_maps]
            design = pd.DataFrame({"intercept": np.ones(len(imgs))})
            print(f"  {contrast} / {label}: n={len(imgs)}, TFCE perms={n_perm}")
            out = non_parametric_inference(
                imgs,
                design_matrix=design,
                second_level_contrast="intercept",
                mask=mask_img,
                n_perm=n_perm,
                tfce=True,
                two_sided_test=True,
                n_jobs=-1,
                verbose=0,
            )
            logp = out["logp_max_tfce"]
            stem = f"{contrast}__{label}"
            logp.to_filename(str(args.out_dir / f"{stem}_logp_fwe_tfce.nii.gz"))
            out["t"].to_filename(str(args.out_dir / f"{stem}_t.nii.gz"))
            data = logp.get_fdata()
            n_sig = int((data >= -np.log10(0.05)).sum())
            min_p = float(10 ** (-np.nanmax(data))) if np.isfinite(np.nanmax(data)) else float("nan")
            summary_rows.append(
                {"contrast": contrast, "analysis": label, "n_subjects": len(imgs),
                 "n_sig_voxels_fwe05": n_sig, "min_fwe_p": min_p}
            )
            print(f"    -> sig voxels (FWE<.05): {n_sig}, min FWE p={min_p:.4g}")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.out_dir / "tfce_summary.csv", index=False)
    print("\n=== TFCE summary ===")
    print(summary.to_string(index=False) if not summary.empty else "(no analyses run)")
    print(f"\nOutputs in {args.out_dir}")


if __name__ == "__main__":
    main()
