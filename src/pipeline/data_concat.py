#!/usr/bin/env python3
"""
Concatenate preprocessed beta/BOLD data across all subjects/sessions/runs.

This script builds a group voxel mask based on cleaned_beta_volume files
(voxels with trial coverage above a configured threshold across all runs,
after trial_keep masking when available). It then concatenates trials across
all inputs (all sessions/runs) and writes group-level outputs that can be
consumed by group_analysis/obj_param.py (via SUB/SES and FMRI_OPT_DATA_DIR).

Note: To avoid a massive 4D volume, cleaned_beta_volume is stored as a 2D
array (n_union_voxels x total_trials). Use common_mask_* to rebuild volumes
if needed.
"""

import argparse
import csv
import os
import re
from glob import glob

import numpy as np
from numpy.lib.format import open_memmap


FILE_RE = re.compile(r"cleaned_beta_volume_(sub-[^_]+)_ses-(\d+)_run-(\d+)\.npy$")


def _safe_symlink(src, dest):
    if os.path.islink(dest):
        if os.readlink(dest) == src:
            return
        os.unlink(dest)
    elif os.path.exists(dest):
        raise ValueError(f"{dest} exists and is not a symlink; remove it to continue.")
    os.symlink(src, dest)


def _discover_inputs(in_root):
    pattern = os.path.join(in_root, "sub-*", "cleaned_beta_volume_sub-*_ses-*_run-*.npy")
    files = sorted(glob(pattern))
    entries = []
    for path in files:
        name = os.path.basename(path)
        match = FILE_RE.match(name)
        if not match:
            continue
        sub_tag = match.group(1)
        ses = int(match.group(2))
        run = int(match.group(3))
        tag = f"{sub_tag}_ses-{ses}_run-{run}"
        base_dir = os.path.dirname(path)
        entries.append(
            {
                "sub_tag": sub_tag,
                "ses": ses,
                "run": run,
                "tag": tag,
                "cleaned_beta": path,
                "active_bold": os.path.join(base_dir, f"active_bold_{tag}.npy.npy"),
                "active_coords": os.path.join(base_dir, f"active_coords_{tag}.npy"),
                "active_flat": os.path.join(base_dir, f"active_flat_indices__{tag}.npy"),
                "beta_filter": os.path.join(base_dir, f"beta_volume_filter_{tag}.npy.npy"),
                "mask_all_nan": os.path.join(base_dir, f"mask_all_nan_{tag}.npy"),
                "nan_mask_flat": os.path.join(base_dir, f"nan_mask_flat_{tag}.npy"),
            }
        )
    return entries


def _validate_inputs(entries):
    required_keys = [
        "cleaned_beta",
        "active_bold",
        "active_coords",
        "active_flat",
        "beta_filter",
        "mask_all_nan",
        "nan_mask_flat",
    ]
    missing = []
    for entry in entries:
        for key in required_keys:
            path = entry[key]
            if not os.path.exists(path):
                missing.append(path)
    if missing:
        msg = ["Missing required inputs:"]
        msg.extend([f"  - {path}" for path in missing])
        raise FileNotFoundError("\n".join(msg))


def _build_union_mask(entries, trial_keep_root=None, min_trial_coverage=0.70):
    coverage_counts_flat = None
    volume_shape = None
    trial_len = None
    total_trials = 0
    missing_trial_keep = []
    for entry in entries:
        vol = np.load(entry["cleaned_beta"], mmap_mode="r")
        if volume_shape is None:
            volume_shape = vol.shape[:3]
        elif vol.shape[:3] != volume_shape:
            raise ValueError(
                f"Volume shape mismatch: {entry['cleaned_beta']} has {vol.shape[:3]} "
                f"but expected {volume_shape}."
            )
        entry["n_trials"] = int(vol.shape[3])

        active_bold = np.load(entry["active_bold"], mmap_mode="r")
        if active_bold.shape[1] != entry["n_trials"]:
            raise ValueError(
                f"Trial mismatch between cleaned_beta and active_bold in {entry['tag']}: "
                f"{entry['n_trials']} vs {active_bold.shape[1]}"
            )
        if trial_len is None:
            trial_len = int(active_bold.shape[2])
        elif trial_len != int(active_bold.shape[2]):
            raise ValueError(
                f"Trial length mismatch: {entry['active_bold']} has {active_bold.shape[2]} "
                f"but expected {trial_len}."
            )
        n_trials = entry["n_trials"]
        total_trials += n_trials
        trial_keep, trial_keep_path = _load_trial_keep_mask(trial_keep_root, entry, n_trials)
        entry["trial_keep"] = trial_keep
        entry["trial_keep_path"] = trial_keep_path
        if trial_keep is None and trial_keep_path is not None and trial_keep_root is not None:
            missing_trial_keep.append((entry["tag"], trial_keep_path))

        if trial_keep is not None:
            valid_count = np.count_nonzero(np.isfinite(vol[..., trial_keep]), axis=-1)
        else:
            valid_count = np.count_nonzero(np.isfinite(vol), axis=-1)
        flat_count = valid_count.reshape(-1).astype(np.uint32, copy=False)
        if coverage_counts_flat is None:
            coverage_counts_flat = np.zeros(flat_count.size, dtype=np.uint32)
        coverage_counts_flat += flat_count

    if coverage_counts_flat is None:
        raise ValueError("No cleaned_beta_volume files found to build a union mask.")
    if total_trials <= 0:
        raise ValueError("No trials found across inputs.")
    coverage_flat = coverage_counts_flat.astype(np.float32) / float(total_trials)
    union_mask = coverage_flat.reshape(volume_shape) > float(min_trial_coverage)
    if not np.any(union_mask):
        raise ValueError("Union mask is empty after aggregation.")
    return union_mask, volume_shape, trial_len, coverage_flat, missing_trial_keep


def _resolve_trial_keep_root(path):
    if path is None:
        return None
    root = os.path.expanduser(path)
    if os.path.isdir(root):
        return root
    print(f"Warning: trial-keep root not found: {root}. Trial keep masking disabled.", flush=True)
    return None


def _load_trial_keep_mask(trial_keep_root, entry, n_trials):
    if trial_keep_root is None:
        return None, None
    path = os.path.join(
        trial_keep_root,
        entry["sub_tag"],
        f"ses-{entry['ses']}",
        "GLMOutputs-mni-std",
        f"trial_keep_run{entry['run']}.npy",
    )
    if not os.path.exists(path):
        return None, path
    keep = np.load(path)
    keep = np.asarray(keep, dtype=bool)
    if keep.ndim != 1:
        raise ValueError(f"trial_keep must be 1D, got shape {keep.shape} for {path}")
    if keep.size != n_trials:
        raise ValueError(
            f"trial_keep length mismatch for {entry['tag']}: {keep.size} vs {n_trials}"
        )
    return keep, path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-root", default="/Data/zahra/results_beta_preprocessed")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--group-sub", default="99")
    parser.add_argument("--obj-ses", default="1")
    parser.add_argument("--obj-run", default="2")
    parser.add_argument("--tag-ses", default="all")
    parser.add_argument("--tag-run", default="all")
    parser.add_argument("--group-label", default="group")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--trial-keep-root", default="/Data/zahra/results_glm")
    parser.add_argument("--min-trial-coverage", type=float, default=0.70)
    parser.add_argument(
        "--keep-masked-trials",
        action="store_true",
        help="Keep trial_keep-masked trials as all-NaN columns in group outputs.",
    )
    args = parser.parse_args()

    in_root = os.path.expanduser(args.in_root)
    if not os.path.isdir(in_root):
        alt_root = "/Data/zahra/data/results_beta_preprocessed"
        if os.path.isdir(alt_root):
            in_root = alt_root
        else:
            raise FileNotFoundError(f"Input root not found: {args.in_root}")

    out_dir = args.out_dir or os.path.join(in_root, "group_concat")
    out_dir = os.path.expanduser(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    entries = _discover_inputs(in_root)
    if not entries:
        raise FileNotFoundError(f"No cleaned_beta_volume files found under {in_root}")
    _validate_inputs(entries)
    if not (0.0 <= float(args.min_trial_coverage) < 1.0):
        raise ValueError(f"--min-trial-coverage must be in [0, 1), got {args.min_trial_coverage}.")

    entries.sort(key=lambda e: (e["sub_tag"], e["ses"], e["run"]))
    trial_keep_root = _resolve_trial_keep_root(args.trial_keep_root)
    union_mask, volume_shape, trial_len, coverage_flat, missing_trial_keep = _build_union_mask(
        entries, trial_keep_root=trial_keep_root, min_trial_coverage=args.min_trial_coverage
    )
    union_flat = np.flatnonzero(union_mask)
    n_union = int(union_flat.size)
    total_trials = int(sum(entry["n_trials"] for entry in entries))
    drop_masked_trials = not bool(args.keep_masked_trials)

    source_trial_keep_parts = []
    for entry in entries:
        trial_keep = entry.get("trial_keep")
        if trial_keep is None:
            source_keep = np.ones(entry["n_trials"], dtype=bool)
        else:
            source_keep = np.asarray(trial_keep, dtype=bool)
        entry["source_trial_keep"] = source_keep
        source_trial_keep_parts.append(source_keep)
    source_trial_keep_concat = np.concatenate(source_trial_keep_parts, axis=0)
    trial_kept_indices = np.flatnonzero(source_trial_keep_concat).astype(np.int64, copy=False)
    trial_removed_indices = np.flatnonzero(~source_trial_keep_concat).astype(np.int64, copy=False)
    output_source_indices = (
        trial_kept_indices if drop_masked_trials else np.arange(total_trials, dtype=np.int64)
    )
    output_trials = int(output_source_indices.size)
    retained_coverage = coverage_flat[union_flat]
    if missing_trial_keep:
        print(
            f"Warning: missing trial_keep for {len(missing_trial_keep)} run(s); using all trials for those runs.",
            flush=True,
        )

    print(f"Found {len(entries)} runs across subjects.", flush=True)
    print(
        f"Voxel trial coverage filter: > {args.min_trial_coverage:.2f} "
        f"(computed across {total_trials} concatenated trials).",
        flush=True,
    )
    print(f"Union voxel count: {n_union}", flush=True)
    print(
        f"Retained voxel coverage range: [{retained_coverage.min():.3f}, {retained_coverage.max():.3f}]",
        flush=True,
    )
    print(f"Total source trials (all runs): {total_trials}", flush=True)
    if drop_masked_trials:
        print(
            f"Dropping trial_keep-masked trials: removed={trial_removed_indices.size}, kept={output_trials}",
            flush=True,
        )
    else:
        print(
            f"Keeping masked trials as NaN columns: masked={trial_removed_indices.size}, output_trials={output_trials}",
            flush=True,
        )

    group_label = str(args.group_label)
    tag_ses = str(args.tag_ses)
    tag_run = str(args.tag_run)
    obj_ses = str(args.obj_ses)
    obj_run = str(args.obj_run)
    group_sub = str(args.group_sub)

    tag_hyphen = group_label
    tag_obj = f"sub0{group_sub}_ses{obj_ses}_run{obj_run}"

    common_mask_path = os.path.join(out_dir, f"common_mask_{tag_hyphen}.npy")
    common_mask_flat_path = os.path.join(out_dir, f"common_mask_flat_{tag_hyphen}.npy")
    np.save(common_mask_path, union_mask)
    np.save(common_mask_flat_path, union_mask.ravel())
    np.save(os.path.join(out_dir, f"trial_keep_concat_{tag_hyphen}.npy"), source_trial_keep_concat)
    np.save(os.path.join(out_dir, f"trial_kept_indices_{tag_hyphen}.npy"), trial_kept_indices)
    np.save(os.path.join(out_dir, f"trial_removed_indices_{tag_hyphen}.npy"), trial_removed_indices)
    np.save(os.path.join(out_dir, f"trial_output_source_indices_{tag_hyphen}.npy"), output_source_indices)

    active_coords = np.unravel_index(union_flat, volume_shape)
    active_flat_indices = union_flat
    mask_all_nan = ~union_mask.ravel()

    active_coords_path = os.path.join(out_dir, f"active_coords_{tag_hyphen}.npy")
    active_flat_path = os.path.join(out_dir, f"active_flat_indices__{tag_hyphen}.npy")
    mask_all_nan_path = os.path.join(out_dir, f"mask_all_nan_{tag_hyphen}.npy")
    nan_mask_flat_path = os.path.join(out_dir, f"nan_mask_flat_{tag_hyphen}.npy")
    np.save(active_coords_path, active_coords)
    np.save(active_flat_path, active_flat_indices)
    np.save(mask_all_nan_path, mask_all_nan)
    np.save(nan_mask_flat_path, mask_all_nan)

    beta_dtype = np.dtype(args.dtype)
    if not np.issubdtype(beta_dtype, np.floating):
        raise ValueError(f"beta dtype must be floating to support NaN masking, got {beta_dtype}")
    beta_path = os.path.join(out_dir, f"beta_volume_filter_{tag_hyphen}.npy")
    beta_mm = open_memmap(beta_path, mode="w+", dtype=beta_dtype, shape=(n_union, output_trials))

    cleaned_path = os.path.join(out_dir, f"cleaned_beta_volume_{tag_hyphen}.npy")
    _safe_symlink(beta_path, cleaned_path)
    cleaned_mm = beta_mm

    active_dtype = np.load(entries[0]["active_bold"], mmap_mode="r").dtype
    if not np.issubdtype(active_dtype, np.floating):
        raise ValueError(
            f"active_bold dtype must be floating to support NaN masking, got {active_dtype}"
        )
    active_bold_path = os.path.join(out_dir, f"active_bold_{tag_hyphen}.npy")
    active_mm = open_memmap(
        active_bold_path,
        mode="w+",
        dtype=active_dtype,
        shape=(n_union, output_trials, trial_len),
    )

    manifest_path = os.path.join(out_dir, f"concat_manifest_{tag_hyphen}.tsv")
    with open(manifest_path, "w", newline="") as manifest_file:
        writer = csv.writer(manifest_file, delimiter="\t")
        writer.writerow(
            [
                "offset_start",
                "offset_end",
                "source_offset_start",
                "source_offset_end",
                "sub_tag",
                "ses",
                "run",
                "n_trials",
                "n_trials_source",
                "cleaned_beta",
                "trial_keep_path",
                "trial_keep_kept",
            ]
        )

        offset = 0
        source_offset = 0
        for entry in entries:
            vol = np.load(entry["cleaned_beta"], mmap_mode="r")
            n_trials = entry["n_trials"]
            flat_view = vol.reshape(-1, n_trials)
            trial_keep = entry.get("trial_keep")
            source_keep = entry["source_trial_keep"]
            write_keep = source_keep if drop_masked_trials else np.ones(n_trials, dtype=bool)
            n_trials_kept = int(np.count_nonzero(write_keep))

            beta_chunk = flat_view[union_flat][:, write_keep]
            trial_keep_path = entry.get("trial_keep_path")
            if trial_keep is not None and not drop_masked_trials:
                beta_chunk[:, ~trial_keep] = np.nan
            beta_mm[:, offset : offset + n_trials_kept] = beta_chunk
            if cleaned_mm is not None and cleaned_mm is not beta_mm:
                cleaned_mm[:, offset : offset + n_trials_kept] = beta_chunk

            coords = np.load(entry["active_coords"], allow_pickle=True)
            flat_active = np.ravel_multi_index(coords, volume_shape)
            sorter = np.argsort(flat_active)
            sorted_flat = flat_active[sorter]
            positions = np.searchsorted(sorted_flat, union_flat)
            valid = positions < sorted_flat.size
            present = np.zeros(union_flat.size, dtype=bool)
            if np.any(valid):
                present[valid] = sorted_flat[positions[valid]] == union_flat[valid]
            idx = np.full(union_flat.size, -1, dtype=int)
            if np.any(present):
                idx[present] = sorter[positions[present]]

            active_bold = np.load(entry["active_bold"], mmap_mode="r")
            active_slice = active_mm[:, offset : offset + n_trials_kept, :]
            active_slice[:] = np.nan
            if np.any(present):
                active_slice[present, :, :] = active_bold[idx[present]][:, write_keep, :]
            if trial_keep is not None and not drop_masked_trials:
                active_slice[:, ~trial_keep, :] = np.nan

            writer.writerow(
                [
                    offset,
                    offset + n_trials_kept,
                    source_offset,
                    source_offset + n_trials,
                    entry["sub_tag"],
                    entry["ses"],
                    entry["run"],
                    n_trials_kept,
                    n_trials,
                    entry["cleaned_beta"],
                    trial_keep_path or "",
                    int(np.count_nonzero(source_keep)),
                ]
            )
            offset += n_trials_kept
            source_offset += n_trials

        if offset != output_trials:
            raise ValueError(
                f"Output trial count mismatch while writing: wrote {offset}, expected {output_trials}."
            )
        if source_offset != total_trials:
            raise ValueError(
                f"Source trial count mismatch while writing: consumed {source_offset}, expected {total_trials}."
            )

    beta_mm.flush()
    active_mm.flush()

    print(f"Wrote outputs to: {out_dir}", flush=True)
    print(f"Group tag (output): {tag_hyphen}", flush=True)


if __name__ == "__main__":
    main()
