# Group fMRI optimization workflow

# =============================================================================
# Imports and runtime paths
# =============================================================================
import argparse
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from glob import glob
from collections import defaultdict
from datetime import datetime
from os.path import join

from empca.empca import empca
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors, ticker
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import plotting, image, datasets
from scipy import sparse
from scipy import ndimage
from scipy.linalg import eigh
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.stats import t as student_t
import pingouin as pg

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
DEFAULT_RESULTS_DIR = join(PROJECT_ROOT, "results")
RESULTS_DIR = os.path.abspath(os.path.expanduser(os.environ.get("FMRI_RESULTS_DIR", DEFAULT_RESULTS_DIR)))
os.makedirs(RESULTS_DIR, exist_ok=True)
DEFAULT_DATA_DIR = join(PROJECT_ROOT, "data")

def _is_path_within_dir(path, directory):
    return os.path.commonpath([os.path.abspath(path), os.path.abspath(directory)]) == os.path.abspath(directory)

def _result_path(path, for_write=False, allow_absolute=False):
    expanded = os.path.expanduser(str(path))
    if os.path.isabs(expanded):
        candidate = os.path.abspath(expanded)
        if allow_absolute or _is_path_within_dir(candidate, RESULTS_DIR):
            resolved = candidate
        else:
            resolved = join(RESULTS_DIR, os.path.basename(candidate))
    else:
        resolved = join(RESULTS_DIR, expanded)

    resolved = os.path.abspath(resolved)
    if not allow_absolute and not _is_path_within_dir(resolved, RESULTS_DIR):
        resolved = os.path.abspath(join(RESULTS_DIR, os.path.basename(resolved)))

    if for_write:
        os.makedirs(os.path.dirname(resolved), exist_ok=True)
    return resolved

LOG_FILE = _result_path("run_metrics_log.jsonl", for_write=True)


# =============================================================================
# Logging and summaries
# =============================================================================

def _array_summary(array):
    flat = np.asarray(array, dtype=np.float64).ravel()
    finite = flat[np.isfinite(flat)]
    if finite.size == 0:
        return {"min": np.nan, "max": np.nan, "mean": np.nan, "var": np.nan}
    return {"min": float(np.min(finite)), "max": float(np.max(finite)), "mean": float(np.mean(finite)), "var": float(np.var(finite))}

def _matrix_norm_summary(matrix):
    mat = np.asarray(matrix, dtype=np.float64)
    finite = np.isfinite(mat)
    if not finite.any():
        return {"fro": np.nan, "mean_abs": np.nan, "max_abs": np.nan}
    safe_mat = np.where(finite, mat, 0.0)
    fro_norm = float(np.linalg.norm(safe_mat, ord="fro"))
    abs_vals = np.abs(safe_mat[finite])
    return {"fro": fro_norm, "mean_abs": float(np.mean(abs_vals)), "max_abs": float(np.max(abs_vals))}

def _sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.floating, np.integer, float, int)):
        value = float(obj)
        if not math.isfinite(value):
            return None
        return value
    if isinstance(obj, str) or obj is None:
        return obj
    return str(obj)

def _append_run_log(entry, path=LOG_FILE):
    payload = dict(entry)
    payload["timestamp"] = datetime.utcnow().isoformat() + "Z"
    payload = _sanitize_for_json(payload)
    path = _result_path(path, for_write=True)
    with open(path, "a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(payload) + "\n")


# =============================================================================
# Anatomy, masks, and input discovery
# =============================================================================

def mask_anatomical_image(anat_img, mask_flat, volume_shape):
    anat_data = anat_img.get_fdata()
    mask_flat = mask_flat.ravel()
    mask_3d = mask_flat.reshape(volume_shape)
    anat_data = np.where(mask_3d, np.nan, anat_data)
    return anat_data

def compute_distance_weighted_adjacency(coords, affine, radius_mm=3.0, sigma_mm=None):
    if sigma_mm is None:
        sigma_mm = radius_mm / 2.0
    coord_array = np.column_stack(coords).astype(np.float32, copy=False)
    coords_mm = nib.affines.apply_affine(affine, coord_array)
    tree = cKDTree(coords_mm)
    dist_matrix = tree.sparse_distance_matrix(tree, max_distance=radius_mm, output_type="coo_matrix")
    rows, cols, data = dist_matrix.row, dist_matrix.col, dist_matrix.data
    if data.size:
        off_diag = rows != cols
        rows, cols, data = rows[off_diag], cols[off_diag], data[off_diag]
    weights = np.exp(-(data ** 2) / (sigma_mm ** 2))
    adjacency = sparse.csr_matrix((weights, (rows, cols)), shape=(coords_mm.shape[0], coords_mm.shape[0]))
    if adjacency.nnz:
        adjacency = adjacency.maximum(adjacency.T)
    degrees = np.asarray(adjacency.sum(axis=1)).ravel()
    degree_matrix = sparse.diags(degrees).tocsr()  # CSR keeps diagonal structure but allows slicing.
    return adjacency, degree_matrix
# %%
# Output labels (SUB/SES env vars are only used for naming outputs)
sub = os.environ.get("SUB", "9")
ses = int(os.environ.get("SES", "1"))
sub_dir = str(sub).zfill(2)
ses_dir = str(ses).zfill(2)
num_trials = 180  # default; updated after loading data
trial_len = 9
behavior_source_column = int(os.environ.get("FMRI_BEHAVIOR_SOURCE_COLUMN", "1"))
behave_indice = 0  # the behavior loader collapses inputs to one target/RT column

# Save a quick visualization of the mean beta activation (averaged over trials) as an interactive HTML.


data_base = os.path.abspath(os.path.expanduser(os.environ.get("FMRI_DATA_DIR", DEFAULT_DATA_DIR)))

def _first_existing_path(*candidates):
    seen = set()
    for candidate in candidates:
        if not candidate:
            continue
        expanded = os.path.abspath(os.path.expanduser(str(candidate)))
        if expanded in seen:
            continue
        seen.add(expanded)
        if os.path.exists(expanded):
            return expanded
    return None

def _default_behavior_root(data_root):
    behavior_candidates = [join(data_root, "behaviour"), join(data_root, "behavior")]
    existing = _first_existing_path(*behavior_candidates)
    return existing if existing else behavior_candidates[0]

def _default_trial_keep_roots(data_root):
    env_override = os.environ.get("FMRI_TRIAL_KEEP_DIR", "").strip()
    candidates = [env_override, join(data_root, "glm_output")]
    deduped = []
    seen = set()
    for candidate in candidates:
        if not candidate:
            continue
        normalized = os.path.abspath(os.path.expanduser(candidate))
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped if deduped else [os.path.abspath(os.path.expanduser(join(data_root, "glm_output")))]

TRIAL_KEEP_FILES_DIRS = _default_trial_keep_roots(data_base)
TRIAL_KEEP_FILES_DIR = TRIAL_KEEP_FILES_DIRS[0]

def _trial_keep_data_candidates(sub_tag, ses_id, run_id, declared_path):
    path = str(declared_path or "").strip()
    sub_tag = str(sub_tag)
    ses_id = int(ses_id)
    run_id = int(run_id)
    has_declared_path = bool(path) and path.lower() != "nan"
    keep_basename = os.path.basename(os.path.expanduser(path)) if has_declared_path else f"trial_keep_run{run_id}.npy"
    candidates = []
    for trial_keep_root in TRIAL_KEEP_FILES_DIRS:
        candidates.extend([
            join(trial_keep_root, sub_tag, f"ses-{ses_id}", "GLMOutputs-mni-std", keep_basename),
            join(trial_keep_root, sub_tag, f"ses-{ses_id}", keep_basename),
            join(trial_keep_root, sub_tag, keep_basename),
        ])
    deduped = []
    seen = set()
    for candidate in candidates:
        if not candidate:
            continue
        normalized = os.path.abspath(os.path.expanduser(candidate))
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped

def _resolve_trial_keep_path(sub_tag, ses_id, run_id, declared_path):
    return _first_existing_path(*_trial_keep_data_candidates(sub_tag, ses_id, run_id, declared_path))

def _behavior_data_candidates(behavior_root, behavior_filename, data_root):
    candidates = [join(behavior_root, behavior_filename), join(data_root, "behaviour", behavior_filename), join(data_root, "behavior", behavior_filename)]
    deduped = []
    seen = set()
    for candidate in candidates:
        normalized = os.path.abspath(os.path.expanduser(candidate))
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped

def _resolve_behavior_file(behavior_root, behavior_filename, data_root):
    return _first_existing_path(*_behavior_data_candidates(behavior_root, behavior_filename, data_root))

def _resolve_existing_path(*candidates, glob_candidates=()):
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    for pattern in glob_candidates:
        matches = sorted(glob(pattern))
        if matches:
            return matches[0]
    raise RuntimeError("None of the candidate paths exist:\n" + "\n".join([f"- {c}" for c in candidates if c] + [f"- {p} (glob)" for p in glob_candidates if p]))

def _resolve_numpy_path(*candidates, glob_candidates=()):
    return _resolve_existing_path(*candidates, glob_candidates=glob_candidates)

def _group_concat_required_files(group_label):
    return [f"nan_mask_flat_{group_label}.npy", f"active_coords_{group_label}.npy", f"active_flat_indices__{group_label}.npy",
            f"beta_volume_filter_{group_label}.npy", f"active_bold_{group_label}.npy"]

def _is_group_concat_dir(path, group_label):
    if not path or not os.path.isdir(path):
        return False
    required = _group_concat_required_files(group_label)
    return all(os.path.exists(join(path, name)) for name in required)

def _resolve_group_concat_dir(data_root, group_label):
    explicit_group_dir = os.environ.get("FMRI_GROUP_CONCAT_DIR", "").strip()
    legacy_group_dir = os.environ.get("FMRI_OPT_DATA_DIR", "").strip()
    candidates = [
        explicit_group_dir,
        legacy_group_dir,
        join(data_root, "data_concatenated"),
        join(data_root, "group_concat"),
        data_root,
    ]

    seen = set()
    for candidate in candidates:
        expanded = os.path.abspath(os.path.expanduser(candidate))
        if expanded in seen:
            continue
        seen.add(expanded)
        if _is_group_concat_dir(expanded, group_label):
            return expanded
    return None

def _normalize_active_coords(active_coords):
    coords = np.asarray(active_coords)
    if coords.dtype == object and coords.size == 3:
        axes = [np.asarray(axis, dtype=np.int64).ravel() for axis in coords]
        return np.vstack(axes)
    if coords.ndim == 2 and coords.shape[0] == 3:
        return coords.astype(np.int64, copy=False)
    if coords.ndim == 2 and coords.shape[1] == 3:
        return coords.T.astype(np.int64, copy=False)
    raise RuntimeError(f"Unexpected active_coords shape: {coords.shape}")

def _extract_subject_digits(sub_tag):
    match = re.search(r"(\d+)$", str(sub_tag))
    if not match:
        raise RuntimeError(f"Could not parse subject digits from '{sub_tag}'.")
    return match.group(1)

def _load_manifest_segments(manifest_path, expected_trials):
    if not manifest_path or not os.path.exists(manifest_path):
        return [expected_trials], []
    manifest = pd.read_csv(manifest_path, sep="\t").sort_values("offset_start")
    segment_lengths = []
    transition_boundaries = []
    cumulative = 0
    for row in manifest.itertuples(index=False):
        start = int(row.offset_start)
        end = int(row.offset_end)
        length = max(0, end - start)
        if length == 0:
            continue
        segment_lengths.append(length)
        cumulative += length
        if cumulative < expected_trials:
            transition_boundaries.append(cumulative)
    if cumulative != expected_trials:
        print(f"WARNING: Manifest trials ({cumulative}) do not match expected_trials ({expected_trials}).", flush=True)
        if cumulative < expected_trials:
            segment_lengths.append(expected_trials - cumulative)
    transition_boundaries = sorted({int(boundary) for boundary in transition_boundaries if 0 < int(boundary) < expected_trials})
    return segment_lengths, transition_boundaries

def _load_session_labels(manifest_path, expected_trials):
    if not manifest_path or not os.path.exists(manifest_path):
        return np.zeros(int(expected_trials), dtype=np.int32), [{"session_id": 0, "sub_tag": "unknown", "ses": 0, "n_trials": int(expected_trials)}]

    manifest = pd.read_csv(manifest_path, sep="\t").sort_values("offset_start")
    session_labels = np.full(int(expected_trials), -1, dtype=np.int32)
    session_lookup = {}
    session_metadata = []

    for row in manifest.itertuples(index=False):
        start = max(0, int(row.offset_start))
        end = min(int(expected_trials), int(row.offset_end))
        if end <= start:
            continue
        session_key = (str(row.sub_tag), int(row.ses))
        if session_key not in session_lookup:
            session_id = len(session_metadata)
            session_lookup[session_key] = session_id
            session_metadata.append({"session_id": session_id, "sub_tag": session_key[0], "ses": session_key[1], "n_trials": 0})
        session_id = session_lookup[session_key]
        session_labels[start:end] = session_id
        session_metadata[session_id]["n_trials"] += end - start

    unlabeled_mask = session_labels < 0
    if np.any(unlabeled_mask):
        fallback_id = len(session_metadata)
        session_labels[unlabeled_mask] = fallback_id
        session_metadata.append({"session_id": fallback_id, "sub_tag": "unlabeled", "ses": -1, "n_trials": int(np.count_nonzero(unlabeled_mask))})
        print(f"WARNING: Assigned {int(np.count_nonzero(unlabeled_mask))} unlabeled trials to fallback session.", flush=True)

    return session_labels, session_metadata

def _build_block_boundaries(total_trials, block_size):
    total_trials = int(total_trials)
    block_size = int(block_size)
    if total_trials <= 0 or block_size <= 0:
        return []
    return list(range(block_size, total_trials, block_size))

def _select_behavior_target_column(run_behavior, sub_tag, ses_id, run_id):
    run_behavior = np.asarray(run_behavior, dtype=np.float64)
    if run_behavior.ndim == 1:
        return run_behavior[:, None]
    if run_behavior.shape[1] == 1:
        return run_behavior
    return run_behavior[:, [behavior_source_column]]

def _load_group_behavior_matrix(group_base_path, group_label, imaging_trials):
    manifest_path = _resolve_existing_path(join(group_base_path, f"concat_manifest_{group_label}.tsv"))
    behavior_root = _default_behavior_root(data_base)
    manifest = pd.read_csv(manifest_path, sep="\t").sort_values("offset_start")
    concat_trial_keep = None
    concat_trial_keep_path = join(group_base_path, f"trial_keep_concat_{group_label}.npy")
    has_source_offsets = {"source_offset_start", "source_offset_end"}.issubset(manifest.columns)
    if has_source_offsets and os.path.exists(concat_trial_keep_path):
        concat_trial_keep = np.asarray(np.load(concat_trial_keep_path), dtype=bool).ravel()
        print(
            f"INFO: Using stored concatenated trial_keep mask for behavior alignment: {concat_trial_keep_path}",
            flush=True,
        )
    behavior_chunks = []
    missing_behavior = []

    for row in manifest.itertuples(index=False):
        sub_digits = _extract_subject_digits(row.sub_tag)
        ses_id = int(row.ses)
        run_id = int(row.run)
        run_trials = int(row.n_trials)
        behavior_filename = f"PSPD{sub_digits}_ses_{ses_id}_run_{run_id}.npy"
        behavior_path = _resolve_behavior_file(behavior_root, behavior_filename, data_base)
        if not behavior_path:
            candidates = _behavior_data_candidates(behavior_root, behavior_filename, data_base)
            missing_behavior.append((row.sub_tag, ses_id, run_id, candidates[0] if candidates else behavior_filename))
            continue

        run_behavior_raw = np.asarray(np.load(behavior_path), dtype=np.float64)

        behavior_trial_count = run_behavior_raw.shape[0]
        if concat_trial_keep is not None:
            source_start = int(row.source_offset_start)
            source_end = int(row.source_offset_end)
            trial_keep = np.asarray(concat_trial_keep[source_start:source_end], dtype=bool).ravel()
            if trial_keep.size != behavior_trial_count:
                raise RuntimeError(
                    f"Stored concatenated trial_keep length mismatch for {row.sub_tag} ses-{ses_id} run-{run_id}: "
                    f"mask has {trial_keep.size} entries but behavior has {behavior_trial_count} trials."
                )
        else:
            trial_keep_path = str(getattr(row, "trial_keep_path", "") or "").strip()
            resolved_trial_keep = _resolve_trial_keep_path(row.sub_tag, ses_id, run_id, trial_keep_path)
            if not resolved_trial_keep:
                tk_candidates = _trial_keep_data_candidates(row.sub_tag, ses_id, run_id, trial_keep_path)
                tk_hint = "\n".join([f"- {candidate}" for candidate in tk_candidates]) if tk_candidates else f"- {trial_keep_path}"
                raise RuntimeError(
                    f"Missing trial_keep file for {row.sub_tag} ses-{ses_id} run-{run_id}. "
                    f"Tried:\n{tk_hint}"
                )

            trial_keep = np.asarray(np.load(resolved_trial_keep), dtype=bool).ravel()
            if trial_keep.size != behavior_trial_count:
                raise RuntimeError(
                    f"trial_keep length mismatch for {row.sub_tag} ses-{ses_id} run-{run_id}: "
                    f"mask has {trial_keep.size} entries but behavior has {behavior_trial_count} trials."
                )

        run_behavior = _select_behavior_target_column(run_behavior_raw, row.sub_tag, ses_id, run_id)
        run_behavior = run_behavior[trial_keep, :]
        kept_trials = int(run_behavior.shape[0])
        if kept_trials != run_trials:
            print(
                f"WARNING: Manifest n_trials={run_trials} differs from trial_keep kept trials={kept_trials} "
                f"for {row.sub_tag} ses-{ses_id} run-{run_id}; using trial_keep count.",
                flush=True,
            )
        behavior_chunks.append(run_behavior)

    if missing_behavior:
        lines = ["Some required input files are missing in data/. Please add these files:"]
        lines.append("Missing behavior files:")
        lines.extend([f"- {path} (manifest row: {sub_tag} ses-{ses_id} run-{run_id})" for sub_tag, ses_id, run_id, path in missing_behavior])
        raise RuntimeError("\n".join(lines))

    if not behavior_chunks:
        raise RuntimeError("Manifest is empty; cannot build group behavior matrix.")
    behavior_matrix = np.concatenate(behavior_chunks, axis=0)
    observed_trials = int(behavior_matrix.shape[0])
    if observed_trials != int(imaging_trials):
        raise RuntimeError(
            f"Behavior matrix has {observed_trials} trials after applying trial_keep, "
            f"but imaging data has {imaging_trials} trials."
        )
    return behavior_matrix, manifest_path

# =============================================================================
# Runtime data loading
# =============================================================================

def load_group_inputs():
    group_label = os.environ.get("FMRI_GROUP_LABEL", "group").strip() or "group"
    group_concat_dir = _resolve_group_concat_dir(data_base, group_label)
    if group_concat_dir is None:
        raise RuntimeError(
            "Group mode is required, but no valid group-concat directory was found. "
            "Set FMRI_GROUP_CONCAT_DIR or place group files under data/data_concatenated or data/group_concat."
        )
    base_path = group_concat_dir
    print(f"Data mode: group-concat | base_path={base_path}", flush=True)
    print(f"Results directory: {RESULTS_DIR}", flush=True)

    anatomy_root = join(data_base, "anatomy_masks")
    anat_img_path = _resolve_existing_path(join(anatomy_root, "MNI152_T1_2mm_brain.nii.gz"), join(anatomy_root, "MNI152_T1_1mm_brain.nii.gz"))
    anat_path = anat_img_path
    brain_mask_path = _resolve_existing_path(join(anatomy_root, "MNI152_T1_2mm_brain_mask.nii.gz"), join(anatomy_root, "MNI152_T1_1mm_brain_mask.nii.gz"))
    csf_mask_path = _resolve_existing_path(join(anatomy_root, "MNI152_T1_2mm_brain_seg_csf.nii.gz"), join(anatomy_root, "MNI152_T1_1mm_brain_seg_csf.nii.gz"))
    gray_mask_path = _resolve_existing_path(join(anatomy_root, "MNI152_T1_2mm_brain_seg_gm.nii.gz"), join(anatomy_root, "MNI152_T1_1mm_brain_seg_gm.nii.gz"))

    anat_img = nib.load(anat_img_path)
    brain_mask = nib.load(brain_mask_path).get_fdata().astype(np.float16)
    csf_mask = nib.load(csf_mask_path).get_fdata().astype(np.float16)
    gray_mask = nib.load(gray_mask_path).get_fdata().astype(np.float16)

    nan_mask_group_path = _resolve_numpy_path(join(base_path, f"nan_mask_flat_{group_label}.npy"), join(base_path, f"mask_all_nan_{group_label}.npy"))
    active_coords_group_path = _resolve_numpy_path(join(base_path, f"active_coords_{group_label}.npy"))
    active_flat_group_path = _resolve_numpy_path(join(base_path, f"active_flat_indices__{group_label}.npy"))
    beta_group_path = _resolve_numpy_path(join(base_path, f"beta_volume_filter_{group_label}.npy"), join(base_path, f"cleaned_beta_volume_{group_label}.npy"))
    bold_group_path = _resolve_numpy_path(join(base_path, f"active_bold_{group_label}.npy"))

    shared_nan_mask_flat = np.asarray(np.load(nan_mask_group_path, mmap_mode="r")).ravel()
    nan_mask_flat = shared_nan_mask_flat.copy()
    active_coords = _normalize_active_coords(np.load(active_coords_group_path, allow_pickle=True))
    active_flat_idx = np.asarray(np.load(active_flat_group_path, mmap_mode="r"), dtype=np.int64).ravel()
    beta_clean = np.load(beta_group_path, mmap_mode="r")
    bold_clean = np.load(bold_group_path, mmap_mode="r")
    volume_shape = anat_img.shape[:3]
    expected_mask_size = int(np.prod(volume_shape))
    if shared_nan_mask_flat.size != expected_mask_size:
        raise RuntimeError(f"Group nan mask size ({shared_nan_mask_flat.size}) does not match anatomy volume size ({expected_mask_size}).")

    num_trials = int(bold_clean.shape[1])
    if beta_clean.shape[1] != num_trials:
        print(f"WARNING: Trial counts differ (bold={num_trials}, beta={beta_clean.shape[1]}); using bold count.", flush=True)

    group_manifest_path = join(base_path, f"concat_manifest_{group_label}.tsv")
    _, transition_boundaries = _load_manifest_segments(group_manifest_path, num_trials)
    session_labels, session_metadata = _load_session_labels(group_manifest_path, num_trials)
    print(f"Loaded {len(session_metadata)} subject-session blocks for session-level EMPCA penalties.", flush=True)
    group_block_trials = int(os.environ.get("FMRI_GROUP_BLOCK_TRIALS", "90"))
    fixed_group_boundaries = _build_block_boundaries(num_trials, group_block_trials)
    if fixed_group_boundaries:
        manifest_boundary_set = set(transition_boundaries)
        missing_boundaries = [boundary for boundary in fixed_group_boundaries if boundary not in manifest_boundary_set]
        if missing_boundaries:
            preview = missing_boundaries[:8]
            tail = "..." if len(missing_boundaries) > 8 else ""
            print(f"INFO: Adding {len(missing_boundaries)} fixed {group_block_trials}-trial block boundaries "
                  f"for transition safety: {preview}{tail}", flush=True)
        transition_boundaries = sorted(set(transition_boundaries).union(fixed_group_boundaries))

    behavior_matrix, used_manifest = _load_group_behavior_matrix(base_path, group_label, num_trials)
    print(f"Loaded group behavior matrix from manifest: {used_manifest}", flush=True)

    active_coords = np.asarray(active_coords)
    transition_boundaries = sorted({int(boundary) for boundary in (transition_boundaries or []) if 0 < int(boundary) < num_trials})
    print(f"Combined active_flat_idx: {active_flat_idx.shape}", flush=True)
    print(f"Combined clean_active_bold: {bold_clean.shape}", flush=True)
    print(f"Combined active_coords: {active_coords.shape}", flush=True)
    print(f"Using num_trials={num_trials}, boundaries={transition_boundaries[:8]}{'...' if len(transition_boundaries) > 8 else ''}.", flush=True)
    print(f"Behavior matrix shape: {behavior_matrix.shape}", flush=True)

    masked_anat = mask_anatomical_image(anat_img, shared_nan_mask_flat, volume_shape)
    masked_active_anat = masked_anat.ravel()[active_flat_idx]
    finite_anat = masked_active_anat[np.isfinite(masked_active_anat)]
    anat_range = (float(np.min(finite_anat)), float(np.max(finite_anat))) if finite_anat.size else (np.nan, np.nan)
    print(f"Masked anatomical volume shape: {masked_anat.shape}, active voxels: {masked_active_anat.size}", flush=True)
    print(f"Masked anatomical intensity range (finite): [{anat_range[0]}, {anat_range[1]}]", flush=True)

    adjacency_radius_mm = 3.0
    adjacency_sigma_mm = adjacency_radius_mm / 2.0
    adjacency_matrix, degree_matrix = compute_distance_weighted_adjacency(active_coords, anat_img.affine, radius_mm=adjacency_radius_mm, sigma_mm=adjacency_sigma_mm)
    adj_data = adjacency_matrix.data
    adj_range = (float(adj_data.min()), float(adj_data.max())) if adj_data.size else (0.0, 0.0)
    degree_values = degree_matrix.data.ravel()
    degree_range = (float(degree_values.min()), float(degree_values.max())) if degree_values.size else (0.0, 0.0)
    print(f"Adjacency matrix (distance-weighted) shape: {adjacency_matrix.shape}, "
        f"nnz={adjacency_matrix.nnz}, range: [{adj_range[0]}, {adj_range[1]}]", flush=True)
    print(f"Degree matrix shape: {degree_matrix.shape}, range: [{degree_range[0]}, {degree_range[1]}]", flush=True)
    laplacian_matrix = degree_matrix - adjacency_matrix

    local_values = locals()
    runtime_names = (
        "group_label", "group_concat_dir", "base_path", "anatomy_root", "anat_img_path", "anat_path",
        "brain_mask_path", "csf_mask_path", "gray_mask_path", "anat_img", "brain_mask", "csf_mask", "gray_mask",
        "nan_mask_group_path", "active_coords_group_path", "active_flat_group_path", "beta_group_path", "bold_group_path",
        "shared_nan_mask_flat", "nan_mask_flat", "active_coords", "active_flat_idx", "beta_clean", "bold_clean",
        "volume_shape", "expected_mask_size", "num_trials", "group_manifest_path", "transition_boundaries",
        "session_labels", "session_metadata",
        "group_block_trials", "fixed_group_boundaries", "behavior_matrix", "used_manifest", "masked_anat",
        "masked_active_anat", "finite_anat", "anat_range", "adjacency_radius_mm", "adjacency_sigma_mm",
        "adjacency_matrix", "degree_matrix", "adj_data", "adj_range", "degree_values", "degree_range",
        "laplacian_matrix",
    )
    runtime = {name: local_values[name] for name in runtime_names}
    globals().update(runtime)
    return runtime


# =============================================================================
# PCA setup and normalization
# =============================================================================

def standardize_matrix(matrix):
    sym_array = 0.5 * (matrix + matrix.T)
    trace_value = np.trace(sym_array)
    if not np.isfinite(trace_value) or trace_value == 0:
        trace_value = 1.0
    normalized_mat = sym_array / trace_value * sym_array.shape[0]
    return normalized_mat

def select_empca_components(model, variance_threshold=0.8):
    explained = np.var(model.coeff, axis=0)
    total = explained.sum()
    if not np.isfinite(total) or total <= 0:
        print("EMPCA variance check failed; keeping all components.", flush=True)
        return model

    explained_ratio = explained / total
    cumulative_ratio = np.cumsum(explained_ratio)
    # for idx, (var, frac, cum_frac) in enumerate(zip(explained, explained_ratio, cumulative_ratio), start=1):
    #     print(f"EMPCA component {idx:3d}: variance={var:.6f}, ratio={frac*100:.2f}%, cumulative={cum_frac*100:.2f}%", flush=True)

    target_components = int(np.searchsorted(cumulative_ratio, variance_threshold) + 1)
    target_components = min(target_components, model.nvec)
    print(f"Using {target_components} EMPCA components for {variance_threshold*100:.0f}% variance.", flush=True)

    model.nvec = target_components
    model.eigvec = model.eigvec[:target_components]
    model.coeff = model.coeff[:, :target_components]
    model.solve_model()
    model.dof = model.data[model._unmasked].size - model.eigvec.size - model.nvec * model.nobs
    return model

def _empca_model_path():
    cache_dir_env = os.environ.get("FMRI_EMPCA_CACHE_DIR", "").strip()
    if cache_dir_env:
        cache_dir = _result_path(cache_dir_env, for_write=True)
    else:
        cache_dir = RESULTS_DIR
    return _result_path(join(cache_dir, f"empca_model_{group_label}.npy"), for_write=True)

def apply_empca(bold_clean):
    def prepare_for_empca(data):
        W = np.isfinite(data)
        Y = np.where(np.isfinite(data), data, 0.0)
        row_weight = W.sum(axis=0, keepdims=True).astype(np.float64)
        mean = np.divide((Y * W).sum(axis=0, keepdims=True), row_weight, out=np.zeros_like(row_weight), where=row_weight > 0)
        centered = Y - mean
        var = np.divide((W * centered**2).sum(axis=0, keepdims=True), row_weight, out=np.zeros_like(row_weight), where=row_weight > 0)
        scale = np.sqrt(var)
        z = np.divide(centered, np.maximum(scale, 1e-6), out=np.zeros_like(centered), where=row_weight > 0)
        return z, W
    
    X_reshap = bold_clean.reshape(bold_clean.shape[0], -1)
    Yc, W = prepare_for_empca(X_reshap.T)
    Yc = Yc.T
    W = W.T
    print("begin empca...", flush=True)
    m = empca(Yc.astype(np.float32, copy=False), W.astype(np.float32, copy=False), nvec=100, niter=10)
    model_path = _empca_model_path()
    np.save(model_path, m)
    print(f"Saved EMPCA model to {model_path}", flush=True)
    return m

def load_or_fit_empca_model(bold_clean):
    model_path = _empca_model_path()
    if os.path.exists(model_path):
        print(f"Loading existing EMPCA model from {model_path}", flush=True)
        m = np.load(model_path, allow_pickle=True).item()
        m = select_empca_components(m, variance_threshold=0.80)
        return m
    return apply_empca(bold_clean)

def build_pca_dataset(bold_clean, beta_clean, behavioral_matrix, nan_mask_flat, active_coords, active_flat_indices, trial_length, num_trials):
    pca_model = load_or_fit_empca_model(bold_clean)
    bold_pca_components = pca_model.eigvec
    print(f"***bold_pca_components, number of trials: {num_trials}")
    bold_pca_trials = bold_pca_components.reshape(bold_pca_components.shape[0], num_trials, trial_length)
    
    coeff_pinv = np.linalg.pinv(pca_model.coeff)
    beta_pca_full = coeff_pinv @ np.nan_to_num(beta_clean)

    normalization = _compute_global_normalization(behavioral_matrix, beta_pca_full, behave_indice)

    return {"pca_model": pca_model, "bold_pca_trials": bold_pca_trials, "coeff_pinv": coeff_pinv, "beta_pca_full": beta_pca_full, 
            "bold_clean": bold_clean, "beta_clean": beta_clean, "behavior_matrix": behavioral_matrix, "nan_mask_flat": nan_mask_flat, 
            "active_coords": active_coords, "active_flat_indices": active_flat_indices, **normalization}

def _compute_global_normalization(behavior_matrix, beta_pca_full, behavior_index):
    behavior_vector = behavior_matrix[:, behavior_index].ravel()
    behavior_finite_mask = np.isfinite(behavior_vector)

    behavior_centered_full = np.full_like(behavior_vector, np.nan)
    behavior_mean = np.nanmean(behavior_vector[behavior_finite_mask])
    behavior_centered_full[behavior_finite_mask] = behavior_vector[behavior_finite_mask] - behavior_mean
    behavior_norm = np.linalg.norm(behavior_centered_full[behavior_finite_mask])
    if not np.isfinite(behavior_norm) or behavior_norm <= 0:
        behavior_norm = 1.0

    behavior_normalized_full = np.full_like(behavior_vector, np.nan)
    behavior_normalized_full[behavior_finite_mask] = behavior_centered_full[behavior_finite_mask] / behavior_norm

    behavior_mask = behavior_finite_mask[None, :]
    beta_valid_mask = np.isfinite(beta_pca_full) & behavior_mask
    beta_counts = beta_valid_mask.sum(axis=1, keepdims=True)
    beta_sum = np.nansum(np.where(beta_valid_mask, beta_pca_full, 0.0), axis=1, keepdims=True)
    beta_mean = np.divide(beta_sum, beta_counts, out=np.zeros_like(beta_sum), where=beta_counts > 0)
    beta_centered_full = beta_pca_full - beta_mean

    return {"behavior_vector_full": behavior_vector, "behavior_centered_full": behavior_centered_full,
            "behavior_normalized_full": behavior_normalized_full, "behavior_mean": behavior_mean,
            "behavior_norm": behavior_norm, "beta_centered_full": beta_centered_full, "beta_mean_full": beta_mean}

def compute_fold_normalization(projection_data, train_indices, behavior_index):
    """Center/scale behaviors (and beta means) using only the training trials, then apply the scale to all trials."""
    behavior_vector_full = projection_data["behavior_vector_full"]
    beta_pca_full = projection_data["beta_pca_full"]

    total_trials = behavior_vector_full.size
    train_mask = np.zeros(total_trials, dtype=bool)
    train_mask[np.asarray(train_indices, dtype=int)] = True
    behavior_mask = np.isfinite(behavior_vector_full)
    behavior_train_mask = train_mask & behavior_mask

    behavior_mean = float(np.nanmean(behavior_vector_full[behavior_train_mask])) if np.any(behavior_train_mask) else 0.0
    behavior_centered_full = np.full_like(behavior_vector_full, np.nan)
    behavior_centered_full[behavior_mask] = behavior_vector_full[behavior_mask] - behavior_mean
    behavior_norm = float(np.linalg.norm(behavior_centered_full[behavior_train_mask])) if np.any(behavior_train_mask) else 1.0
    if not np.isfinite(behavior_norm) or behavior_norm <= 0:
        behavior_norm = 1.0

    behavior_normalized_full = np.full_like(behavior_vector_full, np.nan)
    behavior_normalized_full[behavior_mask] = behavior_centered_full[behavior_mask] / behavior_norm

    beta_valid_train = np.isfinite(beta_pca_full) & behavior_train_mask[None, :]
    beta_counts = beta_valid_train.sum(axis=1, keepdims=True)
    beta_sum = np.nansum(np.where(beta_valid_train, beta_pca_full, 0.0), axis=1, keepdims=True)
    beta_mean = np.divide(beta_sum, beta_counts, out=np.zeros_like(beta_sum), where=beta_counts > 0)
    beta_centered_full = beta_pca_full - beta_mean

    return {"behavior_vector_full": behavior_vector_full, "behavior_centered_full": behavior_centered_full, "behavior_normalized_full": behavior_normalized_full,
        "behavior_mean": behavior_mean, "behavior_norm": behavior_norm, "beta_centered_full": beta_centered_full, "beta_mean_full": beta_mean}

def build_loss_corr_term(beta_components, normalized_behaviors, ridge=1e-6):
    beta_components = np.nan_to_num(beta_components, nan=0.0, posinf=0.0, neginf=0.0)
    behavior_vector = normalized_behaviors.reshape(1, -1)
    behavior_projection = beta_components @ behavior_vector.T
    C_b = behavior_projection @ behavior_projection.T
    C_d = beta_components @ beta_components.T
    skew_b = C_b - C_b.T
    skew_d = C_d - C_d.T
    C_b = 0.5 * (C_b + C_b.T)
    C_d = 0.5 * (C_d + C_d.T)
    eye = np.eye(beta_components.shape[0])
    C_b = C_b + ridge * eye
    C_d = C_d + ridge * eye
    return C_b, C_d

def _safe_pearsonr(x, y, return_p=False):
    finite_mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite_mask], y[finite_mask]
    sample_size = x.size
    x -= x.mean()
    y -= y.mean()
    denom = np.sqrt(np.dot(x, x) * np.dot(y, y))
    correlation = np.dot(x, y) / denom
    if not return_p:
        return correlation
    r = float(np.clip(correlation, -0.999999, 0.999999))
    dof = sample_size - 2
    t_statistic = r * np.sqrt(dof) / np.sqrt(max(1e-12, 1.0 - r**2))
    p_value = student_t.sf(abs(t_statistic), dof)
    return correlation, float(p_value)

def _aggregate_trials(active_bold):
    # downsample bold data by reducer metrice
    num_voxels, num_trials, trial_length = active_bold.shape
    reshaped = active_bold.reshape(-1, trial_length)
    finite_counts = np.count_nonzero(np.isfinite(reshaped), axis=1)
    valid_mask = finite_counts > 0
    reduced_flat = np.full(reshaped.shape[0], np.nan)

    if np.any(valid_mask):
        valid_values = reshaped[valid_mask]
        reduced_values = np.nanmedian(valid_values, axis=-1)
        reduced_flat[valid_mask] = reduced_values
    return reduced_flat.reshape(num_voxels, num_trials)

def _compute_matrix_icc(data):
    """ICC(A,1) reliability across folds (raters) and voxels (targets) using pingouin."""
    n_folds, n_vox = data.shape
    fold_idx, voxel_idx = np.meshgrid(np.arange(n_folds, dtype=int), np.arange(n_vox, dtype=int), indexing="ij")
    df = pd.DataFrame({"targets": voxel_idx.ravel(), "raters": fold_idx.ravel(), "ratings": data.ravel()})
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["ratings"])
    icc_table = pg.intraclass_corr(data=df, targets="targets", raters="raters", ratings="ratings")
    icc2 = icc_table.loc[icc_table["Type"] == "ICC2", "ICC"]
    if icc2.empty or not np.isfinite(icc2.iloc[0]):
        return np.nan
    return icc2.iloc[0]

def save_brain_map(correlations, active_coords, volume_shape, anat_img, file_prefix, result_prefix="active_bold_corr", map_title=None, display_threshold_ratio=None):
    coord_arrays = tuple(np.asarray(axis, dtype=int) for axis in active_coords)
    display_values = np.abs(correlations.ravel())
    volume = np.full(volume_shape, np.nan)
    colorbar_max = None
    if coord_arrays and coord_arrays[0].size:
        volume[coord_arrays] = display_values
        finite_values = display_values[np.isfinite(display_values)]
        colorbar_max = float(np.percentile(finite_values, 99.5))

    corr_img = nib.Nifti1Image(volume, anat_img.affine, anat_img.header)
    volume_path = _result_path(f"{result_prefix}_{file_prefix}.nii.gz", for_write=True)
    nib.save(corr_img, volume_path)

    if np.any(np.isfinite(display_values)):
        display_volume = np.nan_to_num(volume, nan=0.0, posinf=0.0, neginf=0.0)
        display_volume = np.clip(display_volume, 0.0, colorbar_max)
        clamped_ratio = np.clip(display_threshold_ratio, 0.0, 1.0)
        threshold_value = colorbar_max * clamped_ratio
        display_volume[display_volume < threshold_value] = 0.0
        display_img = nib.Nifti1Image(display_volume, anat_img.affine, anat_img.header)
        display = plotting.view_img(display_img, bg_img=anat_img, colorbar=True, symmetric_cmap=False,cmap='jet', threshold=threshold_value)
        display.save_as_html(_result_path(f"{result_prefix}_{file_prefix}.html", for_write=True))
        # display.close()
    return

def enhance_bold_visualization(input_file, anat_img=None, output_prefix=None, percentiles=(90, 95, 99), min_cluster_sizes=(100, 75, 50), vmax_percentile=99.9, map_title=None):
    input_file = _result_path(input_file, allow_absolute=True)
    if not os.path.exists(input_file):
        raise RuntimeError(f"Input file not found: {input_file}")

    if output_prefix is None:
        base_name = os.path.basename(input_file)
        if base_name.endswith(".nii.gz"):
            base_name = base_name[:-7]
        else:
            base_name = os.path.splitext(base_name)[0]
        output_prefix = os.path.join(os.path.dirname(input_file), f"{base_name}_bold")
    output_prefix = _result_path(output_prefix, for_write=True)

    if len(percentiles) != len(min_cluster_sizes):
        raise RuntimeError("percentiles and min_cluster_sizes must be the same length.")

    print("==============================================", flush=True)
    print("Bold Visualization Enhancement", flush=True)
    print("==============================================", flush=True)
    print(f"Input file: {input_file}", flush=True)
    print(f"Output prefix: {output_prefix}", flush=True)
    print("", flush=True)
    print("Applying bold visualization enhancements...", flush=True)
    print("  - NO smoothing (preserves original activations only)", flush=True)
    print("  - Cluster filtering (min_size=100/75/50 voxels)", flush=True)
    print("  - Multiple threshold levels (90th/95th/99th percentile)", flush=True)
    print("", flush=True)

    img = nib.load(input_file)
    data = img.get_fdata()
    processed_data = data.copy()

    finite_vals = processed_data[np.isfinite(processed_data) & (processed_data > 0)]
    if finite_vals.size == 0:
        raise RuntimeError("No finite positive values found in data.")

    thresholds = [np.percentile(finite_vals, pct) for pct in percentiles]
    vmax = float(np.percentile(finite_vals, vmax_percentile))

    print("Thresholds:", flush=True)
    for pct, thr in zip(percentiles, thresholds):
        print(f"  {pct}th percentile: {thr:.3f}", flush=True)
    print(f"  {vmax_percentile}th percentile (vmax): {vmax:.3f}", flush=True)

    def _cluster_filter(data_in, threshold, min_cluster_size=50):
        thresholded = data_in > threshold
        labeled, num_features = ndimage.label(thresholded)
        cluster_sizes = np.bincount(labeled.ravel())
        small_clusters = cluster_sizes < min_cluster_size
        keep_mask = thresholded & ~small_clusters[labeled]
        result = np.zeros_like(data_in)
        result[keep_mask] = data_in[keep_mask]
        kept_clusters = int(num_features - np.sum(small_clusters[1:]))
        total_kept_voxels = int(np.sum(keep_mask))
        return result, kept_clusters, total_kept_voxels

    filtered_results = []
    for pct, thr, min_size in zip(percentiles, thresholds, min_cluster_sizes):
        data_thr, clusters, voxels = _cluster_filter(processed_data, thr, min_cluster_size=min_size)
        filtered_results.append((pct, thr, min_size, data_thr, clusters, voxels))

    print("\nCluster-filtered results:", flush=True)
    for pct, _, min_size, _, clusters, voxels in filtered_results:
        print(f"  {pct}th pctl (min_size={min_size}): {clusters} clusters, {voxels} voxels", flush=True)

    print("\nGenerating visualizations...", flush=True)
    for pct, thr, min_size, data_thr, clusters, _ in filtered_results:
        img_thr = nib.Nifti1Image(data_thr.astype(np.float32), img.affine, img.header)
        view = plotting.view_img(img_thr, bg_img=anat_img, cmap="hot", symmetric_cmap=False, threshold=thr, vmax=vmax, colorbar=True, title=map_title)
        out_html = f"{output_prefix}_thr{pct}.html"
        view.save_as_html(out_html)
        print(f"Saved: {out_html}", flush=True)

        display = plotting.plot_stat_map(img_thr, bg_img=anat_img,cmap="hot", symmetric_cbar=False, threshold=thr, vmax=vmax, colorbar=True, title=map_title, display_mode="ortho")
        png_path = f"{output_prefix}_thr{pct}.png"
        display.savefig(png_path, dpi=150)
        display.close()
        print(f"Saved: {png_path}", flush=True)

        nii_path = f"{output_prefix}_thr{pct}.nii.gz"
        img_thr.to_filename(nii_path)
        print(f"Saved: {nii_path}", flush=True)

    original_file = f"{output_prefix}_original.nii.gz"
    nib.save(img, original_file)
    print(f"\nSaved original activation map: {original_file}", flush=True)
    print("\nProcessing complete!", flush=True)
    print("==============================================", flush=True)
    print("", flush=True)
    return output_prefix


# =============================================================================
# Atlas alignment and region summaries
# =============================================================================

def _normalize_label_list(labels):
    normalized = []
    for label in labels:
        if isinstance(label, bytes):
            label = label.decode("utf-8", errors="replace")
        normalized.append(str(label))
    return normalized

def _fetch_ho_atlas(atlas_name, data_dir=None):
    atlas = datasets.fetch_atlas_harvard_oxford(atlas_name, data_dir=str(data_dir) if data_dir else None)
    atlas_img = atlas.maps if isinstance(atlas.maps, nib.Nifti1Image) else nib.load(atlas.maps)
    labels = _normalize_label_list(atlas.labels)
    atlas_path = atlas.maps if isinstance(atlas.maps, str) else None
    return atlas_img, labels, atlas_path

def _resolve_fsl_dir(flirt_path=None):
    if flirt_path:
        flirt_path = os.path.realpath(flirt_path)
    if flirt_path:
        candidate = os.path.dirname(os.path.dirname(flirt_path))
        if os.path.isdir(os.path.join(candidate, "data", "standard")):
            return candidate
    fsl_dir = os.environ.get("FSLDIR")
    if fsl_dir:
        return fsl_dir
    return None

def _default_mni_template(flirt_path=None):
    fsl_dir = _resolve_fsl_dir(flirt_path)
    if not fsl_dir:
        return None
    for name in ("MNI152_T1_2mm_brain.nii.gz", "MNI152_T1_1mm_brain.nii.gz"):
        candidate = os.path.join(fsl_dir, "data", "standard", name)
        if os.path.exists(candidate):
            return candidate
    return None

def _find_flirt():
    flirt_path = shutil.which("flirt")
    if flirt_path:
        return flirt_path
    fsl_dir = os.environ.get("FSLDIR")
    if not fsl_dir:
        return None
    candidate = os.path.join(fsl_dir, "bin", "flirt")
    if os.path.exists(candidate):
        return candidate
    return None

def _run_flirt(cmd):
    env = os.environ.copy()
    env.setdefault("FSLOUTPUTTYPE", "NIFTI_GZ")
    subprocess.run(cmd, check=True, env=env)

def _compute_mni_to_anat(mni_template, anat_path, out_dir, flirt_path):
    os.makedirs(out_dir, exist_ok=True)
    mat_path = os.path.join(out_dir, "mni_to_anat_flirt.mat")
    warped_path = os.path.join(out_dir, "mni_template_in_anat.nii.gz")
    if os.path.exists(mat_path) and os.path.exists(warped_path):
        return mat_path, warped_path
    cmd = [flirt_path, "-in", mni_template, "-ref", anat_path, "-omat", mat_path, "-out", warped_path, "-dof", "12"]
    _run_flirt(cmd)
    return mat_path, warped_path

def _apply_flirt(in_path, ref_path, mat_path, out_path, flirt_path, interp="nearestneighbour"):
    if os.path.exists(out_path):
        return out_path
    cmd = [flirt_path, "-in", in_path, "-ref", ref_path, "-applyxfm", "-init", mat_path, "-interp", interp, "-out", out_path]
    _run_flirt(cmd)
    return out_path

def _align_atlas_to_reference(atlas_img, anat_img, anat_path, ref_img, out_dir, assume_mni=False, return_method=False):
    use_flirt = False
    flirt_path = None
    mni_template = None
    registration_method = "resample"

    if not assume_mni:
        flirt_path = _find_flirt()
        mni_template = _default_mni_template(flirt_path)
        if flirt_path and mni_template and os.path.exists(mni_template):
            _compute_mni_to_anat(mni_template, anat_path, out_dir, flirt_path)
            use_flirt = True
            registration_method = "flirt"
            print("Registered MNI template to anatomy with FLIRT.", flush=True)
        else:
            print("WARNING: FLIRT or MNI template not available; using header-based resampling.", flush=True)

    if use_flirt:
        mat_path = os.path.join(out_dir, "mni_to_anat_flirt.mat")
        with tempfile.TemporaryDirectory(dir=out_dir) as tmpdir:
            atlas_mni_path = os.path.join(tmpdir, "atlas_mni.nii.gz")
            atlas_anat_path = os.path.join(tmpdir, "atlas_in_anat.nii.gz")
            mni_template_img = nib.load(mni_template)
            atlas_mni_img = image.resample_to_img(atlas_img, mni_template_img, interpolation="nearest", force_resample=True, copy_header=True)
            atlas_mni_img.to_filename(atlas_mni_path)
            _apply_flirt(atlas_mni_path, anat_path, mat_path, atlas_anat_path, flirt_path)
            atlas_in_anat_img = nib.load(atlas_anat_path)
            atlas_in_anat = nib.Nifti1Image(atlas_in_anat_img.get_fdata(), atlas_in_anat_img.affine, atlas_in_anat_img.header)
    else:
        atlas_in_anat = image.resample_to_img(atlas_img, anat_img, interpolation="nearest", force_resample=True, copy_header=True)

    if (atlas_in_anat.shape[:3] != ref_img.shape[:3] or not np.allclose(atlas_in_anat.affine, ref_img.affine)):
        atlas_in_ref = image.resample_to_img(atlas_in_anat, ref_img, interpolation="nearest", force_resample=True, copy_header=True)
    else:
        atlas_in_ref = atlas_in_anat

    if return_method:
        return atlas_in_ref, registration_method
    return atlas_in_ref

def _combine_atlas_data(cort_data, cort_labels, sub_data, sub_labels):
    if cort_data.shape != sub_data.shape:
        raise RuntimeError("Cortical and subcortical atlas shapes do not match.")
    combined = np.asarray(cort_data).copy()
    sub_data = np.asarray(sub_data)
    offset = len(cort_labels)
    insert_mask = (combined == 0) & (sub_data > 0)
    combined[insert_mask] = sub_data[insert_mask] + offset
    combined_labels = list(cort_labels) + [f"sub:{label}" for label in sub_labels]
    return combined, combined_labels

def _select_label_indices(labels, label_patterns):
    patterns = []
    if label_patterns:
        patterns = [p.strip().lower() for p in label_patterns if p and p.strip()]
    indices = []
    for idx, name in enumerate(labels):
        if idx == 0 or name.strip().lower() == "background":
            continue
        if not patterns:
            indices.append(idx)
            continue
        lname = name.lower()
        if any(pattern in lname for pattern in patterns):
            indices.append(idx)
    return indices

def _prepare_atlas_context(anat_img, anat_path, ref_img, output_dir, atlas_threshold=25, data_dir=None, assume_mni=False):
    output_dir = _result_path(output_dir, for_write=True)
    os.makedirs(output_dir, exist_ok=True)
    atlas_cache_path = os.path.join(output_dir, f"atlas_thr{atlas_threshold}_combined.nii.gz")
    labels_path = os.path.join(output_dir, f"atlas_thr{atlas_threshold}_combined_labels.json")

    if os.path.exists(atlas_cache_path) and os.path.exists(labels_path):
        atlas_img = nib.load(atlas_cache_path)
        if atlas_img.shape[:3] == ref_img.shape[:3] and np.allclose(atlas_img.affine, ref_img.affine):
            with open(labels_path, "r", encoding="utf-8") as handle:
                labels = json.load(handle)
            atlas_data = atlas_img.get_fdata().astype(np.int32)
            return {"atlas_data": atlas_data, "labels": labels, "shape": atlas_img.shape[:3], "affine": atlas_img.affine, "path": atlas_cache_path,
                    "registration_method": "cached", "atlas_threshold": atlas_threshold}

    cort_name = f"cort-maxprob-thr{atlas_threshold}-2mm"
    sub_name = f"sub-maxprob-thr{atlas_threshold}-2mm"
    cort_img, cort_labels, _ = _fetch_ho_atlas(cort_name, data_dir=data_dir)
    sub_img, sub_labels, _ = _fetch_ho_atlas(sub_name, data_dir=data_dir)

    cort_in_ref, registration_method = _align_atlas_to_reference(cort_img, anat_img, anat_path, ref_img, output_dir, assume_mni=assume_mni, return_method=True)
    sub_in_ref = _align_atlas_to_reference(sub_img, anat_img, anat_path, ref_img, output_dir, assume_mni=assume_mni, return_method=False)

    cort_data = np.rint(cort_in_ref.get_fdata())
    sub_data = np.rint(sub_in_ref.get_fdata())
    combined_data, combined_labels = _combine_atlas_data(cort_data, cort_labels, sub_data, sub_labels)

    combined_img = nib.Nifti1Image(combined_data, ref_img.affine, ref_img.header)
    nib.save(combined_img, atlas_cache_path)
    with open(labels_path, "w", encoding="utf-8") as handle:
        json.dump(combined_labels, handle, indent=2)

    return {"atlas_data": combined_data, "labels": combined_labels, "shape": combined_img.shape[:3], "affine": combined_img.affine, "path": atlas_cache_path,
            "registration_method": registration_method, "atlas_threshold": atlas_threshold}

def _analyze_weight_map_regions(voxel_weights_path, atlas_context, output_prefix, motor_label_patterns, threshold_percentile=95):
    weights_img = nib.load(voxel_weights_path)
    weights_data = weights_img.get_fdata()
    atlas_data = atlas_context["atlas_data"]
    labels = atlas_context["labels"]
    output_prefix = _result_path(output_prefix, for_write=True)

    active_mask = np.isfinite(weights_data)
    if not np.any(active_mask):
        print("WARNING: No finite voxels found in voxel-weights map.", flush=True)
        return None

    active_values = np.abs(weights_data[active_mask])
    active_values = active_values[np.isfinite(active_values)]
    if active_values.size == 0:
        print("WARNING: No finite weights available for percentile thresholding.", flush=True)
        return None

    threshold_value = float(np.percentile(active_values, threshold_percentile))
    suprath_mask = active_mask & (np.abs(weights_data) >= threshold_value)

    active_labels = atlas_data[active_mask]
    active_labels = active_labels[(active_labels > 0) & (active_labels < len(labels))]
    active_counts = np.bincount(active_labels, minlength=len(labels))

    suprath_labels = atlas_data[suprath_mask]
    suprath_labels = suprath_labels[(suprath_labels > 0) & (suprath_labels < len(labels))]
    suprath_counts = np.bincount(suprath_labels, minlength=len(labels))

    total_suprath = int(np.sum(suprath_counts))
    total_active = int(np.sum(active_counts))

    motor_indices = _select_label_indices(labels, motor_label_patterns) if motor_label_patterns else []
    motor_mask = active_mask & np.isin(atlas_data, motor_indices)
    motor_coords = np.column_stack(np.nonzero(motor_mask))
    np.savez(f"{output_prefix}_motor_voxel_indicies.npz", indices=motor_coords)
    suprath_coords = np.column_stack(np.nonzero(suprath_mask))
    np.savez(f"{output_prefix}_selected_voxel_indicies.npz", indices=suprath_coords)

    records = []
    for idx in range(1, len(labels)):
        active_count = int(active_counts[idx]) if idx < active_counts.size else 0
        suprath_count = int(suprath_counts[idx]) if idx < suprath_counts.size else 0
        if active_count == 0 and suprath_count == 0:
            continue
        records.append({"label_index": idx, "label": labels[idx], "active_voxels": active_count, "suprathreshold_voxels": suprath_count,
                        "pct_suprathreshold": (suprath_count / total_suprath * 100.0) if total_suprath else 0.0,
                        "pct_of_active_region": (suprath_count / active_count * 100.0) if active_count else 0.0})

    if records:
        summary_df = pd.DataFrame(records).sort_values("suprathreshold_voxels", ascending=False)
    else:
        summary_df = pd.DataFrame(columns=["label_index", "label", "active_voxels", "suprathreshold_voxels", "pct_suprathreshold", "pct_of_active_region"])
    summary_csv = f"{output_prefix}_atlas_region_distribution_thr{int(threshold_percentile)}.csv"
    summary_df.to_csv(summary_csv, index=False)

    motor_suprath = int(np.sum([suprath_counts[idx] for idx in motor_indices])) if motor_indices else 0
    motor_active = int(np.sum([active_counts[idx] for idx in motor_indices])) if motor_indices else 0
    motor_pct = (motor_suprath / total_suprath * 100.0) if total_suprath else 0.0
    motor_region_pct = (motor_suprath / motor_active * 100.0) if motor_active else 0.0

    summary_json = f"{output_prefix}_atlas_region_distribution_thr{int(threshold_percentile)}.json"
    summary_payload = {"voxel_weights_path": voxel_weights_path,
        "atlas_path": atlas_context.get("path"),
        "atlas_threshold": atlas_context.get("atlas_threshold"),
        "atlas_registration_method": atlas_context.get("registration_method"),
        "threshold_percentile": float(threshold_percentile),
        "threshold_value": threshold_value,
        "total_active_voxels": total_active,
        "total_suprathreshold_voxels": total_suprath,
        "motor_suprathreshold_voxels": motor_suprath,
        "motor_suprathreshold_pct": motor_pct,
        "motor_active_voxels": motor_active,
        "motor_active_suprath_pct": motor_region_pct,
        "motor_label_patterns": list(motor_label_patterns) if motor_label_patterns else []}
    with open(summary_json, "w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2)

    print(f"Atlas summary saved: {summary_csv} (thr={threshold_value:.4f}, suprath={total_suprath}, motor={motor_pct:.1f}%)", flush=True)
    return summary_payload

def _load_projection_series(series_path):
    series_path = _result_path(series_path, allow_absolute=True)
    if not os.path.exists(series_path):
        return {}
    loaded = np.load(series_path, allow_pickle=True)
    if isinstance(loaded, np.ndarray):
        data = loaded.item()
    else:
        data = dict(loaded)
    series = {}
    for key, values in data.items():
        gamma_key = float(key)
        series[gamma_key] = values
    return series

def _merge_projection_series(existing_series, new_series):
    merged = dict(existing_series)
    for gamma_value, projection in new_series:
        if projection is None:
            continue
        merged[float(gamma_value)] = np.asarray(projection).ravel()
    return merged


# =============================================================================
# Objective matrices and correlations
# =============================================================================

def calcu_penalty_terms(run_data, alpha_task, alpha_bold, alpha_beta, alpha_smooth):
    beta_centered = run_data["beta_centered"]
    n_components = beta_centered.shape[0]

    task_penalty = alpha_task * run_data["C_task"]
    bold_penalty = alpha_bold * run_data["C_bold"]
    beta_penalty = alpha_beta * run_data["C_beta"]
    smooth_matrix = run_data.get("C_smooth")
    skew_smooth = smooth_matrix - smooth_matrix.T
    # print(f"C_smooth symmetry deviation: fro_norm={np.linalg.norm(skew_smooth, ord='fro'):.6e}, max_abs={np.max(np.abs(skew_smooth)):.6e}",flush=True)
    smooth_penalty = alpha_smooth * smooth_matrix

    total_penalty = task_penalty + bold_penalty + beta_penalty + smooth_penalty
    total_penalty = 0.5 * (total_penalty + total_penalty.T)
    total_penalty = total_penalty + 1e-6 * np.eye(n_components)
    return {"task": task_penalty, "bold": bold_penalty, "beta": beta_penalty, "smooth": smooth_penalty}, total_penalty

# def _corr_ratio_and_components(run_data, weights, eps=1e-8):
#     weights = np.asarray(weights, dtype=np.float64)
#     beta_centered = run_data["beta_centered"]
#     behavior_vec = run_data["normalized_behaviors"]
#     y = beta_centered.T @ weights
#     corr_num = float(np.dot(y, behavior_vec))
#     corr_den = float(np.dot(y, y) + eps)
#     corr_ratio = (corr_num ** 2) / corr_den if np.isfinite(corr_den) and corr_den > 0 else np.inf
#     return corr_ratio, corr_num, corr_den, y

# def compute_total_loss(run_data, weights, total_penalty, gamma_value, penalty_terms=None, label=None, eps=1e-8):
#     weights = np.asarray(weights, dtype=np.float64)
#     corr_ratio, corr_num, corr_den, _ = _corr_ratio_and_components(run_data, weights, eps=eps)
#     penalty_value = float(weights.T @ total_penalty @ weights)
#     if penalty_terms:
#         label_prefix = f"[{label}] " if label else ""
#         for term_label, matrix in penalty_terms.items():
#             contribution = float(weights.T @ matrix @ weights)
#             print(f"{label_prefix}{term_label}_penalty: {contribution:.6f}", flush=True)
#     total_loss = gamma_value * penalty_value - corr_ratio
#     return total_loss, penalty_value, corr_ratio, corr_num, corr_den

def _build_objective_matrices(total_penalty, corr_num, corr_den, gamma_value, corr_weight=1.0, eps=1e-8):
    A_mat = gamma_value * total_penalty - corr_weight * corr_num
    A_mat = 0.5 * (A_mat + A_mat.T)
    eye = np.eye(A_mat.shape[0])
    if corr_weight == 0:
        B_mat = eye
        return A_mat + eps * eye, B_mat
    B_mat = 0.5 * (corr_den + corr_den.T)
    return A_mat + eps * eye, B_mat + eps * eye

def _total_loss_from_penalty(weights, total_penalty, gamma_value, corr_weight=1.0, corr_num=None, corr_den=None, penalty_terms=None, label=None, A_mat=None, B_mat=None):
    weights = np.where(np.isfinite(weights), weights, 0.0)
    A_mat, B_mat = _build_objective_matrices(total_penalty, corr_num, corr_den, gamma_value, corr_weight=corr_weight)
    numerator = float(weights.T @ A_mat @ weights)
    denominator = float(weights.T @ B_mat @ weights)
    if penalty_terms:
        label_prefix = f"[{label}] " if label else ""
        for term_label, matrix in penalty_terms.items():
            contribution = float(weights.T @ matrix @ weights)
            print(f"{label_prefix}{term_label}_penalty: {contribution:.6f}", flush=True)
    return numerator / denominator

def evaluate_projection_corr(data, weights):
    #corr(W*beta, behave)
    beta_centered = data["beta_centered"]
    behavior_centered = data["behavior_centered"]
    normalized_behaviors = data["normalized_behaviors"]

    Y = beta_centered.T @ weights
    finite_mask = np.isfinite(Y) & np.isfinite(behavior_centered)
    Y = Y[finite_mask]
    behavior_centered = behavior_centered[finite_mask]
    normalized_behaviors = normalized_behaviors[finite_mask]

    metrics = {"pearson": np.nan, "pearson_p": np.nan}
    metrics["pearson"], metrics["pearson_p"] = _safe_pearsonr(Y, behavior_centered, return_p=True)
    return metrics

def _compute_projection(weights, mat):
    if mat.ndim != 2:
        mat = mat.reshape(mat.shape[0], -1)
    finite_mask = np.isfinite(mat)
    weighted = mat * weights[:, None]
    projection = np.where(np.any(finite_mask, axis=0), np.nansum(weighted, axis=0), np.nan)
    return projection

def _compute_bold_projection(voxel_weights, data):
    active_bold = data.get("active_bold")
    bold_matrix = active_bold.reshape(active_bold.shape[0], -1).copy()
    voxel_means = np.nanmean(bold_matrix, axis=1, keepdims=True)
    voxel_means = np.where(np.isfinite(voxel_means), voxel_means, 0.0)
    bold_matrix -= voxel_means
    projection = _compute_projection(voxel_weights, bold_matrix)
    return projection, bold_matrix

def evaluate_beta_bold_projection_corr(voxel_weights, data):
    #corr(w*beta, bold)
    active_beta = data.get("beta_clean")
    active_bold = data.get("active_bold")
    bold_trial_metric = _aggregate_trials(active_bold)
    projection = _compute_projection(voxel_weights, active_beta)
    correlations = np.full(active_beta.shape[0], np.nan)
    projection_finite = np.isfinite(projection)

    for voxel_idx, voxel_series in enumerate(bold_trial_metric):
        voxel_finite = np.isfinite(voxel_series)
        joint_mask = projection_finite & voxel_finite
        if np.count_nonzero(joint_mask) < 2:
            continue
        correlations[voxel_idx] = _safe_pearsonr(projection[joint_mask], voxel_series[joint_mask])
    return projection, correlations

def evaluate_bold_bold_projection_corr(voxel_weights, data):
    # corr(weight * bold, bold)
    projection, bold_matrix = _compute_bold_projection(voxel_weights, data)
    correlations = np.full(bold_matrix.shape[0], np.nan)
    projection_finite = np.isfinite(projection)

    for voxel_idx, voxel_series in enumerate(bold_matrix):
        voxel_finite = np.isfinite(voxel_series)
        joint_mask = projection_finite & voxel_finite
        if np.count_nonzero(joint_mask) < 2:
            continue
        correlations[voxel_idx] = _safe_pearsonr(projection[joint_mask], voxel_series[joint_mask])
    return projection, correlations


# =============================================================================
# Cross-validation data preparation
# =============================================================================

def _split_contiguous_folds(trial_count, num_folds):
    trial_count = int(trial_count)
    num_folds = int(num_folds)
    if trial_count < 0:
        raise RuntimeError(f"trial_count must be >= 0, got {trial_count}.")
    if num_folds <= 0:
        raise RuntimeError(f"num_folds must be > 0, got {num_folds}.")
    base_fold = trial_count // num_folds
    remainder = trial_count % num_folds
    fold_sizes = np.full(num_folds, base_fold, dtype=int)
    if remainder > 0:
        fold_sizes[:remainder] += 1
    folds = []
    start = 0
    for block_size in fold_sizes:
        end = start + block_size
        folds.append(np.arange(start, end))
        start = end
    return folds

def build_custom_kfold_splits(total_trials, num_folds=10):
    total_trials = int(total_trials)
    num_folds = int(num_folds)
    all_trials = np.arange(total_trials)
    test_folds = _split_contiguous_folds(total_trials, num_folds)
    folds = []
    for fold_idx, test_indices in enumerate(test_folds, start=1):
        test_indices = np.asarray(test_indices)
        train_mask = np.ones(total_trials, dtype=bool)
        train_mask[test_indices] = False
        train_indices = all_trials[train_mask]
        folds.append({"fold_id": fold_idx, "train_indices": train_indices, "test_indices": test_indices})
    return folds

def _crosses_any_boundary(idx_current, idx_next, transition_boundaries):
    if not transition_boundaries:
        return False
    for boundary in transition_boundaries:
        if idx_current < boundary <= idx_next:
            return True
    return False

_TASK_RANKING_REPORTED_KEYS = set()

def _component_task_matrix(beta_values):
    counts = np.count_nonzero(np.isfinite(beta_values), axis=-1)
    sums = np.nansum(np.abs(beta_values), axis=-1)
    mean_beta = np.zeros(beta_values.shape[0])
    mask = counts > 0
    mean_beta[mask] = sums[mask] / counts[mask]

    task_diag = np.zeros_like(mean_beta)
    valid = mean_beta > 0
    task_diag[valid] = 1.0 / mean_beta[valid]
    return np.diag(task_diag), int(np.count_nonzero(valid))

def _component_beta_transition_matrix(beta_values, global_trial_indices, transition_boundaries):
    c_beta = np.zeros((beta_values.shape[0], beta_values.shape[0]))
    transition_count = 0
    global_trial_indices = np.asarray(global_trial_indices, dtype=int).ravel()
    for i in range(global_trial_indices.size - 1):
        idx_current = int(global_trial_indices[i])
        idx_next = int(global_trial_indices[i + 1])
        if (idx_next - idx_current) != 1:
            continue
        if _crosses_any_boundary(idx_current, idx_next, transition_boundaries):
            continue
        diff = beta_values[:, i] - beta_values[:, i + 1]
        c_beta += np.outer(diff, diff)
        transition_count += 1
    if transition_count > 0:
        c_beta /= transition_count
    return c_beta, transition_count

def _ordered_unique(values):
    seen = set()
    ordered = []
    for value in np.asarray(values).ravel():
        key = int(value)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(key)
    return ordered

def _rank_vector(values):
    values = np.asarray(values, dtype=np.float64).ravel()
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks

def _finite_corr(x, y):
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    finite = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(finite) < 2:
        return np.nan
    x = x[finite] - np.mean(x[finite])
    y = y[finite] - np.mean(y[finite])
    denom = np.sqrt(np.dot(x, x) * np.dot(y, y))
    if not np.isfinite(denom) or denom <= 0:
        return np.nan
    return float(np.dot(x, y) / denom)

def _print_task_ranking_comparison(pooled_task, session_task, diagnostics, trial_indices):
    trial_indices = np.asarray(trial_indices, dtype=np.int64).ravel()
    report_key = (int(trial_indices.size), int(trial_indices[0]) if trial_indices.size else -1,
                  int(trial_indices[-1]) if trial_indices.size else -1, trial_indices.tobytes())
    if report_key in _TASK_RANKING_REPORTED_KEYS:
        return
    _TASK_RANKING_REPORTED_KEYS.add(report_key)

    pooled_diag = np.diag(pooled_task)
    session_diag = np.diag(session_task)
    finite = np.isfinite(pooled_diag) & np.isfinite(session_diag)
    if not np.any(finite):
        print("C_task session-average ranking check: no finite component penalties.", flush=True)
        return

    pooled_finite = pooled_diag[finite]
    session_finite = session_diag[finite]
    spearman = _finite_corr(_rank_vector(pooled_finite), _rank_vector(session_finite))
    pearson = _finite_corr(pooled_finite, session_finite)
    top_k = min(8, pooled_diag.size)
    pooled_top = (np.argsort(-pooled_diag)[:top_k] + 1).astype(int).tolist()
    session_top = (np.argsort(-session_diag)[:top_k] + 1).astype(int).tolist()
    overlap = len(set(pooled_top).intersection(session_top))
    max_abs_delta = float(np.nanmax(np.abs(session_diag - pooled_diag)))

    beta_counts = np.asarray(diagnostics.get("beta_transition_counts", []), dtype=np.float64)
    if beta_counts.size:
        beta_summary = f"beta session transitions min/median/max={int(np.min(beta_counts))}/{float(np.median(beta_counts)):.1f}/{int(np.max(beta_counts))}"
    else:
        beta_summary = "beta session transitions n/a"

    print(
        "C_task session-average ranking check "
        f"(trials={trial_indices.size}, task_sessions={diagnostics.get('task_session_count', 0)}, "
        f"beta_sessions={diagnostics.get('beta_session_count', 0)}, {beta_summary}): "
        f"diag Pearson={pearson:.4f}, Spearman={spearman:.4f}, top{top_k}_overlap={overlap}/{top_k}, "
        f"max_abs_diag_delta={max_abs_delta:.4g}, pooled_top={pooled_top}, session_top={session_top}",
        flush=True,
    )

def _session_average_component_penalties(beta_pca, trial_indices, transition_boundaries, session_labels):
    pooled_task_raw, _ = _component_task_matrix(beta_pca)
    pooled_beta_raw, pooled_transition_count = _component_beta_transition_matrix(beta_pca, trial_indices, transition_boundaries)
    pooled_task = standardize_matrix(pooled_task_raw)
    pooled_beta = standardize_matrix(pooled_beta_raw)

    if session_labels is None:
        diagnostics = {"task_session_count": 0, "beta_session_count": 0, "pooled_beta_transition_count": pooled_transition_count,
                       "beta_transition_counts": []}
        return pooled_task, pooled_beta, pooled_task, diagnostics

    session_labels = np.asarray(session_labels, dtype=np.int32).ravel()
    if session_labels.size != beta_pca.shape[1]:
        raise RuntimeError(
            f"session_labels length ({session_labels.size}) does not match beta_pca trial count ({beta_pca.shape[1]})."
        )

    task_blocks = []
    beta_blocks = []
    beta_transition_counts = []
    trial_indices = np.asarray(trial_indices, dtype=int).ravel()

    for session_id in _ordered_unique(session_labels):
        session_positions = np.flatnonzero(session_labels == session_id)
        if session_positions.size == 0:
            continue

        task_raw, task_valid_count = _component_task_matrix(beta_pca[:, session_positions])
        if task_valid_count > 0:
            task_blocks.append(standardize_matrix(task_raw))

        beta_raw, transition_count = _component_beta_transition_matrix(
            beta_pca[:, session_positions],
            trial_indices[session_positions],
            transition_boundaries,
        )
        if transition_count > 0:
            beta_blocks.append(standardize_matrix(beta_raw))
            beta_transition_counts.append(transition_count)

    c_task = standardize_matrix(np.mean(np.stack(task_blocks, axis=0), axis=0)) if task_blocks else pooled_task
    c_beta = standardize_matrix(np.mean(np.stack(beta_blocks, axis=0), axis=0)) if beta_blocks else pooled_beta
    diagnostics = {
        "task_session_count": len(task_blocks),
        "beta_session_count": len(beta_blocks),
        "pooled_beta_transition_count": pooled_transition_count,
        "beta_transition_counts": beta_transition_counts,
    }
    return c_task, c_beta, pooled_task, diagnostics

def calcu_matrices_func(beta_pca, bold_pca, behave_mat, behave_indice, trial_len, num_trials, trial_indices, transition_boundaries, session_labels=None):
    bold_pca_reshape = bold_pca.reshape(bold_pca.shape[0], num_trials, trial_len)
    behavior_selected = behave_mat[:, behave_indice]
    transition_boundaries = sorted({int(boundary) for boundary in (transition_boundaries or []) if int(boundary) > 0})

    C_bold = np.zeros((bold_pca_reshape.shape[0], bold_pca_reshape.shape[0]))
    bold_transition_count = 0
    for i in range(num_trials - 1):
        idx_current = trial_indices[i]
        idx_next = trial_indices[i + 1]
        if (idx_next - idx_current) != 1:
            continue
        if _crosses_any_boundary(idx_current, idx_next, transition_boundaries):
            continue
        x1 = bold_pca_reshape[:, i, :]
        x2 = bold_pca_reshape[:, i + 1, :]
        C_bold += (x1 - x2) @ (x1 - x2).T
        bold_transition_count += 1
    if bold_transition_count > 0:
        C_bold /= bold_transition_count

    C_task, C_beta, pooled_C_task, diagnostics = _session_average_component_penalties(
        beta_pca, trial_indices, transition_boundaries, session_labels
    )
    _print_task_ranking_comparison(pooled_C_task, C_task, diagnostics, trial_indices)

    return C_task, C_bold, C_beta, behavior_selected

def prepare_data_func(projection_data, trial_indices, trial_length, transition_boundaries, normalization_info):
    effective_num_trials = trial_indices.size
    behavior_subset = projection_data["behavior_matrix"][trial_indices]
    behavior_vector_full = normalization_info["behavior_vector_full"]
    behavior_centered_full_all = normalization_info["behavior_centered_full"]
    behavior_normalized_full_all = normalization_info["behavior_normalized_full"]
    beta_centered_full_all = normalization_info["beta_centered_full"]
    beta_pca_full = projection_data["beta_pca_full"]

    behavior_vector = behavior_vector_full[trial_indices]
    behavior_centered_full = behavior_centered_full_all[trial_indices]
    behavior_normalized_full = behavior_normalized_full_all[trial_indices]
    beta_centered_full = beta_centered_full_all
    beta_pca = beta_pca_full[:, trial_indices]
    beta_centered_subset = beta_centered_full[:, trial_indices]
    bold_pca_trials = projection_data["bold_pca_trials"][:, trial_indices, :]
    bold_pca_components = bold_pca_trials.reshape(bold_pca_trials.shape[0], effective_num_trials * trial_length)
    session_labels_full = projection_data.get("session_labels")
    session_labels_subset = None
    if session_labels_full is not None:
        session_labels_subset = np.asarray(session_labels_full, dtype=np.int32).ravel()[trial_indices]

    (C_task, C_bold, C_beta, _) = calcu_matrices_func(beta_pca, bold_pca_components, behavior_subset, behave_indice,
                                                      trial_len=trial_length, num_trials=effective_num_trials,
                                                      trial_indices=trial_indices, transition_boundaries=transition_boundaries,
                                                      session_labels=session_labels_subset)
    skew_task = C_task - C_task.T
    skew_bold = C_bold - C_bold.T
    skew_beta = C_beta - C_beta.T
    # print(f"C_task symmetry deviation: fro_norm={np.linalg.norm(skew_task, ord='fro'):.6e}, max_abs={np.max(np.abs(skew_task)):.6e}",flush=True)
    # print(f"C_bold symmetry deviation: fro_norm={np.linalg.norm(skew_bold, ord='fro'):.6e}, max_abs={np.max(np.abs(skew_bold)):.6e}",flush=True)
    # print(f"C_beta symmetry deviation: fro_norm={np.linalg.norm(skew_beta, ord='fro'):.6e}, max_abs={np.max(np.abs(skew_beta)):.6e}",flush=True)
    C_task, C_bold, C_beta = standardize_matrix(C_task), standardize_matrix(C_bold), standardize_matrix(C_beta)

    trial_mask = np.isfinite(behavior_vector)
    behavior_observed = behavior_vector[trial_mask]
    behavior_centered = behavior_centered_full[trial_mask]
    normalized_behaviors = behavior_normalized_full[trial_mask]

    beta_observed = beta_pca[:, trial_mask]
    beta_centered = beta_centered_subset[:, trial_mask]

    # Correlation term (r^2) — keep the ratio intact, but rescale both matrices by a shared scalar
    # so their magnitude is comparable to the (standardized) penalty matrices. Using a common scale
    # preserves w^T C_corr_num w / w^T C_corr_den w while preventing tiny denominators.
    C_corr_num, C_corr_den = build_loss_corr_term(beta_centered, normalized_behaviors)
    corr_scale = float(np.trace(C_corr_den)) / max(1, C_corr_den.shape[0])
    if not np.isfinite(corr_scale) or corr_scale <= 0:
        corr_scale = 1.0
    C_corr_num = C_corr_num / corr_scale
    C_corr_den = C_corr_den / corr_scale

    beta_subset = projection_data["beta_clean"][:, trial_indices]
    bold_subset = projection_data["bold_clean"][:, trial_indices, :]

    return {"nan_mask_flat": projection_data["nan_mask_flat"],
        "active_coords": tuple(np.array(coord) for coord in projection_data["active_coords"]),
        "active_flat_indices": projection_data["active_flat_indices"],
        "coeff_pinv": projection_data["coeff_pinv"], "beta_centered": beta_centered,
        "behavior_centered": behavior_centered, "normalized_behaviors": normalized_behaviors,
        "behavior_observed": behavior_observed, "beta_observed": beta_observed,
        "beta_clean": beta_subset, "C_task": C_task, "C_bold": C_bold, "C_beta": C_beta,
        "C_smooth": projection_data.get("C_smooth"),
        "C_corr_num": C_corr_num, "C_corr_den": C_corr_den,
        "active_bold": bold_subset, "trial_indices": trial_indices, "num_trials": effective_num_trials}


# =============================================================================
# Optimization solvers
# =============================================================================

def solve_soc_problem(run_data, alpha_task, alpha_bold, alpha_beta, alpha_smooth, gamma, corr_weight=1.0):
    beta_centered = run_data["beta_centered"]
    n_components = beta_centered.shape[0]
    penalty_matrices, total_penalty = calcu_penalty_terms(run_data, alpha_task, alpha_bold, alpha_beta, alpha_smooth)

    corr_num, corr_den = run_data["C_corr_num"], run_data["C_corr_den"]
    A_mat, B_mat = _build_objective_matrices(total_penalty, corr_num, corr_den, gamma, corr_weight=corr_weight)

    def _objective_with_grad(weights):
        numerator = weights.T @ A_mat @ weights
        denominator = weights.T @ B_mat @ weights
        grad_num = 2.0 * (A_mat @ weights)
        grad_den = 2.0 * (B_mat @ weights)
        objective_value = numerator / denominator
        # objective_value = numerator / 1
        gradient = (grad_num * denominator - grad_den * numerator) / (denominator**2)
        # gradient = grad_num
        return objective_value, gradient
    def _objective(weights):
        value, _ = _objective_with_grad(weights)
        return value
    def _objective_grad(weights):
        _, grad = _objective_with_grad(weights)
        return grad

    constraints = [{"type": "eq", "fun": lambda w: np.linalg.norm(w) - 1.0, "jac": lambda w: w / (np.linalg.norm(w) + 1e-12)}]
    initial_weights = np.full(n_components, 1.0 / np.sqrt(n_components))
    result = minimize(_objective, initial_weights, jac=_objective_grad, constraints=constraints, 
                      method="SLSQP", options={"maxiter": 1000, "ftol": 1e-8, "disp": True})
    solution_weights = np.asarray(result.x)
    denominator_value = solution_weights.T @ B_mat @ solution_weights

    contributions = {label: solution_weights.T @ matrix @ solution_weights for label, matrix in penalty_matrices.items()}
    y = beta_centered.T @ solution_weights
    penalty_value = float(solution_weights.T @ total_penalty @ solution_weights)
    gamma_penalty_value = float(gamma * penalty_value)
    gamma_penalty_ratio = gamma_penalty_value / denominator_value
    numerator_value = solution_weights.T @ A_mat @ solution_weights
    correlation_numerator = solution_weights.T @ corr_num @ solution_weights
    total_loss = _total_loss_from_penalty(solution_weights, total_penalty, gamma, corr_weight=corr_weight, corr_num=corr_num, corr_den=corr_den, A_mat=A_mat, B_mat=B_mat)
    fractional_objective = numerator_value / denominator_value
    correlation_ratio = correlation_numerator / denominator_value

    return {"weights": solution_weights, "total_loss": total_loss, "Y": y, "fractional_objective": fractional_objective,
            "numerator": numerator_value, "denominator": denominator_value, "penalty_contributions": contributions, "corr_ratio": correlation_ratio,
            "penalty_value": penalty_value, "gamma_penalty": gamma_penalty_value, "gamma_penalty_ratio": gamma_penalty_ratio}

def solve_soc_problem_eig(run_data, alpha_task, alpha_bold, alpha_beta, alpha_smooth, gamma, corr_weight=1.0):
    beta_centered = run_data["beta_centered"]
    n_components = beta_centered.shape[0]
    penalty_matrices, total_penalty = calcu_penalty_terms(run_data, alpha_task, alpha_bold, alpha_beta, alpha_smooth)

    corr_num, corr_den = run_data["C_corr_num"], run_data["C_corr_den"]
    A_mat, B_mat = _build_objective_matrices(total_penalty, corr_num, corr_den, gamma, corr_weight=corr_weight)
    A_mat = 0.5 * (A_mat + A_mat.T)
    B_mat = 0.5 * (B_mat + B_mat.T)

    # Ensure B is positive definite for the generalized eigenvalue problem.
    min_eig_B = float(np.min(np.linalg.eigvalsh(B_mat)))
    if not np.isfinite(min_eig_B):
        raise RuntimeError("Invalid correlation denominator matrix (non-finite eigenvalues).")
    if min_eig_B <= 0:
        B_mat = B_mat + (abs(min_eig_B) + 1e-8) * np.eye(n_components)

    eigvals, eigvecs = eigh(A_mat, B_mat)
    finite_mask = np.isfinite(eigvals)
    min_idx = int(np.nanargmin(eigvals))
    solution_weights = np.asarray(eigvecs[:, min_idx]).ravel()
    b_norm = float(np.sqrt(solution_weights.T @ B_mat @ solution_weights))
    solution_weights = solution_weights / b_norm

    contributions = {label: float(solution_weights.T @ matrix @ solution_weights) for label, matrix in penalty_matrices.items()}
    y = beta_centered.T @ solution_weights
    penalty_value = float(solution_weights.T @ total_penalty @ solution_weights)
    gamma_penalty_value = float(gamma * penalty_value)
    denominator_value = float(solution_weights.T @ B_mat @ solution_weights)
    numerator_value = float(solution_weights.T @ A_mat @ solution_weights)
    correlation_numerator = float(solution_weights.T @ corr_num @ solution_weights)
    gamma_penalty_ratio = gamma_penalty_value / denominator_value
    total_loss = _total_loss_from_penalty(solution_weights, total_penalty, gamma, corr_weight=corr_weight, corr_num=corr_num, corr_den=corr_den, A_mat=A_mat, B_mat=B_mat)
    fractional_objective = numerator_value / denominator_value
    correlation_ratio = correlation_numerator / denominator_value

    return {"weights": solution_weights, "total_loss": total_loss, "Y": y, "fractional_objective": fractional_objective,
            "numerator": numerator_value, "denominator": denominator_value, "penalty_contributions": contributions, "corr_ratio": correlation_ratio,
            "penalty_value": penalty_value, "gamma_penalty": gamma_penalty_value, "gamma_penalty_ratio": gamma_penalty_ratio}


# =============================================================================
# Plotting and saved outputs
# =============================================================================

def plot_projection_bold(y_trials, task_alpha, bold_alpha, beta_alpha, gamma_value, output_path, max_trials=10, series_label=None, trial_indices_array=None, trial_windows=None):
    output_path = _result_path(output_path, for_write=True)
    num_trials_total, trial_length = y_trials.shape
    time_axis = np.arange(trial_length)
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    if trial_windows is None:
        window_span = 10
        first_start = 10 if num_trials_total > 20 else max(0, num_trials_total // 4)
        second_start = num_trials_total // 2
        def _build_window(start_idx):
            start_idx = int(max(0, min(start_idx, max(num_trials_total - 1, 0))))
            end_idx = min(start_idx + window_span, num_trials_total)
            if end_idx <= start_idx:
                end_idx = min(num_trials_total, start_idx + max(1, window_span))
            return start_idx, end_idx
        first_window = _build_window(first_start)
        second_window = _build_window(max(second_start, first_window[1]))
        trial_windows = [first_window, second_window]

    def _get_light_colors(count):
        cmap = plt.cm.get_cmap("tab20", count)
        raw_colors = cmap(np.linspace(0, 1, count))
        pastel = raw_colors * 0.55 + 0.45
        return np.clip(pastel, 0.0, 1.0)

    for ax, (start, end) in zip(axes, trial_windows):
        start_idx = min(start, num_trials_total)
        end_idx = min(end, num_trials_total)
        if start_idx >= end_idx:
            ax.axis("off")
            continue
        window_trials = y_trials[start_idx:end_idx]
        trials_to_plot = window_trials[:max_trials]
        colors = _get_light_colors(trials_to_plot.shape[0])
        for trial_offset, (trial_values, color) in enumerate(zip(trials_to_plot, colors)):
            if trial_indices_array is not None:
                global_idx = int(trial_indices_array[start_idx + trial_offset]) + 1
            else:
                global_idx = start_idx + trial_offset + 1
            label = f"Trial {global_idx}"
            ax.plot(time_axis, trial_values, linewidth=1.0, color=color, alpha=0.9, label=label)

        ax.set_ylabel("BOLD projection")
        if trial_indices_array is not None:
            label_start = int(trial_indices_array[start_idx]) + 1
            label_end = int(trial_indices_array[end_idx - 1]) + 1
        else:
            label_start = start_idx + 1
            label_end = end_idx
        ax.set_title(f"Trials {label_start}-{label_end}")
        ax.grid(True, linestyle="--", alpha=0.25)
        ax.legend(loc="upper right", fontsize=8, ncol=2)

    axes[-1].set_xlabel("Time point")
    if series_label:
        figure_title = f"{series_label} BOLD projection"
    else:
        figure_title = "BOLD projection"
    hyperparam_label = f"alpha_task={task_alpha:g}, alpha_bold={bold_alpha:g}, alpha_beta={beta_alpha:g}, gamma={gamma_value:g}"
    fig.suptitle(f"{figure_title}\n{hyperparam_label}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path

def plot_projection_beta_sweep(gamma_projection_series, task_alpha, bold_alpha, beta_alpha, output_path, series_label=None, trial_indices=None):
    output_path = _result_path(output_path, for_write=True)
    sorted_series = sorted(gamma_projection_series, key=lambda item: item[0])
    max_trials = max(np.asarray(series).ravel().size for _, series in sorted_series)
    fig_height = 5.0 if max_trials <= 120 else 6.5
    trial_indices_array = None
    if trial_indices is not None:
        trial_indices_array = np.asarray(trial_indices).ravel()
    axis_reference = None
    if trial_indices_array is not None and trial_indices_array.size:
        axis_reference = trial_indices_array[:min(trial_indices_array.size, max_trials)] + 1
    use_global_axis = axis_reference is not None
    cmap = plt.cm.get_cmap("tab10", len(sorted_series))

    fig, ax = plt.subplots(figsize=(10, fig_height))
    for idx, (gamma_value, projection) in enumerate(sorted_series):
        y_trials = np.asarray(projection).ravel()
        finite_vals = y_trials[np.isfinite(y_trials)]
        cv_val = np.nanstd(finite_vals) / abs(np.nanmean(finite_vals))
        if trial_indices_array is not None:
                axis_values = np.arange(y_trials.size)
                axis_values = trial_indices_array[: y_trials.size] + 1
        else:
            axis_values = np.arange(y_trials.size)
        variability_label = f"gamma={gamma_value:g} (cv={cv_val})"
        ax.plot(axis_values, y_trials, linewidth=1.2, alpha=0.9, label=variability_label, color=cmap(idx))

    axis_label = "Trial (global index)" if use_global_axis else "Trial index"
    ax.set_xlabel(axis_label)
    ax.set_ylabel("Voxel-space projection")
    label_suffix = f" [{series_label}]" if series_label else ""
    gamma_labels = ", ".join(f"{gamma:g}" for gamma, _ in sorted_series)
    ax.set_title(f"y_beta across gammas ({gamma_labels}) | alpha_task={task_alpha:g}, alpha_bold={bold_alpha:g}, alpha_beta={beta_alpha:g}{label_suffix}")
    if max_trials > 1:
        approx_ticks = 10
        if use_global_axis:
            tick_count = min(approx_ticks, axis_reference.size)
            tick_indices = np.linspace(0, axis_reference.size - 1, num=tick_count, dtype=int)
            xtick_positions = axis_reference[tick_indices]
            ax.set_xticks(xtick_positions)
        else:
            tick_step = max(1, max_trials // approx_ticks)
            xtick_positions = np.arange(0, max_trials, tick_step, dtype=int)
            if xtick_positions[-1] != max_trials - 1:
                xtick_positions = np.append(xtick_positions, max_trials - 1)
            ax.set_xticks(xtick_positions)
            ax.set_xticklabels([str(pos + 1) for pos in xtick_positions])
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best", fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path

def plot_fold_metric_box(fold_values, title, ylabel, output_path, highlight_threshold=None):
    output_path = _result_path(output_path, for_write=True)
    sorted_values = sorted(fold_values, key=lambda item: item[0])
    fold_ids = np.array([int(idx) for idx, _ in sorted_values])
    raw_values = np.array([float(val) if np.isfinite(val) else np.nan for _, val in sorted_values])
    finite_mask = np.isfinite(raw_values)
    finite_values = raw_values[finite_mask]
    finite_ids = fold_ids[finite_mask]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    box = ax.boxplot([finite_values], vert=True, patch_artist=True, widths=0.3,
                     boxprops={"facecolor": "#b5cde0", "color": "#4c72b0", "linewidth": 1.5},
                     medianprops={"color": "#c44e52", "linewidth": 2.0},
                     whiskerprops={"color": "#4c72b0"}, capprops={"color": "#4c72b0"},
                     flierprops={"marker": "o", "markerfacecolor": "#c44e52", "markeredgecolor": "#c44e52", "markersize": 5})
    x_center = 1.0
    if finite_values.size > 1:
        jitter = np.linspace(-0.07, 0.07, finite_values.size)
    else:
        jitter = np.array([0.0])
    scatter_positions = x_center + jitter
    ax.scatter(scatter_positions, finite_values, color="#4c72b0", alpha=0.7, s=25, zorder=3, label="Fold values")
    for fold_id, x_pos, y_val in zip(finite_ids, scatter_positions, finite_values):
        ax.text(x_pos, y_val, f"{fold_id}", fontsize=8, color="#1a1a1a", ha="center", va="bottom", rotation=0, clip_on=True)

    ax.set_xticks([x_center])
    ax.set_xticklabels([f"{finite_values.size} folds"])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    use_scalar_formatter = True
    if finite_values.size:
        max_abs = np.nanmax(np.abs(finite_values))
        if np.isfinite(max_abs) and max_abs < 0.1:
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.4f"))
            use_scalar_formatter = False

    if use_scalar_formatter:
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    if highlight_threshold is not None:
        ax.axhline(highlight_threshold, color="#c44e52", linestyle="--", linewidth=1.0, label=f"threshold={highlight_threshold:g}")
    ax.legend(loc="best")

    nan_count = np.count_nonzero(~finite_mask)
    if nan_count:
        ax.annotate(f"{nan_count} fold(s) skipped (NaN)", xy=(0.02, 0.95), xycoords="axes fraction",fontsize=8, color="#c44e52", ha="left", va="top")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path

def plot_train_test_total_loss_box(train_values, test_values, title, ylabel, output_path):
    output_path = _result_path(output_path, for_write=True)
    def _prepare(entries):
        sorted_entries = sorted(entries, key=lambda item: item[0])
        fold_ids = np.array([int(idx) for idx, _ in sorted_entries])
        raw_values = np.array([float(val) if np.isfinite(val) else np.nan for _, val in sorted_entries])
        finite_mask = np.isfinite(raw_values)
        return fold_ids[finite_mask], raw_values[finite_mask], np.count_nonzero(~finite_mask)

    train_ids, train_vals, train_nan = _prepare(train_values)
    test_ids, test_vals, test_nan = _prepare(test_values)

    plot_data = []
    positions = []
    labels = []
    colors = []
    scatter_data = []
    position_cursor = 1

    if train_vals.size:
        plot_data.append(train_vals)
        positions.append(position_cursor)
        labels.append(f"Train ({train_vals.size} folds)")
        colors.append("#4c72b0")
        scatter_data.append((position_cursor, train_ids, train_vals, "#4c72b0"))
        position_cursor += 1

    if test_vals.size:
        plot_data.append(test_vals)
        positions.append(position_cursor)
        labels.append(f"Test ({test_vals.size} folds)")
        colors.append("#dd8452")
        scatter_data.append((position_cursor, test_ids, test_vals, "#dd8452"))

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    box = ax.boxplot(plot_data, positions=positions, vert=True, patch_artist=True, widths=0.35)

    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor("#1a1a1a")
    for median in box["medians"]:
        median.set_color("#1a1a1a")
        median.set_linewidth(2.0)
    for whisker in box["whiskers"]:
        whisker.set_color("#1a1a1a")
    for cap in box["caps"]:
        cap.set_color("#1a1a1a")
    for flier, color in zip(box["fliers"], colors):
        flier.set_markerfacecolor(color)
        flier.set_markeredgecolor(color)

    for x_pos, fold_ids, values, color in scatter_data:
        if values.size > 1:
            jitter = np.linspace(-0.07, 0.07, values.size)
        else:
            jitter = np.array([0.0])
        scatter_positions = x_pos + jitter
        ax.scatter(scatter_positions, values, color=color, alpha=0.75, s=25, zorder=3)
        for fold_id, x_coord, y_val in zip(fold_ids, scatter_positions, values):
            ax.text(x_coord, y_val, f"{fold_id}", fontsize=8, color="#1a1a1a",
                    ha="center", va="bottom", rotation=0, clip_on=True)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    all_values = np.concatenate(plot_data) if plot_data else np.array([])
    use_scalar_formatter = True
    if all_values.size:
        max_abs = np.nanmax(np.abs(all_values))
        if np.isfinite(max_abs) and max_abs < 0.1:
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.4f"))
            use_scalar_formatter = False
    if use_scalar_formatter:
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)

    skipped = train_nan + test_nan
    if skipped:
        ax.annotate(f"{skipped} value(s) skipped (NaN)", xy=(0.02, 0.95), xycoords="axes fraction",
                    fontsize=8, color="#c44e52", ha="left", va="top")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path

def save_projection_outputs(pca_weights, bold_pca_components, trial_length, file_prefix, task_alpha, bold_alpha, beta_alpha, gamma_value, voxel_weights, beta_clean, data=None, bold_projection=None, plot_trials=True):
    bold_projection_signal, _ = _compute_bold_projection(voxel_weights, data)
    num_trials = bold_projection_signal.size // trial_length
    y_trials = bold_projection_signal.reshape(num_trials, trial_length)
    trial_indices = None
    if data is not None:
        trial_indices = data.get("trial_indices")
    if plot_trials:
        plot_path = f"y_projection_trials_{file_prefix}.png"
        plot_projection_bold(y_trials, task_alpha, bold_alpha, beta_alpha, gamma_value, plot_path, series_label="Active BOLD space", trial_indices_array=trial_indices)

    voxel_weights = voxel_weights.ravel()
    beta_matrix = np.nan_to_num(beta_clean, nan=0.0, posinf=0.0, neginf=0.0)
    y_projection_voxel_trials = voxel_weights @ beta_matrix
    y_projection_voxel = y_projection_voxel_trials.ravel()
    return y_projection_voxel


# =============================================================================
# Experiment configuration and main flow
# =============================================================================

ridge_penalty = 1e-3
solver_name = "MOSEK"

def _dedupe_penalty_tuples(candidates, decimals=6):
    seen = set()
    deduped = []
    for candidate in candidates:
        values = tuple(float(value) for value in candidate)
        if len(values) != 5:
            raise RuntimeError(f"Penalty candidate must have 5 values, got {len(values)}: {candidate}")
        key = tuple(round(value, decimals) for value in values)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(values)
    return deduped

def _build_default_coarse_penalty_sweep():
    # The penalty matrices are trace-normalized, so these are relative weights around a neutral start.
    candidates = [
        (1.0, 1.0, 1.0, 1.0, 1.0),      # balanced default for a new dataset
        (0.8, 1.0, 0.5, 1.2, 1.0),      # previous subject-specific setting, kept as a reference
        (0.7, 0.7, 0.7, 0.7, 1.0),      # weaker overall regularization
        (1.3, 1.3, 1.3, 1.3, 1.0),      # stronger overall regularization
        (0.5, 1.0, 1.0, 1.0, 1.0),      # weaker task penalty
        (1.5, 1.0, 1.0, 1.0, 1.0),      # stronger task penalty
        (2, 1.0, 1.0, 1.0, 1.0),      # stronger task penalty
        (1.0, 0.6, 0.6, 1.0, 1.0),      # weaker transition penalties
        (1.0, 1.4, 1.4, 1.0, 1.0),      # stronger transition penalties
        (1.0, 1.0, 0.5, 1.0, 1.0),      # weaker beta transition penalty
        (1.0, 1.0, 1.0, 0.6, 1.0),      # weaker smoothness penalty
        (1.0, 1.0, 1.0, 1.8, 1.0),      # stronger smoothness penalty
    ]
    return _dedupe_penalty_tuples(candidates)

def _penalty_tuple_from_config(combo):
    if isinstance(combo, dict):
        values = (
            combo["task_penalty"],
            combo["bold_penalty"],
            combo["beta_penalty"],
            combo["smooth_penalty"],
            combo.get("corr_weight", 1.0),
        )
    else:
        values = combo
    return tuple(float(value) for value in values)

def _load_penalty_sweep(default_candidates):
    penalty_sweep_env = os.environ.get("FMRI_PENALTY_SWEEP_JSON", "").strip()
    if not penalty_sweep_env:
        return list(default_candidates)
    parsed_penalties = json.loads(penalty_sweep_env)
    candidates = [_penalty_tuple_from_config(combo) for combo in parsed_penalties]
    candidates = _dedupe_penalty_tuples(candidates)
    print(f"Using penalty_sweep from FMRI_PENALTY_SWEEP_JSON with {len(candidates)} candidates.", flush=True)
    return candidates

def _alpha_settings_from_penalty_sweep(penalty_candidates):
    return [{"task_penalty": task_alpha, "bold_penalty": bold_alpha, "beta_penalty": beta_alpha,
        "smooth_penalty": smooth_alpha, "corr_weight": corr_weight}
        for task_alpha, bold_alpha, beta_alpha, smooth_alpha, corr_weight in penalty_candidates]

def _load_gamma_sweep(default_values):
    gamma_sweep_env = os.environ.get("FMRI_GAMMA_SWEEP_JSON", "").strip()
    if not gamma_sweep_env:
        return list(default_values)
    values = [float(value) for value in json.loads(gamma_sweep_env)]
    print(f"Using gamma_sweep from FMRI_GAMMA_SWEEP_JSON: {values}", flush=True)
    return values

def resolve_penalty_search_mode(default_mode="coarse_to_fine"):
    mode = os.environ.get("FMRI_PENALTY_SEARCH_MODE", default_mode).strip().lower().replace("-", "_")
    if mode in ("0", "false", "off", "manual", "none"):
        return "manual"
    if mode in ("1", "true", "on", "auto", "coarse", "coarse_to_fine"):
        return "coarse_to_fine"
    raise RuntimeError(f"Unknown FMRI_PENALTY_SEARCH_MODE={mode!r}. Use 'coarse_to_fine' or 'manual'.")

penalty_sweep = _load_penalty_sweep(_build_default_coarse_penalty_sweep())
alpha_sweep = _alpha_settings_from_penalty_sweep(penalty_sweep)
gamma_sweep = _load_gamma_sweep([1.5])
# SAVE_PER_FOLD_VOXEL_MAPS = False  # disable individual fold voxel-weight plots; averages saved later

def build_projection_data():
    projection_data = build_pca_dataset(bold_clean, beta_clean, behavior_matrix, nan_mask_flat, active_coords, active_flat_idx, trial_len, num_trials)
    projection_data["session_labels"] = session_labels
    projection_data["session_metadata"] = session_metadata
    coeff_pinv = projection_data["coeff_pinv"]
    projected = coeff_pinv @ (laplacian_matrix @ coeff_pinv.T)
    projection_data["C_smooth"] = standardize_matrix(projected)
    globals().update({"projection_data": projection_data, "coeff_pinv": coeff_pinv, "projected": projected})
    return projection_data

def _describe_trial_span(indices):
    indices = np.asarray(indices).ravel()
    if indices.size == 0:
        return None
    return (int(indices[0] + 1), int(indices[-1] + 1))

def _save_fold_metric_plots(fold_metric_records, metric_plot_configs):
    if not fold_metric_records:
        return
    print("\n===== Saving fold-wise metric box plots =====", flush=True)
    for metrics_key in sorted(fold_metric_records.keys()):
        task_alpha, bold_alpha, beta_alpha, smooth_alpha, gamma_value, corr_weight = metrics_key
        combo_label = (f"task={task_alpha:g}, bold={bold_alpha:g}, beta={beta_alpha:g}, smooth={smooth_alpha:g}, gamma={gamma_value:g}")
        metrics_prefix = (f"foldmetrics_sub{sub}_ses{ses}_task{task_alpha:g}_bold{bold_alpha:g}_beta{beta_alpha:g}_smooth{smooth_alpha:g}_gamma{gamma_value:g}")
        
        metric_records = fold_metric_records[metrics_key]
        train_loss_entries = metric_records.get("train_total_loss", [])
        test_loss_entries = metric_records.get("test_total_loss", [])
        for metric_name, entries in metric_records.items():
            if metric_name in ("train_total_loss", "test_total_loss", "train_gamma_ratio", "test_gamma_ratio"):
                continue
            config = metric_plot_configs.get(metric_name)
            plot_title = f"{config['title']} across folds ({combo_label})"
            plot_path = f"{metric_name}_{metrics_prefix}.png"
            created_path = plot_fold_metric_box(entries, plot_title, config["ylabel"], plot_path, highlight_threshold=config.get("threshold"))
            if created_path:
                print(f"  Saved {metric_name} fold plot for {combo_label} -> {created_path}", flush=True)
        if train_loss_entries or test_loss_entries:
            loss_title = f"Total loss across folds ({combo_label})"
            loss_path = f"total_loss_{metrics_prefix}_train_vs_test.png"
            created_loss_plot = plot_train_test_total_loss_box(train_loss_entries, test_loss_entries, loss_title, "Objective value", loss_path)
            if created_loss_plot:
                print(f"  Saved train/test total loss fold plot for {combo_label} -> {created_loss_plot}", flush=True)

        train_gamma_entries = metric_records.get("train_gamma_ratio", [])
        test_gamma_entries = metric_records.get("test_gamma_ratio", [])
        if train_gamma_entries or test_gamma_entries:
            gamma_title = f"Gamma-penalty ratio across folds ({combo_label})"
            gamma_path = f"gamma_penalty_ratio_{metrics_prefix}_train_vs_test.png"
            created_gamma_plot = plot_train_test_total_loss_box(train_gamma_entries, test_gamma_entries, gamma_title, "gamma * penalty / corr_den", gamma_path)
            if created_gamma_plot:
                print(f"  Saved gamma-penalty ratio fold plot for {combo_label} -> {created_gamma_plot}", flush=True)

def _atlas_output_options():
    atlas_analysis_enabled = os.environ.get("FMRI_ATLAS_ANALYSIS", "1").strip().lower() not in ("0", "false", "no")
    atlas_threshold = int(os.environ.get("FMRI_ATLAS_THRESHOLD", "25"))
    motor_label_patterns = os.environ.get("FMRI_MOTOR_LABEL_PATTERNS", "precentral,supplementary motor,cerebellum")
    motor_label_patterns = [p.strip() for p in motor_label_patterns.split(",") if p.strip()]
    atlas_assume_mni = os.environ.get("FMRI_ATLAS_ASSUME_MNI", "0").strip().lower() in ("1", "true", "yes")
    atlas_data_dir = os.environ.get("FMRI_ATLAS_DIR", "").strip()
    if atlas_data_dir:
        atlas_data_dir = _result_path(atlas_data_dir, for_write=True)
    else:
        atlas_data_dir = _result_path("atlas_cache", for_write=True)
    if atlas_analysis_enabled:
        os.makedirs(atlas_data_dir, exist_ok=True)
    return {"enabled": atlas_analysis_enabled, "threshold": atlas_threshold, "motor_label_patterns": motor_label_patterns,
            "assume_mni": atlas_assume_mni, "data_dir": atlas_data_dir}

def _save_fold_average_outputs(fold_output_tracker, projection_data):
    if not fold_output_tracker:
        return
    print("\n===== Saving fold-averaged spatial maps and projections =====", flush=True)
    active_coords = projection_data.get("active_coords")
    volume_shape = anat_img.shape[:3]
    fold_avg_projection_series = defaultdict(list)
    atlas_context = None
    atlas_options = _atlas_output_options()

    for metrics_key in sorted(fold_output_tracker.keys()):
        task_alpha, bold_alpha, beta_alpha, smooth_alpha, gamma_value, corr_weight = metrics_key
        fold_outputs = fold_output_tracker[metrics_key]
        avg_prefix = (f"foldavg_sub{sub}_ses{ses}_task{task_alpha:g}_bold{bold_alpha:g}_beta{beta_alpha:g}_smooth{smooth_alpha:g}_gamma{gamma_value:g}")

        # comp_weights_stack = np.stack(fold_outputs["component_weights"], axis=0)
        # comp_weights_icc = _compute_matrix_icc(comp_weights_stack)
        # print(f"  ICC (component weights across folds/components): {comp_weights_icc:.4f}" if np.isfinite(comp_weights_icc) else "  ICC (component weights) could not be computed", flush=True)

        weights_stack = np.stack(fold_outputs["voxel_weights"], axis=0)
        weights_avg = np.nanmean(weights_stack, axis=0)
        weights_icc = _compute_matrix_icc(weights_stack)
        print(f"  ICC (weights across folds/voxels): {weights_icc:.4f}" if np.isfinite(weights_icc) else "  ICC (weights) could not be computed", flush=True)
        weights_title = f"ICC={weights_icc:.3f}"
        save_brain_map(weights_avg * 1000, active_coords, volume_shape, anat_img, avg_prefix, result_prefix="voxel_weights_mean", map_title=weights_title,
            display_threshold_ratio=0.8)
        voxel_weights_path = _result_path(f"voxel_weights_mean_{avg_prefix}.nii.gz")
        if os.path.exists(voxel_weights_path):
            enhance_bold_visualization(voxel_weights_path, anat_img=anat_img, map_title=weights_title)
            if atlas_options["enabled"]:
                weights_img = nib.load(voxel_weights_path)
                if (atlas_context is None or atlas_context.get("shape") != weights_img.shape[:3]
                    or not np.allclose(atlas_context.get("affine"), weights_img.affine)):
                    atlas_context = _prepare_atlas_context(anat_img, anat_path, weights_img, RESULTS_DIR, atlas_threshold=atlas_options["threshold"],
                                                           data_dir=atlas_options["data_dir"], assume_mni=atlas_options["assume_mni"])
                output_prefix = f"voxel_weights_mean_{avg_prefix}"
                _analyze_weight_map_regions(voxel_weights_path, atlas_context, output_prefix, atlas_options["motor_label_patterns"], threshold_percentile=95)
        else:
            print(f"  WARNING: Expected voxel weights map not found: {voxel_weights_path}", flush=True)

        bold_corr_stack = np.stack(np.abs(fold_outputs["bold_corr"]), axis=0)
        bold_corr_avg = np.nanmean(bold_corr_stack, axis=0)
        bold_corr_icc = _compute_matrix_icc(bold_corr_stack)
        print(f"  ICC (bold corr across folds/voxels): {bold_corr_icc:.4f}" if np.isfinite(bold_corr_icc) else "  ICC (bold corr) could not be computed", flush=True)
        bold_title = f"ICC={bold_corr_icc:.3f}"
        save_brain_map(bold_corr_avg, active_coords, volume_shape, anat_img, avg_prefix, result_prefix="active_bold_corr_mean", map_title=bold_title,
                       display_threshold_ratio=0.8)

        beta_corr_stack = np.stack(np.abs(fold_outputs["beta_corr"]), axis=0)
        beta_corr_avg = np.nanmean(beta_corr_stack, axis=0)
        beta_corr_icc = _compute_matrix_icc(beta_corr_stack)
        print(f"  ICC (beta corr across folds/voxels): {beta_corr_icc:.4f}" if np.isfinite(beta_corr_icc) else "  ICC (beta corr) could not be computed", flush=True)
        beta_title = f"ICC={beta_corr_icc:.3f}"
        save_brain_map(beta_corr_avg, active_coords, volume_shape, anat_img, avg_prefix, result_prefix=f"active_beta_corr_mean", map_title=beta_title,
                       display_threshold_ratio=0.8)

        projection_avg = _compute_projection(weights_avg, projection_data["beta_clean"])
        avg_series_key = (task_alpha, bold_alpha, beta_alpha, smooth_alpha, corr_weight)
        fold_avg_projection_series[avg_series_key].append((gamma_value, projection_avg))

        bold_clean_full = projection_data.get("bold_clean")
        if bold_clean_full is not None:
            bold_projection_signal, _ = _compute_bold_projection(weights_avg, {"active_bold": bold_clean_full})
            num_trials = bold_projection_signal.size // trial_len
            bold_projection_trials= bold_projection_signal.reshape(num_trials, trial_len)
            bold_plot_path = f"y_projection_bold_{avg_prefix}.png"
            plot_projection_bold(bold_projection_trials, task_alpha, bold_alpha, beta_alpha, gamma_value, bold_plot_path, 
                                 series_label="Active BOLD space (weights avg)", trial_indices_array=np.arange(num_trials))

    if fold_avg_projection_series:
        for avg_series_key in sorted(fold_avg_projection_series.keys()):
            task_alpha, bold_alpha, beta_alpha, smooth_alpha, corr_weight = avg_series_key
            gamma_series = fold_avg_projection_series[avg_series_key]
            avg_base_prefix = (f"foldavg_sub{sub}_ses{ses}_task{task_alpha:g}_bold{bold_alpha:g}_beta{beta_alpha:g}_smooth{smooth_alpha:g}")
            avg_series_storage = _result_path(f"gamma_series_voxel_{avg_base_prefix}.npy", for_write=True)
            existing_avg = _load_projection_series(avg_series_storage)
            merged_avg = _merge_projection_series(existing_avg, gamma_series)
            np.save(avg_series_storage, merged_avg, allow_pickle=True)
            aggregate_plot_path = f"y_projection_trials_voxel_{avg_base_prefix}_all_gammas.png"
            merged_avg_sorted = sorted(merged_avg.items(), key=lambda item: item[0])
            plot_projection_beta_sweep(merged_avg_sorted, task_alpha, bold_alpha, beta_alpha, aggregate_plot_path, series_label="Voxel space (fold avg)")

def _print_aggregate_metrics(aggregate_metrics):
    if not aggregate_metrics:
        return
    print("\n===== Cross-fold average metrics =====", flush=True)
    for metrics_key in sorted(aggregate_metrics.keys()):
        task_alpha, bold_alpha, beta_alpha, smooth_alpha, gamma_value, corr_weight = metrics_key
        bucket = aggregate_metrics[metrics_key]

        train_corr_mean = np.nanmean(np.abs(bucket["train_corr"]))
        train_p_mean = np.nanmean(bucket["train_p"])
        test_corr_mean = np.nanmean(np.abs(bucket["test_corr"]))
        test_p_mean = np.nanmean(bucket["test_p"])
        train_loss_mean = np.nanmean(bucket["train_total_loss"])
        test_loss_mean = np.nanmean(bucket["test_total_loss"])
        train_gamma_ratio_mean = np.nanmean(bucket["train_gamma_ratio"])
        test_gamma_ratio_mean = np.nanmean(bucket["test_gamma_ratio"])
        fold_count = len(bucket["test_corr"])

        combo_label = (f"task={task_alpha:g}\n bold={bold_alpha:g}\n beta={beta_alpha:g}\n smooth={smooth_alpha:g}\n gamma={gamma_value:g}")
        print(f"task={task_alpha:g}, bold={bold_alpha:g}, beta={beta_alpha:g}, smooth={smooth_alpha:g}, gamma={gamma_value:g}"
            f"(folds contributing={fold_count}): "
            f"train corr={train_corr_mean:.4f} (p={train_p_mean:.4f}), "
            f"test corr={test_corr_mean:.4f} (p={test_p_mean:.4f}), "
            f"avg loss train={train_loss_mean:.4f}, test={test_loss_mean:.4f}, "
            f"gamma_ratio train={train_gamma_ratio_mean:.4f}, test={test_gamma_ratio_mean:.4f}", flush=True)

def run_cross_run_experiment(alpha_settings, gamma_values, fold_splits, projection_data,
                             save_artifacts=True, compute_spatial_outputs=True,
                             append_log=True, run_label=None):
    total_folds = len(fold_splits)
    bold_pca_components = projection_data["pca_model"].eigvec if compute_spatial_outputs else None
    aggregate_metrics = defaultdict(lambda: {"train_corr": [], "train_p": [], "test_corr": [], "test_p": [],
                                             "train_total_loss": [], "test_total_loss": [], "train_gamma_ratio": [], "test_gamma_ratio": []})
    fold_metric_records = defaultdict(lambda: defaultdict(list))
    fold_output_tracker = defaultdict(lambda: {"component_weights": [], "voxel_weights": [], "bold_corr": [], "beta_corr": []})
    metric_plot_configs = {"train_corr": {"title": "Train correlation", "ylabel": "Corr"}, "test_corr": {"title": "Test correlation", "ylabel": "Corr"},
                           "train_p": {"title": "Train correlation p-value", "ylabel": "p-value", "threshold": 0.05},
                           "test_p": {"title": "Test correlation p-value", "ylabel": "p-value", "threshold": 0.05}}
    if run_label:
        print(f"\n===== Parameter run: {run_label} ({len(alpha_settings)} alpha candidates, {len(gamma_values)} gamma values) =====", flush=True)

    for fold_idx, split in enumerate(fold_splits, start=1):
        print(f"\n===== Fold {fold_idx}/{total_folds} =====", flush=True)
        test_indices, train_indices = split["test_indices"], split["train_indices"]
        test_desc = _describe_trial_span(test_indices)
        test_text = f"{test_desc[0]}-{test_desc[1]}" if test_desc else "n/a"
        print(f"Test trials (1-based): {test_text}", flush=True)
        normalization_info = compute_fold_normalization(projection_data, train_indices, behave_indice)
        train_data = prepare_data_func(projection_data, train_indices, trial_len, transition_boundaries, normalization_info)
        test_data = prepare_data_func(projection_data, test_indices, trial_len, transition_boundaries, normalization_info)

        train_corr_norms = {"num": _matrix_norm_summary(train_data["C_corr_num"]), "den": _matrix_norm_summary(train_data["C_corr_den"])}
        test_corr_norms = {"num": _matrix_norm_summary(test_data["C_corr_num"]), "den": _matrix_norm_summary(test_data["C_corr_den"])}
        print(f"  Corr matrix norms (train) num_fro={train_corr_norms['num']['fro']:.4f}, den_fro={train_corr_norms['den']['fro']:.4f}; "
            f"(test) num_fro={test_corr_norms['num']['fro']:.4f}, den_fro={test_corr_norms['den']['fro']:.4f}", flush=True)

        for alpha_setting in alpha_settings:
            task_alpha, bold_alpha, beta_alpha, smooth_alpha = alpha_setting["task_penalty"], alpha_setting["bold_penalty"], alpha_setting["beta_penalty"], alpha_setting["smooth_penalty"]
            corr_weight = float(alpha_setting.get("corr_weight", 1.0))
            alpha_prefix = f"fold{fold_idx}_sub{sub}_ses{ses}_task{task_alpha:g}_bold{bold_alpha:g}_beta{beta_alpha:g}_smooth{smooth_alpha:g}"

            for gamma_value in gamma_values:
                metrics_key = (task_alpha, bold_alpha, beta_alpha, smooth_alpha, gamma_value, corr_weight)
                file_prefix = f"{alpha_prefix}_gamma{gamma_value:g}"
                print(f"Fold {fold_idx}: optimization with task={task_alpha}, bold={bold_alpha}, beta={beta_alpha}, "
                      f"smooth={smooth_alpha}, gamma={gamma_value}", flush=True)
                solution = solve_soc_problem(train_data, task_alpha, bold_alpha, beta_alpha, smooth_alpha, gamma_value, corr_weight=corr_weight)
                numerator_value = solution.get("numerator")
                denominator_value = solution.get("denominator")
                print(f"  Objective terms -> numerator: {numerator_value:.6f}, denominator: {denominator_value:.6f}", flush=True)

                penalty_contributions = solution.get("penalty_contributions", {})
                loss_report = (f"C_task: {penalty_contributions.get('task')}, " f"C_bold: {penalty_contributions.get('bold')}, "
                               f"C_beta: {penalty_contributions.get('beta')}, " f"C_smooth: {penalty_contributions.get('smooth')}")
                print(f"  Loss terms -> {loss_report}", flush=True)

                # Preserve the optimized weight signs so correlation metrics match the optimized objective
                component_weights = np.asarray(solution["weights"])
                voxel_weights = None

                if compute_spatial_outputs:
                    coeff_pinv = np.asarray(train_data["coeff_pinv"])
                    voxel_weights = coeff_pinv.T @ component_weights

                    projection_signal, voxel_correlations = evaluate_bold_bold_projection_corr(voxel_weights, train_data)
                    beta_projection_signal, beta_voxel_correlations = evaluate_beta_bold_projection_corr(voxel_weights, train_data)

                    save_projection_outputs(component_weights, bold_pca_components, trial_len, file_prefix,
                                            task_alpha, bold_alpha, beta_alpha, gamma_value,
                                            voxel_weights=voxel_weights, beta_clean=train_data["beta_clean"],
                                            data=train_data, bold_projection=projection_signal, plot_trials=False)
                    fold_outputs = fold_output_tracker[metrics_key]
                    fold_outputs["component_weights"].append(np.abs(component_weights))
                    fold_outputs["voxel_weights"].append(np.abs(voxel_weights))
                    fold_outputs["bold_corr"].append(np.abs(voxel_correlations))
                    fold_outputs["beta_corr"].append(np.abs(beta_voxel_correlations))

                train_metrics = evaluate_projection_corr(train_data, component_weights)
                test_metrics = evaluate_projection_corr(test_data, component_weights)

                train_penalties, total_penalty_train = calcu_penalty_terms(train_data, task_alpha, bold_alpha, beta_alpha, smooth_alpha)
                train_A, train_B = _build_objective_matrices(total_penalty_train, train_data["C_corr_num"], train_data["C_corr_den"], gamma_value, corr_weight=corr_weight)
                train_total_loss = solution.get("total_loss")
                train_numerator = float(component_weights.T @ train_A @ component_weights)
                train_denominator = float(component_weights.T @ train_B @ component_weights)
                train_corr_num_quad = float(component_weights.T @ train_data["C_corr_num"] @ component_weights)
                train_gamma_penalty = float(gamma_value * (component_weights.T @ total_penalty_train @ component_weights))
                train_gamma_ratio = train_gamma_penalty / train_denominator if np.isfinite(train_denominator) and train_denominator > 0 else np.nan
                train_corr_ratio = train_corr_num_quad / train_denominator if np.isfinite(train_denominator) and train_denominator > 0 else np.nan

                test_penalties, total_penalty_test = calcu_penalty_terms(test_data, task_alpha, bold_alpha, beta_alpha, smooth_alpha)
                test_penalty_terms = test_penalties if save_artifacts else None
                test_total_loss = _total_loss_from_penalty(solution["weights"], total_penalty_test, gamma_value, corr_weight=corr_weight, corr_num=test_data["C_corr_num"], corr_den=test_data["C_corr_den"], penalty_terms=test_penalty_terms, label="test")
                test_A, test_B = _build_objective_matrices(total_penalty_test, test_data["C_corr_num"], test_data["C_corr_den"], gamma_value, corr_weight=corr_weight)
                test_numerator = float(component_weights.T @ test_A @ component_weights)
                test_denominator = float(component_weights.T @ test_B @ component_weights)
                test_corr_num_quad = float(component_weights.T @ test_data["C_corr_num"] @ component_weights)
                test_gamma_penalty = float(gamma_value * (component_weights.T @ total_penalty_test @ component_weights))
                test_gamma_ratio = test_gamma_penalty / test_denominator if np.isfinite(test_denominator) and test_denominator > 0 else np.nan
                test_corr_ratio = test_corr_num_quad / test_denominator if np.isfinite(test_denominator) and test_denominator > 0 else np.nan

                train_metrics["total_loss"] = train_total_loss
                test_metrics["total_loss"] = test_total_loss

                print(f"  Total loss (train objective): {train_total_loss:.6f}", flush=True)
                print(f"  Total loss (test objective):  {test_total_loss:.6f}", flush=True)
                print(f"  Gamma-penalty ratio -> train: {train_gamma_ratio:.6f}, test: {test_gamma_ratio:.6f}", flush=True)
                print(f"  Train metrics -> corr: {train_metrics['pearson']:.4f}", flush=True)
                print(f"  Test metrics  -> corr: {test_metrics['pearson']:.4f},", flush=True)

                if append_log:
                    input_stats = {"beta_clean": _array_summary(train_data["beta_clean"]),
                        "active_bold": _array_summary(train_data["active_bold"]),
                        "behavior_observed": _array_summary(train_data["behavior_observed"]),
                        "beta_centered": _array_summary(train_data["beta_centered"]),
                        "normalized_behaviors": _array_summary(train_data["normalized_behaviors"])}
                    weight_stats = {"component_weights": _array_summary(component_weights)}
                    if voxel_weights is not None:
                        weight_stats["voxel_weights"] = _array_summary(voxel_weights)
                    log_entry = {"fold": fold_idx,
                        "test_span": test_desc,
                        "alphas": {"task": task_alpha, "bold": bold_alpha, "beta": beta_alpha, "smooth": smooth_alpha},
                        "gamma": gamma_value,
                        "corr_weight": corr_weight,
                        "numerator": numerator_value,
                        "denominator": denominator_value,
                        "fractional_objective": solution.get("fractional_objective"),
                        "gamma_penalty": solution.get("gamma_penalty"),
                        "gamma_penalty_ratio": solution.get("gamma_penalty_ratio"),
                        "train_total_loss": train_total_loss,
                        "test_total_loss": test_total_loss,
                        "train_gamma_ratio": train_gamma_ratio,
                        "test_gamma_ratio": test_gamma_ratio,
                        "train_corr": float(train_metrics.get("pearson", np.nan)),
                        "test_corr": float(test_metrics.get("pearson", np.nan)),
                        "train_numerator": train_numerator,
                        "train_denominator": train_denominator,
                        "train_corr_ratio": train_corr_ratio,
                        "test_numerator": test_numerator,
                        "test_denominator": test_denominator,
                        "test_corr_ratio": test_corr_ratio,
                        "train_gamma_penalty": train_gamma_penalty,
                        "test_gamma_penalty": test_gamma_penalty,
                        "train_corr_num_quad": train_corr_num_quad,
                        "test_corr_num_quad": test_corr_num_quad,
                        "penalty_contributions": {k: float(v) for k, v in penalty_contributions.items()},
                        "train_corr_norms": train_corr_norms,
                        "test_corr_norms": test_corr_norms,
                        "input_stats": input_stats,
                        "weight_stats": weight_stats}
                    if run_label:
                        log_entry["run_label"] = run_label
                    _append_run_log(log_entry)

                metrics_key = (task_alpha, bold_alpha, beta_alpha, smooth_alpha, gamma_value, corr_weight)
                bucket = aggregate_metrics[metrics_key]
                bucket["train_corr"].append(np.abs(train_metrics["pearson"]))
                bucket["train_p"].append(train_metrics["pearson_p"])
                bucket["test_corr"].append(np.abs(test_metrics["pearson"]))
                bucket["test_p"].append(test_metrics["pearson_p"])
                bucket["train_total_loss"].append(train_total_loss)
                bucket["test_total_loss"].append(test_total_loss)
                bucket["train_gamma_ratio"].append(train_gamma_ratio)
                bucket["test_gamma_ratio"].append(test_gamma_ratio)

                fold_metrics = fold_metric_records[metrics_key]
                fold_metrics["train_corr"].append((fold_idx, np.abs(train_metrics["pearson"])))
                fold_metrics["train_p"].append((fold_idx, train_metrics["pearson_p"]))
                fold_metrics["test_corr"].append((fold_idx, np.abs(test_metrics["pearson"])))
                fold_metrics["test_p"].append((fold_idx, test_metrics["pearson_p"]))
                fold_metrics["train_total_loss"].append((fold_idx, train_total_loss))
                fold_metrics["test_total_loss"].append((fold_idx, test_total_loss))
                fold_metrics["train_gamma_ratio"].append((fold_idx, train_gamma_ratio))
                fold_metrics["test_gamma_ratio"].append((fold_idx, test_gamma_ratio))

    if save_artifacts:
        _save_fold_metric_plots(fold_metric_records, metric_plot_configs)
    if save_artifacts and compute_spatial_outputs:
        _save_fold_average_outputs(fold_output_tracker, projection_data)
    _print_aggregate_metrics(aggregate_metrics)
    return aggregate_metrics

def _finite_float_array(values):
    arr = np.asarray(values, dtype=np.float64).ravel()
    return arr[np.isfinite(arr)]

def _finite_mean(values):
    arr = _finite_float_array(values)
    return float(np.mean(arr)) if arr.size else np.nan

def _parameter_balance_distance(values):
    distance = 0.0
    for value in values:
        value = float(value)
        if value <= 0 or not np.isfinite(value):
            distance += 10.0
        else:
            distance += abs(math.log(value))
    return float(distance)

def summarize_aggregate_metrics(aggregate_metrics):
    summaries = []
    for metrics_key in sorted(aggregate_metrics.keys()):
        task_alpha, bold_alpha, beta_alpha, smooth_alpha, gamma_value, corr_weight = metrics_key
        bucket = aggregate_metrics[metrics_key]
        train_corr = _finite_float_array(bucket["train_corr"])
        test_corr = _finite_float_array(bucket["test_corr"])
        test_corr_mean = float(np.mean(test_corr)) if test_corr.size else np.nan
        test_corr_std = float(np.std(test_corr)) if test_corr.size else np.nan
        test_corr_se = float(test_corr_std / np.sqrt(test_corr.size)) if test_corr.size else np.nan
        train_corr_mean = float(np.mean(train_corr)) if train_corr.size else np.nan
        train_test_gap = train_corr_mean - test_corr_mean if np.isfinite(train_corr_mean) and np.isfinite(test_corr_mean) else np.nan
        positive_gap = max(0.0, train_test_gap) if np.isfinite(train_test_gap) else np.nan
        selection_score = test_corr_mean
        if np.isfinite(test_corr_se):
            selection_score -= 0.5 * test_corr_se
        if np.isfinite(positive_gap):
            selection_score -= 0.25 * positive_gap
        balance_distance = _parameter_balance_distance((task_alpha, bold_alpha, beta_alpha, smooth_alpha))
        summaries.append({
            "task_penalty": float(task_alpha),
            "bold_penalty": float(bold_alpha),
            "beta_penalty": float(beta_alpha),
            "smooth_penalty": float(smooth_alpha),
            "gamma": float(gamma_value),
            "corr_weight": float(corr_weight),
            "fold_count": int(test_corr.size),
            "train_corr_mean": train_corr_mean,
            "test_corr_mean": test_corr_mean,
            "test_corr_std": test_corr_std,
            "test_corr_se": test_corr_se,
            "train_test_gap": float(train_test_gap) if np.isfinite(train_test_gap) else np.nan,
            "train_total_loss_mean": _finite_mean(bucket["train_total_loss"]),
            "test_total_loss_mean": _finite_mean(bucket["test_total_loss"]),
            "train_gamma_ratio_mean": _finite_mean(bucket["train_gamma_ratio"]),
            "test_gamma_ratio_mean": _finite_mean(bucket["test_gamma_ratio"]),
            "selection_score": float(selection_score) if np.isfinite(selection_score) else np.nan,
            "balance_distance": balance_distance,
        })
    return summaries

def select_best_parameter_summary(summaries):
    valid = [summary for summary in summaries
             if summary.get("fold_count", 0) > 0 and np.isfinite(summary.get("test_corr_mean", np.nan))]
    if not valid:
        raise RuntimeError("No valid penalty candidate produced finite test correlations.")
    best_by_corr = max(valid, key=lambda item: item["test_corr_mean"])
    tolerance = best_by_corr.get("test_corr_se", np.nan)
    if not np.isfinite(tolerance):
        tolerance = 0.0
    threshold = best_by_corr["test_corr_mean"] - tolerance
    eligible = [summary for summary in valid if summary["test_corr_mean"] >= threshold]

    def selection_key(summary):
        score = summary.get("selection_score", np.nan)
        test_std = summary.get("test_corr_std", np.nan)
        gap = summary.get("train_test_gap", np.nan)
        balance = summary.get("balance_distance", np.nan)
        return (
            score if np.isfinite(score) else -np.inf,
            -(test_std if np.isfinite(test_std) else np.inf),
            -(max(0.0, gap) if np.isfinite(gap) else np.inf),
            -(balance if np.isfinite(balance) else np.inf),
        )

    return max(eligible, key=selection_key)

def _alpha_setting_from_summary(summary):
    return {"task_penalty": float(summary["task_penalty"]),
            "bold_penalty": float(summary["bold_penalty"]),
            "beta_penalty": float(summary["beta_penalty"]),
            "smooth_penalty": float(summary["smooth_penalty"]),
            "corr_weight": float(summary.get("corr_weight", 1.0))}

def _clamp_alpha(value, lower=0.05, upper=4.0):
    return float(np.clip(float(value), lower, upper))

def build_fine_alpha_sweep(best_summary):
    base = [float(best_summary["task_penalty"]), float(best_summary["bold_penalty"]),
            float(best_summary["beta_penalty"]), float(best_summary["smooth_penalty"]),
            float(best_summary.get("corr_weight", 1.0))]
    candidates = [tuple(base)]
    for alpha_idx in range(4):
        for factor in (0.75, 1.25):
            candidate = list(base)
            candidate[alpha_idx] = _clamp_alpha(candidate[alpha_idx] * factor)
            candidates.append(tuple(candidate))
    for factor in (0.85, 1.15):
        candidate = [_clamp_alpha(value * factor) for value in base[:4]] + [base[4]]
        candidates.append(tuple(candidate))
    return _alpha_settings_from_penalty_sweep(_dedupe_penalty_tuples(candidates))

def _print_parameter_ranking(label, summaries, selected_summary, max_rows=8):
    print(f"\n===== {label} parameter ranking =====", flush=True)
    ranked = sorted(summaries, key=lambda item: item.get("selection_score", -np.inf), reverse=True)
    for idx, summary in enumerate(ranked[:max_rows], start=1):
        print(
            f"{idx:02d}. task={summary['task_penalty']:g}, bold={summary['bold_penalty']:g}, "
            f"beta={summary['beta_penalty']:g}, smooth={summary['smooth_penalty']:g}, "
            f"gamma={summary['gamma']:g}: test_corr={summary['test_corr_mean']:.4f} "
            f"+/- {summary['test_corr_se']:.4f}, gap={summary['train_test_gap']:.4f}, "
            f"score={summary['selection_score']:.4f}",
            flush=True,
        )
    print(
        f"Selected {label}: task={selected_summary['task_penalty']:g}, "
        f"bold={selected_summary['bold_penalty']:g}, beta={selected_summary['beta_penalty']:g}, "
        f"smooth={selected_summary['smooth_penalty']:g}, gamma={selected_summary['gamma']:g}",
        flush=True,
    )

def _write_penalty_search_summary(coarse_summaries, coarse_best, fine_summaries, final_best, final_summaries=None):
    payload = {
        "mode": "coarse_to_fine",
        "selection_rule": "one-standard-error candidate set, ranked by held-out correlation penalized for fold SE and train-test gap",
        "coarse_best": coarse_best,
        "final_best": final_best,
        "coarse_summaries": coarse_summaries,
        "fine_summaries": fine_summaries,
        "final_summaries": final_summaries or [],
    }
    output_path = _result_path(f"penalty_search_summary_sub{sub}_ses{ses}.json", for_write=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(_sanitize_for_json(payload), handle, indent=2)
    print(f"Saved penalty search summary: {output_path}", flush=True)
    return output_path

def run_coarse_to_fine_search(alpha_settings, gamma_values, fold_splits, projection_data):
    print("\n===== Coarse-to-fine penalty search =====", flush=True)
    print("Coarse and fine search stages skip spatial map outputs; final selected setting saves the full outputs.", flush=True)

    coarse_metrics = run_cross_run_experiment(alpha_settings, gamma_values, fold_splits, projection_data,
                                              save_artifacts=False, compute_spatial_outputs=False,
                                              append_log=False, run_label="coarse-search")
    coarse_summaries = summarize_aggregate_metrics(coarse_metrics)
    coarse_best = select_best_parameter_summary(coarse_summaries)
    _print_parameter_ranking("coarse-search", coarse_summaries, coarse_best)

    fine_alphas = build_fine_alpha_sweep(coarse_best)
    fine_gamma_values = [float(coarse_best["gamma"])]
    fine_metrics = run_cross_run_experiment(fine_alphas, fine_gamma_values, fold_splits, projection_data,
                                            save_artifacts=False, compute_spatial_outputs=False,
                                            append_log=False, run_label="fine-search")
    fine_summaries = summarize_aggregate_metrics(fine_metrics)
    final_best = select_best_parameter_summary(fine_summaries)
    _print_parameter_ranking("fine-search", fine_summaries, final_best)

    final_alpha = [_alpha_setting_from_summary(final_best)]
    final_gamma_values = [float(final_best["gamma"])]
    final_metrics = run_cross_run_experiment(final_alpha, final_gamma_values, fold_splits, projection_data,
                                             save_artifacts=True, compute_spatial_outputs=True,
                                             append_log=True, run_label="selected-final")
    final_summaries = summarize_aggregate_metrics(final_metrics)
    _write_penalty_search_summary(coarse_summaries, coarse_best, fine_summaries, final_best, final_summaries=final_summaries)
    return final_metrics


def select_parameter_sweep(combo_env=None):
    combo_idx = None
    combo_env = combo_env if combo_env is not None else (os.environ.get("FMRI_COMBO_IDX") or os.environ.get("SLURM_ARRAY_TASK_ID"))
    if combo_env not in (None, ""):
        combo_idx = int(combo_env)

    selected_alphas = alpha_sweep
    selected_gammas = gamma_sweep
    if combo_idx is not None:
        total = len(alpha_sweep) * len(gamma_sweep)
        alpha_idx, gamma_idx = divmod(combo_idx, len(gamma_sweep))
        selected_alphas = [alpha_sweep[alpha_idx]]
        selected_gammas = [gamma_sweep[gamma_idx]]
        print(f"Running combo_idx={combo_idx} (alpha_idx={alpha_idx}, gamma_idx={gamma_idx})", flush=True)
    else:
        print("Running full alpha/gamma sweep (no combo index provided).", flush=True)
    return selected_alphas, selected_gammas

def resolve_num_folds(default_num_folds=10):
    num_folds_env = os.environ.get("FMRI_NUM_FOLDS", "").strip()
    if num_folds_env:
        return int(num_folds_env)
    return int(default_num_folds)

def main():
    load_group_inputs()
    projection_data = build_projection_data()
    search_mode = resolve_penalty_search_mode()
    if search_mode == "manual":
        selected_alphas, selected_gammas = select_parameter_sweep()
    else:
        selected_alphas, selected_gammas = alpha_sweep, gamma_sweep
        print(f"Penalty search mode: {search_mode}. Using all {len(selected_alphas)} coarse candidates.", flush=True)
    num_folds = resolve_num_folds()

    fold_splits = build_custom_kfold_splits(num_trials, num_folds=num_folds)
    fold_sizes = [int(split["test_indices"].size) for split in fold_splits]
    print(f"Constructed {len(fold_splits)} folds using num_folds={num_folds} (global-contiguous).", flush=True)
    print(f"Fold test sizes: min={min(fold_sizes)}, max={max(fold_sizes)}, "
          f"mean={float(np.mean(fold_sizes)):.2f}", flush=True)
    if search_mode == "coarse_to_fine":
        run_coarse_to_fine_search(selected_alphas, selected_gammas, fold_splits, projection_data)
    else:
        run_cross_run_experiment(selected_alphas, selected_gammas, fold_splits, projection_data)

    # trial_indices = np.arange(projection_data["behavior_matrix"].shape[0], dtype=int)
    # run_data = prepare_data_func(projection_data, trial_indices, trial_len, transition_boundaries, projection_data)

if __name__ == "__main__":
    main()
