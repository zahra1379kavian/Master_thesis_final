# %%
import os
import re
from glob import glob

import nibabel as nib
import numpy as np
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests
import scipy.ndimage as ndimage

# %%
TRIAL_LEN = 9

DATA_ROOT = "/Data/zahra"
BOLD_DIR = os.path.join(DATA_ROOT, "bold_data")
GLM_ROOT = os.path.join(DATA_ROOT, "results_glm")
OUT_ROOT = os.path.join(DATA_ROOT, "data", "results_beta_preprocessed")
ALT_OUT_ROOT = os.path.join(DATA_ROOT, "results_beta_preprocessed")
OUTPUT_ROOTS = [OUT_ROOT]
if ALT_OUT_ROOT != OUT_ROOT and os.path.isdir(ALT_OUT_ROOT):
    OUTPUT_ROOTS.append(ALT_OUT_ROOT)

BRAIN_MASK_TEMPLATE = os.path.join(DATA_ROOT, "anatomy_masks", "MNI152_T1_2mm_brain_mask.nii.gz")
CSF_MASK_TEMPLATE = os.path.join(DATA_ROOT, "anatomy_masks", "MNI152_T1_2mm_brain_seg_csf.nii.gz")
GRAY_MASK_TEMPLATE = os.path.join(DATA_ROOT, "anatomy_masks", "MNI152_T1_2mm_brain_seg_gm.nii.gz")
ANAT_TEMPLATE = os.path.join(DATA_ROOT, "anatomy_masks", "MNI152_T1_2mm_brain.nii.gz")
GO_TIMES_DIR = os.path.join(DATA_ROOT, "go_times")

BOLD_RE = re.compile(r"sub-(?P<sub>[^_]+)_ses-(?P<ses>\d+)_run-(?P<run>\d+)_")

# %%
back_mask = nib.load(BRAIN_MASK_TEMPLATE).get_fdata().astype(np.float32)
csf_mask = nib.load(CSF_MASK_TEMPLATE).get_fdata().astype(np.float32)
gray_mask = nib.load(GRAY_MASK_TEMPLATE).get_fdata().astype(np.float32)

back_mask_data = back_mask > 0
csf_mask_data = csf_mask > 0
gray_mask_data = gray_mask > 0.5
mask = np.logical_and(back_mask_data, ~csf_mask_data)
nonzero_mask = np.where(mask)
keep_voxels = gray_mask_data[nonzero_mask]
masked_coords_base = tuple(ax[keep_voxels] for ax in nonzero_mask)

bold_files = sorted(glob(os.path.join(BOLD_DIR, "sub-*_ses-*_run-*_task-*_bold_*mnireg-2mm.nii.gz")))
if not bold_files:
    raise FileNotFoundError(f"No BOLD files found in {BOLD_DIR}")

print(f"Found {len(bold_files)} BOLD files", flush=True)
os.makedirs(OUT_ROOT, exist_ok=True)

def hampel_filter_image(image, window_size, threshold_factor, return_stats=False):
    footprint = np.ones((window_size,) * 3, dtype=bool)
    insufficient_any = np.zeros(image.shape[:3], dtype=bool)
    corrected_any = np.zeros(image.shape[:3], dtype=bool)

    for t in range(image.shape[3]):
        print(f"Trial Number: {t}", flush=True)
        vol = image[..., t]
        valid = np.isfinite(vol)

        med = ndimage.generic_filter(vol, np.nanmedian, footprint=footprint, mode='constant', cval=np.nan).astype(np.float32)
        mad = ndimage.generic_filter(np.abs(vol - med), np.nanmedian, footprint=footprint, mode='constant', cval=np.nan).astype(np.float32)
        counts = ndimage.generic_filter(np.isfinite(vol).astype(np.float32), np.sum, footprint=footprint, mode='constant', cval=0).astype(np.float32)
        neighbor_count = counts - valid.astype(np.float32)

        scaled_mad = 1.4826 * mad
        insufficient = valid & (neighbor_count < 3)
        insufficient_any |= insufficient
        image[..., t][insufficient] = np.nan

        enough_data = (neighbor_count >= 3) & valid
        outliers = enough_data & (np.abs(vol - med) > threshold_factor * scaled_mad)

        corrected_any |= outliers
        image[..., t][outliers] = med[outliers]

    if return_stats:
        stats = {'insufficient_total': int(np.count_nonzero(insufficient_any)), 'corrected_total': int(np.count_nonzero(corrected_any))}
        return image, stats

    return image

for bold_path in bold_files:
    match = BOLD_RE.search(os.path.basename(bold_path))
    if not match:
        print(f"Skipping unrecognized BOLD filename: {bold_path}", flush=True)
        continue

    sub = match.group("sub")
    ses = int(match.group("ses"))
    run = int(match.group("run"))

    tag = f"sub-{sub}_ses-{ses}_run-{run}"
    found_existing = False
    for output_root in OUTPUT_ROOTS:
        out_dir = os.path.join(output_root, f"sub-{sub}")
        existing = glob(os.path.join(out_dir, f"*{tag}*.npy"))
        if existing:
            print(f"Skipping sub-{sub} ses-{ses} run-{run}: existing outputs in {out_dir}", flush=True)
            found_existing = True
            break
    if found_existing:
        continue

    glm_dir = os.path.join(GLM_ROOT, f"sub-{sub}", f"ses-{ses}", "GLMOutputs-mni-std")
    glm_path = os.path.join(glm_dir, "TYPED_FITHRF_GLMDENOISE_RR.npy")
    trial_keep_run1_path = os.path.join(glm_dir, "trial_keep_run1.npy")
    trial_keep_run2_path = os.path.join(glm_dir, "trial_keep_run2.npy")

    if not os.path.exists(glm_path):
        print(f"Missing GLM file for {bold_path}: {glm_path}", flush=True)
        continue
    if not (os.path.exists(trial_keep_run1_path) and os.path.exists(trial_keep_run2_path)):
        print(f"Missing trial_keep files in {glm_dir}", flush=True)
        continue

    print(f"Processing sub-{sub} ses-{ses} run-{run}", flush=True)

    glm_dict = np.load(glm_path, allow_pickle=True).item()
    beta_glm = glm_dict["betasmd"][:, 0, 0, :]

    trial_keep_run1 = np.load(trial_keep_run1_path)
    trial_keep_run2 = np.load(trial_keep_run2_path)
    num_trials = int(trial_keep_run1.size)
    if trial_keep_run2.size != num_trials:
        raise ValueError(f"trial_keep sizes differ in {glm_dir}")

    n_keep_run1 = int(trial_keep_run1.sum())
    n_keep_run2 = int(trial_keep_run2.sum())
    if n_keep_run1 + n_keep_run2 != beta_glm.shape[-1]:
        raise ValueError(f"betasmd length mismatch in {glm_dir}: {beta_glm.shape[-1]} vs {n_keep_run1 + n_keep_run2}")

    beta_run1 = np.full((beta_glm.shape[0], num_trials), np.nan, dtype=beta_glm.dtype)
    beta_run2 = np.full((beta_glm.shape[0], num_trials), np.nan, dtype=beta_glm.dtype)
    beta_run1[:, trial_keep_run1] = beta_glm[:, :n_keep_run1]
    beta_run2[:, trial_keep_run2] = beta_glm[:, n_keep_run1 : n_keep_run1 + n_keep_run2]

    if run == 1:
        beta = beta_run1
    elif run == 2:
        beta = beta_run2
    else:
        print(f"Skipping unsupported run number: {run}", flush=True)
        continue

    bold_img = nib.load(bold_path)
    bold_data = bold_img.get_fdata(dtype=np.float32)
    if bold_data.shape[:3] != back_mask.shape:
        print(f"Skipping {bold_path}: mask shape {back_mask.shape}, does not match bold shape {bold_data.shape[:3]}", flush=True)
        continue

    bold_flat = bold_data[nonzero_mask]
    masked_bold = bold_flat[keep_voxels]
    masked_coords = tuple(coord.copy() for coord in masked_coords_base)

    masked_bold = masked_bold.astype(np.float32)
    num_voxels, num_timepoints = masked_bold.shape
    bold_data_reshape = np.full((num_voxels, num_trials, TRIAL_LEN), np.nan, dtype=np.float32)

    start = 0
    for i in range(num_trials):
        end = start + TRIAL_LEN
        if end > num_timepoints:
            raise ValueError("Masked BOLD data does not contain enough timepoints for all trials")
        bold_data_reshape[:, i, :] = masked_bold[:, start:end]
        start += TRIAL_LEN
        if start in (270, 560):
            start += 20  # skip discarded timepoints

    nan_voxels = np.isnan(beta).all(axis=1)
    if np.any(nan_voxels):
        beta = beta[~nan_voxels]
        bold_data_reshape = bold_data_reshape[~nan_voxels]
        masked_coords = tuple(coord[~nan_voxels] for coord in masked_coords)

    med = np.nanmedian(beta, keepdims=True)
    mad = np.nanmedian(np.abs(beta - med), keepdims=True)
    scale = 1.4826 * np.maximum(mad, 1e-9)
    beta_norm = (beta - med) / scale
    thr = np.nanpercentile(np.abs(beta_norm), 99.9)
    outlier_mask = np.abs(beta_norm) > thr

    clean_beta = beta.copy()
    voxel_outlier_fraction = np.mean(outlier_mask, axis=1)
    valid_voxels = voxel_outlier_fraction <= 0.5
    clean_beta[~valid_voxels] = np.nan
    clean_beta[np.logical_and(outlier_mask, valid_voxels[:, None])] = np.nan
    keeped_mask = ~np.all(np.isnan(clean_beta), axis=1)
    clean_beta = clean_beta[keeped_mask]
    keeped_indices = np.flatnonzero(keeped_mask)

    bold_data_reshape[~valid_voxels, :, :] = np.nan
    trial_outliers = np.logical_and(outlier_mask, valid_voxels[:, None])
    bold_data_reshape = np.where(trial_outliers[:, :, None], np.nan, bold_data_reshape)
    bold_data_reshape = bold_data_reshape[keeped_mask]

    # Apply t-test and FDR, detect & remove non-active voxels
    tvals, pvals = ttest_1samp(clean_beta, popmean=0, axis=1, nan_policy="omit")

    # FDR correction
    tested = np.isfinite(pvals)
    alpha = 0.05
    rej, q, _, _ = multipletests(pvals[tested], alpha=alpha, method="fdr_bh")

    n_voxel = clean_beta.shape[0]
    qvals = np.full(n_voxel, np.nan)
    reject = np.zeros(n_voxel, dtype=bool)
    reject[tested] = rej
    qvals[tested] = q

    # reject non-active voxels
    clean_active_beta = clean_beta[reject]
    clean_active_idx = keeped_indices[reject]
    clean_active_bold = bold_data_reshape[reject]

    clean_active_volume = np.full(bold_data.shape[:3] + (num_trials,), np.nan)
    active_coords = tuple(coord[clean_active_idx] for coord in masked_coords)
    clean_active_volume[active_coords[0], active_coords[1], active_coords[2], :] = clean_active_beta

    beta_volume_filter, hampel_stats = hampel_filter_image(clean_active_volume.astype(np.float32), window_size=5, threshold_factor=3, return_stats=True)
    print("Total voxels with <3 neighbours:", hampel_stats["insufficient_total"], flush=True)
    print("Total corrected voxels:", hampel_stats["corrected_total"], flush=True)

    nan_voxels = np.all(np.isnan(beta_volume_filter), axis=-1)
    mask_2d = nan_voxels.reshape(-1)

    out_dir = os.path.join(OUT_ROOT, f"sub-{sub}")
    os.makedirs(out_dir, exist_ok=True)

    cleaned_beta_path = os.path.join(out_dir, f"cleaned_beta_volume_{tag}.npy")
    np.save(cleaned_beta_path, beta_volume_filter)
    np.save(os.path.join(out_dir, f"mask_all_nan_{tag}.npy"), mask_2d)

    mask_2d = np.load(os.path.join(out_dir, f"mask_all_nan_{tag}.npy"))
    nan_mask_flat_path = os.path.join(out_dir, f"nan_mask_flat_{tag}.npy")
    np.save(nan_mask_flat_path, mask_2d)
    beta_volume_clean_2d = beta_volume_filter[~nan_voxels]
    np.save(os.path.join(out_dir, f"beta_volume_filter_{tag}.npy.npy"), beta_volume_clean_2d)

    active_flat_idx = np.ravel_multi_index(active_coords, nan_voxels.shape)
    np.save(os.path.join(out_dir, f"active_flat_indices__{tag}.npy"), active_flat_idx)
    keep_mask = ~mask_2d[active_flat_idx]
    clean_active_bold = clean_active_bold[keep_mask, ...]
    np.save(os.path.join(out_dir, f"active_bold_{tag}.npy.npy"), clean_active_bold)
    clean_active_beta = clean_active_beta[keep_mask, ...]

    # Drop voxels everywhere our Hampel mask removed them
    clean_active_idx = clean_active_idx[keep_mask]
    active_coords = tuple(coord[keep_mask] for coord in active_coords)
    np.save(os.path.join(out_dir, f"active_coords_{tag}.npy"), active_coords)

    print(f"Finished sub-{sub} ses-{ses} run-{run}", flush=True)
