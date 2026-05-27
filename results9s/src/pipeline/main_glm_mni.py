# %%
from __future__ import annotations

import gc
import os
import time
import re
import sys
from pathlib import Path
import nibabel as nib
import numpy as np


DATA_ROOT_DEFAULT = Path('/Data/zahra')
RESULTS_GLM_ROOT_DEFAULT = Path('/Data/zahra/results_glm')
REPO_ROOT = Path(__file__).resolve().parents[2]
GLMSINGLE_SOURCE_DIR = REPO_ROOT / 'GLMsingle'
if GLMSINGLE_SOURCE_DIR.is_dir() and str(GLMSINGLE_SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(GLMSINGLE_SOURCE_DIR))

BOLD_FILENAME_RE = re.compile(r'^sub-pd(?P<sub>\d+)_ses-(?P<ses>\d+)_run-(?P<run>\d+)_task-mv_bold_corrected_smoothed_mnireg-2mm\.nii\.gz$')

brain_mask_template = 'MNI152_T1_2mm_brain_mask.nii.gz'
csf_mask_template = 'MNI152_T1_2mm_brain_seg_csf.nii.gz'
gray_mask_template = 'MNI152_T1_2mm_brain_seg_gm.nii.gz'
anat_template = 'MNI152_T1_2mm_brain.nii.gz'
go_times_template = 'PSPD{sub}-ses-{ses}-go-times.txt'

mask_mode = 'brain_csf_gray'

num_timepoints = 850
num_trials = 90
stimdur = 9
tr = 1.0
trial_block_size = 30
trial_rest_trs = 20

def _env_override(name, cast, default):
    value = os.getenv(name)
    if value is None:
        return default
    return cast(value)

trial_metric = _env_override('GLM_TRIAL_METRIC', str, 'std')
trial_z = _env_override('GLM_TRIAL_Z', float, 3)
trial_fallback = _env_override('GLM_TRIAL_FALLBACK', float, 95)
trial_max_drop = _env_override('GLM_TRIAL_MAX_DROP', float, 0.15)
trial_onsets_source = _env_override('GLM_TRIAL_ONSETS', str, 'go_times').lower()
results_cache_name = 'results_glmsingle.npy'
load_results_dir = _env_override('GLM_LOAD_DIR', str, '').strip()
load_results_dir = Path(load_results_dir).expanduser().resolve() if load_results_dir else None

glmsingle_wantlibrary = 1
glmsingle_wantglmdenoise = 1
glmsingle_wantfracridge = 1
glmsingle_wantfileoutputs = [1, 1, 1, 1]
glmsingle_wantmemoryoutputs = [1, 1, 1, 1]

def _resolve_zahra_paths():
    root = Path(os.getenv('ZAHRA_ROOT', str(DATA_ROOT_DEFAULT))).expanduser().resolve()
    bold_root = Path(os.getenv('ZAHRA_BOLD_DIR', str(root / 'bold_data'))).expanduser().resolve()
    masks_root = Path(os.getenv('ZAHRA_MASK_DIR', str(root / 'anatomy_masks'))).expanduser().resolve()
    go_times_root = Path(os.getenv('ZAHRA_GO_DIR', str(root / 'go_times'))).expanduser().resolve()
    results_root = Path(os.getenv('ZAHRA_RESULTS_DIR', str(root / 'results'))).expanduser().resolve()
    results_glm_root = Path(os.getenv('ZAHRA_RESULTS_GLM_DIR', str(root / 'results_glm'))).expanduser().resolve()
    return root, bold_root, masks_root, go_times_root, results_root, results_glm_root

def _require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f'Missing {label}: {path}')
    return path

def _discover_bold_runs(bold_root: Path):
    if not bold_root.is_dir():
        raise FileNotFoundError(f'BOLD directory not found: {bold_root}')

    by_sub_ses = {}
    unmatched = []
    for p in sorted(bold_root.glob('*.nii.gz')):
        m = BOLD_FILENAME_RE.match(p.name)
        if not m:
            unmatched.append(p.name)
            continue
        sub = m.group('sub')
        ses = m.group('ses')
        run = int(m.group('run'))
        by_sub_ses.setdefault((sub, ses), []).append((run, p))

    if unmatched:
        preview = '\n'.join(unmatched[:10])
        raise ValueError(f'Found unexpected filenames in {bold_root} (showing up to 10):\n{preview}\n'
            f'Expected pattern: {BOLD_FILENAME_RE.pattern}')

    for k in list(by_sub_ses.keys()):
        by_sub_ses[k] = sorted(by_sub_ses[k], key=lambda x: x[0])

    return by_sub_ses


def apply_mask(anat_data, bold_data, nonzero_mask):
    cleaned_bold_data = bold_data[nonzero_mask]
    cleaned_anat_data = anat_data[nonzero_mask]
    print('combined mask voxels:', nonzero_mask[0].size)
    print('bold_data masked shape:', cleaned_bold_data.shape)
    return cleaned_bold_data, cleaned_anat_data

def _build_mask(brain_mask, csf_mask, gray_mask, mask_mode):
    if mask_mode == 'brain_csf':
        return np.logical_and(brain_mask, ~csf_mask)
    if mask_mode == 'brain_csf_gray':
        return np.logical_and(brain_mask, np.logical_and(gray_mask, ~csf_mask))

def _trial_onsets_from_blocks(num_trials, stimdur, trials_per_block, rest_tr):
    onsets = []
    onset = 1
    for trial_idx in range(num_trials):
        onsets.append(onset)
        onset += stimdur
        if (trial_idx + 1) % trials_per_block == 0 and (trial_idx + 1) < num_trials:
            onset += rest_tr
    onsets = np.array(onsets, dtype=np.int32)
    return onsets

def _standardize_regressor(values):
    values = np.asarray(values, dtype=np.float32)
    mean = float(np.nanmean(values))
    std = float(np.nanstd(values))
    if std > 0:
        return (values - mean) / std
    return values - mean

def _trial_metrics(masked_bold, onsets, stimdur, metric):
    metrics = []
    for onset in onsets:
        start = int(onset) - 1
        end = start + int(round(stimdur / tr))
        segment = masked_bold[:, start:end]
        if metric == 'mean_abs':
            value = float(np.nanmean(np.abs(segment)))
        elif metric == 'std':
            value = float(np.nanstd(segment))
        elif metric == 'dvars':
            diff = np.diff(segment, axis=1)
            value = float(np.sqrt(np.nanmean(diff * diff)))
        else:
            raise ValueError(f'Unknown trial metric: {metric}')
        metrics.append(value)
    return np.array(metrics, dtype=np.float32)

def _trial_keep_mask(metrics, z_thr, fallback_pct, max_drop_fraction, metric):
    if z_thr <= 0:
        return np.ones_like(metrics, dtype=bool)

    med = float(np.nanmedian(metrics))
    mad = float(np.nanmedian(np.abs(metrics - med)))
    scale = 1.4826 * max(mad, 1e-9)
    z = (metrics - med) / scale

    if metric == 'mean_abs':
        direction = 'low'
    else:
        direction = 'high'

    if direction == 'high':
        keep = z <= z_thr
    elif direction == 'low':
        keep = z >= -z_thr
    else:
        raise ValueError(f'Unknown trial metric direction: {direction}')
    keep &= np.isfinite(metrics)

    if np.all(keep) and fallback_pct > 0:
        if direction == 'high':
            cutoff = float(np.nanpercentile(metrics, fallback_pct))
            keep = metrics <= cutoff
        else:
            cutoff = float(np.nanpercentile(metrics, 100 - fallback_pct))
            keep = metrics >= cutoff

    drop_fraction = 1 - np.mean(keep)
    if drop_fraction > max_drop_fraction:
        n_keep = int(np.ceil(len(metrics) * (1 - max_drop_fraction)))
        order = np.argsort(metrics)
        keep = np.zeros_like(metrics, dtype=bool)
        if direction == 'high':
            keep[order[:n_keep]] = True
        else:
            keep[order[-n_keep:]] = True

    return keep

def _load_existing_results(load_dir, cache_name):
    if load_dir is None:
        return None
    cache_path = load_dir / cache_name
    if cache_path.is_file():
        print(f'Loading cached GLMsingle results from {cache_path}')
        return np.load(cache_path, allow_pickle=True).item()
    typed_candidates = sorted(load_dir.glob('TYPED_FITHRF_GLMDENOISE_RR*.npy'))
    if typed_candidates:
        typed_path = typed_candidates[0]
        print(f'Loading cached GLMsingle results from {typed_path}')
        return {'typed': np.load(typed_path, allow_pickle=True).item()}
    return None

def _has_existing_results(output_dir: Path, cache_name: str) -> bool:
    if output_dir is None:
        return False
    cache_path = output_dir / cache_name
    if cache_path.is_file():
        return True
    typed_candidates = sorted(output_dir.glob('TYPED_FITHRF_GLMDENOISE_RR*.npy'))
    return bool(typed_candidates)

def _has_existing_results_glm(results_glm_root: Path, sub: str, ses: str, trial_metric: str) -> bool:
    if results_glm_root is None:
        return False
    outputdir_glm = results_glm_root / f'sub-pd{sub}' / f'ses-{ses}' / f'GLMOutputs-mni-{trial_metric}'
    return _has_existing_results(outputdir_glm, results_cache_name)


def _load_templates(masks_root: Path):
    brain_mask_path = _require_file(masks_root / brain_mask_template, 'brain_mask_template')
    csf_mask_path = _require_file(masks_root / csf_mask_template, 'csf_mask_template')
    gray_mask_path = _require_file(masks_root / gray_mask_template, 'gray_mask_template')
    anat_path = _require_file(masks_root / anat_template, 'anat_template')
    return brain_mask_path, csf_mask_path, gray_mask_path, anat_path

def _load_go_times(go_times_root: Path, sub: str, ses: str, n_runs: int) -> np.ndarray:
    go_times_path = _require_file(go_times_root / go_times_template.format(sub=sub, ses=ses), 'go_times_template')
    go_flag = np.loadtxt(go_times_path, dtype=int)
    if go_flag.ndim == 1:
        go_flag = go_flag[None, :]
    if go_flag.shape[0] < n_runs:
        raise ValueError(f'Go times file has {go_flag.shape[0]} runs but bold_data has {n_runs} runs: {go_times_path}')
    return go_flag

def _validate_mask_compat(bold_img: nib.Nifti1Image, brain_mask: nib.Nifti1Image, csf_mask: nib.Nifti1Image, gray_mask: nib.Nifti1Image, anat_img: nib.Nifti1Image):
    bold_shape = tuple(bold_img.shape[:3])
    for label, img in [('brain_mask', brain_mask), ('csf_mask', csf_mask), ('gray_mask', gray_mask), ('anat', anat_img)]:
        if tuple(img.shape[:3]) != bold_shape:
            raise ValueError(f'Shape mismatch: bold={bold_shape} vs {label}={img.shape[:3]}. Check that bold_data is in MNI 2mm space and templates are MNI152_T1_2mm.')

def _binary_mask_data(mask_img: nib.Nifti1Image, *, label: str) -> np.ndarray:
    data = mask_img.get_fdata(dtype=np.float32)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        raise ValueError(f'{label} mask has no finite values.')

    tol = 1e-6
    minv = float(np.nanmin(finite))
    maxv = float(np.nanmax(finite))
    if minv < -tol or maxv > 1.0 + tol:
        raise ValueError(f'{label} mask is not binary (min={minv:.4g}, max={maxv:.4g}).')
    if np.any((finite > tol) & (finite < 1.0 - tol)):
        raise ValueError(f'{label} mask is not binary (contains values between 0 and 1).')
    return data > 0.5

def run_glmsingle_mni_for_subject_session(*, sub: str, ses: str, run_paths: list[Path], results_root: Path, masks_root: Path, go_times_root: Path, dry_run: bool, force: bool):
    runs = [int(BOLD_FILENAME_RE.match(p.name).group('run')) for p in run_paths]
    run_labels = [str(r) for r in runs]
    print(f'\n=== sub-pd{sub} ses-{ses} runs={run_labels} ===')

    brain_mask_path, csf_mask_path, gray_mask_path, anat_path = _load_templates(masks_root)
    first_bold_img = nib.load(str(run_paths[0]))
    brain_mask_img = nib.load(str(brain_mask_path))
    csf_mask_img = nib.load(str(csf_mask_path))
    gray_mask_img = nib.load(str(gray_mask_path))
    anat_img = nib.load(str(anat_path))
    _validate_mask_compat(first_bold_img, brain_mask_img, csf_mask_img, gray_mask_img, anat_img)

    outputdir_glmsingle = results_root / f'sub-pd{sub}' / f'ses-{ses}' / f'GLMOutputs-mni-{trial_metric}'
    if dry_run:
        print(f'[dry-run] Would write results to: {outputdir_glmsingle}')
        _ = _binary_mask_data(brain_mask_img, label='brain_mask')
        _ = _binary_mask_data(csf_mask_img, label='csf_mask')
        _ = _binary_mask_data(gray_mask_img, label='gray_mask')
        _load_go_times(go_times_root, sub, ses, n_runs=len(run_paths))
        return
    outputdir_glmsingle.mkdir(parents=True, exist_ok=True)

    active_load_results_dir = None if force else (load_results_dir if load_results_dir is not None else outputdir_glmsingle)
    if force:
        print('Force enabled: ignoring cached GLMsingle results and refitting.')

    print('apply masking...')
    brain_mask_data = _binary_mask_data(brain_mask_img, label='brain_mask')
    csf_mask_data = _binary_mask_data(csf_mask_img, label='csf_mask')
    gray_mask_data = _binary_mask_data(gray_mask_img, label='gray_mask')
    mask = _build_mask(brain_mask_data, csf_mask_data, gray_mask_data, mask_mode)
    mask_indices = np.where(mask)
    print('Combined mask voxels:', mask_indices[0].size)

    mask_indices_path = outputdir_glmsingle / 'mask_indices.npy'
    np.save(mask_indices_path, np.array(mask_indices, dtype=object))
    print(f'Saved mask indices: {mask_indices_path}')
    anat_data = anat_img.get_fdata(dtype=np.float32)

    data = []
    extraregressors = []
    trial_keep_by_run = []

    print('Load BOLD data...')
    for run_label, bold_path in zip(run_labels, run_paths):
        bold_run = nib.load(str(bold_path)).get_fdata(dtype=np.float32)
        masked_bold, _ = apply_mask(anat_data, bold_run, mask_indices)
        data.append(masked_bold)
        csf_ts = _standardize_regressor(np.mean(bold_run[csf_mask_data], axis=0))
        extraregressors.append(csf_ts[:, None])
        print(f'  run {run_label}: {bold_run.shape} -> {masked_bold.shape}')

    del anat_data

    print('Select trials...')
    go_flag = _load_go_times(go_times_root, sub, ses, n_runs=len(run_paths))
    run_onsets_metric = _trial_onsets_from_blocks(num_trials, stimdur, trial_block_size, trial_rest_trs)

    trial_keep_by_run = []
    trial_metrics_by_run = []
    for idx, run_label in enumerate(run_labels):
        if trial_onsets_source == 'go_times':
            run_onsets = go_flag[idx][:num_trials]
        else:
            run_onsets = run_onsets_metric
        metrics = _trial_metrics(data[idx], run_onsets, stimdur, trial_metric)
        keep = _trial_keep_mask(metrics, trial_z, trial_fallback, trial_max_drop, trial_metric)
        trial_keep_by_run.append(keep)
        trial_metrics_by_run.append(metrics)
        dropped = int(np.count_nonzero(~keep))
        print(f'Run {run_label}: dropping {dropped}/{len(keep)} trials (metric={trial_metric}).')

    print('Create design matrices...')
    design_matrix = []
    for idx, run_label in enumerate(run_labels):
        design = np.zeros((num_timepoints, 1), dtype=int)
        if trial_onsets_source == 'go_times':
            run_onsets_design = go_flag[idx][:num_trials]
        else:
            run_onsets_design = run_onsets_metric
        keep = trial_keep_by_run[idx]
        for onset, keep_trial in zip(run_onsets_design, keep):
            if not keep_trial:
                continue
            onset_idx = int(onset) - 1
            if onset_idx < 0 or onset_idx >= num_timepoints:
                raise ValueError(f'Invalid onset {onset} for run {run_label} with T={num_timepoints}')
            design[onset_idx, 0] = 1
        design_matrix.append(design)
        print(
            f'Run {run_label}: design onsets source={trial_onsets_source}, '
            f'first onset TR={int(run_onsets_design[0])}, last onset TR={int(run_onsets_design[-1])}.'
        )

    opt = {'wantlibrary': glmsingle_wantlibrary, 'wantglmdenoise': int(glmsingle_wantglmdenoise), 'wantfracridge': int(glmsingle_wantfracridge),
        'wantfileoutputs': glmsingle_wantfileoutputs, 'wantmemoryoutputs': glmsingle_wantmemoryoutputs, 'chunklen': 8000}
    if len(extraregressors) == len(run_paths):
        opt['extra_regressors'] = extraregressors

    results_glmsingle = _load_existing_results(active_load_results_dir, results_cache_name)
    if results_glmsingle is None and active_load_results_dir is not None:
        print(f'No cached results found in {active_load_results_dir}; running GLMsingle.')

    if results_glmsingle is None:
        print('running GLMsingle...')
        from glmsingle.glmsingle import GLM_single
        glmsingle_obj = GLM_single(opt)
        results_glmsingle = glmsingle_obj.fit(design_matrix, data, stimdur, tr, outputdir=str(outputdir_glmsingle))
        np.save(outputdir_glmsingle / results_cache_name, results_glmsingle)

        # GLMsingle wipes outputdir at start, so save sidecar outputs after fit.
        np.save(outputdir_glmsingle / 'mask_indices.npy', np.array(mask_indices, dtype=object))
        for run_label, keep, metrics in zip(run_labels, trial_keep_by_run, trial_metrics_by_run):
            np.save(outputdir_glmsingle / f'trial_keep_run{run_label}.npy', keep)
            np.save(outputdir_glmsingle / f'trial_metric_run{run_label}.npy', metrics)

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Run GLMsingle in MNI space for all subjects/sessions/runs in /Data/zahra/bold_data.')
    parser.add_argument('--dry-run', action='store_true', help='Only validate inputs and print planned outputs; do not write results.')
    parser.add_argument('--force', action='store_true', help='Run even if results already exist in the results directory.')
    parser.add_argument('--subjects', nargs='*', default=None, help='Optional list of subject IDs (e.g., 004 009). Defaults to all discovered.')
    parser.add_argument('--sessions', nargs='*', default=None, help='Optional list of session IDs (e.g., 1 2). Defaults to all discovered.')
    args = parser.parse_args()

    _, bold_root, masks_root, go_times_root, results_root, results_glm_root = _resolve_zahra_paths()
    by_sub_ses = _discover_bold_runs(bold_root)

    selected = []
    for (sub, ses), runs_and_paths in sorted(by_sub_ses.items(), key=lambda x: (int(x[0][0]), int(x[0][1]))):
        if args.subjects is not None and sub not in args.subjects:
            continue
        if args.sessions is not None and ses not in args.sessions:
            continue
        run_paths = [p for _, p in runs_and_paths]
        selected.append((sub, ses, run_paths))

    if not selected:
        raise ValueError('No subject/session matched the requested filters.')

    print('Zahra paths:')
    print(f'  bold_data: {bold_root}')
    print(f'  anatomy_masks: {masks_root}')
    print(f'  go_times: {go_times_root}')
    print(f'  results: {results_root}')
    print(f'  results_glm: {results_glm_root}')
    print(f'Found {len(selected)} subject-session entries.')

    for sub, ses, run_paths in selected:
        if not args.force and _has_existing_results_glm(results_glm_root, sub, ses, trial_metric):
            print(f'[skip] sub-pd{sub} ses-{ses}: existing results found in {results_glm_root}')
            continue

        outputdir_glmsingle = results_root / f'sub-pd{sub}' / f'ses-{ses}' / f'GLMOutputs-mni-{trial_metric}'
        if not args.force and _has_existing_results(outputdir_glmsingle, results_cache_name):
            print(f'[skip] sub-pd{sub} ses-{ses}: existing results found at {outputdir_glmsingle}')
            continue

        t0 = time.time()
        run_glmsingle_mni_for_subject_session(sub=sub, ses=ses, run_paths=run_paths, results_root=results_root, 
                                              masks_root=masks_root, go_times_root=go_times_root, dry_run=args.dry_run,
                                              force=args.force)
        gc.collect()
        print(f'Finished sub-pd{sub} ses-{ses} in {time.time() - t0:.1f}s')

if __name__ == '__main__':
    main()
