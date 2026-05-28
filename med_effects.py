#!/usr/bin/env python3
from collections import namedtuple
import argparse
import itertools
import json
import re
import warnings
from pathlib import Path
import matplotlib
warnings.filterwarnings('ignore', message='Unable to import Axes3D.*')
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import datasets
from scipy import stats
from sklearn.feature_selection import mutual_info_regression
import statsmodels.formula.api as smf
from threshold_robustness_voxel_network import DEFAULT_AAL_VERSION, REFERENCE_THRESHOLD, ROIGroup, UNASSIGNED_ROI, _build_roi_groups, _coarse_aal_group_name, _resample_label_img
DEFAULT_WEIGHT_MAP = Path('data/voxel_weights_task1_bold1_beta0.75_smooth1.8_gamma1.5.nii.gz')
DEFAULT_ROI_FIGURE = Path('figures/task1_bold1_beta0.75_smooth1.8_gamma1.5_threshold_robustness_atlas_regions.png')
DEFAULT_ROI_REGION_TABLE = Path('figures/task1_bold1_beta0.75_smooth1.8_gamma1.5_threshold_robustness_regions.csv')
DEFAULT_SESSION_MANIFEST = Path('data/med_effects_session_manifest.csv')
DEFAULT_BETA_ROOT = Path('/mnt/TeamShare/Data_Masterfile/H20-00572_All-Dressed/Zahra-Thesis-Data/fmri_opt_group/results_beta_preprocessed')
DEFAULT_OUT_DIR = Path('figures/med_effects')
DEFAULT_ATLAS_CACHE_DIR = Path('/home/zkavian/nilearn_data')
DEFAULT_MIN_REPORT_VOXELS = 25
DEFAULT_MI_NEIGHBORS = 3
CONNECTIVITY_METRIC = 'mutual_information_ksg'
COMPARISON_METRIC = 'laplacian_spectral_distance_signed'
BETA_FILE_RE = re.compile('cleaned_beta_volume_(?P<subject>sub-pd\\d+)_ses-(?P<session>\\d+)_run-(?P<run>\\d+)\\.npy$')
DEFAULT_SESSION_STATES = {'1': 'off', '2': 'on'}
WeightedROI = namedtuple('WeightedROI', ('name', 'mask', 'weights', 'n_voxels'))
SessionSpec = namedtuple('SessionSpec', ('label', 'subject', 'session', 'state', 'bold_path', 'timeseries_path', 'beta_paths'))

def _resolve_path(value, base_dir):
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    path = Path(text).expanduser()
    if path.is_absolute():
        return path
    manifest_relative = (base_dir / path).resolve()
    if manifest_relative.exists():
        return manifest_relative
    return path.resolve()

def _default_region_table_for(roi_figure):
    name = roi_figure.name
    if name.endswith('_atlas_regions.png'):
        return roi_figure.with_name(name.replace('_atlas_regions.png', '_regions.csv'))
    if name.endswith('_atlas_regions.pdf'):
        return roi_figure.with_name(name.replace('_atlas_regions.pdf', '_regions.csv'))
    return DEFAULT_ROI_REGION_TABLE

def _read_manifest(path):
    if not path.exists():
        raise RuntimeError(f'Missing session manifest: {path}. Expected columns: label, subject, session, state, and either bold_path or timeseries_path.')
    manifest = pd.read_csv(path)
    required = {'subject', 'session', 'state'}
    missing_columns = sorted(required - set(manifest.columns))
    if missing_columns:
        raise RuntimeError(f"{path} is missing required columns: {', '.join(missing_columns)}")
    input_columns = {'bold_path', 'timeseries_path', 'beta_path', 'beta_paths'} & set(manifest.columns)
    if not input_columns:
        raise RuntimeError(f'{path} must include one of: bold_path, timeseries_path, beta_path, beta_paths')
    specs = []
    base_dir = path.parent.resolve()
    for (row_index, row) in manifest.iterrows():
        subject = str(row['subject']).strip()
        session = str(row['session']).strip()
        state = str(row['state']).strip().lower()
        if not subject or not session or (not state):
            raise RuntimeError(f'{path} row {row_index + 2} has an empty subject, session, or state')
        label = str(row.get('label', f'{subject}_ses-{session}')).strip()
        bold_path = _resolve_path(row.get('bold_path'), base_dir)
        timeseries_path = _resolve_path(row.get('timeseries_path'), base_dir)
        beta_paths = []
        beta_path = _resolve_path(row.get('beta_path'), base_dir)
        if beta_path is not None:
            beta_paths.append(beta_path)
        beta_path_text = row.get('beta_paths')
        if not pd.isna(beta_path_text):
            for item in str(beta_path_text).split(';'):
                beta_item = _resolve_path(item, base_dir)
                if beta_item is not None:
                    beta_paths.append(beta_item)
        if bold_path is None and timeseries_path is None and (not beta_paths):
            raise RuntimeError(f'{path} row {row_index + 2} must provide bold_path, timeseries_path, or beta_path')
        specs.append(SessionSpec(label=label, subject=subject, session=session, state=state, bold_path=bold_path, timeseries_path=timeseries_path, beta_paths=tuple(beta_paths)))
    return specs

def _discover_beta_sessions(beta_root, session_states):
    if not beta_root.exists():
        raise RuntimeError(f'Missing beta root: {beta_root}')
    grouped = {}
    for path in sorted(beta_root.glob('sub-pd*/cleaned_beta_volume_sub-pd*_ses-*_run-*.npy')):
        match = BETA_FILE_RE.match(path.name)
        if match is None:
            continue
        subject = match.group('subject')
        session = match.group('session')
        run = int(match.group('run'))
        grouped.setdefault((subject, session), []).append((run, path))
    if not grouped:
        raise RuntimeError(f'No cleaned_beta_volume files found under {beta_root}')
    specs = []
    for ((subject, session), run_paths) in sorted(grouped.items()):
        state = session_states.get(str(session))
        if state is None:
            raise RuntimeError(f'No medication state mapping was provided for session {session}')
        paths = tuple((path for (_, path) in sorted(run_paths)))
        specs.append(SessionSpec(label=f'{subject}_ses-{session}', subject=subject, session=str(session), state=state, bold_path=None, timeseries_path=None, beta_paths=paths))
    return specs

def _normalize_subject(subject):
    text = str(subject).strip()
    return text if text.startswith('sub-') else f'sub-{text}'

def _filter_session_specs(args, specs):
    if args.subjects:
        keep = {_normalize_subject(subject) for subject in args.subjects}
        specs = [spec for spec in specs if spec.subject in keep]
    if args.complete_subjects_only:
        states_by_subject = {}
        for spec in specs:
            states_by_subject.setdefault(spec.subject, set()).add(spec.state)
        complete_subjects = {subject for (subject, states) in states_by_subject.items() if {'off', 'on'}.issubset(states)}
        specs = [spec for spec in specs if spec.subject in complete_subjects]
    if not specs:
        raise RuntimeError('No sessions remain after subject/session filtering')
    return specs

def _load_session_specs(args):
    if args.session_manifest.exists():
        specs = _read_manifest(args.session_manifest)
    else:
        specs = _discover_beta_sessions(args.beta_root, DEFAULT_SESSION_STATES)
    return _filter_session_specs(args, specs)

def _session_summary(specs):
    rows = []
    for spec in specs:
        input_kind = 'timeseries' if spec.timeseries_path is not None else 'bold' if spec.bold_path is not None else 'beta'
        rows.append({'label': spec.label, 'subject': spec.subject, 'session': spec.session, 'state': spec.state, 'input_kind': input_kind, 'n_beta_runs': len(spec.beta_paths)})
    return pd.DataFrame(rows)

def _beta_shape_counts(specs):
    counts = {}
    for spec in specs:
        for beta_path in spec.beta_paths:
            data = np.load(beta_path, mmap_mode='r')
            key = f'{tuple(data.shape)} {data.dtype}'
            counts[key] = counts.get(key, 0) + 1
    return counts

def _missing_inputs(args):
    missing = []
    if not args.weight_map.exists():
        missing.append(f'{args.weight_map} (optimization-weight NIfTI)')
    if not args.roi_definition_figure.exists():
        missing.append(f'{args.roi_definition_figure} (ROI definition reference figure)')
    if not args.roi_region_table.exists():
        missing.append(f'{args.roi_region_table} (machine-readable ROI table; the PNG alone is not a voxel mask)')
    try:
        specs = _load_session_specs(args)
    except RuntimeError as exc:
        missing.append(str(exc))
        return missing
    for spec in specs:
        if spec.timeseries_path is not None and (not spec.timeseries_path.exists()):
            missing.append(f'{spec.timeseries_path} (ROI time-series CSV for {spec.label})')
        if spec.timeseries_path is None and spec.bold_path is not None and (not spec.bold_path.exists()):
            missing.append(f'{spec.bold_path} (4D BOLD NIfTI for {spec.label})')
        for beta_path in spec.beta_paths:
            if not beta_path.exists():
                missing.append(f'{beta_path} (4D cleaned beta volume for {spec.label})')
    return missing

def _load_roi_names(region_table, roi_percentile, min_report_voxels):
    regions = pd.read_csv(region_table)
    required = {'percentile', 'roi_name', 'n_voxels'}
    missing = sorted(required - set(regions.columns))
    if missing:
        raise RuntimeError(f"{region_table} is missing required columns: {', '.join(missing)}")
    rows = regions[np.isclose(regions['percentile'].astype(float), float(roi_percentile)) & ~regions['roi_name'].eq(UNASSIGNED_ROI) & (regions['n_voxels'].astype(int) >= int(min_report_voxels))].copy()
    if 'present_for_report' in rows.columns:
        rows = rows[rows['present_for_report'].astype(bool)]
    if rows.empty:
        raise RuntimeError(f'No reportable p{roi_percentile:g} ROI rows found in {region_table}; try lowering --min-report-voxels.')
    rows = rows.sort_values('n_voxels', ascending=False)
    return rows['roi_name'].astype(str).tolist()

def _build_lateralized_roi_groups(reference_img, aal_version, cache_dir, roi_names):
    data_dir = str(cache_dir) if cache_dir is not None else None
    atlas = datasets.fetch_atlas_aal(version=aal_version, data_dir=data_dir, verbose=0)
    atlas_img = atlas.maps if isinstance(atlas.maps, nib.Nifti1Image) else nib.load(atlas.maps)
    atlas_data = _resample_label_img(atlas_img, reference_img)
    atlas_source = f'AAL3v2 ({Path(str(atlas.maps)).name})' if aal_version == '3v2' else f'AAL {aal_version}'
    keep = set(roi_names)
    group_masks = {}
    group_labels = {}
    for (label_value, label_name) in zip(atlas.indices, atlas.labels):
        label_value = int(label_value)
        label_name = str(label_name)
        if label_value == 0 or label_name.lower() == 'background':
            continue
        if label_name.endswith('_L'):
            hemisphere = 'L'
        elif label_name.endswith('_R'):
            hemisphere = 'R'
        else:
            continue
        group_name = _coarse_aal_group_name(label_name)
        if group_name not in keep:
            continue
        roi_name = f'{group_name}_{hemisphere}'
        mask = atlas_data == label_value
        if not np.any(mask):
            continue
        if roi_name in group_masks:
            group_masks[roi_name] |= mask
            group_labels[roi_name].append(label_name)
        else:
            group_masks[roi_name] = mask.copy()
            group_labels[roi_name] = [label_name]
    ordered_names = [f'{name}_{hemisphere}' for name in roi_names for hemisphere in ('L', 'R')]
    missing = [name for name in ordered_names if name not in group_masks]
    if missing:
        raise RuntimeError('Missing lateralized AAL ROI masks: ' + ', '.join(missing))
    groups = [ROIGroup(name=name, source=atlas_source, mask=group_masks[name], matched_labels=tuple(group_labels[name])) for name in ordered_names]
    metadata = {'roi_definition': 'aal3_left_right_coarse_anatomical_groups', 'priority_order': ordered_names + [UNASSIGNED_ROI], 'atlas_info': {'name': 'AAL3v2 (Automated Anatomical Labeling 3)' if aal_version == '3v2' else f'AAL {aal_version}', 'description': atlas_source, 'version': aal_version, 'map': str(atlas.maps), 'n_labels': len([label for label in atlas.indices if int(label) != 0]), 'n_regions': len(groups), 'grouping': 'Original AAL left/right labels merged into coarse anatomical groups within hemisphere; midline labels excluded.'}, 'roi_sources': {group.name: group.source for group in groups}, 'roi_matched_labels': {group.name: group.matched_labels for group in groups}}
    return (groups, metadata)

def _analysis_roi_setup(args, reference_img):
    roi_names = _load_roi_names(args.roi_region_table, args.roi_percentile, args.min_report_voxels)
    excluded_rois = {str(name).strip() for name in args.exclude_rois if str(name).strip()}
    if excluded_rois:
        roi_names = [name for name in roi_names if name not in excluded_rois]
    if args.split_hemispheres:
        (groups, metadata) = _build_lateralized_roi_groups(reference_img, args.aal_version, args.atlas_cache_dir, roi_names)
        weighted_roi_names = [group.name for group in groups]
        min_roi_voxels = int(args.min_lateralized_voxels)
    else:
        (groups, metadata) = _build_roi_groups(reference_img, args.aal_version, args.atlas_cache_dir)
        weighted_roi_names = roi_names
        min_roi_voxels = int(args.min_report_voxels)
    metadata.update({'split_hemispheres': bool(args.split_hemispheres), 'source_roi_names': roi_names, 'excluded_rois': sorted(excluded_rois), 'min_roi_voxels': int(min_roi_voxels)})
    return (groups, metadata, weighted_roi_names, min_roi_voxels)

def _build_weighted_rois(weight_values, roi_names, groups, roi_percentile, min_report_voxels, min_roi_voxels=None):
    if min_roi_voxels is None:
        min_roi_voxels = min_report_voxels
    finite_nonzero = np.isfinite(weight_values) & (weight_values != 0)
    if not np.any(finite_nonzero):
        raise RuntimeError('No finite nonzero voxels found in the weight map')
    threshold = float(np.percentile(weight_values[finite_nonzero], roi_percentile))
    selected = finite_nonzero & (weight_values >= threshold)
    group_lookup = {group.name: group for group in groups}
    rois = []
    missing_groups = [name for name in roi_names if name not in group_lookup]
    if missing_groups:
        raise RuntimeError('ROI table names were not found in the AAL grouping: ' + ', '.join(missing_groups))
    for name in roi_names:
        mask = selected & group_lookup[name].mask
        n_voxels = int(np.count_nonzero(mask))
        if n_voxels < min_roi_voxels:
            continue
        roi_weights = np.asarray(weight_values[mask], dtype=np.float64)
        roi_weights = np.where(np.isfinite(roi_weights) & (roi_weights > 0), roi_weights, 0.0)
        if float(np.sum(roi_weights)) <= 0.0:
            roi_weights = np.ones(n_voxels, dtype=np.float64)
        rois.append(WeightedROI(name=name, mask=mask, weights=roi_weights, n_voxels=n_voxels))
    if len(rois) < 2:
        raise RuntimeError('At least two weighted ROI masks are required for edge-network analysis')
    return (rois, threshold)

def _check_image_grid(reference, img, label):
    if img.shape[:3] != reference.shape[:3]:
        raise RuntimeError(f'{label} shape {img.shape[:3]} differs from the weight-map grid {reference.shape[:3]}')
    if not np.allclose(img.affine, reference.affine):
        raise RuntimeError(f'{label} affine differs from the weight-map affine')

def _weighted_mean_timeseries(data, roi):
    roi_data = np.asarray(data[roi.mask, :], dtype=np.float64)
    weights = roi.weights.astype(np.float64, copy=False)
    finite = np.isfinite(roi_data)
    weighted = np.where(finite, roi_data, 0.0) * weights[:, None]
    denom = np.sum(np.where(finite, weights[:, None], 0.0), axis=0)
    out = np.full(roi_data.shape[1], np.nan, dtype=np.float64)
    valid = denom > 0
    out[valid] = np.sum(weighted[:, valid], axis=0) / denom[valid]
    return out

def _extract_roi_timeseries(bold_path, reference_img, rois):
    img = nib.load(str(bold_path))
    _check_image_grid(reference_img, img, str(bold_path))
    if len(img.shape) != 4:
        raise RuntimeError(f'{bold_path} must be a 4D BOLD NIfTI')
    data = img.get_fdata(dtype=np.float32)
    roi_series = {roi.name: _weighted_mean_timeseries(data, roi) for roi in rois}
    return pd.DataFrame(roi_series)

def _extract_roi_timeseries_from_beta(beta_path, reference_img, rois):
    data = np.load(beta_path, mmap_mode='r')
    if data.ndim != 4:
        raise RuntimeError(f'{beta_path} must be a 4D cleaned beta volume with shape (x, y, z, trials)')
    if tuple(data.shape[:3]) != tuple(reference_img.shape[:3]):
        raise RuntimeError(f'{beta_path} shape {data.shape[:3]} differs from the weight-map grid {reference_img.shape[:3]}')
    roi_series = {roi.name: _weighted_mean_timeseries(data, roi) for roi in rois}
    return pd.DataFrame(roi_series)

def _read_roi_timeseries(path, roi_names):
    df = pd.read_csv(path)
    missing = [name for name in roi_names if name not in df.columns]
    if missing:
        raise RuntimeError(f"{path} is missing ROI time-series columns: {', '.join(missing)}")
    return df.loc[:, roi_names].apply(pd.to_numeric, errors='coerce')

def _load_session_timeseries(spec, reference_img, rois):
    roi_names = [roi.name for roi in rois]
    if spec.timeseries_path is not None:
        return _read_roi_timeseries(spec.timeseries_path, roi_names)
    if spec.beta_paths:
        frames = [_extract_roi_timeseries_from_beta(path, reference_img, rois) for path in spec.beta_paths]
        return pd.concat(frames, ignore_index=True)
    if spec.bold_path is None:
        raise RuntimeError(f'{spec.label} has no bold_path, timeseries_path, or beta_path')
    return _extract_roi_timeseries(spec.bold_path, reference_img, rois)

def _clean_timeseries(df):
    values = df.to_numpy(dtype=np.float64)
    values = values[np.all(np.isfinite(values), axis=1)]
    if values.shape[0] < 4:
        raise RuntimeError('ROI time series has fewer than four complete time points')
    centered = values - np.mean(values, axis=0, keepdims=True)
    scale = np.std(centered, axis=0, ddof=1)
    valid = scale > 0
    centered[:, valid] /= scale[valid]
    centered[:, ~valid] = 0.0
    return centered

def _mutual_information_matrix(timeseries, n_neighbors, random_state):
    (n_timepoints, n_rois) = timeseries.shape
    matrix = np.zeros((n_rois, n_rois), dtype=np.float64)
    if n_rois < 2:
        return matrix
    neighbors = min(int(n_neighbors), max(1, n_timepoints - 1))
    for (i, j) in itertools.combinations(range(n_rois), 2):
        xi = timeseries[:, [i]]
        xj = timeseries[:, [j]]
        yi = timeseries[:, i]
        yj = timeseries[:, j]
        if np.std(yi) <= 0 or np.std(yj) <= 0:
            score = 0.0
        else:
            mi_ij = mutual_info_regression(xi, yj, discrete_features=False, n_neighbors=neighbors, random_state=random_state)[0]
            mi_ji = mutual_info_regression(xj, yi, discrete_features=False, n_neighbors=neighbors, random_state=random_state)[0]
            score = max(0.0, float((mi_ij + mi_ji) / 2.0))
        matrix[i, j] = score
        matrix[j, i] = score
    return matrix

def _signed_laplacian_spectrum(adjacency):
    matrix = np.asarray(adjacency, dtype=np.float64)
    matrix = 0.5 * (matrix + matrix.T)
    np.fill_diagonal(matrix, 0.0)
    degree = np.sum(np.abs(matrix), axis=1)
    laplacian = np.diag(degree) - matrix
    scale = np.zeros_like(degree)
    active = degree > 0
    scale[active] = 1.0 / np.sqrt(degree[active])
    normalized = laplacian * scale[:, None] * scale[None, :]
    return np.sort(np.linalg.eigvalsh(normalized))

def _laplacian_spectral_distance_signed(left, right):
    left_spectrum = _signed_laplacian_spectrum(left)
    right_spectrum = _signed_laplacian_spectrum(right)
    return float(np.linalg.norm(left_spectrum - right_spectrum) / np.sqrt(left_spectrum.size))

def _pair_label(state_a, state_b):
    left = str(state_a).lower()
    right = str(state_b).lower()
    if left == right:
        return ('within_condition', f'{left}-{right}')
    if {left, right} == {'off', 'on'}:
        return ('between_condition', 'off-on')
    return ('between_condition', '-'.join(sorted([left, right])))

def _pairwise_network_distances(specs, networks):
    rows = []
    for (left, right) in itertools.combinations(specs, 2):
        (pair_class, pair_label) = _pair_label(left.state, right.state)
        distance = _laplacian_spectral_distance_signed(networks[left.label], networks[right.label])
        rows.append({'connectivity_metric': CONNECTIVITY_METRIC, 'label_a': left.label, 'label_b': right.label, 'subject_a': left.subject, 'subject_b': right.subject, 'session_a': left.session, 'session_b': right.session, 'state_a': left.state, 'state_b': right.state, 'pair_class': pair_class, 'pair_label': pair_label, 'same_subject': bool(left.subject == right.subject), 'comparison_metric': COMPARISON_METRIC, 'comparison_kind': 'graph_distance', 'higher_is_more_similar': False, 'raw_score': distance, 'oriented_score': -distance})
    return pd.DataFrame(rows)

def _metric_pairwise_rows(pairwise):
    return pairwise.loc[(pairwise['connectivity_metric'] == CONNECTIVITY_METRIC) & (pairwise['comparison_metric'] == COMPARISON_METRIC)].copy()

def _session_rows_from_pairwise(pairwise):
    left = pairwise[['label_a', 'subject_a', 'state_a']].rename(columns={'label_a': 'label', 'subject_a': 'subject', 'state_a': 'state'})
    right = pairwise[['label_b', 'subject_b', 'state_b']].rename(columns={'label_b': 'label', 'subject_b': 'subject', 'state_b': 'state'})
    sessions = pd.concat([left, right], ignore_index=True).drop_duplicates()
    sessions['state'] = sessions['state'].astype(str).str.lower()
    return sessions.sort_values(['subject', 'state', 'label']).reset_index(drop=True)

def _complete_off_on_subjects(pairwise):
    sessions = _session_rows_from_pairwise(pairwise)
    complete = []
    incomplete = []
    for (subject, rows) in sessions.groupby('subject', sort=True):
        off_labels = rows.loc[rows['state'] == 'off', 'label'].astype(str).tolist()
        on_labels = rows.loc[rows['state'] == 'on', 'label'].astype(str).tolist()
        if len(off_labels) == 1 and len(on_labels) == 1:
            complete.append({'subject': str(subject), 'off_label': off_labels[0], 'on_label': on_labels[0]})
        else:
            incomplete.append({'subject': str(subject), 'n_off': int(len(off_labels)), 'n_on': int(len(on_labels))})
    return (complete, incomplete)

def _distance_lookup(pairwise):
    lookup = {}
    for row in pairwise.itertuples(index=False):
        if not np.isfinite(float(row.raw_score)):
            continue
        key = tuple(sorted([str(row.label_a), str(row.label_b)]))
        lookup[key] = float(row.raw_score)
    return lookup

def _lookup_distance(lookup, left_label, right_label):
    key = tuple(sorted([str(left_label), str(right_label)]))
    if key not in lookup:
        raise RuntimeError(f'Missing pairwise distance for {left_label} and {right_label}')
    return lookup[key]

def _paired_assignment_group_means(assignments, lookup):
    values = {'off-off': [], 'on-on': [], 'off-on': []}
    for (left, right) in itertools.combinations(assignments, 2):
        if left['subject'] == right['subject']:
            continue
        if left['state'] == right['state'] == 'off':
            key = 'off-off'
        elif left['state'] == right['state'] == 'on':
            key = 'on-on'
        else:
            key = 'off-on'
        values[key].append(_lookup_distance(lookup, left['label'], right['label']))
    means = {key: float(np.mean(group_values)) if group_values else np.nan for (key, group_values) in values.items()}
    counts = {key: int(len(group_values)) for (key, group_values) in values.items()}
    return (means, counts)

def _observed_paired_assignments(complete_subjects):
    assignments = []
    for subject in complete_subjects:
        assignments.append({'subject': subject['subject'], 'label': subject['off_label'], 'state': 'off'})
        assignments.append({'subject': subject['subject'], 'label': subject['on_label'], 'state': 'on'})
    return assignments

def _swapped_paired_assignments(complete_subjects, mask):
    assignments = []
    for (subject_index, subject) in enumerate(complete_subjects):
        swapped = bool((mask >> subject_index) & 1)
        assignments.append({'subject': subject['subject'], 'label': subject['off_label'], 'state': 'on' if swapped else 'off'})
        assignments.append({'subject': subject['subject'], 'label': subject['on_label'], 'state': 'off' if swapped else 'on'})
    return assignments

def _exact_paired_permutation_test(complete_subjects, lookup):
    observed_assignments = _observed_paired_assignments(complete_subjects)
    (observed_means, observed_counts) = _paired_assignment_group_means(observed_assignments, lookup)
    observed_contrast = float(observed_means['off-off'] - observed_means['on-on'])
    n_subjects = len(complete_subjects)
    n_permutations = 1 << n_subjects
    permutation_contrasts = np.empty(n_permutations, dtype=np.float64)
    for mask in range(n_permutations):
        assignments = _swapped_paired_assignments(complete_subjects, mask)
        (means, _) = _paired_assignment_group_means(assignments, lookup)
        permutation_contrasts[mask] = means['off-off'] - means['on-on']
    tolerance = 1e-12
    return {'observed_off_off_mean': float(observed_means['off-off']), 'observed_on_on_mean': float(observed_means['on-on']), 'observed_off_minus_on': observed_contrast, 'pair_counts': observed_counts, 'n_permutations': int(n_permutations), 'permutation_p_value_two_sided': float(np.mean(np.abs(permutation_contrasts) >= abs(observed_contrast) - tolerance)), 'permutation_p_value_greater': float(np.mean(permutation_contrasts >= observed_contrast - tolerance))}

def _subject_level_similarity_values(complete_subjects, lookup):
    rows = []
    for subject in complete_subjects:
        other_subjects = [other for other in complete_subjects if other['subject'] != subject['subject']]
        off_distances = [_lookup_distance(lookup, subject['off_label'], other['off_label']) for other in other_subjects]
        on_distances = [_lookup_distance(lookup, subject['on_label'], other['on_label']) for other in other_subjects]
        off_mean = float(np.mean(off_distances))
        on_mean = float(np.mean(on_distances))
        rows.append({'subject': subject['subject'], 'off_label': subject['off_label'], 'on_label': subject['on_label'], 'off_mean_to_other_off': off_mean, 'on_mean_to_other_on': on_mean, 'off_minus_on': off_mean - on_mean})
    return pd.DataFrame(rows)

def _subject_level_similarity_tests(subject_values):
    differences = subject_values['off_minus_on'].to_numpy(dtype=np.float64)
    differences = differences[np.isfinite(differences)]
    if differences.size < 2:
        return {'n_subjects': int(differences.size), 'mean_off_minus_on': float(np.mean(differences)) if differences.size else float('nan'), 'paired_t_p_value_two_sided': float('nan'), 'wilcoxon_p_value_two_sided': float('nan')}
    mean_difference = float(np.mean(differences))
    sd_difference = float(np.std(differences, ddof=1))
    sem_difference = float(stats.sem(differences))
    t_result = stats.ttest_1samp(differences, 0.0)
    ci_low, ci_high = stats.t.interval(0.95, differences.size - 1, loc=mean_difference, scale=sem_difference)
    try:
        wilcoxon_result = stats.wilcoxon(differences, alternative='two-sided')
        wilcoxon_statistic = float(wilcoxon_result.statistic)
        wilcoxon_p_value = float(wilcoxon_result.pvalue)
    except ValueError:
        wilcoxon_statistic = float('nan')
        wilcoxon_p_value = float('nan')
    return {'n_subjects': int(differences.size), 'mean_off_minus_on': mean_difference, 'sd_off_minus_on': sd_difference, 'sem_off_minus_on': sem_difference, 'ci95_low': float(ci_low), 'ci95_high': float(ci_high), 'cohen_dz': float(mean_difference / sd_difference) if sd_difference > 0 else np.nan, 'paired_t_statistic': float(t_result.statistic), 'paired_t_p_value_two_sided': float(t_result.pvalue), 'wilcoxon_statistic': wilcoxon_statistic, 'wilcoxon_p_value_two_sided': wilcoxon_p_value}

def _paired_similarity_tests(pairwise):
    metric_rows = _metric_pairwise_rows(pairwise)
    (complete_subjects, incomplete_subjects) = _complete_off_on_subjects(metric_rows)
    if len(complete_subjects) < 2:
        raise RuntimeError('At least two complete OFF/ON subjects are required for paired medication similarity tests')
    lookup = _distance_lookup(metric_rows)
    permutation = _exact_paired_permutation_test(complete_subjects, lookup)
    subject_values = _subject_level_similarity_values(complete_subjects, lookup)
    subject_tests = _subject_level_similarity_tests(subject_values)
    stats_summary = {'test_scope': 'complete subjects with one OFF and one ON session; cross-subject distances only', 'hypothesis': 'OFF-OFF distances are greater than ON-ON distances when medication makes networks more similar across subjects', 'complete_subjects': complete_subjects, 'incomplete_subjects': incomplete_subjects, 'permutation': permutation, 'subject_level': subject_tests}
    return (stats_summary, subject_values)

def _save_paired_similarity_tests(pairwise, out_dir):
    (stats_summary, subject_values) = _paired_similarity_tests(pairwise)
    subject_path = out_dir / 'paired_subject_similarity_values.csv'
    stats_path = out_dir / 'paired_subject_similarity_stats.json'
    subject_values.to_csv(subject_path, index=False)
    stats_path.write_text(json.dumps(stats_summary, indent=2), encoding='utf-8')
    return (stats_summary, subject_path, stats_path)

def _holm_adjusted_pvalues(p_values):
    p_values = np.asarray(p_values, dtype=np.float64)
    adjusted = np.full(p_values.shape, np.nan, dtype=np.float64)
    finite_indices = np.flatnonzero(np.isfinite(p_values))
    if finite_indices.size == 0:
        return adjusted
    ordered = finite_indices[np.argsort(p_values[finite_indices])]
    running_max = 0.0
    n_tests = ordered.size
    for (rank, original_index) in enumerate(ordered):
        adjusted_value = (n_tests - rank) * p_values[original_index]
        running_max = max(running_max, adjusted_value)
        adjusted[original_index] = min(running_max, 1.0)
    return adjusted

def _pvalue_stars(p_value):
    if not np.isfinite(p_value) or p_value >= 0.05:
        return ''
    if p_value < 0.001:
        return '***'
    if p_value < 0.01:
        return '**'
    return '*'

def _fixed_effect_vector(fe_names, pair_key, reference_key):
    vector = np.zeros(len(fe_names), dtype=np.float64)
    vector[list(fe_names).index('Intercept')] = 1.0
    if pair_key == reference_key:
        return vector
    matches = [idx for (idx, name) in enumerate(fe_names) if name.endswith(f'[T.{pair_key}]')]
    if not matches:
        raise KeyError(f'Missing fixed-effect term for {pair_key}')
    vector[matches[0]] = 1.0
    return vector

def _pairwise_group_tests(subset, class_order, groups):
    tests = []
    key_by_name = dict(class_order)
    group_keys = [key_by_name[name] for (name, _, _) in groups]
    for ((left_index, (left_name, _, _)), (right_index, (right_name, _, _))) in itertools.combinations(enumerate(groups), 2):
        tests.append({'left': left_index, 'right': right_index, 'left_name': left_name, 'right_name': right_name, 'left_key': group_keys[left_index], 'right_key': group_keys[right_index], 'p_value': np.nan})
    if len(group_keys) < 2:
        return tests
    model_data = subset.loc[subset['pair_label'].isin(group_keys), ['raw_score', 'pair_label', 'subject_a', 'subject_b']].copy()
    model_data = model_data[np.isfinite(model_data['raw_score'].to_numpy(dtype=np.float64))]
    model_data = model_data.dropna(subset=['pair_label', 'subject_a', 'subject_b'])
    if model_data.empty:
        return tests
    model_data['pair_label'] = pd.Categorical(model_data['pair_label'], categories=group_keys, ordered=True)
    model_data['all_pairs'] = 'all'
    subjects = sorted(set(model_data['subject_a'].astype(str)).union(model_data['subject_b'].astype(str)))
    subject_columns = []
    for (subject_index, subject) in enumerate(subjects):
        column = f'subject_member_{subject_index}'
        model_data[column] = ((model_data['subject_a'].astype(str) == subject) | (model_data['subject_b'].astype(str) == subject)).astype(float)
        subject_columns.append(column)
    if not subject_columns:
        return tests
    reference_key = group_keys[0]
    formula = f'raw_score ~ C(pair_label, Treatment(reference="{reference_key}"))'
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = smf.mixedlm(formula, data=model_data, groups=model_data['all_pairs'], re_formula='0', vc_formula={'subject': '0 + ' + ' + '.join(subject_columns)}).fit(reml=True, method='lbfgs', maxiter=1000, disp=False)
    except (ValueError, np.linalg.LinAlgError) as exc:
        warnings.warn(f'Mixed-effects significance model failed: {exc}', RuntimeWarning)
        return tests
    fe_params = result.fe_params
    fe_names = fe_params.index
    fe_cov = result.cov_params().loc[fe_names, fe_names].to_numpy(dtype=np.float64)
    fe_values = fe_params.to_numpy(dtype=np.float64)
    vectors = {pair_key: _fixed_effect_vector(fe_names, pair_key, reference_key) for pair_key in group_keys}
    for test in tests:
        contrast = vectors[test['left_key']] - vectors[test['right_key']]
        estimate = float(contrast @ fe_values)
        se_squared = float(contrast @ fe_cov @ contrast)
        if not np.isfinite(se_squared) or se_squared <= 0:
            continue
        standard_error = float(np.sqrt(se_squared))
        z_value = estimate / standard_error
        p_value = float(2.0 * stats.norm.sf(abs(z_value)))
        test.update({'estimate': estimate, 'standard_error': standard_error, 'z_value': z_value, 'p_value': p_value})
    adjusted = _holm_adjusted_pvalues([test['p_value'] for test in tests])
    for (test, p_adjusted) in zip(tests, adjusted):
        test['p_value_holm'] = float(p_adjusted)
        test['stars'] = _pvalue_stars(p_adjusted)
    return tests

def _add_significance_stars(ax, groups, tests, y_max, y_span):
    significant = [test for test in tests if test['stars']]
    significant = sorted(significant, key=lambda test: (test['right'] - test['left'], test['left']))
    if not significant:
        return
    positions = [pos for (_, pos, _) in groups]
    bracket_height = 0.025 * y_span
    baseline = y_max + 0.08 * y_span
    step = 0.075 * y_span
    for (level, test) in enumerate(significant):
        x_left = positions[test['left']]
        x_right = positions[test['right']]
        y = baseline + level * step
        ax.plot([x_left, x_left, x_right, x_right], [y, y + bracket_height, y + bracket_height, y], color='#222222', linewidth=1.0, clip_on=False)
        ax.text((x_left + x_right) / 2.0, y + bracket_height + 0.008 * y_span, test['stars'], ha='center', va='bottom', color='#222222', fontsize=11, clip_on=False)

def _paired_permutation_plot_tests(groups, paired_stats):
    if paired_stats is None:
        return []
    p_value = paired_stats.get('permutation', {}).get('permutation_p_value_two_sided', np.nan)
    if not np.isfinite(p_value):
        return []
    group_index = {name: index for (index, (name, _, _)) in enumerate(groups)}
    if 'OFF-OFF' not in group_index or 'ON-ON' not in group_index:
        return []
    return [{'left': group_index['OFF-OFF'], 'right': group_index['ON-ON'], 'left_name': 'OFF-OFF', 'right_name': 'ON-ON', 'p_value': float(p_value), 'stars': _pvalue_stars(float(p_value))}]

def _complete_subject_labels_from_stats(paired_stats):
    if paired_stats is None:
        return None
    labels = set()
    for subject in paired_stats.get('complete_subjects', []):
        labels.add(subject['off_label'])
        labels.add(subject['on_label'])
    return labels if labels else None

def _plot_cross_subject_distribution(pairwise, out_dir, paired_stats=None):
    subset = _metric_pairwise_rows(pairwise)
    subset = subset.loc[~subset['same_subject'].astype(bool)].copy()
    complete_labels = _complete_subject_labels_from_stats(paired_stats)
    if complete_labels is not None:
        subset = subset.loc[subset['label_a'].isin(complete_labels) & subset['label_b'].isin(complete_labels)].copy()
    class_order = [('OFF-OFF', 'off-off'), ('ON-ON', 'on-on'), ('OFF-ON', 'off-on')]
    colors_by_name = {'OFF-OFF': '#4c78a8', 'ON-ON': '#e9a3a3', 'OFF-ON': '#54a24b'}
    groups = [(name, idx * 1.15, subset.loc[subset['pair_label'] == key, 'raw_score'].to_numpy(dtype=np.float64)) for (idx, (name, key)) in enumerate(class_order)]
    groups = [(name, pos, values[np.isfinite(values)]) for (name, pos, values) in groups if np.isfinite(values).any()]
    if not groups:
        raise RuntimeError('No finite cross-subject pairwise distances were available for plotting')
    group_tests = _paired_permutation_plot_tests(groups, paired_stats)
    (fig, ax) = plt.subplots(figsize=(5.2, 4.1))
    box = ax.boxplot([values for (_, _, values) in groups], positions=[pos for (_, pos, _) in groups], widths=0.56, patch_artist=True, flierprops={'markeredgecolor': '#444444', 'markerfacecolor': '#444444', 'markersize': 2.5})
    for (patch, (name, _, _)) in zip(box['boxes'], groups):
        patch.set_facecolor(colors_by_name.get(name, '#7f7f7f'))
        patch.set_alpha(0.55)
    rng = np.random.default_rng(0)
    for (_, pos, values) in groups:
        ax.scatter(rng.normal(loc=pos, scale=0.04, size=values.size), values, s=14, alpha=0.55, color='black', linewidths=0.0)
    positions = [pos for (_, pos, _) in groups]
    ax.set_xticks(positions)
    ax.set_xticklabels([name for (name, _, _) in groups])
    ax.set_xlim(min(positions) - 0.48, max(positions) + 0.48)
    ax.set_ylabel('Laplacian spectral distance')
    y_values = np.concatenate([values for (_, _, values) in groups])
    y_min = float(np.min(y_values))
    y_max = float(np.max(y_values))
    y_span = max(y_max - y_min, 1e-06)
    significant_count = sum(1 for test in group_tests if test['stars'])
    upper_padding = 0.3 if significant_count == 0 else 0.18 + 0.095 * significant_count
    ax.set_ylim(y_min - 0.05 * y_span, y_max + upper_padding * y_span)
    _add_significance_stars(ax, groups, group_tests, y_max, y_span)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / 'cross_subject_only_laplacian_spectral_distance_signed_distribution.png'
    pdf_path = png_path.with_suffix('.pdf')
    fig.savefig(png_path, dpi=190, bbox_inches='tight', pad_inches=0.04)
    fig.savefig(pdf_path, bbox_inches='tight', pad_inches=0.04)
    plt.close(fig)
    return png_path

def _save_networks(networks, roi_names, out_dir):
    network_dir = out_dir / 'network_matrices'
    network_dir.mkdir(parents=True, exist_ok=True)
    for (label, matrix) in networks.items():
        pd.DataFrame(matrix, index=roi_names, columns=roi_names).to_csv(network_dir / f'{label}.csv')

def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight-map', type=Path, default=DEFAULT_WEIGHT_MAP)
    parser.add_argument('--roi-definition-figure', type=Path, default=DEFAULT_ROI_FIGURE)
    parser.add_argument('--roi-region-table', type=Path, default=None)
    parser.add_argument('--roi-percentile', type=float, default=REFERENCE_THRESHOLD)
    parser.add_argument('--min-report-voxels', type=int, default=DEFAULT_MIN_REPORT_VOXELS)
    parser.add_argument('--session-manifest', type=Path, default=DEFAULT_SESSION_MANIFEST)
    parser.add_argument('--beta-root', type=Path, default=DEFAULT_BETA_ROOT)
    parser.add_argument('--subjects', nargs='+', default=None)
    parser.add_argument('--complete-subjects-only', action='store_true')
    parser.add_argument('--out-dir', type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument('--split-hemispheres', action='store_true')
    parser.add_argument('--exclude-rois', nargs='*', default=())
    parser.add_argument('--min-lateralized-voxels', type=int, default=1)
    parser.add_argument('--aal-version', default=DEFAULT_AAL_VERSION)
    parser.add_argument('--atlas-cache-dir', type=Path, default=DEFAULT_ATLAS_CACHE_DIR)
    parser.add_argument('--mi-neighbors', type=int, default=DEFAULT_MI_NEIGHBORS)
    parser.add_argument('--random-state', type=int, default=0)
    parser.add_argument('--check-inputs', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    return parser

def _print_dry_run(args):
    weight_img = nib.load(str(args.weight_map))
    weight_values = np.asarray(weight_img.get_fdata(), dtype=np.float64)
    (groups, _, roi_names, min_roi_voxels) = _analysis_roi_setup(args, weight_img)
    (rois, roi_threshold) = _build_weighted_rois(weight_values=weight_values, roi_names=roi_names, groups=groups, roi_percentile=args.roi_percentile, min_report_voxels=args.min_report_voxels, min_roi_voxels=min_roi_voxels)
    specs = _load_session_specs(args)
    session_df = _session_summary(specs)
    beta_shape_counts = _beta_shape_counts(specs)
    print('Dry run only; no connectivity matrices, pairwise distances, or figures were computed.')
    print()
    print('Planned steps:')
    print(f'1. Load weight map: {args.weight_map}')
    print(f'2. Rebuild p{args.roi_percentile:g} weighted AAL ROI masks from: {args.roi_region_table}')
    if args.split_hemispheres:
        print('   Split selected AAL groups into left/right hemisphere ROI masks.')
    print(f'3. Extract weighted mean beta trial series per ROI from {len(specs)} subject/session inputs.')
    print('4. Concatenate runs within each subject/session.')
    print(f'5. Compute {CONNECTIVITY_METRIC} ROI-edge matrices for each subject/session.')
    print(f'6. Compute {COMPARISON_METRIC} for all session pairs.')
    print('7. Compute paired OFF/ON subject-level and exact label-swap similarity tests.')
    print('8. Plot cross-subject-only OFF-OFF, ON-ON, and OFF-ON distance distributions.')
    print()
    print(f'ROI count: {len(rois)}')
    print(f'Weight threshold: {roi_threshold:.8g}')
    print(f'Minimum selected voxels per ROI: {min_roi_voxels}')
    print('Top ROI masks: ' + ', '.join((f'{roi.name} ({roi.n_voxels})' for roi in rois[:8])))
    print()
    print('Input sessions by state:')
    print(session_df.groupby('state')['label'].count().to_string())
    print()
    print('Input sessions:')
    print(session_df.to_string(index=False))
    if beta_shape_counts:
        print()
        print('Beta volume shapes detected:')
        for (shape, count) in beta_shape_counts.items():
            print(f'- {shape}: {count} files')

def main():
    args = build_parser().parse_args()
    if args.roi_region_table is None:
        args.roi_region_table = _default_region_table_for(args.roi_definition_figure)
    missing = _missing_inputs(args)
    if missing:
        print('Missing required inputs:')
        for item in missing:
            print(f'- {item}')
        return 1
    if args.check_inputs:
        print('All required inputs are present.')
        return 0
    if args.dry_run:
        _print_dry_run(args)
        return 0
    weight_img = nib.load(str(args.weight_map))
    weight_values = np.asarray(weight_img.get_fdata(), dtype=np.float64)
    (groups, metadata, roi_names, min_roi_voxels) = _analysis_roi_setup(args, weight_img)
    (rois, roi_threshold) = _build_weighted_rois(weight_values=weight_values, roi_names=roi_names, groups=groups, roi_percentile=args.roi_percentile, min_report_voxels=args.min_report_voxels, min_roi_voxels=min_roi_voxels)
    specs = _load_session_specs(args)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    roi_names = [roi.name for roi in rois]
    roi_summary = pd.DataFrame({'roi_name': [roi.name for roi in rois], 'n_weighted_voxels': [roi.n_voxels for roi in rois], 'roi_percentile': float(args.roi_percentile), 'weight_threshold': roi_threshold})
    roi_summary.to_csv(out_dir / 'weighted_roi_definition.csv', index=False)
    timeseries_dir = out_dir / 'roi_timeseries'
    timeseries_dir.mkdir(parents=True, exist_ok=True)
    networks = {}
    for spec in specs:
        session_df = _load_session_timeseries(spec, weight_img, rois)
        session_df.to_csv(timeseries_dir / f'{spec.label}.csv', index=False)
        cleaned = _clean_timeseries(session_df)
        networks[spec.label] = _mutual_information_matrix(cleaned, n_neighbors=args.mi_neighbors, random_state=args.random_state)
    _save_networks(networks, roi_names, out_dir)
    pairwise = _pairwise_network_distances(specs, networks)
    pairwise_path = out_dir / 'pairwise_metric_values.csv'
    pairwise.to_csv(pairwise_path, index=False)
    (paired_stats, paired_subject_path, paired_stats_path) = _save_paired_similarity_tests(pairwise, out_dir)
    figure_path = _plot_cross_subject_distribution(pairwise, out_dir, paired_stats=paired_stats)
    metadata.update({'weight_map': str(args.weight_map), 'roi_definition_figure': str(args.roi_definition_figure), 'roi_region_table': str(args.roi_region_table), 'session_manifest': str(args.session_manifest) if args.session_manifest.exists() else None, 'beta_root': str(args.beta_root), 'roi_percentile': float(args.roi_percentile), 'weight_threshold': roi_threshold, 'min_report_voxels': int(args.min_report_voxels), 'min_roi_voxels': int(min_roi_voxels), 'connectivity_metric': CONNECTIVITY_METRIC, 'comparison_metric': COMPARISON_METRIC, 'mi_neighbors': int(args.mi_neighbors), 'paired_subject_similarity_values': str(paired_subject_path), 'paired_subject_similarity_stats': str(paired_stats_path), 'paired_subject_similarity_primary_p': paired_stats['permutation']['permutation_p_value_two_sided'], 'sessions': [{'label': spec.label, 'subject': spec.subject, 'session': spec.session, 'state': spec.state, 'bold_path': str(spec.bold_path) if spec.bold_path else None, 'timeseries_path': str(spec.timeseries_path) if spec.timeseries_path else None, 'beta_paths': [str(path) for path in spec.beta_paths]} for spec in specs]})
    (out_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(f'Saved {pairwise_path}')
    print(f'Saved {figure_path}')
    print(f"Saved {figure_path.with_suffix('.pdf')}")
    print(f"Saved {out_dir / 'weighted_roi_definition.csv'}")
    print(f'Saved {paired_subject_path}')
    print(f'Saved {paired_stats_path}')
    print(f"Saved {out_dir / 'metadata.json'}")
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
