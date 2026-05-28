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
from matplotlib.lines import Line2D
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
DEFAULT_BOOTSTRAP_ITERATIONS = 10000
DEFAULT_BOOTSTRAP_RANDOM_STATE = 0
DEFAULT_NODE_STRENGTH_TOP_N = 16
CONNECTIVITY_METRIC = 'mutual_information_ksg'
COMPARISON_METRIC = 'laplacian_spectral_distance_signed'
INTRA_BETWEEN_FC_METRIC = 'pearson_fisher_z'
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

CONTRAST_DEFINITIONS = (
    ('off_minus_on', 'off-off', 'on-on', 'OFF-OFF - ON-ON'),
    ('off_minus_off_on', 'off-off', 'off-on', 'OFF-OFF - OFF-ON'),
    ('on_minus_off_on', 'on-on', 'off-on', 'ON-ON - OFF-ON'),
)

def _contrast_values(means):
    return {name: float(means[left_key] - means[right_key]) for (name, left_key, right_key, _) in CONTRAST_DEFINITIONS}

def _p_values_from_permutation(observed, permutation_values):
    values = np.asarray(permutation_values, dtype=np.float64)
    tolerance = 1e-12
    return {'observed': float(observed), 'p_value_two_sided': float(np.mean(np.abs(values) >= abs(observed) - tolerance)), 'p_value_greater': float(np.mean(values >= observed - tolerance)), 'p_value_less': float(np.mean(values <= observed + tolerance))}

def _exact_paired_permutation_test(complete_subjects, lookup):
    observed_assignments = _observed_paired_assignments(complete_subjects)
    (observed_means, observed_counts) = _paired_assignment_group_means(observed_assignments, lookup)
    observed_contrasts = _contrast_values(observed_means)
    n_subjects = len(complete_subjects)
    n_permutations = 1 << n_subjects
    permutation_contrasts = {name: np.empty(n_permutations, dtype=np.float64) for (name, _, _, _) in CONTRAST_DEFINITIONS}
    for mask in range(n_permutations):
        assignments = _swapped_paired_assignments(complete_subjects, mask)
        (means, _) = _paired_assignment_group_means(assignments, lookup)
        contrasts = _contrast_values(means)
        for (name, _, _, _) in CONTRAST_DEFINITIONS:
            permutation_contrasts[name][mask] = contrasts[name]
    contrast_tests = {name: _p_values_from_permutation(observed_contrasts[name], permutation_contrasts[name]) for (name, _, _, _) in CONTRAST_DEFINITIONS}
    return {'method': 'exact within-subject OFF/ON medication-label swaps for complete subjects', 'observed_means': {'off-off': float(observed_means['off-off']), 'on-on': float(observed_means['on-on']), 'off-on': float(observed_means['off-on'])}, 'observed_contrasts': observed_contrasts, 'pair_counts': observed_counts, 'n_permutations': int(n_permutations), 'contrasts': contrast_tests, 'observed_off_off_mean': float(observed_means['off-off']), 'observed_on_on_mean': float(observed_means['on-on']), 'observed_off_minus_on': observed_contrasts['off_minus_on'], 'permutation_p_value_two_sided': contrast_tests['off_minus_on']['p_value_two_sided'], 'permutation_p_value_greater': contrast_tests['off_minus_on']['p_value_greater']}

def _subject_level_similarity_values(complete_subjects, lookup):
    rows = []
    for subject in complete_subjects:
        other_subjects = [other for other in complete_subjects if other['subject'] != subject['subject']]
        off_distances = [_lookup_distance(lookup, subject['off_label'], other['off_label']) for other in other_subjects]
        on_distances = [_lookup_distance(lookup, subject['on_label'], other['on_label']) for other in other_subjects]
        off_to_on_distances = [_lookup_distance(lookup, subject['off_label'], other['on_label']) for other in other_subjects]
        on_to_off_distances = [_lookup_distance(lookup, subject['on_label'], other['off_label']) for other in other_subjects]
        off_mean = float(np.mean(off_distances))
        on_mean = float(np.mean(on_distances))
        off_on_mean = float(np.mean(off_to_on_distances + on_to_off_distances))
        rows.append({'subject': subject['subject'], 'off_label': subject['off_label'], 'on_label': subject['on_label'], 'off_mean_to_other_off': off_mean, 'on_mean_to_other_on': on_mean, 'off_on_mean_to_other_mixed': off_on_mean, 'off_minus_on': off_mean - on_mean, 'off_minus_off_on': off_mean - off_on_mean, 'on_minus_off_on': on_mean - off_on_mean})
    return pd.DataFrame(rows)

def _single_subject_level_test(subject_values, column):
    differences = subject_values[column].to_numpy(dtype=np.float64)
    differences = differences[np.isfinite(differences)]
    if differences.size < 2:
        return {'n_subjects': int(differences.size), 'mean': float(np.mean(differences)) if differences.size else float('nan'), 'paired_t_p_value_two_sided': float('nan'), 'wilcoxon_p_value_two_sided': float('nan')}
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
    return {'n_subjects': int(differences.size), 'mean': mean_difference, 'sd': sd_difference, 'sem': sem_difference, 'ci95_low': float(ci_low), 'ci95_high': float(ci_high), 'cohen_dz': float(mean_difference / sd_difference) if sd_difference > 0 else float('nan'), 'paired_t_statistic': float(t_result.statistic), 'paired_t_p_value_two_sided': float(t_result.pvalue), 'wilcoxon_statistic': wilcoxon_statistic, 'wilcoxon_p_value_two_sided': wilcoxon_p_value}

def _subject_level_similarity_tests(subject_values):
    contrast_tests = {name: _single_subject_level_test(subject_values, name) for (name, _, _, _) in CONTRAST_DEFINITIONS}
    primary = contrast_tests['off_minus_on']
    return {'n_subjects': primary['n_subjects'], 'mean_off_minus_on': primary['mean'], 'sd_off_minus_on': primary.get('sd', float('nan')), 'sem_off_minus_on': primary.get('sem', float('nan')), 'ci95_low': primary.get('ci95_low', float('nan')), 'ci95_high': primary.get('ci95_high', float('nan')), 'cohen_dz': primary.get('cohen_dz', float('nan')), 'paired_t_statistic': primary.get('paired_t_statistic', float('nan')), 'paired_t_p_value_two_sided': primary['paired_t_p_value_two_sided'], 'wilcoxon_statistic': primary.get('wilcoxon_statistic', float('nan')), 'wilcoxon_p_value_two_sided': primary['wilcoxon_p_value_two_sided'], 'contrasts': contrast_tests}

def _percentile_ci(values):
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return (float('nan'), float('nan'))
    low, high = np.percentile(finite, [2.5, 97.5])
    return (float(low), float(high))

def _summary_with_bootstrap_ci(estimate, bootstrap_values):
    ci_low, ci_high = _percentile_ci(bootstrap_values)
    return {'estimate': float(estimate), 'ci95_low': ci_low, 'ci95_high': ci_high}

def _resampled_subject_group_means(complete_subjects, lookup, subject_indices):
    values = {'off-off': [], 'on-on': [], 'off-on': []}
    for (left_position, right_position) in itertools.combinations(range(len(subject_indices)), 2):
        left_index = int(subject_indices[left_position])
        right_index = int(subject_indices[right_position])
        if left_index == right_index:
            continue
        left = complete_subjects[left_index]
        right = complete_subjects[right_index]
        values['off-off'].append(_lookup_distance(lookup, left['off_label'], right['off_label']))
        values['on-on'].append(_lookup_distance(lookup, left['on_label'], right['on_label']))
        values['off-on'].append(_lookup_distance(lookup, left['off_label'], right['on_label']))
        values['off-on'].append(_lookup_distance(lookup, left['on_label'], right['off_label']))
    means = {key: float(np.mean(group_values)) if group_values else float('nan') for (key, group_values) in values.items()}
    counts = {key: int(len(group_values)) for (key, group_values) in values.items()}
    return (means, counts)

def _subject_level_bootstrap(complete_subjects, lookup, n_bootstrap=DEFAULT_BOOTSTRAP_ITERATIONS, random_state=DEFAULT_BOOTSTRAP_RANDOM_STATE):
    n_subjects = int(len(complete_subjects))
    if n_subjects < 2:
        return {'method': 'subject-level bootstrap: resample complete subjects and recompute pairwise group means', 'n_subjects': n_subjects, 'n_bootstrap': 0}
    rng = np.random.default_rng(random_state)
    observed_means, observed_counts = _resampled_subject_group_means(complete_subjects, lookup, np.arange(n_subjects))
    bootstrap_means = {key: np.empty(int(n_bootstrap), dtype=np.float64) for key in observed_means}
    for bootstrap_index in range(int(n_bootstrap)):
        sample_indices = rng.integers(0, n_subjects, size=n_subjects)
        means, _ = _resampled_subject_group_means(complete_subjects, lookup, sample_indices)
        for key in bootstrap_means:
            bootstrap_means[key][bootstrap_index] = means[key]
    bootstrap_contrasts = {
        'off_minus_on': bootstrap_means['off-off'] - bootstrap_means['on-on'],
        'off_minus_off_on': bootstrap_means['off-off'] - bootstrap_means['off-on'],
        'on_minus_off_on': bootstrap_means['on-on'] - bootstrap_means['off-on'],
    }
    observed_contrasts = _contrast_values(observed_means)
    valid_iterations = int(np.count_nonzero(np.isfinite(bootstrap_contrasts['off_minus_on'])))
    return {'method': 'subject-level bootstrap: resample complete subjects and recompute OFF-OFF, ON-ON, and OFF-ON pairwise means', 'n_subjects': n_subjects, 'n_bootstrap': int(n_bootstrap), 'valid_bootstrap_iterations': valid_iterations, 'random_state': int(random_state), 'observed_pair_counts': observed_counts, 'means': {key: _summary_with_bootstrap_ci(observed_means[key], bootstrap_means[key]) for key in observed_means}, 'contrasts': {key: _summary_with_bootstrap_ci(observed_contrasts[key], bootstrap_contrasts[key]) for key in observed_contrasts}}

def _coded_pearson(distance_values, design_values):
    distance_values = np.asarray(distance_values, dtype=np.float64)
    design_values = np.asarray(design_values, dtype=np.float64)
    valid = np.isfinite(distance_values) & np.isfinite(design_values)
    if np.count_nonzero(valid) < 3 or np.std(distance_values[valid]) <= 0 or np.std(design_values[valid]) <= 0:
        return float('nan')
    return float(np.corrcoef(distance_values[valid], design_values[valid])[0, 1])

def _mantel_style_permutation_test(complete_subjects, lookup):
    labels = []
    subject_indices = []
    observed_states = []
    for (subject_index, subject) in enumerate(complete_subjects):
        labels.extend([subject['off_label'], subject['on_label']])
        subject_indices.extend([subject_index, subject_index])
        observed_states.extend([0, 1])
    subject_indices = np.asarray(subject_indices, dtype=np.int64)
    observed_states = np.asarray(observed_states, dtype=np.int64)
    pair_indices = [(left, right) for (left, right) in itertools.combinations(range(len(labels)), 2) if subject_indices[left] != subject_indices[right]]
    left_indices = np.asarray([left for (left, _) in pair_indices], dtype=np.int64)
    right_indices = np.asarray([right for (_, right) in pair_indices], dtype=np.int64)
    distances = np.asarray([_lookup_distance(lookup, labels[left], labels[right]) for (left, right) in pair_indices], dtype=np.float64)

    def design_for_mask(mask):
        swaps = np.asarray([(mask >> subject_index) & 1 for subject_index in range(len(complete_subjects))], dtype=np.int64)
        states = observed_states ^ swaps[subject_indices]
        left_states = states[left_indices]
        right_states = states[right_indices]
        return np.where((left_states == 0) & (right_states == 0), 1.0, np.where((left_states == 1) & (right_states == 1), -1.0, 0.0))

    observed_r = _coded_pearson(distances, design_for_mask(0))
    n_permutations = 1 << len(complete_subjects)
    permutation_r = np.empty(n_permutations, dtype=np.float64)
    for mask in range(n_permutations):
        permutation_r[mask] = _coded_pearson(distances, design_for_mask(mask))
    p_values = _p_values_from_permutation(observed_r, permutation_r)
    return {'method': 'Mantel-style test: correlate network-distance vector with medication contrast coding, then exactly swap OFF/ON labels within complete subjects', 'coding': 'OFF-OFF=1, ON-ON=-1, OFF-ON=0; positive r means OFF-OFF distances are larger than ON-ON distances', 'n_pairs': int(distances.size), 'n_permutations': int(n_permutations), 'pearson_r': float(observed_r), 'p_value_two_sided': p_values['p_value_two_sided'], 'p_value_greater': p_values['p_value_greater'], 'p_value_less': p_values['p_value_less']}

def _effect_size_summary(stats_summary, subject_tests, bootstrap):
    observed_means = stats_summary['observed_means']
    off_minus_on = stats_summary['observed_contrasts']['off_minus_on']
    percent_on_lower = 100.0 * off_minus_on / observed_means['off-off'] if observed_means['off-off'] else float('nan')
    primary_subject = subject_tests['contrasts']['off_minus_on']
    primary_bootstrap = bootstrap['contrasts']['off_minus_on']
    return {'primary_contrast': 'OFF-OFF - ON-ON', 'raw_difference': float(off_minus_on), 'bootstrap_ci95_low': primary_bootstrap['ci95_low'], 'bootstrap_ci95_high': primary_bootstrap['ci95_high'], 'percent_on_on_lower_than_off_off': float(percent_on_lower), 'cohen_dz_subject_level': primary_subject.get('cohen_dz', float('nan')), 'subject_t_ci95_low': primary_subject.get('ci95_low', float('nan')), 'subject_t_ci95_high': primary_subject.get('ci95_high', float('nan'))}

def _paired_similarity_tests(pairwise):
    metric_rows = _metric_pairwise_rows(pairwise)
    (complete_subjects, incomplete_subjects) = _complete_off_on_subjects(metric_rows)
    if len(complete_subjects) < 2:
        raise RuntimeError('At least two complete OFF/ON subjects are required for paired medication similarity tests')
    lookup = _distance_lookup(metric_rows)
    permutation = _exact_paired_permutation_test(complete_subjects, lookup)
    subject_values = _subject_level_similarity_values(complete_subjects, lookup)
    subject_tests = _subject_level_similarity_tests(subject_values)
    bootstrap = _subject_level_bootstrap(complete_subjects, lookup)
    mantel_style = _mantel_style_permutation_test(complete_subjects, lookup)
    observed = {'means': permutation['observed_means'], 'contrasts': permutation['observed_contrasts'], 'pair_counts': permutation['pair_counts']}
    stats_summary = {'test_scope': 'complete subjects with one OFF and one ON session; cross-subject distances only', 'hypothesis': 'OFF-OFF distances are greater than ON-ON distances when medication makes networks more similar across subjects', 'complete_subjects': complete_subjects, 'incomplete_subjects': incomplete_subjects, 'observed': observed, 'medication_label_permutation': permutation, 'permutation': permutation, 'subject_level': subject_tests, 'subject_level_bootstrap': bootstrap, 'mantel_style': mantel_style}
    stats_summary['effect_size'] = _effect_size_summary(permutation, subject_tests, bootstrap)
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

def _benjamini_hochberg_pvalues(p_values):
    p_values = np.asarray(p_values, dtype=np.float64)
    adjusted = np.full(p_values.shape, np.nan, dtype=np.float64)
    finite_indices = np.flatnonzero(np.isfinite(p_values))
    if finite_indices.size == 0:
        return adjusted
    ordered = finite_indices[np.argsort(p_values[finite_indices])]
    ranked = p_values[ordered]
    ranks = np.arange(1, ordered.size + 1, dtype=np.float64)
    ordered_adjusted = ranked * float(ordered.size) / ranks
    ordered_adjusted = np.minimum.accumulate(ordered_adjusted[::-1])[::-1]
    adjusted[ordered] = np.minimum(ordered_adjusted, 1.0)
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

def _add_star_legend(ax, tests):
    shown_stars = {str(test.get('stars', '')) for test in tests}
    labels = []
    if '***' in shown_stars:
        labels.append('*** Holm p < .001')
    if '**' in shown_stars:
        labels.append('** Holm p < .01')
    if '*' in shown_stars:
        labels.append('* Holm p < .05')
    if not labels:
        return
    handles = [Line2D([], [], linestyle='none', label=label) for label in labels]
    ax.legend(handles=handles, loc='upper right', frameon=False, handlelength=0, handletextpad=0, borderpad=0.1, labelspacing=0.2, fontsize=7)

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
    group_tests = _pairwise_group_tests(subset, class_order, groups)
    if not any(np.isfinite(float(test.get('p_value', np.nan))) for test in group_tests):
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
    _add_star_legend(ax, group_tests)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / 'cross_subject_only_laplacian_spectral_distance_signed_distribution.png'
    pdf_path = png_path.with_suffix('.pdf')
    fig.savefig(png_path, dpi=190, bbox_inches='tight', pad_inches=0.04)
    fig.savefig(pdf_path, bbox_inches='tight', pad_inches=0.04)
    plt.close(fig)
    return png_path

def _node_strength_values(specs, networks, roi_names, paired_stats):
    complete_labels = _complete_subject_labels_from_stats(paired_stats)
    rows = []
    for spec in specs:
        if complete_labels is not None and spec.label not in complete_labels:
            continue
        matrix = np.asarray(networks[spec.label], dtype=np.float64)
        strengths = np.sum(matrix, axis=1)
        for (roi_name, strength) in zip(roi_names, strengths):
            rows.append({'connectivity_metric': CONNECTIVITY_METRIC, 'analysis': 'node_strength_mi', 'label': spec.label, 'subject': spec.subject, 'session': spec.session, 'state': spec.state, 'roi': roi_name, 'node_strength': float(strength)})
    values = pd.DataFrame(rows)
    if values.empty:
        raise RuntimeError('No node-strength values were available for complete OFF/ON subjects')
    return values.sort_values(['roi', 'subject', 'state']).reset_index(drop=True)

def _node_strength_summary(values):
    rows = []
    for (roi_name, roi_values) in values.groupby('roi', sort=False):
        pivot = roi_values.pivot_table(index='subject', columns='state', values='node_strength', aggfunc='first')
        if 'off' not in pivot.columns or 'on' not in pivot.columns:
            continue
        pivot = pivot.loc[:, ['off', 'on']].dropna()
        if pivot.empty:
            continue
        differences = pivot['on'].to_numpy(dtype=np.float64) - pivot['off'].to_numpy(dtype=np.float64)
        try:
            wilcoxon = stats.wilcoxon(differences, alternative='two-sided')
            wilcoxon_stat = float(wilcoxon.statistic)
            p_value = float(wilcoxon.pvalue)
        except ValueError:
            wilcoxon_stat = float('nan')
            p_value = float('nan')
        rows.append({'roi': roi_name, 'mean_delta': float(np.mean(differences)), 'median_delta': float(np.median(differences)), 'wilcoxon_stat': wilcoxon_stat, 'p_value': p_value, 'mean_off': float(np.mean(pivot['off'])), 'mean_on': float(np.mean(pivot['on'])), 'n_subjects': int(pivot.shape[0])})
    summary = pd.DataFrame(rows)
    if summary.empty:
        raise RuntimeError('No paired OFF/ON node-strength summaries could be computed')
    summary['p_fdr'] = _benjamini_hochberg_pvalues(summary['p_value'].to_numpy(dtype=np.float64))
    summary['sig_fdr05'] = summary['p_fdr'] < 0.05
    columns = ['roi', 'mean_delta', 'median_delta', 'wilcoxon_stat', 'p_value', 'mean_off', 'mean_on', 'p_fdr', 'sig_fdr05', 'n_subjects']
    return summary.loc[:, columns].sort_values(['p_value', 'roi']).reset_index(drop=True)

def _format_roi_label(roi_name):
    parts = str(roi_name).rsplit('_', 1)
    if len(parts) == 2 and parts[1] in {'L', 'R'}:
        return f"{parts[0].replace('_', ' ')} ({parts[1]})"
    return str(roi_name).replace('_', ' ')

def _format_p_value(p_value):
    if not np.isfinite(p_value):
        return 'n/a'
    if p_value < 0.001:
        return '<0.001'
    return f'{p_value:.3f}'.lstrip('0')

def _plot_node_strength_boxplots(values, summary, out_dir, top_n=DEFAULT_NODE_STRENGTH_TOP_N):
    selected = summary.head(int(top_n))['roi'].astype(str).tolist()
    if not selected:
        raise RuntimeError('No node-strength ROIs were available for plotting')
    stats_by_roi = summary.set_index('roi')
    n_cols = 4
    n_rows = int(np.ceil(len(selected) / n_cols))
    (fig, axes) = plt.subplots(n_rows, n_cols, figsize=(10.6, 2.2 * n_rows), squeeze=False)
    colors = {'off': '#4C78A8', 'on': '#D65F5F'}
    rng = np.random.default_rng(0)
    for (roi_name, ax) in zip(selected, axes.ravel()):
        roi_values = values.loc[values['roi'] == roi_name]
        pivot = roi_values.pivot_table(index='subject', columns='state', values='node_strength', aggfunc='first')
        pivot = pivot.loc[:, ['off', 'on']].dropna().sort_index()
        off = pivot['off'].to_numpy(dtype=np.float64)
        on = pivot['on'].to_numpy(dtype=np.float64)
        box = ax.boxplot([off, on], positions=[0, 1], widths=0.48, patch_artist=True, showfliers=False, medianprops={'color': '#1a1a1a', 'linewidth': 1.1}, whiskerprops={'color': '#333333', 'linewidth': 0.9}, capprops={'color': '#333333', 'linewidth': 0.9})
        for (patch, state) in zip(box['boxes'], ['off', 'on']):
            patch.set_facecolor(colors[state])
            patch.set_alpha(0.32)
            patch.set_edgecolor('#333333')
            patch.set_linewidth(0.9)
        for (_, row) in pivot.iterrows():
            ax.plot([0, 1], [row['off'], row['on']], color='#9a9a9a', linewidth=0.55, alpha=0.38, zorder=1)
        for (x_value, state_values, color) in [(0, off, colors['off']), (1, on, colors['on'])]:
            jitter = rng.normal(loc=x_value, scale=0.035, size=state_values.size)
            ax.scatter(jitter, state_values, s=13, color=color, edgecolor='white', linewidth=0.35, alpha=0.85, zorder=3)
        roi_stats = stats_by_roi.loc[roi_name]
        title = _format_roi_label(roi_name)
        subtitle = f"p={_format_p_value(float(roi_stats['p_value']))}, q={_format_p_value(float(roi_stats['p_fdr']))}"
        ax.set_title(f'{title}\n{subtitle}', fontsize=8.7, pad=5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['OFF', 'ON'], fontsize=8)
        ax.tick_params(axis='y', labelsize=8)
        ax.grid(axis='y', color='#dddddd', linewidth=0.55, alpha=0.8)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        y_values = np.concatenate([off, on])
        y_min = float(np.min(y_values))
        y_max = float(np.max(y_values))
        y_span = max(y_max - y_min, 1e-06)
        ax.set_ylim(y_min - 0.1 * y_span, y_max + 0.14 * y_span)
    for ax in axes.ravel()[len(selected):]:
        ax.axis('off')
    for row_index in range(n_rows):
        axes[row_index, 0].set_ylabel('Node strength', fontsize=8.5)
    fig.suptitle('Node strength by medication state (MI connectivity)', fontsize=12.5, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / 'node_strength_mi_boxplot.png'
    pdf_path = png_path.with_suffix('.pdf')
    fig.savefig(png_path, dpi=320, bbox_inches='tight', pad_inches=0.04)
    fig.savefig(pdf_path, bbox_inches='tight', pad_inches=0.04)
    plt.close(fig)
    return png_path

def _save_node_strength_analysis(specs, networks, roi_names, out_dir, paired_stats, top_n=DEFAULT_NODE_STRENGTH_TOP_N):
    values = _node_strength_values(specs, networks, roi_names, paired_stats)
    summary = _node_strength_summary(values)
    values_path = out_dir / 'node_strength_mi_values.csv'
    summary_path = out_dir / 'node_strength_mi_results.csv'
    values.to_csv(values_path, index=False)
    summary.to_csv(summary_path, index=False)
    figure_path = _plot_node_strength_boxplots(values, summary, out_dir, top_n=top_n)
    return (summary, values_path, summary_path, figure_path)

def _roi_hemisphere(roi_name):
    parts = str(roi_name).rsplit('_', 1)
    if len(parts) == 2 and parts[1] in {'L', 'R'}:
        return parts[1]
    return None

def _hemisphere_edge_values(specs, networks, roi_names):
    hemispheres = [_roi_hemisphere(name) for name in roi_names]
    if any(hemisphere is None for hemisphere in hemispheres):
        raise RuntimeError('Within-vs-between hemisphere analysis requires lateralized ROI names ending in _L or _R; rerun with --split-hemispheres.')
    rows = []
    for spec in specs:
        matrix = np.asarray(networks[spec.label], dtype=np.float64)
        for (left, right) in itertools.combinations(range(len(roi_names)), 2):
            left_hemi = hemispheres[left]
            right_hemi = hemispheres[right]
            edge_class = 'within_hemisphere' if left_hemi == right_hemi else 'between_hemisphere'
            edge_subclass = f'within_{left_hemi.lower()}_hemisphere' if left_hemi == right_hemi else 'between_hemisphere'
            rows.append({
                'connectivity_metric': CONNECTIVITY_METRIC,
                'analysis': 'within_vs_between_hemisphere_fc',
                'label': spec.label,
                'subject': spec.subject,
                'session': spec.session,
                'state': spec.state,
                'roi_a': roi_names[left],
                'roi_b': roi_names[right],
                'hemisphere_a': left_hemi,
                'hemisphere_b': right_hemi,
                'edge_class': edge_class,
                'edge_subclass': edge_subclass,
                'edge_weight': float(matrix[left, right]),
            })
    values = pd.DataFrame(rows)
    if values.empty:
        raise RuntimeError('No hemisphere-classified ROI edges were available')
    return values

def _mean_finite(series):
    values = np.asarray(series, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else float('nan')

def _count_finite(series):
    return int(np.count_nonzero(np.isfinite(np.asarray(series, dtype=np.float64))))

def _hemisphere_fc_session_values(edge_values):
    rows = []
    for ((label, subject, session, state), values) in edge_values.groupby(['label', 'subject', 'session', 'state'], sort=True):
        within = values.loc[values['edge_class'] == 'within_hemisphere', 'edge_weight']
        between = values.loc[values['edge_class'] == 'between_hemisphere', 'edge_weight']
        left = values.loc[values['edge_subclass'] == 'within_l_hemisphere', 'edge_weight']
        right = values.loc[values['edge_subclass'] == 'within_r_hemisphere', 'edge_weight']
        rows.append({
            'connectivity_metric': CONNECTIVITY_METRIC,
            'analysis': 'within_vs_between_hemisphere_fc',
            'label': label,
            'subject': subject,
            'session': session,
            'state': state,
            'within_hemisphere_mean_edge': _mean_finite(within),
            'between_hemisphere_mean_edge': _mean_finite(between),
            'left_within_hemisphere_mean_edge': _mean_finite(left),
            'right_within_hemisphere_mean_edge': _mean_finite(right),
            'within_hemisphere_n_edges': _count_finite(within),
            'between_hemisphere_n_edges': _count_finite(between),
            'left_within_hemisphere_n_edges': _count_finite(left),
            'right_within_hemisphere_n_edges': _count_finite(right),
        })
    return pd.DataFrame(rows).sort_values(['subject', 'session']).reset_index(drop=True)

def _complete_hemisphere_subject_deltas(session_values):
    rows = []
    for (subject, subject_values) in session_values.groupby('subject', sort=True):
        off_rows = subject_values.loc[subject_values['state'] == 'off']
        on_rows = subject_values.loc[subject_values['state'] == 'on']
        if off_rows.shape[0] != 1 or on_rows.shape[0] != 1:
            continue
        off = off_rows.iloc[0]
        on = on_rows.iloc[0]
        within_off = float(off['within_hemisphere_mean_edge'])
        within_on = float(on['within_hemisphere_mean_edge'])
        between_off = float(off['between_hemisphere_mean_edge'])
        between_on = float(on['between_hemisphere_mean_edge'])
        left_off = float(off['left_within_hemisphere_mean_edge'])
        left_on = float(on['left_within_hemisphere_mean_edge'])
        right_off = float(off['right_within_hemisphere_mean_edge'])
        right_on = float(on['right_within_hemisphere_mean_edge'])
        rows.append({
            'subject': subject,
            'off_label': off['label'],
            'on_label': on['label'],
            'within_hemisphere_off': within_off,
            'within_hemisphere_on': within_on,
            'within_hemisphere_delta_on_minus_off': within_on - within_off,
            'between_hemisphere_off': between_off,
            'between_hemisphere_on': between_on,
            'between_hemisphere_delta_on_minus_off': between_on - between_off,
            'within_minus_between_hemisphere_delta': (within_on - within_off) - (between_on - between_off),
            'left_within_hemisphere_off': left_off,
            'left_within_hemisphere_on': left_on,
            'left_within_hemisphere_delta_on_minus_off': left_on - left_off,
            'right_within_hemisphere_off': right_off,
            'right_within_hemisphere_on': right_on,
            'right_within_hemisphere_delta_on_minus_off': right_on - right_off,
            'left_minus_right_within_hemisphere_delta': (left_on - left_off) - (right_on - right_off),
        })
    values = pd.DataFrame(rows)
    if values.empty:
        return values
    return values.sort_values('subject').reset_index(drop=True)

def _hemisphere_fc_test_rows(subject_deltas):
    tests = [
        ('within_hemisphere_on_minus_off', 'within_hemisphere_delta_on_minus_off', 'ON - OFF within-hemisphere FC'),
        ('between_hemisphere_on_minus_off', 'between_hemisphere_delta_on_minus_off', 'ON - OFF between-hemisphere FC'),
        ('within_minus_between_hemisphere_delta', 'within_minus_between_hemisphere_delta', '(ON - OFF within-hemisphere FC) - (ON - OFF between-hemisphere FC)'),
        ('left_within_hemisphere_on_minus_off', 'left_within_hemisphere_delta_on_minus_off', 'ON - OFF left within-hemisphere FC'),
        ('right_within_hemisphere_on_minus_off', 'right_within_hemisphere_delta_on_minus_off', 'ON - OFF right within-hemisphere FC'),
        ('left_minus_right_within_hemisphere_delta', 'left_minus_right_within_hemisphere_delta', '(ON - OFF left within-hemisphere FC) - (ON - OFF right within-hemisphere FC)'),
    ]
    rows = []
    details = {}
    for (analysis, column, description) in tests:
        result = _single_subject_level_test(subject_deltas, column)
        details[analysis] = result
        row = {'analysis': analysis, 'value_column': column, 'description': description}
        row.update(result)
        rows.append(row)
    return (pd.DataFrame(rows), details)

def _plot_hemisphere_fc(subject_deltas, results, out_dir):
    if subject_deltas.empty:
        raise RuntimeError('No complete OFF/ON subjects were available for hemisphere FC plotting')
    colors = {'off': '#4C78A8', 'on': '#D65F5F', 'delta': '#333333'}
    (fig, axes) = plt.subplots(1, 2, figsize=(8.0, 3.65))
    rng = np.random.default_rng(0)
    paired_specs = [
        ('Within hemi', 'within_hemisphere_off', 'within_hemisphere_on', 0.0),
        ('Between hemi', 'between_hemisphere_off', 'between_hemisphere_on', 1.1),
    ]
    for (_, off_column, on_column, offset) in paired_specs:
        off_values = subject_deltas[off_column].to_numpy(dtype=np.float64)
        on_values = subject_deltas[on_column].to_numpy(dtype=np.float64)
        for (off_value, on_value) in zip(off_values, on_values):
            axes[0].plot([offset, offset + 0.32], [off_value, on_value], color='#9a9a9a', linewidth=0.7, alpha=0.55, zorder=1)
        axes[0].scatter(rng.normal(offset, 0.018, off_values.size), off_values, s=20, color=colors['off'], edgecolor='white', linewidth=0.35, zorder=2, label='OFF' if offset == 0.0 else None)
        axes[0].scatter(rng.normal(offset + 0.32, 0.018, on_values.size), on_values, s=20, color=colors['on'], edgecolor='white', linewidth=0.35, zorder=2, label='ON' if offset == 0.0 else None)
    axes[0].set_xticks([0.16, 1.26])
    axes[0].set_xticklabels([item[0] for item in paired_specs])
    axes[0].set_ylabel('Mean MI edge weight')
    axes[0].set_title('Medication State', fontsize=10)
    axes[0].legend(frameon=False, fontsize=8, loc='best')

    delta_specs = [
        ('Within hemi', 'within_hemisphere_delta_on_minus_off', 0.0),
        ('Between hemi', 'between_hemisphere_delta_on_minus_off', 1.0),
    ]
    for (_, row) in subject_deltas.iterrows():
        values = [row['within_hemisphere_delta_on_minus_off'], row['between_hemisphere_delta_on_minus_off']]
        axes[1].plot([0, 1], values, color='#9a9a9a', linewidth=0.7, alpha=0.55, zorder=1)
    for (name, column, x_value) in delta_specs:
        values = subject_deltas[column].to_numpy(dtype=np.float64)
        axes[1].scatter(rng.normal(x_value, 0.025, values.size), values, s=22, color=colors['delta'], edgecolor='white', linewidth=0.35, zorder=2)
        mean_value = float(np.nanmean(values))
        axes[1].plot([x_value - 0.16, x_value + 0.16], [mean_value, mean_value], color='#d62728', linewidth=1.4, zorder=3)
    axes[1].axhline(0, color='#666666', linewidth=0.8, linestyle='--')
    axes[1].set_xticks([item[2] for item in delta_specs])
    axes[1].set_xticklabels([item[0] for item in delta_specs])
    axes[1].set_ylabel('ON - OFF edge-weight change')
    p_value = results.get('within_minus_between_hemisphere_delta', {}).get('paired_t_p_value_two_sided', np.nan)
    axes[1].set_title(f'Change Difference p={_format_p_value(float(p_value))}', fontsize=10)

    for ax in axes:
        ax.grid(axis='y', color='#dddddd', linewidth=0.55, alpha=0.8)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=8.5)
    fig.suptitle('Within-hemisphere vs between-hemisphere FC', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / 'within_vs_between_hemisphere_fc_medication_change.png'
    pdf_path = png_path.with_suffix('.pdf')
    fig.savefig(png_path, dpi=320, bbox_inches='tight', pad_inches=0.04)
    fig.savefig(pdf_path, bbox_inches='tight', pad_inches=0.04)
    plt.close(fig)
    return png_path

def _write_hemisphere_fc_method(path):
    text = (
        '# Within-Hemisphere vs Between-Hemisphere FC Method\n\n'
        'This analysis uses the lateralized ROI network matrices produced by the medication-effects pipeline. '
        'ROIs must have names ending in `_L` or `_R`. For each subject/session, upper-triangle ROI edges were '
        'classified as within-hemisphere when both ROIs had the same hemisphere suffix, and between-hemisphere '
        'when one ROI was left-lateralized and the other was right-lateralized. The session-level within- and '
        'between-hemisphere values are unweighted means of the mutual-information edge weights in each class.\n\n'
        'Medication effects were evaluated within complete subjects as ON minus OFF separately for within- and '
        'between-hemisphere edges. The primary comparison was the paired subject-level contrast '
        '(ON - OFF within-hemisphere FC) - (ON - OFF between-hemisphere FC). Positive values mean medication '
        'increased within-hemisphere connectivity more than between-hemisphere connectivity.\n'
    )
    path.write_text(text, encoding='utf-8')

def _save_hemisphere_fc_analysis(specs, networks, roi_names, out_dir):
    edge_values = _hemisphere_edge_values(specs, networks, roi_names)
    session_values = _hemisphere_fc_session_values(edge_values)
    subject_deltas = _complete_hemisphere_subject_deltas(session_values)
    if subject_deltas.empty:
        raise RuntimeError('No complete OFF/ON subjects were available for hemisphere FC analysis')
    (results_table, results) = _hemisphere_fc_test_rows(subject_deltas)
    result_summary = {
        'connectivity_metric': CONNECTIVITY_METRIC,
        'method': 'Mean mutual-information ROI edge weights are summarized separately for within-hemisphere and between-hemisphere ROI pairs.',
        'n_sessions': int(session_values.shape[0]),
        'n_complete_subjects': int(subject_deltas.shape[0]),
        'tests': results,
    }
    edge_path = out_dir / 'within_vs_between_hemisphere_fc_edge_values.csv'
    session_path = out_dir / 'within_vs_between_hemisphere_fc_session_values.csv'
    delta_path = out_dir / 'within_vs_between_hemisphere_fc_subject_deltas.csv'
    results_path = out_dir / 'within_vs_between_hemisphere_fc_results.csv'
    json_path = out_dir / 'within_vs_between_hemisphere_fc_results.json'
    method_path = out_dir / 'within_vs_between_hemisphere_fc_method.md'
    edge_values.to_csv(edge_path, index=False)
    session_values.to_csv(session_path, index=False)
    subject_deltas.to_csv(delta_path, index=False)
    results_table.to_csv(results_path, index=False)
    json_path.write_text(json.dumps(result_summary, indent=2), encoding='utf-8')
    _write_hemisphere_fc_method(method_path)
    figure_path = _plot_hemisphere_fc(subject_deltas, results, out_dir)
    return {
        'summary': result_summary,
        'edge_values': edge_path,
        'session_values': session_path,
        'subject_deltas': delta_path,
        'results': results_path,
        'results_json': json_path,
        'method': method_path,
        'figure': figure_path,
    }

def _extract_roi_voxel_timeseries(bold_path, reference_img, rois):
    img = nib.load(str(bold_path))
    _check_image_grid(reference_img, img, str(bold_path))
    if len(img.shape) != 4:
        raise RuntimeError(f'{bold_path} must be a 4D BOLD NIfTI')
    data = img.get_fdata(dtype=np.float32)
    return {roi.name: np.asarray(data[roi.mask, :], dtype=np.float64).T for roi in rois}

def _extract_roi_voxel_timeseries_from_beta(beta_path, reference_img, rois):
    data = np.load(beta_path, mmap_mode='r')
    if data.ndim != 4:
        raise RuntimeError(f'{beta_path} must be a 4D cleaned beta volume with shape (x, y, z, trials)')
    if tuple(data.shape[:3]) != tuple(reference_img.shape[:3]):
        raise RuntimeError(f'{beta_path} shape {data.shape[:3]} differs from the weight-map grid {reference_img.shape[:3]}')
    return {roi.name: np.asarray(data[roi.mask, :], dtype=np.float64).T for roi in rois}

def _load_session_voxel_timeseries(spec, reference_img, rois):
    roi_names = [roi.name for roi in rois]
    frames = {name: [] for name in roi_names}
    if spec.beta_paths:
        for beta_path in spec.beta_paths:
            run_data = _extract_roi_voxel_timeseries_from_beta(beta_path, reference_img, rois)
            for name in roi_names:
                frames[name].append(run_data[name])
        return {name: np.concatenate(frames[name], axis=0) for name in roi_names}
    if spec.bold_path is not None:
        return _extract_roi_voxel_timeseries(spec.bold_path, reference_img, rois)
    raise RuntimeError(f'{spec.label} has only ROI mean time series; voxel-level intra-ROI FC requires beta_path or bold_path input')

def _clean_feature_timeseries(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise RuntimeError(f'Expected a 2D time-by-feature matrix, got shape {values.shape}')
    finite_timepoints = np.all(np.isfinite(values), axis=1)
    values = values[finite_timepoints, :]
    if values.shape[0] < 4:
        return (np.empty((values.shape[0], 0), dtype=np.float64), np.zeros(values.shape[1], dtype=bool))
    finite_features = np.all(np.isfinite(values), axis=0)
    values = values[:, finite_features]
    if values.shape[1] < 2:
        return (np.empty((values.shape[0], 0), dtype=np.float64), finite_features)
    centered = values - np.mean(values, axis=0, keepdims=True)
    scale = np.std(centered, axis=0, ddof=1)
    valid_scaled = np.isfinite(scale) & (scale > 0)
    valid_features = finite_features.copy()
    valid_features[finite_features] = valid_scaled
    centered = centered[:, valid_scaled]
    if centered.shape[1] == 0:
        return (centered, valid_features)
    centered /= scale[valid_scaled][None, :]
    return (centered, valid_features)

def _mean_pairwise_fisher_z(cleaned, weights=None):
    n_timepoints, n_features = cleaned.shape
    if n_timepoints < 4 or n_features < 2:
        return {
            'mean_z': float('nan'),
            'mean_r': float('nan'),
            'n_features': int(n_features),
            'n_pairs': 0,
        }
    corr = (cleaned.T @ cleaned) / float(n_timepoints - 1)
    corr = np.clip(corr, -0.999999, 0.999999)
    pair_indices = np.triu_indices(n_features, k=1)
    pair_z = np.arctanh(corr[pair_indices])
    valid = np.isfinite(pair_z)
    if weights is not None:
        weights = np.asarray(weights, dtype=np.float64)
        pair_weights = weights[pair_indices[0]] * weights[pair_indices[1]]
        valid &= np.isfinite(pair_weights) & (pair_weights > 0)
        if np.any(valid):
            mean_z = float(np.average(pair_z[valid], weights=pair_weights[valid]))
        else:
            mean_z = float('nan')
    else:
        mean_z = float(np.mean(pair_z[valid])) if np.any(valid) else float('nan')
    return {
        'mean_z': mean_z,
        'mean_r': float(np.tanh(mean_z)) if np.isfinite(mean_z) else float('nan'),
        'n_features': int(n_features),
        'n_pairs': int(np.count_nonzero(valid)),
    }

def _mean_pairwise_fisher_z_pairwise_complete(values, weights=None, min_overlap=4):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise RuntimeError(f'Expected a 2D time-by-feature matrix, got shape {values.shape}')
    finite = np.isfinite(values)
    usable_features = np.sum(finite, axis=0) >= int(min_overlap)
    values = values[:, usable_features]
    finite = finite[:, usable_features]
    n_features = values.shape[1]
    if n_features < 2:
        return {
            'mean_z': float('nan'),
            'mean_r': float('nan'),
            'n_features': int(n_features),
            'n_pairs': 0,
        }
    if weights is not None:
        weights = np.asarray(weights, dtype=np.float64)[usable_features]
    mask = finite.astype(np.float64)
    filled = np.where(finite, values, 0.0)
    filled_squared = filled * filled
    counts = mask.T @ mask
    sums_left = filled.T @ mask
    sums_right = sums_left.T
    sums_sq_left = filled_squared.T @ mask
    sums_sq_right = sums_sq_left.T
    sums_cross = filled.T @ filled
    with np.errstate(divide='ignore', invalid='ignore'):
        covariance_num = sums_cross - (sums_left * sums_right / counts)
        variance_left = sums_sq_left - (sums_left * sums_left / counts)
        variance_right = sums_sq_right - (sums_right * sums_right / counts)
        corr = covariance_num / np.sqrt(variance_left * variance_right)
    pair_indices = np.triu_indices(n_features, k=1)
    pair_corr = corr[pair_indices]
    valid = (counts[pair_indices] >= int(min_overlap)) & np.isfinite(pair_corr)
    valid &= (variance_left[pair_indices] > 0) & (variance_right[pair_indices] > 0)
    if not np.any(valid):
        return {
            'mean_z': float('nan'),
            'mean_r': float('nan'),
            'n_features': int(n_features),
            'n_pairs': 0,
        }
    pair_z = np.arctanh(np.clip(pair_corr[valid], -0.999999, 0.999999))
    if weights is not None:
        pair_weights = weights[pair_indices[0]] * weights[pair_indices[1]]
        pair_weights = pair_weights[valid]
        weight_valid = np.isfinite(pair_weights) & (pair_weights > 0)
        pair_z = pair_z[weight_valid]
        pair_weights = pair_weights[weight_valid]
        if pair_z.size == 0:
            mean_z = float('nan')
        else:
            mean_z = float(np.average(pair_z, weights=pair_weights))
    else:
        mean_z = float(np.mean(pair_z))
    return {
        'mean_z': mean_z,
        'mean_r': float(np.tanh(mean_z)) if np.isfinite(mean_z) else float('nan'),
        'n_features': int(n_features),
        'n_pairs': int(pair_z.size),
    }

def _between_roi_fc_summary(cleaned_roi_timeseries):
    summary = _mean_pairwise_fisher_z(cleaned_roi_timeseries)
    return {
        'between_roi_mean_z': summary['mean_z'],
        'between_roi_mean_r': summary['mean_r'],
        'n_rois': summary['n_features'],
        'n_between_roi_pairs': summary['n_pairs'],
    }

def _intra_roi_fc_values(spec, voxel_timeseries, rois):
    roi_rows = []
    roi_mean_values = []
    for roi in rois:
        cleaned, valid_voxels = _clean_feature_timeseries(voxel_timeseries[roi.name])
        weights = np.asarray(roi.weights, dtype=np.float64)[valid_voxels]
        summary = _mean_pairwise_fisher_z(cleaned, weights=weights)
        estimation = 'common_complete_timepoints'
        if not np.isfinite(summary['mean_z']):
            summary = _mean_pairwise_fisher_z_pairwise_complete(voxel_timeseries[roi.name], weights=roi.weights)
            estimation = 'pairwise_complete_timepoints'
        if np.isfinite(summary['mean_z']):
            roi_mean_values.append(summary['mean_z'])
        roi_rows.append({
            'connectivity_metric': INTRA_BETWEEN_FC_METRIC,
            'analysis': 'intra_roi_voxel_fc',
            'label': spec.label,
            'subject': spec.subject,
            'session': spec.session,
            'state': spec.state,
            'roi': roi.name,
            'n_weighted_voxels': int(roi.n_voxels),
            'n_used_voxels': summary['n_features'],
            'n_voxel_pairs': summary['n_pairs'],
            'intra_roi_mean_z': summary['mean_z'],
            'intra_roi_mean_r': summary['mean_r'],
            'intra_roi_estimation': estimation,
        })
    session_mean_z = float(np.mean(roi_mean_values)) if roi_mean_values else float('nan')
    session_summary = {
        'within_roi_mean_z': session_mean_z,
        'within_roi_mean_r': float(np.tanh(session_mean_z)) if np.isfinite(session_mean_z) else float('nan'),
        'n_within_roi_values': int(len(roi_mean_values)),
        'within_roi_aggregation': 'equal_roi_mean_of_roi_level_weighted_voxel_pair_fisher_z',
    }
    return (roi_rows, session_summary)

def _complete_intra_between_subject_deltas(session_values):
    rows = []
    for (subject, subject_values) in session_values.groupby('subject', sort=True):
        off_rows = subject_values.loc[subject_values['state'] == 'off']
        on_rows = subject_values.loc[subject_values['state'] == 'on']
        if off_rows.shape[0] != 1 or on_rows.shape[0] != 1:
            continue
        off = off_rows.iloc[0]
        on = on_rows.iloc[0]
        within_off_z = float(off['within_roi_mean_z'])
        within_on_z = float(on['within_roi_mean_z'])
        between_off_z = float(off['between_roi_mean_z'])
        between_on_z = float(on['between_roi_mean_z'])
        rows.append({
            'subject': subject,
            'off_label': off['label'],
            'on_label': on['label'],
            'within_roi_off_z': within_off_z,
            'within_roi_on_z': within_on_z,
            'within_roi_off_r': float(np.tanh(within_off_z)),
            'within_roi_on_r': float(np.tanh(within_on_z)),
            'within_roi_delta_z_on_minus_off': within_on_z - within_off_z,
            'within_roi_delta_r_on_minus_off': float(np.tanh(within_on_z) - np.tanh(within_off_z)),
            'between_roi_off_z': between_off_z,
            'between_roi_on_z': between_on_z,
            'between_roi_off_r': float(np.tanh(between_off_z)),
            'between_roi_on_r': float(np.tanh(between_on_z)),
            'between_roi_delta_z_on_minus_off': between_on_z - between_off_z,
            'between_roi_delta_r_on_minus_off': float(np.tanh(between_on_z) - np.tanh(between_off_z)),
            'within_minus_between_delta_z': (within_on_z - within_off_z) - (between_on_z - between_off_z),
            'within_minus_between_delta_r': (float(np.tanh(within_on_z) - np.tanh(within_off_z)) - float(np.tanh(between_on_z) - np.tanh(between_off_z))),
        })
    values = pd.DataFrame(rows)
    if values.empty:
        return values
    return values.sort_values('subject').reset_index(drop=True)

def _intra_between_fc_test_rows(subject_deltas):
    tests = [
        ('within_roi_on_minus_off', 'within_roi_delta_z_on_minus_off', 'ON - OFF intra-ROI voxel FC'),
        ('between_roi_on_minus_off', 'between_roi_delta_z_on_minus_off', 'ON - OFF between-ROI FC'),
        ('within_minus_between_delta', 'within_minus_between_delta_z', '(ON - OFF intra-ROI FC) - (ON - OFF between-ROI FC)'),
    ]
    rows = []
    details = {}
    for (analysis, column, description) in tests:
        result = _single_subject_level_test(subject_deltas, column)
        details[analysis] = result
        row = {'analysis': analysis, 'value_column': column, 'description': description}
        row.update(result)
        rows.append(row)
    return (pd.DataFrame(rows), details)

def _plot_intra_between_fc(subject_deltas, results, out_dir):
    if subject_deltas.empty:
        raise RuntimeError('No complete OFF/ON subjects were available for intra-vs-between FC plotting')
    colors = {'off': '#4C78A8', 'on': '#D65F5F', 'delta': '#333333'}
    (fig, axes) = plt.subplots(1, 2, figsize=(8.0, 3.65))
    rng = np.random.default_rng(0)

    paired_specs = [
        ('Intra-ROI', 'within_roi_off_r', 'within_roi_on_r', 0.0),
        ('Between ROI', 'between_roi_off_r', 'between_roi_on_r', 1.1),
    ]
    for (_, off_column, on_column, offset) in paired_specs:
        off_values = subject_deltas[off_column].to_numpy(dtype=np.float64)
        on_values = subject_deltas[on_column].to_numpy(dtype=np.float64)
        for (off_value, on_value) in zip(off_values, on_values):
            axes[0].plot([offset, offset + 0.32], [off_value, on_value], color='#9a9a9a', linewidth=0.7, alpha=0.55, zorder=1)
        axes[0].scatter(rng.normal(offset, 0.018, off_values.size), off_values, s=20, color=colors['off'], edgecolor='white', linewidth=0.35, zorder=2, label='OFF' if offset == 0.0 else None)
        axes[0].scatter(rng.normal(offset + 0.32, 0.018, on_values.size), on_values, s=20, color=colors['on'], edgecolor='white', linewidth=0.35, zorder=2, label='ON' if offset == 0.0 else None)
    axes[0].set_xticks([0.16, 1.26])
    axes[0].set_xticklabels([item[0] for item in paired_specs])
    axes[0].set_ylabel('Mean FC (Pearson r)')
    axes[0].set_title('Medication State', fontsize=10)
    axes[0].legend(frameon=False, fontsize=8, loc='best')

    delta_specs = [
        ('Intra-ROI', 'within_roi_delta_z_on_minus_off', 0.0),
        ('Between ROI', 'between_roi_delta_z_on_minus_off', 1.0),
    ]
    for (_, row) in subject_deltas.iterrows():
        values = [row['within_roi_delta_z_on_minus_off'], row['between_roi_delta_z_on_minus_off']]
        axes[1].plot([0, 1], values, color='#9a9a9a', linewidth=0.7, alpha=0.55, zorder=1)
    for (name, column, x_value) in delta_specs:
        values = subject_deltas[column].to_numpy(dtype=np.float64)
        axes[1].scatter(rng.normal(x_value, 0.025, values.size), values, s=22, color=colors['delta'], edgecolor='white', linewidth=0.35, zorder=2)
        mean_value = float(np.nanmean(values))
        axes[1].plot([x_value - 0.16, x_value + 0.16], [mean_value, mean_value], color='#d62728', linewidth=1.4, zorder=3)
    axes[1].axhline(0, color='#666666', linewidth=0.8, linestyle='--')
    axes[1].set_xticks([item[2] for item in delta_specs])
    axes[1].set_xticklabels([item[0] for item in delta_specs])
    axes[1].set_ylabel('ON - OFF FC change (Fisher z)')
    p_value = results.get('within_minus_between_delta', {}).get('paired_t_p_value_two_sided', np.nan)
    axes[1].set_title(f'Change Difference p={_format_p_value(float(p_value))}', fontsize=10)

    for ax in axes:
        ax.grid(axis='y', color='#dddddd', linewidth=0.55, alpha=0.8)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=8.5)
    fig.suptitle('Intra-ROI voxel FC vs between-ROI FC', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / 'intra_vs_between_fc_medication_change.png'
    pdf_path = png_path.with_suffix('.pdf')
    fig.savefig(png_path, dpi=320, bbox_inches='tight', pad_inches=0.04)
    fig.savefig(pdf_path, bbox_inches='tight', pad_inches=0.04)
    plt.close(fig)
    return png_path

def _write_intra_between_method(path):
    text = (
        '# Intra-ROI vs Between-ROI FC Method\n\n'
        'For each subject/session, beta-series were extracted from the selected weighted voxels in each ROI. '
        'Intra-ROI FC was defined as the mean Pearson correlation between voxel beta-series within the same ROI. '
        'Time points with missing values in any selected voxel of that ROI were excluded before voxel correlations '
        'were computed. If fewer than four complete common time points remained for an ROI, voxel correlations were '
        'computed with pairwise-complete observations instead. Voxel-pair correlations were Fisher z transformed '
        'before averaging, and voxel-pair averages within an ROI were weighted by the product of the two optimization '
        'weights. To avoid bias from unequal ROI sizes, voxel pairs were not pooled across ROIs: each ROI contributed '
        'one intra-ROI FC value, and the session-level intra-ROI FC was the unweighted mean of these ROI-level '
        'Fisher-z values.\n\n'
        'Between-ROI FC was computed in the same Fisher-z Pearson-correlation scale using the weighted mean ROI '
        'beta-series. Correlations were averaged across the upper triangle of the ROI-by-ROI correlation matrix. '
        'Medication effects were evaluated within complete subjects as ON minus OFF separately for intra-ROI and '
        'between-ROI FC. The primary comparison was the paired subject-level contrast '
        '(ON - OFF intra-ROI FC) - (ON - OFF between-ROI FC).\n'
    )
    path.write_text(text, encoding='utf-8')

def _save_intra_between_fc_analysis(session_rows, roi_rows, out_dir):
    roi_values = pd.DataFrame(roi_rows)
    session_values = pd.DataFrame(session_rows).sort_values(['subject', 'session']).reset_index(drop=True)
    subject_deltas = _complete_intra_between_subject_deltas(session_values)
    if subject_deltas.empty:
        raise RuntimeError('No complete OFF/ON subjects were available for intra-vs-between FC analysis')
    (results_table, results) = _intra_between_fc_test_rows(subject_deltas)
    result_summary = {
        'connectivity_metric': INTRA_BETWEEN_FC_METRIC,
        'method': 'intra-ROI voxel-pair FC and between-ROI FC are both summarized as Fisher-z Pearson correlations; intra-ROI session means average ROI-level values equally so ROIs with more voxels do not dominate.',
        'n_sessions': int(session_values.shape[0]),
        'n_complete_subjects': int(subject_deltas.shape[0]),
        'tests': results,
    }
    roi_path = out_dir / 'intra_vs_between_fc_roi_values.csv'
    session_path = out_dir / 'intra_vs_between_fc_session_values.csv'
    delta_path = out_dir / 'intra_vs_between_fc_subject_deltas.csv'
    results_path = out_dir / 'intra_vs_between_fc_results.csv'
    json_path = out_dir / 'intra_vs_between_fc_results.json'
    method_path = out_dir / 'intra_vs_between_fc_method.md'
    roi_values.to_csv(roi_path, index=False)
    session_values.to_csv(session_path, index=False)
    subject_deltas.to_csv(delta_path, index=False)
    results_table.to_csv(results_path, index=False)
    json_path.write_text(json.dumps(result_summary, indent=2), encoding='utf-8')
    _write_intra_between_method(method_path)
    figure_path = _plot_intra_between_fc(subject_deltas, results, out_dir)
    return {
        'summary': result_summary,
        'roi_values': roi_path,
        'session_values': session_path,
        'subject_deltas': delta_path,
        'results': results_path,
        'results_json': json_path,
        'method': method_path,
        'figure': figure_path,
    }

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
    parser.add_argument('--node-strength-top-n', type=int, default=DEFAULT_NODE_STRENGTH_TOP_N)
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
    print('7. Compute paired OFF/ON subject-level, bootstrap, label-swap, and Mantel-style similarity tests.')
    print('8. Plot cross-subject-only OFF-OFF, ON-ON, and OFF-ON distance distributions.')
    print(f'9. Compute paired MI node-strength summaries and plot the top {args.node_strength_top_n} ROI panels.')
    print('10. Compute within-hemisphere vs between-hemisphere MI edge-change summaries.')
    print('11. Compute intra-ROI voxel FC vs between-ROI FC medication-change summaries.')
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
    intra_between_session_rows = []
    intra_between_roi_rows = []
    intra_between_fc_skipped = []
    for spec in specs:
        session_df = _load_session_timeseries(spec, weight_img, rois)
        session_df.to_csv(timeseries_dir / f'{spec.label}.csv', index=False)
        cleaned = _clean_timeseries(session_df)
        networks[spec.label] = _mutual_information_matrix(cleaned, n_neighbors=args.mi_neighbors, random_state=args.random_state)
        session_fc_row = {'label': spec.label, 'subject': spec.subject, 'session': spec.session, 'state': spec.state, 'connectivity_metric': INTRA_BETWEEN_FC_METRIC}
        session_fc_row.update(_between_roi_fc_summary(cleaned))
        try:
            voxel_timeseries = _load_session_voxel_timeseries(spec, weight_img, rois)
            (roi_fc_rows, intra_session_summary) = _intra_roi_fc_values(spec, voxel_timeseries, rois)
            session_fc_row.update(intra_session_summary)
            intra_between_roi_rows.extend(roi_fc_rows)
            intra_between_session_rows.append(session_fc_row)
        except RuntimeError as exc:
            intra_between_fc_skipped.append({'label': spec.label, 'reason': str(exc)})
    _save_networks(networks, roi_names, out_dir)
    pairwise = _pairwise_network_distances(specs, networks)
    pairwise_path = out_dir / 'pairwise_metric_values.csv'
    pairwise.to_csv(pairwise_path, index=False)
    (paired_stats, paired_subject_path, paired_stats_path) = _save_paired_similarity_tests(pairwise, out_dir)
    figure_path = _plot_cross_subject_distribution(pairwise, out_dir, paired_stats=paired_stats)
    (node_strength_summary, node_strength_values_path, node_strength_summary_path, node_strength_figure_path) = _save_node_strength_analysis(specs, networks, roi_names, out_dir, paired_stats=paired_stats, top_n=args.node_strength_top_n)
    hemisphere_fc_paths = None
    hemisphere_fc_skipped = None
    try:
        hemisphere_fc_paths = _save_hemisphere_fc_analysis(specs, networks, roi_names, out_dir)
    except RuntimeError as exc:
        hemisphere_fc_skipped = str(exc)
        warnings.warn(f'Skipped within-vs-between hemisphere FC analysis: {exc}', RuntimeWarning)
    intra_between_paths = None
    if intra_between_session_rows:
        intra_between_paths = _save_intra_between_fc_analysis(intra_between_session_rows, intra_between_roi_rows, out_dir)
    elif intra_between_fc_skipped:
        warnings.warn('Skipped intra-vs-between FC analysis because no sessions had voxel-level beta or BOLD inputs.', RuntimeWarning)
    metadata.update({'weight_map': str(args.weight_map), 'roi_definition_figure': str(args.roi_definition_figure), 'roi_region_table': str(args.roi_region_table), 'session_manifest': str(args.session_manifest) if args.session_manifest.exists() else None, 'beta_root': str(args.beta_root), 'roi_percentile': float(args.roi_percentile), 'weight_threshold': roi_threshold, 'min_report_voxels': int(args.min_report_voxels), 'min_roi_voxels': int(min_roi_voxels), 'connectivity_metric': CONNECTIVITY_METRIC, 'comparison_metric': COMPARISON_METRIC, 'mi_neighbors': int(args.mi_neighbors), 'paired_subject_similarity_values': str(paired_subject_path), 'paired_subject_similarity_stats': str(paired_stats_path), 'paired_subject_similarity_primary_p': paired_stats['permutation']['permutation_p_value_two_sided'], 'paired_subject_similarity_primary_effect': paired_stats['effect_size']['raw_difference'], 'node_strength_mi_values': str(node_strength_values_path), 'node_strength_mi_results': str(node_strength_summary_path), 'node_strength_mi_figure': str(node_strength_figure_path), 'node_strength_mi_top_n': int(args.node_strength_top_n), 'node_strength_mi_min_p': float(node_strength_summary['p_value'].min()), 'intra_vs_between_fc_metric': INTRA_BETWEEN_FC_METRIC, 'intra_vs_between_fc_skipped': intra_between_fc_skipped, 'intra_vs_between_fc_outputs': {key: str(value) for (key, value) in intra_between_paths.items() if key != 'summary'} if intra_between_paths else None, 'intra_vs_between_fc_primary_p': intra_between_paths['summary']['tests']['within_minus_between_delta']['paired_t_p_value_two_sided'] if intra_between_paths else None, 'intra_vs_between_fc_primary_effect': intra_between_paths['summary']['tests']['within_minus_between_delta']['mean'] if intra_between_paths else None, 'sessions': [{'label': spec.label, 'subject': spec.subject, 'session': spec.session, 'state': spec.state, 'bold_path': str(spec.bold_path) if spec.bold_path else None, 'timeseries_path': str(spec.timeseries_path) if spec.timeseries_path else None, 'beta_paths': [str(path) for path in spec.beta_paths]} for spec in specs]})
    metadata.update({
        'within_vs_between_hemisphere_fc_skipped': hemisphere_fc_skipped,
        'within_vs_between_hemisphere_fc_outputs': {key: str(value) for (key, value) in hemisphere_fc_paths.items() if key != 'summary'} if hemisphere_fc_paths else None,
        'within_vs_between_hemisphere_fc_primary_p': hemisphere_fc_paths['summary']['tests']['within_minus_between_hemisphere_delta']['paired_t_p_value_two_sided'] if hemisphere_fc_paths else None,
        'within_vs_between_hemisphere_fc_primary_effect': hemisphere_fc_paths['summary']['tests']['within_minus_between_hemisphere_delta']['mean'] if hemisphere_fc_paths else None,
    })
    (out_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(f'Saved {pairwise_path}')
    print(f'Saved {figure_path}')
    print(f"Saved {figure_path.with_suffix('.pdf')}")
    print(f'Saved {node_strength_figure_path}')
    print(f"Saved {node_strength_figure_path.with_suffix('.pdf')}")
    print(f'Saved {node_strength_values_path}')
    print(f'Saved {node_strength_summary_path}')
    if hemisphere_fc_paths:
        print(f"Saved {hemisphere_fc_paths['figure']}")
        print(f"Saved {hemisphere_fc_paths['figure'].with_suffix('.pdf')}")
        print(f"Saved {hemisphere_fc_paths['edge_values']}")
        print(f"Saved {hemisphere_fc_paths['session_values']}")
        print(f"Saved {hemisphere_fc_paths['subject_deltas']}")
        print(f"Saved {hemisphere_fc_paths['results']}")
        print(f"Saved {hemisphere_fc_paths['results_json']}")
        print(f"Saved {hemisphere_fc_paths['method']}")
    if intra_between_paths:
        print(f"Saved {intra_between_paths['figure']}")
        print(f"Saved {intra_between_paths['figure'].with_suffix('.pdf')}")
        print(f"Saved {intra_between_paths['roi_values']}")
        print(f"Saved {intra_between_paths['session_values']}")
        print(f"Saved {intra_between_paths['subject_deltas']}")
        print(f"Saved {intra_between_paths['results']}")
        print(f"Saved {intra_between_paths['results_json']}")
        print(f"Saved {intra_between_paths['method']}")
    print(f"Saved {out_dir / 'weighted_roi_definition.csv'}")
    print(f'Saved {paired_subject_path}')
    print(f'Saved {paired_stats_path}')
    print(f"Saved {out_dir / 'metadata.json'}")
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
