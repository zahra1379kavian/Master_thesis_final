#!/usr/bin/env python3
import argparse
import json
import warnings
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

import med_effects as M


DEFAULT_TASK_ACTIVATION_MAP = Path('data/z_valu_standard_glm.nii.gz')
DEFAULT_TASK_Z_THRESHOLD = 3.1
DEFAULT_OUT_DIR = Path('figures/med_effects_task_activation')
DEFAULT_MIN_LATERALIZED_VOXELS = 30
TASK_VOXEL_SELECTION = 'task-activation'


def _lateralized_source_roi_names(reference_img, aal_version, cache_dir):
    groups, _ = M._build_roi_groups(reference_img, aal_version, cache_dir)
    names = []
    for group in groups:
        has_left = any(str(label).endswith('_L') for label in group.matched_labels)
        has_right = any(str(label).endswith('_R') for label in group.matched_labels)
        if has_left and has_right:
            names.append(group.name)
    return names


def _task_activation_roi_setup(args, reference_img):
    excluded_rois = {str(name).strip() for name in args.exclude_rois if str(name).strip()}
    if args.split_hemispheres:
        source_roi_names = _lateralized_source_roi_names(reference_img, args.aal_version, args.atlas_cache_dir)
        source_roi_names = [name for name in source_roi_names if name not in excluded_rois]
        groups, metadata = M._build_lateralized_roi_groups(reference_img, args.aal_version, args.atlas_cache_dir, source_roi_names)
        roi_names = [group.name for group in groups]
        min_roi_voxels = int(args.min_lateralized_voxels)
    else:
        groups, metadata = M._build_roi_groups(reference_img, args.aal_version, args.atlas_cache_dir)
        groups = [group for group in groups if group.name not in excluded_rois]
        roi_names = [group.name for group in groups]
        source_roi_names = roi_names
        min_roi_voxels = int(args.min_report_voxels)
    metadata.update({
        'split_hemispheres': bool(args.split_hemispheres),
        'source_roi_names': source_roi_names,
        'excluded_rois': sorted(excluded_rois),
        'min_roi_voxels': int(min_roi_voxels),
    })
    return groups, metadata, roi_names, min_roi_voxels


def _build_task_activation_rois(task_values, roi_names, groups, z_threshold, min_roi_voxels):
    selected = np.isfinite(task_values) & (task_values >= float(z_threshold))
    if not np.any(selected):
        raise RuntimeError(f'No finite task-activation voxels found at z >= {float(z_threshold):g}')
    group_lookup = {group.name: group for group in groups}
    missing_groups = [name for name in roi_names if name not in group_lookup]
    if missing_groups:
        raise RuntimeError('ROI names were not found in the AAL grouping: ' + ', '.join(missing_groups))
    rois = []
    for name in roi_names:
        mask = selected & group_lookup[name].mask
        n_voxels = int(np.count_nonzero(mask))
        if n_voxels < int(min_roi_voxels):
            continue
        rois.append(M.WeightedROI(
            name=name,
            mask=mask,
            weights=np.ones(n_voxels, dtype=np.float64),
            n_voxels=n_voxels,
        ))
    rois = sorted(rois, key=lambda roi: (-roi.n_voxels, roi.name))
    if len(rois) < 2:
        raise RuntimeError('At least two task-activation ROI masks are required for edge-network analysis')
    return rois


def _missing_inputs(args):
    missing = []
    if not args.task_activation_map.exists():
        missing.append(f'{args.task_activation_map} (standard-GLM z map)')
    try:
        specs = M._load_session_specs(args)
    except RuntimeError as exc:
        missing.append(str(exc))
        return missing
    for spec in specs:
        for beta_path in spec.beta_paths:
            if not beta_path.exists():
                missing.append(f'{beta_path} (4D cleaned beta volume for {spec.label})')
        if spec.bold_path is not None and not spec.bold_path.exists():
            missing.append(f'{spec.bold_path} (4D BOLD NIfTI for {spec.label})')
        if spec.timeseries_path is not None and not spec.timeseries_path.exists():
            missing.append(f'{spec.timeseries_path} (ROI time-series CSV for {spec.label})')
    return missing


def _roi_summary(rois, z_threshold):
    summary = pd.DataFrame({
        'roi_name': [roi.name for roi in rois],
        'n_voxels': [roi.n_voxels for roi in rois],
        'task_activation_voxels': [roi.n_voxels for roi in rois],
        'voxel_selection': TASK_VOXEL_SELECTION,
        'voxel_weighting': 'unit_weights',
        'task_z_threshold': float(z_threshold),
    })
    summary['n_weighted_voxels'] = summary['n_voxels']
    return summary


def _write_task_intra_between_method(path, z_threshold):
    text = (
        '# Intra-ROI vs Between-ROI FC Method\n\n'
        'For each subject/session, beta-series were extracted from task-activation voxels '
        f'defined by the standard-GLM z map thresholded at z >= {float(z_threshold):g}. Because the task map is '
        'binary after thresholding and has no optimization weights, ROI beta-series are '
        'unweighted means of selected voxel beta values.\n\n'
        'Intra-ROI FC was defined as the mean Pearson correlation between voxel beta-series '
        'within the same ROI. Voxel-pair correlations were Fisher z transformed before '
        'averaging, with equal weight for each voxel pair. Between-ROI FC was computed from '
        'the unweighted mean ROI beta-series in the same Fisher-z Pearson-correlation scale. '
        'Medication effects were evaluated within complete subjects as ON minus OFF, and '
        'the primary comparison was (ON - OFF intra-ROI FC) - (ON - OFF between-ROI FC).\n'
    )
    path.write_text(text, encoding='utf-8')


def _prepare_task_rois(args):
    task_img = nib.load(str(args.task_activation_map))
    task_values = np.asarray(task_img.get_fdata(), dtype=np.float64)
    groups, metadata, roi_names, min_roi_voxels = _task_activation_roi_setup(args, task_img)
    rois = _build_task_activation_rois(
        task_values=task_values,
        roi_names=roi_names,
        groups=groups,
        z_threshold=args.task_z_threshold,
        min_roi_voxels=min_roi_voxels,
    )
    return task_img, task_values, metadata, rois, min_roi_voxels


def _print_dry_run(args):
    task_img, task_values, _, rois, min_roi_voxels = _prepare_task_rois(args)
    specs = M._load_session_specs(args)
    session_df = M._session_summary(specs)
    beta_shape_counts = M._beta_shape_counts(specs)
    selected = np.isfinite(task_values) & (task_values >= float(args.task_z_threshold))
    print('Dry run only; no connectivity matrices, pairwise distances, or figures were computed.')
    print()
    print('Planned steps:')
    print(f'1. Load task-activation map: {args.task_activation_map}')
    print(f'2. Select voxels with z >= {float(args.task_z_threshold):g}.')
    print('3. Assign selected voxels to AAL ROI masks and use unit voxel weights.')
    if args.split_hemispheres:
        print('   Split selected AAL groups into left/right hemisphere ROI masks.')
    print(f'4. Extract unweighted ROI mean beta trial series from {len(specs)} subject/session inputs.')
    print('5. Run the existing medication network, cross-subject distance, and intra-vs-between FC analyses.')
    print()
    print(f'Task-map grid: {task_img.shape[:3]}')
    print(f'Task activation voxels at threshold: {int(np.count_nonzero(selected))}')
    print(f'ROI count: {len(rois)}')
    print(f'Minimum selected voxels per ROI: {min_roi_voxels}')
    print('Top ROI masks: ' + ', '.join((f'{roi.name} ({roi.n_voxels})' for roi in rois[:10])))
    print()
    print('Input sessions by state:')
    print(session_df.groupby('state')['label'].count().to_string())
    if beta_shape_counts:
        print()
        print('Beta volume shapes detected:')
        for shape, count in beta_shape_counts.items():
            print(f'- {shape}: {count} files')


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task-activation-map', type=Path, default=DEFAULT_TASK_ACTIVATION_MAP)
    parser.add_argument('--task-z-threshold', type=float, default=DEFAULT_TASK_Z_THRESHOLD)
    parser.add_argument('--session-manifest', type=Path, default=M.DEFAULT_SESSION_MANIFEST)
    parser.add_argument('--beta-root', type=Path, default=M.DEFAULT_BETA_ROOT)
    parser.add_argument('--subjects', nargs='+', default=None)
    parser.add_argument('--complete-subjects-only', action='store_true')
    parser.add_argument('--out-dir', type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument('--split-hemispheres', dest='split_hemispheres', action='store_true', default=True)
    parser.add_argument('--no-split-hemispheres', dest='split_hemispheres', action='store_false')
    parser.add_argument('--exclude-rois', nargs='*', default=())
    parser.add_argument('--min-report-voxels', type=int, default=M.DEFAULT_MIN_REPORT_VOXELS)
    parser.add_argument('--min-lateralized-voxels', type=int, default=DEFAULT_MIN_LATERALIZED_VOXELS)
    parser.add_argument('--aal-version', default=M.DEFAULT_AAL_VERSION)
    parser.add_argument('--atlas-cache-dir', type=Path, default=M.DEFAULT_ATLAS_CACHE_DIR)
    parser.add_argument('--connectivity-metric', choices=M.CONNECTIVITY_METRICS, default=M.CONNECTIVITY_METRIC)
    parser.add_argument('--mi-neighbors', type=int, default=M.DEFAULT_MI_NEIGHBORS)
    parser.add_argument('--node-strength-top-n', type=int, default=M.DEFAULT_NODE_STRENGTH_TOP_N)
    parser.add_argument('--random-state', type=int, default=0)
    parser.add_argument('--check-inputs', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    return parser


def main():
    args = build_parser().parse_args()
    M.CONNECTIVITY_METRIC = args.connectivity_metric
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

    task_img, task_values, metadata, rois, min_roi_voxels = _prepare_task_rois(args)
    selected_task = np.isfinite(task_values) & (task_values >= float(args.task_z_threshold))
    specs = M._load_session_specs(args)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    roi_names = [roi.name for roi in rois]
    roi_definition_path = out_dir / 'task_activation_roi_definition.csv'
    _roi_summary(rois, args.task_z_threshold).to_csv(roi_definition_path, index=False)

    timeseries_dir = out_dir / 'roi_timeseries'
    timeseries_dir.mkdir(parents=True, exist_ok=True)
    networks = {}
    intra_between_session_rows = []
    intra_between_roi_rows = []
    intra_between_fc_skipped = []
    for spec in specs:
        session_df = M._load_session_timeseries(spec, task_img, rois)
        session_df.to_csv(timeseries_dir / f'{spec.label}.csv', index=False)
        cleaned = M._clean_timeseries(session_df)
        networks[spec.label] = M._connectivity_matrix(cleaned, n_neighbors=args.mi_neighbors, random_state=args.random_state)
        session_fc_row = {
            'label': spec.label,
            'subject': spec.subject,
            'session': spec.session,
            'state': spec.state,
            'connectivity_metric': M.INTRA_BETWEEN_FC_METRIC,
        }
        session_fc_row.update(M._between_roi_fc_summary(cleaned))
        try:
            voxel_timeseries = M._load_session_voxel_timeseries(spec, task_img, rois)
            roi_fc_rows, intra_session_summary = M._intra_roi_fc_values(spec, voxel_timeseries, rois)
            session_fc_row.update(intra_session_summary)
            intra_between_roi_rows.extend(roi_fc_rows)
            intra_between_session_rows.append(session_fc_row)
        except RuntimeError as exc:
            intra_between_fc_skipped.append({'label': spec.label, 'reason': str(exc)})

    M._save_networks(networks, roi_names, out_dir)
    pairwise = M._pairwise_network_distances(specs, networks)
    pairwise_path = out_dir / 'pairwise_metric_values.csv'
    pairwise.to_csv(pairwise_path, index=False)
    paired_stats, paired_subject_path, paired_stats_path = M._save_paired_similarity_tests(pairwise, out_dir)
    figure_path = M._plot_cross_subject_distribution(pairwise, out_dir, paired_stats=paired_stats)
    node_strength_summary, node_strength_values_path, node_strength_summary_path, node_strength_figure_path = M._save_node_strength_analysis(
        specs,
        networks,
        roi_names,
        out_dir,
        paired_stats=paired_stats,
        top_n=args.node_strength_top_n,
    )

    hemisphere_fc_paths = None
    hemisphere_fc_skipped = None
    try:
        hemisphere_fc_paths = M._save_hemisphere_fc_analysis(specs, networks, roi_names, out_dir)
    except RuntimeError as exc:
        hemisphere_fc_skipped = str(exc)
        M._remove_hemisphere_fc_outputs(out_dir)
        warnings.warn(f'Skipped within-vs-between hemisphere FC analysis: {exc}', RuntimeWarning)

    intra_between_paths = None
    if intra_between_session_rows:
        intra_between_paths = M._save_intra_between_fc_analysis(
            intra_between_session_rows,
            intra_between_roi_rows,
            out_dir,
            voxel_selection=TASK_VOXEL_SELECTION,
        )
        _write_task_intra_between_method(intra_between_paths['method'], args.task_z_threshold)
    elif intra_between_fc_skipped:
        warnings.warn('Skipped intra-vs-between FC analysis because no sessions had voxel-level beta or BOLD inputs.', RuntimeWarning)

    metadata.update({
        'task_activation_map': str(args.task_activation_map),
        'task_activation_threshold': f'z >= {float(args.task_z_threshold):g}',
        'task_activation_voxels': int(np.count_nonzero(selected_task)),
        'task_activation_roi_definition': str(roi_definition_path),
        'session_manifest': str(args.session_manifest) if args.session_manifest.exists() else None,
        'beta_root': str(args.beta_root),
        'voxel_selection': TASK_VOXEL_SELECTION,
        'voxel_weighting': 'unit_weights',
        'min_report_voxels': int(args.min_report_voxels),
        'min_roi_voxels': int(min_roi_voxels),
        'connectivity_metric': M.CONNECTIVITY_METRIC,
        'comparison_metric': M.COMPARISON_METRIC,
        'mi_neighbors': int(args.mi_neighbors),
        'paired_subject_similarity_values': str(paired_subject_path),
        'paired_subject_similarity_stats': str(paired_stats_path),
        'paired_subject_similarity_primary_p': paired_stats['permutation']['permutation_p_value_two_sided'],
        'paired_subject_similarity_primary_effect': paired_stats['effect_size']['raw_difference'],
        'node_strength_mi_values': str(node_strength_values_path),
        'node_strength_mi_results': str(node_strength_summary_path),
        'node_strength_mi_figure': str(node_strength_figure_path),
        'node_strength_mi_top_n': int(args.node_strength_top_n),
        'node_strength_mi_min_p': float(node_strength_summary['p_value'].min()),
        'intra_vs_between_fc_metric': M.INTRA_BETWEEN_FC_METRIC,
        'intra_vs_between_fc_skipped': intra_between_fc_skipped,
        'intra_vs_between_fc_outputs': {key: str(value) for key, value in intra_between_paths.items() if key != 'summary'} if intra_between_paths else None,
        'intra_vs_between_fc_primary_p': intra_between_paths['summary']['tests']['within_minus_between_delta']['paired_t_p_value_two_sided'] if intra_between_paths else None,
        'intra_vs_between_fc_primary_effect': intra_between_paths['summary']['tests']['within_minus_between_delta']['mean'] if intra_between_paths else None,
        'within_vs_between_hemisphere_fc_skipped': hemisphere_fc_skipped,
        'within_vs_between_hemisphere_fc_outputs': {key: str(value) for key, value in hemisphere_fc_paths.items() if key != 'summary'} if hemisphere_fc_paths else None,
        'within_vs_between_hemisphere_fc_primary_p': hemisphere_fc_paths['summary']['tests']['within_minus_between_hemisphere_delta']['paired_t_p_value_two_sided'] if hemisphere_fc_paths else None,
        'within_vs_between_hemisphere_fc_primary_effect': hemisphere_fc_paths['summary']['tests']['within_minus_between_hemisphere_delta']['mean'] if hemisphere_fc_paths else None,
        'sessions': [{
            'label': spec.label,
            'subject': spec.subject,
            'session': spec.session,
            'state': spec.state,
            'bold_path': str(spec.bold_path) if spec.bold_path else None,
            'timeseries_path': str(spec.timeseries_path) if spec.timeseries_path else None,
            'beta_paths': [str(path) for path in spec.beta_paths],
        } for spec in specs],
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
    print(f'Saved {roi_definition_path}')
    print(f'Saved {paired_subject_path}')
    print(f'Saved {paired_stats_path}')
    print(f"Saved {out_dir / 'metadata.json'}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
