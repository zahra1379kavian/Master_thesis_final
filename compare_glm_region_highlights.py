#!/usr/bin/env python3
from collections import namedtuple
import argparse
import itertools
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage
from threshold_robustness_voxel_network import BILATERAL_HEMISPHERE_LABEL, DEFAULT_AAL_VERSION, UNASSIGNED_ROI, _build_roi_groups, _display_region_name
DEFAULT_STANDARD_GLM = Path('data/z_valu_standard_glm.nii.gz')
DEFAULT_GLM_SINGLE_A = Path('data/z_value_typaA.nii.gz')
DEFAULT_GLM_SINGLE_D = Path('data/z_value_typeD.nii.gz')
DEFAULT_WEIGHT_MAP = Path('data/voxel_weights_task1_bold0.6_beta0.6_smooth1.25_gamma1.5.nii.gz')
DEFAULT_OUT_BASE = Path('figures/glm_glmsingle_optimization_region_comparison')
DEFAULT_STANDARD_Z_THRESHOLD = 3.1
DEFAULT_GLM_SINGLE_Z_THRESHOLD = 1.96
DEFAULT_WEIGHT_PERCENTILES = (80.0, 90.0)
DEFAULT_MIN_REPORT_VOXELS = 25
DEFAULT_HEATMAP_MIN_ROW_PERCENT = 2.0
DEFAULT_ATLAS_CACHE_DIR = Path('/home/zkavian/nilearn_data')
MapSpec = namedtuple('MapSpec', ('method', 'path', 'kind'))
MaskSpec = namedtuple('MaskSpec', ('method', 'mask_name', 'family', 'threshold_definition', 'threshold_value', 'values', 'mask'))

def _load_img(path):
    if not path.exists():
        raise RuntimeError(f'Missing input map: {path}')
    return nib.load(str(path))

def _check_same_grid(reference, images):
    for (name, img) in images.items():
        if img.shape[:3] != reference.shape[:3]:
            raise RuntimeError(f'{name} shape {img.shape[:3]} differs from reference {reference.shape[:3]}')
        if not np.allclose(img.affine, reference.affine):
            raise RuntimeError(f'{name} affine differs from the reference image')

def _build_analysis_mask(map_data):
    support_names = ('GLMsingle Type A', 'GLMsingle Type D', 'Optimization weights')
    mask = np.zeros(next(iter(map_data.values())).shape, dtype=bool)
    for name in support_names:
        values = map_data[name]
        mask |= np.isfinite(values) & (values != 0)
    return mask

def _top_weight_mask(weights, analysis_mask, n_voxels):
    available = analysis_mask & np.isfinite(weights) & (weights != 0)
    available_values = weights[available]
    if n_voxels <= 0 or available_values.size == 0:
        return (np.zeros(weights.shape, dtype=bool), np.nan)
    n_voxels = min(int(n_voxels), int(available_values.size))
    threshold = float(np.partition(available_values, -n_voxels)[-n_voxels])
    mask = available & (weights >= threshold)
    if int(np.count_nonzero(mask)) > n_voxels:
        selected = np.column_stack(np.nonzero(available))
        selected_values = weights[available]
        order = np.argsort(selected_values, kind='stable')[-n_voxels:]
        exact = np.zeros(weights.shape, dtype=bool)
        exact[tuple(selected[order].T)] = True
        mask = exact
    return (mask, threshold)

def _make_masks(map_data, analysis_mask, z_thresholds, weight_percentiles):
    masks = []
    z_methods = ('Standard GLM', 'GLMsingle Type A', 'GLMsingle Type D')
    for method in z_methods:
        values = map_data[method]
        finite = analysis_mask & np.isfinite(values)
        z_threshold = float(z_thresholds[method])
        masks.extend([MaskSpec(method=method, mask_name=f'{method} positive z', family='positive_z', threshold_definition=f'z >= {z_threshold:g}', threshold_value=float(z_threshold), values=values, mask=finite & (values >= z_threshold)), MaskSpec(method=method, mask_name=f'{method} negative z', family='negative_z', threshold_definition=f'z <= -{z_threshold:g}', threshold_value=float(-z_threshold), values=values, mask=finite & (values <= -z_threshold)), MaskSpec(method=method, mask_name=f'{method} absolute z', family='absolute_z', threshold_definition=f'|z| >= {z_threshold:g}', threshold_value=float(z_threshold), values=values, mask=finite & (np.abs(values) >= z_threshold))])
    weights = map_data['Optimization weights']
    finite_weights = analysis_mask & np.isfinite(weights) & (weights != 0)
    weight_values = weights[finite_weights]
    for weight_percentile in weight_percentiles:
        weight_threshold = float(np.percentile(weight_values, weight_percentile))
        masks.append(MaskSpec(method='Optimization weights', mask_name=f'Optimization p{weight_percentile:g}', family='weight_percentile', threshold_definition=f'top {100.0 - weight_percentile:g}% of nonzero weights', threshold_value=weight_threshold, values=weights, mask=finite_weights & (weights >= weight_threshold)))
    for source in [mask for mask in masks if mask.family in {'positive_z', 'absolute_z'}]:
        n_voxels = int(np.count_nonzero(source.mask))
        (matched_mask, matched_threshold) = _top_weight_mask(weights, analysis_mask, n_voxels)
        masks.append(MaskSpec(method='Optimization weights', mask_name=f"Optimization matched to {source.method} {source.family.replace('_', ' ')}", family=f'weight_matched_{source.family}', threshold_definition=f"top {n_voxels:,} weights; voxel-count matched to {source.method} {source.family.replace('_', ' ')}", threshold_value=matched_threshold, values=weights, mask=matched_mask))
    return masks

def _region_denominators(groups, analysis_mask):
    return {group.name: int(np.count_nonzero(group.mask & analysis_mask)) for group in groups}

def _assign_regions(mask_spec, affine, groups, region_sizes, min_report_voxels):
    selected_ijk = np.column_stack(np.nonzero(mask_spec.mask)).astype(np.int32, copy=False)
    if selected_ijk.size == 0:
        return pd.DataFrame([{'method': mask_spec.method, 'mask_name': mask_spec.mask_name, 'family': mask_spec.family, 'threshold_definition': mask_spec.threshold_definition, 'threshold_value': mask_spec.threshold_value, 'roi_name': UNASSIGNED_ROI, 'hemisphere': 'NA', 'n_voxels': 0, 'percent_of_map': np.nan, 'percent_of_assigned_map': np.nan, 'atlas_region_voxels': np.nan, 'percent_of_atlas_region': np.nan, 'present_for_report': False, 'mean_map_value': np.nan, 'min_map_value': np.nan, 'max_map_value': np.nan, 'mean_abs_value': np.nan, 'peak_abs_value': np.nan, 'x_mm': np.nan, 'y_mm': np.nan, 'z_mm': np.nan, 'source': 'No selected voxels'}])
    (x, y, z) = selected_ijk.T
    assigned = np.zeros(selected_ijk.shape[0], dtype=np.int16)
    group_names = [UNASSIGNED_ROI] + [group.name for group in groups]
    group_sources = {UNASSIGNED_ROI: 'Outside atlas labels'}
    for (group_id, group) in enumerate(groups, start=1):
        hit = group.mask[x, y, z] & (assigned == 0)
        assigned[hit] = group_id
        group_sources[group.name] = group.source
    values = mask_spec.values[x, y, z]
    coords_mm = nib.affines.apply_affine(affine, selected_ijk)
    total_voxels = int(selected_ijk.shape[0])
    assigned_voxels = int(np.count_nonzero(assigned != 0))
    rows = []
    for group_id in np.unique(assigned):
        positions = np.flatnonzero(assigned == group_id)
        if positions.size == 0:
            continue
        roi_name = group_names[int(group_id)]
        roi_values = values[positions]
        coords = coords_mm[positions]
        atlas_region_voxels = region_sizes.get(roi_name, np.nan)
        percent_of_atlas_region = float(positions.size / atlas_region_voxels) if roi_name != UNASSIGNED_ROI and atlas_region_voxels else np.nan
        rows.append({'method': mask_spec.method, 'mask_name': mask_spec.mask_name, 'family': mask_spec.family, 'threshold_definition': mask_spec.threshold_definition, 'threshold_value': float(mask_spec.threshold_value), 'roi_name': roi_name, 'hemisphere': 'NA' if roi_name == UNASSIGNED_ROI else BILATERAL_HEMISPHERE_LABEL, 'n_voxels': int(positions.size), 'percent_of_map': float(positions.size / total_voxels) if total_voxels else np.nan, 'percent_of_assigned_map': float(positions.size / assigned_voxels) if assigned_voxels and roi_name != UNASSIGNED_ROI else np.nan, 'atlas_region_voxels': atlas_region_voxels, 'percent_of_atlas_region': percent_of_atlas_region, 'present_for_report': bool(positions.size >= min_report_voxels and roi_name != UNASSIGNED_ROI), 'mean_map_value': float(np.mean(roi_values)), 'min_map_value': float(np.min(roi_values)), 'max_map_value': float(np.max(roi_values)), 'mean_abs_value': float(np.mean(np.abs(roi_values))), 'peak_abs_value': float(np.max(np.abs(roi_values))), 'x_mm': float(np.mean(coords[:, 0])), 'y_mm': float(np.mean(coords[:, 1])), 'z_mm': float(np.mean(coords[:, 2])), 'source': group_sources[roi_name]})
    return pd.DataFrame(rows)

def _component_stats(mask):
    (labels, n_components) = ndimage.label(mask)
    component_sizes = np.bincount(labels.ravel())
    largest_component = int(component_sizes[1:].max()) if component_sizes.size > 1 else 0
    return (int(n_components), largest_component)

def _top_region_text(region_df, mask_name, max_regions=8):
    rows = region_df[region_df['mask_name'].eq(mask_name) & region_df['present_for_report'] & ~region_df['roi_name'].eq(UNASSIGNED_ROI)].sort_values('n_voxels', ascending=False).head(max_regions)
    return '; '.join((f'{row.roi_name} ({int(row.n_voxels)})' for row in rows.itertuples(index=False)))

def _summarize_masks(masks, region_df):
    rows = []
    for spec in masks:
        n_voxels = int(np.count_nonzero(spec.mask))
        unassigned = int(region_df.loc[region_df['mask_name'].eq(spec.mask_name) & region_df['roi_name'].eq(UNASSIGNED_ROI), 'n_voxels'].sum())
        (n_components, largest_component) = _component_stats(spec.mask)
        reportable_regions = int(region_df.loc[region_df['mask_name'].eq(spec.mask_name) & region_df['present_for_report'], 'roi_name'].nunique())
        rows.append({'method': spec.method, 'mask_name': spec.mask_name, 'family': spec.family, 'threshold_definition': spec.threshold_definition, 'threshold_value': spec.threshold_value, 'n_voxels': n_voxels, 'n_unassigned_voxels': unassigned, 'atlas_assigned_fraction': float(1.0 - unassigned / n_voxels) if n_voxels else np.nan, 'n_components': n_components, 'largest_component_voxels': largest_component, 'n_reportable_regions': reportable_regions, 'top_regions_by_voxels': _top_region_text(region_df, spec.mask_name)})
    return pd.DataFrame(rows)

def _intersection_top_regions(intersection_mask, groups, min_report_voxels, max_regions=6):
    rows = []
    for group in groups:
        n_voxels = int(np.count_nonzero(intersection_mask & group.mask))
        if n_voxels >= min_report_voxels:
            rows.append((group.name, n_voxels))
    rows.sort(key=lambda item: item[1], reverse=True)
    return '; '.join((f'{name} ({n_voxels})' for (name, n_voxels) in rows[:max_regions]))

def _overlap_table(masks, groups, min_report_voxels):
    rows = []
    for (left, right) in itertools.combinations(masks, 2):
        left_mask = left.mask
        right_mask = right.mask
        left_n = int(np.count_nonzero(left_mask))
        right_n = int(np.count_nonzero(right_mask))
        intersection_mask = left_mask & right_mask
        intersection = int(np.count_nonzero(intersection_mask))
        union = int(np.count_nonzero(left_mask | right_mask))
        rows.append({'mask_a': left.mask_name, 'family_a': left.family, 'mask_b': right.mask_name, 'family_b': right.family, 'n_voxels_a': left_n, 'n_voxels_b': right_n, 'shared_voxels': intersection, 'dice': float(2 * intersection / (left_n + right_n)) if left_n + right_n else np.nan, 'jaccard': float(intersection / union) if union else np.nan, 'overlap_coefficient': float(intersection / min(left_n, right_n)) if min(left_n, right_n) else np.nan, 'shared_top_regions': _intersection_top_regions(intersection_mask, groups, min_report_voxels)})
    return pd.DataFrame(rows)

def _wide_region_table(region_df, mask_names, value_col='percent_of_map'):
    rows = region_df[region_df['mask_name'].isin(mask_names) & ~region_df['roi_name'].eq(UNASSIGNED_ROI)].copy()
    wide = rows.pivot_table(index='roi_name', columns='mask_name', values=value_col, aggfunc='sum', fill_value=0.0)
    return wide.reindex(columns=[name for name in mask_names if name in wide.columns])

def _plot_region_heatmap(region_df, mask_names, out_base, min_row_percent=DEFAULT_HEATMAP_MIN_ROW_PERCENT):
    wide = _wide_region_table(region_df, mask_names, value_col='percent_of_map')
    if wide.empty:
        return
    wide = wide.loc[wide.max(axis=1).sort_values(ascending=False).index]
    min_row_fraction = max(float(min_row_percent), 0.0) / 100.0
    if min_row_fraction > 0:
        filtered = wide[wide.max(axis=1) >= min_row_fraction]
        if not filtered.empty:
            wide = filtered
    fig_height = max(5.8, 0.28 * len(wide) + 2.0)
    (fig, ax) = plt.subplots(figsize=(11.8, fig_height), facecolor='white')
    values = wide.to_numpy(dtype=float) * 100.0
    im = ax.imshow(values, aspect='auto', cmap='Blues', vmin=0.0)
    ax.set_xticks(np.arange(len(wide.columns)))
    ax.set_xticklabels([_short_mask_label(name) for name in wide.columns], rotation=30, ha='right')
    ax.set_yticks(np.arange(len(wide.index)))
    ax.set_yticklabels([_display_region_name(name) for name in wide.index], fontsize=8)
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            if value >= 2.0:
                (r, g, b, _) = im.cmap(im.norm(value))
                luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
                ax.text(col, row, f'{value:.1f}', ha='center', va='center', fontsize=6.5, color='black' if luminance > 0.55 else 'white')
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(f'{out_base}_region_heatmap.png', dpi=220, bbox_inches='tight', pad_inches=0.04)
    fig.savefig(f'{out_base}_region_heatmap.pdf', bbox_inches='tight', pad_inches=0.04)
    plt.close(fig)

def _short_mask_label(name):
    replacements = {'Standard GLM positive z': 'Standard GLM', 'GLMsingle Type A positive z': 'GLMsingle A', 'GLMsingle Type D positive z': 'GLMsingle D', 'Optimization p80': 'Opt. p80', 'Optimization p90': 'Opt. p90'}
    return replacements.get(name, name.replace('Optimization matched to ', 'Opt. matched\n'))

def _plot_overlap_heatmap(overlap_df, mask_names, out_base):
    matrix = pd.DataFrame(np.eye(len(mask_names)), index=mask_names, columns=mask_names)
    for row in overlap_df.itertuples(index=False):
        if row.mask_a in matrix.index and row.mask_b in matrix.columns:
            matrix.loc[row.mask_a, row.mask_b] = row.dice
            matrix.loc[row.mask_b, row.mask_a] = row.dice
    (fig, ax) = plt.subplots(figsize=(8.4, 7.2), facecolor='white')
    values = matrix.to_numpy(dtype=float)
    im = ax.imshow(values, vmin=0, vmax=1, cmap='Blues')
    ax.set_xticks(np.arange(len(mask_names)))
    ax.set_xticklabels([_short_mask_label(name) for name in mask_names], rotation=35, ha='right')
    ax.set_yticks(np.arange(len(mask_names)))
    ax.set_yticklabels([_short_mask_label(name) for name in mask_names])
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            ax.text(col, row, f'{values[row, col]:.2f}', ha='center', va='center', fontsize=8, color='white' if values[row, col] >= 0.55 else 'black')
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_ticks(np.linspace(0, 1, 6))
    fig.tight_layout()
    fig.savefig(f'{out_base}_overlap_heatmap.png', dpi=220, bbox_inches='tight', pad_inches=0.04)
    fig.savefig(f'{out_base}_overlap_heatmap.pdf', bbox_inches='tight', pad_inches=0.04)
    plt.close(fig)

def _top_region_bullets(region_df, mask_name, max_regions=8):
    rows = region_df[region_df['mask_name'].eq(mask_name) & region_df['present_for_report'] & ~region_df['roi_name'].eq(UNASSIGNED_ROI)].sort_values('n_voxels', ascending=False).head(max_regions)
    return [f'{row.roi_name}: {int(row.n_voxels):,} voxels ({100.0 * float(row.percent_of_map):.1f}% of highlighted map)' for row in rows.itertuples(index=False)]

def _write_report(out_base, map_specs, summary_df, region_df, overlap_df, primary_mask_names, metadata, z_thresholds, weight_percentiles, analysis_mask):
    atlas_info = metadata.get('atlas_info', {})
    atlas_name = atlas_info.get('description', 'AAL3v2') if isinstance(atlas_info, dict) else 'AAL3v2'
    summary_lookup = summary_df.set_index('mask_name')
    input_lines = '\n'.join((f'- {spec.method}: `{spec.path}`' for spec in map_specs))
    primary_lines = []
    for mask_name in primary_mask_names:
        row = summary_lookup.loc[mask_name]
        top_regions = _top_region_bullets(region_df, mask_name)
        primary_lines.append('\n'.join([f'### {mask_name}', f'- Highlighted voxels: {int(row.n_voxels):,}', f'- Atlas-assigned fraction: {float(row.atlas_assigned_fraction):.1%}', f'- Reportable regions: {int(row.n_reportable_regions):,}', '- Top regions: ' + '; '.join(top_regions)]))
    primary_overlap = overlap_df[overlap_df['mask_a'].isin(primary_mask_names) & overlap_df['mask_b'].isin(primary_mask_names)].copy()
    primary_overlap = primary_overlap.sort_values('dice', ascending=False)
    overlap_lines = [f"- {row.mask_a} vs {row.mask_b}: Dice={row.dice:.3f}, Jaccard={row.jaccard:.3f}, shared={int(row.shared_voxels):,}; {row.shared_top_regions or 'no shared reportable region'}" for row in primary_overlap.itertuples(index=False)]
    z_threshold_text = '; '.join((f'{method}: z >= {threshold:g}' for (method, threshold) in z_thresholds.items()))
    weight_threshold_text = ', '.join((f'p{percentile:g}' for percentile in weight_percentiles))
    report = f'# GLM, GLMsingle, and Optimization Region Comparison\n\n## Inputs\n\n{input_lines}\n\n## Method\n\n- All images were checked on the same grid and summarized in MNI 2 mm space.\n- The analysis mask was defined as the union of nonzero GLMsingle Type A, GLMsingle Type D, and optimization-weight voxels ({int(np.count_nonzero(analysis_mask)):,} voxels). This avoids treating the standard-GLM full field of view as highlighted signal.\n- Positive z-map highlights used {z_threshold_text}. Negative and absolute-z summaries were also exported for sensitivity checks.\n- The optimization map was summarized as a weight-importance map using {weight_threshold_text} nonzero-weight percentile thresholds, plus voxel-count-matched top-weight masks for each positive-z and absolute-z map.\n- Atlas summaries used {atlas_name}, with the same bilateral coarse AAL3 grouping as the threshold-robustness figure.\n\n## Primary Positive-Z vs Weight Summary\n\n{chr(10).join(primary_lines)}\n\n## Primary Pairwise Spatial Overlap\n\n{chr(10).join(overlap_lines)}\n\n## Outputs\n\n- `{out_base}_summary.csv`\n- `{out_base}_regions.csv`\n- `{out_base}_overlaps.csv`\n- `{out_base}_region_by_method.csv`\n- `{out_base}_region_heatmap.png`\n- `{out_base}_region_heatmap.pdf`\n- `{out_base}_overlap_heatmap.png`\n- `{out_base}_overlap_heatmap.pdf`\n- `{out_base}.json`\n'
    Path(f'{out_base}.md').write_text(report, encoding='utf-8')

def _write_json(out_base, map_specs, metadata, z_thresholds, weight_percentiles, min_report_voxels, analysis_mask):
    payload = {'inputs': [{'method': spec.method, 'path': str(spec.path), 'kind': spec.kind} for spec in map_specs], 'z_thresholds': {method: float(threshold) for (method, threshold) in z_thresholds.items()}, 'weight_percentiles': [float(percentile) for percentile in weight_percentiles], 'min_report_voxels': int(min_report_voxels), 'analysis_mask': {'definition': 'union of nonzero GLMsingle Type A, GLMsingle Type D, and optimization-weight voxels', 'n_voxels': int(np.count_nonzero(analysis_mask))}, 'atlas': metadata}
    Path(f'{out_base}.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')

def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--standard-glm', type=Path, default=DEFAULT_STANDARD_GLM)
    parser.add_argument('--glmsingle-a', type=Path, default=DEFAULT_GLM_SINGLE_A)
    parser.add_argument('--glmsingle-d', type=Path, default=DEFAULT_GLM_SINGLE_D)
    parser.add_argument('--weight-map', type=Path, default=DEFAULT_WEIGHT_MAP)
    parser.add_argument('--out-base', type=Path, default=DEFAULT_OUT_BASE)
    parser.add_argument('--standard-z-threshold', type=float, default=DEFAULT_STANDARD_Z_THRESHOLD)
    parser.add_argument('--glmsingle-z-threshold', type=float, default=DEFAULT_GLM_SINGLE_Z_THRESHOLD)
    parser.add_argument('--weight-percentiles', type=float, nargs='+', default=list(DEFAULT_WEIGHT_PERCENTILES))
    parser.add_argument('--min-report-voxels', type=int, default=DEFAULT_MIN_REPORT_VOXELS)
    parser.add_argument('--heatmap-min-row-percent', type=float, default=DEFAULT_HEATMAP_MIN_ROW_PERCENT)
    parser.add_argument('--aal-version', default=DEFAULT_AAL_VERSION)
    parser.add_argument('--atlas-cache-dir', type=Path, default=DEFAULT_ATLAS_CACHE_DIR)
    return parser

def main():
    args = build_parser().parse_args()
    map_specs = [MapSpec('Standard GLM', args.standard_glm, 'z'), MapSpec('GLMsingle Type A', args.glmsingle_a, 'z'), MapSpec('GLMsingle Type D', args.glmsingle_d, 'z'), MapSpec('Optimization weights', args.weight_map, 'weight')]
    images = {spec.method: _load_img(spec.path) for spec in map_specs}
    reference_img = images['Optimization weights']
    _check_same_grid(reference_img, images)
    map_data = {name: np.asarray(img.get_fdata(), dtype=float) for (name, img) in images.items()}
    analysis_mask = _build_analysis_mask(map_data)
    if not np.any(analysis_mask):
        raise RuntimeError('The analysis mask is empty.')
    (groups, metadata) = _build_roi_groups(reference_img, args.aal_version, args.atlas_cache_dir)
    region_sizes = _region_denominators(groups, analysis_mask)
    z_thresholds = {'Standard GLM': float(args.standard_z_threshold), 'GLMsingle Type A': float(args.glmsingle_z_threshold), 'GLMsingle Type D': float(args.glmsingle_z_threshold)}
    weight_percentiles = sorted(set((float(percentile) for percentile in args.weight_percentiles)))
    masks = _make_masks(map_data, analysis_mask, z_thresholds, weight_percentiles)
    region_df = pd.concat([_assign_regions(mask_spec=mask, affine=reference_img.affine, groups=groups, region_sizes=region_sizes, min_report_voxels=args.min_report_voxels) for mask in masks], ignore_index=True)
    summary_df = _summarize_masks(masks, region_df)
    primary_mask_names = [f'Standard GLM positive z', f'GLMsingle Type A positive z', f'GLMsingle Type D positive z'] + [f'Optimization p{percentile:g}' for percentile in weight_percentiles]
    heatmap_mask_names = [name for name in primary_mask_names if name != 'Optimization p80']
    primary_masks = [mask for mask in masks if mask.mask_name in primary_mask_names]
    overlap_df = _overlap_table(primary_masks, groups, args.min_report_voxels)
    matched_mask_names = primary_mask_names + [mask.mask_name for mask in masks if mask.family == 'weight_matched_positive_z']
    matched_overlap_df = _overlap_table([mask for mask in masks if mask.mask_name in matched_mask_names], groups, args.min_report_voxels)
    matched_overlap_df = matched_overlap_df[~(matched_overlap_df['mask_a'].isin(primary_mask_names) & matched_overlap_df['mask_b'].isin(primary_mask_names))]
    overlap_df = pd.concat([overlap_df.assign(overlap_set='primary'), matched_overlap_df.assign(overlap_set='count_matched')], ignore_index=True).drop_duplicates(subset=['overlap_set', 'mask_a', 'mask_b'])
    out_base = args.out_base
    out_base.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(f'{out_base}_summary.csv', index=False)
    region_df.to_csv(f'{out_base}_regions.csv', index=False)
    overlap_df.to_csv(f'{out_base}_overlaps.csv', index=False)
    _wide_region_table(region_df, primary_mask_names).to_csv(f'{out_base}_region_by_method.csv')
    _plot_region_heatmap(region_df, heatmap_mask_names, out_base, args.heatmap_min_row_percent)
    _plot_overlap_heatmap(overlap_df[overlap_df['overlap_set'].eq('primary')], heatmap_mask_names, out_base)
    _write_report(out_base=out_base, map_specs=map_specs, summary_df=summary_df, region_df=region_df, overlap_df=overlap_df[overlap_df['overlap_set'].eq('primary')], primary_mask_names=primary_mask_names, metadata=metadata, z_thresholds=z_thresholds, weight_percentiles=weight_percentiles, analysis_mask=analysis_mask)
    _write_json(out_base=out_base, map_specs=map_specs, metadata=metadata, z_thresholds=z_thresholds, weight_percentiles=weight_percentiles, min_report_voxels=args.min_report_voxels, analysis_mask=analysis_mask)
    print(summary_df[summary_df['mask_name'].isin(primary_mask_names)].to_string(index=False))
    print(f'Saved {out_base}_summary.csv')
    print(f'Saved {out_base}_regions.csv')
    print(f'Saved {out_base}_overlaps.csv')
    print(f'Saved {out_base}_region_by_method.csv')
    print(f'Saved {out_base}.md')
if __name__ == '__main__':
    main()
