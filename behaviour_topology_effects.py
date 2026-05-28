#!/usr/bin/env python3
import argparse
import json
import re
import warnings
from pathlib import Path

import matplotlib

warnings.filterwarnings('ignore', message='Unable to import Axes3D.*')
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from med_effects import COMPARISON_METRIC, CONNECTIVITY_METRIC, _laplacian_spectral_distance_signed


DEFAULT_BEHAVIOUR_ROOT = Path('/mnt/TeamShare/Data_Masterfile/H20-00572_All-Dressed/Zahra-Thesis-Data/fmri_opt_group/behaviour')
DEFAULT_NETWORK_DIR = Path('figures/med_effects/network_matrices')
DEFAULT_OUT_DIR = Path('figures/med_effects')
DEFAULT_RT_COLUMN_ONE_BASED = 2
DEFAULT_MIN_RT = 0.0
DEFAULT_N_PERMUTATIONS = 10000
DEFAULT_RANDOM_STATE = 0
DEFAULT_SESSION_STATES = {'1': 'off', '2': 'on'}

BEHAVIOUR_FILE_RE = re.compile(r'^PSPD(?P<subject>\d+)_ses_(?P<session>\d+)_run_(?P<run>\d+)\.npy$')
NETWORK_FILE_RE = re.compile(r'^(?P<subject>sub-pd\d+)_ses-(?P<session>\d+)\.csv$')


ASSOCIATIONS = (
    {
        'analysis': 'topology_change_predicts_mean_rt_change',
        'predictor': 'topology_change_distance',
        'outcome': 'delta_mean_rt_on_minus_off',
        'xlabel': 'OFF-to-ON topology change\n(Laplacian spectral distance)',
        'ylabel': 'Mean RT change (ON - OFF, s)',
        'title': 'Topology Change vs Mean RT Change',
        'figure_stem': 'behaviour_topology_change_mean_rt_change',
    },
    {
        'analysis': 'topology_change_predicts_rt_variability_change',
        'predictor': 'topology_change_distance',
        'outcome': 'delta_rt_sd_on_minus_off',
        'xlabel': 'OFF-to-ON topology change\n(Laplacian spectral distance)',
        'ylabel': 'RT variability change (ON - OFF SD, s)',
        'title': 'Topology Change vs RT Variability Change',
        'figure_stem': 'behaviour_topology_change_rt_variability_change',
    },
    {
        'analysis': 'off_distance_to_on_centroid_predicts_mean_rt_improvement',
        'predictor': 'off_distance_to_loo_on_centroid',
        'outcome': 'mean_rt_improvement_off_minus_on',
        'xlabel': 'OFF distance to leave-one-out ON centroid',
        'ylabel': 'Mean RT improvement (OFF - ON, s)',
        'title': 'ON Centroid Distance vs Mean RT Improvement',
        'figure_stem': 'behaviour_on_centroid_distance_mean_rt_improvement',
    },
    {
        'analysis': 'movement_toward_on_centroid_predicts_mean_rt_improvement',
        'predictor': 'movement_toward_loo_on_centroid',
        'outcome': 'mean_rt_improvement_off_minus_on',
        'xlabel': 'Movement toward leave-one-out ON centroid',
        'ylabel': 'Mean RT improvement (OFF - ON, s)',
        'title': 'Centroid Movement vs Mean RT Improvement',
        'figure_stem': 'behaviour_on_centroid_movement_mean_rt_improvement',
    },
)


def _subject_id_from_pspd(number):
    return f'sub-pd{str(number).zfill(3)}'


def _state_for_session(session):
    state = DEFAULT_SESSION_STATES.get(str(session))
    if state is None:
        raise RuntimeError(f'No OFF/ON state mapping is defined for session {session}')
    return state


def _rt_values_from_npy(path, rt_column_index, min_rt=None, max_rt=None):
    data = np.load(path, allow_pickle=False)
    values = np.asarray(data, dtype=np.float64)
    if values.ndim == 1:
        rt = values
        source_kind = 'rt_vector'
    elif values.ndim == 2:
        if values.shape[1] <= rt_column_index:
            raise RuntimeError(f'{path} has {values.shape[1]} columns; cannot read RT column {rt_column_index + 1}')
        rt = values[:, rt_column_index]
        source_kind = f'matrix_column_{rt_column_index + 1}'
    else:
        raise RuntimeError(f'{path} must be a 1D RT vector or 2D behavioural matrix, got shape {values.shape}')
    rt = np.asarray(rt, dtype=np.float64)
    keep = np.isfinite(rt)
    if min_rt is not None:
        keep &= rt >= float(min_rt)
    if max_rt is not None:
        keep &= rt <= float(max_rt)
    return (rt[keep], source_kind)


def _session_summary(subject, session, values, source_files, source_kinds):
    if values.size == 0:
        raise RuntimeError(f'No finite RT values remained for {subject} session {session}')
    q25, q75 = np.percentile(values, [25, 75])
    state = _state_for_session(session)
    label = f'{subject}_ses-{session}'
    return {
        'subject': subject,
        'session': str(session),
        'state': state,
        'label': label,
        'n_rt': int(values.size),
        'mean_rt': float(np.mean(values)),
        'sd_rt': float(np.std(values, ddof=1)) if values.size > 1 else float('nan'),
        'median_rt': float(np.median(values)),
        'iqr_rt': float(q75 - q25),
        'source_files': ';'.join(source_files),
        'source_kinds': ';'.join(sorted(set(source_kinds))),
    }


def _load_behaviour_sessions(root, rt_column_index, min_rt=None, max_rt=None):
    grouped = {}
    source_kinds = {}
    for path in sorted(root.glob('PSPD*_ses_*_run_*.npy')):
        match = BEHAVIOUR_FILE_RE.match(path.name)
        if match is None:
            continue
        subject = _subject_id_from_pspd(match.group('subject'))
        session = str(match.group('session'))
        run = int(match.group('run'))
        (rt, source_kind) = _rt_values_from_npy(path, rt_column_index, min_rt=min_rt, max_rt=max_rt)
        key = (subject, session)
        grouped.setdefault(key, []).append((run, path.name, rt))
        source_kinds.setdefault(key, []).append(source_kind)
    if not grouped:
        raise RuntimeError(f'No PSPD*_ses_*_run_*.npy behaviour files found in {root}')
    rows = []
    for ((subject, session), run_values) in sorted(grouped.items()):
        ordered = sorted(run_values, key=lambda item: item[0])
        values = np.concatenate([rt for (_, _, rt) in ordered])
        source_files = [name for (_, name, _) in ordered]
        rows.append(_session_summary(subject, session, values, source_files, source_kinds[(subject, session)]))
    return pd.DataFrame(rows).sort_values(['subject', 'session']).reset_index(drop=True)


def _load_networks(network_dir):
    networks = {}
    rows = []
    for path in sorted(network_dir.glob('sub-pd*_ses-*.csv')):
        match = NETWORK_FILE_RE.match(path.name)
        if match is None:
            continue
        subject = str(match.group('subject'))
        session = str(match.group('session'))
        label = f'{subject}_ses-{session}'
        matrix = pd.read_csv(path, index_col=0).to_numpy(dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise RuntimeError(f'{path} is not a square network matrix')
        networks[label] = matrix
        rows.append({'subject': subject, 'session': session, 'state': _state_for_session(session), 'label': label})
    if not networks:
        raise RuntimeError(f'No network matrices found in {network_dir}')
    return (networks, pd.DataFrame(rows).sort_values(['subject', 'session']).reset_index(drop=True))


def _complete_subjects(behaviour, network_sessions):
    behaviour_subjects = set()
    for (subject, rows) in behaviour.groupby('subject'):
        states = set(rows['state'].astype(str))
        if {'off', 'on'}.issubset(states):
            behaviour_subjects.add(str(subject))
    network_subjects = set()
    for (subject, rows) in network_sessions.groupby('subject'):
        states = set(rows['state'].astype(str))
        if {'off', 'on'}.issubset(states):
            network_subjects.add(str(subject))
    return sorted(behaviour_subjects & network_subjects)


def _row_for_state(behaviour, subject, state):
    rows = behaviour.loc[(behaviour['subject'] == subject) & (behaviour['state'] == state)]
    if rows.shape[0] != 1:
        raise RuntimeError(f'Expected one {state.upper()} behaviour row for {subject}, found {rows.shape[0]}')
    return rows.iloc[0]


def _label_for_state(network_sessions, subject, state):
    rows = network_sessions.loc[(network_sessions['subject'] == subject) & (network_sessions['state'] == state)]
    if rows.shape[0] != 1:
        raise RuntimeError(f'Expected one {state.upper()} network row for {subject}, found {rows.shape[0]}')
    return str(rows.iloc[0]['label'])


def _mean_matrix(matrices):
    first_shape = matrices[0].shape
    for matrix in matrices:
        if matrix.shape != first_shape:
            raise RuntimeError('Network matrices have inconsistent shapes')
    return np.mean(np.stack(matrices, axis=0), axis=0)


def _build_subject_table(behaviour, networks, network_sessions):
    subjects = _complete_subjects(behaviour, network_sessions)
    if len(subjects) < 3:
        raise RuntimeError('At least three complete OFF/ON subjects are required for behaviour-topology associations')
    on_labels = {subject: _label_for_state(network_sessions, subject, 'on') for subject in subjects}
    rows = []
    for subject in subjects:
        off_behaviour = _row_for_state(behaviour, subject, 'off')
        on_behaviour = _row_for_state(behaviour, subject, 'on')
        off_label = _label_for_state(network_sessions, subject, 'off')
        on_label = on_labels[subject]
        off_matrix = networks[off_label]
        on_matrix = networks[on_label]
        other_on_matrices = [networks[label] for (other_subject, label) in on_labels.items() if other_subject != subject]
        if not other_on_matrices:
            raise RuntimeError('Leave-one-out ON centroid requires at least two complete subjects')
        on_centroid = _mean_matrix(other_on_matrices)
        off_distance_to_centroid = _laplacian_spectral_distance_signed(off_matrix, on_centroid)
        on_distance_to_centroid = _laplacian_spectral_distance_signed(on_matrix, on_centroid)
        row = {
            'subject': subject,
            'off_label': off_label,
            'on_label': on_label,
            'n_rt_off': int(off_behaviour['n_rt']),
            'n_rt_on': int(on_behaviour['n_rt']),
            'mean_rt_off': float(off_behaviour['mean_rt']),
            'mean_rt_on': float(on_behaviour['mean_rt']),
            'sd_rt_off': float(off_behaviour['sd_rt']),
            'sd_rt_on': float(on_behaviour['sd_rt']),
            'median_rt_off': float(off_behaviour['median_rt']),
            'median_rt_on': float(on_behaviour['median_rt']),
            'iqr_rt_off': float(off_behaviour['iqr_rt']),
            'iqr_rt_on': float(on_behaviour['iqr_rt']),
            'delta_mean_rt_on_minus_off': float(on_behaviour['mean_rt'] - off_behaviour['mean_rt']),
            'delta_rt_sd_on_minus_off': float(on_behaviour['sd_rt'] - off_behaviour['sd_rt']),
            'mean_rt_improvement_off_minus_on': float(off_behaviour['mean_rt'] - on_behaviour['mean_rt']),
            'rt_variability_improvement_off_minus_on': float(off_behaviour['sd_rt'] - on_behaviour['sd_rt']),
            'topology_change_distance': _laplacian_spectral_distance_signed(off_matrix, on_matrix),
            'off_distance_to_loo_on_centroid': off_distance_to_centroid,
            'on_distance_to_loo_on_centroid': on_distance_to_centroid,
            'delta_distance_to_loo_on_centroid_on_minus_off': on_distance_to_centroid - off_distance_to_centroid,
            'movement_toward_loo_on_centroid': off_distance_to_centroid - on_distance_to_centroid,
            'loo_on_centroid_n_subjects': int(len(other_on_matrices)),
            'off_behaviour_files': str(off_behaviour['source_files']),
            'on_behaviour_files': str(on_behaviour['source_files']),
            'off_behaviour_source_kinds': str(off_behaviour['source_kinds']),
            'on_behaviour_source_kinds': str(on_behaviour['source_kinds']),
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values('subject').reset_index(drop=True)


def _permutation_corr_p_value(x_values, y_values, observed_r, n_permutations, random_state):
    if not np.isfinite(observed_r):
        return float('nan')
    rng = np.random.default_rng(random_state)
    count = 0
    y_values = np.asarray(y_values, dtype=np.float64)
    for _ in range(int(n_permutations)):
        permuted = rng.permutation(y_values)
        if np.std(permuted) <= 0:
            continue
        permuted_r = float(np.corrcoef(x_values, permuted)[0, 1])
        if abs(permuted_r) >= abs(observed_r) - 1e-12:
            count += 1
    return float((count + 1) / (int(n_permutations) + 1))


def _association_stats(subject_table, analysis, predictor, outcome, n_permutations, random_state):
    subset = subject_table[['subject', predictor, outcome]].replace([np.inf, -np.inf], np.nan).dropna()
    x_values = subset[predictor].to_numpy(dtype=np.float64)
    y_values = subset[outcome].to_numpy(dtype=np.float64)
    result = {
        'analysis': analysis,
        'predictor': predictor,
        'outcome': outcome,
        'n_subjects': int(subset.shape[0]),
    }
    if subset.shape[0] < 3 or np.std(x_values) <= 0 or np.std(y_values) <= 0:
        result.update({
            'pearson_r': float('nan'),
            'pearson_p_value': float('nan'),
            'spearman_r': float('nan'),
            'spearman_p_value': float('nan'),
            'permutation_p_value_two_sided': float('nan'),
            'slope': float('nan'),
            'intercept': float('nan'),
            'r_squared': float('nan'),
        })
        return result
    linear = stats.linregress(x_values, y_values)
    pearson = stats.pearsonr(x_values, y_values)
    spearman = stats.spearmanr(x_values, y_values)
    df = int(subset.shape[0] - 2)
    t_critical = float(stats.t.ppf(0.975, df)) if df > 0 else float('nan')
    slope_ci_low = float(linear.slope - t_critical * linear.stderr) if np.isfinite(t_critical) else float('nan')
    slope_ci_high = float(linear.slope + t_critical * linear.stderr) if np.isfinite(t_critical) else float('nan')
    permutation_p = _permutation_corr_p_value(
        x_values,
        y_values,
        float(pearson.statistic),
        n_permutations=n_permutations,
        random_state=random_state,
    )
    result.update({
        'pearson_r': float(pearson.statistic),
        'pearson_p_value': float(pearson.pvalue),
        'spearman_r': float(spearman.statistic),
        'spearman_p_value': float(spearman.pvalue),
        'permutation_p_value_two_sided': permutation_p,
        'slope': float(linear.slope),
        'intercept': float(linear.intercept),
        'slope_standard_error': float(linear.stderr),
        'slope_ci95_low': slope_ci_low,
        'slope_ci95_high': slope_ci_high,
        'r_squared': float(linear.rvalue ** 2),
    })
    return result


def _format_p_value(p_value):
    if not np.isfinite(p_value):
        return 'n/a'
    if p_value < 0.001:
        return '<0.001'
    return f'{p_value:.3f}'.lstrip('0')


def _plot_association(subject_table, definition, stats_row, out_dir):
    predictor = definition['predictor']
    outcome = definition['outcome']
    subset = subject_table[['subject', predictor, outcome]].replace([np.inf, -np.inf], np.nan).dropna()
    x_values = subset[predictor].to_numpy(dtype=np.float64)
    y_values = subset[outcome].to_numpy(dtype=np.float64)
    fig, ax = plt.subplots(figsize=(4.9, 4.15))
    ax.scatter(x_values, y_values, s=34, color='#4C78A8', edgecolor='white', linewidth=0.7, zorder=3)
    if subset.shape[0] >= 2 and np.isfinite(stats_row.get('slope', np.nan)):
        x_grid = np.linspace(float(np.min(x_values)), float(np.max(x_values)), 100)
        y_grid = float(stats_row['intercept']) + float(stats_row['slope']) * x_grid
        ax.plot(x_grid, y_grid, color='#D65F5F', linewidth=1.4, zorder=2)
    for row in subset.itertuples(index=False):
        label = str(row.subject).replace('sub-pd', 'pd')
        ax.annotate(label, (float(getattr(row, predictor)), float(getattr(row, outcome))), xytext=(3, 2), textcoords='offset points', fontsize=6.4, color='#444444', alpha=0.82)
    ax.axhline(0.0, color='#aaaaaa', linewidth=0.8, zorder=1)
    ax.set_xlabel(definition['xlabel'])
    ax.set_ylabel(definition['ylabel'])
    ax.set_title(definition['title'], fontsize=11.5)
    annotation = (
        f"r={stats_row.get('pearson_r', np.nan):.2f}\n"
        f"p={_format_p_value(stats_row.get('pearson_p_value', np.nan))}\n"
        f"perm p={_format_p_value(stats_row.get('permutation_p_value_two_sided', np.nan))}\n"
        f"n={int(stats_row.get('n_subjects', 0))}"
    )
    ax.text(0.03, 0.97, annotation, transform=ax.transAxes, ha='left', va='top', fontsize=8.3, bbox={'facecolor': 'white', 'edgecolor': '#dddddd', 'alpha': 0.9, 'boxstyle': 'round,pad=0.32'})
    ax.grid(color='#dddddd', linewidth=0.55, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    png_path = out_dir / f"{definition['figure_stem']}.png"
    pdf_path = png_path.with_suffix('.pdf')
    fig.savefig(png_path, dpi=320, bbox_inches='tight', pad_inches=0.04)
    fig.savefig(pdf_path, bbox_inches='tight', pad_inches=0.04)
    plt.close(fig)
    return png_path


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for (key, item) in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--behaviour-root', type=Path, default=DEFAULT_BEHAVIOUR_ROOT)
    parser.add_argument('--network-dir', type=Path, default=DEFAULT_NETWORK_DIR)
    parser.add_argument('--out-dir', type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument('--rt-column', type=int, default=DEFAULT_RT_COLUMN_ONE_BASED, help='One-based RT column for 2D behavioural .npy matrices.')
    parser.add_argument('--min-rt', type=float, default=DEFAULT_MIN_RT)
    parser.add_argument('--max-rt', type=float, default=None)
    parser.add_argument('--n-permutations', type=int, default=DEFAULT_N_PERMUTATIONS)
    parser.add_argument('--random-state', type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument('--check-inputs', action='store_true')
    return parser


def _check_inputs(args):
    missing = []
    if not args.behaviour_root.exists():
        missing.append(f'{args.behaviour_root} (behaviour root)')
    if not args.network_dir.exists():
        missing.append(f'{args.network_dir} (network matrices)')
    if args.rt_column < 1:
        missing.append('--rt-column must be one-based and >= 1')
    return missing


def main():
    args = build_parser().parse_args()
    missing = _check_inputs(args)
    if missing:
        print('Missing required inputs:')
        for item in missing:
            print(f'- {item}')
        return 1
    if args.check_inputs:
        print('All required inputs are present.')
        return 0
    rt_column_index = int(args.rt_column) - 1
    args.out_dir.mkdir(parents=True, exist_ok=True)
    behaviour = _load_behaviour_sessions(
        args.behaviour_root,
        rt_column_index=rt_column_index,
        min_rt=args.min_rt,
        max_rt=args.max_rt,
    )
    (networks, network_sessions) = _load_networks(args.network_dir)
    subject_table = _build_subject_table(behaviour, networks, network_sessions)
    behaviour_path = args.out_dir / 'behaviour_rt_session_summary.csv'
    subject_path = args.out_dir / 'behaviour_topology_subject_values.csv'
    stats_path = args.out_dir / 'behaviour_topology_association_stats.csv'
    json_path = args.out_dir / 'behaviour_topology_association_stats.json'
    behaviour.to_csv(behaviour_path, index=False)
    subject_table.to_csv(subject_path, index=False)
    stats_rows = []
    figure_paths = []
    for definition in ASSOCIATIONS:
        stats_row = _association_stats(
            subject_table,
            analysis=definition['analysis'],
            predictor=definition['predictor'],
            outcome=definition['outcome'],
            n_permutations=args.n_permutations,
            random_state=args.random_state,
        )
        stats_rows.append(stats_row)
        figure_paths.append(_plot_association(subject_table, definition, stats_row, args.out_dir))
    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(stats_path, index=False)
    summary = {
        'inputs': {
            'behaviour_root': str(args.behaviour_root),
            'network_dir': str(args.network_dir),
            'out_dir': str(args.out_dir),
            'rt_column_one_based_for_2d_matrices': int(args.rt_column),
            'min_rt': args.min_rt,
            'max_rt': args.max_rt,
            'connectivity_metric': CONNECTIVITY_METRIC,
            'comparison_metric': COMPARISON_METRIC,
            'n_permutations': int(args.n_permutations),
            'random_state': int(args.random_state),
        },
        'notes': [
            '2D behavioural .npy files use the requested second column as RT.',
            '1D behavioural .npy files are treated as already-extracted RT vectors.',
            'RT change is ON minus OFF; RT improvement is OFF minus ON.',
            'ON centroid distances use a leave-one-subject-out ON centroid.',
        ],
        'n_behaviour_sessions': int(behaviour.shape[0]),
        'n_network_sessions': int(network_sessions.shape[0]),
        'n_analysis_subjects': int(subject_table.shape[0]),
        'analysis_subjects': subject_table['subject'].astype(str).tolist(),
        'outputs': {
            'behaviour_session_summary': str(behaviour_path),
            'subject_values': str(subject_path),
            'association_stats_csv': str(stats_path),
            'association_figures': [str(path) for path in figure_paths],
        },
        'associations': stats_rows,
    }
    json_path.write_text(json.dumps(_json_safe(summary), indent=2), encoding='utf-8')
    print(f'Saved {behaviour_path}')
    print(f'Saved {subject_path}')
    print(f'Saved {stats_path}')
    print(f'Saved {json_path}')
    for figure_path in figure_paths:
        print(f'Saved {figure_path}')
        print(f"Saved {figure_path.with_suffix('.pdf')}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
