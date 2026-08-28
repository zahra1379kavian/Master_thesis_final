#!/usr/bin/env python3
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib

warnings.filterwarnings("ignore", message="Unable to import Axes3D.*")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spagetti_plot import (
    DEFAULT_ACTIVE_BOLD_GROUP,
    DEFAULT_ACTIVE_FLAT_INDICES,
    DEFAULT_GVS_DIR,
    DEFAULT_GVS_OUT_DIR,
    DEFAULT_GVS_SHAM,
    DEFAULT_MANIFEST,
    DEFAULT_TASK_MAP,
    DEFAULT_WEIGHT_HTML,
    DEFAULT_WEIGHT_MAP,
    _gvs_display_label,
    _gvs_subject_timecourse_pairs,
    _gvs_trials_from_full_trials,
    _make_gvs_frames,
    _reduce_task_map_bold_trials,
    _weighted_html_projection_from_active_bold,
)


SESSION_SPECS = (
    (1, "OFF", "session1_off", "Medication OFF, session 1"),
    (2, "ON", "session2_on", "Medication ON, session 2"),
)

GVS_STYLES = {
    "sham": ("#111111", "o", "-"),
    "gvs1": ("#0072B2", "o", "-"),
    "gvs2": ("#D55E00", "o", "-"),
    "gvs3": ("#009E73", "o", "-"),
    "gvs4": ("#CC79A7", "o", "-"),
    "gvs5": ("#E69F00", "o", "-"),
    "gvs6": ("#56B4E9", "o", "-"),
    "gvs7": ("#7F3C8D", "o", "-"),
    "gvs8": ("#A6761D", "o", "-"),
}

GVS_OFFSET_MULTIPLIERS = {
    "sham": -4,
    "gvs1": -3,
    "gvs2": -2,
    "gvs3": -1,
    "gvs4": 0,
    "gvs5": 1,
    "gvs6": 2,
    "gvs7": 3,
    "gvs8": 4,
}


def _time_indices(timecourse_pairs: pd.DataFrame) -> list[int]:
    return sorted(
        int(col[1:].split("_", 1)[0])
        for col in timecourse_pairs.columns
        if col.startswith("t") and col.endswith("_active")
    )


def _mean_lines(
    timecourse_pairs: pd.DataFrame,
    active_codes: list[str],
    sham_code: str,
) -> tuple[np.ndarray, list[tuple[str, str, np.ndarray]]]:
    time_indices = _time_indices(timecourse_pairs)
    if not time_indices:
        raise ValueError("No active timecourse columns were found")

    time = np.asarray(time_indices, dtype=int)
    lines: list[tuple[str, str, np.ndarray]] = []

    sham_cols = [f"t{idx}_sham" for idx in time_indices]
    dedupe_cols = [col for col in ["subject", "session", "medication", "run"] if col in timecourse_pairs.columns]
    sham_data = timecourse_pairs.drop_duplicates(dedupe_cols) if dedupe_cols else timecourse_pairs
    sham_values = sham_data[sham_cols].to_numpy(dtype=np.float64)
    lines.append((str(sham_code), _gvs_display_label(sham_code), np.nanmean(sham_values, axis=0)))

    active_cols = [f"t{idx}_active" for idx in time_indices]
    for active_code in active_codes:
        data = timecourse_pairs.loc[timecourse_pairs["active_gvs"].eq(active_code)]
        if data.empty:
            continue
        active_values = data[active_cols].to_numpy(dtype=np.float64)
        lines.append((str(active_code), _gvs_display_label(active_code), np.nanmean(active_values, axis=0)))

    return time, lines


def _session_lines(
    signal_frame: pd.DataFrame,
    active_codes: list[str],
    sham_code: str,
) -> list[tuple[str, np.ndarray, list[tuple[str, str, np.ndarray]]]]:
    panels = []
    for session, medication, _suffix, title in SESSION_SPECS:
        session_signal = signal_frame.loc[
            signal_frame["session"].astype(int).eq(session)
            & signal_frame["medication"].astype(str).str.upper().eq(medication)
        ].copy()
        if session_signal.empty:
            continue

        timecourse_pairs = _gvs_subject_timecourse_pairs(session_signal, sham_code)
        time, lines = _mean_lines(timecourse_pairs, active_codes, sham_code)
        panels.append((title, time, lines))

    if not panels:
        raise RuntimeError("No session-specific GVS timecourses were found")
    return panels


def _plot_colored_session_subplots(
    panels: list[tuple[str, np.ndarray, list[tuple[str, str, np.ndarray]]]],
    output_path: Path,
    line_offset_step: float,
    y_label: str,
) -> None:
    fig, axes = plt.subplots(1, len(panels), figsize=(7.2 * len(panels), 5.4), sharex=True)
    axes = np.asarray(axes).ravel()
    legend_handles: dict[str, object] = {}
    axis_label_fontsize = 15
    tick_fontsize = 12
    legend_fontsize = 12

    for panel_index, (ax, (_title, time, lines)) in enumerate(zip(axes, panels)):
        for _code, label, mean_signal in lines:
            color, marker, linestyle = GVS_STYLES.get(label, ("#4D4D4D", "o", "-"))
            offset = GVS_OFFSET_MULTIPLIERS.get(label, 0) * line_offset_step
            (handle,) = ax.plot(
                time,
                mean_signal + offset,
                color=color,
                marker=marker,
                linestyle=linestyle,
                markersize=4.8,
                lw=2.2,
                label=label,
                alpha=0.95,
            )
            legend_handles.setdefault(label, handle)

        ax.set_xlabel("Time (TR)", fontsize=axis_label_fontsize, fontweight="bold")
        ax.set_ylabel(
            y_label if panel_index == 0 else "",
            fontsize=axis_label_fontsize,
            fontweight="bold",
        )
        ax.tick_params(axis="both", labelsize=tick_fontsize)
        for tick_label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
            tick_label.set_fontweight("bold")
        ax.grid(color="0.88", lw=0.8)
        ax.margins(x=0.02)

    ordered_labels = [label for label in GVS_STYLES if label in legend_handles]
    fig.legend(
        [legend_handles[label] for label in ordered_labels],
        ordered_labels,
        loc="lower center",
        ncol=min(5, len(ordered_labels)),
        frameon=False,
        prop={"size": legend_fontsize, "weight": "bold"},
    )
    fig.subplots_adjust(bottom=0.24, top=0.97, wspace=0.18)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot session-specific original GVS projection signals with one color per GVS condition."
    )
    parser.add_argument("--gvs-dir", type=Path, default=DEFAULT_GVS_DIR)
    parser.add_argument("--gvs-sham", default=DEFAULT_GVS_SHAM)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_GVS_OUT_DIR)
    parser.add_argument("--active-bold-group", type=Path, default=DEFAULT_ACTIVE_BOLD_GROUP)
    parser.add_argument("--active-flat-indices", type=Path, default=DEFAULT_ACTIVE_FLAT_INDICES)
    parser.add_argument("--projection-weight-map", type=Path, default=DEFAULT_WEIGHT_MAP)
    parser.add_argument("--projection-html-mask", type=Path, default=DEFAULT_WEIGHT_HTML)
    parser.add_argument("--projection-voxel-chunk-size", type=int, default=512)
    parser.add_argument(
        "--task-map-bold-trials",
        type=Path,
        default=None,
        help=f"Optional voxel-by-trial-by-time task-map BOLD array, e.g. {DEFAULT_TASK_MAP}.",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--task-map-reducer", choices=("mean",), default="mean")
    parser.add_argument("--task-map-chunk-size", type=int, default=512)
    parser.add_argument(
        "--line-offset-step",
        type=float,
        default=0.008,
        help="Vertical spacing added between consecutive GVS lines. Use 0 for unshifted lines.",
    )
    parser.add_argument(
        "--use-precomputed-gvs-arrays",
        action="store_true",
        help="Use Projection_BOLD_trials_gvs-*.npy directly instead of recomputing the HTML-mask weighted projection.",
    )
    parser.add_argument(
        "--output-name",
        default="original_signal_gvs_01_to_09_colored_subplots_vertically_offset_session1_off_session2_on.png",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.use_precomputed_gvs_arrays:
        _feature_frame, signal_frame, trials_by_code = _make_gvs_frames(args.gvs_dir)
    elif args.task_map_bold_trials is not None:
        full_trials = _reduce_task_map_bold_trials(
            args.task_map_bold_trials,
            args.manifest,
            args.task_map_reducer,
            args.task_map_chunk_size,
        )
        trials_by_code = _gvs_trials_from_full_trials(full_trials, args.gvs_dir)
        _feature_frame, signal_frame, trials_by_code = _make_gvs_frames(args.gvs_dir, trials_by_code=trials_by_code)
    else:
        full_trials, _projection_metadata = _weighted_html_projection_from_active_bold(
            args.active_bold_group,
            args.active_flat_indices,
            args.projection_weight_map,
            args.projection_html_mask,
            args.projection_voxel_chunk_size,
        )
        trials_by_code = _gvs_trials_from_full_trials(full_trials, args.gvs_dir)
        _feature_frame, signal_frame, trials_by_code = _make_gvs_frames(args.gvs_dir, trials_by_code=trials_by_code)
    y_label = "Projected signal (+offset)" if args.line_offset_step else "Projected signal"
    active_codes = sorted(code for code in trials_by_code if code != args.gvs_sham)
    panels = _session_lines(signal_frame, active_codes, args.gvs_sham)
    _plot_colored_session_subplots(
        panels,
        args.out_dir / args.output_name,
        args.line_offset_step,
        y_label,
    )
    print(f"Wrote {args.out_dir / args.output_name}")


if __name__ == "__main__":
    main()
