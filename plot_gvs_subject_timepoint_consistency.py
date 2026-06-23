#!/usr/bin/env python3
"""Plot subject-level GVS timepoint consistency summaries."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spagetti_plot import DEFAULT_GVS_OUT_DIR, _gvs_display_label


TIME_INDICES = tuple(range(9))
N_BOOTSTRAP = 10_000


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create subject-level timepoint consistency figures for the GVS-minus-sham "
            "and original projection signal plots."
        )
    )
    parser.add_argument(
        "--subject-timecourse-pairs",
        type=Path,
        default=DEFAULT_GVS_OUT_DIR / "gvs_vs_sham_subject_timecourse_pairs.csv",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_GVS_OUT_DIR)
    parser.add_argument("--random-state", type=int, default=0)
    return parser.parse_args()


def _bootstrap_mean_ci(matrix: np.ndarray, random_state: int) -> tuple[np.ndarray, np.ndarray]:
    if matrix.shape[0] <= 1:
        mean = np.nanmean(matrix, axis=0)
        return mean, mean
    rng = np.random.default_rng(random_state)
    sample_indices = rng.integers(0, matrix.shape[0], size=(N_BOOTSTRAP, matrix.shape[0]))
    bootstrap_means = np.nanmean(matrix[sample_indices], axis=1)
    low, high = np.nanpercentile(bootstrap_means, [2.5, 97.5], axis=0)
    return low, high


def _same_direction(values: np.ndarray, reference: float) -> tuple[int, float, str]:
    finite = values[np.isfinite(values)]
    if finite.size == 0 or not np.isfinite(reference):
        return 0, np.nan, "nan"
    if reference < 0:
        count = int(np.count_nonzero(finite < 0))
        direction = "negative"
    elif reference > 0:
        count = int(np.count_nonzero(finite > 0))
        direction = "positive"
    else:
        positive = int(np.count_nonzero(finite > 0))
        negative = int(np.count_nonzero(finite < 0))
        count = max(positive, negative)
        direction = "positive" if positive >= negative else "negative"
    return count, float(count / finite.size), direction


def _records_for_panel(
    *,
    figure: str,
    subplot: str,
    gvs_code: str,
    subjects: pd.Series,
    raw_matrix: np.ndarray,
    plot_matrix: np.ndarray,
    direction_matrix: np.ndarray,
    direction_reference: np.ndarray,
    random_state: int,
) -> list[dict[str, object]]:
    ci_low, ci_high = _bootstrap_mean_ci(plot_matrix, random_state)
    rows: list[dict[str, object]] = []
    for time_pos, time_index in enumerate(TIME_INDICES):
        raw_values = raw_matrix[:, time_pos]
        plot_values = plot_matrix[:, time_pos]
        direction_values = direction_matrix[:, time_pos]
        n_subjects = int(np.count_nonzero(np.isfinite(plot_values)))
        n_same, prop_same, direction = _same_direction(direction_values, direction_reference[time_pos])
        rows.append(
            {
                "figure": figure,
                "subplot": subplot,
                "gvs_code": gvs_code,
                "time_index": time_index,
                "n_subjects": n_subjects,
                "mean": float(np.nanmean(plot_values)),
                "ci95_low": float(ci_low[time_pos]),
                "ci95_high": float(ci_high[time_pos]),
                "sd_between_subjects": float(np.nanstd(plot_values, ddof=1)),
                "iqr_between_subjects": float(np.nanpercentile(plot_values, 75) - np.nanpercentile(plot_values, 25)),
                "raw_mean": float(np.nanmean(raw_values)),
                "raw_sd_between_subjects": float(np.nanstd(raw_values, ddof=1)),
                "direction_reference": float(direction_reference[time_pos]),
                "direction": direction,
                "n_same_direction": n_same,
                "prop_same_direction": prop_same,
            }
        )
    return rows


def _build_delta_panels(pairs: pd.DataFrame, random_state: int) -> tuple[list[dict[str, object]], pd.DataFrame]:
    panels = []
    rows: list[dict[str, object]] = []
    delta_cols = [f"t{idx}_delta" for idx in TIME_INDICES]
    for active_gvs, group in pairs.groupby("active_gvs", sort=True):
        matrix = group[delta_cols].to_numpy(dtype=np.float64)
        mean = np.nanmean(matrix, axis=0)
        panels.append(
            {
                "subplot": _gvs_display_label(active_gvs),
                "gvs_code": active_gvs,
                "subjects": group["subject"].reset_index(drop=True),
                "plot_matrix": matrix,
                "raw_matrix": matrix,
                "direction_matrix": matrix,
                "direction_reference": mean,
                "ylabel": "Active minus sham",
            }
        )
        rows.extend(
            _records_for_panel(
                figure="gvs_minus_sham",
                subplot=_gvs_display_label(active_gvs),
                gvs_code=active_gvs,
                subjects=group["subject"],
                raw_matrix=matrix,
                plot_matrix=matrix,
                direction_matrix=matrix,
                direction_reference=mean,
                random_state=random_state,
            )
        )
    return panels, pd.DataFrame(rows)


def _build_original_panels(pairs: pd.DataFrame, random_state: int) -> tuple[list[dict[str, object]], pd.DataFrame]:
    panels = []
    rows: list[dict[str, object]] = []
    time_cols = [f"t{idx}" for idx in TIME_INDICES]

    sham_cols = [f"t{idx}_sham" for idx in TIME_INDICES]
    sham = pairs.groupby("subject", as_index=False)[sham_cols].mean()
    sham = sham.rename(columns={f"t{idx}_sham": f"t{idx}" for idx in TIME_INDICES})
    original_frames = [sham.assign(gvs_code="gvs-01")]

    for active_gvs, group in pairs.groupby("active_gvs", sort=True):
        active = group[["subject", *[f"t{idx}_active" for idx in TIME_INDICES]]].copy()
        active = active.rename(columns={f"t{idx}_active": f"t{idx}" for idx in TIME_INDICES})
        original_frames.append(active.assign(gvs_code=active_gvs))

    original = pd.concat(original_frames, ignore_index=True)
    code_order = ["gvs-01", *sorted(code for code in pairs["active_gvs"].unique())]
    for gvs_code in code_order:
        group = original.loc[original["gvs_code"].eq(gvs_code)].sort_values("subject")
        raw_matrix = group[time_cols].to_numpy(dtype=np.float64)
        subject_means = np.nanmean(raw_matrix, axis=1, keepdims=True)
        grand_mean = float(np.nanmean(raw_matrix))
        centered_matrix = raw_matrix - subject_means + grand_mean
        direction_matrix = raw_matrix - subject_means
        direction_reference = np.nanmean(direction_matrix, axis=0)

        panels.append(
            {
                "subplot": _gvs_display_label(gvs_code),
                "gvs_code": gvs_code,
                "subjects": group["subject"].reset_index(drop=True),
                "plot_matrix": centered_matrix,
                "raw_matrix": raw_matrix,
                "direction_matrix": direction_matrix,
                "direction_reference": direction_reference,
                "ylabel": "Subject-centered signal",
            }
        )
        rows.extend(
            _records_for_panel(
                figure="original_signal_subject_centered",
                subplot=_gvs_display_label(gvs_code),
                gvs_code=gvs_code,
                subjects=group["subject"],
                raw_matrix=raw_matrix,
                plot_matrix=centered_matrix,
                direction_matrix=direction_matrix,
                direction_reference=direction_reference,
                random_state=random_state,
            )
        )
    return panels, pd.DataFrame(rows)


def _plot_panels(
    panels: list[dict[str, object]],
    output_path: Path,
    *,
    title: str,
    random_state: int,
    show_zero: bool,
) -> None:
    time = np.asarray(TIME_INDICES)
    fig, axes = plt.subplots(
        len(panels),
        3,
        figsize=(15.5, max(2.1 * len(panels), 4.8)),
        sharex="col",
        constrained_layout=True,
    )
    if len(panels) == 1:
        axes = axes[None, :]

    for row_index, panel in enumerate(panels):
        values_ax, sd_ax, consistency_ax = axes[row_index]
        matrix = np.asarray(panel["plot_matrix"], dtype=np.float64)
        direction_matrix = np.asarray(panel["direction_matrix"], dtype=np.float64)
        direction_reference = np.asarray(panel["direction_reference"], dtype=np.float64)
        mean = np.nanmean(matrix, axis=0)
        sd = np.nanstd(matrix, axis=0, ddof=1)
        ci_low, ci_high = _bootstrap_mean_ci(matrix, random_state + row_index)

        values_ax.fill_between(time, ci_low, ci_high, color="0.65", alpha=0.22, lw=0)
        for subject_values in matrix:
            values_ax.plot(time, subject_values, color="#4C78A8", alpha=0.18, lw=0.9)
            values_ax.scatter(time, subject_values, color="#4C78A8", alpha=0.25, s=8, linewidths=0)
        values_ax.plot(time, mean, color="black", lw=2.1)
        if show_zero:
            values_ax.axhline(0.0, color="0.45", lw=0.9)
        values_ax.set_ylabel(str(panel["subplot"]), rotation=0, ha="right", va="center", labelpad=32, fontsize=10)
        values_ax.grid(color="0.9", lw=0.7)

        sd_ax.plot(time, sd, color="#D55E00", lw=1.8, marker="o", ms=3.5)
        sd_ax.set_ylim(bottom=0)
        sd_ax.grid(color="0.9", lw=0.7)

        proportions = []
        counts = []
        n_subjects = []
        for time_pos in range(len(TIME_INDICES)):
            vals = direction_matrix[:, time_pos]
            n = int(np.count_nonzero(np.isfinite(vals)))
            count, prop, _ = _same_direction(vals, direction_reference[time_pos])
            counts.append(count)
            n_subjects.append(n)
            proportions.append(prop * 100 if np.isfinite(prop) else np.nan)
        consistency_ax.bar(time, proportions, color="#009E73", alpha=0.78, width=0.72)
        consistency_ax.axhline(50.0, color="0.45", lw=0.9)
        consistency_ax.set_ylim(0, 105)
        consistency_ax.grid(axis="y", color="0.9", lw=0.7)
        for x_pos, prop, count, n in zip(time, proportions, counts, n_subjects):
            if np.isfinite(prop):
                consistency_ax.text(x_pos, min(prop + 3, 101), f"{count}/{n}", ha="center", va="bottom", fontsize=6)

        if row_index == 0:
            values_ax.set_title("Subject values + mean", fontsize=12)
            sd_ax.set_title("Between-subject SD", fontsize=12)
            consistency_ax.set_title("Same direction as group", fontsize=12)
        if row_index == len(panels) - 1:
            for ax in (values_ax, sd_ax, consistency_ax):
                ax.set_xlabel("Time index")

    fig.suptitle(title, fontsize=15)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _plot_original_style_with_sd(
    panels: list[dict[str, object]],
    output_path: Path,
    *,
    title: str,
    random_state: int,
    n_cols: int,
    sd_label: str,
    show_zero: bool,
) -> None:
    n_rows = int(np.ceil(len(panels) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.7 * n_cols, 4.7 * n_rows), constrained_layout=True)
    axes = np.asarray(axes).ravel()
    time = np.asarray(TIME_INDICES)

    for panel_index, (ax, panel) in enumerate(zip(axes, panels)):
        matrix = np.asarray(panel["plot_matrix"], dtype=np.float64)
        mean = np.nanmean(matrix, axis=0)
        sd = np.nanstd(matrix, axis=0, ddof=1)
        ci_low, ci_high = _bootstrap_mean_ci(matrix, random_state + panel_index)

        if show_zero:
            ax.axhline(0.0, color="0.35", lw=1)
        ax.fill_between(time, ci_low, ci_high, color="#7a7a7a", alpha=0.22, label="95% bootstrap CI")
        ax.errorbar(
            time,
            mean,
            yerr=sd,
            fmt="none",
            ecolor="#D55E00",
            elinewidth=1.05,
            capsize=3,
            capthick=1.05,
            alpha=0.8,
            label=sd_label,
            zorder=2,
        )
        ax.plot(time, mean, color="black", lw=2.5, label=str(panel["line_label"]), zorder=3)
        ax.set_title(str(panel["subplot"]), fontsize=11)
        ax.set_xlabel("Time index")
        ax.set_ylabel(str(panel["ylabel"]))
        ax.grid(color="0.9", lw=0.8)
        ax.legend(frameon=False, fontsize=8)

    for ax in axes[len(panels) :]:
        ax.set_axis_off()

    fig.suptitle(title, fontsize=15)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    pairs = pd.read_csv(args.subject_timecourse_pairs)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    delta_panels, delta_summary = _build_delta_panels(pairs, args.random_state)
    original_panels, original_summary = _build_original_panels(pairs, args.random_state)

    delta_summary.to_csv(args.out_dir / "gvs_minus_sham_timepoint_subject_consistency.csv", index=False)
    original_summary.to_csv(args.out_dir / "original_signal_timepoint_subject_consistency.csv", index=False)

    _plot_panels(
        delta_panels,
        args.out_dir / "subject_timepoint_consistency_gvs_02_to_09_vs_sham.png",
        title="GVS minus sham: subject-level timepoint consistency",
        random_state=args.random_state,
        show_zero=True,
    )
    _plot_panels(
        original_panels,
        args.out_dir / "subject_timepoint_consistency_original_signal_gvs_01_to_09.png",
        title=(
            "Original projection signal: subject-centered timepoint consistency "
            "(subject mean removed, grand mean restored)"
        ),
        random_state=args.random_state,
        show_zero=False,
    )

    for panel in delta_panels:
        panel["line_label"] = f"{panel['subplot']} minus sham"
    _plot_original_style_with_sd(
        delta_panels,
        args.out_dir / "second_subplots_gvs_02_to_09_vs_sham_with_subject_sd.png",
        title="GVS minus sham with between-subject SD at each time point",
        random_state=args.random_state,
        n_cols=4,
        sd_label="Between-subject SD",
        show_zero=True,
    )

    for panel in original_panels:
        panel["line_label"] = str(panel["subplot"])
        panel["ylabel"] = "Original projection signal"
    _plot_original_style_with_sd(
        original_panels,
        args.out_dir / "original_signal_gvs_01_to_09_3x3_with_subject_centered_sd.png",
        title="Original projection signal with subject-centered SD at each time point",
        random_state=args.random_state,
        n_cols=3,
        sd_label="Subject-centered SD",
        show_zero=False,
    )

    print(f"Wrote {args.out_dir / 'subject_timepoint_consistency_gvs_02_to_09_vs_sham.png'}")
    print(f"Wrote {args.out_dir / 'subject_timepoint_consistency_original_signal_gvs_01_to_09.png'}")
    print(f"Wrote {args.out_dir / 'second_subplots_gvs_02_to_09_vs_sham_with_subject_sd.png'}")
    print(f"Wrote {args.out_dir / 'original_signal_gvs_01_to_09_3x3_with_subject_centered_sd.png'}")
    print(f"Wrote {args.out_dir / 'gvs_minus_sham_timepoint_subject_consistency.csv'}")
    print(f"Wrote {args.out_dir / 'original_signal_timepoint_subject_consistency.csv'}")


if __name__ == "__main__":
    main()
