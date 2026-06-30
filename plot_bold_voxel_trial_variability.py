#!/usr/bin/env python3
"""Plot per-voxel consecutive-trial BOLD variability histograms."""

from __future__ import annotations

import argparse
import csv
import re
import warnings
from pathlib import Path

import matplotlib

warnings.filterwarnings("ignore", message="Unable to import Axes3D.*")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance


ROOT = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = ROOT / "figures" / "ablation" / "BOLD_var"
DEFAULT_MANIFEST = ROOT / "data" / "concat_manifest_group.tsv"
REFERENCE_LABEL = "standard_glm_zgte3.1"
LOW_VARIABILITY_THRESHOLD = 0.01
SELECTED_COMPARISON_STEM = "voxel_trial_variability_comparison_blue_orange_purple"
SELECTED_COMPARISON = (
    {
        "label": REFERENCE_LABEL,
        "legend_label": "task activation network",
        "table_label": "Task activation",
        "color": "#ff7f0e",
    },
    {
        "label": "task0_bold0_beta0_smooth0_gamma1.5",
        "legend_label": "model without low-variability constraint",
        "table_label": "Without variability penalty",
        "color": "#9467bd",
    },
    {
        "label": "task1_bold0.6_beta0.6_smooth1.25_gamma1.5",
        "legend_label": "motor vigour network",
        "table_label": "Final vigour network",
        "color": "#1f77b4",
    },
)
DEFAULT_INPUTS = (
    ROOT / "data" / "z_valu_standard_glm_zgte3.1_active_bold.npy",
    ROOT
    / "data"
    / "ablation"
    / "voxel_weights_mean_foldavg_sub9_ses1_task0_bold0_beta0_smooth0_gamma1.5_bold_thr90_active_bold.npy",
    ROOT
    / "data"
    / "ablation"
    / "voxel_weights_mean_foldavg_sub9_ses1_task1_bold0_beta0.6_smooth1.25_gamma1.5_bold_thr90_active_bold.npy",
    ROOT
    / "data"
    / "ablation"
    / "voxel_weights_mean_foldavg_sub9_ses1_task1_bold0.6_beta0_smooth1.25_gamma1.5_bold_thr90_active_bold.npy",
    ROOT
    / "data"
    / "ablation"
    / "voxel_weights_mean_foldavg_sub9_ses1_task1_bold0.6_beta0.6_smooth1.25_gamma1.5_bold_thr90_active_bold.npy",
)
MODEL_RE = re.compile(
    r"task(?P<task>[0-9.]+)_bold(?P<bold>[0-9.]+)_beta(?P<beta>[0-9.]+)"
    r"_smooth(?P<smooth>[0-9.]+)_gamma(?P<gamma>[0-9.]+)"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute each voxel's consecutive-trial variability, "
            "(X_i - X_i+1)^2 / (X_i^2 + X_i+1^2), for each time point, "
            "average over valid adjacent trial pairs and time points, and plot histograms."
        )
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Group concatenation manifest used for subject-level sign-flip summaries.",
    )
    parser.add_argument("--chunk-size", type=int, default=256, help="Number of voxels to process at once.")
    parser.add_argument("--bins", type=int, default=80, help="Histogram bin count.")
    parser.add_argument(
        "inputs",
        type=Path,
        nargs="*",
        default=list(DEFAULT_INPUTS),
        help="Optional replacement input .npy files. Defaults to the five requested files.",
    )
    return parser.parse_args()


def _short_label(path: Path) -> str:
    if path.name.startswith("z_valu_standard_glm"):
        return "standard_glm_zgte3.1"
    match = MODEL_RE.search(path.name)
    if match:
        return (
            f"task{match.group('task')}_bold{match.group('bold')}_beta{match.group('beta')}"
            f"_smooth{match.group('smooth')}_gamma{match.group('gamma')}"
        )
    return path.stem.replace("_active_bold", "")


def _safe_stem(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_")


def _display_label(label: str) -> str:
    if label == REFERENCE_LABEL:
        return "standard GLM z>=3.1"
    match = MODEL_RE.search(label)
    if match:
        return (
            f"task {match.group('task')}, BOLD {match.group('bold')}, "
            f"beta {match.group('beta')}, smooth {match.group('smooth')}"
        )
    return label


def _voxel_trial_variability(path: Path, chunk_size: int) -> np.ndarray:
    data = np.load(path, mmap_mode="r")
    if data.ndim != 3:
        raise ValueError(f"Expected a 3D array in {path}, got shape {data.shape}")
    if data.shape[1] < 2:
        raise ValueError(f"Need at least two trials in {path}, got shape {data.shape}")

    n_voxels, _, n_timepoints = data.shape
    variability = np.full(n_voxels, np.nan, dtype=np.float64)

    for start in range(0, n_voxels, chunk_size):
        stop = min(start + chunk_size, n_voxels)
        timepoint_sum = np.zeros(stop - start, dtype=np.float64)
        timepoint_count = np.zeros(stop - start, dtype=np.int64)

        for timepoint in range(n_timepoints):
            values = np.asarray(data[start:stop, :, timepoint], dtype=np.float64)
            x0 = values[:, :-1]
            x1 = values[:, 1:]
            denominator = x0 * x0 + x1 * x1
            valid = np.isfinite(x0) & np.isfinite(x1) & np.isfinite(denominator) & (denominator > 0)

            ratio = np.zeros_like(denominator, dtype=np.float64)
            np.divide((x0 - x1) ** 2, denominator, out=ratio, where=valid)
            pair_count = np.count_nonzero(valid, axis=1)
            timepoint_mean = np.full(stop - start, np.nan, dtype=np.float64)
            np.divide(ratio.sum(axis=1), pair_count, out=timepoint_mean, where=pair_count > 0)

            has_timepoint = pair_count > 0
            timepoint_sum[has_timepoint] += timepoint_mean[has_timepoint]
            timepoint_count[has_timepoint] += 1

        np.divide(
            timepoint_sum,
            timepoint_count,
            out=variability[start:stop],
            where=timepoint_count > 0,
        )

    return variability


def _summarize(path: Path, label: str, variability: np.ndarray) -> dict[str, float | int | str]:
    finite = variability[np.isfinite(variability)]
    summary: dict[str, float | int | str] = {
        "input_file": str(path),
        "label": label,
        "n_voxels": int(variability.size),
        "n_finite_voxels": int(finite.size),
    }
    if finite.size == 0:
        for key in ("min", "p01", "p05", "p25", "median", "mean", "std", "p75", "p95", "p99", "max"):
            summary[key] = np.nan
        return summary

    percentiles = np.percentile(finite, [1, 5, 25, 50, 75, 95, 99])
    summary.update(
        {
            "min": float(np.min(finite)),
            "p01": float(percentiles[0]),
            "p05": float(percentiles[1]),
            "p25": float(percentiles[2]),
            "median": float(percentiles[3]),
            "mean": float(np.mean(finite)),
            "std": float(np.std(finite)),
            "p75": float(percentiles[4]),
            "p95": float(percentiles[5]),
            "p99": float(percentiles[6]),
            "max": float(np.max(finite)),
        }
    )
    return summary


def _plot_histogram(variability: np.ndarray, label: str, out_base: Path, bins: int) -> None:
    finite = variability[np.isfinite(variability)]
    if finite.size == 0:
        raise ValueError(f"No finite variability values available for {label}")

    png_path = out_base.parent / f"{out_base.name}.png"
    pdf_path = out_base.parent / f"{out_base.name}.pdf"

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.hist(finite, bins=bins, density=True, color="#2563eb", alpha=0.78, edgecolor="white", linewidth=0.3)
    ax.axvline(float(np.median(finite)), color="#dc2626", lw=1.8, label=f"Median = {np.median(finite):.4g}")
    ax.set_title(label)
    ax.set_xlabel("Voxel consecutive-trial variability")
    ax.set_ylabel("Probability density")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(png_path, dpi=200)
    fig.savefig(pdf_path)
    plt.close(fig)


def _plot_comparison(
    variability_by_label: list[tuple[str, np.ndarray]],
    out_dir: Path,
    bins: int,
) -> None:
    finite_by_label = [
        (label, variability[np.isfinite(variability)])
        for label, variability in variability_by_label
    ]
    finite_by_label = [(label, finite) for label, finite in finite_by_label if finite.size]
    if not finite_by_label:
        raise ValueError("No finite variability values available for comparison")

    all_values = np.concatenate([finite for _, finite in finite_by_label])
    x_max = float(np.max(all_values))
    if not np.isfinite(x_max) or x_max <= 0:
        raise ValueError("Need positive finite variability values for comparison")

    bin_edges = np.linspace(0.0, x_max, bins + 1)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, (ax_hist, ax_ecdf) = plt.subplots(1, 2, figsize=(13.2, 5.2), sharex=True)
    for index, (label, finite) in enumerate(finite_by_label):
        color = colors[index % len(colors)]
        weights = np.full(finite.size, 1.0 / finite.size, dtype=np.float64)
        ax_hist.hist(
            finite,
            bins=bin_edges,
            weights=weights,
            histtype="step",
            linewidth=1.8,
            color=color,
            label=_display_label(label),
        )

        sorted_values = np.sort(finite)
        cumulative = np.arange(1, sorted_values.size + 1, dtype=np.float64) / sorted_values.size
        ax_ecdf.step(sorted_values, cumulative, where="post", linewidth=1.8, color=color, label=_display_label(label))

    ax_hist.set_title("Shared-bin histogram")
    ax_hist.set_xlabel("Voxel consecutive-trial variability")
    ax_hist.set_ylabel("Proportion of voxels per bin")

    ax_ecdf.set_title("ECDF")
    ax_ecdf.set_xlabel("Voxel consecutive-trial variability")
    ax_ecdf.set_ylabel("Cumulative proportion of voxels")
    ax_ecdf.set_ylim(0.0, 1.0)

    for ax in (ax_hist, ax_ecdf):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(alpha=0.22, linewidth=0.5)

    handles, labels = ax_ecdf.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(out_dir / "voxel_trial_variability_comparison.png", dpi=200)
    fig.savefig(out_dir / "voxel_trial_variability_comparison.pdf")
    plt.close(fig)


def _selected_finite_by_label(
    variability_by_label: list[tuple[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    finite_by_label = {
        label: variability[np.isfinite(variability)]
        for label, variability in variability_by_label
        if np.isfinite(variability).any()
    }
    missing = [spec["label"] for spec in SELECTED_COMPARISON if spec["label"] not in finite_by_label]
    if missing:
        raise ValueError(f"Missing selected comparison labels: {', '.join(missing)}")
    return finite_by_label


def _selected_summary_rows(
    variability_by_label: list[tuple[str, np.ndarray]],
) -> list[dict[str, float | int | str]]:
    finite_by_label = _selected_finite_by_label(variability_by_label)
    rows: list[dict[str, float | int | str]] = []
    for spec in SELECTED_COMPARISON:
        finite = finite_by_label[str(spec["label"])]
        p25, median, p75 = np.percentile(finite, [25, 50, 75])
        rows.append(
            {
                "network": spec["table_label"],
                "label": spec["label"],
                "n_finite_voxels": int(finite.size),
                "median_variability": float(median),
                "iqr": float(p75 - p25),
                "p25": float(p25),
                "p75": float(p75),
                "pct_voxels_below_0p01": float(100.0 * np.mean(finite < LOW_VARIABILITY_THRESHOLD)),
            }
        )
    return rows


def _write_selected_summary_table(
    variability_by_label: list[tuple[str, np.ndarray]],
    out_dir: Path,
) -> tuple[Path, list[dict[str, float | int | str]]]:
    rows = _selected_summary_rows(variability_by_label)
    table_path = out_dir / f"{SELECTED_COMPARISON_STEM}_summary_table.csv"
    fieldnames = [
        "network",
        "label",
        "n_finite_voxels",
        "median_variability",
        "iqr",
        "p25",
        "p75",
        "pct_voxels_below_0p01",
    ]
    with table_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return table_path, rows


def _format_table_float(value: float) -> str:
    return f"{value:.4f}"


def _format_table_pct(value: float) -> str:
    return f"{value:.1f}%"


def _latex_escape(value: object) -> str:
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(char, char) for char in str(value))


def _format_latex_p(value: float) -> str:
    if not np.isfinite(value):
        return r"\mathrm{NA}"
    if value < 0.001 and value > 0:
        exponent = int(np.floor(np.log10(value)))
        mantissa = value / (10.0**exponent)
        return rf"{mantissa:.3g}\times 10^{{{exponent}}}"
    return f"{value:.3f}"


def _write_selected_summary_latex(
    summary_rows: list[dict[str, float | int | str]],
    signflip_rows: list[dict[str, float | int | str]],
    out_dir: Path,
) -> Path:
    latex_path = out_dir / f"{SELECTED_COMPARISON_STEM}_summary_table.tex"
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Voxel consecutive-trial variability summary.}",
        r"\label{tab:voxel-trial-variability}",
        r"\begin{tabular}{lrrr}",
        r"\hline",
        r"Network & Median variability & IQR & \% voxels $< 0.01$ \\",
        r"\hline",
    ]
    for row in summary_rows:
        lines.append(
            " & ".join(
                [
                    _latex_escape(row["network"]),
                    _format_table_float(float(row["median_variability"])),
                    _format_table_float(float(row["iqr"])),
                    _latex_escape(_format_table_pct(float(row["pct_voxels_below_0p01"]))),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\hline", r"\end{tabular}"])

    if signflip_rows:
        lines.extend([r"\vspace{0.4em}", r"\begin{minipage}{0.96\linewidth}", r"\footnotesize"])
        for row in signflip_rows:
            lines.append(
                (
                    f"{_latex_escape(row['comparison_network'])} $<$ "
                    f"{_latex_escape(row['reference_network'])}: "
                    f"one-sided $p={_format_latex_p(float(row['p_signflip_less']))}$, "
                    f"two-sided $p={_format_latex_p(float(row['p_signflip_two_sided']))}$."
                )
                + r" \\"
            )
        lines.append(r"\end{minipage}")

    lines.append(r"\end{table}")
    with latex_path.open("w") as f:
        f.write("\n".join(lines) + "\n")
    return latex_path


def _load_subject_segments(manifest_path: Path, expected_trials: int) -> dict[str, list[tuple[int, int]]]:
    with manifest_path.open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        required = {"sub_tag", "offset_start", "offset_end"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{manifest_path} is missing required columns: {', '.join(sorted(missing))}")

        rows = sorted(reader, key=lambda row: int(row["offset_start"]))

    subject_segments: dict[str, list[tuple[int, int]]] = {}
    for row in rows:
        start = int(row["offset_start"])
        stop = int(row["offset_end"])
        if start < 0 or stop > expected_trials or stop <= start:
            raise ValueError(
                f"Invalid manifest offsets for {row['sub_tag']}: {start}-{stop} "
                f"with expected trial count {expected_trials}"
            )
        subject_segments.setdefault(str(row["sub_tag"]), []).append((start, stop))

    if not subject_segments:
        raise ValueError(f"No subject rows found in {manifest_path}")
    return subject_segments


def _subject_median_voxel_variability(
    path: Path,
    subject_segments: dict[str, list[tuple[int, int]]],
    chunk_size: int,
) -> dict[str, float]:
    data = np.load(path, mmap_mode="r")
    if data.ndim != 3:
        raise ValueError(f"Expected a 3D array in {path}, got shape {data.shape}")
    n_voxels, _, n_timepoints = data.shape
    subject_medians: dict[str, float] = {}

    for subject, segments in subject_segments.items():
        variability = np.full(n_voxels, np.nan, dtype=np.float64)
        for start_voxel in range(0, n_voxels, chunk_size):
            stop_voxel = min(start_voxel + chunk_size, n_voxels)
            timepoint_sum = np.zeros((stop_voxel - start_voxel, n_timepoints), dtype=np.float64)
            timepoint_count = np.zeros((stop_voxel - start_voxel, n_timepoints), dtype=np.int64)

            for start_trial, stop_trial in segments:
                if stop_trial - start_trial < 2:
                    continue
                values = np.asarray(data[start_voxel:stop_voxel, start_trial:stop_trial, :], dtype=np.float64)
                x0 = values[:, :-1, :]
                x1 = values[:, 1:, :]
                denominator = x0 * x0 + x1 * x1
                valid = np.isfinite(x0) & np.isfinite(x1) & np.isfinite(denominator) & (denominator > 0)
                ratio = np.zeros_like(denominator, dtype=np.float64)
                np.divide((x0 - x1) ** 2, denominator, out=ratio, where=valid)
                timepoint_sum += ratio.sum(axis=1)
                timepoint_count += np.count_nonzero(valid, axis=1)

            timepoint_mean = np.full_like(timepoint_sum, np.nan, dtype=np.float64)
            np.divide(timepoint_sum, timepoint_count, out=timepoint_mean, where=timepoint_count > 0)
            finite_timepoints = np.isfinite(timepoint_mean)
            valid_timepoint_count = np.count_nonzero(finite_timepoints, axis=1)
            chunk_variability = np.full(stop_voxel - start_voxel, np.nan, dtype=np.float64)
            np.divide(
                np.nansum(timepoint_mean, axis=1),
                valid_timepoint_count,
                out=chunk_variability,
                where=valid_timepoint_count > 0,
            )
            variability[start_voxel:stop_voxel] = chunk_variability

        finite = variability[np.isfinite(variability)]
        if finite.size == 0:
            subject_medians[subject] = np.nan
        else:
            subject_medians[subject] = float(np.median(finite))

    return subject_medians


def _paired_sign_flip(diff: np.ndarray) -> dict[str, float | int]:
    finite = diff[np.isfinite(diff)]
    if finite.size == 0:
        return {
            "n_subjects": 0,
            "mean_delta": np.nan,
            "median_delta": np.nan,
            "p_less": np.nan,
            "p_two_sided": np.nan,
        }

    observed = float(np.mean(finite))
    n_subjects = int(finite.size)
    if n_subjects <= 20:
        signs = np.ones((1 << n_subjects, n_subjects), dtype=np.float64)
        indices = np.arange(1 << n_subjects, dtype=np.uint64)
        for index in range(n_subjects):
            signs[((indices >> index) & 1).astype(bool), index] = -1.0
        null = signs @ finite / n_subjects
    else:
        rng = np.random.default_rng(0)
        signs = rng.choice([-1.0, 1.0], size=(200_000, n_subjects))
        null = signs @ finite / n_subjects

    return {
        "n_subjects": n_subjects,
        "mean_delta": observed,
        "median_delta": float(np.median(finite)),
        "p_less": float(np.mean(null <= observed)),
        "p_two_sided": float(np.mean(np.abs(null) >= abs(observed))),
    }


def _write_selected_subject_tables(
    input_paths_by_label: dict[str, Path],
    out_dir: Path,
    manifest_path: Path,
    chunk_size: int,
) -> tuple[Path | None, Path | None, list[dict[str, float | int | str]]]:
    missing_inputs = [str(spec["label"]) for spec in SELECTED_COMPARISON if str(spec["label"]) not in input_paths_by_label]
    if missing_inputs or not manifest_path.exists():
        return None, None, []

    first_path = input_paths_by_label[str(SELECTED_COMPARISON[0]["label"])]
    expected_trials = int(np.load(first_path, mmap_mode="r").shape[1])
    subject_segments = _load_subject_segments(manifest_path, expected_trials)

    subject_rows: list[dict[str, float | str]] = []
    medians_by_label: dict[str, dict[str, float]] = {}
    for spec in SELECTED_COMPARISON:
        label = str(spec["label"])
        medians = _subject_median_voxel_variability(input_paths_by_label[label], subject_segments, chunk_size)
        medians_by_label[label] = medians
        for subject, median in medians.items():
            subject_rows.append(
                {
                    "subject": subject,
                    "network": spec["table_label"],
                    "label": label,
                    "subject_median_variability": median,
                }
            )

    subject_path = out_dir / f"{SELECTED_COMPARISON_STEM}_subject_median_variability.csv"
    with subject_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["subject", "network", "label", "subject_median_variability"],
        )
        writer.writeheader()
        writer.writerows(subject_rows)

    stats_rows: list[dict[str, float | int | str]] = []
    for reference_index, reference_spec in enumerate(SELECTED_COMPARISON[:-1]):
        reference_label = str(reference_spec["label"])
        reference_subjects = medians_by_label[reference_label]
        for comparison_spec in SELECTED_COMPARISON[reference_index + 1:]:
            comparison_label = str(comparison_spec["label"])
            comparison_subjects = medians_by_label[comparison_label]
            paired_subjects = sorted(set(reference_subjects) & set(comparison_subjects))
            reference_values = np.asarray([reference_subjects[subject] for subject in paired_subjects], dtype=np.float64)
            comparison_values = np.asarray([comparison_subjects[subject] for subject in paired_subjects], dtype=np.float64)
            stats = _paired_sign_flip(comparison_values - reference_values)
            stats_rows.append(
                {
                    "reference_network": reference_spec["table_label"],
                    "reference_label": reference_label,
                    "comparison_network": comparison_spec["table_label"],
                    "comparison_label": comparison_label,
                    "tested_direction": f"{comparison_spec['table_label']} < {reference_spec['table_label']}",
                    "n_subjects": stats["n_subjects"],
                    "mean_delta_subject_median_variability": stats["mean_delta"],
                    "median_delta_subject_median_variability": stats["median_delta"],
                    "p_signflip_less": stats["p_less"],
                    "p_signflip_two_sided": stats["p_two_sided"],
                }
            )

    stats_path = out_dir / f"{SELECTED_COMPARISON_STEM}_subject_signflip_stats.csv"
    with stats_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(stats_rows[0].keys()))
        writer.writeheader()
        writer.writerows(stats_rows)

    return subject_path, stats_path, stats_rows


def _plot_selected_comparison(
    variability_by_label: list[tuple[str, np.ndarray]],
    out_dir: Path,
    bins: int,
) -> None:
    finite_by_label = _selected_finite_by_label(variability_by_label)
    all_values = np.concatenate([finite_by_label[str(spec["label"])] for spec in SELECTED_COMPARISON])
    x_max = float(np.max(all_values))
    if not np.isfinite(x_max) or x_max <= 0:
        raise ValueError("Need positive finite variability values for selected comparison")

    bin_edges = np.linspace(0.0, x_max, bins + 1)

    fig, (ax_hist, ax_ecdf) = plt.subplots(1, 2, figsize=(13.2, 4.9), sharex=True)

    for spec in SELECTED_COMPARISON:
        finite = finite_by_label[str(spec["label"])]
        weights = np.full(finite.size, 1.0 / finite.size, dtype=np.float64)
        ax_hist.hist(
            finite,
            bins=bin_edges,
            weights=weights,
            histtype="step",
            linewidth=1.9,
            color=str(spec["color"]),
            label=str(spec["legend_label"]),
        )

        sorted_values = np.sort(finite)
        cumulative = np.arange(1, sorted_values.size + 1, dtype=np.float64) / sorted_values.size
        ax_ecdf.step(
            sorted_values,
            cumulative,
            where="post",
            linewidth=1.9,
            color=str(spec["color"]),
            label=str(spec["legend_label"]),
        )

    ax_hist.set_xlabel("Voxel variability", fontsize=15, fontweight="bold")
    ax_hist.set_ylabel("Proportion of voxels", fontsize=15, fontweight="bold")

    ax_ecdf.set_xlabel("Voxel variability", fontsize=15, fontweight="bold")
    ax_ecdf.set_ylabel("Cumulative proportion", fontsize=15, fontweight="bold")
    ax_ecdf.set_ylim(0.0, 1.0)

    for ax in (ax_hist, ax_ecdf):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(alpha=0.22, linewidth=0.5)
        ax.tick_params(axis="both", labelsize=13)

    handles, labels = ax_ecdf.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.005),
        frameon=False,
        prop={"size": 13, "weight": "bold"},
    )

    fig.tight_layout(rect=(0, 0.15, 1, 1), pad=0.25, w_pad=2.0)
    fig.savefig(out_dir / f"{SELECTED_COMPARISON_STEM}.png", dpi=200, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(out_dir / f"{SELECTED_COMPARISON_STEM}.pdf", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def _write_pairwise_comparison(
    variability_by_label: list[tuple[str, np.ndarray]],
    out_dir: Path,
) -> Path:
    finite_by_label = {
        label: variability[np.isfinite(variability)]
        for label, variability in variability_by_label
        if np.isfinite(variability).any()
    }
    if not finite_by_label:
        raise ValueError("No finite variability values available for pairwise comparison")

    reference_label = REFERENCE_LABEL if REFERENCE_LABEL in finite_by_label else next(iter(finite_by_label))
    reference = finite_by_label[reference_label]
    rows = []
    for label, finite in finite_by_label.items():
        ks = ks_2samp(finite, reference, alternative="two-sided", method="auto")
        rows.append(
            {
                "reference_label": reference_label,
                "label": label,
                "n_finite_voxels": int(finite.size),
                "median": float(np.median(finite)),
                "mean": float(np.mean(finite)),
                "p75": float(np.percentile(finite, 75)),
                "p95": float(np.percentile(finite, 95)),
                "delta_median_vs_reference": float(np.median(finite) - np.median(reference)),
                "wasserstein_vs_reference": float(wasserstein_distance(finite, reference)),
                "ks_stat_vs_reference": float(ks.statistic),
                "ks_p_vs_reference": float(ks.pvalue),
            }
        )

    comparison_path = out_dir / "voxel_trial_variability_pairwise_comparison.csv"
    with comparison_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return comparison_path


def main() -> None:
    args = _parse_args()
    if args.chunk_size < 1:
        raise ValueError("--chunk-size must be at least 1")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    variability_by_label = []
    input_paths_by_label: dict[str, Path] = {}

    for input_path in args.inputs:
        input_path = input_path.expanduser().resolve()
        label = _short_label(input_path)
        input_paths_by_label[label] = input_path
        stem = _safe_stem(label)
        out_base = args.out_dir / f"{stem}_voxel_trial_variability_hist"

        print(f"Processing {input_path}")
        variability = _voxel_trial_variability(input_path, args.chunk_size)
        np.save(args.out_dir / f"{stem}_voxel_trial_variability.npy", variability)
        _plot_histogram(variability, label, out_base, args.bins)
        summaries.append(_summarize(input_path, label, variability))
        variability_by_label.append((label, variability))
        print(f"  saved {out_base.parent / f'{out_base.name}.png'}")
        print(f"  saved {out_base.parent / f'{out_base.name}.pdf'}")

    summary_path = args.out_dir / "voxel_trial_variability_summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)
    print(f"Saved {summary_path}")

    _plot_comparison(variability_by_label, args.out_dir, args.bins)
    comparison_path = _write_pairwise_comparison(variability_by_label, args.out_dir)
    selected_table_path, selected_summary_rows = _write_selected_summary_table(variability_by_label, args.out_dir)
    subject_table_path, signflip_path, signflip_rows = _write_selected_subject_tables(
        input_paths_by_label,
        args.out_dir,
        args.manifest.expanduser().resolve(),
        args.chunk_size,
    )
    selected_latex_path = _write_selected_summary_latex(selected_summary_rows, signflip_rows, args.out_dir)
    _plot_selected_comparison(variability_by_label, args.out_dir, args.bins)
    print(f"Saved {args.out_dir / 'voxel_trial_variability_comparison.png'}")
    print(f"Saved {args.out_dir / 'voxel_trial_variability_comparison.pdf'}")
    print(f"Saved {comparison_path}")
    print(f"Saved {args.out_dir / f'{SELECTED_COMPARISON_STEM}.png'}")
    print(f"Saved {args.out_dir / f'{SELECTED_COMPARISON_STEM}.pdf'}")
    print(f"Saved {selected_table_path}")
    print(f"Saved {selected_latex_path}")
    if subject_table_path is not None:
        print(f"Saved {subject_table_path}")
    if signflip_path is not None:
        print(f"Saved {signflip_path}")


if __name__ == "__main__":
    main()
