#!/usr/bin/env python3
"""Plot ablation model projected-beta RT coupling versus trial variability."""

from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path

import matplotlib

warnings.filterwarnings("ignore", message="Unable to import Axes3D.*")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd

from projected_sig_vs_RT import (
    AXIS_TICK_FONT_SIZE,
    DEFAULT_BEHAVIOUR_DIR,
    DEFAULT_DATA_DIR,
    PAPER_FONT_FAMILY,
    PROJECTION_TRIAL_CHUNK_SIZE,
    VARIABILITY_AXIS_LABEL,
    _adjacent_diff_ratio_sum,
    _align_trials,
    _behaviour_path,
    _category_sort_key,
    _discover_beta_runs,
    _load_behaviour_rt,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_MAIN_HTML = ROOT / "data" / "voxel_weights_task1_bold0.6_beta0.6_smooth1.25_gamma1.5_bold_thr90.html"
DEFAULT_ABLATION_DIR = ROOT / "data" / "ablation"
DEFAULT_ABLATION_SUMMARY = ROOT / "figures" / "ablation" / "ablation_metric_summary.csv"
DEFAULT_OUT_DIR = ROOT / "figures" / "projected_RT"
DEFAULT_FIGURE_STEM = "ablation_projected_beta_rt_corr_vs_variability"
NUMBER_PATTERN = r"[0-9]+(?:\.[0-9]+)?"
WEIGHT_RE = re.compile(
    rf"task(?P<task>{NUMBER_PATTERN})_bold(?P<bold>{NUMBER_PATTERN})_beta(?P<beta>{NUMBER_PATTERN})"
    rf"_smooth(?P<smooth>{NUMBER_PATTERN})_gamma(?P<gamma>{NUMBER_PATTERN})"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Project cleaned beta volumes through the main vigour-network and ablation maps, "
            "then plot RT coupling against consecutive-trial variability."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--behaviour-dir", type=Path, default=DEFAULT_BEHAVIOUR_DIR)
    parser.add_argument("--main-html", type=Path, default=DEFAULT_MAIN_HTML)
    parser.add_argument("--ablation-dir", type=Path, default=DEFAULT_ABLATION_DIR)
    parser.add_argument("--ablation-summary", type=Path, default=DEFAULT_ABLATION_SUMMARY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--figure-stem", default=DEFAULT_FIGURE_STEM)
    parser.add_argument(
        "--behaviour-column",
        type=int,
        default=1,
        help="Zero-based RT column for 2D behaviour arrays; default 1 uses the second column.",
    )
    parser.add_argument(
        "--main-percentile",
        type=float,
        default=90.0,
        help="Percentile threshold applied to the main unthresholded NIfTI map to match the *_bold_thr90 HTML.",
    )
    parser.add_argument("--max-runs", type=int, default=None, help="Optional debug cap on the number of beta runs.")
    parser.add_argument("--progress-every", type=int, default=5, help="Print progress every N beta runs.")
    return parser.parse_args()


def _html_to_weight_map(html_path: Path) -> Path:
    if html_path.suffix != ".html":
        return html_path
    candidate = html_path.with_suffix(".nii.gz")
    if candidate.exists():
        return candidate
    candidate = html_path.with_name(html_path.name.replace("_bold_thr90.html", ".nii.gz"))
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Could not find a NIfTI map paired with {html_path}")


def _candidate_id(path: Path) -> str:
    match = WEIGHT_RE.search(path.name)
    if not match:
        return path.stem
    return (
        f"task{match.group('task')}_bold{match.group('bold')}_beta{match.group('beta')}"
        f"_smooth{match.group('smooth')}_gamma{match.group('gamma')}"
    )


def _candidate_values(candidate_id: str) -> dict[str, float]:
    match = WEIGHT_RE.search(candidate_id)
    if not match:
        return {"task": np.nan, "bold": np.nan, "beta": np.nan, "smooth": np.nan, "gamma": np.nan}
    return {key: float(value) for key, value in match.groupdict().items()}


def _load_candidate_labels(summary_path: Path) -> dict[str, str]:
    if not summary_path.exists():
        return {}
    summary = pd.read_csv(summary_path)
    if not {"candidate_id", "candidate_label"}.issubset(summary.columns):
        return {}
    return dict(zip(summary["candidate_id"].astype(str), summary["candidate_label"].astype(str)))


def _discover_model_specs(main_html: Path, ablation_dir: Path, summary_path: Path) -> list[dict[str, object]]:
    label_by_id = _load_candidate_labels(summary_path)
    main_map = _html_to_weight_map(main_html)
    main_id = _candidate_id(main_map)
    specs: list[dict[str, object]] = [
        {
            "model_key": "main_vigour_network",
            "candidate_id": main_id,
            "label": "Vigour network",
            "weight_map": main_map,
            "source": "main",
            "threshold_main": True,
        }
    ]

    for html_path in sorted(ablation_dir.glob("*_bold_thr90.html")):
        weight_map = _html_to_weight_map(html_path)
        candidate_id = _candidate_id(weight_map)
        label = label_by_id.get(candidate_id, candidate_id)
        if label == "Full model":
            label = "Full model (ablation)"
        specs.append(
            {
                "model_key": f"ablation_{candidate_id}",
                "candidate_id": candidate_id,
                "label": label,
                "weight_map": weight_map,
                "source": "ablation",
                "threshold_main": False,
            }
        )

    if len(specs) == 1:
        raise FileNotFoundError(f"No ablation *_bold_thr90.html files found in {ablation_dir}")
    return specs


def _load_weights(weight_map: Path, threshold_percentile: float | None = None) -> np.ndarray:
    weights = nib.load(str(weight_map)).get_fdata(dtype=np.float32)
    mask = np.isfinite(weights) & (weights != 0)
    if not np.any(mask):
        raise ValueError(f"No nonzero finite weights found in {weight_map}")
    if threshold_percentile is not None:
        threshold = float(np.percentile(weights[mask], threshold_percentile))
        weights = np.where(mask & (weights >= threshold), weights, 0.0).astype(np.float32, copy=False)
        mask = np.isfinite(weights) & (weights != 0)
        if not np.any(mask):
            raise ValueError(f"No weights remained after p{threshold_percentile:g} thresholding {weight_map}")
    return weights


def _project_beta(beta_path: Path, weights: np.ndarray) -> np.ndarray:
    beta = np.load(beta_path, mmap_mode="r")
    if beta.ndim != 4:
        raise ValueError(f"Expected 4D beta volume in {beta_path}, got shape {beta.shape}")
    if beta.shape[:3] != weights.shape:
        raise ValueError(f"Spatial shape mismatch for {beta_path}: beta {beta.shape[:3]} vs weights {weights.shape}")

    weight_mask = np.isfinite(weights) & (weights != 0)
    selected_weights = weights[weight_mask].astype(np.float64)
    projected_signal = np.full(beta.shape[3], np.nan, dtype=np.float64)

    for start in range(0, beta.shape[3], PROJECTION_TRIAL_CHUNK_SIZE):
        stop = min(start + PROJECTION_TRIAL_CHUNK_SIZE, beta.shape[3])
        selected_beta = np.asarray(beta[weight_mask, start:stop], dtype=np.float64)
        finite_beta = np.isfinite(selected_beta)
        filled_beta = np.nan_to_num(selected_beta, nan=0.0, posinf=0.0, neginf=0.0)
        chunk_projection = selected_weights @ filled_beta
        chunk_projection[~np.any(finite_beta, axis=0)] = np.nan
        projected_signal[start:stop] = chunk_projection

    return projected_signal


def _pearsonr(x_values: np.ndarray, y_values: np.ndarray) -> tuple[float, int]:
    x_values = np.asarray(x_values, dtype=np.float64)
    y_values = np.asarray(y_values, dtype=np.float64)
    keep = np.isfinite(x_values) & np.isfinite(y_values)
    if np.count_nonzero(keep) < 3:
        return np.nan, int(np.count_nonzero(keep))
    x_keep = x_values[keep]
    y_keep = y_values[keep]
    if np.nanstd(x_keep) == 0 or np.nanstd(y_keep) == 0:
        return np.nan, int(np.count_nonzero(keep))
    return float(np.corrcoef(x_keep, y_keep)[0, 1]), int(np.count_nonzero(keep))


def _project_beta_all_models(
    beta_path: Path,
    union_mask: np.ndarray,
    weight_matrix: np.ndarray,
    active_weight_matrix: np.ndarray,
    spatial_shape: tuple[int, int, int],
) -> np.ndarray:
    beta = np.load(beta_path, mmap_mode="r")
    if beta.ndim != 4:
        raise ValueError(f"Expected 4D beta volume in {beta_path}, got shape {beta.shape}")
    if beta.shape[:3] != spatial_shape:
        raise ValueError(f"Spatial shape mismatch for {beta_path}: beta {beta.shape[:3]} vs weights {spatial_shape}")

    projected_signals = np.full((weight_matrix.shape[0], beta.shape[3]), np.nan, dtype=np.float64)
    for start in range(0, beta.shape[3], PROJECTION_TRIAL_CHUNK_SIZE):
        stop = min(start + PROJECTION_TRIAL_CHUNK_SIZE, beta.shape[3])
        selected_beta = np.asarray(beta[union_mask, start:stop], dtype=np.float64)
        finite_beta = np.isfinite(selected_beta)
        filled_beta = np.nan_to_num(selected_beta, nan=0.0, posinf=0.0, neginf=0.0)
        chunk_projection = weight_matrix @ filled_beta
        any_finite = active_weight_matrix @ finite_beta.astype(np.float64) > 0
        chunk_projection[~any_finite] = np.nan
        projected_signals[:, start:stop] = chunk_projection
    return projected_signals


def _build_model_tables(
    model_specs: list[dict[str, object]],
    data_dir: Path,
    behaviour_dir: Path,
    behaviour_column: int,
    main_percentile: float,
    max_runs: int | None = None,
    progress_every: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    beta_runs = _discover_beta_runs(data_dir)
    if max_runs is not None:
        beta_runs = beta_runs[:max_runs]
    missing_behaviour = []
    for run_info in beta_runs:
        behaviour_path = _behaviour_path(
            behaviour_dir,
            str(run_info["sub"]),
            int(run_info["ses"]),
            int(run_info["run"]),
        )
        if not behaviour_path.exists():
            missing_behaviour.append(str(behaviour_path))
    if missing_behaviour:
        missing_lines = "\n".join(f"- {path}" for path in missing_behaviour)
        raise FileNotFoundError(f"Missing behaviour files:\n{missing_lines}")

    loaded_weights = []
    weight_counts = []
    spatial_shape = None
    for spec in model_specs:
        threshold = main_percentile if bool(spec["threshold_main"]) else None
        weights = _load_weights(Path(spec["weight_map"]), threshold_percentile=threshold)
        if spatial_shape is None:
            spatial_shape = weights.shape
        elif weights.shape != spatial_shape:
            raise ValueError(f"Weight-map shape mismatch for {spec['weight_map']}: {weights.shape} vs {spatial_shape}")
        loaded_weights.append(weights)
        weight_counts.append(int(np.count_nonzero(np.isfinite(weights) & (weights != 0))))
    if spatial_shape is None:
        raise ValueError("No model weights were loaded.")

    union_mask = np.zeros(spatial_shape, dtype=bool)
    for weights in loaded_weights:
        union_mask |= np.isfinite(weights) & (weights != 0)
    if not np.any(union_mask):
        raise ValueError("No nonzero finite weights found across model maps.")

    weight_matrix = np.vstack([weights[union_mask].astype(np.float64) for weights in loaded_weights])
    active_weight_matrix = (weight_matrix != 0).astype(np.float64)

    run_rows = []
    subject_trial_values: dict[tuple[str, str], dict[str, list[np.ndarray]]] = {}

    for run_number, run_info in enumerate(beta_runs, start=1):
        sub = str(run_info["sub"])
        ses = int(run_info["ses"])
        run = int(run_info["run"])
        if progress_every > 0 and (run_number == 1 or run_number % progress_every == 0 or run_number == len(beta_runs)):
            print(f"Projecting beta run {run_number}/{len(beta_runs)}: {sub} ses-{ses} run-{run}", flush=True)
        beta_path = Path(run_info["path"])
        behaviour_path = _behaviour_path(behaviour_dir, sub, ses, run)
        behaviour_rt = _load_behaviour_rt(behaviour_path, behaviour_column)
        projected_by_model = _project_beta_all_models(
            beta_path=beta_path,
            union_mask=union_mask,
            weight_matrix=weight_matrix,
            active_weight_matrix=active_weight_matrix,
            spatial_shape=spatial_shape,
        )

        for model_index, spec in enumerate(model_specs):
            label = f"{spec['label']} {sub} ses-{ses} run-{run}"
            projected_signal, aligned_rt = _align_trials(projected_by_model[model_index], behaviour_rt, label)
            finite = np.isfinite(projected_signal) & np.isfinite(aligned_rt)
            projection_values = projected_signal[finite]
            behaviour_values = aligned_rt[finite]
            run_corr, n_corr_trials = _pearsonr(projection_values, behaviour_values)
            run_variability, n_adjacent_pairs = _adjacent_diff_ratio_sum(projection_values)

            run_rows.append(
                {
                    "model_key": spec["model_key"],
                    "candidate_id": spec["candidate_id"],
                    "model_label": spec["label"],
                    "source": spec["source"],
                    "weight_map": str(spec["weight_map"]),
                    "sub_tag": sub,
                    "ses": ses,
                    "run": run,
                    "n_trials_paired_finite": int(np.count_nonzero(finite)),
                    "corr_projected_signal_rt": run_corr,
                    "n_corr_trials": n_corr_trials,
                    "adjacent_diff_ratio_sum_projection": run_variability,
                    "n_adjacent_pairs_projection": n_adjacent_pairs,
                }
            )

            key = (str(spec["model_key"]), sub)
            if key not in subject_trial_values:
                subject_trial_values[key] = {"projection": [], "rt": []}
            if projection_values.size:
                subject_trial_values[key]["projection"].append(projection_values)
                subject_trial_values[key]["rt"].append(behaviour_values)

    run_table = pd.DataFrame(run_rows)
    subject_rows = []
    summary_rows = []
    for model_index, spec in enumerate(model_specs):
        model_key = str(spec["model_key"])
        model_run_df = run_table.loc[run_table["model_key"].eq(model_key)].copy()
        subjects = sorted(model_run_df["sub_tag"].unique(), key=_category_sort_key)
        for sub in subjects:
            sub_run_df = model_run_df.loc[model_run_df["sub_tag"].eq(sub)].copy()
            values = subject_trial_values.get((model_key, sub), {"projection": [], "rt": []})
            if values["projection"]:
                subject_projection = np.concatenate(values["projection"])
                subject_rt = np.concatenate(values["rt"])
            else:
                subject_projection = np.array([], dtype=np.float64)
                subject_rt = np.array([], dtype=np.float64)
            subject_corr, n_trials = _pearsonr(subject_projection, subject_rt)
            variability_values = sub_run_df["adjacent_diff_ratio_sum_projection"].to_numpy(dtype=np.float64)
            finite_variability = variability_values[np.isfinite(variability_values)]
            subject_variability = float(np.mean(finite_variability)) if finite_variability.size else np.nan
            subject_rows.append(
                {
                    "model_key": model_key,
                    "candidate_id": spec["candidate_id"],
                    "model_label": spec["label"],
                    "source": spec["source"],
                    "sub_tag": sub,
                    "n_trials_paired_finite": n_trials,
                    "n_runs": int(sub_run_df.shape[0]),
                    "corr_projected_signal_rt": subject_corr,
                    "adjacent_diff_ratio_sum_projection": subject_variability,
                }
            )

        subject_df = pd.DataFrame([row for row in subject_rows if row["model_key"] == model_key])
        finite_corr = subject_df["corr_projected_signal_rt"].to_numpy(dtype=np.float64)
        finite_corr = finite_corr[np.isfinite(finite_corr)]
        finite_var = subject_df["adjacent_diff_ratio_sum_projection"].to_numpy(dtype=np.float64)
        finite_var = finite_var[np.isfinite(finite_var)]
        values = _candidate_values(str(spec["candidate_id"]))
        summary_rows.append(
            {
                "model_key": model_key,
                "candidate_id": spec["candidate_id"],
                "model_label": spec["label"],
                "source": spec["source"],
                "weight_map": str(spec["weight_map"]),
                **values,
                "n_subjects_corr": int(finite_corr.size),
                "n_subjects_variability": int(finite_var.size),
                "mean_subject_corr_projected_signal_rt": float(np.mean(finite_corr)) if finite_corr.size else np.nan,
                "sem_subject_corr_projected_signal_rt": (
                    float(np.std(finite_corr, ddof=1) / np.sqrt(finite_corr.size)) if finite_corr.size > 1 else np.nan
                ),
                "mean_subject_projection_variability": float(np.mean(finite_var)) if finite_var.size else np.nan,
                "sem_subject_projection_variability": (
                    float(np.std(finite_var, ddof=1) / np.sqrt(finite_var.size)) if finite_var.size > 1 else np.nan
                ),
                "n_nonzero_weights": weight_counts[model_index],
            }
        )

    run_table = run_table.sort_values(["model_label", "sub_tag", "ses", "run"]).reset_index(drop=True)
    subject_table = pd.DataFrame(subject_rows)
    subject_table["_sort_key"] = subject_table["sub_tag"].astype(str).map(_category_sort_key)
    subject_table = (
        subject_table.sort_values(["model_label", "_sort_key"])
        .drop(columns="_sort_key")
        .reset_index(drop=True)
    )
    summary_table = pd.DataFrame(summary_rows)
    return summary_table, subject_table, run_table


def _expanded_limits(values: np.ndarray, pad_fraction: float = 0.11, include_zero: bool = False) -> tuple[float, float]:
    finite_values = np.asarray(values, dtype=np.float64)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return -1.0, 1.0
    low = float(np.min(finite_values))
    high = float(np.max(finite_values))
    if include_zero:
        low = min(low, 0.0)
        high = max(high, 0.0)
    if high <= low:
        pad = 1.0 if high == 0 else abs(high) * pad_fraction
    else:
        pad = (high - low) * pad_fraction
    return low - pad, high + pad


def _save_scatter(summary_table: pd.DataFrame, out_dir: Path, figure_stem: str) -> tuple[Path, Path]:
    x_col = "mean_subject_corr_projected_signal_rt"
    y_col = "mean_subject_projection_variability"
    plot_df = summary_table.loc[np.isfinite(summary_table[x_col]) & np.isfinite(summary_table[y_col])].copy()
    if plot_df.empty:
        raise ValueError("No finite model summaries available for plotting.")

    color_by_source = {"main": "#111111", "ablation": "#0072B2"}
    marker_by_source = {"main": "o", "ablation": "o"}
    size_by_source = {"main": 92, "ablation": 58}

    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [PAPER_FONT_FAMILY, "Arial", "DejaVu Sans"],
            "font.size": AXIS_TICK_FONT_SIZE,
            "axes.labelsize": AXIS_TICK_FONT_SIZE,
            "xtick.labelsize": AXIS_TICK_FONT_SIZE,
            "ytick.labelsize": AXIS_TICK_FONT_SIZE,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    ):
        fig, ax = plt.subplots(figsize=(8.1, 5.7))
        for _, row in plot_df.iterrows():
            source = str(row["source"])
            ax.scatter(
                row[x_col],
                row[y_col],
                s=size_by_source.get(source, 58),
                marker=marker_by_source.get(source, "o"),
                facecolor=color_by_source.get(source, "#0072B2"),
                edgecolor="white" if source != "main" else "#111111",
                linewidth=0.9,
                alpha=0.92,
                zorder=4 if source == "main" else 3,
            )

        label_offsets = {
            "Vigour network": (8, 13),
            "No task": (9, -24),
            "BOLD-only": (-16, 12),
            "Beta-only": (-14, -24),
            "No objective penalties": (10, 0),
            "Smooth-only": (-12, 0),
            "Full model (ablation)": (-18, -18),
            "No beta stability": (11, 16),
            "No BOLD stability": (-12, 18),
            "Task-only reference": (-10, -24),
        }
        for index, (_, row) in enumerate(plot_df.reset_index(drop=True).iterrows()):
            x_offset, y_offset = label_offsets.get(str(row["model_label"]), (7, 6 if index % 2 == 0 else -10))
            ha = "left" if x_offset >= 0 else "right"
            ax.annotate(
                str(row["model_label"]),
                (row[x_col], row[y_col]),
                xytext=(x_offset, y_offset),
                textcoords="offset points",
                ha=ha,
                va="center",
                fontsize=10.2,
                color="0.10",
                arrowprops={"arrowstyle": "-", "linewidth": 0.45, "color": "0.55", "shrinkA": 2, "shrinkB": 4},
            )

        ax.axvline(0.0, color="0.45", linestyle=(0, (4, 2)), linewidth=0.8, zorder=0)
        ax.set_xlabel("corr(projected beta signal, RT)")
        ax.set_ylabel(VARIABILITY_AXIS_LABEL)
        ax.set_xlim(_expanded_limits(plot_df[x_col].to_numpy(dtype=np.float64), include_zero=True))
        ax.set_ylim(_expanded_limits(plot_df[y_col].to_numpy(dtype=np.float64)))
        ax.grid(True, color="0.85", linewidth=0.55, alpha=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("0.25")
        ax.spines["bottom"].set_color("0.25")
        fig.tight_layout()

        out_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = out_dir / f"{figure_stem}.pdf"
        png_path = out_dir / f"{figure_stem}.png"
        fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.03)
        fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.03)
        plt.close(fig)
    return pdf_path, png_path


def main() -> None:
    args = _parse_args()
    model_specs = _discover_model_specs(args.main_html, args.ablation_dir, args.ablation_summary)
    summary_table, subject_table, run_table = _build_model_tables(
        model_specs=model_specs,
        data_dir=args.data_dir,
        behaviour_dir=args.behaviour_dir,
        behaviour_column=args.behaviour_column,
        main_percentile=args.main_percentile,
        max_runs=args.max_runs,
        progress_every=args.progress_every,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / f"{args.figure_stem}_summary.csv"
    subject_path = args.out_dir / f"{args.figure_stem}_subject_metrics.csv"
    run_path = args.out_dir / f"{args.figure_stem}_run_metrics.csv"
    summary_table.to_csv(summary_path, index=False)
    subject_table.to_csv(subject_path, index=False)
    run_table.to_csv(run_path, index=False)
    pdf_path, png_path = _save_scatter(summary_table, args.out_dir, args.figure_stem)

    print(f"Saved model summary to {summary_path}")
    print(f"Saved subject metrics to {subject_path}")
    print(f"Saved run metrics to {run_path}")
    print(f"Saved figure PNG to {png_path}")
    print(f"Saved figure PDF to {pdf_path}")


if __name__ == "__main__":
    main()
