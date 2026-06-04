#!/usr/bin/env python3
"""Rerun edge-connectogram sensitivity using task-activation ROI voxels."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import med_effects as M
import med_effects_task_activation as task


DEFAULT_OUT_DIR = ROOT / "figures" / "GVS_effects" / "08_connectivity_coactivation" / "task_activation_z3p1"
DEFAULT_RUN_INVENTORY = ROOT / "figures" / "GVS_effects" / "common" / "run_condition_inventory.csv"
DEFAULT_METRICS = ("mutual_info_quantile", "spearman_rho")


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return str(value)


def validate_metric_names(metric_names: tuple[str, ...], sensitivity) -> list[tuple[str, str, Any]]:
    metric_lookup = {metric: (metric, family, fn) for metric, family, fn in sensitivity.METRICS}
    missing = [metric for metric in metric_names if metric not in metric_lookup]
    if missing:
        available = ", ".join(sorted(metric_lookup))
        raise RuntimeError(f"Unknown metric(s): {', '.join(missing)}. Available metrics: {available}")
    return [metric_lookup[metric] for metric in metric_names]


def write_roi_definition(rois, path: Path, z_threshold: float) -> pd.DataFrame:
    rows = []
    for roi in rois:
        weights = np.asarray(roi.weights, dtype=np.float64)
        rows.append(
            {
                "roi_label": roi.name,
                "n_selected_voxels": int(roi.n_voxels),
                "roi_weight_sum": float(np.sum(weights)),
                "roi_weight_min": float(np.min(weights)) if weights.size else np.nan,
                "roi_weight_max": float(np.max(weights)) if weights.size else np.nan,
                "roi_weight_mean": float(np.mean(weights)) if weights.size else np.nan,
                "voxel_selection": task.TASK_VOXEL_SELECTION,
                "voxel_weighting": "unit_weights",
                "task_z_threshold": float(z_threshold),
            }
        )
    roi_def = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    roi_def.to_csv(path, index=False)
    return roi_def


def build_trial_table(args: argparse.Namespace, rois, reference_img, out_path: Path) -> pd.DataFrame:
    inventory = pd.read_csv(args.run_inventory)
    required = {
        "subject",
        "session",
        "medication",
        "run",
        "condition_code",
        "condition_label",
        "trial_start",
        "trial_stop",
        "source_beta_path",
    }
    missing = sorted(required - set(inventory.columns))
    if missing:
        raise RuntimeError(f"{args.run_inventory} is missing required columns: {', '.join(missing)}")

    roi_names = [roi.name for roi in rois]
    frames = []
    grouped = list(inventory.groupby("source_beta_path", sort=False))
    for run_index, (source_beta_path, run_rows) in enumerate(grouped, start=1):
        beta_path = Path(str(source_beta_path))
        if not beta_path.exists():
            raise RuntimeError(f"Missing beta file listed in run inventory: {beta_path}")
        print(f"Extracting task-activation ROI betas {run_index}/{len(grouped)}: {beta_path.name}", flush=True)
        run_ts = M._extract_roi_timeseries_from_beta(beta_path, reference_img, rois)
        run_rows = run_rows.sort_values(["trial_start", "condition_code"]).reset_index(drop=True)
        for row in run_rows.itertuples(index=False):
            start = int(row.trial_start)
            stop = int(row.trial_stop)
            if start < 0 or stop > run_ts.shape[0] or stop <= start:
                raise RuntimeError(
                    f"Invalid trial slice {start}:{stop} for {beta_path}; beta run has {run_ts.shape[0]} trials"
                )
            block = run_ts.iloc[start:stop].reset_index(drop=True).copy()
            block.insert(0, "trial_in_condition", np.arange(block.shape[0], dtype=int))
            block.insert(0, "condition_label", str(row.condition_label).replace("\n", " "))
            block.insert(0, "condition_code", row.condition_code)
            block.insert(0, "run", int(row.run))
            block.insert(0, "medication", str(row.medication).upper())
            block.insert(0, "session", int(row.session))
            block.insert(0, "subject", row.subject)
            frames.append(block[["subject", "session", "medication", "run", "condition_code", "condition_label", "trial_in_condition", *roi_names]])

    trial = pd.concat(frames, ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    trial.to_csv(out_path, index=False)
    return trial


def run_metric_sensitivity(args: argparse.Namespace, trial_table: Path, roi_def: Path, sensitivity) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_dir = args.out_dir / "metric_sensitivity"
    metric_dir.mkdir(parents=True, exist_ok=True)

    sensitivity.OUT = metric_dir
    sensitivity.base.TRIAL_TABLE = trial_table
    sensitivity.base.ROI_DEF = roi_def
    sensitivity.base.OUT_DIR = args.out_dir
    sensitivity.METRICS = validate_metric_names(tuple(args.metrics), sensitivity)

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    rng = np.random.default_rng(sensitivity.base.RNG_SEED)
    data = sensitivity.base.load_and_prepare()
    edge_index = sensitivity.base.roi_edge_index(data.roi_cols)

    all_stats = []
    for metric, metric_family, metric_fn in sensitivity.METRICS:
        print(f"Running {metric}", flush=True)
        edge_delta = sensitivity.block_edge_deltas(data, metric, metric_fn, edge_index)
        all_stats.append(sensitivity.metric_stats(edge_delta, metric, metric_family, rng))

    stats_df = pd.concat(all_stats, ignore_index=True)
    stats_df["abs_mean"] = stats_df["mean"].abs()
    stats_df["sig_uncorrected"] = stats_df["p_signflip"].lt(sensitivity.base.ALPHA)
    stats_df = sensitivity.base.add_groupwise_fdr(stats_df, "p_signflip", ["metric", "analysis_view", "fdr_scope"])
    front = [
        "metric",
        "metric_family",
        "analysis_view",
        "fdr_scope",
        "edge_id",
        "roi_i",
        "roi_j",
        "edge_label",
        "label",
        "n",
        "mean",
        "t_stat",
        "p_signflip",
        "q_fdr",
        "sig_fdr",
    ]
    stats_df = stats_df[front + [col for col in stats_df.columns if col not in front]]
    stats_df.to_csv(metric_dir / "edge_connectivity_metric_sensitivity_stats.csv", index=False)
    sig_df = stats_df.loc[stats_df["sig_fdr"]].copy()
    sig_df.to_csv(metric_dir / "fdr_significant_edge_connectivity_metric_sensitivity.csv", index=False)

    summary = sensitivity.summarize_stats(stats_df)
    summary.to_csv(metric_dir / "edge_connectivity_metric_sensitivity_summary.csv", index=False)
    top = (
        stats_df.sort_values(["sig_fdr", "q_fdr", "p_signflip", "abs_mean"], ascending=[False, True, True, False])
        .groupby(["metric", "analysis_view", "fdr_scope"], dropna=False)
        .head(10)
        .reset_index(drop=True)
    )
    top.to_csv(metric_dir / "top_edge_connectivity_metric_sensitivity.csv", index=False)
    return stats_df, sig_df


def run_connectogram_report(args: argparse.Namespace, roi_def: Path, plot_module) -> pd.DataFrame:
    metric_dir = args.out_dir / "metric_sensitivity"
    sig_csv = metric_dir / "fdr_significant_edge_connectivity_metric_sensitivity.csv"
    plot_module.ROI_DEF = roi_def
    plot_module.SIG_CSV = sig_csv
    plot_module.OUT_DIR = metric_dir / "connectogram_reports"
    plot_module.main()
    return pd.read_csv(plot_module.OUT_DIR / "fdr_connectogram_summary.csv")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-activation-map", type=Path, default=task.DEFAULT_TASK_ACTIVATION_MAP)
    parser.add_argument("--task-z-threshold", type=float, default=task.DEFAULT_TASK_Z_THRESHOLD)
    parser.add_argument("--run-inventory", type=Path, default=DEFAULT_RUN_INVENTORY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--metrics", nargs="+", default=list(DEFAULT_METRICS))
    parser.add_argument("--split-hemispheres", dest="split_hemispheres", action="store_true", default=True)
    parser.add_argument("--no-split-hemispheres", dest="split_hemispheres", action="store_false")
    parser.add_argument("--exclude-rois", nargs="*", default=())
    parser.add_argument("--min-report-voxels", type=int, default=M.DEFAULT_MIN_REPORT_VOXELS)
    parser.add_argument("--min-lateralized-voxels", type=int, default=task.DEFAULT_MIN_LATERALIZED_VOXELS)
    parser.add_argument("--aal-version", default=M.DEFAULT_AAL_VERSION)
    parser.add_argument("--atlas-cache-dir", type=Path, default=M.DEFAULT_ATLAS_CACHE_DIR)
    parser.add_argument("--reuse-trial-table", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    task_img, task_values, metadata, rois, min_roi_voxels = task._prepare_task_rois(args)
    selected_task = np.isfinite(task_values) & (task_values >= float(args.task_z_threshold))

    trial_table = args.out_dir / "per_trial_task_activation_roi_betas.csv"
    roi_def = args.out_dir / "task_activation_roi_definition.csv"
    write_roi_definition(rois, roi_def, args.task_z_threshold)
    if args.reuse_trial_table and trial_table.exists():
        trial = pd.read_csv(trial_table)
    else:
        trial = build_trial_table(args, rois, task_img, trial_table)

    sensitivity = load_module(HERE / "run_edge_connectivity_metric_sensitivity.py", "task_activation_metric_sensitivity")
    plot_module = load_module(HERE / "plot_fdr_significant_edge_connectograms.py", "task_activation_connectogram_plot")

    stats_df, sig_df = run_metric_sensitivity(args, trial_table, roi_def, sensitivity)
    connectogram_summary = run_connectogram_report(args, roi_def, plot_module) if not sig_df.empty else pd.DataFrame()

    metadata.update(
        {
            "task_activation_map": str(args.task_activation_map),
            "task_activation_threshold": f"z >= {float(args.task_z_threshold):g}",
            "task_activation_voxels": int(np.count_nonzero(selected_task)),
            "task_activation_roi_definition": str(roi_def),
            "trial_table": str(trial_table),
            "run_inventory": str(args.run_inventory),
            "n_trial_rows": int(trial.shape[0]),
            "n_subjects": int(trial["subject"].nunique()),
            "n_rois": int(len(rois)),
            "roi_cols": [roi.name for roi in rois],
            "min_roi_voxels": int(min_roi_voxels),
            "metrics": list(args.metrics),
            "n_metric_rows": int(stats_df.shape[0]),
            "n_fdr_significant_rows": int(sig_df.shape[0]),
            "connectogram_report_rows": int(connectogram_summary.shape[0]),
        }
    )
    (args.out_dir / "task_activation_connectogram_manifest.json").write_text(
        json.dumps(metadata, indent=2, default=json_default),
        encoding="utf-8",
    )
    print(f"Saved task-activation connectogram outputs under {args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
