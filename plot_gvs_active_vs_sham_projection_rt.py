#!/usr/bin/env python3
"""Plot active-GVS versus sham effects on RT and vigour projection metrics."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib

warnings.filterwarnings("ignore", message="Unable to import Axes3D.*")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from gvs_behaviour_effects import _rt_seconds
from projected_sig_vs_RT import (
    DEFAULT_WEIGHT_MAP,
    PAPER_FONT_FAMILY,
    _load_behaviour_rt,
    _load_weights,
    _project_beta,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_INVENTORY = ROOT / "figures" / "GVS_effects" / "common" / "run_condition_inventory.csv"
DEFAULT_RUN_METRICS = (
    ROOT / "figures" / "projected_RT" / "revised_behaviour_from_mat" / "projection_behavior_run_metrics.csv"
)
DEFAULT_OUT_DIR = ROOT / "figures" / "GVS_effects" / "projected_RT"
SHAM_CODE = "gvs-01"
FIGURE_STEM = "gvs_active_vs_sham_projection_rt_four_panel"

METRICS = {
    "rt_ms": {
        "label": "RT",
        "ylabel": "Mean RT (ms)",
        "delta_label": "ms",
        "color": "#4C78A8",
    },
    "projection": {
        "label": "Vigour projection",
        "ylabel": "Mean projection",
        "delta_label": "a.u.",
        "color": "#D55E00",
    },
    "projection_variability": {
        "label": "Projection variability",
        "ylabel": "Mean normalized adjacent change",
        "delta_label": "a.u.",
        "color": "#009E73",
    },
    "projection_rt_coupling_z": {
        "label": "Projection-RT coupling",
        "ylabel": "Fisher z Pearson r",
        "delta_label": "z",
        "color": "#7A5195",
    },
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use the GVS trial inventory beta volumes and the revised behaviour inputs "
            "from the projected-RT panel to plot active-GVS versus sham effects."
        )
    )
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--run-metrics", type=Path, default=DEFAULT_RUN_METRICS)
    parser.add_argument("--weight-map", type=Path, default=DEFAULT_WEIGHT_MAP)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--behaviour-column", type=int, default=1, help="Zero-based RT column for 2D behaviour arrays.")
    parser.add_argument("--min-coupling-pairs", type=int, default=4)
    return parser.parse_args()


def _resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return ROOT / path


def _mean_normalized_adjacent_change(values: np.ndarray) -> tuple[float, int]:
    values = np.asarray(values, dtype=np.float64)
    if values.size < 2:
        return np.nan, 0
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.nan, 0

    centered = values.copy()
    centered[finite] -= np.nanmean(centered[finite])
    previous = centered[:-1]
    following = centered[1:]
    denominator = previous * previous + following * following
    keep = np.isfinite(previous) & np.isfinite(following) & np.isfinite(denominator) & (denominator > 0)
    if not np.any(keep):
        return np.nan, 0
    score = ((previous[keep] - following[keep]) ** 2) / denominator[keep]
    return float(np.mean(score)), int(np.count_nonzero(keep))


def _coupling_fisher_z(projection: np.ndarray, rt_ms: np.ndarray, min_pairs: int) -> tuple[float, float, int]:
    projection = np.asarray(projection, dtype=np.float64)
    rt_ms = np.asarray(rt_ms, dtype=np.float64)
    keep = np.isfinite(projection) & np.isfinite(rt_ms)
    n = int(np.count_nonzero(keep))
    if n < int(min_pairs):
        return np.nan, np.nan, n
    x = projection[keep]
    y = rt_ms[keep]
    if np.std(x, ddof=1) <= 0 or np.std(y, ddof=1) <= 0:
        return np.nan, np.nan, n
    r = float(np.corrcoef(x, y)[0, 1])
    r = float(np.clip(r, -0.999999, 0.999999))
    return float(np.arctanh(r)), r, n


def _load_inputs(inventory_path: Path, run_metrics_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    inventory = pd.read_csv(inventory_path)
    required_inventory = {
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
    missing = sorted(required_inventory - set(inventory.columns))
    if missing:
        raise RuntimeError(f"{inventory_path} is missing columns: {', '.join(missing)}")

    run_metrics = pd.read_csv(run_metrics_path)
    required_metrics = {"sub_tag", "ses", "run", "behaviour_path"}
    missing = sorted(required_metrics - set(run_metrics.columns))
    if missing:
        raise RuntimeError(f"{run_metrics_path} is missing columns: {', '.join(missing)}")

    run_metrics = run_metrics.rename(columns={"sub_tag": "subject", "ses": "session"})
    return inventory, run_metrics


def _run_projection_and_rt(
    row: pd.Series,
    beta_path: Path,
    weights: np.ndarray,
    *,
    behaviour_column: int,
) -> tuple[np.ndarray, np.ndarray, Path, Path]:
    projection_path = _resolve_path(beta_path)
    behaviour_path = _resolve_path(row["behaviour_path"])
    if not projection_path.exists():
        raise FileNotFoundError(projection_path)
    if not behaviour_path.exists():
        raise FileNotFoundError(behaviour_path)

    projection = _project_beta(projection_path, weights)
    behaviour_metric = _load_behaviour_rt(behaviour_path, behaviour_column)
    n_keep = min(projection.shape[0], behaviour_metric.shape[0])
    projection = projection[:n_keep]
    behaviour_metric = behaviour_metric[:n_keep]
    rt_s = _rt_seconds(behaviour_metric, input_is_inverse_rt=True)
    return projection, rt_s * 1000.0, projection_path, behaviour_path


def _build_trial_table(
    inventory: pd.DataFrame,
    run_metrics: pd.DataFrame,
    weights: np.ndarray,
    *,
    behaviour_column: int,
) -> pd.DataFrame:
    metric_lookup = {
        (str(row.subject), int(row.session), int(row.run)): row
        for row in run_metrics.itertuples(index=False)
    }
    beta_lookup = (
        inventory[["subject", "session", "run", "source_beta_path"]]
        .drop_duplicates()
        .set_index(["subject", "session", "run"])["source_beta_path"]
        .to_dict()
    )
    missing_runs = []
    rows: list[dict[str, object]] = []
    cache: dict[tuple[str, int, int], tuple[np.ndarray, np.ndarray, Path, Path]] = {}

    run_keys = inventory[["subject", "session", "run"]].drop_duplicates().sort_values(["subject", "session", "run"])
    for run_index, run_row in enumerate(run_keys.itertuples(index=False), start=1):
        key = (str(run_row.subject), int(run_row.session), int(run_row.run))
        metric_row = metric_lookup.get(key)
        if metric_row is None:
            missing_runs.append(key)
            continue
        beta_path = beta_lookup.get(key)
        if beta_path is None:
            missing_runs.append(key)
            continue
        if run_index == 1 or run_index % 10 == 0:
            print(f"Projecting run {run_index}/{len(run_keys)}: {key[0]} ses-{key[1]} run-{key[2]}", flush=True)
        cache[key] = _run_projection_and_rt(
            pd.Series(metric_row._asdict()),
            _resolve_path(beta_path),
            weights,
            behaviour_column=behaviour_column,
        )

    if missing_runs:
        raise RuntimeError(f"Run metrics are missing {len(missing_runs)} inventory runs.")

    for row in inventory.itertuples(index=False):
        key = (str(row.subject), int(row.session), int(row.run))
        projection, rt_ms, projection_path, behaviour_path = cache[key]
        start = int(row.trial_start)
        stop = min(int(row.trial_stop), projection.shape[0], rt_ms.shape[0])
        if start >= stop:
            continue
        for trial_index in range(start, stop):
            rows.append(
                {
                    "subject": str(row.subject),
                    "session": int(row.session),
                    "medication": str(row.medication),
                    "run": int(row.run),
                    "condition_code": str(row.condition_code),
                    "condition_label": str(row.condition_label),
                    "active_gvs": str(row.condition_code) != SHAM_CODE,
                    "trial_in_condition": int(trial_index - start + 1),
                    "task_trial_index_zero_based": int(trial_index),
                    "rt_ms": float(rt_ms[trial_index]) if np.isfinite(rt_ms[trial_index]) else np.nan,
                    "projection": float(projection[trial_index]) if np.isfinite(projection[trial_index]) else np.nan,
                    "projection_path": str(projection_path),
                    "behaviour_path": str(behaviour_path),
                }
            )

    if not rows:
        raise RuntimeError("No GVS projection/RT trial rows were built.")
    return pd.DataFrame(rows)


def _block_metrics(trials: pd.DataFrame, min_coupling_pairs: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_cols = [
        "subject",
        "session",
        "medication",
        "run",
        "condition_code",
        "condition_label",
        "active_gvs",
    ]
    for key, group in trials.sort_values("task_trial_index_zero_based").groupby(group_cols, sort=True):
        record = dict(zip(group_cols, key))
        projection = group["projection"].to_numpy(dtype=np.float64)
        rt_ms = group["rt_ms"].to_numpy(dtype=np.float64)
        projection_variability, n_projection_adjacent_pairs = _mean_normalized_adjacent_change(projection)
        coupling_z, coupling_r, n_coupling_pairs = _coupling_fisher_z(
            projection,
            rt_ms,
            min_pairs=min_coupling_pairs,
        )
        record.update(
            {
                "n_trials": int(group.shape[0]),
                "n_rt": int(np.count_nonzero(np.isfinite(rt_ms))),
                "n_projection": int(np.count_nonzero(np.isfinite(projection))),
                "rt_ms": float(np.nanmean(rt_ms)) if np.any(np.isfinite(rt_ms)) else np.nan,
                "projection": float(np.nanmean(projection)) if np.any(np.isfinite(projection)) else np.nan,
                "projection_variability": projection_variability,
                "n_projection_adjacent_pairs": n_projection_adjacent_pairs,
                "projection_rt_coupling_z": coupling_z,
                "projection_rt_coupling_r": coupling_r,
                "n_coupling_pairs": n_coupling_pairs,
            }
        )
        rows.append(record)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def _run_pairs(blocks: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    index_cols = ["subject", "session", "medication", "run"]
    metric_cols = list(METRICS)
    for key, group in blocks.groupby(index_cols, sort=True):
        sham = group.loc[group["condition_code"].eq(SHAM_CODE)]
        active = group.loc[~group["condition_code"].eq(SHAM_CODE)]
        if sham.empty or active.empty:
            continue
        record_base = dict(zip(index_cols, key))
        for metric in metric_cols:
            sham_value = float(sham[metric].mean(skipna=True))
            active_value = float(active[metric].mean(skipna=True))
            rows.append(
                {
                    **record_base,
                    "metric": metric,
                    "sham_value": sham_value,
                    "active_value": active_value,
                    "delta_active_minus_sham": active_value - sham_value,
                    "n_active_blocks": int(active[metric].notna().sum()),
                    "n_sham_blocks": int(sham[metric].notna().sum()),
                }
            )
    return pd.DataFrame(rows)


def _subject_pairs(run_pairs: pd.DataFrame) -> pd.DataFrame:
    subject = (
        run_pairs.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["sham_value", "active_value", "delta_active_minus_sham"])
        .groupby(["subject", "metric"], as_index=False)
        .agg(
            sham_value=("sham_value", "mean"),
            active_value=("active_value", "mean"),
            delta_active_minus_sham=("delta_active_minus_sham", "mean"),
            n_runs=("run", "size"),
        )
    )
    return subject.sort_values(["metric", "subject"]).reset_index(drop=True)


def _one_sample_stats(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    row: dict[str, float | int] = {
        "n_subjects": int(values.size),
        "mean_delta": float(np.mean(values)) if values.size else np.nan,
        "ci95_low": np.nan,
        "ci95_high": np.nan,
        "t_statistic": np.nan,
        "p_value": np.nan,
    }
    if values.size > 1:
        sem = float(stats.sem(values))
        ci_low, ci_high = stats.t.interval(0.95, values.size - 1, loc=float(np.mean(values)), scale=sem)
        t_result = stats.ttest_1samp(values, 0.0)
        row.update(
            {
                "ci95_low": float(ci_low),
                "ci95_high": float(ci_high),
                "t_statistic": float(t_result.statistic),
                "p_value": float(t_result.pvalue),
            }
        )
    return row


def _stats_table(subject_pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric, group in subject_pairs.groupby("metric", sort=False):
        row = {"metric": metric, **_one_sample_stats(group["delta_active_minus_sham"].to_numpy(dtype=np.float64))}
        rows.append(row)
    return pd.DataFrame(rows)


def _format_p(p_value: float) -> str:
    if not np.isfinite(p_value):
        return "p = n/a"
    if p_value < 0.001:
        return "p < 0.001"
    return f"p = {p_value:.3f}"


def _expanded_limits(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 1.0
    low = float(np.min(values))
    high = float(np.max(values))
    if high <= low:
        pad = 1.0 if high == 0 else abs(high) * 0.08
    else:
        pad = (high - low) * 0.10
    return low - pad, high + pad


def _draw_metric_panel(ax: plt.Axes, subject_pairs: pd.DataFrame, stats_df: pd.DataFrame, metric: str, label: str) -> None:
    spec = METRICS[metric]
    data = subject_pairs.loc[subject_pairs["metric"].eq(metric)].copy()
    stat = stats_df.loc[stats_df["metric"].eq(metric)].iloc[0]
    data = data.sort_values("delta_active_minus_sham").reset_index(drop=True)

    rng = np.random.default_rng(20240616)
    jitter = rng.uniform(-0.035, 0.035, size=data.shape[0])
    x_sham = 0.0 + jitter
    x_active = 1.0 + jitter
    sham = data["sham_value"].to_numpy(dtype=np.float64)
    active = data["active_value"].to_numpy(dtype=np.float64)
    color = str(spec["color"])

    for x0, x1, y0, y1 in zip(x_sham, x_active, sham, active):
        ax.plot([x0, x1], [y0, y1], color="0.66", linewidth=0.65, alpha=0.45, zorder=1)
    ax.scatter(x_sham, sham, s=24, color="#777777", edgecolor="white", linewidth=0.35, alpha=0.82, zorder=3)
    ax.scatter(x_active, active, s=24, color=color, edgecolor="white", linewidth=0.35, alpha=0.88, zorder=3)

    for x, values, point_color in [(0.0, sham, "#333333"), (1.0, active, color)]:
        finite = values[np.isfinite(values)]
        mean = float(np.mean(finite)) if finite.size else np.nan
        yerr = None
        if finite.size > 1:
            sem = float(stats.sem(finite))
            ci = stats.t.interval(0.95, finite.size - 1, loc=mean, scale=sem)
            yerr = np.array([[mean - ci[0]], [ci[1] - mean]])
        ax.errorbar(
            [x],
            [mean],
            yerr=yerr,
            fmt="D",
            markersize=5.2,
            markerfacecolor="white",
            markeredgecolor=point_color,
            markeredgewidth=1.1,
            ecolor=point_color,
            elinewidth=1.1,
            capsize=3.0,
            zorder=4,
        )

    delta = float(stat["mean_delta"])
    p_value = float(stat["p_value"])
    ax.set_title(
        f"{label}. {spec['label']}\nDelta = {delta:.3g} {spec['delta_label']}; {_format_p(p_value)}",
        fontsize=11,
        pad=8,
    )
    ax.set_xlim(-0.35, 1.35)
    ax.set_xticks([0.0, 1.0])
    ax.set_xticklabels(["Sham", "Active GVS"])
    ax.set_ylabel(str(spec["ylabel"]))
    ax.set_ylim(_expanded_limits(np.concatenate([sham, active])))
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)


def _save_figure(subject_pairs: pd.DataFrame, stats_df: pd.DataFrame, out_dir: Path) -> tuple[Path, Path]:
    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [PAPER_FONT_FAMILY, "Arial", "DejaVu Sans"],
            "font.size": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    ):
        fig, axes = plt.subplots(2, 2, figsize=(9.4, 7.0))
        for ax, metric, label in zip(axes.ravel(), METRICS, ["A", "B", "C", "D"]):
            _draw_metric_panel(ax, subject_pairs, stats_df, metric, label)
        fig.text(
            0.5,
            0.015,
            (
                "Each point is one subject. Active values average the eight active-GVS blocks within each run; "
                "deltas are paired to the same-run sham block and then averaged per subject."
            ),
            ha="center",
            va="bottom",
            fontsize=8.5,
        )
        fig.tight_layout(rect=(0.02, 0.045, 1.0, 1.0), w_pad=2.0, h_pad=2.2)
        out_dir.mkdir(parents=True, exist_ok=True)
        png = out_dir / f"{FIGURE_STEM}.png"
        pdf = out_dir / f"{FIGURE_STEM}.pdf"
        fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.04)
        fig.savefig(pdf, bbox_inches="tight", pad_inches=0.04)
        plt.close(fig)
    return png, pdf


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def main() -> None:
    args = _parse_args()
    inventory, run_metrics = _load_inputs(args.inventory, args.run_metrics)
    weights = _load_weights(args.weight_map)
    trial_table = _build_trial_table(
        inventory,
        run_metrics,
        weights,
        behaviour_column=int(args.behaviour_column),
    )
    blocks = _block_metrics(trial_table, min_coupling_pairs=int(args.min_coupling_pairs))
    run_pair_df = _run_pairs(blocks)
    subject_pair_df = _subject_pairs(run_pair_df)
    stats_df = _stats_table(subject_pair_df)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    trial_path = args.out_dir / "gvs_projection_rt_trial_table.csv"
    block_path = args.out_dir / "gvs_projection_rt_block_metrics.csv"
    run_pair_path = args.out_dir / "gvs_active_vs_sham_run_paired_metrics.csv"
    subject_pair_path = args.out_dir / "gvs_active_vs_sham_subject_paired_metrics.csv"
    stats_path = args.out_dir / "gvs_active_vs_sham_projection_rt_stats.csv"
    trial_table.to_csv(trial_path, index=False)
    blocks.to_csv(block_path, index=False)
    run_pair_df.to_csv(run_pair_path, index=False)
    subject_pair_df.to_csv(subject_pair_path, index=False)
    stats_df.to_csv(stats_path, index=False)
    png, pdf = _save_figure(subject_pair_df, stats_df, args.out_dir)

    summary = {
        "inventory": str(args.inventory),
        "run_metrics": str(args.run_metrics),
        "weight_map": str(args.weight_map),
        "projection_source": "beta",
        "behaviour_column_zero_based": int(args.behaviour_column),
        "min_coupling_pairs": int(args.min_coupling_pairs),
        "n_trials": int(trial_table.shape[0]),
        "n_blocks": int(blocks.shape[0]),
        "n_run_metric_rows": int(run_pair_df.shape[0]),
        "n_subject_metric_rows": int(subject_pair_df.shape[0]),
        "outputs": [str(path) for path in [trial_path, block_path, run_pair_path, subject_pair_path, stats_path, png, pdf]],
    }
    (args.out_dir / f"{FIGURE_STEM}_summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Saved trial table to {trial_path}")
    print(f"Saved block metrics to {block_path}")
    print(f"Saved run-paired metrics to {run_pair_path}")
    print(f"Saved subject-paired metrics to {subject_pair_path}")
    print(f"Saved stats to {stats_path}")
    print(f"Saved figure to {png}")
    print(f"Saved PDF to {pdf}")


if __name__ == "__main__":
    main()
