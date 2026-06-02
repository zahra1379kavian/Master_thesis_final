#!/usr/bin/env python3
"""Matched non-vigour voxel null distribution for medication effects.

Each randomization samples one non-vigour voxel set per ROI, matched to the
final vigour-network voxel count in that ROI, then reruns the same medication
effect summaries used by med_effects.py. The output is one compact row per
randomization plus histograms showing where the actual vigour-network effect
falls relative to the matched same-ROI null distribution.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib

warnings.filterwarnings("ignore", message="Unable to import Axes3D.*")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd

import med_effects as M


DEFAULT_OUT_DIR = Path("figures/med_effects/baselines/matched_nonvigour_null_distribution")


def _default_region_table(value: Path | None, roi_figure: Path) -> Path:
    return M._default_region_table_for(roi_figure) if value is None else value


def _roi_mean_timeseries_from_voxels(voxel_timeseries: dict[str, np.ndarray], rois) -> pd.DataFrame:
    columns = {}
    for roi in rois:
        values = np.asarray(voxel_timeseries[roi.name], dtype=np.float64)
        weights = np.asarray(roi.weights, dtype=np.float64)
        finite = np.isfinite(values)
        weighted = np.where(finite, values, 0.0) * weights[None, :]
        denom = finite.astype(np.float64) @ weights
        series = np.full(values.shape[0], np.nan, dtype=np.float64)
        valid = denom > 0
        series[valid] = np.sum(weighted[valid, :], axis=1) / denom[valid]
        columns[roi.name] = series
    return pd.DataFrame(columns)


def _read_intra_actual(main_dir: Path) -> dict[str, float]:
    table = pd.read_csv(main_dir / "intra_vs_between_fc_results.csv")
    rows = {str(row.analysis): row for row in table.itertuples(index=False)}
    primary = rows["within_minus_between_delta"]
    within = rows["within_roi_on_minus_off"]
    between = rows["between_roi_on_minus_off"]
    return {
        "intra_primary_effect": float(primary.mean),
        "intra_primary_p": float(primary.paired_t_p_value_two_sided),
        "intra_within_effect": float(within.mean),
        "intra_between_effect": float(between.mean),
    }


def _read_spectral_actual(main_dir: Path) -> dict[str, float]:
    stats = json.loads((main_dir / "paired_subject_similarity_stats.json").read_text())
    return {
        "spectral_primary_effect": float(stats["observed"]["contrasts"]["off_minus_on"]),
        "spectral_primary_p": float(stats["permutation"]["contrasts"]["off_minus_on"]["p_value_two_sided"]),
        "spectral_subject_p": float(
            stats["subject_level"]["contrasts"]["off_minus_on"]["paired_t_p_value_two_sided"]
        ),
    }


def _load_actual_values(main_dir: Path) -> dict[str, float]:
    actual = {}
    actual.update(_read_intra_actual(main_dir))
    actual.update(_read_spectral_actual(main_dir))
    return actual


def _single_randomization(
    seed: int,
    args: argparse.Namespace,
    reference_img,
    weight_values: np.ndarray,
    groups,
    weighted_rois,
    roi_threshold: float,
    specs,
) -> dict[str, float | int]:
    rois = M._build_matched_nonvigour_rois(
        weight_values=weight_values,
        weighted_rois=weighted_rois,
        groups=groups,
        roi_threshold=roi_threshold,
        random_state=int(seed),
    )

    networks = {}
    session_rows = []
    for spec in specs:
        voxel_timeseries = M._load_session_voxel_timeseries(spec, reference_img, rois)
        roi_timeseries = _roi_mean_timeseries_from_voxels(voxel_timeseries, rois)
        cleaned = M._clean_timeseries(roi_timeseries)

        networks[spec.label] = M._connectivity_matrix(
            cleaned,
            n_neighbors=args.mi_neighbors,
            random_state=int(seed),
        )

        session_row = {
            "label": spec.label,
            "subject": spec.subject,
            "session": spec.session,
            "state": spec.state,
            "connectivity_metric": M.INTRA_BETWEEN_FC_METRIC,
        }
        session_row.update(M._between_roi_fc_summary(cleaned))
        _, intra_summary = M._intra_roi_fc_values(spec, voxel_timeseries, rois)
        session_row.update(intra_summary)
        session_rows.append(session_row)

    subject_deltas = M._complete_intra_between_subject_deltas(pd.DataFrame(session_rows))
    _, intra_results = M._intra_between_fc_test_rows(subject_deltas)

    pairwise = M._pairwise_network_distances(specs, networks)
    paired_stats, _ = M._paired_similarity_tests(pairwise)

    within = intra_results["within_roi_on_minus_off"]
    between = intra_results["between_roi_on_minus_off"]
    primary = intra_results["within_minus_between_delta"]
    spectral_subject = paired_stats["subject_level"]["contrasts"]["off_minus_on"]
    spectral_perm = paired_stats["permutation"]["contrasts"]["off_minus_on"]

    return {
        "seed": int(seed),
        "n_rois": int(len(rois)),
        "n_voxels_total": int(sum(roi.n_voxels for roi in rois)),
        "n_complete_subjects": int(subject_deltas.shape[0]),
        "intra_within_effect": float(within["mean"]),
        "intra_within_p": float(within["paired_t_p_value_two_sided"]),
        "intra_between_effect": float(between["mean"]),
        "intra_between_p": float(between["paired_t_p_value_two_sided"]),
        "intra_primary_effect": float(primary["mean"]),
        "intra_primary_p": float(primary["paired_t_p_value_two_sided"]),
        "intra_primary_dz": float(primary["cohen_dz"]),
        "spectral_primary_effect": float(paired_stats["observed"]["contrasts"]["off_minus_on"]),
        "spectral_primary_p": float(spectral_perm["p_value_two_sided"]),
        "spectral_primary_p_greater": float(spectral_perm["p_value_greater"]),
        "spectral_subject_p": float(spectral_subject["paired_t_p_value_two_sided"]),
        "spectral_subject_dz": float(spectral_subject["cohen_dz"]),
        "spectral_percent_on_lower_than_off": float(
            paired_stats["effect_size"]["percent_on_on_lower_than_off_off"]
        ),
    }


def _finite_values(values: pd.Series) -> np.ndarray:
    out = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    return out[np.isfinite(out)]


def _empirical_summary(null_values: np.ndarray, actual_value: float) -> dict[str, float | int]:
    null_values = np.asarray(null_values, dtype=np.float64)
    null_values = null_values[np.isfinite(null_values)]
    if null_values.size == 0:
        return {"n": 0}
    count_ge = int(np.count_nonzero(null_values >= actual_value))
    count_le = int(np.count_nonzero(null_values <= actual_value))
    return {
        "n": int(null_values.size),
        "actual": float(actual_value),
        "null_mean": float(np.mean(null_values)),
        "null_sd": float(np.std(null_values, ddof=1)) if null_values.size > 1 else None,
        "null_median": float(np.median(null_values)),
        "null_ci95_low": float(np.percentile(null_values, 2.5)),
        "null_ci95_high": float(np.percentile(null_values, 97.5)),
        "null_min": float(np.min(null_values)),
        "null_max": float(np.max(null_values)),
        "n_null_ge_actual": count_ge,
        "n_null_le_actual": count_le,
        "empirical_p_greater": float((count_ge + 1) / (null_values.size + 1)),
        "empirical_p_less": float((count_le + 1) / (null_values.size + 1)),
        "actual_percentile": float(100.0 * np.mean(null_values <= actual_value)),
    }


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    return value


def _plot_histogram(
    values: np.ndarray,
    actual_value: float,
    summary: dict[str, float | int],
    out_path: Path,
    xlabel: str,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 4.1))
    ax.hist(values, bins="auto", color="#6f8fb7", edgecolor="#ffffff", linewidth=0.7, alpha=0.9)
    ax.axvline(float(summary["null_mean"]), color="#333333", linestyle="--", linewidth=1.4, label="Null mean")
    ax.axvline(actual_value, color="#d62728", linewidth=2.2, label="Actual vigour network")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Random matched non-vigour draws")
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    label = (
        f"actual percentile = {summary['actual_percentile']:.1f}%\n"
        f"empirical p(greater) = {summary['empirical_p_greater']:.4f}"
    )
    ax.text(
        0.98,
        0.95,
        label,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#dddddd", "alpha": 0.9},
    )
    ax.legend(frameon=False, loc="upper left")
    M._apply_paper_typography(fig, [ax])
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"pdf.fonttype": 42, "ps.fonttype": 42}):
        fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.04)
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def _write_summary_outputs(values_path: Path, out_dir: Path, actual: dict[str, float], args: argparse.Namespace) -> None:
    table = pd.read_csv(values_path)
    summaries = {
        "intra_vs_between_fc_primary_effect": _empirical_summary(
            _finite_values(table["intra_primary_effect"]),
            actual["intra_primary_effect"],
        ),
        "cross_subject_spectral_primary_effect": _empirical_summary(
            _finite_values(table["spectral_primary_effect"]),
            actual["spectral_primary_effect"],
        ),
    }
    summary_payload = {
        "method": "Repeated matched non-vigour same-ROI voxel randomization; one full medication analysis per seed.",
        "n_requested_randomizations": int(args.n_randomizations),
        "start_seed": int(args.start_seed),
        "main_dir": str(args.main_dir),
        "values_csv": str(values_path),
        "actual_values": actual,
        "summaries": summaries,
    }
    (out_dir / "matched_nonvigour_null_distribution_summary.json").write_text(
        json.dumps(_json_safe(summary_payload), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    method = (
        "# Matched Non-Vigour Null Distribution\n\n"
        "For each randomization seed, non-vigour voxels were sampled without replacement from the same "
        "lateralized AAL ROIs as the final vigour network. The number of sampled voxels was matched to "
        "the final vigour-network voxel count separately within each ROI. The medication analyses were "
        "then rerun once for that sampled network, and only the seed-level effect estimates were stored. "
        "P-values are not averaged across seeds; inference is based on the empirical null distribution "
        "of effect sizes and the location of the actual vigour-network effect within that distribution.\n"
    )
    (out_dir / "matched_nonvigour_null_distribution_method.md").write_text(method, encoding="utf-8")

    _plot_histogram(
        _finite_values(table["intra_primary_effect"]),
        actual["intra_primary_effect"],
        summaries["intra_vs_between_fc_primary_effect"],
        out_dir / "intra_vs_between_fc_matched_nonvigour_null_histogram.png",
        "(ON - OFF intra-ROI FC) - (ON - OFF between-ROI FC)",
        "Intra-vs-Between FC Null Distribution",
    )
    _plot_histogram(
        _finite_values(table["spectral_primary_effect"]),
        actual["spectral_primary_effect"],
        summaries["cross_subject_spectral_primary_effect"],
        out_dir / "cross_subject_spectral_matched_nonvigour_null_histogram.png",
        "OFF-OFF minus ON-ON spectral distance",
        "Cross-Subject Spectral-Distance Null Distribution",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-randomizations", type=int, default=1000)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--main-dir", type=Path, default=M.DEFAULT_OUT_DIR)
    parser.add_argument("--weight-map", type=Path, default=M.DEFAULT_WEIGHT_MAP)
    parser.add_argument("--roi-definition-figure", type=Path, default=M.DEFAULT_ROI_FIGURE)
    parser.add_argument("--roi-region-table", type=Path, default=None)
    parser.add_argument("--roi-percentile", type=float, default=M.REFERENCE_THRESHOLD)
    parser.add_argument("--min-report-voxels", type=int, default=M.DEFAULT_MIN_REPORT_VOXELS)
    parser.add_argument("--session-manifest", type=Path, default=M.DEFAULT_SESSION_MANIFEST)
    parser.add_argument("--beta-root", type=Path, default=M.DEFAULT_BETA_ROOT)
    parser.add_argument("--subjects", nargs="+", default=None)
    parser.add_argument("--complete-subjects-only", action="store_true")
    parser.add_argument("--split-hemispheres", dest="split_hemispheres", action="store_true", default=True)
    parser.add_argument("--no-split-hemispheres", dest="split_hemispheres", action="store_false")
    parser.add_argument("--exclude-rois", nargs="*", default=())
    parser.add_argument("--min-lateralized-voxels", type=int, default=1)
    parser.add_argument("--aal-version", default=M.DEFAULT_AAL_VERSION)
    parser.add_argument("--atlas-cache-dir", type=Path, default=M.DEFAULT_ATLAS_CACHE_DIR)
    parser.add_argument("--connectivity-metric", choices=M.CONNECTIVITY_METRICS, default=M.CONNECTIVITY_METRIC)
    parser.add_argument("--mi-neighbors", type=int, default=M.DEFAULT_MI_NEIGHBORS)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.roi_region_table = _default_region_table(args.roi_region_table, args.roi_definition_figure)
    M.CONNECTIVITY_METRIC = args.connectivity_metric

    if args.n_randomizations < 1:
        raise RuntimeError("--n-randomizations must be at least 1")
    if args.checkpoint_every < 1:
        raise RuntimeError("--checkpoint-every must be at least 1")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    values_path = out_dir / "matched_nonvigour_null_distribution_values.csv"
    actual = _load_actual_values(args.main_dir)

    existing = pd.DataFrame()
    completed_seeds: set[int] = set()
    if args.resume and values_path.exists():
        existing = pd.read_csv(values_path)
        completed_seeds = set(pd.to_numeric(existing["seed"], errors="coerce").dropna().astype(int).tolist())

    reference_img = nib.load(str(args.weight_map))
    weight_values = np.asarray(reference_img.get_fdata(), dtype=np.float64)
    groups, _, roi_names, min_roi_voxels = M._analysis_roi_setup(args, reference_img)
    weighted_rois, roi_threshold = M._build_weighted_rois(
        weight_values=weight_values,
        roi_names=roi_names,
        groups=groups,
        roi_percentile=args.roi_percentile,
        min_report_voxels=args.min_report_voxels,
        min_roi_voxels=min_roi_voxels,
    )
    specs = M._load_session_specs(args)

    rows = existing.to_dict("records") if not existing.empty else []
    seeds = range(int(args.start_seed), int(args.start_seed) + int(args.n_randomizations))
    for index, seed in enumerate(seeds, start=1):
        if seed in completed_seeds:
            continue
        row = _single_randomization(
            seed=seed,
            args=args,
            reference_img=reference_img,
            weight_values=weight_values,
            groups=groups,
            weighted_rois=weighted_rois,
            roi_threshold=roi_threshold,
            specs=specs,
        )
        rows.append(row)
        if index % args.checkpoint_every == 0:
            pd.DataFrame(rows).sort_values("seed").to_csv(values_path, index=False)
            print(f"Saved checkpoint after seed {seed}: {values_path}", flush=True)

    pd.DataFrame(rows).sort_values("seed").to_csv(values_path, index=False)
    _write_summary_outputs(values_path, out_dir, actual, args)
    print(f"Saved {values_path}")
    print(f"Saved {out_dir / 'matched_nonvigour_null_distribution_summary.json'}")
    print(f"Saved {out_dir / 'intra_vs_between_fc_matched_nonvigour_null_histogram.png'}")
    print(f"Saved {out_dir / 'cross_subject_spectral_matched_nonvigour_null_histogram.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
