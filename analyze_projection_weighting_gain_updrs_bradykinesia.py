#!/usr/bin/env python3
"""Relate the vigour-network weighting gain to the state-matched bradykinesia subscore.

The projected signal behind ``figures/gvs_projection_features_vs_sham`` is a weighted
sum of the HTML-mask voxels of ``data/active_bold_group.npy``.  Its raw amplitude is
dominated by subject-specific BOLD scaling, so no plain amplitude or shape feature of
the projection tracks clinical severity.  The weighting gain removes that scaling:

    kappa = SD( sum_v wn_v * x_v(t) ) / SD( mean_v x_v(t) ),   wn_v = w_v / sum(w)

Both terms are averages of the *same* voxels over the *same* time points; they differ
only in how the vigour weight map distributes emphasis across them.  kappa > 1 means
BOLD fluctuation inside the vigour mask is concentrated on the voxels the optimization
weighted most heavily; kappa < 1 means the weighted core carries less of the fluctuation
than the mask as a whole.  kappa is computed per run and averaged, because concatenating
separately acquired runs folds between-run mean shifts into the standard deviations.
"""

from __future__ import annotations

import argparse
import os
import re
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-gvs-updrs")
warnings.filterwarnings("ignore", message="Unable to import Axes3D.*")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

import spagetti_plot as sp
from analyze_rt_updrs_associations import load_updrs_bradykinesia


ROOT = Path(__file__).resolve().parent
DEFAULT_WORKBOOK = ROOT / "data" / "subjects_info" / "AllDressed_PD_Participant_Study_Visit_Info.xlsx"
DEFAULT_METADATA = ROOT / "data" / "GVS_projection_BOLD" / "gvs_projection_trial_metadata.tsv"
DEFAULT_FD_SUMMARY = ROOT / "figures" / "framewise_displacement" / "subject_session_fd_summary.csv"
DEFAULT_OUT_DIR = ROOT / "figures" / "gvs_projection_features_vs_sham"
OUT_STEM = "projection_weighting_gain_updrs_iii_bradykinesia"
MEDICATION_SESSION = {"OFF": 1, "ON": 2}
MIN_TRIALS_PER_RUN = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--active-bold-group", type=Path, default=ROOT / sp.DEFAULT_ACTIVE_BOLD_GROUP)
    parser.add_argument("--active-flat-indices", type=Path, default=ROOT / sp.DEFAULT_ACTIVE_FLAT_INDICES)
    parser.add_argument("--projection-weight-map", type=Path, default=ROOT / sp.DEFAULT_WEIGHT_MAP)
    parser.add_argument("--projection-html-mask", type=Path, default=ROOT / sp.DEFAULT_WEIGHT_HTML)
    parser.add_argument("--trial-metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--clinical-workbook", type=Path, default=DEFAULT_WORKBOOK)
    parser.add_argument("--fd-summary", type=Path, default=DEFAULT_FD_SUMMARY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--n-weight-permutations", type=int, default=1000)
    parser.add_argument("--n-label-permutations", type=int, default=20000)
    parser.add_argument("--random-state", type=int, default=20260819)
    return parser.parse_args()


def canonical_projection_subject(value: object) -> str:
    match = re.search(r"(?:PS)?[_-]?PD\s*0*(\d+)", str(value), flags=re.IGNORECASE)
    if match is None:
        raise ValueError(f"Could not parse subject identifier: {value!r}")
    return f"sub-pd{int(match.group(1)):03d}"


def load_mask_timeseries(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, int]:
    """Return the voxel-centred timeseries of the weighted mask and its normalised weights."""
    active_bold = np.load(args.active_bold_group, mmap_mode="r")
    if active_bold.ndim != 3:
        raise ValueError(f"Expected a 3D voxel-by-trial-by-time array, got {active_bold.shape}")
    active_flat = np.asarray(np.load(args.active_flat_indices, allow_pickle=False), dtype=np.int64).ravel()
    if active_flat.size != active_bold.shape[0]:
        raise ValueError("Active index count does not match the active BOLD voxel count")

    weight_img = nib.load(str(args.projection_weight_map))
    flat_weights = weight_img.get_fdata(dtype=np.float32).ravel()[active_flat].astype(np.float64)
    html_selected = sp._load_html_selected_mask(args.projection_html_mask, weight_img).ravel()
    keep = html_selected[active_flat] & np.isfinite(flat_weights) & (flat_weights != 0)
    selected_rows = np.flatnonzero(keep)
    if selected_rows.size == 0:
        raise ValueError("No active BOLD voxels overlap nonzero weights in the HTML mask")
    if np.any(flat_weights[selected_rows] < 0):
        raise RuntimeError("Weighting gain assumes an all-positive weight map")

    # Match the projection in spagetti_plot: centre every voxel over its whole timeseries.
    signal = np.asarray(active_bold[selected_rows], dtype=np.float32).reshape(selected_rows.size, -1)
    finite = np.isfinite(signal)
    signal = np.where(finite, signal, 0.0)
    voxel_mean = signal.sum(axis=1, keepdims=True) / np.maximum(finite.sum(axis=1, keepdims=True), 1)
    signal = np.where(finite, signal - voxel_mean, 0.0).astype(np.float32)

    weights = flat_weights[selected_rows]
    return signal, weights / weights.sum(), int(active_bold.shape[2])


def weighting_gain(signal: np.ndarray, weights: np.ndarray, columns: np.ndarray) -> float:
    block = signal[:, columns]
    uniform = np.full(weights.size, 1.0 / weights.size)
    return float((weights @ block).std(ddof=1) / (uniform @ block).std(ddof=1))


def run_level_gains(
    signal: np.ndarray,
    weights: np.ndarray,
    metadata: pd.DataFrame,
    trial_length: int,
    permuted_weights: np.ndarray,
) -> pd.DataFrame:
    uniform = np.full(weights.size, 1.0 / weights.size)
    records: list[dict[str, object]] = []
    for (subject, medication, run), group in metadata.groupby(["subject", "medication", "run"]):
        trials = np.sort(group["projected_trial_index"].to_numpy(dtype=np.int64))
        if trials.size < MIN_TRIALS_PER_RUN:
            continue
        columns = (trials[:, None] * trial_length + np.arange(trial_length)[None, :]).ravel()
        block = signal[:, columns]
        uniform_sd = (uniform @ block).std(ddof=1)
        record: dict[str, object] = {
            "subject": subject,
            "medication": medication,
            "run": int(run),
            "n_trials": int(trials.size),
            "weighting_gain": float((weights @ block).std(ddof=1) / uniform_sd),
        }
        gains = (permuted_weights @ block).std(axis=1, ddof=1) / uniform_sd
        record.update({f"weighting_gain_perm_{index}": float(v) for index, v in enumerate(gains)})
        records.append(record)
    return pd.DataFrame(records)


def subject_table(run_gains: pd.DataFrame, updrs: pd.DataFrame, fd: pd.DataFrame | None) -> pd.DataFrame:
    value_columns = [c for c in run_gains.columns if c.startswith("weighting_gain")]
    subjects = (
        run_gains.groupby(["subject", "medication"], as_index=False)
        .agg(n_runs=("run", "nunique"), n_trials=("n_trials", "sum"))
        .merge(
            run_gains.groupby(["subject", "medication"], as_index=False)[value_columns].mean(),
            on=["subject", "medication"],
        )
    )
    clinical = updrs.copy()
    clinical["subject"] = clinical["subject"].map(canonical_projection_subject)
    merged = subjects.merge(
        clinical[["subject", "medication", "updrs_iii_bradykinesia_subscore"]],
        on=["subject", "medication"],
        how="left",
        validate="one_to_one",
    )
    if fd is not None:
        merged["session"] = merged["medication"].map(MEDICATION_SESSION)
        merged = merged.merge(fd, on=["subject", "session"], how="left")
    return merged.dropna(subset=["updrs_iii_bradykinesia_subscore"]).reset_index(drop=True)


def load_fd(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    fd = pd.read_csv(path)
    return fd[["subject", "session", "mean_fd_mm"]].copy()


def partial_rank_correlation(x: np.ndarray, y: np.ndarray, covariate: np.ndarray) -> tuple[float, float]:
    ranks = [stats.rankdata(v) for v in (x, y, covariate)]
    residuals = [r - np.polyval(np.polyfit(ranks[2], r, 1), ranks[2]) for r in ranks[:2]]
    r = float(np.corrcoef(*residuals)[0, 1])
    n = x.size
    t = r * np.sqrt((n - 3) / max(1e-12, 1 - r**2))
    return r, float(2 * stats.t.sf(abs(t), n - 3))


def analyze(subjects: pd.DataFrame, n_weight_perm: int, n_label_perm: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for medication in ("OFF", "ON"):
        data = subjects.loc[subjects["medication"].eq(medication)]
        x = data["weighting_gain"].to_numpy(dtype=float)
        y = data["updrs_iii_bradykinesia_subscore"].to_numpy(dtype=float)
        spearman = stats.spearmanr(x, y)
        pearson = stats.pearsonr(x, y)

        observed = abs(spearman.statistic)
        label_null = np.array([abs(stats.spearmanr(x, rng.permutation(y)).statistic) for _ in range(n_label_perm)])
        weight_null = np.array(
            [stats.spearmanr(data[f"weighting_gain_perm_{i}"].to_numpy(dtype=float), y).statistic
             for i in range(n_weight_perm)]
        )
        loo = [stats.spearmanr(np.delete(x, i), np.delete(y, i)) for i in range(x.size)]

        row = {
            "medication": medication,
            "n_subjects": int(x.size),
            "spearman_rho": float(spearman.statistic),
            "spearman_p_value": float(spearman.pvalue),
            "pearson_r": float(pearson.statistic),
            "pearson_p_value": float(pearson.pvalue),
            "p_label_permutation": float((1 + np.sum(label_null >= observed)) / (1 + label_null.size)),
            "p_weight_permutation": float((1 + np.sum(np.abs(weight_null) >= observed)) / (1 + weight_null.size)),
            "weight_null_mean_rho": float(weight_null.mean()),
            "weight_null_sd_rho": float(weight_null.std()),
            "loo_min_rho": float(min(v.statistic for v in loo)),
            "loo_max_rho": float(max(v.statistic for v in loo)),
            "loo_max_p_value": float(max(v.pvalue for v in loo)),
        }
        if "mean_fd_mm" in data.columns and data["mean_fd_mm"].notna().all():
            fd = data["mean_fd_mm"].to_numpy(dtype=float)
            partial_r, partial_p = partial_rank_correlation(x, y, fd)
            row["fd_partial_rho"] = partial_r
            row["fd_partial_p_value"] = partial_p
            row["feature_vs_fd_rho"] = float(stats.spearmanr(x, fd).statistic)
        rows.append(row)
    return pd.DataFrame(rows)


def add_panel_labels(axes, labels: tuple[str, ...]) -> None:
    for ax, label in zip(np.ravel(axes), labels):
        ax.text(-0.14, 1.05, label, transform=ax.transAxes, ha="left", va="bottom",
                fontsize=22, fontweight="bold", color="black", clip_on=False)


def plot(subjects: pd.DataFrame, results: pd.DataFrame, n_weight_perm: int, output_stem: Path) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.2), constrained_layout=True)
    gains = subjects["weighting_gain"].to_numpy(dtype=float)
    pad = 0.12 * (gains.max() - gains.min())
    y_limits = (gains.min() - pad, gains.max() + pad)

    for ax, medication in zip(axes[:2], ("OFF", "ON")):
        data = subjects.loc[subjects["medication"].eq(medication)]
        x = data["updrs_iii_bradykinesia_subscore"].to_numpy(dtype=float)
        y = data["weighting_gain"].to_numpy(dtype=float)
        model = sm.OLS(y, sm.add_constant(x)).fit()
        grid = np.linspace(x.min(), x.max(), 200)
        prediction = model.get_prediction(sm.add_constant(grid)).summary_frame(alpha=0.05)
        ax.fill_between(grid, prediction["mean_ci_lower"], prediction["mean_ci_upper"],
                        color="#4C78A8", alpha=0.18, linewidth=0)
        ax.plot(grid, prediction["mean"], color="#245A86", linewidth=2.2)
        ax.scatter(x, y, s=86, color="#E07A5F", edgecolor="white", linewidth=0.9, zorder=3)
        ax.axhline(1.0, color="#888888", linestyle="--", linewidth=1.1, zorder=1)

        result = results.loc[results["medication"].eq(medication)].iloc[0]
        ax.text(0.04, 0.96,
                f"Spearman $\\rho$ = {result['spearman_rho']:.3f}\n"
                f"p = {result['spearman_p_value']:.4g}\nn = {int(result['n_subjects'])}",
                transform=ax.transAxes, va="top", fontsize=12,
                bbox={"boxstyle": "round,pad=0.35", "facecolor": "white",
                      "edgecolor": "#BBBBBB", "alpha": 0.9})
        ax.set_xlabel(f"MDS-UPDRS III bradykinesia subscore\n({medication} medication)", fontsize=13)
        ax.set_ylabel("Vigour-network weighting gain $\\kappa$", fontsize=13)
        ax.set_title(f"Medication {medication}", fontsize=14, fontweight="bold")
        ax.set_ylim(*y_limits)
        ax.tick_params(axis="both", labelsize=12)
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)

    ax = axes[2]
    off = subjects.loc[subjects["medication"].eq("OFF")]
    y_off = off["updrs_iii_bradykinesia_subscore"].to_numpy(dtype=float)
    null = np.array([stats.spearmanr(off[f"weighting_gain_perm_{i}"].to_numpy(dtype=float), y_off).statistic
                     for i in range(n_weight_perm)])
    observed = float(results.loc[results["medication"].eq("OFF"), "spearman_rho"].iloc[0])
    p_spec = float(results.loc[results["medication"].eq("OFF"), "p_weight_permutation"].iloc[0])
    ax.hist(null, bins=32, color="#9EB9D4", edgecolor="white", linewidth=0.6)
    ax.axvline(observed, color="#C1352B", linewidth=2.6)
    ax.text(0.03, 0.96,
            f"vigour weight map\n$\\rho$ = {observed:.3f}\np = {p_spec:.4g}\n"
            f"({n_weight_perm} shuffled weight maps)",
            transform=ax.transAxes, va="top", fontsize=11.5,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white",
                  "edgecolor": "#BBBBBB", "alpha": 0.9})
    ax.set_xlabel("Spearman $\\rho$ with bradykinesia\n(weights shuffled across the same voxels, OFF)", fontsize=13)
    ax.set_ylabel("Shuffled weight maps", fontsize=13)
    ax.set_title("Spatial-pattern specificity", fontsize=14, fontweight="bold")
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)

    add_panel_labels(axes, ("A", "B", "C"))
    fig.savefig(output_stem.with_suffix(".png"), dpi=220)
    fig.savefig(output_stem.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    args = parse_args()
    metadata = pd.read_csv(args.trial_metadata, sep="\t")
    signal, weights, trial_length = load_mask_timeseries(args)
    print(f"Weighted mask: {weights.size} voxels; trial length {trial_length}")

    rng = np.random.default_rng(args.random_state)
    permuted_weights = np.stack([weights[rng.permutation(weights.size)] for _ in range(args.n_weight_permutations)])

    run_gains = run_level_gains(signal, weights, metadata, trial_length, permuted_weights)
    subjects = subject_table(run_gains, load_updrs_bradykinesia(args.clinical_workbook), load_fd(args.fd_summary))
    results = analyze(subjects, args.n_weight_permutations, args.n_label_permutations, args.random_state)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    keep = ["subject", "medication", "run", "n_trials", "weighting_gain"]
    run_gains[keep].to_csv(args.out_dir / f"{OUT_STEM}_run_data.csv", index=False)
    subjects[[c for c in subjects.columns if not c.startswith("weighting_gain_perm_")]].to_csv(
        args.out_dir / f"{OUT_STEM}_subject_data.csv", index=False)
    results.to_csv(args.out_dir / f"{OUT_STEM}_statistics.csv", index=False)
    plot(subjects, results, args.n_weight_permutations, args.out_dir / OUT_STEM)

    pd.set_option("display.width", 220)
    print(results.to_string(index=False))
    print(f"Saved outputs with stem {args.out_dir / OUT_STEM}")


if __name__ == "__main__":
    main()
