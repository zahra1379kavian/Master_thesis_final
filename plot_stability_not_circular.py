#!/usr/bin/env python3
"""Rebuttal figure: the vigour-network projection is more stable than the behaviour
it tracks, and that stability is NOT manufactured by the stability penalty.

Circularity concern (reviewer):
    "The paper finds a stable network because stability was built into the
     optimisation, so the stable-central-vigour interpretation may be circular."

This script answers it with the same consecutive-trial variability metric that
appears in the objective (``_adjacent_diff_ratio_sum``), evaluated on:

  * the reaction-time series itself (the behaviour being tracked),
  * the vigour-network / full-objective projection (stability terms included),
  * ablation projections in which one or more objective terms were removed,
  * a permutation null (full-model weights shuffled across the SAME voxels).

Key facts the figure makes visible:
  1. Every projection is much more stable (lower variability) than RT.
  2. Removing the stability penalty does NOT move variability back toward RT;
     the penalty explains only a small fraction of the stability margin.
=> The low consecutive-trial variability is an intrinsic property of projecting
   distributed single-trial betas, not an artefact of the penalty.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

warnings.filterwarnings("ignore", message="Unable to import Axes3D.*")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from projected_sig_vs_RT import (
    AXIS_TICK_FONT_SIZE,
    DEFAULT_BEHAVIOUR_DIR,
    DEFAULT_DATA_DIR,
    PAPER_FONT_FAMILY,
    VARIABILITY_AXIS_LABEL,
    _adjacent_diff_ratio_sum,
    _align_trials,
    _behaviour_path,
    _category_sort_key,
    _discover_beta_runs,
    _load_behaviour_rt,
)
from plot_ablation_projection_rt_variability import (
    DEFAULT_ABLATION_DIR,
    DEFAULT_ABLATION_SUMMARY,
    DEFAULT_MAIN_HTML,
    PROJECTION_TRIAL_CHUNK_SIZE,
    _discover_model_specs,
    _load_weights,
)

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "figures" / "projected_RT"
FIGURE_STEM = "stability_not_circular"
N_NULL = 500
RNG_SEED = 20240617
MAIN_PERCENTILE = 90.0
BEHAVIOUR_COLUMN = 1

# The two maps optimised with the COMPLETE objective (task + BOLD-stability +
# beta-stability + smoothness). Every other map is an ablation in which one or
# more objective terms were removed.
COMPLETE_OBJECTIVE_LABELS = {"Vigour network", "Full model (ablation)"}

# Colour scheme (colour-blind friendly).
COLOR_BEHAVIOUR = "#D55E00"   # vermillion  – the reference behaviour
COLOR_ON = "#0072B2"          # blue        – stability penalty ON
COLOR_OFF = "#56B4E9"         # light blue  – stability penalty OFF
COLOR_NULL = "0.55"           # grey        – permutation null


def _project_matrix(beta_path: Path, union_mask: np.ndarray, weight_matrix: np.ndarray,
                    active_weight_matrix: np.ndarray, spatial_shape) -> np.ndarray:
    """Project one run's beta volume through every weight row at once."""
    beta = np.load(beta_path, mmap_mode="r")
    if beta.shape[:3] != spatial_shape:
        raise ValueError(f"Shape mismatch for {beta_path}: {beta.shape[:3]} vs {spatial_shape}")
    out = np.full((weight_matrix.shape[0], beta.shape[3]), np.nan, dtype=np.float64)
    for start in range(0, beta.shape[3], PROJECTION_TRIAL_CHUNK_SIZE):
        stop = min(start + PROJECTION_TRIAL_CHUNK_SIZE, beta.shape[3])
        selected = np.asarray(beta[union_mask, start:stop], dtype=np.float64)
        finite = np.isfinite(selected)
        filled = np.nan_to_num(selected, nan=0.0, posinf=0.0, neginf=0.0)
        proj = weight_matrix @ filled
        proj[(active_weight_matrix @ finite.astype(np.float64)) <= 0] = np.nan
        out[:, start:stop] = proj
    return out


def _build_subject_table() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    specs = _discover_model_specs(DEFAULT_MAIN_HTML, DEFAULT_ABLATION_DIR, DEFAULT_ABLATION_SUMMARY)
    model_labels = [str(s["label"]) for s in specs]
    main_index = next(i for i, s in enumerate(specs) if s["source"] == "main")

    # Load weights, build union mask and aligned weight matrix.
    spatial_shape = None
    loaded = []
    for spec in specs:
        thr = MAIN_PERCENTILE if bool(spec["threshold_main"]) else None
        w = _load_weights(Path(spec["weight_map"]), threshold_percentile=thr)
        spatial_shape = w.shape if spatial_shape is None else spatial_shape
        loaded.append(w)
    union_mask = np.zeros(spatial_shape, dtype=bool)
    for w in loaded:
        union_mask |= np.isfinite(w) & (w != 0)
    weight_matrix = np.vstack([w[union_mask].astype(np.float64) for w in loaded])

    # Permutation null: shuffle the vigour-network weights across its own voxels.
    rng = np.random.default_rng(RNG_SEED)
    main_row = weight_matrix[main_index]
    nz = np.nonzero(main_row)[0]
    null_matrix = np.zeros((N_NULL, weight_matrix.shape[1]), dtype=np.float64)
    for i in range(N_NULL):
        null_matrix[i, nz] = rng.permutation(main_row[nz])

    combined = np.vstack([weight_matrix, null_matrix])
    active = (combined != 0).astype(np.float64)

    beta_runs = _discover_beta_runs(DEFAULT_DATA_DIR)
    n_models = len(specs)

    # Accumulators keyed by subject.
    proj_var: dict[str, list[np.ndarray]] = {}   # per run: array over models
    null_var: dict[str, list[np.ndarray]] = {}   # per run: array over nulls
    rt_var: dict[str, list[float]] = {}

    for k, info in enumerate(beta_runs, 1):
        sub, ses, run = str(info["sub"]), int(info["ses"]), int(info["run"])
        if k == 1 or k % 10 == 0 or k == len(beta_runs):
            print(f"  projecting run {k}/{len(beta_runs)}: {sub} ses-{ses} run-{run}", flush=True)
        rt = _load_behaviour_rt(_behaviour_path(DEFAULT_BEHAVIOUR_DIR, sub, ses, run), BEHAVIOUR_COLUMN)
        projected = _project_matrix(Path(info["path"]), union_mask, combined, active, spatial_shape)

        # Behaviour reference (model-independent).
        rt_score, _ = _adjacent_diff_ratio_sum(np.asarray(rt, float)[np.isfinite(rt)])
        rt_var.setdefault(sub, []).append(rt_score)

        model_scores = np.full(n_models, np.nan)
        for m in range(n_models):
            sig, aligned = _align_trials(projected[m], rt, f"{model_labels[m]} {sub}")
            keep = np.isfinite(sig) & np.isfinite(aligned)
            score, _ = _adjacent_diff_ratio_sum(sig[keep])
            model_scores[m] = score
        proj_var.setdefault(sub, []).append(model_scores)

        nscores = np.full(N_NULL, np.nan)
        for j in range(N_NULL):
            sig, aligned = _align_trials(projected[n_models + j], rt, "null")
            keep = np.isfinite(sig) & np.isfinite(aligned)
            score, _ = _adjacent_diff_ratio_sum(sig[keep])
            nscores[j] = score
        null_var.setdefault(sub, []).append(nscores)

    # Aggregate to subject level (mean across that subject's runs).
    subjects = sorted(proj_var, key=_category_sort_key)
    rows = []
    for sub in subjects:
        model_mean = np.nanmean(np.vstack(proj_var[sub]), axis=0)
        rt_mean = float(np.nanmean(rt_var[sub]))
        null_mean = float(np.nanmean(np.vstack(null_var[sub])))  # mean over runs & permutations
        row = {"sub": sub, "rt_variability": rt_mean, "null_variability": null_mean}
        for m, lab in enumerate(model_labels):
            row[lab] = float(model_mean[m])
        rows.append(row)
    subject_df = pd.DataFrame(rows)

    # Per-permutation null grand means (for the null band).
    null_perm_subject = np.vstack([np.nanmean(np.vstack(null_var[s]), axis=0) for s in subjects])  # subjects x N_NULL
    null_perm_grand = np.nanmean(null_perm_subject, axis=0)  # length N_NULL

    return subject_df, pd.DataFrame({"null_grandmean_variability": null_perm_grand}), model_labels


def _ci95(values: np.ndarray) -> tuple[float, float, float]:
    """Mean and 95% CI half-width (t) for a 1D sample. Returns (mean, lo, hi)."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    n = v.size
    mean = float(np.mean(v))
    if n < 2:
        return mean, mean, mean
    from scipy import stats
    sem = float(np.std(v, ddof=1) / np.sqrt(n))
    half = float(stats.t.ppf(0.975, n - 1)) * sem
    return mean, mean - half, mean + half


def _paired_diff_stats(a: np.ndarray, b: np.ndarray) -> dict:
    """Paired a-b mean, 95% CI, t-test p (a,b same subjects)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    keep = np.isfinite(a) & np.isfinite(b)
    d = a[keep] - b[keep]
    from scipy import stats
    mean, lo, hi = _ci95(d)
    t, p = stats.ttest_rel(a[keep], b[keep])
    return {"mean": mean, "lo": lo, "hi": hi, "p": float(p), "n": int(d.size)}


def make_figure(subject_df: pd.DataFrame, null_perm: pd.DataFrame, model_labels: list[str]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- statistics -------------------------------------------------------
    rt_mean, rt_lo, rt_hi = _ci95(subject_df["rt_variability"].to_numpy())
    null_band_lo, null_band_hi = np.percentile(null_perm["null_grandmean_variability"], [2.5, 97.5])
    null_mean = float(null_perm["null_grandmean_variability"].mean())

    # Order models: stability ON first (highlight), then OFF, sorted by variability.
    def _grp(lab: str) -> int:
        return 0 if lab in COMPLETE_OBJECTIVE_LABELS else 1
    stats_by_model = {lab: _ci95(subject_df[lab].to_numpy()) for lab in model_labels}
    ordered = sorted(model_labels, key=lambda L: (_grp(L), stats_by_model[L][0]))

    # Penalty-contribution comparisons (paired vs Vigour network).
    vigour = subject_df["Vigour network"].to_numpy()
    margin = _paired_diff_stats(subject_df["rt_variability"].to_numpy(), vigour)
    contrib_specs = [
        ("Remove the\nentire objective", "No objective penalties"),
        ("Remove BOLD\nstability term", "No BOLD stability"),
        ("Remove beta\nstability term", "No beta stability"),
    ]
    contribs = []
    for name, lab in contrib_specs:
        if lab in subject_df.columns:
            contribs.append((name, _paired_diff_stats(subject_df[lab].to_numpy(), vigour)))

    # ---- figure -----------------------------------------------------------
    with plt.rc_context({
        "font.family": "sans-serif",
        "font.sans-serif": [PAPER_FONT_FAMILY, "Arial", "DejaVu Sans"],
        "font.size": AXIS_TICK_FONT_SIZE,
        "axes.labelsize": AXIS_TICK_FONT_SIZE,
        "xtick.labelsize": AXIS_TICK_FONT_SIZE - 1,
        "ytick.labelsize": AXIS_TICK_FONT_SIZE - 1,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }):
        fig, (axA, axB) = plt.subplots(
            1, 2, figsize=(13.4, 6.2), gridspec_kw={"width_ratios": [1.55, 1.0]}
        )

        # ===== Panel A: forest plot ======================================
        y = np.arange(len(ordered))[::-1]
        # RT reference band spanning the panel.
        axA.axvspan(rt_lo, rt_hi, color=COLOR_BEHAVIOUR, alpha=0.16, zorder=0)
        axA.axvline(rt_mean, color=COLOR_BEHAVIOUR, lw=2.0, zorder=1)
        axA.text(rt_mean, (len(ordered) - 1) / 2.0, "Reaction time (behaviour tracked)",
                 color=COLOR_BEHAVIOUR, ha="center", va="center", fontsize=11.5,
                 fontweight="bold", rotation=90,
                 bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=COLOR_BEHAVIOUR, alpha=0.92))
        # Permutation-null band.
        axA.axvspan(null_band_lo, null_band_hi, color=COLOR_NULL, alpha=0.30, zorder=0)
        axA.axvline(null_mean, color=COLOR_NULL, lw=1.3, ls=(0, (4, 2)), zorder=1)

        for yi, lab in zip(y, ordered):
            mean, lo, hi = stats_by_model[lab]
            on = lab in COMPLETE_OBJECTIVE_LABELS
            color = COLOR_ON if on else COLOR_OFF
            axA.errorbar(mean, yi, xerr=[[mean - lo], [hi - mean]], fmt="o",
                         ms=11 if on else 8, color=color,
                         ecolor=color, elinewidth=1.8, capsize=3.5,
                         markeredgecolor="#1a1a1a" if on else "white",
                         markeredgewidth=1.1 if on else 0.7, zorder=4)
        axA.set_yticks(y)
        axA.set_yticklabels(
            [f"{lab}  ●" if lab in COMPLETE_OBJECTIVE_LABELS else lab for lab in ordered],
            fontsize=11.2)
        for tick, lab in zip(axA.get_yticklabels(), ordered):
            if lab in COMPLETE_OBJECTIVE_LABELS:
                tick.set_fontweight("bold")
        n_rows = len(ordered)
        axA.set_ylim(-0.95, n_rows + 0.25)
        axA.set_xlabel(f"{VARIABILITY_AXIS_LABEL}  (objective metric)")
        axA.grid(axis="x", color="0.88", lw=0.5)
        axA.set_axisbelow(True)
        for s in ("top", "right"):
            axA.spines[s].set_visible(False)

        # stability-margin bracket (complete-objective projection -> RT), drawn above the points
        vig_mean = stats_by_model["Vigour network"][0]
        y_br = n_rows - 0.35
        axA.annotate("", xy=(rt_mean, y_br), xytext=(vig_mean, y_br),
                     arrowprops=dict(arrowstyle="<->", color="0.30", lw=1.4))
        axA.text((rt_mean + vig_mean) / 2.0, y_br + 0.16,
                 f"stability margin ≈ {rt_mean - vig_mean:.0f}",
                 color="0.25", ha="center", va="bottom", fontsize=10.5)
        axA.text(null_mean, -0.42, "permutation\nnull", color="0.40",
                 ha="center", va="top", fontsize=8.6, linespacing=0.95)

        legend_handles = [
            Patch(facecolor=COLOR_ON, edgecolor="#1a1a1a", label="Complete objective ●"),
            Patch(facecolor=COLOR_OFF, edgecolor="white", label="Ablation"),
            Patch(facecolor=COLOR_NULL, alpha=0.5, label="Permutation null (95%)"),
            Patch(facecolor=COLOR_BEHAVIOUR, alpha=0.4, label="RT reference (95% CI)"),
        ]
        axA.legend(handles=legend_handles, loc="lower right", fontsize=9.2,
                   frameon=True, framealpha=0.95, borderpad=0.5)
        axA.set_title("A   Every projection is more stable than the behaviour it tracks",
                      loc="left", fontsize=13.5, fontweight="bold", pad=10)

        # ===== Panel B: stability-margin decomposition ===================
        labels_B = ["Margin vs\nbehaviour\n(RT − full model)"] + [n for n, _ in contribs]
        means_B = [margin["mean"]] + [c["mean"] for _, c in contribs]
        los_B = [margin["lo"]] + [c["lo"] for _, c in contribs]
        his_B = [margin["hi"]] + [c["hi"] for _, c in contribs]
        ps_B = [None] + [c["p"] for _, c in contribs]
        colors_B = [COLOR_BEHAVIOUR] + [COLOR_OFF] * len(contribs)

        xb = np.arange(len(labels_B))
        for xi, m, lo, hi, c in zip(xb, means_B, los_B, his_B, colors_B):
            axB.bar(xi, m, width=0.62, color=c, alpha=0.85, edgecolor="#1a1a1a", lw=0.8, zorder=3)
            axB.errorbar(xi, m, yerr=[[m - lo], [hi - m]], fmt="none",
                         ecolor="#1a1a1a", elinewidth=1.5, capsize=4, zorder=4)
        axB.axhline(0, color="0.3", lw=1.0)
        # annotate p-values for penalty-removal bars
        for xi, m, hi, p in zip(xb, means_B, his_B, ps_B):
            if p is None:
                txt = "***" if margin["p"] < 1e-3 else f"p={margin['p']:.3f}"
                axB.text(xi, hi + 0.4, txt, ha="center", va="bottom", fontsize=10.5, fontweight="bold")
            else:
                txt = "n.s." if p >= 0.05 else f"p={p:.3f}"
                axB.text(xi, (hi if m >= 0 else m) + 0.4, txt, ha="center", va="bottom",
                         fontsize=10, color="0.25")
        axB.set_xticks(xb)
        axB.set_xticklabels(labels_B, fontsize=10.2)
        axB.set_ylabel("Δ consecutive-trial variability\n(subject-paired)")
        axB.grid(axis="y", color="0.88", lw=0.5)
        axB.set_axisbelow(True)
        for s in ("top", "right"):
            axB.spines[s].set_visible(False)
        axB.set_title("B   Removing the penalty recovers little of that margin",
                      loc="left", fontsize=13.5, fontweight="bold", pad=10)
        # fraction-of-margin annotation
        if contribs:
            worst = max(c["mean"] for _, c in contribs)
            frac = 100.0 * worst / margin["mean"] if margin["mean"] else float("nan")
            axB.text(0.97, 0.97,
                     f"objective terms explain ≤ {frac:.0f}%\nof the stability margin vs behaviour",
                     transform=axB.transAxes, ha="right", va="top", fontsize=10,
                     bbox=dict(boxstyle="round,pad=0.4", fc="#f3f3f3", ec="0.7"))

        fig.tight_layout(w_pad=2.5)
        png = OUT_DIR / f"{FIGURE_STEM}.png"
        pdf = OUT_DIR / f"{FIGURE_STEM}.pdf"
        fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.04)
        fig.savefig(pdf, bbox_inches="tight", pad_inches=0.04)
        plt.close(fig)

    # ---- save tables + print summary -------------------------------------
    subject_df.to_csv(OUT_DIR / f"{FIGURE_STEM}_subject_metrics.csv", index=False)
    null_perm.to_csv(OUT_DIR / f"{FIGURE_STEM}_null_grandmeans.csv", index=False)
    summary_rows = [{"quantity": "Reaction time (behaviour)", "mean": rt_mean, "ci_lo": rt_lo, "ci_hi": rt_hi}]
    summary_rows.append({"quantity": "Permutation null", "mean": null_mean,
                         "ci_lo": null_band_lo, "ci_hi": null_band_hi})
    for lab in model_labels:
        m, lo, hi = stats_by_model[lab]
        summary_rows.append({"quantity": lab, "mean": m, "ci_lo": lo, "ci_hi": hi})
    pd.DataFrame(summary_rows).to_csv(OUT_DIR / f"{FIGURE_STEM}_summary.csv", index=False)

    print("\n================ STABILITY-NOT-CIRCULAR SUMMARY ================")
    print(f"RT (behaviour) variability:      {rt_mean:6.2f}  [{rt_lo:.2f}, {rt_hi:.2f}]")
    print(f"Permutation null variability:    {null_mean:6.2f}  [{null_band_lo:.2f}, {null_band_hi:.2f}]")
    for lab in model_labels:
        m, lo, hi = stats_by_model[lab]
        flag = "  <-- stability ON" if lab in COMPLETE_OBJECTIVE_LABELS else ""
        print(f"  {lab:24s} {m:6.2f}  [{lo:.2f}, {hi:.2f}]{flag}")
    print(f"\nStability margin (RT - vigour): {margin['mean']:.2f} "
          f"[{margin['lo']:.2f}, {margin['hi']:.2f}], p={margin['p']:.4g}, n={margin['n']}")
    for name, c in contribs:
        nm = name.replace(chr(10), ' ')
        print(f"Penalty contribution [{nm}]: {c['mean']:.2f} "
              f"[{c['lo']:.2f}, {c['hi']:.2f}], p={c['p']:.4g}")
    print("===============================================================\n")


def main() -> None:
    import sys
    if "--replot" in sys.argv:
        print("Replotting from cached CSVs ...")
        subject_df = pd.read_csv(OUT_DIR / f"{FIGURE_STEM}_subject_metrics.csv")
        null_perm = pd.read_csv(OUT_DIR / f"{FIGURE_STEM}_null_grandmeans.csv")
        reserved = {"sub", "rt_variability", "null_variability"}
        model_labels = [c for c in subject_df.columns if c not in reserved]
        make_figure(subject_df, null_perm, model_labels)
        print(f"Saved figure to {OUT_DIR / (FIGURE_STEM + '.png')}")
        return
    print("Projecting betas through all models + permutation null ...")
    subject_df, null_perm, model_labels = _build_subject_table()
    make_figure(subject_df, null_perm, model_labels)
    print(f"Saved figure to {OUT_DIR / (FIGURE_STEM + '.png')}")


if __name__ == "__main__":
    main()
