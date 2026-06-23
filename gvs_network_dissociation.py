#!/usr/bin/env python3
"""
GVS effect: vigour-projection network vs task-map network DIRECT dissociation.

Two parts, both using the same 18 subjects so the contrast is within-subject:

(A) Direct interaction (difference-of-differences) on the block-shape features
    that already exist in the *_subject_signal_feature_pairs.csv files. For every
    (active GVS condition x feature), test the within-subject contrast:

        (active-sham vigour delta) - (active-sham task delta)

    against zero with an exact paired sign-flip permutation test. FDR (BH) across
    all condition x feature tests. Output: table CSV + direct-contrast heatmap.

(B) Two extra trial-derived measures the existing pipeline does not compute,
    built from the per-trial feature CSVs with the same run-wise sham pairing:
      - frac_amp : fractional change in block amplitude vs that run's sham,
                   amplitude = mean peak-to-peak across trials in the block.
      - trial_var: across-trial variability change = SD across trials of the
                   feature, active minus sham (fractional).
    Computed for a small panel of features and run through the same interaction
    test + FDR. Output: table CSV + figure.

All permutation p-values are exact sign-flip (2^18 = 262144 sign patterns).
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
VIG_DIR = FIG / "gvs_projection_features_vs_sham"
TASK_DIR = FIG / "gvs_task_map_bold_features_vs_sham"
OUT = FIG / "gvs_network_dissociation"
OUT.mkdir(exist_ok=True)

SHAM = "gvs-01"
FEAT_LABELS = {
    "mean_level": "Mean level",
    "auc": "Area under curve",
    "peak_to_peak": "Peak-to-peak amplitude",
    "baseline_to_peak": "Baseline to peak",
    "baseline_to_trough": "Baseline to trough",
    "abs_baseline_response": "Max abs. baseline response",
    "early_late_change": "Late minus early",
    "slope": "Linear slope",
    "temporal_sd": "Temporal SD",
}

def gvs_display_label(gvs_code):
    code = str(gvs_code).strip().lower().replace("_", "-").replace(" ", "-")
    if code in {"gvs-01", "sham"}:
        return "sham"
    if code.startswith("gvs-"):
        try:
            gvs_index = int(code.split("-", 1)[1])
        except ValueError:
            return str(gvs_code)
        if gvs_index == 1:
            return "sham"
        return f"gvs{gvs_index - 1}"
    return str(gvs_code)

# ----------------------------------------------------------------------------
# exact paired sign-flip permutation test on a 1-D vector of subject values
# ----------------------------------------------------------------------------
_SIGN_CACHE = {}
def _signs(n):
    if n not in _SIGN_CACHE:
        # rows = 2^n sign patterns, columns = subjects
        grid = ((np.arange(2 ** n)[:, None] >> np.arange(n)[None, :]) & 1)
        _SIGN_CACHE[n] = np.where(grid == 1, 1.0, -1.0)
    return _SIGN_CACHE[n]

def signflip_p(x):
    """Two-sided exact sign-flip p-value for H0: mean(x)=0."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n == 0:
        return np.nan, 0
    obs = abs(x.mean())
    null = np.abs(_signs(n) @ x) / n
    p = (null >= obs - 1e-12).mean()
    return p, n

def cohen_dz(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    sd = x.std(ddof=1)
    return x.mean() / sd if sd > 0 else np.nan

def bh_fdr(p):
    p = np.asarray(p, float)
    ok = ~np.isnan(p)
    q = np.full_like(p, np.nan)
    idx = np.where(ok)[0]
    pp = p[idx]
    order = np.argsort(pp)
    m = len(pp)
    ranked = pp[order]
    qv = ranked * m / (np.arange(m) + 1)
    qv = np.minimum.accumulate(qv[::-1])[::-1]
    out = np.empty(m); out[order] = np.clip(qv, 0, 1)
    q[idx] = out
    return q

# ----------------------------------------------------------------------------
# interaction test for one aligned pair of per-subject vectors
# ----------------------------------------------------------------------------
def interaction(vig_by_sub, task_by_sub):
    subs = sorted(set(vig_by_sub) & set(task_by_sub))
    v = np.array([vig_by_sub[s] for s in subs], float)
    t = np.array([task_by_sub[s] for s in subs], float)
    m = ~(np.isnan(v) | np.isnan(t))
    v, t = v[m], t[m]
    diff = v - t
    p, n = signflip_p(diff)
    diff_mean = float(np.mean(diff)) if n else np.nan
    diff_sd = float(np.std(diff, ddof=1)) if n > 1 else np.nan
    if n > 1 and np.isfinite(diff_sd):
        sem = diff_sd / np.sqrt(n)
        half_width = stats_t_critical(n - 1) * sem
        diff_ci_low = diff_mean - half_width
        diff_ci_high = diff_mean + half_width
    else:
        diff_ci_low = np.nan
        diff_ci_high = np.nan
    return dict(
        n=n,
        vigour_mean_delta=float(np.mean(v)) if n else np.nan,
        task_mean_delta=float(np.mean(t)) if n else np.nan,
        direct_contrast_mean=diff_mean,
        direct_contrast_ci95_low=diff_ci_low,
        direct_contrast_ci95_high=diff_ci_high,
        direct_contrast_dz=cohen_dz(diff),
        p=p,
    )

def stats_t_critical(df):
    from scipy import stats

    return float(stats.t.ppf(0.975, df))

# ============================================================================
# PART A
# ============================================================================
def part_a():
    vig = pd.read_csv(VIG_DIR / "gvs_vs_sham_subject_signal_feature_pairs.csv")
    task = pd.read_csv(TASK_DIR / "gvs_vs_sham_subject_signal_feature_pairs.csv")
    conds = sorted(vig["active_gvs"].unique())
    feats = [feat for feat in FEAT_LABELS if feat in set(vig["feature"]) and feat in set(task["feature"])]
    recs = []
    for c in conds:
        for f in feats:
            vd = vig[(vig.active_gvs == c) & (vig.feature == f)]
            td = task[(task.active_gvs == c) & (task.feature == f)]
            vb = dict(zip(vd.subject, vd.delta_active_minus_sham))
            tb = dict(zip(td.subject, td.delta_active_minus_sham))
            r = interaction(vb, tb)
            r.update(active_gvs=c, gvs_label=gvs_display_label(c), feature=f, label=FEAT_LABELS[f])
            recs.append(r)
    df = pd.DataFrame(recs)
    df["q"] = bh_fdr(df["p"].values)
    df = df[["active_gvs", "gvs_label", "feature", "label", "n",
             "vigour_mean_delta", "task_mean_delta",
             "direct_contrast_mean", "direct_contrast_ci95_low",
             "direct_contrast_ci95_high", "direct_contrast_dz", "p", "q"]]
    df.to_csv(OUT / "partA_feature_interaction.csv", index=False)
    return df, conds, feats

# ============================================================================
# PART B  -- trial-derived measures
# ============================================================================
def block_table(trial_csv):
    """Per (subject,session,run,gvs) block: mean & SD across trials of each feature."""
    df = pd.read_csv(trial_csv)
    df["gvs"] = "gvs-" + df["gvs_id"].astype(int).astype(str).str.zfill(2)
    keys = ["subject", "session", "run", "gvs"]
    agg = df.groupby(keys)[list(FEAT_LABELS)].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    return agg.reset_index()

def derived_subject_vectors(trial_csv, feature, measure):
    """
    Returns {subject: {active_gvs: value}} for a given feature and measure.
    measure = 'frac_amp'  -> per run (mean_active-mean_sham)/|mean_sham|
            = 'trial_var' -> per run (std_active-std_sham)/std_sham
    Run-level values averaged within subject (matches existing pipeline).
    """
    bt = block_table(trial_csv)
    mcol = f"{feature}_mean"
    scol = f"{feature}_std"
    out = {}
    for (subj, sess, run), g in bt.groupby(["subject", "session", "run"]):
        sham = g[g.gvs == SHAM]
        if sham.empty:
            continue
        sham = sham.iloc[0]
        for _, row in g.iterrows():
            if row.gvs == SHAM:
                continue
            if measure == "frac_amp":
                denom = abs(sham[mcol])
                val = (row[mcol] - sham[mcol]) / denom if denom > 1e-12 else np.nan
            else:  # trial_var
                denom = sham[scol]
                val = (row[scol] - sham[scol]) / denom if denom and denom > 1e-12 else np.nan
            out.setdefault(subj, {}).setdefault(row.gvs, []).append(val)
    # average run-level values within subject -> {subject: {gvs: mean}}
    res = {}
    for subj, d in out.items():
        for gvs, vals in d.items():
            vals = [x for x in vals if not np.isnan(x)]
            if vals:
                res.setdefault(gvs, {})[subj] = float(np.mean(vals))
    return res  # {gvs: {subject: value}}

def part_b():
    # amplitude-type features make sense for frac_amp; variability for trial_var.
    panel = {
        "frac_amp": ["peak_to_peak", "abs_baseline_response", "auc", "temporal_sd"],
        "trial_var": ["peak_to_peak", "mean_level", "slope", "abs_baseline_response"],
    }
    vig_csv = VIG_DIR / "gvs_trial_signal_features.csv"
    task_csv = TASK_DIR / "gvs_trial_signal_features.csv"
    recs = []
    for measure, feats in panel.items():
        for f in feats:
            vmap = derived_subject_vectors(vig_csv, f, measure)
            tmap = derived_subject_vectors(task_csv, f, measure)
            for c in sorted(vmap):
                if c not in tmap:
                    continue
                r = interaction(vmap[c], tmap[c])
                r.update(measure=measure, feature=f, label=FEAT_LABELS[f], active_gvs=c, gvs_label=gvs_display_label(c))
                recs.append(r)
    df = pd.DataFrame(recs)
    df["q"] = bh_fdr(df["p"].values)
    df = df[["measure", "active_gvs", "gvs_label", "feature", "label", "n",
             "vigour_mean_delta", "task_mean_delta",
             "direct_contrast_mean", "direct_contrast_ci95_low",
             "direct_contrast_ci95_high", "direct_contrast_dz", "p", "q"]]
    df.to_csv(OUT / "partB_derived_measures_interaction.csv", index=False)
    return df

# ============================================================================
# FIGURES
# ============================================================================
def heatmap(df, conds, feats, fname, title):
    M = np.full((len(feats), len(conds)), np.nan)
    Q = np.full_like(M, np.nan)
    for _, r in df.iterrows():
        i = feats.index(r.feature); j = conds.index(r.active_gvs)
        M[i, j] = r.direct_contrast_mean; Q[i, j] = r.q
    vmax = np.nanmax(np.abs(M))
    fig, ax = plt.subplots(figsize=(8.5, 6))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([gvs_display_label(c) for c in conds], fontsize=10)
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels([FEAT_LABELS[f] for f in feats], fontsize=10)
    for i in range(len(feats)):
        for j in range(len(conds)):
            if np.isnan(M[i, j]):
                continue
            star = "*" if Q[i, j] < 0.05 else ""
            ax.text(j, i, f"{M[i,j]:.2g}{star}", ha="center", va="center",
                    fontsize=8, fontweight="bold" if star else "normal",
                    color="black")
    ax.set_xlabel("GVS condition", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold")
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("(active-sham vigour) - (active-sham task)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=200)
    plt.close(fig)

def barfig_b(df, fname):
    df = df.copy()
    df["tag"] = df["measure"] + " · " + df["label"] + " · " + df["active_gvs"].str.replace("gvs-", "G")
    df = df.sort_values("direct_contrast_mean")
    colors = ["#c0392b" if q < 0.05 else "#b8c4cf" for q in df["q"]]
    fig, ax = plt.subplots(figsize=(8.5, 0.32 * len(df) + 1))
    ax.barh(range(len(df)), df["direct_contrast_mean"], color=colors)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["tag"], fontsize=7)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("(active-sham vigour) - (active-sham task)", fontsize=11, fontweight="bold")
    ax.set_title("Part B: trial-derived measures — network dissociation\n"
                 "(red = FDR q<0.05)", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=200)
    plt.close(fig)

# ============================================================================
if __name__ == "__main__":
    a, conds, feats = part_a()
    heatmap(a, conds, feats, "partA_feature_interaction_heatmap.png",
            "Part A: GVS network dissociation (active−sham, vigour vs task)\n"
            "Direct paired contrast; * = FDR q<0.05")
    b = part_b()
    barfig_b(b, "partB_derived_measures_interaction.png")

    print("=" * 78)
    print("PART A — feature interaction, FDR-significant (q<0.05):")
    sa = a[a.q < 0.05].sort_values("p")
    print(sa.to_string(index=False) if len(sa) else "  none")
    print("\nTop 8 by p (uncorrected):")
    print(a.sort_values("p").head(8).to_string(index=False))
    print("=" * 78)
    print("PART B — derived measures, FDR-significant (q<0.05):")
    sb = b[b.q < 0.05].sort_values("p")
    print(sb.to_string(index=False) if len(sb) else "  none")
    print("\nTop 8 by p (uncorrected):")
    print(b.sort_values("p").head(8).to_string(index=False))
    print("=" * 78)
    print("Outputs written to", OUT)
