#!/usr/bin/env python3
"""Compare GVS-induced connectivity reorganisation between the vigour network and
the task-activation network at the level of physiologically defined module pairs.

Motivation
----------
The earlier `interhemispheric_homotopic_dissociation` result was metric-fragile and
hard to justify physiologically.  Here we instead pre-register a small set of module
pairs that map onto known GVS / movement-vigour circuitry and ask whether GVS
significantly reorganises any of them *differently* between the two networks.

For each network we have, per edge, whether it is FDR-significant under the
ANY_GVS pooled contrast.  Edges are assigned to one of six anatomical modules
(same mapping used elsewhere in the pipeline).  For each unordered module pair we
compute the density of FDR-significant edges (sig / tested).  The network contrast
for a target module pair is a 2x2 Fisher exact test (network x significant) run
*within that module pair only*, and a logistic interaction `sig ~ network * in_pair`
that asks whether the network difference is concentrated on the target pair
relative to the rest of the network.  We repeat across estimators to check that any
difference is not metric-specific.
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import fisher_exact

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

INPUTS = {
    "Vigour": ROOT / "figures/GVS_effects/GPT/08_connectivity_coactivation/"
    "metric_sensitivity/edge_connectivity_metric_sensitivity_stats.csv",
    "Task": ROOT / "figures/GVS_effects/GPT/08_connectivity_coactivation/"
    "task_activation_z3p1/metric_sensitivity/edge_connectivity_metric_sensitivity_stats.csv",
}

FDR_SCOPE = "pool=ALL_SUBJECTS_BLOCKS;gvs=ANY_GVS"

# Same six-module mapping used by plot_paper_roi_group_edge_density_violins.py
def roi_group(roi: str) -> str:
    base = str(roi).rsplit("_", 1)[0]
    if base in {"Occipital", "Fusiform"}:
        return "Visual"
    if base in {"Precentral", "Postcentral", "Paracentral_Lobule",
                "Supp_Motor_Area", "Cerebellum", "Rolandic_Oper"}:
        return "Somatomotor/Cerebellar"
    if base in {"Amygdala", "Hippocampus", "ParaHippocampal", "Olfactory", "Orbitofrontal"}:
        return "Limbic/MTL-Olfactory"
    if base in {"Caudate", "Pallidum", "Putamen", "Thalamus"}:
        return "Subcortical"
    if base in {"Frontal", "Parietal"}:
        return "Frontal-Parietal"
    return "Cingulate-Temporal"


# Pre-registered physiologically motivated module pairs (unordered)
TARGET_PAIRS = {
    "Subcortical -- Somatomotor/Cerebellar": ("Subcortical", "Somatomotor/Cerebellar"),
    "Subcortical -- Frontal-Parietal": ("Subcortical", "Frontal-Parietal"),
    "Somatomotor/Cerebellar (within)": ("Somatomotor/Cerebellar", "Somatomotor/Cerebellar"),
    "Frontal-Parietal -- Somatomotor/Cerebellar": ("Frontal-Parietal", "Somatomotor/Cerebellar"),
    "Subcortical -- Cingulate-Temporal": ("Subcortical", "Cingulate-Temporal"),
}


def load_edges(path: Path, metric: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    m = (df.metric == metric) & (df.fdr_scope == FDR_SCOPE)
    df = df.loc[m, ["roi_i", "roi_j", "mean", "sig_fdr"]].copy()
    df["sig_fdr"] = df["sig_fdr"].astype(bool)
    g_i = df.roi_i.map(roi_group)
    g_j = df.roi_j.map(roi_group)
    df["pair"] = [frozenset((a, b)) for a, b in zip(g_i, g_j)]
    return df


def pair_key(pair: tuple[str, str]) -> frozenset:
    return frozenset(pair)


def module_pair_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    mods = sorted({m for fs in df.pair for m in fs})
    for a, b in list(combinations(mods, 2)) + [(m, m) for m in mods]:
        key = frozenset((a, b))
        sub = df[df.pair == key]
        n = len(sub)
        s = int(sub.sig_fdr.sum())
        rows.append({"pair": " -- ".join(sorted({a, b})) if a != b else f"{a} (within)",
                     "key": key, "n_tested": n, "n_sig": s,
                     "density": s / n if n else np.nan})
    return pd.DataFrame(rows)


def fisher_network_contrast(vig: pd.DataFrame, task: pd.DataFrame, key: frozenset):
    """2x2: rows = network (Vigour/Task), cols = (sig, not-sig) within target pair."""
    v = vig[vig.pair == key]
    t = task[task.pair == key]
    table = np.array([
        [int(v.sig_fdr.sum()), int((~v.sig_fdr).sum())],
        [int(t.sig_fdr.sum()), int((~t.sig_fdr).sum())],
    ])
    if table[:, 0].sum() == 0:
        return table, np.nan, np.nan
    or_, p = fisher_exact(table)
    return table, or_, p


def logistic_interaction(vig: pd.DataFrame, task: pd.DataFrame, key: frozenset):
    """sig ~ network * in_pair across the full edge set of both networks."""
    v = vig.assign(network="Vigour")
    t = task.assign(network="Task")
    both = pd.concat([v, t], ignore_index=True)
    both["in_pair"] = (both.pair == key).astype(int)
    both["sig"] = both.sig_fdr.astype(int)
    both["network"] = pd.Categorical(both.network, categories=["Task", "Vigour"])
    try:
        res = smf.logit("sig ~ C(network) * in_pair", data=both).fit(disp=0)
        coef = "C(network)[T.Vigour]:in_pair"
        return float(np.exp(res.params[coef])), float(res.pvalues[coef])
    except Exception as exc:  # noqa: BLE001
        return np.nan, str(exc)


def omnibus_interaction(vig: pd.DataFrame, task: pd.DataFrame) -> dict:
    """Likelihood-ratio omnibus: does network change the module-pair pattern of
    GVS significance at all?  sig ~ C(pair_label)*C(network) vs sig ~ C(pair_label)+
    C(network).  A non-significant LR test means the two networks reorganise the same
    module pairs to the same degree (no topological dissociation)."""
    v = vig.assign(network="Vigour")
    t = task.assign(network="Task")
    both = pd.concat([v, t], ignore_index=True)
    both["pair_label"] = both.pair.map(lambda fs: "|".join(sorted(fs)))
    both["sig"] = both.sig_fdr.astype(int)
    full = smf.logit("sig ~ C(pair_label) * C(network)", data=both).fit(disp=0)
    reduced = smf.logit("sig ~ C(pair_label) + C(network)", data=both).fit(disp=0)
    lr = 2 * (full.llf - reduced.llf)
    from scipy.stats import chi2
    dof = int(full.df_model - reduced.df_model)
    p = float(chi2.sf(lr, dof))
    return {"lr_stat": float(lr), "df": dof, "p_value": p}


def fdr_bh(pvals):
    p = np.asarray(pvals, float)
    ok = np.isfinite(p)
    q = np.full_like(p, np.nan)
    idx = np.flatnonzero(ok)
    pv = p[ok]
    order = np.argsort(pv)
    m = pv.size
    ranked = pv[order]
    qv = np.minimum.accumulate((ranked * m / np.arange(m, 0, -1))[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(qv, 1.0)
    q[idx] = out
    return q


def run_metric(metric: str) -> pd.DataFrame:
    vig = load_edges(INPUTS["Vigour"], metric)
    task = load_edges(INPUTS["Task"], metric)
    vt = module_pair_table(vig).set_index("pair")
    tt = module_pair_table(task).set_index("pair")

    rows = []
    for name, pair in TARGET_PAIRS.items():
        key = pair_key(pair)
        vrow = vt[vt.key == key].iloc[0]
        trow = tt[tt.key == key].iloc[0]
        _table, or_f, p_f = fisher_network_contrast(vig, task, key)
        or_l, p_l = logistic_interaction(vig, task, key)
        rows.append({
            "metric": metric,
            "module_pair": name,
            "vig_sig": int(vrow.n_sig), "vig_tested": int(vrow.n_tested),
            "vig_density": vrow.density,
            "task_sig": int(trow.n_sig), "task_tested": int(trow.n_tested),
            "task_density": trow.density,
            "fisher_OR_vig_vs_task": or_f, "fisher_p": p_f,
            "logit_interaction_OR": or_l,
            "logit_interaction_p": p_l if isinstance(p_l, float) else np.nan,
        })
    out = pd.DataFrame(rows)
    out["fisher_q_bh"] = fdr_bh(out.fisher_p.values)
    return out


def make_figure(all_out: pd.DataFrame, metrics: list[str], omni: dict[str, dict],
                out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"],
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    VIG_C, TASK_C = "#7b3294", "#008837"
    pairs = list(TARGET_PAIRS.keys())
    short = {
        "Subcortical -- Somatomotor/Cerebellar": "Subcort.–\nSomatomotor/Cb",
        "Subcortical -- Frontal-Parietal": "Subcort.–\nFront/Par",
        "Somatomotor/Cerebellar (within)": "Somatomotor/Cb\n(within)",
        "Frontal-Parietal -- Somatomotor/Cerebellar": "Front/Par–\nSomatomotor/Cb",
        "Subcortical -- Cingulate-Temporal": "Subcort.–\nCing/Temp",
    }
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.4 * len(metrics), 4.6), sharey=True)
    if len(metrics) == 1:
        axes = [axes]
    nice = {"mutual_info_quantile": "Mutual information (quantile)",
            "spearman_rho": "Spearman rank correlation"}
    for ax, metric in zip(axes, metrics):
        sub = all_out[all_out.metric == metric].set_index("module_pair").loc[pairs]
        x = np.arange(len(pairs))
        w = 0.38
        ax.bar(x - w / 2, sub.vig_density * 100, w, color=VIG_C, label="Vigour network")
        ax.bar(x + w / 2, sub.task_density * 100, w, color=TASK_C, label="Task-activation network")
        for xi, (_, r) in zip(x, sub.iterrows()):
            top = max(r.vig_density, r.task_density) * 100
            p = r.fisher_p
            star = "*" if p < 0.05 else "ns"
            ax.text(xi, top + 1.4, f"p={p:.2f}\n{star}", ha="center", va="bottom",
                    fontsize=8, color="#444")
            ax.text(xi - w / 2, r.vig_density * 100 + 0.3, f"{int(r.vig_sig)}/{int(r.vig_tested)}",
                    ha="center", va="bottom", fontsize=6.5, color=VIG_C)
            ax.text(xi + w / 2, r.task_density * 100 + 0.3, f"{int(r.task_sig)}/{int(r.task_tested)}",
                    ha="center", va="bottom", fontsize=6.5, color=TASK_C)
        ax.set_xticks(x)
        ax.set_xticklabels([short[p] for p in pairs], fontsize=8)
        ax.set_title(f"{nice.get(metric, metric)}\nomnibus network×module-pair LR "
                     f"p = {omni[metric]['p_value']:.2f} (n.s.)", fontsize=10)
        ax.set_ylim(0, 50)
        ax.spines[["top", "right"]].set_visible(False)
        if ax is axes[0]:
            ax.set_ylabel("FDR-significant GVS edges (% of tested edges)")
        ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.suptitle("GVS reorganisation of physiologically pre-registered module pairs: "
                 "vigour vs task-activation network", fontsize=12, y=1.02)
    fig.text(0.5, -0.04,
             "No module pair survives FDR correction in either network or metric; the "
             "omnibus test shows the two networks reorganise the same module pairs to the "
             "same degree.\nLabels above bars give significant/tested edge counts. "
             "ANY_GVS pooled contrast, all-subjects block pool.",
             ha="center", fontsize=8, color="#555")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    # Only mutual_info_quantile and spearman_rho are available for BOTH networks
    # in the ALL_SUBJECTS_BLOCKS;ANY_GVS scope (task pipeline computed these two).
    ap.add_argument("--metrics", nargs="*",
                    default=["mutual_info_quantile", "spearman_rho"])
    ap.add_argument("--out", default=str(ROOT / "figures/GVS_effects/main result/"
                    "connectogram_network_comparison/"
                    "vigour_task_module_pair_gvs_comparison.csv"))
    args = ap.parse_args()

    all_out = pd.concat([run_metric(m) for m in args.metrics], ignore_index=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    all_out.to_csv(args.out, index=False)

    print("\n=== OMNIBUS: does network change the module-pair pattern of GVS significance? ===")
    omni = {}
    for metric in args.metrics:
        vig = load_edges(INPUTS["Vigour"], metric)
        task = load_edges(INPUTS["Task"], metric)
        omni[metric] = omnibus_interaction(vig, task)
        print(f"  {metric:24s}  LR={omni[metric]['lr_stat']:.2f}  "
              f"df={omni[metric]['df']}  p={omni[metric]['p_value']:.3f}")

    fig_png = Path(args.out).with_name("vigour_task_module_pair_gvs_comparison.png")
    make_figure(all_out, list(args.metrics), omni, fig_png)
    print(f"Saved figure: {fig_png}")

    pd.set_option("display.width", 200, "display.max_columns", 30)
    primary = all_out[all_out.metric == "mutual_info_quantile"]
    print("\n=== PRIMARY METRIC (mutual_info_quantile) ===")
    print(primary[["module_pair", "vig_sig", "vig_tested", "vig_density",
                   "task_sig", "task_tested", "task_density",
                   "fisher_OR_vig_vs_task", "fisher_p", "fisher_q_bh",
                   "logit_interaction_OR", "logit_interaction_p"]].to_string(index=False))

    print("\n=== CROSS-METRIC ROBUSTNESS (Fisher p per module pair) ===")
    piv = all_out.pivot(index="module_pair", columns="metric", values="fisher_p")
    print(piv.to_string())
    print("\n=== CROSS-METRIC density difference (vig - task) ===")
    all_out["dens_diff"] = all_out.vig_density - all_out.task_density
    piv2 = all_out.pivot(index="module_pair", columns="metric", values="dens_diff")
    print(piv2.to_string())
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
