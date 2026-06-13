#!/usr/bin/env python3
"""Focused test of the inter-hemispheric / homotopic dissociation between the
GVS-modulated vigour network and the task-activation network.

Hypothesis (pre-specified, single test per contrast):
    GVS-induced connectivity changes in the *task-activation* network are
    preferentially organised around cross-hemispheric (and homotopic /
    callosal-like) edges, whereas in the *vigour* network they are not.

We use the connectogram metric shown in the figures (mutual_info_quantile,
ANY_GVS pool). Each FDR-significant edge is classified as within-hemisphere,
between-hemisphere, or homotopic (same base region, opposite hemisphere).
We then run a single cross-network 2x2 Fisher test (the network x edge-class
interaction) instead of the large multi-family FDR sweep, which is the more
powerful, hypothesis-driven contrast.
"""
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy.stats import fisher_exact
from itertools import combinations

BASE = "/home/zkavian/Master_thesis_final/figures/GVS_effects/GPT/08_connectivity_coactivation"
METRIC = "mutual_info_quantile"
SCOPE = "pool=ALL_SUBJECTS_BLOCKS;gvs=ANY_GVS"

NETWORKS = {
    "vigour": (f"{BASE}/metric_sensitivity/fdr_significant_edge_connectivity_metric_sensitivity.csv",
               "/home/zkavian/Master_thesis_final/Claude_results/group_analyses/analysis3_lme/per_trial_roi_betas_roi_definition.csv"),
    "task":   (f"{BASE}/task_activation_z3p1/metric_sensitivity/fdr_significant_edge_connectivity_metric_sensitivity.csv",
               f"{BASE}/task_activation_z3p1/task_activation_roi_definition.csv"),
}


def base_and_hemi(name):
    if name.endswith("_L"):
        return name[:-2], "L"
    if name.endswith("_R"):
        return name[:-2], "R"
    return name, None


def classify(i, j):
    bi, hi = base_and_hemi(i)
    bj, hj = base_and_hemi(j)
    if hi is None or hj is None:
        return "unknown"
    if hi == hj:
        return "within"
    # opposite hemispheres
    return "homotopic" if bi == bj else "between_nonhomotopic"


def possible_edge_classes(rois):
    """Count possible (not necessarily significant) edges of each class given the ROI list."""
    counts = {"within": 0, "between_nonhomotopic": 0, "homotopic": 0, "unknown": 0}
    for a, b in combinations(sorted(rois), 2):
        counts[classify(a, b)] += 1
    return counts


def load(net):
    edge_path, roi_path = NETWORKS[net]
    df = pd.read_csv(edge_path)
    df = df[(df.metric == METRIC) & (df.fdr_scope == SCOPE)].copy()
    df["cls"] = [classify(i, j) for i, j in zip(df.roi_i, df.roi_j)]
    df["direction"] = df["mean"].apply(lambda m: "improved" if m > 0 else "decreased")
    rois = pd.read_csv(roi_path)["roi_label"].tolist()
    return df, rois


def main():
    data = {}
    for net in NETWORKS:
        df, rois = load(net)
        poss = possible_edge_classes(rois)
        data[net] = dict(df=df, rois=rois, poss=poss)
        print(f"\n=== {net} network: {len(rois)} ROIs, {len(df)} sig edges (mutual_info_quantile) ===")
        for cls in ["within", "between_nonhomotopic", "homotopic"]:
            sig = (df.cls == cls).sum()
            print(f"  {cls:22s}: {sig:3d} sig / {poss[cls]:3d} possible = {100*sig/poss[cls]:.1f}% density")

    # ---- Contrast 1: inter-hemispheric (between+homotopic) vs within, per network, then cross ----
    print("\n" + "=" * 70)
    print("CONTRAST 1: inter-hemispheric fraction of SIGNIFICANT edges (network x class)")
    print("=" * 70)
    tab = {}
    for net in NETWORKS:
        df = data[net]["df"]
        inter = df.cls.isin(["between_nonhomotopic", "homotopic"]).sum()
        within = (df.cls == "within").sum()
        tab[net] = (inter, within)
        print(f"  {net}: inter={inter}, within={within}  -> {100*inter/(inter+within):.1f}% inter-hemispheric")
    table = [[tab["task"][0], tab["task"][1]], [tab["vigour"][0], tab["vigour"][1]]]
    orr, p = fisher_exact(table)
    print(f"  Fisher 2x2 [task vs vigour] x [inter vs within]: OR={orr:.2f}, p={p:.4f}")

    # density-based version (corrects for differing ROI composition)
    print("\n  Density-corrected (sig/possible) inter vs within:")
    for net in NETWORKS:
        df, poss = data[net]["df"], data[net]["poss"]
        inter_sig = df.cls.isin(["between_nonhomotopic", "homotopic"]).sum()
        inter_pos = poss["between_nonhomotopic"] + poss["homotopic"]
        win_sig = (df.cls == "within").sum()
        win_pos = poss["within"]
        orr2, p2 = fisher_exact([[inter_sig, inter_pos - inter_sig], [win_sig, win_pos - win_sig]])
        print(f"    {net}: inter {inter_sig}/{inter_pos}={100*inter_sig/inter_pos:.1f}%  "
              f"within {win_sig}/{win_pos}={100*win_sig/win_pos:.1f}%  OR={orr2:.2f} p={p2:.3f}")

    # ---- Contrast 2: homotopic enrichment (homotopic vs all other), cross-network ----
    print("\n" + "=" * 70)
    print("CONTRAST 2: homotopic edges -- the strongest dissociation")
    print("=" * 70)
    homo = {}
    for net in NETWORKS:
        df, poss = data[net]["df"], data[net]["poss"]
        hs = (df.cls == "homotopic").sum()
        hp = poss["homotopic"]
        homo[net] = (hs, hp)
        edges = df[df.cls == "homotopic"][["roi_i", "roi_j", "direction", "mean"]]
        print(f"  {net}: homotopic {hs}/{hp} possible = {100*hs/hp:.0f}% of homotopic pairs are GVS-significant")
        for _, r in edges.iterrows():
            print(f"      {r.roi_i:18s}-- {r.roi_j:18s} {r.direction:9s} (mean={r['mean']:+.4f})")
    # cross-network Fisher on homotopic significant counts vs homotopic non-sig
    t = [[homo["task"][0], homo["task"][1] - homo["task"][0]],
         [homo["vigour"][0], homo["vigour"][1] - homo["vigour"][0]]]
    orr, p = fisher_exact(t)
    print(f"\n  Fisher [task vs vigour] homotopic sig vs non-sig: OR={orr:.2f}, p={p:.4f}")

    # ---- Which interhemispheric region pairs are shared vs network-specific? ----
    print("\n" + "=" * 70)
    print("CONTRAST 3: cross-hemispheric region-pair overlap")
    print("=" * 70)
    for net in NETWORKS:
        df = data[net]["df"]
        inter_edges = df[df.cls.isin(["between_nonhomotopic", "homotopic"])]
        bases = inter_edges.apply(lambda r: tuple(sorted([base_and_hemi(r.roi_i)[0],
                                                          base_and_hemi(r.roi_j)[0]])), axis=1)
        data[net]["inter_basepairs"] = set(bases)
        print(f"  {net}: {len(inter_edges)} inter-hemispheric sig edges -> "
              f"{len(set(bases))} unique region-pairs")
    shared = data["task"]["inter_basepairs"] & data["vigour"]["inter_basepairs"]
    print(f"  Shared inter-hemispheric region-pairs (task & vigour): {len(shared)}")
    for s in sorted(shared):
        print(f"      {s[0]} <-> {s[1]}")

    # ---- Single interaction model: does homotopic/inter enrichment differ by network? ----
    # Build one row per POSSIBLE edge in each network, outcome = FDR-significant (0/1).
    # Logistic GLM: sig ~ C(cls) * C(network). The interaction Wald test is the single
    # pre-specified test of the topology dissociation (uses every edge -> more power
    # than the 2x2 Fisher on homotopic cells alone).
    print("\n" + "=" * 70)
    print("CONTRAST 4: logistic interaction model  sig ~ C(cls) * C(network)")
    print("=" * 70)
    rows = []
    for net in NETWORKS:
        df, rois = data[net]["df"], data[net]["rois"]
        sig_pairs = {tuple(sorted([a, b])) for a, b in zip(df.roi_i, df.roi_j)}
        for a, b in combinations(sorted(rois), 2):
            cls = classify(a, b)
            if cls == "unknown":
                continue
            rows.append({"network": net, "cls": cls,
                         "sig": int(tuple(sorted([a, b])) in sig_pairs)})
    edf = pd.DataFrame(rows)
    # reference levels: cls=within, network=vigour
    edf["cls"] = pd.Categorical(edf["cls"], ["within", "between_nonhomotopic", "homotopic"])
    edf["network"] = pd.Categorical(edf["network"], ["vigour", "task"])
    m = smf.glm("sig ~ C(cls) * C(network)", data=edf,
                family=__import__("statsmodels.api", fromlist=["families"]).families.Binomial()).fit()
    print(m.summary().tables[1])
    print("\n  Interaction terms (network-specific extra enrichment vs vigour baseline):")
    for term in m.params.index:
        if ":" in term:
            print(f"    {term:55s} OR={np.exp(m.params[term]):.2f}  p={m.pvalues[term]:.3f}")
    # LR test for the whole interaction block
    m0 = smf.glm("sig ~ C(cls) + C(network)", data=edf,
                 family=__import__("statsmodels.api", fromlist=["families"]).families.Binomial()).fit()
    from scipy.stats import chi2
    lr = 2 * (m.llf - m0.llf)
    dfree = m.df_model - m0.df_model
    print(f"\n  Joint LR test of cls x network interaction: chi2={lr:.2f}, df={int(dfree)}, "
          f"p={chi2.sf(lr, dfree):.4f}")
    homo_p = m.pvalues["C(cls)[T.homotopic]:C(network)[T.task]"]
    homo_or = np.exp(m.params["C(cls)[T.homotopic]:C(network)[T.task]"])

    make_figure(data, homo_or, homo_p)


def make_figure(data, homo_or, homo_p):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle, PathPatch, Wedge
    from matplotlib.path import Path as MplPath
    from gvs_connectivity_coactivation.plot_fdr_significant_edge_connectograms import (
        GROUP_COLORS,
        circular_layout,
        draw_chord,
        label_rotation,
        pol2cart,
        roi_display_label,
        roi_group,
    )

    classes = ["within", "between_nonhomotopic", "homotopic"]
    labels = ["Within\nhemisphere", "Between\n(non-homotopic)", "Homotopic\n(callosal)"]
    colors = {"vigour": "#7570b3", "task": "#1b9e77"}
    direction_colors = {"improved": "#d95f02", "decreased": "#2166ac"}
    fig = plt.figure(figsize=(10.2, 4.15), facecolor="white")
    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.03, 0.92],
        left=0.07,
        right=0.985,
        bottom=0.20,
        top=0.95,
        wspace=-0.08,
    )
    ax = fig.add_subplot(grid[0, 0])
    circle_ax = fig.add_subplot(grid[0, 1])

    width = 0.38
    x = np.arange(len(classes))
    bar_tops = {}
    for k, net in enumerate(["vigour", "task"]):
        df, poss = data[net]["df"], data[net]["poss"]
        dens = [100 * (df.cls == c).sum() / poss[c] for c in classes]
        bars = ax.bar(x + (k - 0.5) * width, dens, width,
                      label=f"{'Vigour' if net=='vigour' else 'Task-activation'} network",
                      color=colors[net], edgecolor="black", linewidth=0.6)
        for b, c in zip(bars, classes):
            s, p = (df.cls == c).sum(), poss[c]
            bar_tops[(net, c)] = b.get_height()
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height() + 0.65,
                f"{s}/{p}",
                ha="center",
                va="bottom",
                fontsize=8.8,
            )
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11.5)
    ax.set_ylabel("GVS-significant edge density (%)", fontsize=12.5)
    ax.tick_params(axis="y", labelsize=11.5)
    ax.legend(frameon=False, fontsize=10.5, loc="upper left")
    if homo_p < 0.05:
        homo_x = x[2]
        x1, x2 = homo_x - 0.5 * width, homo_x + 0.5 * width
        y = max(bar_tops[("vigour", "homotopic")], bar_tops[("task", "homotopic")]) + 4.2
        h = 1.2
        ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color="#202020", linewidth=0.9)
        ax.text((x1 + x2) / 2, y + h + 0.55, "*", ha="center", va="bottom", fontsize=16)
    ax.set_ylim(0, 54)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)

    homotopic_edges = {
        net: data[net]["df"].loc[data[net]["df"].cls == "homotopic"].copy()
        for net in ["task", "vigour"]
    }
    for edge_df in homotopic_edges.values():
        edge_df["abs_mean"] = edge_df["mean"].abs()

    homotopic_rois = sorted(
        {
            roi
            for edge_df in homotopic_edges.values()
            for roi in pd.concat([edge_df["roi_i"], edge_df["roi_j"]]).astype(str)
        }
    )
    ordered_rois, angles, sectors = circular_layout(homotopic_rois)
    max_abs = max(float(edge_df["abs_mean"].max()) for edge_df in homotopic_edges.values())
    if not np.isfinite(max_abs) or max_abs <= 0:
        max_abs = 1.0

    def draw_network_chord(ax, theta1, theta2, color, width, alpha, linestyle, control_scale, zorder):
        p1 = pol2cart(theta1, 0.93)
        p2 = pol2cart(theta2, 0.93)
        c1 = p1 * control_scale
        c2 = p2 * control_scale
        path = MplPath([p1, c1, c2, p2], [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4])
        ax.add_patch(
            PathPatch(
                path,
                facecolor="none",
                edgecolor=color,
                linewidth=width,
                alpha=alpha,
                linestyle=linestyle,
                capstyle="round",
                zorder=zorder,
            )
        )

    def draw_homotopic_panel(ax):
        ax.set_aspect("equal")
        ax.axis("off")
        ax.add_patch(Circle((0, 0), 0.91, facecolor="white", edgecolor="#D0D0D0", linewidth=1.0, zorder=0))

        base_pairs = {}
        for roi in ordered_rois:
            base, hemi = base_and_hemi(roi)
            base_pairs.setdefault(base, {})[hemi] = roi
        for pair in base_pairs.values():
            if "L" in pair and "R" in pair:
                draw_chord(ax, angles[pair["L"]], angles[pair["R"]], "#C8C8C8", 0.45, 0.18)

        network_styles = {
            "task": {"linestyle": "solid", "control_scale": 0.16, "zorder": 2.0},
            "vigour": {"linestyle": "--", "control_scale": 0.34, "zorder": 2.4},
        }
        for net in ["task", "vigour"]:
            style = network_styles[net]
            edge_df = homotopic_edges[net]
            for row in edge_df.sort_values("abs_mean").itertuples(index=False):
                scale = min(float(row.abs_mean) / max_abs, 1.0)
                draw_network_chord(
                    ax,
                    angles[row.roi_i],
                    angles[row.roi_j],
                    direction_colors[row.direction],
                    0.65 + 2.0 * scale,
                    0.76 + 0.18 * scale,
                    style["linestyle"],
                    style["control_scale"],
                    style["zorder"],
                )

        for group, theta1, theta2 in sectors:
            ax.add_patch(
                Wedge(
                    (0, 0),
                    1.16,
                    np.rad2deg(theta1),
                    np.rad2deg(theta2),
                    width=0.075,
                    facecolor=GROUP_COLORS[group],
                    edgecolor="white",
                    linewidth=1.2,
                    alpha=0.95,
                    zorder=3,
                )
            )

        for roi in ordered_rois:
            theta = angles[roi]
            group = roi_group(roi)
            p = pol2cart(theta, 1.01)
            ax.scatter([p[0]], [p[1]], s=30, color=GROUP_COLORS[group], edgecolor="white", linewidth=0.7, zorder=5)
            rot, ha = label_rotation(theta)
            ax.text(
                *pol2cart(theta, 1.15),
                roi_display_label(roi),
                rotation=rot,
                rotation_mode="anchor",
                ha=ha,
                va="center",
                fontsize=7.8,
                color="#222222",
            )

        ax.set_xlim(-1.50, 1.50)
        ax.set_ylim(-1.30, 1.30)

    draw_homotopic_panel(circle_ax)

    legend_handles = [
        Line2D([0], [0], color=direction_colors["improved"], linewidth=2.2, label="Strengthened"),
        Line2D([0], [0], color=direction_colors["decreased"], linewidth=2.2, label="Weakened"),
        Line2D([0], [0], color="#202020", linewidth=2.0, linestyle="solid", label="Task-activation"),
        Line2D([0], [0], color="#202020", linewidth=2.0, linestyle="--", label="Vigour"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower right",
        bbox_to_anchor=(0.985, 0.018),
        frameon=False,
        ncol=4,
        fontsize=9.6,
        handlelength=1.8,
        columnspacing=1.0,
    )
    out = "/home/zkavian/Master_thesis_final/figures/GVS_effects/main result/connectogram_network_comparison/interhemispheric_homotopic_dissociation"
    fig.savefig(out + ".png", dpi=200, bbox_inches="tight")
    fig.savefig(out + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure written to {out}.png / .pdf")


if __name__ == "__main__":
    main()
