#!/usr/bin/env python3
"""Paired vigour-vs-task contrast for the strongest GVS feature effects.

For the top rows in
figures/gvs_vigour_vs_task_significance_comparison/
vigour_only_fdr_significant_gvs_features.csv, this tests the within-subject
difference:

    vigour active-sham feature change - task active-sham feature change

and plots paired active-sham bars plus the paired difference bars.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "figures" / "gvs_vigour_vs_task_significance_comparison"
TOP_FEATURES = OUT_DIR / "vigour_only_fdr_significant_gvs_features.csv"
VIGOUR_STATS = (
    ROOT / "figures" / "gvs_projection_features_vs_sham"
    / "gvs_vs_sham_signal_feature_stats.csv"
)
TASK_STATS = (
    ROOT / "figures" / "gvs_task_map_bold_features_vs_sham"
    / "gvs_vs_sham_signal_feature_stats.csv"
)
VIGOUR_REPORT = (
    ROOT / "figures" / "gvs_projection_features_vs_sham"
    / "gvs_vs_sham_signal_feature_report.txt"
)
VIGOUR_SUBJECTS = (
    ROOT / "figures" / "gvs_projection_features_vs_sham"
    / "gvs_vs_sham_subject_signal_feature_pairs.csv"
)
TASK_SUBJECTS = (
    ROOT / "figures" / "gvs_task_map_bold_features_vs_sham"
    / "gvs_vs_sham_subject_signal_feature_pairs.csv"
)
FEATURE_ORDER = [
    "mean_level",
    "auc",
    "peak_to_peak",
    "baseline_to_peak",
    "baseline_to_trough",
    "abs_baseline_response",
    "early_late_change",
    "slope",
    "temporal_sd",
]
FEATURE_LABELS = {
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


def fdr_bh(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    q = np.full(p.shape, np.nan, dtype=float)
    ok = np.isfinite(p)
    if not ok.any():
        return q
    idx = np.flatnonzero(ok)
    pv = p[ok]
    order = np.argsort(pv)
    ranked = pv[order]
    m = len(ranked)
    q_ranked = ranked * m / (np.arange(m) + 1)
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    out = np.empty_like(q_ranked)
    out[order] = np.clip(q_ranked, 0.0, 1.0)
    q[idx] = out
    return q


def exact_signflip_p(values: np.ndarray) -> tuple[float, int]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return np.nan, 0
    signs = np.where(
        ((np.arange(2**n)[:, None] >> np.arange(n)[None, :]) & 1) == 1,
        1.0,
        -1.0,
    )
    obs = abs(x.mean())
    null = np.abs(signs @ x) / n
    return float(np.mean(null >= obs - 1e-12)), n


def mean_ci(values: np.ndarray) -> tuple[float, float, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    mean = float(x.mean())
    if len(x) < 2:
        return mean, np.nan, np.nan
    sem = stats.sem(x)
    half = stats.t.ppf(0.975, len(x) - 1) * sem
    return mean, float(mean - half), float(mean + half)


def cohen_dz(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan
    sd = x.std(ddof=1)
    return float(x.mean() / sd) if sd > 0 else np.nan


def vigour_source_note() -> str:
    if not VIGOUR_REPORT.exists():
        return "Vigour source: corrected HTML-mask weighted BOLD"
    for line in VIGOUR_REPORT.read_text(encoding="utf-8").splitlines():
        if "selected_active_voxels=" not in line:
            continue
        selected = line.split("selected_active_voxels=", 1)[1].split(";", 1)[0].strip()
        return f"Vigour source: corrected HTML-mask weighted BOLD ({selected} HTML-selected active voxels)"
    return "Vigour source: corrected HTML-mask weighted BOLD"


def add_within_gvs_fdr(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["q_within_gvs_fdr"] = np.nan
    for _, index in out.groupby("active_gvs").groups.items():
        out.loc[index, "q_within_gvs_fdr"] = fdr_bh(out.loc[index, "p_perm"].to_numpy())
    return out


def format_feature_hits(hits: pd.DataFrame, vigour_q_col: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "gvs": hits["active_gvs"],
            "feature": hits["label_vigour"],
            "feature_id": hits["feature"],
            "vigour_mean_active_minus_sham": hits["mean_active_minus_sham_vigour"],
            "vigour_p_perm": hits["p_perm_vigour"],
            "vigour_q_perm_fdr": hits[vigour_q_col],
            "vigour_q_global_fdr": hits["q_perm_fdr_vigour"],
            "task_mean_active_minus_sham": hits["mean_active_minus_sham_task"],
            "task_p_perm": hits["p_perm_task"],
            "task_q_perm_fdr": hits["q_perm_fdr_task"],
        }
    ).reset_index(drop=True)


def rebuild_vigour_only_features() -> pd.DataFrame:
    vigour = add_within_gvs_fdr(pd.read_csv(VIGOUR_STATS))
    task = add_within_gvs_fdr(pd.read_csv(TASK_STATS))
    keep_cols = [
        "active_gvs",
        "feature",
        "label",
        "mean_active_minus_sham",
        "p_perm",
        "q_perm_fdr",
        "q_within_gvs_fdr",
    ]
    merged = vigour[keep_cols].merge(
        task[keep_cols],
        on=["active_gvs", "feature"],
        how="inner",
        suffixes=("_vigour", "_task"),
        validate="one_to_one",
    )

    fdr_hits = merged.loc[
        merged["q_within_gvs_fdr_vigour"].lt(0.05) & merged["q_perm_fdr_task"].ge(0.05)
    ].copy()
    fdr_hits = fdr_hits.sort_values(
        ["q_within_gvs_fdr_vigour", "p_perm_vigour", "active_gvs", "feature"],
        kind="mergesort",
    )
    fdr_out = format_feature_hits(fdr_hits, "q_within_gvs_fdr_vigour")

    nominal_hits = merged.loc[
        merged["p_perm_vigour"].lt(0.05) & merged["q_perm_fdr_task"].ge(0.05)
    ].copy()
    nominal_hits = nominal_hits.sort_values(
        ["p_perm_vigour", "active_gvs", "feature"],
        kind="mergesort",
    )
    nominal_out = format_feature_hits(nominal_hits, "q_perm_fdr_vigour")

    outputs = {
        "vigour_only_fdr_significant_gvs_features": (
            fdr_out,
            "Vigour-only within-GVS FDR-significant GVS-feature pairs",
            "blue = vigour within-GVS q < 0.05 and task global q >= 0.05",
        ),
        "vigour_only_significant_gvs_features": (
            nominal_out,
            "Vigour-only nominally significant GVS-feature pairs",
            "blue = vigour p < 0.05 and task global q >= 0.05",
        ),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for stem, (out, title, criterion_note) in outputs.items():
        csv_path = OUT_DIR / f"{stem}.csv"
        out.to_csv(csv_path, index=False)
        plot_feature_matrix(out, OUT_DIR / f"{stem}_matrix.png", title, criterion_note)
        plot_feature_table(out, OUT_DIR / f"{stem}_table.png", title)
    return fdr_out


def plot_feature_matrix(features: pd.DataFrame, out_png: Path, title: str, criterion_note: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    active_codes = [f"gvs-{idx:02d}" for idx in range(2, 10)]
    matrix = np.zeros((len(active_codes), len(FEATURE_ORDER)), dtype=float)
    hit_pairs = set(zip(features["gvs"], features["feature_id"]))
    for row, gvs in enumerate(active_codes):
        for col, feature in enumerate(FEATURE_ORDER):
            if (gvs, feature) in hit_pairs:
                matrix[row, col] = 1.0

    fig, ax = plt.subplots(figsize=(12.8, 4.8))
    cmap = ListedColormap(["#f1f4f7", "#2c7fb8"])
    ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(FEATURE_ORDER)))
    ax.set_xticklabels([FEATURE_LABELS[name] for name in FEATURE_ORDER], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(active_codes)))
    ax.set_yticklabels(active_codes)
    ax.set_xlabel("Feature", fontweight="bold")
    ax.set_ylabel("Active GVS condition", fontweight="bold")
    ax.set_title(
        f"{title}\n"
        f"{criterion_note}\n"
        f"{vigour_source_note()}",
        fontweight="bold",
    )
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            text = "YES" if matrix[row, col] else "-"
            color = "white" if matrix[row, col] else "#6f7f8f"
            ax.text(col, row, text, ha="center", va="center", color=color, fontweight="bold" if matrix[row, col] else "normal")
    ax.set_xticks(np.arange(-0.5, len(FEATURE_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(active_codes), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def plot_feature_table(features: pd.DataFrame, out_png: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    display = features.copy()
    if display.empty:
        display = pd.DataFrame(
            [
                {
                    "gvs": "None",
                    "feature": "No vigour-only FDR-significant pairs",
                    "vigour_mean_active_minus_sham": np.nan,
                    "vigour_q_perm_fdr": np.nan,
                    "task_mean_active_minus_sham": np.nan,
                    "task_q_perm_fdr": np.nan,
                }
            ]
        )
    display = display.assign(
        vigour_delta=lambda df: df["vigour_mean_active_minus_sham"].map(lambda x: f"{x:.4g}" if np.isfinite(x) else ""),
        vigour_q=lambda df: df["vigour_q_perm_fdr"].map(lambda x: f"{x:.3g}" if np.isfinite(x) else ""),
        task_delta=lambda df: df["task_mean_active_minus_sham"].map(lambda x: f"{x:.4g}" if np.isfinite(x) else ""),
        task_q=lambda df: df["task_q_perm_fdr"].map(lambda x: f"{x:.3g}" if np.isfinite(x) else ""),
    )
    table_data = display[["gvs", "feature", "vigour_delta", "vigour_q", "task_delta", "task_q"]]
    table_data.columns = ["GVS", "Feature", "Vigour delta", "Vigour q", "Task delta", "Task q"]

    fig, ax = plt.subplots(figsize=(10.2, max(1.8, 0.58 * len(table_data) + 1.1)))
    ax.axis("off")
    table = ax.table(cellText=table_data.values, colLabels=table_data.columns, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.35)
    for (row, _), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#2c7fb8")
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("#f6f8fa" if row % 2 else "white")
    ax.set_title(
        f"{title}\n{vigour_source_note()}",
        fontweight="bold",
        pad=10,
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def load_top_features(n_top: int = 3) -> pd.DataFrame:
    top = pd.read_csv(TOP_FEATURES)
    top = top.sort_values(["vigour_q_perm_fdr", "vigour_p_perm"], kind="mergesort")
    return top.head(n_top).reset_index(drop=True)


def aligned_subject_data(row: pd.Series, vigour: pd.DataFrame, task: pd.DataFrame) -> pd.DataFrame:
    active_gvs = row["gvs"]
    feature = row["feature_id"]
    common = {"active_gvs": active_gvs, "sham_gvs": "gvs-01", "feature": feature}
    vig = vigour.loc[
        vigour[list(common)].eq(pd.Series(common)).all(axis=1),
        ["subject", "delta_active_minus_sham"],
    ].rename(columns={"delta_active_minus_sham": "vigour_delta"})
    tas = task.loc[
        task[list(common)].eq(pd.Series(common)).all(axis=1),
        ["subject", "delta_active_minus_sham"],
    ].rename(columns={"delta_active_minus_sham": "task_delta"})
    merged = vig.merge(tas, on="subject", how="inner", validate="one_to_one")
    if merged.empty:
        raise ValueError(f"No aligned subjects for {active_gvs} / {feature}")
    merged["paired_difference"] = merged["vigour_delta"] - merged["task_delta"]
    merged["gvs"] = active_gvs
    merged["feature"] = feature
    merged["label"] = row["feature"]
    return merged


def paired_stats(top: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    vigour = pd.read_csv(VIGOUR_SUBJECTS)
    task = pd.read_csv(TASK_SUBJECTS)
    subject_frames = []
    rows = []
    for _, feature_row in top.iterrows():
        aligned = aligned_subject_data(feature_row, vigour, task)
        subject_frames.append(aligned)
        diff = aligned["paired_difference"].to_numpy()
        vig = aligned["vigour_delta"].to_numpy()
        tas = aligned["task_delta"].to_numpy()
        p_perm, n = exact_signflip_p(diff)
        t_res = stats.ttest_1samp(diff, 0.0, nan_policy="omit")
        try:
            wilcoxon_p = float(stats.wilcoxon(diff).pvalue)
        except ValueError:
            wilcoxon_p = np.nan
        diff_mean, diff_low, diff_high = mean_ci(diff)
        vig_mean, vig_low, vig_high = mean_ci(vig)
        task_mean, task_low, task_high = mean_ci(tas)
        rows.append(
            {
                "gvs": feature_row["gvs"],
                "feature": feature_row["feature_id"],
                "label": feature_row["feature"],
                "n_subjects": n,
                "vigour_mean_active_minus_sham": vig_mean,
                "vigour_ci95_low": vig_low,
                "vigour_ci95_high": vig_high,
                "task_mean_active_minus_sham": task_mean,
                "task_ci95_low": task_low,
                "task_ci95_high": task_high,
                "paired_difference_mean": diff_mean,
                "paired_difference_ci95_low": diff_low,
                "paired_difference_ci95_high": diff_high,
                "paired_difference_cohen_dz": cohen_dz(diff),
                "paired_difference_t": float(t_res.statistic),
                "paired_difference_p_t": float(t_res.pvalue),
                "paired_difference_p_wilcoxon": wilcoxon_p,
                "paired_difference_p_perm": p_perm,
                "source_vigour_q_perm_fdr": feature_row["vigour_q_perm_fdr"],
                "source_vigour_q_global_fdr": feature_row.get("vigour_q_global_fdr", np.nan),
                "source_task_q_perm_fdr": feature_row["task_q_perm_fdr"],
            }
        )
    stats_df = pd.DataFrame(rows)
    stats_df["paired_difference_q_perm_fdr"] = fdr_bh(
        stats_df["paired_difference_p_perm"].to_numpy()
    )
    return stats_df, pd.concat(subject_frames, ignore_index=True)


def fmt_p(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    if value < 0.001:
        return f"{value:.1e}"
    return f"{value:.3f}"


def padded_limits(values: np.ndarray) -> tuple[float, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return -1.0, 1.0
    lo = min(0.0, float(np.min(x)))
    hi = max(0.0, float(np.max(x)))
    span = hi - lo
    if span == 0:
        span = max(abs(hi), 1.0)
    pad = 0.18 * span
    return lo - pad, hi + pad


def plot_bars(stats_df: pd.DataFrame, subjects: pd.DataFrame, out_png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    vigour_c = "#7b3294"
    task_c = "#008837"
    diff_c = "#4c78a8"
    point_c = "#303030"

    n_cols = len(stats_df)
    fig, axes = plt.subplots(2, n_cols, figsize=(3.75 * n_cols, 6.0))
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    for col, row in enumerate(stats_df.itertuples(index=False)):
        sub = subjects.loc[
            (subjects["gvs"] == row.gvs) & (subjects["feature"] == row.feature)
        ].sort_values("subject")
        title = f"{row.gvs.upper()}\n{row.label}"

        ax = axes[0, col]
        means = np.array(
            [row.vigour_mean_active_minus_sham, row.task_mean_active_minus_sham]
        )
        lows = np.array([row.vigour_ci95_low, row.task_ci95_low])
        highs = np.array([row.vigour_ci95_high, row.task_ci95_high])
        err = np.vstack([means - lows, highs - means])
        ax.bar([0, 1], means, yerr=err, width=0.64, color=[vigour_c, task_c], capsize=4)
        jitter = np.linspace(-0.045, 0.045, len(sub))
        for j, (_, srow) in zip(jitter, sub.iterrows()):
            ax.plot(
                [0 + j, 1 + j],
                [srow.vigour_delta, srow.task_delta],
                color="#9a9a9a",
                alpha=0.45,
                linewidth=0.7,
                zorder=1,
            )
        ax.scatter(np.full(len(sub), 0) + jitter, sub["vigour_delta"], s=13, color=point_c, zorder=3)
        ax.scatter(np.full(len(sub), 1) + jitter, sub["task_delta"], s=13, color=point_c, zorder=3)
        ax.axhline(0, color="#222", linewidth=0.8)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Vigour", "Task"])
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_ylim(padded_limits(np.r_[sub["vigour_delta"], sub["task_delta"], lows, highs]))
        if col == 0:
            ax.set_ylabel("Active - sham feature change")
        ax.spines[["top", "right"]].set_visible(False)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))

        ax = axes[1, col]
        diff = sub["paired_difference"].to_numpy()
        mean = row.paired_difference_mean
        low = row.paired_difference_ci95_low
        high = row.paired_difference_ci95_high
        ax.bar(
            [0],
            [mean],
            yerr=np.array([[mean - low], [high - mean]]),
            width=0.58,
            color=diff_c,
            capsize=4,
        )
        ax.scatter(jitter * 2.0, diff, s=18, color=point_c, zorder=3)
        ax.axhline(0, color="#222", linewidth=0.8)
        ax.set_xticks([0])
        ax.set_xticklabels(["Vigour - task"])
        ax.set_ylim(padded_limits(np.r_[diff, low, high]))
        ax.text(
            0.5,
            0.96,
            f"mean={mean:.3g}\np_perm={fmt_p(row.paired_difference_p_perm)}; "
            f"q={fmt_p(row.paired_difference_q_perm_fdr)}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=8,
        )
        if col == 0:
            ax.set_ylabel("Paired difference")
        ax.spines[["top", "right"]].set_visible(False)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))

    fig.suptitle(
        "Three strongest vigour-only GVS feature pairs: paired vigour-vs-task contrast\n"
        f"{vigour_source_note()}",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_png, dpi=220)
    fig.savefig(out_png.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    rebuild_vigour_only_features()
    top = load_top_features(n_top=3)
    stats_df, subjects = paired_stats(top)
    stats_path = OUT_DIR / "vigour_task_paired_feature_difference_stats.csv"
    subjects_path = OUT_DIR / "vigour_task_paired_feature_difference_subjects.csv"
    fig_path = OUT_DIR / "vigour_task_paired_feature_difference_bars.png"
    stats_df.to_csv(stats_path, index=False)
    subjects.to_csv(subjects_path, index=False)
    plot_bars(stats_df, subjects, fig_path)

    print(stats_df.to_string(index=False, float_format=lambda value: f"{value:.6g}"))
    print(f"\nSaved: {stats_path}")
    print(f"Saved: {subjects_path}")
    print(f"Saved: {fig_path}")
    print(f"Saved: {fig_path.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
