#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


DEFAULT_MAIN_DIR = Path("figures/med_effects")
DEFAULT_MATCHED_DIR = Path("figures/med_effects/baselines/matched_nonvigour")
DEFAULT_OUT_DIR = Path("figures/med_effects/specificity")


def _to_float_array(values):
    arr = np.asarray(values, dtype=np.float64)
    return arr[np.isfinite(arr)]


def _exact_sign_flip(values):
    values = _to_float_array(values)
    observed = float(np.mean(values)) if values.size else float("nan")
    if values.size == 0:
        return {
            "method": "exact sign-flip test",
            "n_permutations": 0,
            "observed_mean": observed,
            "p_value_two_sided": float("nan"),
            "p_value_greater": float("nan"),
            "p_value_less": float("nan"),
        }
    if values.size > 20:
        raise RuntimeError("Exact sign-flip test is limited to at most 20 subjects")
    n_perm = 1 << values.size
    means = np.empty(n_perm, dtype=np.float64)
    for mask in range(n_perm):
        signs = np.ones(values.size, dtype=np.float64)
        for idx in range(values.size):
            if (mask >> idx) & 1:
                signs[idx] = -1.0
        means[mask] = float(np.mean(signs * values))
    tol = 1e-12
    return {
        "method": "exact sign-flip test over subject-level paired interaction values",
        "n_permutations": int(n_perm),
        "observed_mean": observed,
        "p_value_two_sided": float(np.mean(np.abs(means) >= abs(observed) - tol)),
        "p_value_greater": float(np.mean(means >= observed - tol)),
        "p_value_less": float(np.mean(means <= observed + tol)),
    }


def _one_sample_summary(values):
    values = _to_float_array(values)
    out = {
        "n_subjects": int(values.size),
        "mean": float(np.mean(values)) if values.size else float("nan"),
        "median": float(np.median(values)) if values.size else float("nan"),
        "sd": float(np.std(values, ddof=1)) if values.size > 1 else float("nan"),
        "sem": float(stats.sem(values)) if values.size > 1 else float("nan"),
    }
    if values.size > 1:
        ci_low, ci_high = stats.t.interval(
            0.95,
            values.size - 1,
            loc=out["mean"],
            scale=out["sem"],
        )
        t_result = stats.ttest_1samp(values, 0.0)
        out.update(
            {
                "ci95_low": float(ci_low),
                "ci95_high": float(ci_high),
                "cohen_dz": float(out["mean"] / out["sd"]) if out["sd"] > 0 else float("nan"),
                "paired_t_statistic": float(t_result.statistic),
                "paired_t_p_value_two_sided": float(t_result.pvalue),
            }
        )
        try:
            wilcoxon_two = stats.wilcoxon(values, alternative="two-sided")
            wilcoxon_greater = stats.wilcoxon(values, alternative="greater")
            wilcoxon_less = stats.wilcoxon(values, alternative="less")
            out.update(
                {
                    "wilcoxon_statistic": float(wilcoxon_two.statistic),
                    "wilcoxon_p_value_two_sided": float(wilcoxon_two.pvalue),
                    "wilcoxon_p_value_greater": float(wilcoxon_greater.pvalue),
                    "wilcoxon_p_value_less": float(wilcoxon_less.pvalue),
                }
            )
        except ValueError:
            out.update(
                {
                    "wilcoxon_statistic": float("nan"),
                    "wilcoxon_p_value_two_sided": float("nan"),
                    "wilcoxon_p_value_greater": float("nan"),
                    "wilcoxon_p_value_less": float("nan"),
                }
            )
    else:
        out.update(
            {
                "ci95_low": float("nan"),
                "ci95_high": float("nan"),
                "cohen_dz": float("nan"),
                "paired_t_statistic": float("nan"),
                "paired_t_p_value_two_sided": float("nan"),
                "wilcoxon_statistic": float("nan"),
                "wilcoxon_p_value_two_sided": float("nan"),
                "wilcoxon_p_value_greater": float("nan"),
                "wilcoxon_p_value_less": float("nan"),
            }
        )
    out["exact_sign_flip"] = _exact_sign_flip(values)
    return out


def _read_spectral_subject_values(path):
    df = pd.read_csv(path / "paired_subject_similarity_values.csv")
    required = {"subject", "off_mean_to_other_off", "on_mean_to_other_on"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"{path} missing spectral columns: {', '.join(missing)}")
    out = df.loc[:, ["subject", "off_mean_to_other_off", "on_mean_to_other_on"]].copy()
    out["spectral_on_minus_off"] = out["on_mean_to_other_on"] - out["off_mean_to_other_off"]
    out["spectral_reduction_off_minus_on"] = -out["spectral_on_minus_off"]
    return out


def _read_fc_subject_values(path):
    df = pd.read_csv(path / "intra_vs_between_fc_subject_deltas.csv")
    required = {"subject", "within_minus_between_delta_z"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"{path} missing FC columns: {', '.join(missing)}")
    return df.loc[:, ["subject", "within_minus_between_delta_z"]].copy()


def _load_subject_interactions(main_dir, matched_dir):
    spectral_main = _read_spectral_subject_values(main_dir).rename(
        columns={
            "off_mean_to_other_off": "vigour_spectral_off",
            "on_mean_to_other_on": "vigour_spectral_on",
            "spectral_on_minus_off": "vigour_spectral_on_minus_off",
            "spectral_reduction_off_minus_on": "vigour_spectral_reduction_off_minus_on",
        }
    )
    spectral_matched = _read_spectral_subject_values(matched_dir).rename(
        columns={
            "off_mean_to_other_off": "matched_spectral_off",
            "on_mean_to_other_on": "matched_spectral_on",
            "spectral_on_minus_off": "matched_spectral_on_minus_off",
            "spectral_reduction_off_minus_on": "matched_spectral_reduction_off_minus_on",
        }
    )
    fc_main = _read_fc_subject_values(main_dir).rename(
        columns={"within_minus_between_delta_z": "vigour_fc_intra_minus_between_delta_z"}
    )
    fc_matched = _read_fc_subject_values(matched_dir).rename(
        columns={"within_minus_between_delta_z": "matched_fc_intra_minus_between_delta_z"}
    )
    merged = spectral_main.merge(spectral_matched, on="subject", how="inner")
    merged = merged.merge(fc_main, on="subject", how="inner").merge(fc_matched, on="subject", how="inner")
    merged["spectral_on_minus_off_interaction_vigour_minus_matched"] = (
        merged["vigour_spectral_on_minus_off"] - merged["matched_spectral_on_minus_off"]
    )
    merged["spectral_reduction_interaction_vigour_minus_matched"] = (
        merged["vigour_spectral_reduction_off_minus_on"]
        - merged["matched_spectral_reduction_off_minus_on"]
    )
    merged["fc_interaction_vigour_minus_matched"] = (
        merged["vigour_fc_intra_minus_between_delta_z"]
        - merged["matched_fc_intra_minus_between_delta_z"]
    )
    return merged.sort_values("subject").reset_index(drop=True)


def _interaction_tests(subject_values):
    tests = {
        "spectral_reduction_vigour_minus_matched": {
            "description": "(OFF - ON cross-subject spectral distance reduction in vigour) - matched non-vigour; positive means stronger medication-related distance reduction in vigour.",
            "column": "spectral_reduction_interaction_vigour_minus_matched",
            "directional_alternative": "greater",
        },
        "spectral_on_minus_off_vigour_minus_matched": {
            "description": "(ON - OFF cross-subject spectral distance change in vigour) - matched non-vigour; negative means stronger medication-related distance reduction in vigour.",
            "column": "spectral_on_minus_off_interaction_vigour_minus_matched",
            "directional_alternative": "less",
        },
        "fc_intra_minus_between_vigour_minus_matched": {
            "description": "((ON - OFF intra-ROI FC) - (ON - OFF between-ROI FC)) in vigour - matched non-vigour; positive means stronger within-network medication effect in vigour.",
            "column": "fc_interaction_vigour_minus_matched",
            "directional_alternative": "greater",
        },
    }
    rows = []
    details = {}
    for name, spec in tests.items():
        summary = _one_sample_summary(subject_values[spec["column"]])
        detail = dict(spec)
        detail.update(summary)
        details[name] = detail
        row = {
            "analysis": name,
            "value_column": spec["column"],
            "description": spec["description"],
            "directional_alternative": spec["directional_alternative"],
        }
        for key, value in summary.items():
            if key != "exact_sign_flip":
                row[key] = value
        row["exact_sign_flip_p_two_sided"] = summary["exact_sign_flip"]["p_value_two_sided"]
        row["exact_sign_flip_p_greater"] = summary["exact_sign_flip"]["p_value_greater"]
        row["exact_sign_flip_p_less"] = summary["exact_sign_flip"]["p_value_less"]
        rows.append(row)
    return pd.DataFrame(rows), details


def _plot_direct_interactions(subject_values, tests, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.1), gridspec_kw={"wspace": 0.32})
    panels = [
        (
            axes[0],
            "Cross-subject distance reduction",
            "matched_spectral_reduction_off_minus_on",
            "vigour_spectral_reduction_off_minus_on",
            "spectral_reduction_vigour_minus_matched",
            "OFF - ON distance",
        ),
        (
            axes[1],
            "Intra-minus-between FC contrast",
            "matched_fc_intra_minus_between_delta_z",
            "vigour_fc_intra_minus_between_delta_z",
            "fc_intra_minus_between_vigour_minus_matched",
            "FC contrast, Fisher z",
        ),
    ]
    colors = {"matched": "#767676", "vigour": "#2f6f9f", "mean": "#c82333"}
    rng = np.random.default_rng(0)
    for ax, title, matched_col, vigour_col, test_key, ylabel in panels:
        matched = subject_values[matched_col].to_numpy(dtype=np.float64)
        vigour = subject_values[vigour_col].to_numpy(dtype=np.float64)
        for left, right in zip(matched, vigour):
            ax.plot([0, 1], [left, right], color="#b7b7b7", linewidth=0.8, alpha=0.75, zorder=1)
        ax.scatter(rng.normal(0, 0.025, matched.size), matched, s=28, color=colors["matched"], edgecolor="white", linewidth=0.4, zorder=2)
        ax.scatter(rng.normal(1, 0.025, vigour.size), vigour, s=30, color=colors["vigour"], edgecolor="white", linewidth=0.4, zorder=2)
        means = [float(np.mean(matched)), float(np.mean(vigour))]
        ax.plot([0, 1], means, color=colors["mean"], linewidth=2.1, zorder=3)
        ax.scatter([0, 1], means, color=colors["mean"], s=34, zorder=4)
        p = tests[test_key]["exact_sign_flip"]["p_value_two_sided"]
        ax.set_title(f"{title}\ninteraction p={p:.4f}", fontsize=11)
        ax.set_ylabel(ylabel)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["matched\nnon-vigour", "vigour"])
        ax.axhline(0, color="#555555", linewidth=0.8, linestyle="--", zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=10)
    fig.tight_layout()
    png_path = out_dir / "specificity_interaction_subject_plot.png"
    pdf_path = png_path.with_suffix(".pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return png_path


def _seed_dirs(null_root):
    if null_root is None or not null_root.exists():
        return []
    return sorted([path for path in null_root.iterdir() if path.is_dir()])


def _effect_means(path):
    spectral = _read_spectral_subject_values(path)
    fc = _read_fc_subject_values(path)
    metadata_path = path / "metadata.json"
    seed = None
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        seed = metadata.get("matched_nonvigour_random_state")
    if seed is None:
        digits = "".join(ch for ch in path.name if ch.isdigit())
        seed = int(digits) if digits else None
    return {
        "seed": seed,
        "path": str(path),
        "n_spectral_subjects": int(spectral.shape[0]),
        "n_fc_subjects": int(fc.shape[0]),
        "spectral_reduction_off_minus_on_mean": float(np.mean(spectral["spectral_reduction_off_minus_on"])),
        "fc_intra_minus_between_delta_z_mean": float(np.mean(fc["within_minus_between_delta_z"])),
    }


def _summarize_null(null_dirs, observed, out_dir):
    rows = []
    for path in null_dirs:
        required = [
            path / "paired_subject_similarity_values.csv",
            path / "intra_vs_between_fc_subject_deltas.csv",
        ]
        if all(item.exists() for item in required):
            rows.append(_effect_means(path))
    null_df = pd.DataFrame(rows)
    if null_df.empty:
        return null_df, {}
    observed_spectral = float(np.mean(observed["vigour_spectral_reduction_off_minus_on"]))
    observed_fc = float(np.mean(observed["vigour_fc_intra_minus_between_delta_z"]))
    null_df["spectral_observed_vigour_minus_null"] = (
        observed_spectral - null_df["spectral_reduction_off_minus_on_mean"]
    )
    null_df["fc_observed_vigour_minus_null"] = (
        observed_fc - null_df["fc_intra_minus_between_delta_z_mean"]
    )

    def percentile_and_p(null_values, observed_value):
        values = np.asarray(null_values, dtype=np.float64)
        values = values[np.isfinite(values)]
        percentile = float(100.0 * np.mean(values <= observed_value))
        p_greater = float((np.count_nonzero(values >= observed_value) + 1) / (values.size + 1))
        return percentile, p_greater

    spectral_percentile, spectral_p = percentile_and_p(
        null_df["spectral_reduction_off_minus_on_mean"], observed_spectral
    )
    fc_percentile, fc_p = percentile_and_p(null_df["fc_intra_minus_between_delta_z_mean"], observed_fc)
    summary = {
        "n_null_sets": int(null_df.shape[0]),
        "observed_vigour_spectral_reduction_off_minus_on_mean": observed_spectral,
        "observed_vigour_fc_intra_minus_between_delta_z_mean": observed_fc,
        "spectral_null_percentile_of_observed": spectral_percentile,
        "spectral_empirical_p_greater_or_equal": spectral_p,
        "fc_null_percentile_of_observed": fc_percentile,
        "fc_empirical_p_greater_or_equal": fc_p,
    }
    null_df.to_csv(out_dir / "matched_nonvigour_null_summary.csv", index=False)
    (out_dir / "matched_nonvigour_null_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    _plot_null_summary(null_df, summary, out_dir)
    return null_df, summary


def _plot_null_summary(null_df, summary, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.8), gridspec_kw={"wspace": 0.32})
    panels = [
        (
            axes[0],
            "Matched non-vigour distance reduction",
            "spectral_reduction_off_minus_on_mean",
            "observed_vigour_spectral_reduction_off_minus_on_mean",
            "spectral_empirical_p_greater_or_equal",
            "OFF - ON distance",
        ),
        (
            axes[1],
            "Matched non-vigour FC contrast",
            "fc_intra_minus_between_delta_z_mean",
            "observed_vigour_fc_intra_minus_between_delta_z_mean",
            "fc_empirical_p_greater_or_equal",
            "FC contrast, Fisher z",
        ),
    ]
    for ax, title, column, observed_key, p_key, xlabel in panels:
        values = null_df[column].to_numpy(dtype=np.float64)
        observed = float(summary[observed_key])
        ax.hist(values, bins=min(30, max(6, int(np.sqrt(values.size)))), color="#b6bec8", edgecolor="white")
        ax.axvline(observed, color="#c82333", linewidth=2.2)
        ax.set_title(f"{title}\nempirical p={summary[p_key]:.4f}", fontsize=11)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("null samples")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    png_path = out_dir / "matched_nonvigour_null_effect_distribution.png"
    pdf_path = png_path.with_suffix(".pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def _metadata_for_command(matched_dir):
    metadata_path = matched_dir / "metadata.json"
    if not metadata_path.exists():
        raise RuntimeError(f"Missing metadata needed to run null seeds: {metadata_path}")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _run_null_seeds(args):
    if args.run_null_seeds <= 0:
        return
    metadata = _metadata_for_command(args.matched_dir)
    args.null_root.mkdir(parents=True, exist_ok=True)
    for seed in range(args.seed_start, args.seed_start + args.run_null_seeds):
        seed_dir = args.null_root / f"seed_{seed:04d}"
        complete_outputs = [
            seed_dir / "paired_subject_similarity_values.csv",
            seed_dir / "intra_vs_between_fc_subject_deltas.csv",
        ]
        if all(path.exists() for path in complete_outputs) and not args.force:
            print(f"Skipping existing seed {seed}: {seed_dir}")
            continue
        cmd = [
            sys.executable,
            str(args.med_effects_script),
            "--weight-map",
            str(metadata["weight_map"]),
            "--roi-definition-figure",
            str(metadata["roi_definition_figure"]),
            "--roi-region-table",
            str(metadata["roi_region_table"]),
            "--roi-percentile",
            str(metadata["roi_percentile"]),
            "--min-report-voxels",
            str(metadata["min_report_voxels"]),
            "--beta-root",
            str(metadata["beta_root"]),
            "--voxel-selection",
            "matched-nonvigour",
            "--random-state",
            str(seed),
            "--out-dir",
            str(seed_dir),
        ]
        if metadata.get("split_hemispheres"):
            cmd.append("--split-hemispheres")
            if metadata.get("min_roi_voxels") is not None:
                cmd.extend(["--min-lateralized-voxels", str(metadata["min_roi_voxels"])])
        if metadata.get("session_manifest"):
            cmd.extend(["--session-manifest", str(metadata["session_manifest"])])
        if metadata.get("connectivity_metric"):
            cmd.extend(["--connectivity-metric", str(metadata["connectivity_metric"])])
        if metadata.get("mi_neighbors") is not None:
            cmd.extend(["--mi-neighbors", str(metadata["mi_neighbors"])])
        print("Running", " ".join(cmd))
        subprocess.run(cmd, check=True)


def _write_summary_md(subject_values, tests_table, null_summary, out_dir):
    spectral_row = tests_table.loc[tests_table["analysis"] == "spectral_reduction_vigour_minus_matched"].iloc[0]
    fc_row = tests_table.loc[tests_table["analysis"] == "fc_intra_minus_between_vigour_minus_matched"].iloc[0]
    lines = [
        "# Vigour vs Matched Non-vigour Specificity Tests",
        "",
        "## Direct Paired Interaction",
        "",
        f"- Spectral distance reduction interaction, vigour - matched: mean = {spectral_row['mean']:.6g}, 95% CI [{spectral_row['ci95_low']:.6g}, {spectral_row['ci95_high']:.6g}], paired t p = {spectral_row['paired_t_p_value_two_sided']:.6g}, exact sign-flip two-sided p = {spectral_row['exact_sign_flip_p_two_sided']:.6g}.",
        f"- FC intra-minus-between interaction, vigour - matched: mean = {fc_row['mean']:.6g}, 95% CI [{fc_row['ci95_low']:.6g}, {fc_row['ci95_high']:.6g}], paired t p = {fc_row['paired_t_p_value_two_sided']:.6g}, exact sign-flip two-sided p = {fc_row['exact_sign_flip_p_two_sided']:.6g}.",
        "",
        f"N complete subjects in the merged direct test: {subject_values.shape[0]}.",
    ]
    if null_summary:
        lines.extend(
            [
                "",
                "## Matched Non-vigour Null Sets",
                "",
                f"- Null sets summarized: {null_summary['n_null_sets']}.",
                f"- Observed vigour spectral reduction percentile among null sets: {null_summary['spectral_null_percentile_of_observed']:.2f}%, empirical p(null >= observed) = {null_summary['spectral_empirical_p_greater_or_equal']:.6g}.",
                f"- Observed vigour FC contrast percentile among null sets: {null_summary['fc_null_percentile_of_observed']:.2f}%, empirical p(null >= observed) = {null_summary['fc_empirical_p_greater_or_equal']:.6g}.",
            ]
        )
    (out_dir / "specificity_interaction_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-dir", type=Path, default=DEFAULT_MAIN_DIR)
    parser.add_argument("--matched-dir", type=Path, default=DEFAULT_MATCHED_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--null-root", type=Path, default=None)
    parser.add_argument("--run-null-seeds", type=int, default=0)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--med-effects-script", type=Path, default=Path("med_effects.py"))
    return parser


def main():
    args = build_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.null_root is None:
        args.null_root = args.out_dir / "matched_nonvigour_null"
    _run_null_seeds(args)
    subject_values = _load_subject_interactions(args.main_dir, args.matched_dir)
    tests_table, tests = _interaction_tests(subject_values)
    subject_path = args.out_dir / "specificity_interaction_subject_values.csv"
    tests_path = args.out_dir / "specificity_interaction_tests.csv"
    json_path = args.out_dir / "specificity_interaction_tests.json"
    subject_values.to_csv(subject_path, index=False)
    tests_table.to_csv(tests_path, index=False)
    json_path.write_text(json.dumps({"tests": tests}, indent=2), encoding="utf-8")
    figure_path = _plot_direct_interactions(subject_values, tests, args.out_dir)
    null_df, null_summary = _summarize_null(_seed_dirs(args.null_root), subject_values, args.out_dir)
    _write_summary_md(subject_values, tests_table, null_summary, args.out_dir)
    print(f"Saved {subject_path}")
    print(f"Saved {tests_path}")
    print(f"Saved {json_path}")
    print(f"Saved {figure_path}")
    print(f"Saved {figure_path.with_suffix('.pdf')}")
    if not null_df.empty:
        print(f"Saved {args.out_dir / 'matched_nonvigour_null_summary.csv'}")
        print(f"Saved {args.out_dir / 'matched_nonvigour_null_summary.json'}")
        print(f"Saved {args.out_dir / 'matched_nonvigour_null_effect_distribution.png'}")
    print(f"Saved {args.out_dir / 'specificity_interaction_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
