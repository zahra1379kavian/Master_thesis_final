#!/usr/bin/env python3
"""Density-normalized comparison of two FDR-significant connectogram networks."""

from __future__ import annotations

import argparse
import itertools
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, norm

from plot_fdr_significant_edge_connectograms import GROUP_LABELS, GROUP_ORDER, roi_group


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCE_ROOT = ROOT / "figures" / "GVS_effects" / "GPT" / "08_connectivity_coactivation"

DEFAULT_MAIN_SIG = SOURCE_ROOT / "metric_sensitivity" / "fdr_significant_edge_connectivity_metric_sensitivity.csv"
DEFAULT_MAIN_STATS = SOURCE_ROOT / "metric_sensitivity" / "edge_connectivity_metric_sensitivity_stats.csv"
DEFAULT_MAIN_ROIS = ROOT / "Claude_results" / "group_analyses" / "analysis3_lme" / "per_trial_roi_betas_roi_definition.csv"
DEFAULT_TASK_SIG = SOURCE_ROOT / "task_activation_z3p1" / "metric_sensitivity" / "fdr_significant_edge_connectivity_metric_sensitivity.csv"
DEFAULT_TASK_STATS = SOURCE_ROOT / "task_activation_z3p1" / "metric_sensitivity" / "edge_connectivity_metric_sensitivity_stats.csv"
DEFAULT_TASK_ROIS = SOURCE_ROOT / "task_activation_z3p1" / "task_activation_roi_definition.csv"
DEFAULT_OUT_DIR = ROOT / "figures" / "GVS_effects" / "main result" / "connectogram_network_comparison"

DEFAULT_METRIC = "mutual_info_quantile"
DEFAULT_ANALYSIS_VIEW = "all_subjects_block_pool"
DEFAULT_FDR_SCOPE = "pool=ALL_SUBJECTS_BLOCKS;gvs=ANY_GVS"
NETWORK_ORDER = ("task_activation_z3p1", "main_result")
DIRECTIONS = ("any", "improved", "decreased")


@dataclass(frozen=True)
class NetworkSpec:
    name: str
    sig_csv: Path
    stats_csv: Path
    roi_csv: Path


@dataclass
class NetworkData:
    spec: NetworkSpec
    rois: list[str]
    sig: pd.DataFrame
    stats: pd.DataFrame


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def roi_hemi(roi: str) -> str:
    text = str(roi)
    if text.endswith("_L"):
        return "L"
    if text.endswith("_R"):
        return "R"
    return "NA"


def roi_base(roi: str) -> str:
    text = str(roi)
    hemi = roi_hemi(text)
    if hemi in {"L", "R"}:
        return text.rsplit("_", 1)[0]
    return text


def display_group(group: str) -> str:
    return GROUP_LABELS.get(group, group)


def ordered_group_pair(group_i: str, group_j: str) -> tuple[str, str]:
    group_index = {group: idx for idx, group in enumerate(GROUP_ORDER)}
    key_i = group_index.get(group_i, len(group_index))
    key_j = group_index.get(group_j, len(group_index))
    if (key_i, group_i) <= (key_j, group_j):
        return group_i, group_j
    return group_j, group_i


def canonical_edge(roi_i: str, roi_j: str) -> str:
    left, right = sorted([str(roi_i), str(roi_j)])
    return f"{left}--{right}"


def filter_scope(df: pd.DataFrame, metric: str, analysis_view: str, fdr_scope: str) -> pd.DataFrame:
    out = df.loc[
        df["metric"].astype(str).eq(metric)
        & df["analysis_view"].astype(str).eq(analysis_view)
        & df["fdr_scope"].astype(str).eq(fdr_scope)
    ].copy()
    out["mean"] = pd.to_numeric(out["mean"], errors="coerce")
    out["abs_mean"] = out["mean"].abs()
    out["edge"] = [canonical_edge(i, j) for i, j in zip(out["roi_i"], out["roi_j"], strict=True)]
    out["direction"] = np.where(out["mean"] > 0, "improved", np.where(out["mean"] < 0, "decreased", "zero"))
    return out


def load_network(spec: NetworkSpec, metric: str, analysis_view: str, fdr_scope: str) -> NetworkData:
    rois = pd.read_csv(spec.roi_csv)["roi_label"].astype(str).tolist()
    sig = filter_scope(pd.read_csv(spec.sig_csv, low_memory=False), metric, analysis_view, fdr_scope)
    if "sig_fdr" in sig.columns:
        sig = sig.loc[as_bool(sig["sig_fdr"])].copy()
    stats = filter_scope(pd.read_csv(spec.stats_csv, low_memory=False), metric, analysis_view, fdr_scope)
    return NetworkData(spec=spec, rois=rois, sig=sig, stats=stats)


def build_possible_edges(rois: list[str]) -> pd.DataFrame:
    rows = []
    for roi_i, roi_j in itertools.combinations(rois, 2):
        hemi_i = roi_hemi(roi_i)
        hemi_j = roi_hemi(roi_j)
        group_i = roi_group(roi_i)
        group_j = roi_group(roi_j)
        first_group, second_group = ordered_group_pair(group_i, group_j)
        within = hemi_i == hemi_j and hemi_i in {"L", "R"}
        between = hemi_i in {"L", "R"} and hemi_j in {"L", "R"} and hemi_i != hemi_j
        homotopic = between and roi_base(roi_i) == roi_base(roi_j)
        rows.append(
            {
                "edge": canonical_edge(roi_i, roi_j),
                "roi_i": roi_i,
                "roi_j": roi_j,
                "roi_i_base": roi_base(roi_i),
                "roi_j_base": roi_base(roi_j),
                "hemi_i": hemi_i,
                "hemi_j": hemi_j,
                "group_i": display_group(group_i),
                "group_j": display_group(group_j),
                "group_pair": f"{display_group(first_group)} -- {display_group(second_group)}",
                "hemisphere_class": "within_hemisphere" if within else "between_hemisphere" if between else "other",
                "hemisphere_subclass": f"within_{hemi_i.lower()}_hemisphere" if within else "between_hemisphere" if between else "other",
                "homotopic": bool(homotopic),
                "between_nonhomotopic": bool(between and not homotopic),
                "anatomical_class": "same_anatomical_group" if group_i == group_j else "cross_anatomical_group",
            }
        )
    return pd.DataFrame(rows)


def profile_network(data: NetworkData, rois: list[str], roi_set: str) -> dict[str, object]:
    roi_set_values = set(rois)
    possible = build_possible_edges(rois)
    sig = data.sig.loc[data.sig["roi_i"].isin(roi_set_values) & data.sig["roi_j"].isin(roi_set_values)].copy()
    stats = data.stats.loc[data.stats["roi_i"].isin(roi_set_values) & data.stats["roi_j"].isin(roi_set_values)].copy()

    sig = sig.merge(possible.drop(columns=["roi_i", "roi_j"]), on="edge", how="left", validate="one_to_one")
    stats = stats.merge(possible.drop(columns=["roi_i", "roi_j"]), on="edge", how="left", validate="one_to_one")
    if sig["hemisphere_class"].isna().any():
        bad = ", ".join(sig.loc[sig["hemisphere_class"].isna(), "edge"].head(5))
        raise RuntimeError(f"{data.spec.name}: significant edges missing from possible edge set: {bad}")

    return {
        "network": data.spec.name,
        "roi_set": roi_set,
        "rois": rois,
        "possible": possible,
        "sig": sig,
        "stats": stats,
    }


def direction_sig(sig: pd.DataFrame, direction: str) -> pd.DataFrame:
    if direction == "any":
        return sig
    return sig.loc[sig["direction"].eq(direction)].copy()


def summarize_effects(sig: pd.DataFrame) -> dict[str, float]:
    if sig.empty:
        return {
            "mean_signed_effect": np.nan,
            "mean_abs_effect": np.nan,
            "median_abs_effect": np.nan,
            "sum_abs_effect": 0.0,
        }
    return {
        "mean_signed_effect": float(sig["mean"].mean()),
        "mean_abs_effect": float(sig["abs_mean"].mean()),
        "median_abs_effect": float(sig["abs_mean"].median()),
        "sum_abs_effect": float(sig["abs_mean"].sum()),
    }


def density_row(profile: dict[str, object], feature_family: str, feature: str, possible: pd.DataFrame, sig: pd.DataFrame, direction: str) -> dict[str, object]:
    n_possible = int(possible.shape[0])
    n_sig = int(sig.shape[0])
    row = {
        "network": profile["network"],
        "roi_set": profile["roi_set"],
        "feature_family": feature_family,
        "feature": feature,
        "direction": direction,
        "n_rois": len(profile["rois"]),
        "n_possible_edges": n_possible,
        "n_significant_edges": n_sig,
        "density": float(n_sig / n_possible) if n_possible else np.nan,
        "density_per_100_edges": float(100 * n_sig / n_possible) if n_possible else np.nan,
    }
    row.update(summarize_effects(sig))
    return row


def network_summary_rows(profiles: list[dict[str, object]]) -> pd.DataFrame:
    rows = []
    for profile in profiles:
        rois = list(profile["rois"])
        n_left = sum(roi_hemi(roi) == "L" for roi in rois)
        n_right = sum(roi_hemi(roi) == "R" for roi in rois)
        for direction in DIRECTIONS:
            sig = direction_sig(profile["sig"], direction)
            row = density_row(profile, "network", "all_edges", profile["possible"], sig, direction)
            row.update({"n_left_rois": n_left, "n_right_rois": n_right})
            rows.append(row)
    return pd.DataFrame(rows)


def hemisphere_density_rows(profiles: list[dict[str, object]]) -> pd.DataFrame:
    selectors = {
        "within_hemisphere": lambda df: df["hemisphere_class"].eq("within_hemisphere"),
        "within_l_hemisphere": lambda df: df["hemisphere_subclass"].eq("within_l_hemisphere"),
        "within_r_hemisphere": lambda df: df["hemisphere_subclass"].eq("within_r_hemisphere"),
        "between_hemisphere": lambda df: df["hemisphere_class"].eq("between_hemisphere"),
        "homotopic": lambda df: df["homotopic"],
        "between_nonhomotopic": lambda df: df["between_nonhomotopic"],
    }
    rows = []
    for profile in profiles:
        possible = profile["possible"]
        sig_all = profile["sig"]
        for feature, selector in selectors.items():
            possible_mask = selector(possible)
            for direction in DIRECTIONS:
                sig = direction_sig(sig_all, direction)
                sig_mask = selector(sig)
                rows.append(density_row(profile, "hemisphere", feature, possible.loc[possible_mask], sig.loc[sig_mask], direction))
    return pd.DataFrame(rows)


def anatomical_density_rows(profiles: list[dict[str, object]]) -> pd.DataFrame:
    rows = []
    for profile in profiles:
        possible = profile["possible"]
        sig_all = profile["sig"]
        for feature in ["same_anatomical_group", "cross_anatomical_group"]:
            for direction in DIRECTIONS:
                sig = direction_sig(sig_all, direction)
                rows.append(
                    density_row(
                        profile,
                        "anatomical_class",
                        feature,
                        possible.loc[possible["anatomical_class"].eq(feature)],
                        sig.loc[sig["anatomical_class"].eq(feature)],
                        direction,
                    )
                )
    return pd.DataFrame(rows)


def group_pair_density_rows(profiles: list[dict[str, object]]) -> pd.DataFrame:
    rows = []
    for profile in profiles:
        possible = profile["possible"]
        sig_all = profile["sig"]
        for group_pair in sorted(possible["group_pair"].unique()):
            possible_pair = possible.loc[possible["group_pair"].eq(group_pair)]
            for direction in DIRECTIONS:
                sig = direction_sig(sig_all, direction)
                sig_pair = sig.loc[sig["group_pair"].eq(group_pair)]
                rows.append(density_row(profile, "group_pair", group_pair, possible_pair, sig_pair, direction))
    return pd.DataFrame(rows)


def node_involvement_rows(profiles: list[dict[str, object]]) -> pd.DataFrame:
    rows = []
    for profile in profiles:
        rois = list(profile["rois"])
        possible_degree = max(len(rois) - 1, 0)
        sig_all = profile["sig"]
        for roi in rois:
            for direction in DIRECTIONS:
                sig = direction_sig(sig_all, direction)
                incident = sig.loc[sig["roi_i"].eq(roi) | sig["roi_j"].eq(roi)]
                row = {
                    "network": profile["network"],
                    "roi_set": profile["roi_set"],
                    "roi": roi,
                    "roi_base": roi_base(roi),
                    "hemisphere": roi_hemi(roi),
                    "group": display_group(roi_group(roi)),
                    "direction": direction,
                    "n_possible_incident_edges": possible_degree,
                    "n_significant_incident_edges": int(incident.shape[0]),
                    "normalized_significant_degree": float(incident.shape[0] / possible_degree) if possible_degree else np.nan,
                    "normalized_degree_per_100_edges": float(100 * incident.shape[0] / possible_degree) if possible_degree else np.nan,
                }
                row.update(summarize_effects(incident))
                rows.append(row)
    return pd.DataFrame(rows)


def odds_ratio_ci(a: int, b: int, c: int, d: int) -> tuple[float, float]:
    cells = np.array([a, b, c, d], dtype=float)
    if np.any(cells == 0):
        cells += 0.5
    aa, bb, cc, dd = cells
    log_or = math.log((aa * dd) / (bb * cc))
    se = math.sqrt(1 / aa + 1 / bb + 1 / cc + 1 / dd)
    return float(math.exp(log_or - 1.96 * se)), float(math.exp(log_or + 1.96 * se))


def log_odds_ratio_and_se(a: int, b: int, c: int, d: int) -> tuple[float, float]:
    cells = np.array([a, b, c, d], dtype=float)
    if np.any(cells == 0):
        cells += 0.5
    aa, bb, cc, dd = cells
    log_or = math.log((aa * dd) / (bb * cc))
    se = math.sqrt(1 / aa + 1 / bb + 1 / cc + 1 / dd)
    return float(log_or), float(se)


def density_ratio(numerator_sig: int, numerator_possible: int, denominator_sig: int, denominator_possible: int) -> float:
    numerator_density = numerator_sig / numerator_possible if numerator_possible else np.nan
    denominator_density = denominator_sig / denominator_possible if denominator_possible else np.nan
    if not np.isfinite(numerator_density) or not np.isfinite(denominator_density):
        return np.nan
    if denominator_density == 0:
        return np.inf if numerator_density > 0 else np.nan
    return float(numerator_density / denominator_density)


def fisher_compare(a_sig: int, a_possible: int, b_sig: int, b_possible: int) -> dict[str, float]:
    a_not = int(a_possible - a_sig)
    b_not = int(b_possible - b_sig)
    if min(a_sig, a_not, b_sig, b_not) < 0:
        raise RuntimeError("Significant edge count cannot exceed possible edge count")
    if a_possible == 0 or b_possible == 0:
        return {
            "density_a": np.nan,
            "density_b": np.nan,
            "density_difference_a_minus_b": np.nan,
            "density_ratio_a_vs_b": np.nan,
            "odds_ratio_a_vs_b": np.nan,
            "odds_ratio_ci95_low": np.nan,
            "odds_ratio_ci95_high": np.nan,
            "fisher_p": np.nan,
        }
    result = fisher_exact([[a_sig, a_not], [b_sig, b_not]], alternative="two-sided")
    ci_low, ci_high = odds_ratio_ci(a_sig, a_not, b_sig, b_not)
    density_a = a_sig / a_possible
    density_b = b_sig / b_possible
    return {
        "density_a": float(density_a),
        "density_b": float(density_b),
        "density_difference_a_minus_b": float(density_a - density_b),
        "density_ratio_a_vs_b": density_ratio(a_sig, a_possible, b_sig, b_possible),
        "odds_ratio_a_vs_b": float(result.statistic),
        "odds_ratio_ci95_low": ci_low,
        "odds_ratio_ci95_high": ci_high,
        "fisher_p": float(result.pvalue),
    }


def bh_q_values(p_values: pd.Series) -> pd.Series:
    p = pd.to_numeric(p_values, errors="coerce").to_numpy(dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    finite = np.isfinite(p)
    if not finite.any():
        return pd.Series(q, index=p_values.index)
    finite_idx = np.where(finite)[0]
    order = finite_idx[np.argsort(p[finite])]
    ranked_p = p[order]
    m = ranked_p.size
    ranked_q = ranked_p * m / np.arange(1, m + 1)
    ranked_q = np.minimum.accumulate(ranked_q[::-1])[::-1]
    ranked_q = np.clip(ranked_q, 0, 1)
    q[order] = ranked_q
    return pd.Series(q, index=p_values.index)


def compare_density_table(density: pd.DataFrame, feature_family: str) -> pd.DataFrame:
    rows = []
    table = density.loc[density["feature_family"].eq(feature_family)].copy()
    key_cols = ["roi_set", "feature", "direction"]
    for key, group in table.groupby(key_cols, sort=False, dropna=False):
        roi_set, feature, direction = key
        if set(group["network"]) != set(NETWORK_ORDER):
            continue
        task = group.loc[group["network"].eq("task_activation_z3p1")].iloc[0]
        main = group.loc[group["network"].eq("main_result")].iloc[0]
        comp = fisher_compare(
            int(task["n_significant_edges"]),
            int(task["n_possible_edges"]),
            int(main["n_significant_edges"]),
            int(main["n_possible_edges"]),
        )
        rows.append(
            {
                "roi_set": roi_set,
                "feature_family": feature_family,
                "feature": feature,
                "direction": direction,
                "task_significant_edges": int(task["n_significant_edges"]),
                "task_possible_edges": int(task["n_possible_edges"]),
                "task_density": comp["density_a"],
                "main_significant_edges": int(main["n_significant_edges"]),
                "main_possible_edges": int(main["n_possible_edges"]),
                "main_density": comp["density_b"],
                "density_difference_task_minus_main": comp["density_difference_a_minus_b"],
                "density_ratio_task_vs_main": comp["density_ratio_a_vs_b"],
                "odds_ratio_task_vs_main": comp["odds_ratio_a_vs_b"],
                "odds_ratio_ci95_low": comp["odds_ratio_ci95_low"],
                "odds_ratio_ci95_high": comp["odds_ratio_ci95_high"],
                "fisher_p": comp["fisher_p"],
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["q_bh"] = bh_q_values(out["fisher_p"])
        out["significant_q05"] = out["q_bh"].lt(0.05)
    return out


def hemisphere_enrichment_rows(hemisphere_density: pd.DataFrame) -> pd.DataFrame:
    return enrichment_rows(
        hemisphere_density,
        numerator_feature="between_hemisphere",
        denominator_feature="within_hemisphere",
        contrast="between_vs_within_hemisphere",
        numerator_label="between",
        denominator_label="within",
    )


def enrichment_rows(
    density: pd.DataFrame,
    numerator_feature: str,
    denominator_feature: str,
    contrast: str,
    numerator_label: str,
    denominator_label: str,
) -> pd.DataFrame:
    rows = []
    base = density.loc[density["feature"].isin([numerator_feature, denominator_feature])].copy()
    for key, group in base.groupby(["network", "roi_set", "direction"], sort=False, dropna=False):
        network, roi_set, direction = key
        if not {numerator_feature, denominator_feature}.issubset(set(group["feature"])):
            continue
        numerator = group.loc[group["feature"].eq(numerator_feature)].iloc[0]
        denominator = group.loc[group["feature"].eq(denominator_feature)].iloc[0]
        comp = fisher_compare(
            int(numerator["n_significant_edges"]),
            int(numerator["n_possible_edges"]),
            int(denominator["n_significant_edges"]),
            int(denominator["n_possible_edges"]),
        )
        rows.append(
            {
                "network": network,
                "roi_set": roi_set,
                "contrast": contrast,
                "direction": direction,
                "numerator_feature": numerator_feature,
                "denominator_feature": denominator_feature,
                f"{numerator_label}_significant_edges": int(numerator["n_significant_edges"]),
                f"{numerator_label}_possible_edges": int(numerator["n_possible_edges"]),
                f"{numerator_label}_density": comp["density_a"],
                f"{denominator_label}_significant_edges": int(denominator["n_significant_edges"]),
                f"{denominator_label}_possible_edges": int(denominator["n_possible_edges"]),
                f"{denominator_label}_density": comp["density_b"],
                f"{numerator_label}_minus_{denominator_label}_density": comp["density_difference_a_minus_b"],
                f"{numerator_label}_{denominator_label}_density_ratio": comp["density_ratio_a_vs_b"],
                f"{numerator_label}_vs_{denominator_label}_odds_ratio": comp["odds_ratio_a_vs_b"],
                "odds_ratio_ci95_low": comp["odds_ratio_ci95_low"],
                "odds_ratio_ci95_high": comp["odds_ratio_ci95_high"],
                "fisher_p": comp["fisher_p"],
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["q_bh"] = bh_q_values(out["fisher_p"])
        out["significant_q05"] = out["q_bh"].lt(0.05)
    return out


def enrichment_network_comparison(
    density: pd.DataFrame,
    numerator_feature: str,
    denominator_feature: str,
    contrast: str,
) -> pd.DataFrame:
    rows = []
    base = density.loc[density["feature"].isin([numerator_feature, denominator_feature])].copy()
    for key, group in base.groupby(["roi_set", "direction"], sort=False, dropna=False):
        roi_set, direction = key
        networks = set(group["network"])
        if set(NETWORK_ORDER) != networks:
            continue
        values = {}
        for network in NETWORK_ORDER:
            network_group = group.loc[group["network"].eq(network)]
            if not {numerator_feature, denominator_feature}.issubset(set(network_group["feature"])):
                continue
            numerator = network_group.loc[network_group["feature"].eq(numerator_feature)].iloc[0]
            denominator = network_group.loc[network_group["feature"].eq(denominator_feature)].iloc[0]
            a = int(numerator["n_significant_edges"])
            b = int(numerator["n_possible_edges"] - numerator["n_significant_edges"])
            c = int(denominator["n_significant_edges"])
            d = int(denominator["n_possible_edges"] - denominator["n_significant_edges"])
            log_or, se = log_odds_ratio_and_se(a, b, c, d)
            values[network] = {
                "numerator_sig": a,
                "numerator_possible": int(numerator["n_possible_edges"]),
                "numerator_density": float(numerator["density"]),
                "denominator_sig": c,
                "denominator_possible": int(denominator["n_possible_edges"]),
                "denominator_density": float(denominator["density"]),
                "log_or": log_or,
                "se": se,
                "odds_ratio": float(math.exp(log_or)),
            }
        if set(values) != set(NETWORK_ORDER):
            continue
        task = values["task_activation_z3p1"]
        main = values["main_result"]
        diff = task["log_or"] - main["log_or"]
        se_diff = math.sqrt(task["se"] ** 2 + main["se"] ** 2)
        z_value = diff / se_diff if se_diff > 0 else np.nan
        p_value = float(2 * norm.sf(abs(z_value))) if np.isfinite(z_value) else np.nan
        rows.append(
            {
                "roi_set": roi_set,
                "feature": contrast,
                "numerator_feature": numerator_feature,
                "denominator_feature": denominator_feature,
                "direction": direction,
                "task_numerator_significant_edges": task["numerator_sig"],
                "task_numerator_possible_edges": task["numerator_possible"],
                "task_numerator_density": task["numerator_density"],
                "task_denominator_significant_edges": task["denominator_sig"],
                "task_denominator_possible_edges": task["denominator_possible"],
                "task_denominator_density": task["denominator_density"],
                "task_enrichment_odds_ratio": task["odds_ratio"],
                "main_numerator_significant_edges": main["numerator_sig"],
                "main_numerator_possible_edges": main["numerator_possible"],
                "main_numerator_density": main["numerator_density"],
                "main_denominator_significant_edges": main["denominator_sig"],
                "main_denominator_possible_edges": main["denominator_possible"],
                "main_denominator_density": main["denominator_density"],
                "main_enrichment_odds_ratio": main["odds_ratio"],
                "log_odds_ratio_difference_task_minus_main": diff,
                "ratio_of_odds_ratios_task_vs_main": float(math.exp(diff)),
                "ratio_of_odds_ratios_ci95_low": float(math.exp(diff - 1.96 * se_diff)),
                "ratio_of_odds_ratios_ci95_high": float(math.exp(diff + 1.96 * se_diff)),
                "wald_z": z_value,
                "wald_p": p_value,
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["q_bh"] = bh_q_values(out["wald_p"])
        out["significant_q05"] = out["q_bh"].lt(0.05)
    return out


def node_comparison_rows(profiles: list[dict[str, object]]) -> pd.DataFrame:
    rows = []
    common_profiles = [profile for profile in profiles if profile["roi_set"] == "common_roi_set"]
    profiles_by_network = {profile["network"]: profile for profile in common_profiles}
    task = profiles_by_network["task_activation_z3p1"]
    main = profiles_by_network["main_result"]
    common_rois = sorted(set(task["rois"]) & set(main["rois"]))
    possible_incident = max(len(common_rois) - 1, 0)
    for roi in common_rois:
        for direction in DIRECTIONS:
            task_sig = direction_sig(task["sig"], direction)
            main_sig = direction_sig(main["sig"], direction)
            task_count = int((task_sig["roi_i"].eq(roi) | task_sig["roi_j"].eq(roi)).sum())
            main_count = int((main_sig["roi_i"].eq(roi) | main_sig["roi_j"].eq(roi)).sum())
            comp = fisher_compare(task_count, possible_incident, main_count, possible_incident)
            rows.append(
                {
                    "roi_set": "common_roi_set",
                    "roi": roi,
                    "roi_base": roi_base(roi),
                    "hemisphere": roi_hemi(roi),
                    "group": display_group(roi_group(roi)),
                    "direction": direction,
                    "task_significant_incident_edges": task_count,
                    "task_possible_incident_edges": possible_incident,
                    "task_normalized_degree": comp["density_a"],
                    "main_significant_incident_edges": main_count,
                    "main_possible_incident_edges": possible_incident,
                    "main_normalized_degree": comp["density_b"],
                    "normalized_degree_difference_task_minus_main": comp["density_difference_a_minus_b"],
                    "normalized_degree_ratio_task_vs_main": comp["density_ratio_a_vs_b"],
                    "odds_ratio_task_vs_main": comp["odds_ratio_a_vs_b"],
                    "odds_ratio_ci95_low": comp["odds_ratio_ci95_low"],
                    "odds_ratio_ci95_high": comp["odds_ratio_ci95_high"],
                    "fisher_p": comp["fisher_p"],
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["q_bh"] = bh_q_values(out["fisher_p"])
        out["significant_q05"] = out["q_bh"].lt(0.05)
    return out


def corr_or_nan(left: pd.Series, right: pd.Series, method: str = "pearson") -> float:
    values = pd.concat([left, right], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if values.shape[0] < 2:
        return np.nan
    if values.iloc[:, 0].nunique() < 2 or values.iloc[:, 1].nunique() < 2:
        return np.nan
    return float(values.iloc[:, 0].corr(values.iloc[:, 1], method=method))


def cosine_or_nan(left: pd.Series, right: pd.Series) -> float:
    values = pd.concat([left, right], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return np.nan
    a = values.iloc[:, 0].to_numpy(dtype=float)
    b = values.iloc[:, 1].to_numpy(dtype=float)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return np.nan
    return float(np.dot(a, b) / denom)


def edge_set_similarity(profiles: list[dict[str, object]]) -> pd.DataFrame:
    profiles_by_network = {profile["network"]: profile for profile in profiles if profile["roi_set"] == "common_roi_set"}
    task = profiles_by_network["task_activation_z3p1"]
    main = profiles_by_network["main_result"]
    common_edges = sorted(set(task["possible"]["edge"]) & set(main["possible"]["edge"]))
    rows = []
    for direction in DIRECTIONS:
        task_sig = direction_sig(task["sig"], direction).set_index("edge")
        main_sig = direction_sig(main["sig"], direction).set_index("edge")
        task_edges = set(task_sig.index) & set(common_edges)
        main_edges = set(main_sig.index) & set(common_edges)
        overlap = task_edges & main_edges
        union = task_edges | main_edges
        row = {
            "roi_set": "common_roi_set",
            "direction": direction,
            "n_common_rois": len(task["rois"]),
            "n_common_possible_edges": len(common_edges),
            "task_significant_edges": len(task_edges),
            "main_significant_edges": len(main_edges),
            "overlap_edges": len(overlap),
            "task_only_edges": len(task_edges - main_edges),
            "main_only_edges": len(main_edges - task_edges),
            "union_edges": len(union),
            "jaccard_similarity": float(len(overlap) / len(union)) if union else np.nan,
            "overlap_coefficient": float(len(overlap) / min(len(task_edges), len(main_edges))) if min(len(task_edges), len(main_edges)) else np.nan,
        }
        if direction == "any" and overlap:
            merged_overlap = (
                task_sig.loc[sorted(overlap), ["mean", "direction"]]
                .rename(columns={"mean": "task_mean", "direction": "task_direction"})
                .join(main_sig.loc[sorted(overlap), ["mean", "direction"]].rename(columns={"mean": "main_mean", "direction": "main_direction"}))
            )
            row["sign_concordant_overlap_edges"] = int(merged_overlap["task_direction"].eq(merged_overlap["main_direction"]).sum())
            row["sign_concordance_overlap"] = float(merged_overlap["task_direction"].eq(merged_overlap["main_direction"]).mean())
            row["overlap_effect_pearson"] = corr_or_nan(merged_overlap["task_mean"], merged_overlap["main_mean"], "pearson")
            row["overlap_effect_spearman"] = corr_or_nan(merged_overlap["task_mean"], merged_overlap["main_mean"], "spearman")
        else:
            row["sign_concordant_overlap_edges"] = np.nan
            row["sign_concordance_overlap"] = np.nan
            row["overlap_effect_pearson"] = np.nan
            row["overlap_effect_spearman"] = np.nan

        task_union = pd.Series(0.0, index=sorted(union), dtype=float)
        main_union = pd.Series(0.0, index=sorted(union), dtype=float)
        if union:
            task_union.update(task["sig"].set_index("edge")["mean"])
            main_union.update(main["sig"].set_index("edge")["mean"])
        row["union_signed_effect_cosine"] = cosine_or_nan(task_union, main_union)
        row["union_signed_effect_pearson"] = corr_or_nan(task_union, main_union, "pearson")
        row["union_signed_effect_spearman"] = corr_or_nan(task_union, main_union, "spearman")

        if direction == "any":
            task_stats = task["stats"].set_index("edge")["mean"].reindex(common_edges)
            main_stats = main["stats"].set_index("edge")["mean"].reindex(common_edges)
            row["common_possible_effect_pearson"] = corr_or_nan(task_stats, main_stats, "pearson")
            row["common_possible_effect_spearman"] = corr_or_nan(task_stats, main_stats, "spearman")
            row["common_possible_signed_effect_cosine"] = cosine_or_nan(task_stats, main_stats)
            row["missing_task_common_stats"] = int(task_stats.isna().sum())
            row["missing_main_common_stats"] = int(main_stats.isna().sum())
        else:
            row["common_possible_effect_pearson"] = np.nan
            row["common_possible_effect_spearman"] = np.nan
            row["common_possible_signed_effect_cosine"] = np.nan
            row["missing_task_common_stats"] = np.nan
            row["missing_main_common_stats"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def distribution_similarity(group_density: pd.DataFrame) -> pd.DataFrame:
    rows = []
    table = group_density.loc[group_density["feature_family"].eq("group_pair")].copy()
    for key, group in table.groupby(["roi_set", "direction"], sort=False, dropna=False):
        roi_set, direction = key
        pivot_density = group.pivot(index="feature", columns="network", values="density").reindex(columns=NETWORK_ORDER)
        pivot_counts = group.pivot(index="feature", columns="network", values="n_significant_edges").reindex(columns=NETWORK_ORDER).fillna(0)
        task_density = pivot_density["task_activation_z3p1"]
        main_density = pivot_density["main_result"]
        task_counts = pivot_counts["task_activation_z3p1"].to_numpy(dtype=float)
        main_counts = pivot_counts["main_result"].to_numpy(dtype=float)
        if task_counts.sum() > 0:
            task_p = task_counts / task_counts.sum()
        else:
            task_p = task_counts
        if main_counts.sum() > 0:
            main_p = main_counts / main_counts.sum()
        else:
            main_p = main_counts
        mix = 0.5 * (task_p + main_p)
        valid_task = task_p > 0
        valid_main = main_p > 0
        jsd = 0.0
        if mix.sum() > 0:
            jsd = 0.5 * np.sum(task_p[valid_task] * np.log2(task_p[valid_task] / mix[valid_task]))
            jsd += 0.5 * np.sum(main_p[valid_main] * np.log2(main_p[valid_main] / mix[valid_main]))
        rows.append(
            {
                "roi_set": roi_set,
                "direction": direction,
                "n_group_pairs": int(pivot_density.shape[0]),
                "density_pearson": corr_or_nan(task_density, main_density, "pearson"),
                "density_spearman": corr_or_nan(task_density, main_density, "spearman"),
                "density_cosine": cosine_or_nan(task_density.fillna(0), main_density.fillna(0)),
                "count_distribution_jsd_bits": float(jsd),
            }
        )
    return pd.DataFrame(rows)


def significant_changes(*comparison_tables: tuple[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for family, table in comparison_tables:
        if table.empty or "q_bh" not in table.columns:
            continue
        sig = table.loc[table["q_bh"].lt(0.05)].copy()
        for row in sig.to_dict("records"):
            feature = row.get("feature", row.get("roi", ""))
            direction = row.get("direction", "")
            density_diff = row.get("density_difference_task_minus_main", row.get("normalized_degree_difference_task_minus_main", np.nan))
            if pd.isna(density_diff):
                density_diff = row.get("log_odds_ratio_difference_task_minus_main", np.nan)
            rows.append(
                {
                    "comparison_family": family,
                    "roi_set": row.get("roi_set", ""),
                    "feature": feature,
                    "direction": direction,
                    "task_density_or_normalized_degree": row.get(
                        "task_density",
                        row.get("task_normalized_degree", row.get("task_numerator_density", np.nan)),
                    ),
                    "main_density_or_normalized_degree": row.get(
                        "main_density",
                        row.get("main_normalized_degree", row.get("main_numerator_density", np.nan)),
                    ),
                    "difference_task_minus_main": density_diff,
                    "higher_density_network": "task_activation_z3p1" if pd.notna(density_diff) and density_diff > 0 else "main_result",
                    "odds_ratio_task_vs_main": row.get(
                        "odds_ratio_task_vs_main",
                        row.get("ratio_of_odds_ratios_task_vs_main", np.nan),
                    ),
                    "odds_ratio_ci95_low": row.get(
                        "odds_ratio_ci95_low",
                        row.get("ratio_of_odds_ratios_ci95_low", np.nan),
                    ),
                    "odds_ratio_ci95_high": row.get(
                        "odds_ratio_ci95_high",
                        row.get("ratio_of_odds_ratios_ci95_high", np.nan),
                    ),
                    "fisher_p": row.get("fisher_p", row.get("wald_p", np.nan)),
                    "q_bh": row.get("q_bh", np.nan),
                }
            )
    return pd.DataFrame(rows).sort_values(["q_bh", "fisher_p"], na_position="last") if rows else pd.DataFrame(
        columns=[
            "comparison_family",
            "roi_set",
            "feature",
            "direction",
            "task_density_or_normalized_degree",
            "main_density_or_normalized_degree",
            "difference_task_minus_main",
            "higher_density_network",
            "odds_ratio_task_vs_main",
            "odds_ratio_ci95_low",
            "odds_ratio_ci95_high",
            "fisher_p",
            "q_bh",
        ]
    )


def nominal_changes(*comparison_tables: tuple[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for family, table in comparison_tables:
        if table.empty:
            continue
        p_col = "fisher_p" if "fisher_p" in table.columns else "wald_p" if "wald_p" in table.columns else None
        if p_col is None:
            continue
        nominal = table.loc[pd.to_numeric(table[p_col], errors="coerce").lt(0.05)].copy()
        for row in nominal.to_dict("records"):
            feature = row.get("feature", row.get("roi", ""))
            direction = row.get("direction", "")
            density_diff = row.get("density_difference_task_minus_main", row.get("normalized_degree_difference_task_minus_main", np.nan))
            if pd.isna(density_diff):
                density_diff = row.get("log_odds_ratio_difference_task_minus_main", np.nan)
            rows.append(
                {
                    "comparison_family": family,
                    "roi_set": row.get("roi_set", ""),
                    "feature": feature,
                    "direction": direction,
                    "task_density_or_normalized_degree": row.get(
                        "task_density",
                        row.get("task_normalized_degree", row.get("task_numerator_density", np.nan)),
                    ),
                    "main_density_or_normalized_degree": row.get(
                        "main_density",
                        row.get("main_normalized_degree", row.get("main_numerator_density", np.nan)),
                    ),
                    "difference_task_minus_main": density_diff,
                    "odds_or_ratio_of_odds_ratios_task_vs_main": row.get(
                        "odds_ratio_task_vs_main",
                        row.get("ratio_of_odds_ratios_task_vs_main", np.nan),
                    ),
                    "p_value": row.get(p_col, np.nan),
                    "q_bh": row.get("q_bh", np.nan),
                    "survives_bh_q05": row.get("significant_q05", False),
                }
            )
    return pd.DataFrame(rows).sort_values(["p_value"], na_position="last") if rows else pd.DataFrame(
        columns=[
            "comparison_family",
            "roi_set",
            "feature",
            "direction",
            "task_density_or_normalized_degree",
            "main_density_or_normalized_degree",
            "difference_task_minus_main",
            "odds_or_ratio_of_odds_ratios_task_vs_main",
            "p_value",
            "q_bh",
            "survives_bh_q05",
        ]
    )


def write_report(out_dir: Path, tables: dict[str, pd.DataFrame], args: argparse.Namespace) -> None:
    summary = tables["network_summary_density"]
    hemi_enrichment = tables["hemisphere_enrichment"]
    sig_changes = tables["significant_network_differences_q05"]
    nominal = tables["nominal_network_differences_p05"]
    similarity = tables["edge_set_similarity"]

    def fmt_pct(value: float) -> str:
        return "NA" if pd.isna(value) else f"{100 * value:.2f}%"

    any_full = summary.loc[summary["roi_set"].eq("full_network") & summary["direction"].eq("any")]
    lines = [
        "# Connectogram Network Density Comparison",
        "",
        f"Metric: `{args.metric}`",
        f"Analysis view: `{args.analysis_view}`",
        f"FDR scope: `{args.fdr_scope}`",
        "",
        "Density is `FDR-significant edges / possible ROI pairs`. Fisher exact tests compare significant vs non-significant possible edges; q-values are Benjamini-Hochberg corrected within each comparison table. Treat edge-level p-values as descriptive because edges are not independent observations.",
        "",
        "## Inputs",
        "",
        f"- Main significant edges: `{args.main_sig}`",
        f"- Main all-edge stats: `{args.main_stats}`",
        f"- Main ROI definition: `{args.main_rois}`",
        f"- Task significant edges: `{args.task_sig}`",
        f"- Task all-edge stats: `{args.task_stats}`",
        f"- Task ROI definition: `{args.task_rois}`",
        "",
        "## Overall Full-Network Density",
        "",
        "| network | ROIs | possible edges | significant edges | density | improved | decreased |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for network in NETWORK_ORDER:
        rows = summary.loc[summary["roi_set"].eq("full_network") & summary["network"].eq(network)]
        any_row = rows.loc[rows["direction"].eq("any")].iloc[0]
        improved = rows.loc[rows["direction"].eq("improved"), "n_significant_edges"].iloc[0]
        decreased = rows.loc[rows["direction"].eq("decreased"), "n_significant_edges"].iloc[0]
        lines.append(
            f"| {network} | {int(any_row.n_rois)} | {int(any_row.n_possible_edges)} | "
            f"{int(any_row.n_significant_edges)} | {fmt_pct(any_row.density)} | {int(improved)} | {int(decreased)} |"
        )

    lines.extend(["", "## Between-vs-Within Hemisphere Enrichment", ""])
    lines.append("| network | ROI set | direction | between density | within density | density ratio | OR | p | q |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
    for row in hemi_enrichment.sort_values(["roi_set", "network", "direction"]).itertuples(index=False):
        lines.append(
            f"| {row.network} | {row.roi_set} | {row.direction} | {fmt_pct(row.between_density)} | "
            f"{fmt_pct(row.within_density)} | {row.between_within_density_ratio:.3g} | "
            f"{row.between_vs_within_odds_ratio:.3g} | {row.fisher_p:.3g} | {row.q_bh:.3g} |"
        )

    lines.extend(["", "## Significant Task-vs-Main Density Differences", ""])
    if sig_changes.empty:
        lines.append("No network-density differences survived BH q < 0.05 in the generated comparison tables.")
    else:
        lines.append("| family | ROI set | feature | direction | task density | main density | OR | p | q |")
        lines.append("|---|---|---|---|---:|---:|---:|---:|---:|")
        for row in sig_changes.head(30).itertuples(index=False):
            lines.append(
                f"| {row.comparison_family} | {row.roi_set} | {row.feature} | {row.direction} | "
                f"{fmt_pct(row.task_density_or_normalized_degree)} | {fmt_pct(row.main_density_or_normalized_degree)} | "
                f"{row.odds_ratio_task_vs_main:.3g} | {row.fisher_p:.3g} | {row.q_bh:.3g} |"
            )

    lines.extend(["", "## Nominal Task-vs-Main Differences", ""])
    if nominal.empty:
        lines.append("No task-vs-main comparisons had nominal p < 0.05.")
    else:
        lines.append("These are uncorrected exploratory contrasts and did not necessarily survive BH correction.")
        lines.append("")
        lines.append("| family | ROI set | feature | direction | task | main | OR/ratio | p | q |")
        lines.append("|---|---|---|---|---:|---:|---:|---:|---:|")
        for row in nominal.head(20).itertuples(index=False):
            lines.append(
                f"| {row.comparison_family} | {row.roi_set} | {row.feature} | {row.direction} | "
                f"{fmt_pct(row.task_density_or_normalized_degree)} | {fmt_pct(row.main_density_or_normalized_degree)} | "
                f"{row.odds_or_ratio_of_odds_ratios_task_vs_main:.3g} | {row.p_value:.3g} | {row.q_bh:.3g} |"
            )

    lines.extend(["", "## Common-ROI Edge Similarity", ""])
    lines.append("| direction | common ROIs | common possible edges | task sig | main sig | overlap | Jaccard | sign concordance |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in similarity.itertuples(index=False):
        sign = "NA" if pd.isna(row.sign_concordance_overlap) else f"{100 * row.sign_concordance_overlap:.1f}%"
        lines.append(
            f"| {row.direction} | {int(row.n_common_rois)} | {int(row.n_common_possible_edges)} | "
            f"{int(row.task_significant_edges)} | {int(row.main_significant_edges)} | {int(row.overlap_edges)} | "
            f"{row.jaccard_similarity:.3g} | {sign} |"
        )

    lines.extend(["", "## Output Tables", ""])
    for name in sorted(tables):
        lines.append(f"- `{name}.csv`")
    (out_dir / "NETWORK_DENSITY_COMPARISON_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric", default=DEFAULT_METRIC)
    parser.add_argument("--analysis-view", default=DEFAULT_ANALYSIS_VIEW)
    parser.add_argument("--fdr-scope", default=DEFAULT_FDR_SCOPE)
    parser.add_argument("--main-sig", type=Path, default=DEFAULT_MAIN_SIG)
    parser.add_argument("--main-stats", type=Path, default=DEFAULT_MAIN_STATS)
    parser.add_argument("--main-rois", type=Path, default=DEFAULT_MAIN_ROIS)
    parser.add_argument("--task-sig", type=Path, default=DEFAULT_TASK_SIG)
    parser.add_argument("--task-stats", type=Path, default=DEFAULT_TASK_STATS)
    parser.add_argument("--task-rois", type=Path, default=DEFAULT_TASK_ROIS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    main_data = load_network(
        NetworkSpec("main_result", args.main_sig, args.main_stats, args.main_rois),
        args.metric,
        args.analysis_view,
        args.fdr_scope,
    )
    task_data = load_network(
        NetworkSpec("task_activation_z3p1", args.task_sig, args.task_stats, args.task_rois),
        args.metric,
        args.analysis_view,
        args.fdr_scope,
    )

    common_rois = sorted(set(main_data.rois) & set(task_data.rois))
    profiles = [
        profile_network(task_data, task_data.rois, "full_network"),
        profile_network(main_data, main_data.rois, "full_network"),
        profile_network(task_data, common_rois, "common_roi_set"),
        profile_network(main_data, common_rois, "common_roi_set"),
    ]

    network_summary = network_summary_rows(profiles)
    hemisphere_density = hemisphere_density_rows(profiles)
    anatomical_density = anatomical_density_rows(profiles)
    group_pair_density = group_pair_density_rows(profiles)
    node_involvement = node_involvement_rows(profiles)

    hemisphere_comparison = compare_density_table(hemisphere_density, "hemisphere")
    anatomical_comparison = compare_density_table(anatomical_density, "anatomical_class")
    group_pair_comparison = compare_density_table(group_pair_density, "group_pair")
    node_comparison = node_comparison_rows(profiles)
    hemisphere_enrichment = hemisphere_enrichment_rows(hemisphere_density)
    anatomical_enrichment = enrichment_rows(
        anatomical_density,
        numerator_feature="same_anatomical_group",
        denominator_feature="cross_anatomical_group",
        contrast="same_vs_cross_anatomical_group",
        numerator_label="same",
        denominator_label="cross",
    )
    homotopic_enrichment = enrichment_rows(
        hemisphere_density,
        numerator_feature="homotopic",
        denominator_feature="between_nonhomotopic",
        contrast="homotopic_vs_between_nonhomotopic",
        numerator_label="homotopic",
        denominator_label="between_nonhomotopic",
    )
    hemisphere_enrichment_comparison = enrichment_network_comparison(
        hemisphere_density,
        numerator_feature="between_hemisphere",
        denominator_feature="within_hemisphere",
        contrast="between_vs_within_hemisphere",
    )
    anatomical_enrichment_comparison = enrichment_network_comparison(
        anatomical_density,
        numerator_feature="same_anatomical_group",
        denominator_feature="cross_anatomical_group",
        contrast="same_vs_cross_anatomical_group",
    )
    homotopic_enrichment_comparison = enrichment_network_comparison(
        hemisphere_density,
        numerator_feature="homotopic",
        denominator_feature="between_nonhomotopic",
        contrast="homotopic_vs_between_nonhomotopic",
    )
    similarity = edge_set_similarity(profiles)
    group_distribution_similarity = distribution_similarity(group_pair_density)

    sig_changes = significant_changes(
        ("hemisphere_density", hemisphere_comparison),
        ("anatomical_class_density", anatomical_comparison),
        ("group_pair_density", group_pair_comparison),
        ("node_normalized_involvement", node_comparison),
        ("hemisphere_enrichment_difference", hemisphere_enrichment_comparison),
        ("anatomical_enrichment_difference", anatomical_enrichment_comparison),
        ("homotopic_enrichment_difference", homotopic_enrichment_comparison),
    )
    nominal = nominal_changes(
        ("hemisphere_density", hemisphere_comparison),
        ("anatomical_class_density", anatomical_comparison),
        ("group_pair_density", group_pair_comparison),
        ("node_normalized_involvement", node_comparison),
        ("hemisphere_enrichment_difference", hemisphere_enrichment_comparison),
        ("anatomical_enrichment_difference", anatomical_enrichment_comparison),
        ("homotopic_enrichment_difference", homotopic_enrichment_comparison),
    )

    tables = {
        "network_summary_density": network_summary,
        "hemisphere_density": hemisphere_density,
        "hemisphere_enrichment": hemisphere_enrichment,
        "hemisphere_enrichment_network_comparison": hemisphere_enrichment_comparison,
        "hemisphere_density_network_comparison": hemisphere_comparison,
        "anatomical_class_density": anatomical_density,
        "anatomical_class_enrichment": anatomical_enrichment,
        "anatomical_class_enrichment_network_comparison": anatomical_enrichment_comparison,
        "anatomical_class_density_network_comparison": anatomical_comparison,
        "homotopic_enrichment": homotopic_enrichment,
        "homotopic_enrichment_network_comparison": homotopic_enrichment_comparison,
        "group_pair_density": group_pair_density,
        "group_pair_density_network_comparison": group_pair_comparison,
        "group_pair_distribution_similarity": group_distribution_similarity,
        "node_normalized_involvement": node_involvement,
        "node_involvement_network_comparison": node_comparison,
        "edge_set_similarity": similarity,
        "significant_network_differences_q05": sig_changes,
        "nominal_network_differences_p05": nominal,
    }
    for name, table in tables.items():
        table.to_csv(args.out_dir / f"{name}.csv", index=False)
    write_report(args.out_dir, tables, args)

    print(f"Saved density-normalized network comparison tables under {args.out_dir}", flush=True)
    print(sig_changes.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
