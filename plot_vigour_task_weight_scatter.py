#!/usr/bin/env python3
"""Scatter vigour-network weights against standard-GLM task z statistics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from analyze_ablation_constraints import _brainstem_mask
from quantify_full_vs_task_only_rois import DEFAULT_ATLAS_CACHE_DIR, _white_matter_mask


DEFAULT_WEIGHT_MAP = Path(
    "data/voxel_weights_task1_bold0.6_beta0.6_smooth1.25_gamma1.5.nii.gz"
)
DEFAULT_TASK_Z_MAP = Path("data/z_valu_standard_glm.nii.gz")
DEFAULT_TASK_VALUE_MAP = Path(
    "../fsl_glm/outputs/feat/mixed_model.gfeat/cope1.feat/stats/zstat1.nii.gz"
)
DEFAULT_OUT_BASE = Path("figures/ablation/ablation_full_vs_task_only_weight_scatter")
DEFAULT_VIGOUR_PERCENTILE = 90.0
DEFAULT_TASK_Z_THRESHOLD = 3.5
DEFAULT_TASK_Z_REFERENCE = 3.1
WEIGHT_DISPLAY_SCALE = 1000.0

COLORS = {
    "Vigour only": "#0072B2",
    "Task only": "#D55E00",
    "Common": "#009E73",
}


def _load_map(path: Path) -> tuple[nib.Nifti1Image, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing input map: {path}")
    img = nib.load(str(path))
    data = np.asarray(img.dataobj, dtype=float)
    if data.ndim != 3:
        raise ValueError(f"Expected a 3D map, but {path} has shape {data.shape}.")
    return img, data


def _resolve_task_value_map(path: Path) -> Path:
    candidates = [path]
    if not path.is_absolute():
        candidates.append(Path("../fsl_glm") / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    searched = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        f"Missing quantitative task z map. Searched: {searched}. "
        "Pass FEAT's cope1.feat/stats/zstat1.nii.gz with --task-value-map."
    )


def _check_same_grid(
    weight_img: nib.Nifti1Image,
    task_img: nib.Nifti1Image,
    task_path: Path,
) -> None:
    if task_img.shape != weight_img.shape:
        raise ValueError(
            f"Task z map {task_path} has shape {task_img.shape}; "
            f"the vigour map has shape {weight_img.shape}."
        )
    if not np.allclose(task_img.affine, weight_img.affine):
        raise ValueError(
            f"Task z map {task_path} is not on the vigour-map grid. "
            "Resample it before making a voxelwise scatter."
        )


def _correlations(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    if x.size < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return {
            "pearson_r": np.nan,
            "pearson_p": np.nan,
            "spearman_rho": np.nan,
            "spearman_p": np.nan,
        }
    pearson = pearsonr(x, y)
    spearman = spearmanr(x, y)
    return {
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman_rho": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
    }


def _category_summary(
    category_masks: dict[str, np.ndarray],
    weight_data: np.ndarray,
    task_data: np.ndarray,
) -> pd.DataFrame:
    union_n = sum(int(np.count_nonzero(mask)) for mask in category_masks.values())
    rows = []
    for category, mask in category_masks.items():
        weights = weight_data[mask]
        task_z = task_data[mask]
        rows.append(
            {
                "category": category,
                "n_voxels": int(mask.sum()),
                "percent_of_union": 100.0 * float(mask.sum()) / union_n if union_n else np.nan,
                "mean_vigour_weight": float(np.mean(weights)) if weights.size else np.nan,
                "median_vigour_weight": float(np.median(weights)) if weights.size else np.nan,
                "mean_task_z": float(np.mean(task_z)) if task_z.size else np.nan,
                "median_task_z": float(np.median(task_z)) if task_z.size else np.nan,
                **_correlations(weights, task_z),
            }
        )
    return pd.DataFrame(rows)


def _scatter(
    weight_data: np.ndarray,
    task_data: np.ndarray,
    category_masks: dict[str, np.ndarray],
    weight_threshold: float,
    task_z_reference: float,
    out_base: Path,
) -> None:
    union_mask = np.logical_or.reduce(list(category_masks.values()))
    x_all = weight_data[union_mask] * WEIGHT_DISPLAY_SCALE
    y_all = task_data[union_mask]
    x_min = min(0.0, float(np.nanmin(x_all)))
    x_max = float(np.nanmax(x_all))
    x_pad = max(0.03 * (x_max - x_min), 0.01)
    y_min = float(np.nanmin(y_all))
    y_max = float(np.nanmax(y_all))
    y_pad = max(0.04 * (y_max - y_min), 0.2)
    x_limits = (x_min - 0.15 * x_pad, x_max + x_pad)
    y_limits = (y_min - y_pad, y_max + y_pad)

    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"],
            "font.size": 11,
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    ):
        fig, ax = plt.subplots(figsize=(10.8, 7.4), facecolor="white")
        ax.set_facecolor("white")

        draw_specs = (
            ("Task only", 7, 0.18),
            ("Vigour only", 9, 0.28),
            ("Common", 14, 0.62),
        )
        for category, size, alpha in draw_specs:
            mask = category_masks[category]
            ax.scatter(
                weight_data[mask] * WEIGHT_DISPLAY_SCALE,
                task_data[mask],
                s=size,
                color=COLORS[category],
                alpha=alpha,
                linewidths=0,
                rasterized=True,
                zorder=2 if category != "Common" else 3,
            )

        ax.set_xlim(x_limits)
        ax.set_ylim(y_limits)
        ax.set_xlabel("Vigour network", fontsize=18)
        ax.set_ylabel("Task activation map", fontsize=18)
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        threshold_line_style = {
            "color": "#808080",
            "linestyle": (0, (5, 4)),
            "linewidth": 1.4,
            "alpha": 0.85,
            "zorder": 1,
        }
        ax.axvline(
            weight_threshold * WEIGHT_DISPLAY_SCALE,
            **threshold_line_style,
        )
        ax.axhline(task_z_reference, **threshold_line_style)

        direct_labels = (
            ("Task network", 0.14, 5.45, COLORS["Task only"]),
            ("Vigour network", 0.53, 1.05, COLORS["Vigour only"]),
            ("Common voxels", 0.48, 5.25, COLORS["Common"]),
        )
        for label, x, y, color in direct_labels:
            text = ax.text(
                x,
                y,
                label,
                color=color,
                fontsize=22,
                fontweight="bold",
                ha="center",
                va="center",
                zorder=4,
            )
            text.set_path_effects(
                [
                    path_effects.Stroke(linewidth=3.5, foreground="white"),
                    path_effects.Normal(),
                ]
            )

        fig.subplots_adjust(left=0.12, right=0.975, top=0.975, bottom=0.12)

        out_base.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(f"{out_base}.png", dpi=300, bbox_inches="tight")
        fig.savefig(f"{out_base}.pdf", bbox_inches="tight")
        plt.close(fig)


def run(args: argparse.Namespace) -> None:
    weight_img, weight_data = _load_map(args.weight_map)
    task_membership_img, task_membership_data = _load_map(args.task_z_map)
    _check_same_grid(weight_img, task_membership_img, args.task_z_map)
    task_value_map = _resolve_task_value_map(args.task_value_map)
    task_value_img, task_value_data = _load_map(task_value_map)
    _check_same_grid(weight_img, task_value_img, task_value_map)

    finite_nonzero_weights = np.isfinite(weight_data) & (weight_data > 0)
    if not np.any(finite_nonzero_weights):
        raise ValueError(f"No positive finite weights found in {args.weight_map}.")
    weight_threshold = float(
        np.percentile(weight_data[finite_nonzero_weights], args.vigour_percentile)
    )
    vigour_mask = finite_nonzero_weights & (weight_data >= weight_threshold)
    task_mask = np.isfinite(task_membership_data) & (
        task_membership_data >= args.task_z_threshold
    )

    brainstem_mask = np.zeros(weight_data.shape, dtype=bool)
    if not args.include_brainstem:
        brainstem_mask = _brainstem_mask(weight_img)
        vigour_mask &= ~brainstem_mask
        task_mask &= ~brainstem_mask

    white_matter_mask = np.zeros(weight_data.shape, dtype=bool)
    if not args.include_white_matter:
        white_matter_mask = _white_matter_mask(weight_img, args.atlas_cache_dir)
        white_matter_mask &= ~brainstem_mask
        vigour_mask &= ~white_matter_mask
        task_mask &= ~white_matter_mask

    category_masks = {
        "Vigour only": vigour_mask & ~task_mask,
        "Task only": task_mask & ~vigour_mask,
        "Common": vigour_mask & task_mask,
    }
    if not any(np.any(mask) for mask in category_masks.values()):
        raise ValueError("The two thresholded maps have no selected voxels.")

    summary = _category_summary(category_masks, weight_data, task_value_data)
    summary_path = Path(f"{args.out_base}_summary.csv")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)

    network_counts = pd.DataFrame(
        [
            {
                "network": "Task-activation map",
                "voxel_count": int(task_mask.sum()),
                "shared_voxels": int(category_masks["Common"].sum()),
                "shared_percent_of_network": (
                    100.0 * float(category_masks["Common"].sum()) / float(task_mask.sum())
                ),
            },
            {
                "network": "Vigour network",
                "voxel_count": int(vigour_mask.sum()),
                "shared_voxels": int(category_masks["Common"].sum()),
                "shared_percent_of_network": (
                    100.0 * float(category_masks["Common"].sum()) / float(vigour_mask.sum())
                ),
            },
        ]
    )
    network_counts_path = Path(f"{args.out_base}_network_counts.csv")
    network_counts.to_csv(network_counts_path, index=False)

    metadata = {
        "vigour_weight_map": str(args.weight_map),
        "task_activation_map": str(args.task_z_map),
        "task_value_map": str(task_value_map),
        "vigour_definition": (
            f"top {100.0 - args.vigour_percentile:g}% of positive nonzero weights"
        ),
        "vigour_percentile": float(args.vigour_percentile),
        "vigour_weight_threshold": weight_threshold,
        "n_positive_nonzero_weight_voxels": int(finite_nonzero_weights.sum()),
        "task_definition": (
            f"published task-selection display map value >= {args.task_z_threshold:g}"
        ),
        "task_membership_map_threshold": float(args.task_z_threshold),
        "raw_task_z_reference": float(args.task_z_reference),
        "brainstem_excluded": not args.include_brainstem,
        "brainstem_voxels": int(brainstem_mask.sum()),
        "white_matter_excluded": not args.include_white_matter,
        "white_matter_voxels": int(white_matter_mask.sum()),
        "atlas_cache_dir": str(args.atlas_cache_dir),
        "grid_shape": list(weight_img.shape),
        "affine": weight_img.affine.tolist(),
        "network_counts": {
            "Vigour network": int(vigour_mask.sum()),
            "Task-activation map": int(task_mask.sum()),
            "Shared": int(category_masks["Common"].sum()),
        },
        "category_counts": {
            category: int(mask.sum()) for category, mask in category_masks.items()
        },
        "union_voxels": int(np.logical_or.reduce(list(category_masks.values())).sum()),
        "note": (
            "Defaults reproduce Table S4 and the existing ablation anatomy/ROI comparison: "
            "the task-activation map is data/z_valu_standard_glm.nii.gz at z >= 3.5, "
            "with brainstem and Harvard-Oxford cerebral white matter excluded. The scatter "
            "y-axis uses raw FEAT stats/zstat1.nii.gz rather than rendered display values."
        ),
    }
    metadata_path = Path(f"{args.out_base}_metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    _scatter(
        weight_data,
        task_value_data,
        category_masks,
        weight_threshold,
        args.task_z_reference,
        args.out_base,
    )

    print(f"Saved {args.out_base}.png")
    print(f"Saved {args.out_base}.pdf")
    print(f"Saved {summary_path}")
    print(f"Saved {network_counts_path}")
    print(f"Saved {metadata_path}")
    print(network_counts.to_string(index=False))
    print(summary[["category", "n_voxels", "percent_of_union"]].to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weight-map", type=Path, default=DEFAULT_WEIGHT_MAP)
    parser.add_argument("--task-z-map", type=Path, default=DEFAULT_TASK_Z_MAP)
    parser.add_argument(
        "--task-value-map",
        type=Path,
        default=DEFAULT_TASK_VALUE_MAP,
        help="Raw FEAT stats/zstat1.nii.gz used for quantitative scatter y-values.",
    )
    parser.add_argument("--vigour-percentile", type=float, default=DEFAULT_VIGOUR_PERCENTILE)
    parser.add_argument("--task-z-threshold", type=float, default=DEFAULT_TASK_Z_THRESHOLD)
    parser.add_argument(
        "--task-z-reference",
        type=float,
        default=DEFAULT_TASK_Z_REFERENCE,
        help="Reference line shown on the raw-z scatter; it does not redefine Table S4 membership.",
    )
    parser.add_argument("--out-base", type=Path, default=DEFAULT_OUT_BASE)
    parser.add_argument("--atlas-cache-dir", type=Path, default=DEFAULT_ATLAS_CACHE_DIR)
    parser.add_argument(
        "--include-brainstem",
        action="store_true",
        help="Keep brainstem voxels; the default excludes them to match the anatomy figure.",
    )
    parser.add_argument(
        "--include-white-matter",
        action="store_true",
        help="Keep cerebral white matter; the default excludes it to match Table S4.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
