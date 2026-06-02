#!/usr/bin/env python3
"""Plot the ROI labels used for the full-vs-task-only quantification."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
import nibabel as nib
import numpy as np

from analyze_ablation_constraints import (
    DEFAULT_ATLAS_CACHE_DIR,
    DEFAULT_FULL_MODEL_HTML,
    DEFAULT_MAIN_MAP,
    DEFAULT_TASK_ONLY_MAP,
    DEFAULT_TASK_ONLY_Z_THRESHOLD,
    _cut_indices_for_mask,
    _html_sprite_volumes,
    _plane_slice,
)
from quantify_full_vs_task_only_rois import (
    _build_regions,
    _exclude_white_matter,
    _load_figure_masks,
    _roi_quantification,
    _white_matter_mask,
)
from threshold_robustness_voxel_network import UNASSIGNED_ROI

DEFAULT_OUT = Path("figures/ablation/ablation_full_vs_task_only_roi_labeling.png")


def _roi_colors(n: int) -> list[tuple[float, float, float, float]]:
    palettes = (plt.cm.tab20, plt.cm.tab20b, plt.cm.tab20c)
    colors: list[tuple[float, float, float, float]] = []
    for palette in palettes:
        for idx in range(palette.N):
            colors.append(palette(idx))
    return colors[:n]


def _make_label_volume(
    roi_df,
    regions,
    assigned_mask: np.ndarray,
    union_mask: np.ndarray,
) -> tuple[np.ndarray, list[str], list[float]]:
    label_volume = np.zeros(union_mask.shape, dtype=np.int16)
    roi_names = roi_df.sort_values(["union_voxels", "roi_name"], ascending=[False, True])["roi_name"].tolist()
    pct_of_roi = []
    for label_id, roi_name in enumerate(roi_names, start=1):
        if roi_name == UNASSIGNED_ROI:
            mask = union_mask & ~assigned_mask
            pct = np.nan
        else:
            mask = union_mask & regions[roi_name].mask
            pct = float(
                roi_df.loc[roi_df["roi_name"].eq(roi_name), "union_pct_of_roi"].iloc[0]
            )
        label_volume[mask] = label_id
        pct_of_roi.append(pct)
    return label_volume, roi_names, pct_of_roi


def _slice_coord_mm(affine: np.ndarray, axis: int, index: int) -> float:
    ijk = np.zeros(3, dtype=float)
    ijk[axis] = float(index)
    return float(nib.affines.apply_affine(affine, ijk)[axis])


def _crop_label_slice(background: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    crop_mask = background > 0
    crop_mask |= labels > 0
    y, x = np.where(crop_mask)
    if y.size == 0:
        return np.pad(background, ((3, 3), (2, 2))), np.pad(labels, ((3, 3), (2, 2)))
    y0, y1 = max(int(y.min()) - 4, 0), min(int(y.max()) + 5, background.shape[0])
    x0, x1 = max(int(x.min()) - 4, 0), min(int(x.max()) + 5, background.shape[1])
    return (
        np.pad(background[y0:y1, x0:x1], ((3, 3), (2, 2))),
        np.pad(labels[y0:y1, x0:x1], ((3, 3), (2, 2))),
    )


def _plot_label_slice(
    ax: plt.Axes,
    background: np.ndarray,
    labels: np.ndarray,
    cmap: ListedColormap,
    norm: BoundaryNorm,
) -> None:
    bg, label_cropped = _crop_label_slice(background, labels)
    vmax = np.percentile(bg[bg > 0], 99) if np.any(bg > 0) else 1.0
    ax.imshow(bg, cmap="gray", vmin=0, vmax=vmax, interpolation="nearest")
    masked = np.ma.masked_where(label_cropped == 0, label_cropped)
    ax.imshow(masked, cmap=cmap, norm=norm, alpha=0.78, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)


def _pct_label(value: float) -> str:
    return "NA" if np.isnan(value) else f"{100.0 * value:.2f}% ROI"


def plot(args: argparse.Namespace) -> None:
    background, _, html_affine = _html_sprite_volumes(args.full_html)
    reference_img, masks, _ = _load_figure_masks(
        args.reference_map,
        args.full_html,
        args.task_map,
        args.task_z_threshold,
    )
    if args.exclude_white_matter:
        masks, _ = _exclude_white_matter(
            masks,
            _white_matter_mask(reference_img, args.atlas_cache_dir) & ~masks["brainstem"],
            {},
        )
    analysis_space_mask = ~masks["brainstem"] & ~masks.get("white_matter", np.zeros(reference_img.shape[:3], dtype=bool))
    regions, assigned_mask, _ = _build_regions(
        reference_img,
        args.atlas_cache_dir,
        analysis_space_mask,
        args.atlas_mode,
        masks["union"],
    )
    roi_df = _roi_quantification(regions, assigned_mask, masks, reference_img)
    label_volume, roi_names, pct_of_roi = _make_label_volume(roi_df, regions, assigned_mask, masks["union"])

    colors = _roi_colors(len(roi_names))
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(0.5, len(roi_names) + 1.5), len(roi_names))

    planes = (("Sagittal", 0, "x"), ("Coronal", 1, "y"), ("Axial", 2, "z"))
    fig = plt.figure(figsize=(18, 10.5), constrained_layout=False)
    grid = fig.add_gridspec(3, 6, width_ratios=[1, 1, 1, 1, 1, 1.55], wspace=0.04, hspace=0.08)
    for row, (plane_name, axis, coord_name) in enumerate(planes):
        cuts = _cut_indices_for_mask(masks["union"], axis, n_cuts=5, min_gap=6)
        for col, cut in enumerate(cuts):
            ax = fig.add_subplot(grid[row, col])
            bg_slice = _plane_slice(background, axis, cut)
            label_slice = _plane_slice(label_volume, axis, cut)
            _plot_label_slice(ax, bg_slice, label_slice, cmap, norm)
            coord = _slice_coord_mm(html_affine, axis, cut)
            ax.set_title(f"{plane_name} {coord_name}={coord:g} mm", fontsize=8, pad=2)

    legend_ax = fig.add_subplot(grid[:, 5])
    legend_ax.axis("off")
    handles = [
        Patch(facecolor=colors[idx], edgecolor="none", label=f"{idx + 1}. {name} ({_pct_label(pct_of_roi[idx])})")
        for idx, name in enumerate(roi_names)
    ]
    legend_ax.legend(
        handles=handles,
        loc="upper left",
        frameon=False,
        fontsize=7.1,
        handlelength=0.9,
        handletextpad=0.4,
        borderaxespad=0.0,
        labelspacing=0.34,
    )
    legend_ax.set_title("ROI label (union % of ROI)", fontsize=10, loc="left", pad=8)

    if args.atlas_mode == "aal3_ho_nearest":
        atlas_note = "AAL3 first; Harvard-Oxford fill; residual non-white-matter voxels assigned to nearest anatomical label"
    elif args.atlas_mode == "aal3_ho_fill":
        atlas_note = "AAL3 first; Harvard-Oxford fill; atlas-uncovered selected voxels left Unassigned"
    else:
        atlas_note = "AAL3 coarse bilateral labels; atlas-uncovered selected voxels left Unassigned"
    fig.suptitle(
        f"Brain ROI labels used for full-model vs task-activation quantification\n{atlas_note}",
        fontsize=14,
        y=0.985,
    )
    fig.text(
        0.02,
        0.015,
        "Colored voxels are the non-white-matter union of the vigour-network HTML overlay and the z>=3.5 task-activation map after brainstem suppression.",
        fontsize=9,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(args.out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-map", type=Path, default=DEFAULT_MAIN_MAP)
    parser.add_argument("--full-html", type=Path, default=DEFAULT_FULL_MODEL_HTML)
    parser.add_argument("--task-map", type=Path, default=DEFAULT_TASK_ONLY_MAP)
    parser.add_argument("--task-z-threshold", type=float, default=DEFAULT_TASK_ONLY_Z_THRESHOLD)
    parser.add_argument("--atlas-cache-dir", type=Path, default=DEFAULT_ATLAS_CACHE_DIR)
    parser.add_argument("--atlas-mode", choices=("aal3", "aal3_ho_fill", "aal3_ho_nearest"), default="aal3_ho_nearest")
    parser.add_argument(
        "--include-white-matter",
        action="store_false",
        dest="exclude_white_matter",
        help="Include Harvard-Oxford cerebral white-matter voxels in the ROI labeling plot.",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> None:
    plot(parse_args())


if __name__ == "__main__":
    main()
