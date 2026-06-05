from pathlib import Path
from io import BytesIO
import argparse
import base64
import json
import re
import warnings

import numpy as np

warnings.filterwarnings("ignore", message="Unable to import Axes3D.*")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
import nibabel as nib
from PIL import Image
from scipy.ndimage import binary_fill_holes

from motor_overlap_overlay import fill_from_nearest_selected, motor_overlap_masks


DEFAULT_HTML = Path("data/voxel_weights_task1_bold0.6_beta0.6_smooth1.25_gamma1.5_bold_thr90.html")
DEFAULT_WEIGHT_MAP = Path("data/voxel_weights_task1_bold0.6_beta0.6_smooth1.25_gamma1.5.nii.gz")
DEFAULT_OUTPUT_BASE = (
    Path("figures")
    / "voxel_weights_task1_bold0.6_beta0.6_smooth1.25_gamma1.5_bold_thr90_multiplane_weights_colorbar"
)
DEFAULT_REQUIRED_Z_CUTS = (64, 66)
AXIS_LABEL_FONTSIZE = 13
COLORBAR_FONTSIZE = 12

plt.rcParams.update(
    {
        "font.family": "Liberation Sans",
        "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot thresholded voxel weights as a multiplane brain montage with a continuous colorbar."
    )
    parser.add_argument("--html", type=Path, default=DEFAULT_HTML, help="Thresholded HTML map used for background/mask.")
    parser.add_argument("--weight-map", type=Path, default=DEFAULT_WEIGHT_MAP, help="NIfTI map containing voxel weights.")
    parser.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT_BASE, help="Output path without extension.")
    parser.add_argument("--n-cuts", type=int, default=8, help="Number of cuts per plane.")
    parser.add_argument("--gap", type=int, default=4, help="Minimum voxel gap between selected cuts.")
    parser.add_argument(
        "--required-z-cuts",
        type=int,
        nargs="*",
        default=DEFAULT_REQUIRED_Z_CUTS,
        help="Axial z coordinates that must be included in the montage.",
    )
    parser.add_argument("--scale", type=float, default=1000.0, help="Multiplier applied before color mapping.")
    parser.add_argument(
        "--colorbar-label",
        default="Voxel weight\n($\\times 10^{-3}$)",
        help="Colorbar label after applying --scale.",
    )
    parser.add_argument(
        "--cmap",
        default="YlOrRd",
        help="Matplotlib colormap. The default is a bright sequential map for positive weights.",
    )
    parser.add_argument("--cmap-min", type=float, default=0.18, help="Lower fraction used when truncating --cmap.")
    parser.add_argument("--cmap-max", type=float, default=0.98, help="Upper fraction used when truncating --cmap.")
    parser.add_argument(
        "--include-weight-map-mask",
        action="store_true",
        help="Display nonzero voxels from --weight-map in addition to the thresholded HTML mask.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="PNG output resolution.")
    return parser.parse_args()


def sprite_volume(sprite, nx, ny, nz):
    volume = np.zeros((nx, ny, nz) + sprite.shape[2:], sprite.dtype)
    n_cols = sprite.shape[1] // ny
    for x in range(nx):
        col, row = x % n_cols, x // n_cols
        tile = sprite[row * nz : (row + 1) * nz, col * ny : (col + 1) * ny]
        volume[x] = tile[::-1].transpose(1, 0, 2)
    return volume


def load_html_display(html_path):
    html = html_path.read_text()
    images = [
        np.asarray(Image.open(BytesIO(base64.b64decode(encoded))).convert("RGBA"))
        for encoded in re.findall(r'src="data:image/png;base64,([^"]+)"', html)
    ]
    cfg = json.loads(re.search(r"brainsprite\((\{.*?\})\);", html).group(1))
    nx, ny, nz = [cfg["nbSlice"][axis] for axis in "XYZ"]
    affine = np.asarray(cfg["affine"], dtype=float)
    background = sprite_volume(images[0], nx, ny, nz)[..., :3].mean(-1).astype(float)
    selected_mask = sprite_volume(images[2], nx, ny, nz)[..., 3] > 0
    return background, selected_mask, affine


def align_weights_to_display(weight_img, target_shape, target_affine):
    weights = np.asarray(weight_img.get_fdata(), dtype=float)
    if weights.shape != target_shape:
        raise ValueError(
            f"Weight map shape {weights.shape} does not match HTML sprite shape {target_shape}."
        )

    aligned = weights
    source_affine = np.asarray(weight_img.affine, dtype=float)
    for axis in range(3):
        source_step = source_affine[axis, axis]
        target_step = target_affine[axis, axis]
        if source_step and target_step and np.sign(source_step) != np.sign(target_step):
            aligned = np.flip(aligned, axis=axis)
    return aligned


def world_coord(affine, mode, index):
    axis = "xyz".index(mode)
    return int(round(affine[axis, axis] * (index + 1) + affine[axis, 3]))


def pick_cuts(mask, affine, mode, n_cuts, gap):
    axis = "xyz".index(mode)
    counts = mask.sum(tuple(i for i in range(3) if i != axis))
    chosen = []
    for index in np.argsort(counts)[::-1]:
        if counts[index] and all(abs(index - previous) >= gap for previous in chosen):
            chosen.append(int(index))
        if len(chosen) == n_cuts:
            break
    return [world_coord(affine, mode, index) for index in sorted(chosen)]


def include_required_cuts(cuts, required_cuts, n_cuts):
    required_cuts = list(dict.fromkeys(required_cuts))
    merged = sorted(set(cuts).union(required_cuts))
    while len(merged) > n_cuts:
        candidates = [cut for cut in merged if cut not in required_cuts]
        if not candidates:
            break
        drop = min(candidates, key=lambda cut: min(abs(cut - required) for required in required_cuts))
        merged.remove(drop)
    return merged


def cut_index(affine, mode, cut):
    axis = "xyz".index(mode)
    return int(round((cut - affine[axis, 3]) / affine[axis, axis] - 1))


def plane(volume, affine, mode, cut):
    index = cut_index(affine, mode, cut)
    if mode == "x":
        return volume[index].T[::-1]
    if mode == "y":
        return volume[:, index, :].T[::-1]
    return volume[:, :, index].T[::-1]


def pad(array, fill_value=0):
    return np.pad(array, ((3, 5), (2, 2)), mode="constant", constant_values=fill_value)


def crop(background, mask, weights, *extra_masks):
    crop_mask = binary_fill_holes(background > 0) | mask
    for extra_mask in extra_masks:
        crop_mask |= extra_mask
    y, x = np.where(crop_mask)
    y0, y1 = max(y.min() - 4, 0), min(y.max() + 5, background.shape[0])
    x0, x1 = max(x.min() - 4, 0), min(x.max() + 5, background.shape[1])
    cropped = (
        pad(background[y0:y1, x0:x1]),
        pad(mask[y0:y1, x0:x1], False),
        pad(weights[y0:y1, x0:x1], np.nan),
    )
    return cropped + tuple(pad(extra_mask[y0:y1, x0:x1], False) for extra_mask in extra_masks)


def anatomical_rgba(background, max_intensity):
    brain = binary_fill_holes(background > 0)
    rgba = plt.cm.gray(np.clip(background / max_intensity, 0, 1))
    rgba[~brain, 3] = 0
    return rgba


def truncate_colormap(cmap_name, vmin=0.0, vmax=1.0, n=256):
    base_cmap = plt.get_cmap(cmap_name)
    colors = base_cmap(np.linspace(vmin, vmax, n))
    return LinearSegmentedColormap.from_list(f"{cmap_name}_{vmin:.2f}_{vmax:.2f}", colors)


def color_norm(values):
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        raise ValueError("No finite weights were found inside the selected HTML mask.")
    vmin = float(np.min(finite_values))
    vmax = float(np.max(finite_values))
    if vmin < 0 < vmax:
        limit = max(abs(vmin), abs(vmax))
        return TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    return Normalize(vmin=vmin, vmax=vmax)


def draw_overlay(ax, weights, mask, cmap, norm):
    overlay = np.ma.masked_invalid(np.where(mask, weights, np.nan))
    ax.imshow(overlay, cmap=cmap, norm=norm, interpolation="nearest", alpha=0.96)
    if np.any(mask):
        ax.contour(mask.astype(float), levels=[0.5], colors="0.12", linewidths=0.34, alpha=0.72)


def main():
    args = parse_args()
    background, selected_mask, display_affine = load_html_display(args.html)
    motor_overlap_display, _ = motor_overlap_masks(selected_mask, display_affine)
    weight_img = nib.load(args.weight_map)
    weights = align_weights_to_display(weight_img, selected_mask.shape, display_affine) * args.scale
    display_source_mask = selected_mask
    if args.include_weight_map_mask:
        display_source_mask = display_source_mask | (np.isfinite(weights) & (weights != 0))
    weights = np.where(display_source_mask & np.isfinite(weights), weights, np.nan)
    weights = fill_from_nearest_selected(weights, display_source_mask, motor_overlap_display)
    display_mask = display_source_mask | motor_overlap_display

    norm = color_norm(weights)
    cmap = truncate_colormap(args.cmap, args.cmap_min, args.cmap_max).copy()
    cmap.set_bad((0, 0, 0, 0))
    max_intensity = np.percentile(background[background > 0], 99.5)

    plane_specs = [("x", "Sagittal"), ("y", "Coronal"), ("z", "Axial")]
    cuts = {mode: pick_cuts(selected_mask, display_affine, mode, args.n_cuts, args.gap) for mode, _ in plane_specs}
    cuts["z"] = include_required_cuts(cuts["z"], args.required_z_cuts, args.n_cuts)
    n_cols = max(len(mode_cuts) for mode_cuts in cuts.values())

    fig = plt.figure(figsize=(15.1, 5.35), facecolor="white")
    grid = fig.add_gridspec(
        len(plane_specs),
        n_cols,
        left=0.008,
        right=0.925,
        bottom=0.065,
        top=0.955,
        wspace=0.012,
        hspace=0.07,
    )
    axes = np.asarray(
        [[fig.add_subplot(grid[row, col]) for col in range(n_cols)] for row in range(len(plane_specs))]
    )
    for row, (mode, _) in enumerate(plane_specs):
        for col in range(n_cols):
            ax = axes[row, col]
            ax.set_facecolor("none")
            ax.patch.set_alpha(0)
            if col >= len(cuts[mode]):
                ax.set_axis_off()
                continue
            cut = cuts[mode][col]
            bg_slice, mask_slice, weight_slice = crop(
                plane(background, display_affine, mode, cut),
                plane(display_mask, display_affine, mode, cut),
                plane(weights, display_affine, mode, cut),
            )
            ax.imshow(anatomical_rgba(bg_slice, max_intensity), interpolation="nearest")
            draw_overlay(ax, weight_slice, mask_slice, cmap, norm)
            height, width = bg_slice.shape[:2]
            brain_y, brain_x = np.where(binary_fill_holes(bg_slice > 0))
            if row == 0:
                label_y = max(float(brain_y.min()) - 2.5, -1.0) if brain_y.size else 0.0
                ax.text(
                    width * 0.14,
                    label_y,
                    "L",
                    ha="center",
                    va="bottom",
                    fontsize=AXIS_LABEL_FONTSIZE,
                    color="0.15",
                    clip_on=False,
                )
                ax.text(
                    width * 0.86,
                    label_y,
                    "R",
                    ha="center",
                    va="bottom",
                    fontsize=AXIS_LABEL_FONTSIZE,
                    color="0.15",
                    clip_on=False,
                )
            if brain_y.size and brain_x.size:
                slice_label_x = (brain_x.min() + brain_x.max()) / 2
                slice_label_y = brain_y.max() + 3
            else:
                slice_label_x = width / 2
                slice_label_y = height + 3
            ax.text(
                slice_label_x,
                slice_label_y,
                f"{mode} = {cut:g}",
                ha="center",
                va="top",
                fontsize=AXIS_LABEL_FONTSIZE,
                color="0.15",
            )
            ax.set_axis_off()

    cax = fig.add_axes([0.948, 0.13, 0.012, 0.74])
    scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar.set_array([])
    colorbar = fig.colorbar(scalar, cax=cax)
    colorbar.ax.set_title(args.colorbar_label, fontsize=COLORBAR_FONTSIZE, pad=7)
    colorbar.ax.tick_params(labelsize=COLORBAR_FONTSIZE, width=0.6, length=2.8, pad=2)
    colorbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    colorbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    colorbar.outline.set_linewidth(0.6)

    args.output_base.parent.mkdir(exist_ok=True)
    fig.savefig(f"{args.output_base}.png", dpi=args.dpi, bbox_inches="tight", pad_inches=0.015)
    fig.savefig(f"{args.output_base}.pdf", bbox_inches="tight", pad_inches=0.015)
    print(f"{args.output_base}.png")
    print(f"{args.output_base}.pdf")


if __name__ == "__main__":
    main()
