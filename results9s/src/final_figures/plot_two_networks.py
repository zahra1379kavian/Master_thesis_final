#!/usr/bin/env python3
"""Create an interactive HTML view for two voxel-weight maps on anatomy."""

from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
from nilearn import datasets, image, plotting


DEFAULT_MAP_1 = (
    "results/ablation/"
    "voxel_weights_mean_foldavg_sub9_ses1_task0.8_bold0.8_beta0.5_smooth0.2_gamma1_"
    "bold_thr90.nii.gz"
)
DEFAULT_MAP_2 = (
    "results/ablation/"
    "voxel_weights_mean_foldavg_sub9_ses1_task0.8_bold0_beta0_smooth0_gamma1_"
    "bold_thr90_postcentral_boosted.nii.gz"
)
DEFAULT_ANAT = "/Data/zahra/anatomy_masks/MNI152_T1_2mm_brain.nii.gz"
DEFAULT_OUT = "results/ablation/two_networks_overlay_sub9_ses1_thr90.html"
DEFAULT_CUT_COORDS = [0.0, -20.0, 52.0]


def _infer_map_labels(map_1_path: str, map_2_path: str) -> tuple[str, str]:
    name_1 = Path(map_1_path).name.lower()
    name_2 = Path(map_2_path).name.lower()

    def _condition_label(name: str) -> str | None:
        if "all_states" in name or "allstates" in name:
            return "All states"
        if "ses1" in name:
            return "Medication off (ses1)"
        if "ses2" in name:
            return "Medication on (ses2)"
        return None

    if "postcentral_boosted" in name_1 and "postcentral_boosted" not in name_2:
        return "Postcentral-boosted map", "Reference map"
    if "postcentral_boosted" in name_2 and "postcentral_boosted" not in name_1:
        return "Reference map", "Postcentral-boosted map"

    label_1 = _condition_label(name_1)
    label_2 = _condition_label(name_2)
    if label_1 is not None and label_2 is not None and label_1 != label_2:
        return label_1, label_2

    return "Map 1", "Map 2"


def _label_cmap() -> ListedColormap:
    return ListedColormap(
        [
            (0.0, 0.0, 0.0, 0.0),      # background
            (0.86, 0.18, 0.18, 1.0),   # map 1 only (red)
            (0.17, 0.42, 0.77, 1.0),   # map 2 only (blue)
            (0.55, 0.00, 0.55, 1.0),   # overlap (magenta)
        ]
    )


def _resolve_cut_coords(
    map_2_img: nib.Nifti1Image, requested_cut_coords: list[float]
) -> tuple[float, float, float]:
    atlas = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
    atlas_img = atlas.maps if isinstance(atlas.maps, nib.Nifti1Image) else nib.load(atlas.maps)
    atlas_img = image.resample_to_img(
        atlas_img,
        map_2_img,
        interpolation="nearest",
        force_resample=True,
        copy_header=True,
    )
    labels = [
        lbl.decode("utf-8", errors="replace") if isinstance(lbl, bytes) else str(lbl)
        for lbl in atlas.labels
    ]
    postcentral_idx = [i for i, lbl in enumerate(labels) if "postcentral gyrus" in lbl.lower()]
    if not postcentral_idx:
        return tuple(requested_cut_coords)

    postcentral_mask = np.isin(atlas_img.get_fdata().astype(int), postcentral_idx)
    active_postcentral = (map_2_img.get_fdata() > 0) & postcentral_mask
    if not np.any(active_postcentral):
        return tuple(requested_cut_coords)

    ijk = np.argwhere(active_postcentral)
    xyz = nib.affines.apply_affine(map_2_img.affine, ijk)
    return tuple(np.mean(xyz, axis=0).tolist())


def _slice_coord_from_index(
    affine: np.ndarray, shape: tuple[int, int, int], axis_idx: int, slice_idx: int
) -> float:
    ijk = np.array([(dim - 1) / 2.0 for dim in shape], dtype=float)
    ijk[axis_idx] = float(slice_idx)
    xyz = nib.affines.apply_affine(affine, ijk)
    return float(xyz[axis_idx])


def _select_spaced_indices(
    scored_indices: list[tuple[float, int]], n_cuts: int, min_gap: int
) -> list[int]:
    selected: list[int] = []
    for _, idx in sorted(scored_indices, key=lambda item: item[0], reverse=True):
        if all(abs(idx - chosen) > min_gap for chosen in selected):
            selected.append(idx)
        if len(selected) == n_cuts:
            break

    if len(selected) < n_cuts:
        for _, idx in sorted(scored_indices, key=lambda item: item[0], reverse=True):
            if idx not in selected:
                selected.append(idx)
            if len(selected) == n_cuts:
                break

    return sorted(selected)


def _paper_cut_coords(
    roi_img: nib.Nifti1Image, n_cuts: int
) -> dict[str, list[float]]:
    data = np.rint(roi_img.get_fdata()).astype(np.int16)
    if not np.any(data > 0):
        return {
            "x": np.linspace(-50.0, 50.0, n_cuts).tolist(),
            "y": np.linspace(-70.0, 30.0, n_cuts).tolist(),
            "z": np.linspace(-20.0, 80.0, n_cuts).tolist(),
        }

    coords: dict[str, list[float]] = {}
    for axis_idx, axis_name in enumerate(("x", "y", "z")):
        scored_indices: list[tuple[float, int]] = []
        for slice_idx in range(data.shape[axis_idx]):
            plane = np.take(data, slice_idx, axis=axis_idx)
            map_1_only = int(np.count_nonzero(plane == 1))
            map_2_only = int(np.count_nonzero(plane == 2))
            overlap = int(np.count_nonzero(plane == 3))
            map_1_total = map_1_only + overlap
            map_2_total = map_2_only + overlap
            active_total = map_1_total + map_2_total
            if active_total == 0:
                continue

            # Favor slices that show substantial signal and preferably both maps.
            score = (
                active_total
                + 2.5 * overlap
                + 0.5 * min(map_1_total, map_2_total)
            )
            scored_indices.append((float(score), slice_idx))

        if not scored_indices:
            coords[axis_name] = np.linspace(-50.0, 50.0, n_cuts).tolist()
            continue

        min_gap = max(2, int(round(data.shape[axis_idx] / (n_cuts * 3))))
        selected_indices = _select_spaced_indices(scored_indices, n_cuts=n_cuts, min_gap=min_gap)
        coords[axis_name] = [
            _slice_coord_from_index(roi_img.affine, data.shape, axis_idx, slice_idx)
            for slice_idx in selected_indices
        ]
    return coords


def _anat_with_white_panel_background(anat_img: nib.Nifti1Image) -> nib.Nifti1Image:
    anat_data = anat_img.get_fdata().astype("float32")
    positive_mask = anat_data > 0
    if not np.any(positive_mask):
        return anat_img

    white_bg_data = anat_data.copy()
    white_bg_data[~positive_mask] = float(np.max(anat_data[positive_mask]))
    return image.new_img_like(anat_img, white_bg_data)


def _paper_layout(anat_img: nib.Nifti1Image, n_cuts: int) -> dict[str, object]:
    left = 0.02
    right = 0.995
    top = 0.99
    bottom = 0.085
    slice_width = 2.05
    legend_height = 0.65
    plane_axes = {
        "x": (1, 2),  # sagittal: y by z
        "y": (0, 2),  # coronal: x by z
        "z": (0, 1),  # axial: x by y
    }
    voxel_sizes = nib.affines.voxel_sizes(anat_img.affine)[:3]
    shape = anat_img.shape[:3]
    height_ratios: list[float] = []
    for display_mode in ("x", "y", "z"):
        width_axis, height_axis = plane_axes[display_mode]
        plane_width = float(shape[width_axis] * voxel_sizes[width_axis])
        plane_height = float(shape[height_axis] * voxel_sizes[height_axis])
        height_ratios.append(max(plane_height / plane_width, 0.6))

    fig_width = max(13.0, n_cuts * slice_width / (right - left))
    row_content_height = sum(slice_width * ratio for ratio in height_ratios)
    fig_height = max(9.1, (row_content_height + legend_height) / (top - bottom))
    return {
        "figsize": (fig_width, fig_height),
        "height_ratios": height_ratios,
        "subplot_adjust": {
            "left": left,
            "right": right,
            "top": top,
            "bottom": bottom,
            "hspace": 0.015,
        },
    }


def _save_paper_figure(
    combined_img: nib.Nifti1Image,
    map_1_binary_img: nib.Nifti1Image,
    map_2_binary_img: nib.Nifti1Image,
    anat_img: nib.Nifti1Image,
    out_png: Path | None,
    out_pdf: Path | None,
    out_eps: Path | None,
    n_cuts: int,
    map_1_label: str,
    map_2_label: str,
    overlap_label: str,
) -> None:
    if out_png is None and out_pdf is None and out_eps is None:
        return

    cut_coords = _paper_cut_coords(combined_img, n_cuts=n_cuts)
    overlap_img = image.math_img("1.0 * (img == 3)", img=combined_img)
    layout = _paper_layout(anat_img, n_cuts=n_cuts)
    panel_specs = [("x", "Sagittal sections"), ("y", "Coronal sections"), ("z", "Axial sections")]

    def _render_figure(
        out_path: Path,
        *,
        overlap_fill_alpha: float,
        overlap_linewidth: float,
        outline_linewidth: float,
    ) -> None:
        fig, axes = plt.subplots(
            3,
            1,
            figsize=layout["figsize"],
            facecolor="white",
            gridspec_kw={"height_ratios": layout["height_ratios"]},
        )
        plt.subplots_adjust(**layout["subplot_adjust"])

        for ax, (display_mode, _title) in zip(axes, panel_specs):
            display = plotting.plot_roi(
                overlap_img,
                bg_img=anat_img,
                axes=ax,
                display_mode=display_mode,
                cut_coords=cut_coords[display_mode],
                cmap=ListedColormap(
                    [
                        (0.0, 0.0, 0.0, 0.0),
                        (0.55, 0.00, 0.55, 1.0),
                    ]
                ),
                threshold=0.5,
                alpha=overlap_fill_alpha,
                black_bg=False,
                dim=0.0,
                draw_cross=False,
                annotate=False,
                colorbar=False,
            )
            # Draw overlap first so session/state contours remain visible on top,
            # especially when one map is fully contained in the other.
            display.add_contours(
                overlap_img,
                levels=[0.5],
                colors=["#8c109c"],
                linewidths=overlap_linewidth,
            )
            display.add_contours(
                map_2_binary_img,
                levels=[0.5],
                colors=["#2b6ac5"],
                linewidths=outline_linewidth,
            )
            display.add_contours(
                map_1_binary_img,
                levels=[0.5],
                colors=["#db2d2d"],
                linewidths=outline_linewidth,
            )

        legend_handles = [
            Line2D([0], [0], color="#db2d2d", lw=2.2, label=f"{map_1_label} (red)"),
            Line2D([0], [0], color="#2b6ac5", lw=2.2, label=f"{map_2_label} (blue)"),
            Line2D([0], [0], color="#8c109c", lw=2.4, label=f"{overlap_label} (magenta)"),
        ]
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=3,
            frameon=False,
            fontsize=11,
            handlelength=1.6,
            columnspacing=2.0,
        )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_kwargs = {"bbox_inches": "tight", "facecolor": "white"}
        if out_path.suffix.lower() == ".eps":
            save_kwargs["format"] = "eps"
        else:
            save_kwargs["dpi"] = 300
        fig.savefig(out_path, **save_kwargs)
        print(f"Saved paper figure: {out_path}")
        plt.close(fig)

    def _save_eps_from_rendered_figure(
        *,
        out_path: Path,
        overlap_fill_alpha: float,
        overlap_linewidth: float,
        outline_linewidth: float,
    ) -> None:
        eps_anat_img = _anat_with_white_panel_background(anat_img)
        fig, axes = plt.subplots(
            3,
            1,
            figsize=layout["figsize"],
            facecolor="white",
            gridspec_kw={"height_ratios": layout["height_ratios"]},
        )
        plt.subplots_adjust(**layout["subplot_adjust"])

        for ax, (display_mode, _title) in zip(axes, panel_specs):
            display = plotting.plot_roi(
                overlap_img,
                bg_img=eps_anat_img,
                axes=ax,
                display_mode=display_mode,
                cut_coords=cut_coords[display_mode],
                cmap=ListedColormap(
                    [
                        (0.0, 0.0, 0.0, 0.0),
                        (0.55, 0.00, 0.55, 1.0),
                    ]
                ),
                threshold=0.5,
                alpha=overlap_fill_alpha,
                black_bg=False,
                dim=0.0,
                draw_cross=False,
                annotate=False,
                colorbar=False,
            )
            display.add_contours(
                overlap_img,
                levels=[0.5],
                colors=["#8c109c"],
                linewidths=overlap_linewidth,
            )
            display.add_contours(
                map_2_binary_img,
                levels=[0.5],
                colors=["#2b6ac5"],
                linewidths=outline_linewidth,
            )
            display.add_contours(
                map_1_binary_img,
                levels=[0.5],
                colors=["#db2d2d"],
                linewidths=outline_linewidth,
            )

        legend_handles = [
            Line2D([0], [0], color="#db2d2d", lw=2.2, label=f"{map_1_label} (red)"),
            Line2D([0], [0], color="#2b6ac5", lw=2.2, label=f"{map_2_label} (blue)"),
            Line2D([0], [0], color="#8c109c", lw=2.4, label=f"{overlap_label} (magenta)"),
        ]
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=3,
            frameon=False,
            fontsize=11,
            handlelength=1.6,
            columnspacing=2.0,
        )

        png_buffer = BytesIO()
        fig.savefig(png_buffer, format="png", dpi=300, bbox_inches="tight", facecolor="white")
        png_buffer.seek(0)
        raster = plt.imread(png_buffer)
        png_buffer.close()
        plt.close(fig)

        height, width = raster.shape[:2]
        eps_fig = plt.figure(figsize=(width / 300.0, height / 300.0), dpi=300, facecolor="white")
        eps_ax = eps_fig.add_axes([0.0, 0.0, 1.0, 1.0])
        eps_ax.imshow(raster, interpolation="nearest")
        eps_ax.axis("off")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        eps_fig.savefig(out_path, format="eps", dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved paper figure: {out_path}")
        plt.close(eps_fig)

    for out_path in (out_png, out_pdf):
        if out_path is None:
            continue
        _render_figure(
            out_path,
            overlap_fill_alpha=0.0,
            overlap_linewidth=1.25,
            outline_linewidth=1.6,
        )

    if out_eps is not None:
        _save_eps_from_rendered_figure(
            out_path=out_eps,
            overlap_fill_alpha=0.0,
            overlap_linewidth=1.35,
            outline_linewidth=1.7,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Overlay two voxel-weight maps on one anatomical figure."
    )
    parser.add_argument("--map-1", default=DEFAULT_MAP_1, help="Path to first NIfTI map.")
    parser.add_argument("--map-2", default=DEFAULT_MAP_2, help="Path to second NIfTI map.")
    parser.add_argument("--anat", default=DEFAULT_ANAT, help="Path to anatomical NIfTI.")
    parser.add_argument("--out", default=DEFAULT_OUT, help="Output HTML path.")
    parser.add_argument(
        "--out-map-1",
        default=None,
        help="Optional output HTML path for the first map viewer.",
    )
    parser.add_argument(
        "--out-map-2",
        default=None,
        help="Optional output HTML path for the second map viewer.",
    )
    parser.add_argument(
        "--cut-coords",
        nargs=3,
        type=float,
        default=DEFAULT_CUT_COORDS,
        metavar=("X", "Y", "Z"),
        help="Ortho cut coordinates. Defaults to active postcentral center when present.",
    )
    parser.add_argument(
        "--out-paper-png",
        default=None,
        help="Optional output PNG path for a paper-ready static figure.",
    )
    parser.add_argument(
        "--out-paper-pdf",
        default=None,
        help="Optional output PDF path for a paper-ready static figure.",
    )
    parser.add_argument(
        "--out-paper-eps",
        default=None,
        help="Optional output EPS path for a paper-ready static figure.",
    )
    parser.add_argument(
        "--paper-slices",
        type=int,
        default=7,
        help="Number of slices per plane in the static paper figure.",
    )
    parser.add_argument(
        "--paper-only",
        action="store_true",
        help="Only write the static paper figure outputs; skip optional HTML viewers.",
    )
    parser.add_argument(
        "--map-1-label",
        default=None,
        help="Legend/title label for the first map.",
    )
    parser.add_argument(
        "--map-2-label",
        default=None,
        help="Legend/title label for the second map.",
    )
    parser.add_argument(
        "--overlap-label",
        default="Overlap",
        help="Legend/title label for overlapping voxels.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    out_path = Path(args.out)
    out_map_1_path = (
        Path(args.out_map_1)
        if args.out_map_1 is not None
        else out_path.with_name(f"{out_path.stem}_map1.html")
    )
    out_map_2_path = (
        Path(args.out_map_2)
        if args.out_map_2 is not None
        else out_path.with_name(f"{out_path.stem}_map2.html")
    )
    out_paper_png_path = Path(args.out_paper_png) if args.out_paper_png is not None else None
    out_paper_pdf_path = Path(args.out_paper_pdf) if args.out_paper_pdf is not None else None
    out_paper_eps_path = Path(args.out_paper_eps) if args.out_paper_eps is not None else None
    inferred_map_1_label, inferred_map_2_label = _infer_map_labels(args.map_1, args.map_2)
    map_1_label = args.map_1_label or inferred_map_1_label
    map_2_label = args.map_2_label or inferred_map_2_label
    overlap_label = args.overlap_label
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_map_1_path.parent.mkdir(parents=True, exist_ok=True)
    out_map_2_path.parent.mkdir(parents=True, exist_ok=True)
    if out_paper_png_path is not None:
        out_paper_png_path.parent.mkdir(parents=True, exist_ok=True)
    if out_paper_pdf_path is not None:
        out_paper_pdf_path.parent.mkdir(parents=True, exist_ok=True)
    if out_paper_eps_path is not None:
        out_paper_eps_path.parent.mkdir(parents=True, exist_ok=True)

    anat_img = image.load_img(args.anat)
    paper_anat_img = anat_img
    if Path(args.anat).name == "MNI152_T1_2mm_brain.nii.gz":
        full_mni_img = datasets.load_mni152_template(resolution=2)
        paper_anat_img = image.resample_to_img(
            full_mni_img,
            anat_img,
            interpolation="continuous",
            force_resample=True,
            copy_header=True,
        )
    map_1_img = image.resample_to_img(
        image.load_img(args.map_1),
        anat_img,
        interpolation="continuous",
        force_resample=True,
        copy_header=True,
    )
    map_2_img = image.resample_to_img(
        image.load_img(args.map_2),
        anat_img,
        interpolation="continuous",
        force_resample=True,
        copy_header=True,
    )

    combined_img = image.math_img(
        "1.0 * (img1 > 0) + 2.0 * (img2 > 0)",
        img1=map_1_img,
        img2=map_2_img,
    )
    map_1_binary_img = image.math_img("1.0 * (img > 0)", img=map_1_img)
    map_2_binary_img = image.math_img("1.0 * (img > 0)", img=map_2_img)
    combined_img = image.new_img_like(anat_img, combined_img.get_fdata().astype("float32"))
    map_1_binary_img = image.new_img_like(
        anat_img, map_1_binary_img.get_fdata().astype("float32")
    )
    map_2_binary_img = image.new_img_like(
        anat_img, map_2_binary_img.get_fdata().astype("float32")
    )
    if not args.paper_only:
        cut_coords = _resolve_cut_coords(map_2_binary_img, args.cut_coords)
        title = f"Red: {map_1_label} | Blue: {map_2_label} | Magenta: {overlap_label}"
        view = plotting.view_img(
            combined_img,
            bg_img=anat_img,
            cut_coords=cut_coords,
            threshold=0.5,
            vmax=3.0,
            symmetric_cmap=False,
            cmap=_label_cmap(),
            colorbar=True,
            title=title,
        )
        view.save_as_html(str(out_path))
        map_1_view = plotting.view_img(
            map_1_binary_img,
            bg_img=anat_img,
            cut_coords=cut_coords,
            threshold=0.5,
            vmax=1.0,
            symmetric_cmap=False,
            cmap="Reds",
            colorbar=True,
            title=map_1_label,
        )
        map_1_view.save_as_html(str(out_map_1_path))
        map_2_view = plotting.view_img(
            map_2_binary_img,
            bg_img=anat_img,
            cut_coords=cut_coords,
            threshold=0.5,
            vmax=1.0,
            symmetric_cmap=False,
            cmap="Blues",
            colorbar=True,
            title=map_2_label,
        )
        map_2_view.save_as_html(str(out_map_2_path))
    _save_paper_figure(
        combined_img=combined_img,
        map_1_binary_img=map_1_binary_img,
        map_2_binary_img=map_2_binary_img,
        anat_img=paper_anat_img,
        out_png=out_paper_png_path,
        out_pdf=out_paper_pdf_path,
        out_eps=out_paper_eps_path,
        n_cuts=args.paper_slices,
        map_1_label=map_1_label,
        map_2_label=map_2_label,
        overlap_label=overlap_label,
    )
    if not args.paper_only:
        print(f"Saved HTML viewer: {out_path}")
        print(f"Saved map 1 HTML viewer: {out_map_1_path}")
        print(f"Saved map 2 HTML viewer: {out_map_2_path}")


if __name__ == "__main__":
    main()
