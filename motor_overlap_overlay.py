from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage
from scipy.ndimage import distance_transform_edt

from analyze_ablation_constraints import (
    DEFAULT_TASK_ONLY_MAP,
    DEFAULT_TASK_ONLY_Z_THRESHOLD,
    MOTOR_CONTOUR_ROIS,
    MOTOR_OVERLAP_DISPLAY_DILATION_VOXELS,
    _atlas_roi_mask,
    _brainstem_mask,
    _load_data,
)


def motor_overlap_masks(
    selected_mask: np.ndarray,
    display_affine: np.ndarray,
    task_only_map: Path = DEFAULT_TASK_ONLY_MAP,
    task_z_threshold: float = DEFAULT_TASK_ONLY_Z_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray]:
    task_img, task_data = _load_data(task_only_map)
    if task_img.shape[:3] != selected_mask.shape:
        raise RuntimeError(f"{task_only_map} shape {task_img.shape[:3]} differs from selected mask {selected_mask.shape}.")

    task_mask = np.isfinite(task_data) & (task_data >= task_z_threshold)
    for axis in range(3):
        task_step = float(task_img.affine[axis, axis])
        display_step = float(display_affine[axis, axis])
        if task_step != 0.0 and display_step != 0.0 and np.sign(task_step) != np.sign(display_step):
            task_mask = np.flip(task_mask, axis=axis)

    display_img = nib.Nifti1Image(np.zeros(selected_mask.shape, dtype=np.uint8), display_affine)
    brainstem_mask = _brainstem_mask(display_img)
    motor_mask = _atlas_roi_mask(display_img, MOTOR_CONTOUR_ROIS)
    shared_motor_mask = (selected_mask & ~brainstem_mask) & (task_mask & ~brainstem_mask) & motor_mask
    display_mask = (
        ndimage.binary_dilation(
            shared_motor_mask,
            structure=ndimage.generate_binary_structure(3, 1),
            iterations=MOTOR_OVERLAP_DISPLAY_DILATION_VOXELS,
        )
        & motor_mask
    )
    return display_mask, shared_motor_mask


def fill_from_nearest_selected(
    weights: np.ndarray,
    selected_mask: np.ndarray,
    fill_mask: np.ndarray,
) -> np.ndarray:
    missing_mask = fill_mask & ~selected_mask & ~np.isfinite(weights)
    valid_mask = selected_mask & np.isfinite(weights)
    if not np.any(missing_mask) or not np.any(valid_mask):
        return weights

    filled = weights.copy()
    nearest = distance_transform_edt(~valid_mask, return_distances=False, return_indices=True)
    filled[missing_mask] = weights[tuple(indices[missing_mask] for indices in nearest)]
    return filled
