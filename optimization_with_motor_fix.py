"""Motor-cortex fix wrapper for optimization_problem_core.py

The fix: replace the per-component diagonal C_task with a spatially-specific
rank-1 projection matrix built entirely from the beta values that are already
in the pipeline.

Why the original C_task fails
------------------------------
The original builds  diag[i] = 1 / mean(|beta_component_i|)  where the mean
is taken over ALL voxels projected onto component i.  This averages the task
signal across the whole brain, erasing spatial structure.  The result is a
penalty matrix that cannot distinguish motor cortex from anywhere else.

Fix (beta-values only, no external prior)
------------------------------------------
For each training fold:
  1. Take beta_voxel  (n_voxels × n_train_trials) — same betas the original
     already uses, just before PCA projection.
  2. Compute  mean_abs = nanmean(|beta_voxel|, axis=trials).
     Motor cortex voxels have consistently large |beta|; other regions are
     near zero.  This is a spatial map of task responsiveness derived purely
     from the data.
  3. Project into PCA space:  v_raw = coeff_pinv @ mean_abs.
  4. Normalise: v = v_raw / ||v_raw||.
  5. Build  C_task = I - outer(v, v).
     Eigenvalue 0 in the motor-aligned direction (no penalty there) and ~1
     everywhere else (~20x stronger signal than the original diagonal).
  6. Trace-normalise with standardize_matrix so the scale matches C_bold and
     C_beta.

This is computed per-fold from training-fold betas only, so there is no data
leakage into the test fold and no external spatial prior is injected.

Usage
-----
Run this script in place of optimization_problem_core.py:

    python optimization_with_motor_fix.py [all original CLI args]

or swap it into the SLURM sbatch script.
"""

import os
import sys
import numpy as np

MOTOR_FIXED_RESULTS_DIR = os.path.join(
    os.path.abspath(os.path.dirname(__file__)), "results", "motor_fixed"
)
os.environ.setdefault("FMRI_RESULTS_DIR", MOTOR_FIXED_RESULTS_DIR)

import optimization_problem_core as _opc

standardize_matrix = _opc.standardize_matrix


# ---------------------------------------------------------------------------
# Spatial C_task from voxel-level mean absolute beta
# ---------------------------------------------------------------------------

def _spatial_ctask_from_betas(beta_voxel, coeff_pinv):
    """Return C_task = I - v v^T  built from mean |beta| per voxel.

    Parameters
    ----------
    beta_voxel : ndarray, shape (n_voxels, n_trials)
        Raw voxel-space beta coefficients for the current (training) fold.
    coeff_pinv : ndarray, shape (n_components, n_voxels)
        Pseudo-inverse of the EMPCA coefficient matrix.

    Returns
    -------
    C_task : ndarray (n_components, n_components), trace-normalised, or None
    """
    mean_abs = np.nanmean(np.abs(beta_voxel), axis=-1)   # (n_voxels,)
    mean_abs = np.where(np.isfinite(mean_abs), mean_abs, 0.0)

    v_raw = coeff_pinv @ mean_abs                          # (n_components,)
    norm  = np.linalg.norm(v_raw)
    if norm < 1e-12:
        return None

    v = v_raw / norm
    n = coeff_pinv.shape[0]
    C = np.eye(n) - np.outer(v, v)
    return standardize_matrix(C)


# ---------------------------------------------------------------------------
# Patched prepare_data_func
# ---------------------------------------------------------------------------

_original_prepare_data_func = _opc.prepare_data_func


def _patched_prepare_data_func(projection_data, trial_indices, trial_length,
                                transition_boundaries, normalization_info):
    run_data = _original_prepare_data_func(
        projection_data, trial_indices, trial_length,
        transition_boundaries, normalization_info,
    )

    # Build the spatial C_task from this fold's training-fold betas only
    beta_voxel = projection_data["beta_clean"][:, trial_indices]
    coeff_pinv = projection_data["coeff_pinv"]

    C_task_spatial = _spatial_ctask_from_betas(beta_voxel, coeff_pinv)
    if C_task_spatial is not None:
        run_data["C_task"] = C_task_spatial
    else:
        print("WARNING: spatial C_task could not be built; "
              "falling back to per-fold diagonal C_task.", flush=True)

    return run_data


# ---------------------------------------------------------------------------
# Apply patch and run
# ---------------------------------------------------------------------------

_opc.prepare_data_func = _patched_prepare_data_func

print("optimization_with_motor_fix: prepare_data_func patched "
      "(spatial C_task from voxel-level mean |beta|).", flush=True)

if __name__ == "__main__":
    _opc.main()
