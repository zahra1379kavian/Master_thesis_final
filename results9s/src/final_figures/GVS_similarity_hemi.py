#!/usr/bin/env python3
"""Run GVS similarity analysis with medication-style hemisphere ROI nodes.

This wrapper reuses ``GVS_similarity.py`` for the actual GVS beta-matrix,
trial-similarity, and sham-referenced ROI-delta analyses.  The only intended
change is ROI membership: ROI families are split into left/right nodes using
the same MNI-x rule as ``roi_edge_connectivity.py``.  Mutual information is not
computed here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from group_analysis.connectivity_new.roi_edge_connectivity_requested import (  # noqa: E402
    ALWAYS_EXCLUDED_ROI_PATTERNS,
    _load_roi_names,
    _load_selected_ijk,
)
from group_analysis.main import GVS_similarity as base  # noqa: E402


DEFAULT_OUTPUT_ROOT = _REPO_ROOT / "results" / "connectivity" / "GVS_effects" / "gvs_similarity_hemi"
DEFAULT_SELECTED_VOXELS_PATH = _REPO_ROOT / "results" / "connectivity" / "tmp" / "data" / "selected_voxel_indices.npz"
DEFAULT_BY_GVS_DIR = _REPO_ROOT / "results" / "connectivity" / "GVS_effects" / "data" / "by_gvs_91601"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the GVS similarity analysis with the medication-state "
            "hemisphere-specific ROI node definition."
        )
    )
    parser.add_argument("--by-gvs-dir", type=Path, default=DEFAULT_BY_GVS_DIR)
    parser.add_argument("--selected-voxels-path", type=Path, default=DEFAULT_SELECTED_VOXELS_PATH)
    parser.add_argument("--roi-img", type=Path, default=base.DEFAULT_ROI_IMG)
    parser.add_argument("--roi-summary", type=Path, default=base.DEFAULT_ROI_SUMMARY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--min-roi-voxels", type=int, default=base.MIN_ROI_VOXELS)
    parser.add_argument(
        "--midline-band-mm",
        type=float,
        default=1.0,
        help="Voxels with abs(MNI x) <= this value are treated as midline.",
    )
    parser.add_argument(
        "--target-trials",
        type=int,
        default=base.DEFAULT_TARGET_TRIALS,
        help="Fixed trial-grid width used when resampling ROI rows before comparison.",
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=base.DEFAULT_N_PERMUTATIONS,
        help="Monte Carlo permutations used for each subject-level sham comparison.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=base.DEFAULT_RANDOM_SEED,
        help="Random seed used for subject-level permutation tests.",
    )
    return parser.parse_args()


def build_hemisphere_roi_membership(
    roi_img_path: Path,
    roi_summary_path: Path,
    selected_voxels_path: Path,
    min_roi_voxels: int,
    *,
    midline_band_mm: float,
) -> tuple[list[np.ndarray], list[str], pd.DataFrame]:
    """Build left/right ROI nodes with the medication-network split rule."""
    roi_img = nib.load(str(roi_img_path))
    roi_data = np.asarray(roi_img.get_fdata(), dtype=np.int32)
    selected_ijk = _load_selected_ijk(selected_voxels_path, roi_data.shape)
    x, y, z = selected_ijk.T
    roi_ids_at_selected = roi_data[x, y, z]
    roi_names = _load_roi_names(roi_summary_path)
    selected_coords_mm = nib.affines.apply_affine(roi_img.affine, selected_ijk)
    selected_x_mm = selected_coords_mm[:, 0]

    min_node_voxels = int(max(1, min_roi_voxels))
    midline_band = float(max(0.0, midline_band_mm))
    exclude_patterns = sorted(str(pattern).lower() for pattern in ALWAYS_EXCLUDED_ROI_PATTERNS)

    unique_ids, counts = np.unique(roi_ids_at_selected[roi_ids_at_selected > 0], return_counts=True)
    members: list[np.ndarray] = []
    labels: list[str] = []
    rows: list[dict[str, Any]] = []

    for roi_id, count in zip(unique_ids.tolist(), counts.tolist(), strict=True):
        if int(count) < min_node_voxels:
            continue
        roi_name = roi_names.get(int(roi_id), f"ROI_{int(roi_id)}")
        if any(pattern in roi_name.lower() for pattern in exclude_patterns):
            continue

        base_members = np.flatnonzero(roi_ids_at_selected == int(roi_id)).astype(np.int64, copy=False)
        roi_x = selected_x_mm[base_members]
        left = base_members[roi_x < -midline_band]
        right = base_members[roi_x > midline_band]
        midline = base_members[np.abs(roi_x) <= midline_band]

        if midline.size > 0:
            if left.size >= right.size and left.size > 0:
                left = np.concatenate([left, midline])
            elif right.size > 0:
                right = np.concatenate([right, midline])

        added = 0
        for hemisphere, node_members in (("L", left), ("R", right)):
            if node_members.size < min_node_voxels:
                continue
            members.append(node_members)
            node_name = f"{hemisphere} {roi_name}"
            labels.append(node_name)
            centroid = np.mean(selected_coords_mm[node_members], axis=0)
            rows.append(
                {
                    "roi_index": len(rows) + 1,
                    "base_roi_id": int(roi_id),
                    "hemisphere": hemisphere,
                    "roi_label": node_name,
                    "n_selected_voxels": int(node_members.size),
                    "x_mm": float(centroid[0]),
                    "y_mm": float(centroid[1]),
                    "z_mm": float(centroid[2]),
                }
            )
            added += 1

        if added == 0 and base_members.size >= min_node_voxels:
            members.append(base_members)
            node_name = f"M {roi_name}"
            labels.append(node_name)
            centroid = np.mean(selected_coords_mm[base_members], axis=0)
            rows.append(
                {
                    "roi_index": len(rows) + 1,
                    "base_roi_id": int(roi_id),
                    "hemisphere": "M",
                    "roi_label": node_name,
                    "n_selected_voxels": int(base_members.size),
                    "x_mm": float(centroid[0]),
                    "y_mm": float(centroid[1]),
                    "z_mm": float(centroid[2]),
                }
            )

    return members, labels, pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    roi_meta_holder: dict[str, pd.DataFrame] = {}

    def _patched_build_roi_membership(
        roi_img_path: Path,
        roi_summary_path: Path,
        selected_voxels_path: Path,
        min_roi_voxels: int,
    ) -> tuple[list[np.ndarray], list[str], pd.DataFrame]:
        out = build_hemisphere_roi_membership(
            roi_img_path=roi_img_path,
            roi_summary_path=roi_summary_path,
            selected_voxels_path=selected_voxels_path,
            min_roi_voxels=min_roi_voxels,
            midline_band_mm=float(args.midline_band_mm),
        )
        roi_meta_holder["roi_nodes"] = out[2]
        return out

    base.build_roi_membership = _patched_build_roi_membership
    base.parse_args = lambda: args
    base.main()

    out_dir = args.out_dir.expanduser().resolve()
    common_dir = out_dir / "common"
    roi_meta = roi_meta_holder.get("roi_nodes")
    if roi_meta is not None and not roi_meta.empty:
        roi_meta.to_csv(common_dir / "roi_nodes.csv", index=False)
        roi_meta.to_csv(common_dir / "roi_nodes_hemi_metadata.csv", index=False)

    manifest_path = out_dir / "analysis_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["roi_definition"] = {
            "source": "medication_state_hemisphere_split",
            "split_hemispheres": True,
            "midline_band_mm": float(args.midline_band_mm),
            "min_roi_voxels": int(args.min_roi_voxels),
            "midline_handling": (
                "Voxels with abs(MNI x) <= midline_band_mm were assigned to the larger "
                "hemisphere within the same ROI family; ties go left when left is non-empty."
            ),
            "small_hemisphere_handling": (
                "A hemisphere node was retained only when it contained at least min_roi_voxels "
                "selected voxels; if neither hemisphere passed but the whole ROI did, one M node "
                "was retained."
            ),
            "mutual_information_computed": False,
            "n_nodes": int(roi_meta.shape[0]) if roi_meta is not None else int(manifest.get("n_rois", 0)),
        }
        if roi_meta is not None and not roi_meta.empty:
            manifest["n_rois"] = int(roi_meta.shape[0])
            manifest["roi_labels"] = roi_meta["roi_label"].astype(str).tolist()
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
