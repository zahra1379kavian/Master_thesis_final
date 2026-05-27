#!/usr/bin/env python3
"""Medication-state ROI network distances from optimization-weighted ROIs.

This script mirrors the final figure logic used for
``cross_subject_only_laplacian_spectral_distance_signed_distribution``:

1. define ROI nodes from the optimization-weight map and AAL3 atlas groups,
2. compute mutual-information ROI connectivity for each subject/session,
3. compare session networks with signed Laplacian spectral distance, and
4. plot the cross-subject OFF-OFF, ON-ON, and OFF-ON distance distribution.

The atlas-regions PNG is a visualization, not a voxel-level ROI mask. The
actual ROI node list is read from the sibling ``*_regions.csv`` file and the
voxel masks are reconstructed with the same AAL grouping code used to make
that figure.
"""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression

from threshold_robustness_voxel_network import (
    DEFAULT_AAL_VERSION,
    REFERENCE_THRESHOLD,
    UNASSIGNED_ROI,
    ROIGroup,
    _build_roi_groups,
)


DEFAULT_WEIGHT_MAP = Path("data/voxel_weights_task1_bold1_beta0.75_smooth1.8_gamma1.5.nii.gz")
DEFAULT_ROI_FIGURE = Path(
    "figures/task1_bold1_beta0.75_smooth1.8_gamma1.5_threshold_robustness_atlas_regions.png"
)
DEFAULT_ROI_REGION_TABLE = Path(
    "figures/task1_bold1_beta0.75_smooth1.8_gamma1.5_threshold_robustness_regions.csv"
)
DEFAULT_SESSION_MANIFEST = Path("data/med_effects_session_manifest.csv")
DEFAULT_OUT_DIR = Path("figures/med_effects")
DEFAULT_ATLAS_CACHE_DIR = Path("/home/zkavian/nilearn_data")
DEFAULT_MIN_REPORT_VOXELS = 25
DEFAULT_MI_NEIGHBORS = 3
CONNECTIVITY_METRIC = "mutual_information_ksg"
COMPARISON_METRIC = "laplacian_spectral_distance_signed"


@dataclass(frozen=True)
class WeightedROI:
    name: str
    mask: np.ndarray
    weights: np.ndarray
    n_voxels: int


@dataclass(frozen=True)
class SessionSpec:
    label: str
    subject: str
    session: str
    state: str
    bold_path: Path | None
    timeseries_path: Path | None


def _resolve_path(value: object, base_dir: Path) -> Path | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    path = Path(text).expanduser()
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _default_region_table_for(roi_figure: Path) -> Path:
    name = roi_figure.name
    if name.endswith("_atlas_regions.png"):
        return roi_figure.with_name(name.replace("_atlas_regions.png", "_regions.csv"))
    if name.endswith("_atlas_regions.pdf"):
        return roi_figure.with_name(name.replace("_atlas_regions.pdf", "_regions.csv"))
    return DEFAULT_ROI_REGION_TABLE


def _read_manifest(path: Path) -> list[SessionSpec]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing session manifest: {path}. Expected columns: label, subject, session, state, "
            "and either bold_path or timeseries_path."
        )
    manifest = pd.read_csv(path)
    required = {"subject", "session", "state"}
    missing_columns = sorted(required - set(manifest.columns))
    if missing_columns:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing_columns)}")
    if "bold_path" not in manifest.columns and "timeseries_path" not in manifest.columns:
        raise ValueError(f"{path} must include either a bold_path column or a timeseries_path column")

    specs: list[SessionSpec] = []
    base_dir = path.parent.resolve()
    for row_index, row in manifest.iterrows():
        subject = str(row["subject"]).strip()
        session = str(row["session"]).strip()
        state = str(row["state"]).strip().lower()
        if not subject or not session or not state:
            raise ValueError(f"{path} row {row_index + 2} has an empty subject, session, or state")
        label = str(row.get("label", f"{subject}_ses-{session}")).strip()
        bold_path = _resolve_path(row.get("bold_path"), base_dir)
        timeseries_path = _resolve_path(row.get("timeseries_path"), base_dir)
        if bold_path is None and timeseries_path is None:
            raise ValueError(f"{path} row {row_index + 2} must provide bold_path or timeseries_path")
        specs.append(
            SessionSpec(
                label=label,
                subject=subject,
                session=session,
                state=state,
                bold_path=bold_path,
                timeseries_path=timeseries_path,
            )
        )
    return specs


def _missing_inputs(args: argparse.Namespace) -> list[str]:
    missing: list[str] = []
    if not args.weight_map.exists():
        missing.append(f"{args.weight_map} (optimization-weight NIfTI)")
    if not args.roi_definition_figure.exists():
        missing.append(f"{args.roi_definition_figure} (ROI definition reference figure)")
    if not args.roi_region_table.exists():
        missing.append(
            f"{args.roi_region_table} (machine-readable ROI table; the PNG alone is not a voxel mask)"
        )
    if not args.session_manifest.exists():
        missing.append(
            f"{args.session_manifest} (session manifest with subject, session, state, and bold_path or timeseries_path)"
        )
        return missing

    try:
        specs = _read_manifest(args.session_manifest)
    except (FileNotFoundError, ValueError) as exc:
        missing.append(str(exc))
        return missing

    for spec in specs:
        if spec.timeseries_path is not None and not spec.timeseries_path.exists():
            missing.append(f"{spec.timeseries_path} (ROI time-series CSV for {spec.label})")
        if spec.timeseries_path is None and spec.bold_path is not None and not spec.bold_path.exists():
            missing.append(f"{spec.bold_path} (4D BOLD NIfTI for {spec.label})")
    return missing


def _load_roi_names(region_table: Path, roi_percentile: float, min_report_voxels: int) -> list[str]:
    regions = pd.read_csv(region_table)
    required = {"percentile", "roi_name", "n_voxels"}
    missing = sorted(required - set(regions.columns))
    if missing:
        raise ValueError(f"{region_table} is missing required columns: {', '.join(missing)}")
    rows = regions[
        np.isclose(regions["percentile"].astype(float), float(roi_percentile))
        & ~regions["roi_name"].eq(UNASSIGNED_ROI)
        & (regions["n_voxels"].astype(int) >= int(min_report_voxels))
    ].copy()
    if "present_for_report" in rows.columns:
        rows = rows[rows["present_for_report"].astype(bool)]
    if rows.empty:
        raise ValueError(
            f"No reportable p{roi_percentile:g} ROI rows found in {region_table}; "
            f"try lowering --min-report-voxels."
        )
    rows = rows.sort_values("n_voxels", ascending=False)
    return rows["roi_name"].astype(str).tolist()


def _build_weighted_rois(
    weight_values: np.ndarray,
    roi_names: list[str],
    groups: list[ROIGroup],
    roi_percentile: float,
    min_report_voxels: int,
) -> tuple[list[WeightedROI], float]:
    finite_nonzero = np.isfinite(weight_values) & (weight_values != 0)
    if not np.any(finite_nonzero):
        raise ValueError("No finite nonzero voxels found in the weight map")
    threshold = float(np.percentile(weight_values[finite_nonzero], roi_percentile))
    selected = finite_nonzero & (weight_values >= threshold)
    group_lookup = {group.name: group for group in groups}

    rois: list[WeightedROI] = []
    missing_groups = [name for name in roi_names if name not in group_lookup]
    if missing_groups:
        raise ValueError("ROI table names were not found in the AAL grouping: " + ", ".join(missing_groups))

    for name in roi_names:
        mask = selected & group_lookup[name].mask
        n_voxels = int(np.count_nonzero(mask))
        if n_voxels < min_report_voxels:
            continue
        roi_weights = np.asarray(weight_values[mask], dtype=np.float64)
        roi_weights = np.where(np.isfinite(roi_weights) & (roi_weights > 0), roi_weights, 0.0)
        if float(np.sum(roi_weights)) <= 0.0:
            roi_weights = np.ones(n_voxels, dtype=np.float64)
        rois.append(WeightedROI(name=name, mask=mask, weights=roi_weights, n_voxels=n_voxels))
    if len(rois) < 2:
        raise ValueError("At least two weighted ROI masks are required for edge-network analysis")
    return rois, threshold


def _check_image_grid(reference: nib.Nifti1Image, img: nib.Nifti1Image, label: str) -> None:
    if img.shape[:3] != reference.shape[:3]:
        raise ValueError(f"{label} shape {img.shape[:3]} differs from the weight-map grid {reference.shape[:3]}")
    if not np.allclose(img.affine, reference.affine):
        raise ValueError(f"{label} affine differs from the weight-map affine")


def _weighted_mean_timeseries(data: np.ndarray, roi: WeightedROI) -> np.ndarray:
    roi_data = np.asarray(data[roi.mask, :], dtype=np.float64)
    weights = roi.weights.astype(np.float64, copy=False)
    finite = np.isfinite(roi_data)
    weighted = np.where(finite, roi_data, 0.0) * weights[:, None]
    denom = np.sum(np.where(finite, weights[:, None], 0.0), axis=0)
    out = np.full(roi_data.shape[1], np.nan, dtype=np.float64)
    valid = denom > 0
    out[valid] = np.sum(weighted[:, valid], axis=0) / denom[valid]
    return out


def _extract_roi_timeseries(
    bold_path: Path,
    reference_img: nib.Nifti1Image,
    rois: list[WeightedROI],
) -> pd.DataFrame:
    img = nib.load(str(bold_path))
    _check_image_grid(reference_img, img, str(bold_path))
    if len(img.shape) != 4:
        raise ValueError(f"{bold_path} must be a 4D BOLD NIfTI")
    data = img.get_fdata(dtype=np.float32)
    roi_series = {roi.name: _weighted_mean_timeseries(data, roi) for roi in rois}
    return pd.DataFrame(roi_series)


def _read_roi_timeseries(path: Path, roi_names: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [name for name in roi_names if name not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing ROI time-series columns: {', '.join(missing)}")
    return df.loc[:, roi_names].apply(pd.to_numeric, errors="coerce")


def _load_session_timeseries(
    spec: SessionSpec,
    reference_img: nib.Nifti1Image,
    rois: list[WeightedROI],
) -> pd.DataFrame:
    roi_names = [roi.name for roi in rois]
    if spec.timeseries_path is not None:
        return _read_roi_timeseries(spec.timeseries_path, roi_names)
    if spec.bold_path is None:
        raise ValueError(f"{spec.label} has no bold_path or timeseries_path")
    return _extract_roi_timeseries(spec.bold_path, reference_img, rois)


def _clean_timeseries(df: pd.DataFrame) -> np.ndarray:
    values = df.to_numpy(dtype=np.float64)
    values = values[np.all(np.isfinite(values), axis=1)]
    if values.shape[0] < 4:
        raise ValueError("ROI time series has fewer than four complete time points")
    centered = values - np.mean(values, axis=0, keepdims=True)
    scale = np.std(centered, axis=0, ddof=1)
    valid = scale > 0
    centered[:, valid] /= scale[valid]
    centered[:, ~valid] = 0.0
    return centered


def _mutual_information_matrix(
    timeseries: np.ndarray,
    n_neighbors: int,
    random_state: int,
) -> np.ndarray:
    n_timepoints, n_rois = timeseries.shape
    matrix = np.zeros((n_rois, n_rois), dtype=np.float64)
    if n_rois < 2:
        return matrix
    neighbors = min(int(n_neighbors), max(1, n_timepoints - 1))
    for i, j in itertools.combinations(range(n_rois), 2):
        xi = timeseries[:, [i]]
        xj = timeseries[:, [j]]
        yi = timeseries[:, i]
        yj = timeseries[:, j]
        if np.std(yi) <= 0 or np.std(yj) <= 0:
            score = 0.0
        else:
            mi_ij = mutual_info_regression(
                xi,
                yj,
                discrete_features=False,
                n_neighbors=neighbors,
                random_state=random_state,
            )[0]
            mi_ji = mutual_info_regression(
                xj,
                yi,
                discrete_features=False,
                n_neighbors=neighbors,
                random_state=random_state,
            )[0]
            score = max(0.0, float((mi_ij + mi_ji) / 2.0))
        matrix[i, j] = score
        matrix[j, i] = score
    return matrix


def _signed_laplacian_spectrum(adjacency: np.ndarray) -> np.ndarray:
    matrix = np.asarray(adjacency, dtype=np.float64)
    matrix = 0.5 * (matrix + matrix.T)
    np.fill_diagonal(matrix, 0.0)
    degree = np.sum(np.abs(matrix), axis=1)
    laplacian = np.diag(degree) - matrix
    scale = np.zeros_like(degree)
    active = degree > 0
    scale[active] = 1.0 / np.sqrt(degree[active])
    normalized = laplacian * scale[:, None] * scale[None, :]
    return np.sort(np.linalg.eigvalsh(normalized))


def _laplacian_spectral_distance_signed(left: np.ndarray, right: np.ndarray) -> float:
    left_spectrum = _signed_laplacian_spectrum(left)
    right_spectrum = _signed_laplacian_spectrum(right)
    return float(np.linalg.norm(left_spectrum - right_spectrum) / np.sqrt(left_spectrum.size))


def _pair_label(state_a: str, state_b: str) -> tuple[str, str]:
    left = str(state_a).lower()
    right = str(state_b).lower()
    if left == right:
        return "within_condition", f"{left}-{right}"
    if {left, right} == {"off", "on"}:
        return "between_condition", "off-on"
    return "between_condition", "-".join(sorted([left, right]))


def _pairwise_network_distances(
    specs: list[SessionSpec],
    networks: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for left, right in itertools.combinations(specs, 2):
        pair_class, pair_label = _pair_label(left.state, right.state)
        distance = _laplacian_spectral_distance_signed(networks[left.label], networks[right.label])
        rows.append(
            {
                "connectivity_metric": CONNECTIVITY_METRIC,
                "label_a": left.label,
                "label_b": right.label,
                "subject_a": left.subject,
                "subject_b": right.subject,
                "session_a": left.session,
                "session_b": right.session,
                "state_a": left.state,
                "state_b": right.state,
                "pair_class": pair_class,
                "pair_label": pair_label,
                "same_subject": bool(left.subject == right.subject),
                "comparison_metric": COMPARISON_METRIC,
                "comparison_kind": "graph_distance",
                "higher_is_more_similar": False,
                "raw_score": distance,
                "oriented_score": -distance,
            }
        )
    return pd.DataFrame(rows)


def _plot_cross_subject_distribution(pairwise: pd.DataFrame, out_dir: Path) -> Path:
    subset = pairwise.loc[
        (pairwise["connectivity_metric"] == CONNECTIVITY_METRIC)
        & (pairwise["comparison_metric"] == COMPARISON_METRIC)
        & (~pairwise["same_subject"].astype(bool))
    ].copy()
    class_order = [("OFF-OFF", "off-off"), ("ON-ON", "on-on"), ("OFF-ON", "off-on")]
    colors_by_name = {"OFF-OFF": "#4c78a8", "ON-ON": "#e9a3a3", "OFF-ON": "#54a24b"}
    groups = [
        (name, idx * 1.15, subset.loc[subset["pair_label"] == key, "raw_score"].to_numpy(dtype=np.float64))
        for idx, (name, key) in enumerate(class_order)
    ]
    groups = [(name, pos, values[np.isfinite(values)]) for name, pos, values in groups if np.isfinite(values).any()]
    if not groups:
        raise ValueError("No finite cross-subject pairwise distances were available for plotting")

    fig, ax = plt.subplots(figsize=(5.2, 4.1))
    box = ax.boxplot(
        [values for _, _, values in groups],
        positions=[pos for _, pos, _ in groups],
        widths=0.56,
        patch_artist=True,
        flierprops={"markeredgecolor": "#444444", "markerfacecolor": "#444444", "markersize": 2.5},
    )
    for patch, (name, _, _) in zip(box["boxes"], groups):
        patch.set_facecolor(colors_by_name.get(name, "#7f7f7f"))
        patch.set_alpha(0.55)

    rng = np.random.default_rng(0)
    for _, pos, values in groups:
        ax.scatter(
            rng.normal(loc=pos, scale=0.04, size=values.size),
            values,
            s=14,
            alpha=0.55,
            color="black",
            linewidths=0.0,
        )
    positions = [pos for _, pos, _ in groups]
    ax.set_xticks(positions)
    ax.set_xticklabels([name for name, _, _ in groups])
    ax.set_xlim(min(positions) - 0.48, max(positions) + 0.48)
    ax.set_ylabel("Laplacian spectral distance")
    y_values = np.concatenate([values for _, _, values in groups])
    y_min = float(np.min(y_values))
    y_max = float(np.max(y_values))
    y_span = max(y_max - y_min, 1e-6)
    ax.set_ylim(y_min - 0.05 * y_span, y_max + 0.30 * y_span)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "cross_subject_only_laplacian_spectral_distance_signed_distribution.png"
    pdf_path = png_path.with_suffix(".pdf")
    fig.savefig(png_path, dpi=190, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return png_path


def _save_networks(networks: dict[str, np.ndarray], roi_names: list[str], out_dir: Path) -> None:
    network_dir = out_dir / "network_matrices"
    network_dir.mkdir(parents=True, exist_ok=True)
    for label, matrix in networks.items():
        pd.DataFrame(matrix, index=roi_names, columns=roi_names).to_csv(network_dir / f"{label}.csv")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze medication-state ROI edge-network distances.")
    parser.add_argument("--weight-map", type=Path, default=DEFAULT_WEIGHT_MAP, help="Optimization-weight NIfTI map.")
    parser.add_argument(
        "--roi-definition-figure",
        type=Path,
        default=DEFAULT_ROI_FIGURE,
        help="Atlas-regions PNG used as provenance for the ROI definition.",
    )
    parser.add_argument(
        "--roi-region-table",
        type=Path,
        default=None,
        help="Machine-readable ROI table. Defaults to the sibling *_regions.csv for the ROI figure.",
    )
    parser.add_argument(
        "--roi-percentile",
        type=float,
        default=REFERENCE_THRESHOLD,
        help="Weight percentile used for ROI nodes.",
    )
    parser.add_argument(
        "--min-report-voxels",
        type=int,
        default=DEFAULT_MIN_REPORT_VOXELS,
        help="Minimum suprathreshold voxels required for an ROI node.",
    )
    parser.add_argument(
        "--session-manifest",
        type=Path,
        default=DEFAULT_SESSION_MANIFEST,
        help="CSV with subject, session, state, and bold_path or timeseries_path columns.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory outside results9s.")
    parser.add_argument("--aal-version", default=DEFAULT_AAL_VERSION, help="AAL atlas version passed to nilearn.")
    parser.add_argument(
        "--atlas-cache-dir",
        type=Path,
        default=DEFAULT_ATLAS_CACHE_DIR,
        help="Optional nilearn atlas cache directory.",
    )
    parser.add_argument("--mi-neighbors", type=int, default=DEFAULT_MI_NEIGHBORS, help="KSG nearest-neighbor count.")
    parser.add_argument("--random-state", type=int, default=0, help="Random seed for mutual_info_regression.")
    parser.add_argument("--check-inputs", action="store_true", help="Report missing inputs and exit without analysis.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.roi_region_table is None:
        args.roi_region_table = _default_region_table_for(args.roi_definition_figure)

    missing = _missing_inputs(args)
    if missing:
        print("Missing required inputs:")
        for item in missing:
            print(f"- {item}")
        return 1
    if args.check_inputs:
        print("All required inputs are present.")
        return 0

    weight_img = nib.load(str(args.weight_map))
    weight_values = np.asarray(weight_img.get_fdata(), dtype=np.float64)
    groups, metadata = _build_roi_groups(weight_img, args.aal_version, args.atlas_cache_dir)
    roi_names = _load_roi_names(args.roi_region_table, args.roi_percentile, args.min_report_voxels)
    rois, roi_threshold = _build_weighted_rois(
        weight_values=weight_values,
        roi_names=roi_names,
        groups=groups,
        roi_percentile=args.roi_percentile,
        min_report_voxels=args.min_report_voxels,
    )

    specs = _read_manifest(args.session_manifest)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    roi_names = [roi.name for roi in rois]
    roi_summary = pd.DataFrame(
        {
            "roi_name": [roi.name for roi in rois],
            "n_weighted_voxels": [roi.n_voxels for roi in rois],
            "roi_percentile": float(args.roi_percentile),
            "weight_threshold": roi_threshold,
        }
    )
    roi_summary.to_csv(out_dir / "weighted_roi_definition.csv", index=False)

    timeseries_dir = out_dir / "roi_timeseries"
    timeseries_dir.mkdir(parents=True, exist_ok=True)
    networks: dict[str, np.ndarray] = {}
    for spec in specs:
        session_df = _load_session_timeseries(spec, weight_img, rois)
        session_df.to_csv(timeseries_dir / f"{spec.label}.csv", index=False)
        cleaned = _clean_timeseries(session_df)
        networks[spec.label] = _mutual_information_matrix(
            cleaned,
            n_neighbors=args.mi_neighbors,
            random_state=args.random_state,
        )

    _save_networks(networks, roi_names, out_dir)
    pairwise = _pairwise_network_distances(specs, networks)
    pairwise_path = out_dir / "pairwise_metric_values.csv"
    pairwise.to_csv(pairwise_path, index=False)
    figure_path = _plot_cross_subject_distribution(pairwise, out_dir)

    metadata.update(
        {
            "weight_map": str(args.weight_map),
            "roi_definition_figure": str(args.roi_definition_figure),
            "roi_region_table": str(args.roi_region_table),
            "session_manifest": str(args.session_manifest),
            "roi_percentile": float(args.roi_percentile),
            "weight_threshold": roi_threshold,
            "min_report_voxels": int(args.min_report_voxels),
            "connectivity_metric": CONNECTIVITY_METRIC,
            "comparison_metric": COMPARISON_METRIC,
            "mi_neighbors": int(args.mi_neighbors),
            "sessions": [
                spec.__dict__
                | {
                    "bold_path": str(spec.bold_path) if spec.bold_path else None,
                    "timeseries_path": str(spec.timeseries_path) if spec.timeseries_path else None,
                }
                for spec in specs
            ],
        }
    )
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved {pairwise_path}")
    print(f"Saved {figure_path}")
    print(f"Saved {figure_path.with_suffix('.pdf')}")
    print(f"Saved {out_dir / 'weighted_roi_definition.csv'}")
    print(f"Saved {out_dir / 'metadata.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
