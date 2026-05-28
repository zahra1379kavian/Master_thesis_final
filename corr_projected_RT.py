#!/usr/bin/env python3
"""Plot trial-wise weighted beta projection against behaviour RT."""

from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_BETA_DIR = Path(
    "/mnt/TeamShare/Data_Masterfile/H20-00572_All-Dressed/Zahra-Thesis-Data/fmri_opt_group/results_beta_preprocessed"
)
DEFAULT_BEHAVIOUR_DIR = Path(
    "/mnt/TeamShare/Data_Masterfile/H20-00572_All-Dressed/Zahra-Thesis-Data/fmri_opt_group/behaviour"
)
DEFAULT_WEIGHT_MAP = ROOT / "data" / "voxel_weights_task1_bold0.6_beta0.6_smooth1.25_gamma1.5.nii.gz"
DEFAULT_OUT_DIR = ROOT / "figures" / "projected_RT"
OUTLIER_MODIFIED_Z_THRESHOLD = 3.5
PROJECTION_TRIAL_CHUNK_SIZE = 8

BETA_RE = re.compile(
    r"cleaned_beta_volume_(?P<sub>sub-pd\d+)_ses-(?P<ses>\d+)_run-(?P<run>\d+)\.npy$"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Project clean beta volumes through a voxel-weight map and scatter against RT."
    )
    parser.add_argument("--beta-dir", type=Path, default=DEFAULT_BETA_DIR)
    parser.add_argument("--behaviour-dir", type=Path, default=DEFAULT_BEHAVIOUR_DIR)
    parser.add_argument("--weight-map", type=Path, default=DEFAULT_WEIGHT_MAP)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--behaviour-column",
        type=int,
        default=1,
        help="Zero-based RT column for 2D behaviour arrays; default 1 uses the second column.",
    )
    return parser.parse_args()


def _subject_digits(sub_tag: str) -> str:
    match = re.search(r"(\d+)$", sub_tag)
    if not match:
        raise ValueError(f"Could not parse subject digits from {sub_tag!r}")
    return match.group(1)


def _load_weights(weight_map: Path) -> np.ndarray:
    weights = nib.load(str(weight_map)).get_fdata(dtype=np.float32)
    mask = np.isfinite(weights) & (weights != 0)
    if not np.any(mask):
        raise ValueError(f"No nonzero finite weights found in {weight_map}")
    return weights


def _load_behaviour_rt(path: Path, column: int) -> np.ndarray:
    behaviour = np.asarray(np.load(path), dtype=np.float64)
    if behaviour.ndim == 1:
        return behaviour
    if behaviour.ndim != 2:
        raise ValueError(f"Expected 1D or 2D behaviour array in {path}, got shape {behaviour.shape}")
    if column >= behaviour.shape[1]:
        raise ValueError(f"Behaviour column {column} is not available in {path} with shape {behaviour.shape}")
    return behaviour[:, column]


def _align_trials(projected_signal: np.ndarray, behaviour_rt: np.ndarray, label: str) -> tuple[np.ndarray, np.ndarray]:
    n_projection = projected_signal.shape[0]
    n_behaviour = behaviour_rt.shape[0]
    if n_projection == n_behaviour:
        return projected_signal, behaviour_rt

    n_keep = min(n_projection, n_behaviour)
    warnings.warn(
        f"{label}: projection has {n_projection} trials and behaviour has {n_behaviour}; "
        f"truncating both to {n_keep}.",
        stacklevel=2,
    )
    return projected_signal[:n_keep], behaviour_rt[:n_keep]


def _rt_outlier_mask(rt: pd.Series) -> np.ndarray:
    values = rt.to_numpy(dtype=np.float64)
    finite = np.isfinite(values)
    outliers = np.zeros(values.shape, dtype=bool)
    if np.count_nonzero(finite) < 3:
        return outliers

    median = np.nanmedian(values[finite])
    mad = np.nanmedian(np.abs(values[finite] - median))
    if not np.isfinite(mad) or mad <= 0:
        return outliers

    modified_z = 0.6745 * (values[finite] - median) / mad
    outliers[finite] = np.abs(modified_z) > OUTLIER_MODIFIED_Z_THRESHOLD
    return outliers


def _remove_rt_outliers(df: pd.DataFrame) -> pd.DataFrame:
    outlier_masks = []
    for _, group in df.groupby(["sub", "ses", "run"], sort=False):
        outlier_masks.append(pd.Series(_rt_outlier_mask(group["behaviour_rt"]), index=group.index))

    if not outlier_masks:
        return df

    outliers = pd.concat(outlier_masks).sort_index()
    n_outliers = int(outliers.sum())
    if n_outliers:
        print(f"Removed {n_outliers} RT outlier trials using per-run modified z>{OUTLIER_MODIFIED_Z_THRESHOLD}.")
    return df.loc[~outliers].copy()


def _project_beta(beta_path: Path, weights: np.ndarray) -> np.ndarray:
    beta = np.load(beta_path, mmap_mode="r")
    if beta.ndim != 4:
        raise ValueError(f"Expected 4D beta volume in {beta_path}, got shape {beta.shape}")
    if beta.shape[:3] != weights.shape:
        raise ValueError(
            f"Spatial shape mismatch for {beta_path}: beta {beta.shape[:3]} vs weights {weights.shape}"
        )

    weight_mask = np.isfinite(weights) & (weights != 0)
    selected_weights = weights[weight_mask].astype(np.float64)
    projected_signal = np.full(beta.shape[3], np.nan, dtype=np.float64)

    for start in range(0, beta.shape[3], PROJECTION_TRIAL_CHUNK_SIZE):
        stop = min(start + PROJECTION_TRIAL_CHUNK_SIZE, beta.shape[3])
        selected_beta = np.asarray(beta[weight_mask, start:stop], dtype=np.float64)
        finite_beta = np.isfinite(selected_beta)
        numerator = np.nansum(selected_beta * selected_weights[:, None], axis=0)
        denominator = np.sum(finite_beta * selected_weights[:, None], axis=0)
        chunk_projection = np.full(stop - start, np.nan, dtype=np.float64)
        valid = denominator != 0
        chunk_projection[valid] = numerator[valid] / denominator[valid]
        projected_signal[start:stop] = chunk_projection

    return projected_signal


def _discover_beta_runs(beta_dir: Path) -> list[dict[str, object]]:
    runs = []
    for path in sorted(beta_dir.glob("sub-*/cleaned_beta_volume_sub-pd*_ses-*_run-*.npy")):
        match = BETA_RE.match(path.name)
        if not match:
            warnings.warn(f"Skipping unrecognized beta filename: {path}", stacklevel=2)
            continue
        runs.append(
            {
                "path": path,
                "sub": match.group("sub"),
                "ses": int(match.group("ses")),
                "run": int(match.group("run")),
            }
        )
    if not runs:
        raise FileNotFoundError(f"No clean beta files found under {beta_dir}")
    return runs


def _build_projection_table(
    beta_dir: Path,
    behaviour_dir: Path,
    weights: np.ndarray,
    behaviour_column: int,
) -> pd.DataFrame:
    beta_runs = _discover_beta_runs(beta_dir)
    missing_behaviour = []
    for beta_run in beta_runs:
        sub = str(beta_run["sub"])
        ses = int(beta_run["ses"])
        run = int(beta_run["run"])
        behaviour_path = behaviour_dir / f"PSPD{_subject_digits(sub)}_ses_{ses}_run_{run}.npy"
        if not behaviour_path.exists():
            missing_behaviour.append(f"{sub} ses-{ses} run-{run}: {behaviour_path}")

    if missing_behaviour:
        missing_lines = "\n".join(f"- {item}" for item in missing_behaviour)
        raise FileNotFoundError(f"Missing behaviour files:\n{missing_lines}")

    rows = []
    for beta_run in beta_runs:
        sub = str(beta_run["sub"])
        ses = int(beta_run["ses"])
        run = int(beta_run["run"])
        beta_path = Path(beta_run["path"])

        behaviour_path = behaviour_dir / f"PSPD{_subject_digits(sub)}_ses_{ses}_run_{run}.npy"
        projected_signal = _project_beta(beta_path, weights)
        behaviour_rt = _load_behaviour_rt(behaviour_path, behaviour_column)
        projected_signal, behaviour_rt = _align_trials(
            projected_signal, behaviour_rt, f"{sub} ses-{ses} run-{run}"
        )

        for trial_idx, (projection, rt) in enumerate(zip(projected_signal, behaviour_rt), start=1):
            rows.append(
                {
                    "sub": sub,
                    "ses": ses,
                    "run": run,
                    "trial": trial_idx,
                    "projected_signal": projection,
                    "behaviour_rt": rt,
                }
            )

    if not rows:
        raise ValueError("No projection/behaviour rows were created.")
    return pd.DataFrame(rows)


def _save_subject_scatter(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    subjects = sorted(df["sub"].unique())
    sessions = sorted(df["ses"].unique())
    n_rows = len(subjects)
    n_cols = len(sessions)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.2 * n_cols, 4.2 * n_rows), squeeze=False)
    session_colors = {1: "#4C78A8", 2: "#C23B4B"}

    for row_idx, sub in enumerate(subjects):
        for col_idx, ses in enumerate(sessions):
            ax = axes[row_idx, col_idx]
            panel_df = df[(df["sub"] == sub) & (df["ses"] == ses)]
            if panel_df.empty:
                ax.axis("off")
                continue

            ax.scatter(
                panel_df["projected_signal"],
                panel_df["behaviour_rt"],
                s=22,
                alpha=0.68,
                linewidths=0,
                color=session_colors.get(int(ses), "#7A7A7A"),
            )

            finite = np.isfinite(panel_df["projected_signal"]) & np.isfinite(panel_df["behaviour_rt"])
            n_finite = int(finite.sum())
            corr_text = "r=NA"
            if n_finite >= 3:
                x = panel_df.loc[finite, "projected_signal"].to_numpy(dtype=np.float64)
                y = panel_df.loc[finite, "behaviour_rt"].to_numpy(dtype=np.float64)
                corr = np.corrcoef(x, y)[0, 1]
                corr_text = f"r={corr:.2f}"
                if np.nanmax(x) > np.nanmin(x):
                    slope, intercept = np.polyfit(x, y, deg=1)
                    x_fit = np.linspace(np.nanmin(x), np.nanmax(x), 100)
                    ax.plot(x_fit, slope * x_fit + intercept, color="black", linewidth=1.8)

            ax.set_title(f"{sub} ses-{int(ses)} ({corr_text}, n={n_finite})")
            ax.set_xlabel("Projected signal")
            ax.set_ylabel("Behaviour RT")
            ax.grid(alpha=0.25)

    fig.suptitle("Projected signal vs behaviour RT by subject (RT outliers removed)", y=0.995)
    fig.tight_layout()
    fig.savefig(out_dir / "projected_signal_vs_behaviour_by_subject.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    weights = _load_weights(args.weight_map)
    projection_df = _build_projection_table(
        beta_dir=args.beta_dir,
        behaviour_dir=args.behaviour_dir,
        weights=weights,
        behaviour_column=args.behaviour_column,
    )
    projection_df = _remove_rt_outliers(projection_df)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _save_subject_scatter(projection_df, args.out_dir)

    print(f"Saved scatter plot PNG to {args.out_dir}")


if __name__ == "__main__":
    main()
