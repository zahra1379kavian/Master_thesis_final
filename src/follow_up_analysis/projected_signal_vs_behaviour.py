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


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BETA_DIR = ROOT / "data" / "clean_beta_values"
DEFAULT_BEHAVIOUR_DIR = Path("/home/zkavian/fsl_glm/data/behaviour")
DEFAULT_WEIGHT_MAP = (
    ROOT
    / "data"
    / "ablation"
    / "maps"
    / "voxel_weights_mean_foldavg_sub9_ses1_task0.8_bold0_beta0_smooth0_gamma1_bold_thr90_postcentral_boosted.nii.gz"
)
DEFAULT_OUT_DIR = ROOT / "results" / "follow_up_analysis"

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
    parser.add_argument("--behaviour-column", type=int, default=1, help="Zero-based RT column for 2D behaviour arrays.")
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
        warnings.warn(f"{path.name} is 1D; using it directly as the RT vector.", stacklevel=2)
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
    selected_beta = np.asarray(beta[weight_mask, :], dtype=np.float64)

    finite_beta = np.isfinite(selected_beta)
    numerator = np.nansum(selected_beta * selected_weights[:, None], axis=0)
    denominator = np.sum(finite_beta * selected_weights[:, None], axis=0)
    projected_signal = np.full(beta.shape[3], np.nan, dtype=np.float64)
    valid = denominator != 0
    projected_signal[valid] = numerator[valid] / denominator[valid]
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
    rows = []
    for beta_run in _discover_beta_runs(beta_dir):
        sub = str(beta_run["sub"])
        ses = int(beta_run["ses"])
        run = int(beta_run["run"])
        beta_path = Path(beta_run["path"])

        behaviour_path = behaviour_dir / f"PSPD{_subject_digits(sub)}_ses_{ses}_run_{run}.npy"
        if not behaviour_path.exists():
            raise FileNotFoundError(f"Missing behaviour file for {sub} ses-{ses} run-{run}: {behaviour_path}")

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
    n_cols = 2
    n_rows = int(np.ceil(len(subjects) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.2 * n_cols, 4.8 * n_rows), squeeze=False)
    session_colors = {1: "#4C78A8", 2: "#C23B4B"}

    for ax, sub in zip(axes.ravel(), subjects):
        sub_df = df[df["sub"] == sub]
        for ses, ses_df in sub_df.groupby("ses", sort=True):
            ax.scatter(
                ses_df["projected_signal"],
                ses_df["behaviour_rt"],
                s=22,
                alpha=0.68,
                linewidths=0,
                color=session_colors.get(int(ses), "#7A7A7A"),
                label=f"ses-{int(ses)}",
            )

        finite = np.isfinite(sub_df["projected_signal"]) & np.isfinite(sub_df["behaviour_rt"])
        n_finite = int(finite.sum())
        corr_text = "r=NA"
        if n_finite >= 3:
            corr = np.corrcoef(
                sub_df.loc[finite, "projected_signal"].to_numpy(dtype=np.float64),
                sub_df.loc[finite, "behaviour_rt"].to_numpy(dtype=np.float64),
            )[0, 1]
            corr_text = f"r={corr:.2f}"

        ax.set_title(f"{sub} ({corr_text}, n={n_finite})")
        ax.set_xlabel("Projected signal")
        ax.set_ylabel("Behaviour RT")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=9)

    for ax in axes.ravel()[len(subjects) :]:
        ax.axis("off")

    fig.suptitle("Projected signal vs behaviour RT by subject", y=0.995)
    fig.tight_layout()
    fig.savefig(out_dir / "projected_signal_vs_behaviour_by_subject.pdf", bbox_inches="tight")
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

    args.out_dir.mkdir(parents=True, exist_ok=True)
    projection_df.to_csv(args.out_dir / "projected_signal_vs_behaviour_trials.csv", index=False)
    _save_subject_scatter(projection_df, args.out_dir)

    print(f"Saved {len(projection_df)} paired trial rows to {args.out_dir}")
    print(f"Saved scatter plots to {args.out_dir}")


if __name__ == "__main__":
    main()
