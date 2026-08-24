#!/usr/bin/env python3
"""Export per-subject GVS trial features in MATLAB format."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import savemat


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = (
    ROOT / "figures" / "gvs_projection_features_vs_sham" / "gvs_trial_signal_features.csv"
)
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "figures"
    / "gvs_projection_features_vs_sham"
    / "trial_feature_matrices_session1_off"
)

FEATURES = [
    "early_late_change",
    "slope",
    "abs_baseline_response",
    "peak_to_peak",
]
GVS_CODES = [f"gvs-{index:02d}" for index in range(1, 10)]
DISPLAY_LABELS = ["sham", *[f"GVS{index}" for index in range(1, 9)]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Write one 4-by-number-of-trials MAT file per subject, with trials ordered "
            "as sham, GVS1, ..., GVS8."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--session", type=int, default=1)
    parser.add_argument("--medication", default="OFF")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trials = pd.read_csv(args.input)
    required = {
        "subject",
        "session",
        "medication",
        "projected_trial_index",
        "source_trial_index",
        "run",
        "run_trial_index",
        "gvs_code",
        *FEATURES,
    }
    missing = sorted(required - set(trials.columns))
    if missing:
        raise ValueError(f"{args.input} is missing required columns: {', '.join(missing)}")

    trials = trials.loc[
        trials["session"].astype(int).eq(args.session)
        & trials["medication"].astype(str).str.upper().eq(args.medication.upper())
    ].copy()
    if trials.empty:
        raise ValueError(
            f"No trials found for session {args.session} / medication {args.medication}"
        )
    unknown_codes = sorted(set(trials["gvs_code"]) - set(GVS_CODES))
    if unknown_codes:
        raise ValueError(f"Unexpected GVS codes: {', '.join(unknown_codes)}")

    trials["gvs_order"] = pd.Categorical(trials["gvs_code"], GVS_CODES, ordered=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, object]] = []
    suffix = f"session{args.session}_{args.medication.lower()}"

    for subject, subject_trials in trials.groupby("subject", sort=True):
        subject_trials = subject_trials.sort_values(
            ["gvs_order", "projected_trial_index"], kind="stable"
        ).reset_index(drop=True)
        condition_index = subject_trials["gvs_code"].map(
            {code: index for index, code in enumerate(GVS_CODES)}
        )
        if condition_index.isna().any():
            raise RuntimeError(f"Could not map every GVS condition for {subject}")

        condition_index_array = condition_index.to_numpy(dtype=np.int16)
        condition_labels = np.asarray(
            [DISPLAY_LABELS[index] for index in condition_index_array], dtype=object
        )[None, :]
        feature_matrix = subject_trials[FEATURES].to_numpy(dtype=np.float64).T
        output_path = args.output_dir / f"{subject}_gvs_trial_features_{suffix}.mat"
        savemat(
            output_path,
            {
                "feature_matrix": feature_matrix,
                "feature_names": np.asarray(FEATURES, dtype=object)[:, None],
                "condition_labels": condition_labels,
                "condition_codes": subject_trials["gvs_code"].to_numpy(dtype=object)[None, :],
                "gvs_number": condition_index_array[None, :],
                "condition_position": (condition_index_array + 1)[None, :],
                "projected_trial_index": subject_trials["projected_trial_index"].to_numpy(
                    dtype=np.int64
                )[None, :],
                "source_trial_index": subject_trials["source_trial_index"].to_numpy(
                    dtype=np.int64
                )[None, :],
                "run": subject_trials["run"].to_numpy(dtype=np.int16)[None, :],
                "run_trial_index": subject_trials["run_trial_index"].to_numpy(
                    dtype=np.int16
                )[None, :],
                "subject": str(subject),
                "session": np.asarray([[args.session]], dtype=np.int16),
                "medication": args.medication.upper(),
            },
            do_compression=True,
        )

        counts = subject_trials["gvs_code"].value_counts()
        row: dict[str, object] = {
            "subject": subject,
            "file": output_path.name,
            "n_trials": feature_matrix.shape[1],
        }
        row.update(
            {label: int(counts.get(code, 0)) for code, label in zip(GVS_CODES, DISPLAY_LABELS)}
        )
        manifest_rows.append(row)

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(args.output_dir / "manifest.csv", index=False)
    print(
        f"Saved {len(manifest)} subject MAT files ({int(manifest['n_trials'].sum())} trials) "
        f"to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
