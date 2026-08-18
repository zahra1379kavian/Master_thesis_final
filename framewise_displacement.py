#!/usr/bin/env python3
"""Compute Power framewise displacement and compare sessions 1 and 2.

The expected inputs are FSL MCFLIRT ``.par`` files with rotations (radians)
in the first three columns and translations (mm) in the final three columns.
Multi-echo runs are detected from ``_e1``, ``_e2``, and ``_e3`` filenames.
FD is computed within each echo before the echo-specific FD traces are averaged.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib

warnings.filterwarnings("ignore", message="Unable to import Axes3D.*")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import scipy
from scipy import stats


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = Path(
    "/mnt/TeamShare/Data_Masterfile/H20-00572_All-Dressed/"
    "PRECISIONSTIM_PD_Data_Results/fMRI_preprocessed_data/"
    "Rev_pipeline/derivatives"
)
DEFAULT_OUT_DIR = ROOT / "figures" / "framewise_displacement"
DEFAULT_HEAD_RADIUS_MM = 50.0
DEFAULT_FD_THRESHOLDS_MM = (0.2, 0.5)
SESSION_COLORS = {1: "#2878B5", 2: "#D65F3C"}

MOTION_FILE_RE = re.compile(
    r"^(?P<subject>sub-[^_]+)_ses-(?P<session>\d+)_run-(?P<run>\d+)_"
    r"task-(?P<task>.+?)_bold(?:_e(?P<echo>\d+))?_mc\.par$"
)


@dataclass(frozen=True, order=True)
class RunKey:
    subject: str
    session: int
    run: int


@dataclass
class RunMotion:
    key: RunKey
    task: str
    echoes: tuple[int | None, ...]
    source_paths: tuple[Path, ...]
    fd_mm: np.ndarray
    translation_fd_mm: np.ndarray
    rotation_fd_mm: np.ndarray
    echo_fd_mm: dict[int | None, np.ndarray]
    repetition_time_s: float | None

    @property
    def n_volumes(self) -> int:
        return int(self.fd_mm.size)

    @property
    def n_echoes(self) -> int:
        return len(self.echoes)

    @property
    def aggregation(self) -> str:
        return "single_echo" if self.n_echoes == 1 else "mean_of_echo_specific_fd"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute Power FD for every subject/session/run and perform a paired "
            "session-2 versus session-1 comparison."
        )
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--task", default="mv", help="Task label in the motion filenames.")
    parser.add_argument(
        "--head-radius-mm",
        type=float,
        default=DEFAULT_HEAD_RADIUS_MM,
        help="Radius used to convert rotation (radians) to displacement (default: 50 mm).",
    )
    parser.add_argument(
        "--fd-thresholds-mm",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        default=DEFAULT_FD_THRESHOLDS_MM,
        help="FD thresholds used for descriptive QC (default: 0.2 0.5 mm).",
    )
    parser.add_argument(
        "--subjects",
        nargs="*",
        help=(
            "Optional subject subset, with or without the 'sub-' prefix; at least two "
            "paired subjects are required for the session comparison."
        ),
    )
    parser.add_argument(
        "--no-subject-traces",
        action="store_true",
        help="Skip the individual subject FD time-series figures.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser


def _normalize_subject(subject: str) -> str:
    text = str(subject).strip()
    if text.startswith("sub-"):
        return text
    if text.startswith("pd") and text[2:].isdigit():
        return f"sub-pd{text[2:].zfill(3)}"
    if text.isdigit():
        return f"sub-pd{text.zfill(3)}"
    return f"sub-{text}"


def discover_motion_files(
    data_root: Path,
    task: str,
    subjects: Iterable[str] | None = None,
) -> dict[RunKey, dict[int | None, Path]]:
    if not data_root.is_dir():
        raise FileNotFoundError(f"Motion-data root does not exist: {data_root}")

    keep_subjects = None
    if subjects:
        keep_subjects = {_normalize_subject(subject) for subject in subjects}

    grouped: dict[RunKey, dict[int | None, Path]] = {}
    candidates = sorted(data_root.glob("sub-*/ses-*/func/mc*/*.par"))
    for path in candidates:
        match = MOTION_FILE_RE.match(path.name)
        if match is None or match.group("task") != task:
            continue

        subject = match.group("subject")
        session = int(match.group("session"))
        run = int(match.group("run"))
        if session not in (1, 2):
            continue
        if keep_subjects is not None and subject not in keep_subjects:
            continue

        echo_text = match.group("echo")
        echo = int(echo_text) if echo_text is not None else None
        expected_dir = "mc" if echo is None else f"mc_e{echo}"
        if path.parent.name != expected_dir:
            raise ValueError(
                f"Echo label/directory mismatch for {path}: expected parent {expected_dir}"
            )

        key = RunKey(subject, session, run)
        echo_paths = grouped.setdefault(key, {})
        if echo in echo_paths:
            raise ValueError(
                f"Duplicate motion file for {subject}, session {session}, run {run}, "
                f"echo {echo}: {echo_paths[echo]} and {path}"
            )
        echo_paths[echo] = path

    if not grouped:
        selection = "" if not keep_subjects else f" for {sorted(keep_subjects)}"
        raise RuntimeError(f"No task-{task} MCFLIRT .par files found under {data_root}{selection}")

    for key, echo_paths in grouped.items():
        echoes = set(echo_paths)
        if None in echoes and len(echoes) != 1:
            raise ValueError(f"Mixed single-echo and echo-labelled files for {key}")
        if None not in echoes and echoes != {1, 2, 3}:
            raise ValueError(
                f"Incomplete multi-echo set for {key}: found {sorted(echoes)}, expected [1, 2, 3]"
            )

    if keep_subjects is not None:
        found_subjects = {key.subject for key in grouped}
        missing_subjects = sorted(keep_subjects - found_subjects)
        if missing_subjects:
            raise RuntimeError(f"No motion files found for: {', '.join(missing_subjects)}")
    return grouped


def load_motion_parameters(path: Path) -> np.ndarray:
    try:
        parameters = np.loadtxt(path, dtype=np.float64, ndmin=2)
    except Exception as exc:
        raise ValueError(f"Could not read numeric motion parameters from {path}") from exc
    if parameters.ndim != 2 or parameters.shape[1] != 6:
        raise ValueError(f"Expected an n x 6 MCFLIRT matrix in {path}, got {parameters.shape}")
    if parameters.shape[0] < 2:
        raise ValueError(f"At least two motion rows are required in {path}, got {parameters.shape[0]}")
    if not np.all(np.isfinite(parameters)):
        raise ValueError(f"Non-finite motion parameters found in {path}")
    return parameters


def compute_power_fd(
    parameters: np.ndarray,
    head_radius_mm: float = DEFAULT_HEAD_RADIUS_MM,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return total, translation, and rotation Power FD arrays.

    The first volume has undefined FD because no preceding transform exists and is
    represented by NaN. MCFLIRT columns are assumed to be rotations in radians
    followed by translations in millimetres.
    """

    if parameters.ndim != 2 or parameters.shape[1] != 6:
        raise ValueError(f"Expected an n x 6 motion matrix, got {parameters.shape}")
    if head_radius_mm <= 0 or not np.isfinite(head_radius_mm):
        raise ValueError("head_radius_mm must be a positive finite number")

    differences = np.diff(parameters, axis=0)
    rotation = head_radius_mm * np.sum(np.abs(differences[:, :3]), axis=1)
    translation = np.sum(np.abs(differences[:, 3:]), axis=1)
    total = rotation + translation

    def with_undefined_first(values: np.ndarray) -> np.ndarray:
        return np.concatenate(([np.nan], values.astype(np.float64, copy=False)))

    return with_undefined_first(total), with_undefined_first(translation), with_undefined_first(rotation)


def read_repetition_time(path: Path, run: int) -> float | None:
    """Read the run TR from the sibling FEAT preprocessing design, when present."""

    func_dir = path.parent.parent
    design_path = func_dir / f"design_preproc_run-{run}.fsf"
    if not design_path.is_file():
        return None
    tr_pattern = re.compile(r"^set\s+fmri\(tr\)\s+([0-9.eE+-]+)\s*$")
    for line in design_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = tr_pattern.match(line.strip())
        if match is not None:
            repetition_time_s = float(match.group(1))
            if repetition_time_s <= 0 or not np.isfinite(repetition_time_s):
                raise ValueError(f"Invalid TR in {design_path}: {repetition_time_s}")
            return repetition_time_s
    raise ValueError(f"Could not find 'set fmri(tr)' in {design_path}")


def load_runs(
    grouped_paths: dict[RunKey, dict[int | None, Path]],
    task: str,
    head_radius_mm: float,
) -> tuple[list[RunMotion], pd.DataFrame]:
    runs: list[RunMotion] = []
    inventory_rows: list[dict[str, object]] = []

    for key in sorted(grouped_paths):
        echo_paths = grouped_paths[key]
        echo_order = tuple(sorted(echo_paths, key=lambda value: -1 if value is None else value))
        echo_fd: dict[int | None, np.ndarray] = {}
        translation_arrays: list[np.ndarray] = []
        rotation_arrays: list[np.ndarray] = []
        row_counts: set[int] = set()
        repetition_times: set[float] = set()

        for echo in echo_order:
            path = echo_paths[echo]
            parameters = load_motion_parameters(path)
            fd, translation, rotation = compute_power_fd(parameters, head_radius_mm)
            echo_fd[echo] = fd
            translation_arrays.append(translation)
            rotation_arrays.append(rotation)
            row_counts.add(int(parameters.shape[0]))
            repetition_time_s = read_repetition_time(path, key.run)
            if repetition_time_s is not None:
                repetition_times.add(repetition_time_s)
            inventory_rows.append(
                {
                    "subject": key.subject,
                    "session": key.session,
                    "run": key.run,
                    "task": task,
                    "echo": "single" if echo is None else echo,
                    "n_volumes": int(parameters.shape[0]),
                    "n_columns": int(parameters.shape[1]),
                    "repetition_time_s": repetition_time_s,
                    "source_file": str(path),
                }
            )

        if len(row_counts) != 1:
            shapes = {echo: echo_fd[echo].size for echo in echo_order}
            raise ValueError(f"Echo length mismatch for {key}: {shapes}")
        if len(repetition_times) > 1:
            raise ValueError(f"Inconsistent TR metadata for {key}: {sorted(repetition_times)}")
        repetition_time_s = next(iter(repetition_times), None)

        def average_traces(arrays: list[np.ndarray]) -> np.ndarray:
            stacked = np.stack(arrays, axis=0)
            averaged = np.full(stacked.shape[1], np.nan, dtype=np.float64)
            averaged[1:] = np.mean(stacked[:, 1:], axis=0)
            return averaged

        paths = tuple(echo_paths[echo] for echo in echo_order)
        runs.append(
            RunMotion(
                key=key,
                task=task,
                echoes=echo_order,
                source_paths=paths,
                fd_mm=average_traces([echo_fd[echo] for echo in echo_order]),
                translation_fd_mm=average_traces(translation_arrays),
                rotation_fd_mm=average_traces(rotation_arrays),
                echo_fd_mm=echo_fd,
                repetition_time_s=repetition_time_s,
            )
        )

    return runs, pd.DataFrame(inventory_rows)


def _threshold_column(threshold: float) -> str:
    return str(float(threshold)).replace("-", "minus_").replace(".", "_")


def framewise_table(runs: list[RunMotion]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for motion in runs:
        frames.append(
            pd.DataFrame(
                {
                    "subject": motion.key.subject,
                    "session": motion.key.session,
                    "run": motion.key.run,
                    "task": motion.task,
                    "volume": np.arange(motion.n_volumes, dtype=int),
                    "fd_mm": motion.fd_mm,
                    "translation_fd_mm": motion.translation_fd_mm,
                    "rotation_fd_mm": motion.rotation_fd_mm,
                    "valid_transition": np.arange(motion.n_volumes) > 0,
                    "n_echoes": motion.n_echoes,
                    "echo_aggregation": motion.aggregation,
                    "repetition_time_s": motion.repetition_time_s,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def summarize_runs(runs: list[RunMotion], thresholds: tuple[float, float]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for motion in runs:
        valid_fd = motion.fd_mm[1:]
        row: dict[str, object] = {
            "subject": motion.key.subject,
            "session": motion.key.session,
            "run": motion.key.run,
            "task": motion.task,
            "n_volumes": motion.n_volumes,
            "n_valid_transitions": int(valid_fd.size),
            "n_echoes": motion.n_echoes,
            "echo_aggregation": motion.aggregation,
            "repetition_time_s": motion.repetition_time_s,
            "mean_fd_mm": float(np.mean(valid_fd)),
            "median_fd_mm": float(np.median(valid_fd)),
            "p95_fd_mm": float(np.percentile(valid_fd, 95)),
            "max_fd_mm": float(np.max(valid_fd)),
            "mean_translation_fd_mm": float(np.mean(motion.translation_fd_mm[1:])),
            "mean_rotation_fd_mm": float(np.mean(motion.rotation_fd_mm[1:])),
            "source_files": ";".join(str(path) for path in motion.source_paths),
        }
        for threshold in thresholds:
            suffix = _threshold_column(threshold)
            above = valid_fd > threshold
            row[f"n_fd_gt_{suffix}_mm"] = int(np.count_nonzero(above))
            row[f"percent_fd_gt_{suffix}_mm"] = float(100.0 * np.mean(above))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["subject", "session", "run"]).reset_index(drop=True)


def matched_run_sets(run_summary: pd.DataFrame) -> dict[str, tuple[int, ...]]:
    common_by_subject: dict[str, tuple[int, ...]] = {}
    for subject, rows in run_summary.groupby("subject", sort=True):
        session_1 = set(rows.loc[rows["session"].eq(1), "run"].astype(int))
        session_2 = set(rows.loc[rows["session"].eq(2), "run"].astype(int))
        common = tuple(sorted(session_1 & session_2))
        if common:
            common_by_subject[str(subject)] = common
    if not common_by_subject:
        raise RuntimeError("No subjects have at least one matching run in sessions 1 and 2")
    return common_by_subject


def summarize_subject_sessions(
    run_summary: pd.DataFrame,
    thresholds: tuple[float, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    common_by_subject = matched_run_sets(run_summary)
    rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []

    for subject in sorted(run_summary["subject"].unique()):
        subject_rows = run_summary[run_summary["subject"].eq(subject)]
        common = common_by_subject.get(str(subject), ())
        available_1 = tuple(sorted(subject_rows.loc[subject_rows["session"].eq(1), "run"].astype(int)))
        available_2 = tuple(sorted(subject_rows.loc[subject_rows["session"].eq(2), "run"].astype(int)))
        excluded_1 = tuple(run for run in available_1 if run not in common)
        excluded_2 = tuple(run for run in available_2 if run not in common)
        audit_rows.append(
            {
                "subject": subject,
                "session_1_available_runs": ";".join(map(str, available_1)),
                "session_2_available_runs": ";".join(map(str, available_2)),
                "matched_runs_used": ";".join(map(str, common)),
                "session_1_unmatched_runs_excluded": ";".join(map(str, excluded_1)),
                "session_2_unmatched_runs_excluded": ";".join(map(str, excluded_2)),
                "included_in_primary": bool(common),
                "included_in_complete_two_run_sensitivity": len(common) >= 2,
            }
        )
        if not common:
            continue

        for session in (1, 2):
            selected = subject_rows[
                subject_rows["session"].eq(session) & subject_rows["run"].isin(common)
            ]
            row: dict[str, object] = {
                "subject": subject,
                "session": session,
                "matched_runs_used": ";".join(map(str, common)),
                "n_matched_runs": len(common),
                "n_valid_transitions": int(selected["n_valid_transitions"].sum()),
                "mean_fd_mm": float(selected["mean_fd_mm"].mean()),
                "mean_run_median_fd_mm": float(selected["median_fd_mm"].mean()),
                "mean_run_p95_fd_mm": float(selected["p95_fd_mm"].mean()),
                "max_fd_mm": float(selected["max_fd_mm"].max()),
                "mean_translation_fd_mm": float(selected["mean_translation_fd_mm"].mean()),
                "mean_rotation_fd_mm": float(selected["mean_rotation_fd_mm"].mean()),
                "multi_echo_subject": bool((selected["n_echoes"] > 1).any()),
            }
            for threshold in thresholds:
                suffix = _threshold_column(threshold)
                column = f"percent_fd_gt_{suffix}_mm"
                row[f"mean_run_{column}"] = float(selected[column].mean())
            rows.append(row)

    return (
        pd.DataFrame(rows).sort_values(["subject", "session"]).reset_index(drop=True),
        pd.DataFrame(audit_rows).sort_values("subject").reset_index(drop=True),
    )


def exact_or_monte_carlo_sign_flip(
    differences: np.ndarray,
    random_state: int = 0,
    monte_carlo_resamples: int = 100_000,
) -> tuple[float, str, int]:
    differences = np.asarray(differences, dtype=np.float64)
    differences = differences[np.isfinite(differences)]
    n = differences.size
    if n == 0:
        return np.nan, "unavailable", 0
    observed = abs(float(np.mean(differences)))
    tolerance = np.finfo(float).eps * max(1.0, observed) * 32

    if n <= 20:
        n_permutations = 1 << n
        exceedances = 0
        bit_positions = np.arange(n, dtype=np.uint64)
        chunk_size = 65_536
        for start in range(0, n_permutations, chunk_size):
            stop = min(start + chunk_size, n_permutations)
            indices = np.arange(start, stop, dtype=np.uint64)[:, None]
            signs = np.where(((indices >> bit_positions) & 1) == 0, 1.0, -1.0)
            permuted = np.abs((signs @ differences) / n)
            exceedances += int(np.count_nonzero(permuted >= observed - tolerance))
        return exceedances / n_permutations, "exact", n_permutations

    rng = np.random.default_rng(random_state)
    exceedances = 0
    completed = 0
    chunk_size = 10_000
    while completed < monte_carlo_resamples:
        size = min(chunk_size, monte_carlo_resamples - completed)
        signs = rng.choice((-1.0, 1.0), size=(size, n))
        permuted = np.abs((signs @ differences) / n)
        exceedances += int(np.count_nonzero(permuted >= observed - tolerance))
        completed += size
    p_value = (exceedances + 1) / (monte_carlo_resamples + 1)
    return p_value, "monte_carlo", monte_carlo_resamples


def paired_statistics(
    session_1: pd.Series,
    session_2: pd.Series,
    analysis: str,
    cohort: str,
    primary: bool,
) -> dict[str, object]:
    paired = pd.concat(
        [session_1.rename("session_1"), session_2.rename("session_2")], axis=1
    ).dropna()
    if len(paired) < 2:
        raise RuntimeError(f"At least two paired subjects are needed for {analysis}")

    differences = (paired["session_2"] - paired["session_1"]).to_numpy(dtype=float)
    n = int(differences.size)
    mean_difference = float(np.mean(differences))
    sd_difference = float(np.std(differences, ddof=1))
    sem_difference = float(stats.sem(differences))
    t_critical = float(stats.t.ppf(0.975, df=n - 1))
    ci_low = mean_difference - t_critical * sem_difference
    ci_high = mean_difference + t_critical * sem_difference
    t_result = stats.ttest_rel(paired["session_2"], paired["session_1"])
    sign_flip_p, sign_flip_method, sign_flip_resamples = exact_or_monte_carlo_sign_flip(differences)

    if np.allclose(differences, 0.0):
        wilcoxon_statistic, wilcoxon_p = 0.0, 1.0
    else:
        wilcoxon = stats.wilcoxon(differences, alternative="two-sided", method="auto")
        wilcoxon_statistic, wilcoxon_p = float(wilcoxon.statistic), float(wilcoxon.pvalue)

    shapiro_statistic, shapiro_p = (np.nan, np.nan)
    if 3 <= n <= 5_000:
        shapiro = stats.shapiro(differences)
        shapiro_statistic, shapiro_p = float(shapiro.statistic), float(shapiro.pvalue)

    mean_session_1 = float(paired["session_1"].mean())
    mean_session_2 = float(paired["session_2"].mean())
    relative_change = (
        float(100 * mean_difference / mean_session_1)
        if not np.isclose(mean_session_1, 0.0)
        else np.nan
    )
    return {
        "analysis": analysis,
        "cohort": cohort,
        "is_primary": primary,
        "outcome": "subject-level equal-weight mean of matched-run mean Power FD (mm)",
        "contrast": "session_2_minus_session_1",
        "n_subjects": n,
        "subjects": ";".join(map(str, paired.index)),
        "session_1_mean_fd_mm": mean_session_1,
        "session_1_sd_fd_mm": float(paired["session_1"].std(ddof=1)),
        "session_2_mean_fd_mm": mean_session_2,
        "session_2_sd_fd_mm": float(paired["session_2"].std(ddof=1)),
        "mean_difference_mm": mean_difference,
        "difference_95ci_low_mm": float(ci_low),
        "difference_95ci_high_mm": float(ci_high),
        "relative_group_mean_change_percent": relative_change,
        "paired_t_statistic": float(t_result.statistic),
        "paired_t_df": n - 1,
        "paired_t_p_two_sided": float(t_result.pvalue),
        "cohen_dz": float(mean_difference / sd_difference) if sd_difference > 0 else np.nan,
        "sign_flip_p_two_sided": float(sign_flip_p),
        "sign_flip_method": sign_flip_method,
        "sign_flip_resamples": sign_flip_resamples,
        "wilcoxon_statistic": wilcoxon_statistic,
        "wilcoxon_p_two_sided": wilcoxon_p,
        "shapiro_difference_statistic": shapiro_statistic,
        "shapiro_difference_p": shapiro_p,
        "n_increased": int(np.count_nonzero(differences > 0)),
        "n_decreased": int(np.count_nonzero(differences < 0)),
        "n_unchanged": int(np.count_nonzero(differences == 0)),
        "paired_t_significant_at_alpha_0_05": bool(float(t_result.pvalue) < 0.05),
    }


def compare_sessions(subject_session_summary: pd.DataFrame) -> pd.DataFrame:
    pivot = subject_session_summary.pivot(index="subject", columns="session", values="mean_fd_mm")
    if 1 not in pivot or 2 not in pivot:
        raise RuntimeError("Both session 1 and session 2 are required for the paired comparison")

    results = [
        paired_statistics(
            pivot[1],
            pivot[2],
            analysis="primary_matched_runs",
            cohort="all subjects; equal-weight mean over run IDs present in both sessions",
            primary=True,
        )
    ]

    complete_subjects = subject_session_summary.loc[
        subject_session_summary["n_matched_runs"].ge(2), "subject"
    ].unique()
    complete_pivot = pivot.loc[pivot.index.intersection(complete_subjects)]
    if len(complete_pivot) >= 2:
        results.append(
            paired_statistics(
                complete_pivot[1],
                complete_pivot[2],
                analysis="sensitivity_at_least_two_matched_runs",
                cohort="subjects with at least two matched runs in both sessions",
                primary=False,
            )
        )

    multi_echo_by_subject = subject_session_summary.groupby("subject")["multi_echo_subject"].any()
    single_echo_subjects = multi_echo_by_subject.index[~multi_echo_by_subject]
    single_echo_pivot = pivot.loc[pivot.index.intersection(single_echo_subjects)]
    if len(single_echo_pivot) >= 2:
        results.append(
            paired_statistics(
                single_echo_pivot[1],
                single_echo_pivot[2],
                analysis="sensitivity_single_echo_only",
                cohort="single-echo subjects only",
                primary=False,
            )
        )
    return pd.DataFrame(results)


def _mean_ci(values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    mean = float(np.mean(values))
    if values.size < 2:
        return mean, mean, mean
    half_width = float(stats.t.ppf(0.975, values.size - 1) * stats.sem(values))
    return mean, mean - half_width, mean + half_width


def _save_figure(fig: plt.Figure, output_base: Path, dpi: int, pdf: bool = True) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=dpi, bbox_inches="tight", facecolor="white")
    if pdf:
        fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def remove_stale_trace_outputs(output_dir: Path) -> None:
    """Remove only subject trace PNGs owned by this script from an earlier run."""

    if not output_dir.is_dir():
        return
    for path in output_dir.glob("sub-*_power_fd_timeseries.png"):
        path.unlink()


def plot_session_comparison(
    subject_session_summary: pd.DataFrame,
    primary_stats: pd.Series,
    output_base: Path,
    dpi: int,
    head_radius_mm: float,
) -> None:
    pivot = subject_session_summary.pivot(index="subject", columns="session", values="mean_fd_mm").dropna()
    differences = (pivot[2] - pivot[1]).sort_values()

    fig = plt.figure(figsize=(13.4, 8.0))
    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=(0.9, 1.35),
        left=0.065,
        right=0.985,
        bottom=0.145,
        top=0.82,
        wspace=0.25,
    )
    ax_pair = fig.add_subplot(grid[0, 0])
    ax_change = fig.add_subplot(grid[0, 1])

    for subject, row in pivot.iterrows():
        ax_pair.plot([0, 1], [row[1], row[2]], color="#A8ADB3", alpha=0.65, lw=1.1, zorder=1)
        ax_pair.scatter(0, row[1], color=SESSION_COLORS[1], s=38, alpha=0.9, zorder=2)
        ax_pair.scatter(1, row[2], color=SESSION_COLORS[2], s=38, alpha=0.9, zorder=2)

    for x, session in enumerate((1, 2)):
        mean, ci_low, ci_high = _mean_ci(pivot[session].to_numpy())
        ax_pair.errorbar(
            x,
            mean,
            yerr=[[mean - ci_low], [ci_high - mean]],
            fmt="D",
            ms=8,
            color="black",
            mfc=SESSION_COLORS[session],
            mec="black",
            capsize=5,
            lw=2.2,
            zorder=4,
        )

    ax_pair.set_xticks([0, 1], ["Session 1", "Session 2"])
    ax_pair.set_xlim(-0.35, 1.35)
    ax_pair.set_ylim(bottom=0)
    ax_pair.set_ylabel("Subject-level mean Power FD (mm)")
    ax_pair.set_title("Paired session means", loc="left", fontweight="bold")
    ax_pair.grid(axis="y", color="#E4E6E8", lw=0.8)
    ax_pair.spines[["top", "right"]].set_visible(False)

    y_positions = np.arange(len(differences))
    colors = np.where(differences.to_numpy() >= 0, SESSION_COLORS[2], SESSION_COLORS[1])
    ax_change.barh(y_positions, differences, color=colors, alpha=0.86, height=0.72)
    ax_change.axvline(0, color="#333333", lw=1.1)
    mean_difference = float(primary_stats["mean_difference_mm"])
    ci_low = float(primary_stats["difference_95ci_low_mm"])
    ci_high = float(primary_stats["difference_95ci_high_mm"])
    ax_change.axvspan(ci_low, ci_high, color="#222222", alpha=0.09, zorder=0)
    ax_change.axvline(mean_difference, color="#222222", lw=2, ls="--")
    ax_change.set_yticks(y_positions, differences.index)
    ax_change.set_xlabel("Change in mean FD (session 2 − session 1, mm)")
    ax_change.set_title("Paired change by subject", loc="left", fontweight="bold")
    ax_change.grid(axis="x", color="#E4E6E8", lw=0.8)
    ax_change.spines[["top", "right", "left"]].set_visible(False)
    ax_change.tick_params(axis="y", length=0, labelsize=9)

    p_value = float(primary_stats["paired_t_p_two_sided"])
    direction = "increased" if mean_difference > 0 else "decreased"
    if p_value < 0.05:
        headline = f"Mean head motion significantly {direction} in session 2"
    else:
        headline = "No statistically significant change in mean head motion"
    fig.suptitle(headline, fontsize=18, fontweight="bold", y=0.975)
    fig.text(
        0.5,
        0.915,
        (
            f"Power FD, {head_radius_mm:g}-mm radius · matched runs · N={int(primary_stats['n_subjects'])} · "
            f"paired t({int(primary_stats['paired_t_df'])})={float(primary_stats['paired_t_statistic']):.2f}, "
            f"p={p_value:.3f}"
        ),
        ha="center",
        va="top",
        fontsize=11,
        color="#444444",
    )
    fig.text(
        0.5,
        0.025,
        (
            f"Session means: {float(primary_stats['session_1_mean_fd_mm']):.3f} vs "
            f"{float(primary_stats['session_2_mean_fd_mm']):.3f} mm; mean paired change "
            f"{mean_difference:+.3f} mm (95% CI {ci_low:+.3f} to {ci_high:+.3f}); "
            f"Cohen dz={float(primary_stats['cohen_dz']):.2f}; "
            f"{str(primary_stats['sign_flip_method']).replace('_', ' ')} sign-flip "
            f"p={float(primary_stats['sign_flip_p_two_sided']):.3f}; Wilcoxon "
            f"p={float(primary_stats['wilcoxon_p_two_sided']):.3f}. Diamonds show means and 95% CIs."
        ),
        ha="center",
        va="bottom",
        fontsize=9.5,
        color="#444444",
    )
    _save_figure(fig, output_base, dpi=dpi, pdf=True)


def plot_paired_session_means_only(
    subject_session_summary: pd.DataFrame,
    output_base: Path,
    dpi: int,
) -> None:
    """Save the paired-means panel as a standalone, title-free figure."""

    pivot = subject_session_summary.pivot(
        index="subject", columns="session", values="mean_fd_mm"
    ).dropna()
    fig, ax = plt.subplots(figsize=(6.4, 7.2))
    fig.subplots_adjust(left=0.2, right=0.97, bottom=0.13, top=0.98)

    for _, row in pivot.iterrows():
        ax.plot(
            [0, 1],
            [row[1], row[2]],
            color="#A8ADB3",
            alpha=0.7,
            lw=1.7,
            zorder=1,
        )
        ax.scatter(0, row[1], color=SESSION_COLORS[1], s=58, alpha=0.9, zorder=2)
        ax.scatter(1, row[2], color=SESSION_COLORS[2], s=58, alpha=0.9, zorder=2)

    for x, session in enumerate((1, 2)):
        mean, ci_low, ci_high = _mean_ci(pivot[session].to_numpy())
        ax.errorbar(
            x,
            mean,
            yerr=[[mean - ci_low], [ci_high - mean]],
            fmt="D",
            ms=10,
            color="black",
            mfc=SESSION_COLORS[session],
            mec="black",
            mew=1.4,
            capsize=7,
            capthick=2.4,
            lw=2.8,
            zorder=4,
        )

    ax.set_xticks([0, 1], ["Session 1", "Session 2"])
    ax.set_xlim(-0.35, 1.35)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Subject-level mean FD (mm)", fontsize=18)
    ax.tick_params(axis="x", labelsize=17, width=1.4, length=6)
    ax.tick_params(axis="y", labelsize=15, width=1.4, length=5)
    ax.grid(axis="y", color="#E4E6E8", lw=1.0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_linewidth(1.4)
    _save_figure(fig, output_base, dpi=dpi, pdf=True)


def plot_run_heatmap(run_summary: pd.DataFrame, output_base: Path, dpi: int) -> None:
    subjects = sorted(run_summary["subject"].unique())
    run_ids = sorted(run_summary["run"].astype(int).unique())
    columns = [(session, run) for session in (1, 2) for run in run_ids]
    matrix = np.full((len(subjects), len(columns)), np.nan, dtype=float)
    lookup = run_summary.set_index(["subject", "session", "run"])["mean_fd_mm"]
    for row_index, subject in enumerate(subjects):
        for column_index, (session, run) in enumerate(columns):
            key = (subject, session, run)
            if key in lookup.index:
                matrix[row_index, column_index] = float(lookup.loc[key])

    cmap = matplotlib.colormaps["viridis"].copy()
    cmap.set_bad("#D9D9D9")
    vmax = max(0.5, float(np.nanmax(matrix)))
    fig_height = max(7.2, 0.42 * len(subjects) + 1.8)
    fig_width = max(8.6, 1.65 * len(columns) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    fig.subplots_adjust(left=0.18, right=0.84, bottom=0.135, top=0.91)
    image = ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap, vmin=0, vmax=vmax, aspect="auto")

    multi_echo_subjects = set(
        run_summary.loc[run_summary["n_echoes"].gt(1), "subject"].astype(str)
    )
    subject_labels = [f"{subject} *" if subject in multi_echo_subjects else subject for subject in subjects]
    ax.set_yticks(np.arange(len(subjects)), subject_labels)
    ax.set_xticks(np.arange(len(columns)), [f"Session {s}\nRun {r}" for s, r in columns])
    ax.tick_params(length=0)
    ax.set_title("Mean Power FD for every available run", fontsize=16, fontweight="bold", pad=14)

    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            if np.isfinite(value):
                color = "white" if value / vmax > 0.53 else "#1F2328"
                ax.text(column, row, f"{value:.3f}", ha="center", va="center", color=color, fontsize=9)
            else:
                ax.text(column, row, "missing", ha="center", va="center", color="#666666", fontsize=8)

    colorbar = fig.colorbar(image, ax=ax, shrink=0.84, pad=0.035)
    colorbar.set_label("Mean Power FD (mm)")
    fig.text(
        0.5,
        0.025,
        "* Three-echo subject; values are framewise means of echo-specific FD. First-volume FD is excluded.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    _save_figure(fig, output_base, dpi=dpi, pdf=True)


def plot_subject_traces(
    runs: list[RunMotion],
    output_dir: Path,
    thresholds: tuple[float, float],
    dpi: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    by_subject: dict[str, dict[tuple[int, int], RunMotion]] = {}
    for motion in runs:
        by_subject.setdefault(motion.key.subject, {})[(motion.key.session, motion.key.run)] = motion
    run_ids = sorted({motion.key.run for motion in runs})

    for subject, subject_runs in sorted(by_subject.items()):
        finite_values = np.concatenate(
            [motion.fd_mm[1:][np.isfinite(motion.fd_mm[1:])] for motion in subject_runs.values()]
        )
        robust_cap = float(np.percentile(finite_values, 99.5))
        y_cap = max(thresholds[1] * 1.15, robust_cap * 1.12)
        true_max = float(np.max(finite_values))

        fig_width = max(8.0, 6.8 * len(run_ids))
        fig, axes = plt.subplots(2, len(run_ids), figsize=(fig_width, 8.3), sharey=True, squeeze=False)
        fig.subplots_adjust(left=0.065, right=0.985, bottom=0.11, top=0.82, hspace=0.34, wspace=0.04)
        for row, session in enumerate((1, 2)):
            for column, run in enumerate(run_ids):
                ax = axes[row, column]
                motion = subject_runs.get((session, run))
                if motion is None:
                    ax.set_facecolor("#F0F0F0")
                    ax.text(0.5, 0.5, "Run not available", ha="center", va="center", transform=ax.transAxes)
                    ax.set_title(f"Session {session} · Run {run}", loc="left", fontweight="bold")
                    ax.set_xticks([])
                    ax.tick_params(axis="y", left=False, labelleft=False)
                    continue

                volumes = np.arange(motion.n_volumes)
                if motion.n_echoes > 1:
                    for echo, echo_fd in motion.echo_fd_mm.items():
                        clipped_echo = np.minimum(echo_fd, y_cap)
                        ax.plot(volumes, clipped_echo, color="#81878E", alpha=0.27, lw=0.7)

                clipped = np.minimum(motion.fd_mm, y_cap)
                ax.plot(volumes, clipped, color=SESSION_COLORS[session], lw=1.0, alpha=0.95)
                clipped_indices = np.flatnonzero(motion.fd_mm > y_cap)
                if clipped_indices.size:
                    ax.scatter(
                        clipped_indices,
                        np.full(clipped_indices.size, y_cap * 0.985),
                        marker="v",
                        s=19,
                        color="#A20D24",
                        zorder=5,
                    )
                ax.axhline(thresholds[0], color="#E69F00", lw=0.9, ls="--", alpha=0.85)
                ax.axhline(thresholds[1], color="#B2182B", lw=0.9, ls=":", alpha=0.9)
                mean_fd = float(np.mean(motion.fd_mm[1:]))
                max_fd = float(np.max(motion.fd_mm[1:]))
                echo_note = " · 3-echo mean" if motion.n_echoes > 1 else ""
                ax.set_title(
                    f"Session {session} · Run {run}{echo_note}\nmean {mean_fd:.3f} mm · max {max_fd:.2f} mm",
                    loc="left",
                    fontsize=11,
                    fontweight="bold",
                )
                ax.set_xlim(0, motion.n_volumes - 1)
                ax.set_ylim(0, y_cap)
                ax.grid(axis="y", color="#E6E7E9", lw=0.6)
                ax.spines[["top", "right"]].set_visible(False)
                if row == 1:
                    ax.set_xlabel("Volume index")
                if column == 0:
                    ax.set_ylabel("Power FD (mm)")

        legend_handles = [
            Line2D([0], [0], color=SESSION_COLORS[1], lw=1.5, label="FD trace"),
            Line2D([0], [0], color="#E69F00", lw=1, ls="--", label=f"{thresholds[0]:g} mm"),
            Line2D([0], [0], color="#B2182B", lw=1, ls=":", label=f"{thresholds[1]:g} mm"),
        ]
        if any(motion.n_echoes > 1 for motion in subject_runs.values()):
            legend_handles.insert(1, Line2D([0], [0], color="#81878E", alpha=0.4, lw=1, label="Individual echoes"))
        if true_max > y_cap:
            legend_handles.append(
                Line2D([0], [0], marker="v", color="#A20D24", lw=0, label=f"Spike above display cap ({y_cap:.2f} mm)")
            )
        fig.legend(handles=legend_handles, loc="upper center", ncol=len(legend_handles), frameon=False, bbox_to_anchor=(0.5, 0.915))
        fig.suptitle(f"{subject}: framewise head motion", fontsize=17, fontweight="bold", y=0.985)
        fig.text(
            0.5,
            0.025,
            "FD is undefined at the first volume of each run. Display cap is based on the subject-specific 99.5th percentile; markers flag averaged-FD spikes above it.",
            ha="center",
            fontsize=8.5,
            color="#444444",
        )
        _save_figure(fig, output_dir / f"{subject}_power_fd_timeseries", dpi=dpi, pdf=False)


def _json_value(value: object) -> object:
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    return value


def write_report(
    output_path: Path,
    data_root: Path,
    run_summary: pd.DataFrame,
    inclusion_audit: pd.DataFrame,
    statistics_table: pd.DataFrame,
    head_radius_mm: float,
    thresholds: tuple[float, float],
) -> None:
    primary = statistics_table.loc[statistics_table["is_primary"]].iloc[0]
    p_value = float(primary["paired_t_p_two_sided"])
    interpretation = (
        "Mean FD changed significantly between sessions at alpha=0.05."
        if p_value < 0.05
        else "Mean FD did not change significantly between sessions at alpha=0.05."
    )
    missing_descriptions: list[str] = []
    for _, row in inclusion_audit.iterrows():
        for session in (1, 2):
            excluded = str(row[f"session_{session}_unmatched_runs_excluded"]).strip()
            if excluded:
                missing_descriptions.append(
                    f"{row['subject']}: session {session} run(s) {excluded} excluded because unmatched"
                )
    missing_text = "; ".join(missing_descriptions) if missing_descriptions else "none"

    sensitivity_lines = []
    for _, row in statistics_table.loc[~statistics_table["is_primary"]].iterrows():
        sensitivity_lines.append(
            f"- {row['analysis']}: N={int(row['n_subjects'])}, "
            f"delta={float(row['mean_difference_mm']):+.4f} mm, "
            f"t({int(row['paired_t_df'])})={float(row['paired_t_statistic']):.3f}, "
            f"p={float(row['paired_t_p_two_sided']):.4f}."
        )

    tr_rows = run_summary.dropna(subset=["repetition_time_s"])
    if tr_rows.empty:
        tr_description = "unavailable"
        tr_consistency = "TR metadata were unavailable, so within-subject consistency could not be checked."
    else:
        tr_parts = []
        tr_rows = tr_rows.assign(
            repetition_time_group_s=tr_rows["repetition_time_s"].round(3)
        )
        for repetition_time_s, rows in tr_rows.groupby("repetition_time_group_s", sort=True):
            tr_parts.append(
                f"approximately {float(repetition_time_s):.3f} s "
                f"({len(rows)} runs; {rows['subject'].nunique()} subjects)"
            )
        tr_description = "; ".join(tr_parts)
        subject_tr_ranges = tr_rows.groupby("subject")["repetition_time_s"].agg(
            lambda values: float(values.max() - values.min())
        )
        tr_consistency = (
            "Observed TR is constant within rounding precision across sessions within every subject."
            if subject_tr_ranges.le(0.001).all()
            else "WARNING: at least one subject has different observed TR values across sessions."
        )

    report = f"""FRAMEWISE DISPLACEMENT ANALYSIS
Generated: {datetime.now().astimezone().isoformat(timespec='seconds')}

RESULT
{interpretation}
Session 1 mean: {float(primary['session_1_mean_fd_mm']):.4f} mm
Session 2 mean: {float(primary['session_2_mean_fd_mm']):.4f} mm
Paired change (session 2 - session 1): {float(primary['mean_difference_mm']):+.4f} mm
95% CI: [{float(primary['difference_95ci_low_mm']):+.4f}, {float(primary['difference_95ci_high_mm']):+.4f}] mm
Paired t-test: t({int(primary['paired_t_df'])})={float(primary['paired_t_statistic']):.4f}, p={p_value:.6f}
Cohen dz: {float(primary['cohen_dz']):.4f}
{str(primary['sign_flip_method']).replace('_', ' ').capitalize()} sign-flip p: {float(primary['sign_flip_p_two_sided']):.6f}
Wilcoxon signed-rank p: {float(primary['wilcoxon_p_two_sided']):.6f}

METHOD
- Power FD = sum(abs(delta translation)) + {head_radius_mm:g} mm * sum(abs(delta rotation)).
- MCFLIRT columns 1-3 are treated as rotations in radians; columns 4-6 as translations in mm.
- FD is computed within runs. The first volume is undefined and excluded from summaries.
- For multi-echo runs, FD is computed per echo and then averaged frame by frame.
- Run mean FD values are equally averaged within subject/session over run IDs present in both sessions.
- The primary inferential unit is the subject; the primary test is a two-sided paired t-test.
- Sign-flip and Wilcoxon tests are reported as robustness checks.
- MNI spatial normalization/template size does not enter the FD calculation.
- Observed acquisition TR: {tr_description}. {tr_consistency}
  FD and threshold exceedance rates depend on sampling interval; the paired design compares
  sessions within subject. The single-echo-only sensitivity removes the multi-echo protocol.

DATA AUDIT
Input root: {data_root}
Subjects: {run_summary['subject'].nunique()}
Logical runs: {len(run_summary)}
Motion parameter files: {int(run_summary['n_echoes'].sum())}
Multi-echo subjects: {run_summary.loc[run_summary['n_echoes'].gt(1), 'subject'].nunique()}
Unmatched run handling: {missing_text}
Descriptive FD thresholds: {thresholds[0]:g} and {thresholds[1]:g} mm

SENSITIVITY ANALYSES
{chr(10).join(sensitivity_lines)}
"""
    output_path.write_text(report, encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    if args.head_radius_mm <= 0 or not np.isfinite(args.head_radius_mm):
        raise SystemExit("--head-radius-mm must be a positive finite value")
    thresholds = tuple(sorted(float(value) for value in args.fd_thresholds_mm))
    if any(value <= 0 or not np.isfinite(value) for value in thresholds):
        raise SystemExit("--fd-thresholds-mm values must be positive and finite")
    if args.dpi <= 0:
        raise SystemExit("--dpi must be positive")

    grouped_paths = discover_motion_files(args.data_root, args.task, args.subjects)
    runs, inventory = load_runs(grouped_paths, args.task, args.head_radius_mm)
    framewise = framewise_table(runs)
    run_summary = summarize_runs(runs, thresholds)
    subject_session_summary, inclusion_audit = summarize_subject_sessions(run_summary, thresholds)
    statistics_table = compare_sessions(subject_session_summary)
    primary = statistics_table.loc[statistics_table["is_primary"]].iloc[0]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    remove_stale_trace_outputs(args.out_dir / "subject_traces")
    inventory.to_csv(args.out_dir / "input_inventory.csv", index=False)
    framewise.to_csv(args.out_dir / "framewise_fd.csv", index=False, float_format="%.10g", na_rep="")
    run_summary.to_csv(args.out_dir / "run_fd_summary.csv", index=False, float_format="%.10g")
    subject_session_summary.to_csv(
        args.out_dir / "subject_session_fd_summary.csv", index=False, float_format="%.10g"
    )
    inclusion_audit.to_csv(args.out_dir / "inclusion_audit.csv", index=False)
    statistics_table.to_csv(
        args.out_dir / "session_comparison_statistics.csv", index=False, float_format="%.10g"
    )

    plot_session_comparison(
        subject_session_summary,
        primary,
        args.out_dir / "session_1_vs_2_mean_fd",
        args.dpi,
        args.head_radius_mm,
    )
    plot_paired_session_means_only(
        subject_session_summary,
        args.out_dir / "session_1_vs_2_mean_fd_paired_only",
        args.dpi,
    )
    plot_run_heatmap(run_summary, args.out_dir / "run_mean_fd_heatmap", args.dpi)
    if not args.no_subject_traces:
        plot_subject_traces(runs, args.out_dir / "subject_traces", thresholds, args.dpi)

    write_report(
        args.out_dir / "analysis_report.txt",
        args.data_root,
        run_summary,
        inclusion_audit,
        statistics_table,
        args.head_radius_mm,
        thresholds,
    )
    summary = {
        "generated": datetime.now().astimezone().isoformat(timespec="seconds"),
        "data_root": str(args.data_root),
        "output_directory": str(args.out_dir.resolve()),
        "task": args.task,
        "fd_definition": "Power FD",
        "head_radius_mm": args.head_radius_mm,
        "first_volume_fd": "undefined_and_excluded",
        "multi_echo_aggregation": "compute_each_echo_then_framewise_mean",
        "session_aggregation": "equal_weight_mean_of_matched_run_means",
        "primary_test": "two-sided paired t-test on subject mean FD",
        "n_subjects": int(run_summary["subject"].nunique()),
        "n_logical_runs": int(len(run_summary)),
        "n_parameter_files": int(run_summary["n_echoes"].sum()),
        "n_multi_echo_subjects": int(
            run_summary.loc[run_summary["n_echoes"].gt(1), "subject"].nunique()
        ),
        "fd_thresholds_mm": list(thresholds),
        "primary_result": {key: _json_value(value) for key, value in primary.items()},
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }
    (args.out_dir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    print(f"Processed {len(run_summary)} logical runs from {run_summary['subject'].nunique()} subjects.")
    print(f"Session 1 mean FD: {float(primary['session_1_mean_fd_mm']):.4f} mm")
    print(f"Session 2 mean FD: {float(primary['session_2_mean_fd_mm']):.4f} mm")
    print(
        f"Paired change: {float(primary['mean_difference_mm']):+.4f} mm; "
        f"t({int(primary['paired_t_df'])})={float(primary['paired_t_statistic']):.3f}, "
        f"p={float(primary['paired_t_p_two_sided']):.4f}"
    )
    print(f"Outputs saved to {args.out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
