# AGENTS.md

## Scope
These instructions apply to this repository root and all subdirectories, unless a deeper `AGENTS.md` overrides them.

## Project Objective
The main objective of this project is to identify a brain network that encodes motor vigour while showing low variability across trials.

## Dataset
fMRI was recorded while subjects complete a behaviour task, pressure a squeeze bulb as fast and as much as possible. 

## Main Repository Structure
- `empca/`: Expectation-Maximization PCA (EM-PCA) implementation.
- `GLMsingle/`: GLMsingle algorithm for extracting trial-wise, voxel-wise beta values from fMRI data, with a unique HRF per voxel.
- `single_analysis/`: Single-subject analysis pipeline.
- `group_analysis/`: Group-level analysis pipeline built on concatenated subject data.

## Group Analysis Pipeline (`group_analysis/`)
- `data_concat.py`: Concatenate all subjects' data across trials.
- `main_glm_mni.py`: Run GLMsingle on group data in MNI space.
- `Beta_preprocessing_mni.py`: Preprocess beta values extracted from GLMsingle.
- `obj_param.py`: Core objective-function workflow used to identify the target network.
- `plotting.py`: Plot and visualize analysis results.

## Dataset Location
- Primary datasets are located at `/Data/zahra/`.
- Treat `/Data/zahra/` as input data and avoid destructive edits there.

## Working Conventions
- Keep edits minimal and focused on the requested task.
- Keep code changes as minimal as possible and consistent with the existing code structure.
- Reuse existing functions when possible instead of adding new ones.
- Do not refactor unrelated code paths unless explicitly requested.
- Preserve algorithmic intent in `empca/` and `GLMsingle/` unless the task specifically asks for method changes.
- use .venv for running and validate a code. 

## Validation Expectations
- For code changes, run the relevant analysis script(s) for the touched stage.
- If preprocessing/objective logic changes, regenerate and inspect relevant plots from `group_analysis/plotting.py`.
- Document significant analysis or behavior changes in `ANALYSIS_LOG.md` when present.

## Results:
saved at results folder and group_analysis_result.pdf single_subject_result.pdf
