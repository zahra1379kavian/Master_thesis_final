# Master Thesis Final Reproducibility Package

This folder contains the code and analysis-ready inputs needed to regenerate the final thesis figures listed in the project cleanup request.

## Structure

- `src/final_figures/`: Clean figure-generation code for the final results.
- `src/pipeline/`: Main pipeline scripts transferred from the original project.
- `GLMsingle/` and `empca/`: Local code dependencies required by the transferred pipeline scripts.
- `data/`: Minimal analysis-ready inputs used by the final figure scripts.
- `results/`: Regenerated final figures.

## Regenerate Final Figures

From this folder:

```bash
/home/zkavian/Thesis_code_Glm_Opt/.venv/bin/python src/final_figures/make_final_results.py
```

The command regenerates all final PDF outputs under `results/`.

## Final Results

- `results/ablation/w_balanced_0p309_0p314_0p377/obj_evaluation_metric_comparison(main).pdf`
- `results/ablation/map1_multiplane_contour_all_regions(main).pdf`
- `results/ablation/two_networks_overlay(main).pdf`
- `results/behave_vs_bold/projection_behavior_subject_panel(main).pdf`
- `results/connectivity/roi_edge_network/mutual_information_ksg/cross_subject_only_laplacian_spectral_distance_signed_distribution.pdf`
- `results/connectivity/roi_edge_network/mutual_information_ksg/top_edges_subject_delta_heatmap.pdf`
- `results/connectivity/GVS_effects/gvs_similarity_hemi/roi_condition_reference_deltas/plots/without_unassigned/off_on_condition_minus_sham_subject_session_roi_burden_heatmaps.pdf`
- `results/prove_hypothesis/trial_variability_hypothesis_norm_diff_cd(main).pdf`
- `results/prove_hypothesis/state_selection_stability_followup/norm_diff_mean_subject_state_lines(main).pdf`

## Input Availability

No inputs are missing for regenerating the final figures from `src/final_figures/make_final_results.py`.

The transferred main pipeline scripts still depend on the original raw/preprocessed dataset tree. In this environment, `/Data/zahra` is not mounted, so these scripts cannot be fully rerun from raw data here:

- `src/pipeline/main_glm_mni.py`: needs `/Data/zahra/bold_data`, `/Data/zahra/masks`, `/Data/zahra/go_times`, and GLMsingle output locations.
- `src/pipeline/Beta_preprocessing_mni.py`: needs `/Data/zahra/results_glm` and MNI anatomy masks.
- `src/pipeline/data_concat.py`: needs `/Data/zahra/results_beta_preprocessed` and `/Data/zahra/results_glm`.
- `src/pipeline/obj_param.py`: needs `/Data/zahra/results_beta_preprocessed/group_concat`, `/Data/zahra/behaviour`, and anatomy masks.

The packaged `data/` directory is intentionally minimal. It contains the analysis-ready arrays, NIfTI maps, and tables required to reproduce the final figures without copying the full raw fMRI dataset.
