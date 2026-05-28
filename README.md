# Master Thesis Final Reproducibility Package

This folder contains the analysis-ready inputs, figure scripts, and generated
outputs for the thesis figures in `figures/` and `Claude_results/`.

## Structure

- `data/`: packaged analysis inputs, including the current optimization weight
  map `voxel_weights_task1_bold0.6_beta0.6_smooth1.25_gamma1.5.nii.gz`.
- `figures/`: generated main figure outputs and machine-readable summaries.
- `Claude_results/`: follow-up connectivity analyses that read outputs from
  `figures/med_effects/`.
- `GLMsingle/` and `empca/`: local method dependencies retained for pipeline
  compatibility.

## Regenerate Main Outputs

Use the local virtual environment from this folder:

```bash
.venv/bin/python threshold_robustness_voxel_network.py
.venv/bin/python plot_weight_hist.py
.venv/bin/python plot_html_multiplane_contours.py
.venv/bin/python compare_glm_region_highlights.py
.venv/bin/python med_effects.py
.venv/bin/python behaviour_topology_effects.py
bash Claude_results/run_all.sh
```

`med_effects.py` requires the mounted beta-preprocessed dataset at
`/mnt/TeamShare/Data_Masterfile/H20-00572_All-Dressed/Zahra-Thesis-Data/fmri_opt_group/results_beta_preprocessed`.
`behaviour_topology_effects.py` requires the mounted behaviour directory at
`/mnt/TeamShare/Data_Masterfile/H20-00572_All-Dressed/Zahra-Thesis-Data/fmri_opt_group/behaviour`.

## Current Weight-Map Outputs

The current figure code uses:

- `data/voxel_weights_task1_bold0.6_beta0.6_smooth1.25_gamma1.5.nii.gz`
- `data/voxel_weights_task1_bold0.6_beta0.6_smooth1.25_gamma1.5_bold_thr90.html`

and writes matching output stems under `figures/`.
