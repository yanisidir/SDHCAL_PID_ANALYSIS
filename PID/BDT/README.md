# BDT PID

This folder contains LightGBM-based particle-identification studies.

## Main Scripts

- `LGBM_classifier_PID.py`: multiclass LightGBM PID pipeline for pi-, K0, and proton.
- `pion_proton_classifier.py`: pairwise pion/proton classifier study.
- `identify_hadron.py`: helper workflow for applying a trained classifier.
- `feature_importance_with_permutation.py`: permutation-importance study.
- `analyse_corr_permutation.py`: correlation and importance diagnostics.
- `plot_trees.py`: LightGBM tree visualization helper.

## Typical Features

The BDT studies use engineered shower observables such as threshold content, shower start, density, radius, barycentre variables, clustering summaries, timing summaries, and longitudinal-profile quantities.

## Outputs

- `lgbm_viz/`: feature-importance and tree-visualization images.
- `results_with_time/`: generated models and plots when scripts are run.
- `run_parameters.csv`, `run_comments.csv`, `hadron_performances.csv`: lightweight run logs currently present in the folder.

The scripts should not be rerun without updating input paths and confirming the intended dataset split.
