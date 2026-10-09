# BDT PID

This folder contains LightGBM-based particle-identification studies.

## Main Scripts

- `LGBM_classifier_PID.py`: **the maintained classifier.** Multiclass LightGBM
  for pi-, p and K0_L, with SMOTE on the training split only, early stopping on
  validation, and `--seeds n` for multi-seed uncertainties. Inputs come from
  `$SDHCAL_DATA_DIR`, outputs from `$SDHCAL_OUT_DIR/pid_lgbm/`.
- `pion_proton_classifier.py`: pairwise pion/proton classifier study.
- `identify_hadron.py`: helper workflow for applying a trained classifier.
- `feature_importance_with_permutation.py`: permutation-importance study.
- `analyse_corr_permutation.py`: correlation and importance diagnostics.
- `plot_trees.py`: LightGBM tree visualization helper.

## Typical Features

The BDT studies use engineered shower observables such as threshold content, shower start, density, radius, barycentre variables, clustering summaries, timing summaries, and longitudinal-profile quantities.

## Outputs

- `../../../results/pid/bdt/feature-interpretation/`: feature-importance and tree-visualization images moved out of the source folder.
- `$SDHCAL_OUT_DIR/pid_lgbm/`: where new runs write - plots, tables, arrays,
  models, and `runs/pid_<id>.json` holding the full configuration of the run.
- `results_with_time/`, `run_parameters.csv`, `run_comments.csv`,
  `hadron_performances.csv`: outputs of the internship runs, kept as a record.
  New runs no longer write here.

The paths no longer need editing: inputs are resolved from `$SDHCAL_DATA_DIR`
and a missing file raises an error naming the variable to set. Two caveats
remain about the older logs in this folder: their columns were shifted by a
schema change, since repaired, and they do not record the feature list or the
smearing width - so the figures they accompany cannot be attributed to a
configuration. See [`../../../docs/corrections.md`](../../../docs/corrections.md).
