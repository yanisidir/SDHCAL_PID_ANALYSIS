# Results Summary

This repository contains representative figures and performance logs from the internship analysis. The documentation intentionally avoids claiming publication-level performance because the full dataset provenance and environment snapshot are not included.

## PID Outputs

Useful locations:

- `results/pid/bdt/feature-interpretation/`
- `PID/BDT/results_with_time/`
- `PID/MLP/results/`
- `PID/MLP/results_with_time/`
- `PID/GNN/plots/`
- `results/pid-energy-coupling/summary/confusion_matrix_pid_LGBM.png`
- `results/pid-energy-coupling/summary/confusion_matrix_pid_param_LGBM.png`

These outputs show confusion matrices, training curves, feature-importance diagnostics, and model-comparison material.

## Energy Reconstruction Outputs

Useful locations:

- `Energy_reconstruction_ml/BDT/plots/`
- `Energy_reconstruction_ml/BDT/results_all_energy_reco/plots/`
- `Energy_reconstruction_ml/MLP/results_*_energy_reco/plots/`
- `results/energy-reconstruction/chi2-tminuit/summary-plots/`

Typical figures include:

- reconstructed energy vs true beam energy;
- linearity profiles;
- relative deviation or bias curves;
- relative resolution curves;
- training curves for ML regressors.

## Shower-Variable Comparisons

Useful locations:

- `compare_parameters/`
- `results/shower-variables/plots/`

The comparison figures summarize differences in selected shower variables such as density, radius, threshold content, and longitudinal quantities.

## PID-Energy Coupling Outputs

Useful locations:

- `PID_RECONSTRUCTION/pi-_proton/plots/`
- `PID_RECONSTRUCTION/kaon_pi-_proton/plots/`

These studies compare energy reconstruction scenarios with and without PID-informed choices. They should be read as exploratory coupling studies rather than final detector-performance results.

## Interpretation Guidance

When reviewing the results, focus on:

- whether the chosen variables are physically plausible;
- whether model comparisons are organized and interpretable;
- whether PID and energy reconstruction are treated as connected tasks;
- whether limitations and missing reproducibility pieces are clear.

The stored figures are useful for portfolio review and technical discussion, but a formal performance statement would require a locked dataset, fixed splits, exact software versions, and rerun logs.
