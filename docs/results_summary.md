# Results Summary

This repository contains representative figures and performance logs from the
internship analysis. The documentation avoids claiming publication-level
performance because the full dataset provenance and environment snapshot are not
included.

> **The figures and logs described here were produced before the review recorded
> in [corrections.md](corrections.md), and several of the corrections change a
> quoted quantity.** In particular: the resolution curves understate sigma by
> about 21 % (a 90 %-truncated RMS with no truncation correction); the quantity
> labelled `sigma_over_E` mixes bias and resolution; the chi2 coefficients were
> fitted and evaluated on the same events; the per-particle MLP energy results
> come from three identical scripts that all read the pion file and could not
> run; and the coupling figures labelled "no PID" were produced by routing on
> the true particle, so they show perfect PID. None of these figures can be
> attributed to a configuration, because the feature list and the smearing width
> were not logged.

## PID Outputs

Useful locations:

- `results/pid/bdt/feature-interpretation/`
- `analysis/3_pid/bdt/results_with_time/`
- `analysis/3_pid/mlp/results/`
- `analysis/3_pid/mlp/results_with_time/`
- `analysis/3_pid/gnn/plots/`
- `results/pid-energy-coupling/summary/confusion_matrix_pid_LGBM.png`
- `results/pid-energy-coupling/summary/confusion_matrix_pid_param_LGBM.png`

These outputs show confusion matrices, training curves, feature-importance diagnostics, and model-comparison material.

## Energy Reconstruction Outputs

Useful locations:

- `analysis/4_energy/ml_bdt/plots/`
- `analysis/4_energy/ml_bdt/results_all_energy_reco/plots/`
- `analysis/4_energy/ml_mlp/results_*_energy_reco/plots/`
- `results/energy-reconstruction/chi2-tminuit/summary-plots/`

Typical figures include:

- reconstructed energy vs true beam energy;
- linearity profiles;
- relative deviation or bias curves;
- relative resolution curves;
- training curves for ML regressors.

## Shower-Variable Comparisons

Useful locations:

- `analysis/2_variable_study/`
- `results/shower-variables/plots/`

The comparison figures summarize differences in selected shower variables such as density, radius, threshold content, and longitudinal quantities.

## PID-Energy Coupling Outputs

Useful locations:

- `analysis/5_pid_energy/pi-_proton/plots/`
- `analysis/5_pid_energy/kaon_pi-_proton/plots/`

These studies compare energy reconstruction scenarios with and without PID-informed choices. They should be read as exploratory coupling studies rather than final detector-performance results.

## Interpretation Guidance

When reviewing the results, focus on:

- whether the chosen variables are physically plausible;
- whether model comparisons are organized and interpretable;
- whether PID and energy reconstruction are treated as connected tasks;
- whether limitations and missing reproducibility pieces are clear.

The stored figures are useful for portfolio review and technical discussion, but
a formal performance statement would require a locked dataset, fixed splits,
exact software versions, and rerun logs. The machinery for all four now exists -
`tools/build_holdout.py` for the sample, `sdhcal.splits` for the splits,
`sdhcal.runlog` for the versions and the logs - but the studies have not been
re-run. [corrections.md](corrections.md) lists in section 5 what that requires.
