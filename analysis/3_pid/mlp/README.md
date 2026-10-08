# MLP PID

This folder contains scikit-learn MLP classifiers for particle identification from engineered shower variables.

## Main Scripts

- `hadron_classifier_MLP.py`: three-class MLP PID workflow.
- `2_hadron_classifier_MLP.py`: pairwise PID workflow.
- `test_mlp_classifier.py`: test or inference-oriented helper.
- `feature_importance_with_permutation.py`: feature-importance diagnostic.

## Method

The MLP workflows typically read ROOT-derived shower variables, clean invalid values, split train and test samples, standardize features, optionally rebalance classes, train an `MLPClassifier`, and save confusion-matrix or performance outputs.

## Notes

These are research workflows from the internship, kept for comparison with the
maintained LightGBM classifier in [`../bdt/`](../bdt/). Their inputs now resolve
through `$SDHCAL_DATA_DIR`, so they no longer need path editing, but they have
not been brought up to the conventions of the maintained pipeline: no multi-seed
uncertainties, no per-run JSON record, and the bias/resolution estimators of
`sdhcal.metrics` are not used.

`hadron_classifier_MLP.py` read three per-particle directories that each held a
file of the same name; it now names the three datasets explicitly, because those
directories all map to one place today.
