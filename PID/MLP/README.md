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

Several paths are local to the original internship environment. Treat these scripts as research workflows that need path configuration before reuse.
