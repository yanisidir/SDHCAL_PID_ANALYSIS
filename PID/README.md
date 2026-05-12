# PID Studies

This directory contains particle-identification studies for SDHCAL hadronic showers. The main target classes used across the repository are pi-, K0, and proton.

## Contents

- `BDT/`: LightGBM classifiers, feature-importance studies, and visual diagnostics.
- `MLP/`: scikit-learn MLP classifiers using engineered shower variables.
- `GNN/`: exploratory PyTorch Geometric graph-based PID studies.
- `RandomForest/`: additional tree-based baseline studies.
- `CNN/`: exploratory convolutional approach.

## Inputs

Most scripts expect ROOT parameter trees or generated CSV files from the shower-parameter extraction pipeline. Several scripts contain absolute paths from the internship environment and should be adapted before rerunning.

## Outputs

Typical outputs include confusion matrices, training curves, model files, performance CSVs, and run-parameter logs. Large generated artifacts are ignored for future commits, while lightweight figures may be kept for review.
