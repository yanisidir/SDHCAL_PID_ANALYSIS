# ML Energy Reconstruction

This directory contains machine-learning studies for reconstructing hadron energy from SDHCAL shower variables.

## Contents

- `BDT/`: LightGBM regression workflows and comparison plots.
- `MLP/`: MLP-based regression workflows for pi-, K0, proton, and combined samples.

## Inputs

The scripts generally expect ROOT parameter trees containing `primaryEnergy` and derived shower variables from `ShowerAnalyzer/`. Some scripts use hard-coded paths from the original analysis environment.

## Outputs

Typical outputs include:

- reconstructed-vs-true energy plots;
- linearity and relative-deviation plots;
- resolution curves;
- training curves;
- model and scaler artifacts;
- run-parameter and performance CSV files.

The stored figures are useful for review, but a formal comparison should be rerun with fixed dataset provenance and software versions.
