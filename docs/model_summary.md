# Model Summary

This document summarizes the modeling approaches present in the repository. It is descriptive rather than a benchmark claim.

## Shared Inputs

Most PID and ML energy-reconstruction scripts use engineered shower variables extracted from ROOT trees. Common inputs include threshold counts, shower start, density, radius, barycentre variables, clustering summaries, timing summaries, and fitted longitudinal-profile quantities.

Some GNN scripts use hit-level or flattened event representations and build graph-like data structures for PyTorch Geometric.

## PID Models

### BDT / LightGBM

Location: `PID/BDT/`

The BDT studies use LightGBM classifiers for hadron identification. Scripts include multiclass PID for pi-, K0, and proton, as well as pairwise studies such as pion-proton separation. Supporting scripts compute feature importances, permutation importances, correlations, and tree visualizations.

Typical outputs:

- confusion matrices;
- training curves;
- model artifacts;
- feature-importance plots;
- run parameter and performance CSV files.

### MLP

Location: `PID/MLP/`

The MLP studies use scikit-learn neural-network classifiers on engineered shower features. The scripts include data cleaning, train/test splitting, standardization, optional balancing, and confusion-matrix outputs.

### GNN

Location: `PID/GNN/`

The GNN studies are exploratory PyTorch Geometric workflows using graph neural-network layers such as GraphConv, TAGConv, and GATConv in different scripts. These studies investigate whether hit-level or graph representations can complement hand-engineered shower variables.

## Energy Reconstruction Models

### ML Regression

Location: `Energy_reconstruction_ml/`

The ML energy reconstruction studies use LightGBM and MLP regressors. The scripts evaluate reconstructed energy against true beam energy and produce linearity, relative-deviation, resolution, and prediction-vs-truth plots.

### chi2/TMinuit

Location: `energy_reconstruction_Tminuit/`

The TMinuit approach reconstructs energy using semi-digital hit counts. A typical parameterization uses N1, N2, and N3 with coefficients depending on total hit count, then minimizes a chi2-like objective against the true beam energy.

This approach is useful as a physics-motivated comparison to ML regression because it keeps the semi-digital calorimeter response explicit.

## PID-Energy Coupling

Location: `PID_RECONSTRUCTION/`

These scripts compare energy reconstruction with and without PID information. The purpose is to study whether selecting particle-specific reconstruction paths or applying PID-informed corrections changes linearity, bias, or resolution.

## Validation Caveats

- Stored figures and CSV files reflect the specific samples and configurations used at the time.
- The repository does not yet provide a single consolidated benchmark table.
- Some scripts are exploratory and should be treated as analysis notebooks in script form.
- Performance claims should be made only after rerunning with documented datasets, splits, and software versions.
