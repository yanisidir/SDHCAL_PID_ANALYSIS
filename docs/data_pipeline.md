# Data Pipeline

This repository does not include the large input datasets. The expected workflow starts from simulated SDHCAL events and ends with ROOT parameter trees, CSV summaries, trained models, and plots.

## Pipeline Stages

```text
Simulation
  -> digitization
  -> LCIO to ROOT conversion
  -> shower parameter extraction
  -> PID studies
  -> energy reconstruction studies
  -> PID-energy coupling comparisons
```

## Expected Data Products

The scripts assume several types of data products:

- `.slcio` files from detector simulation and digitization stages.
- ROOT files containing per-hit information, usually read from a `tree`.
- ROOT files containing derived shower parameters, usually read from `paramsTree`.
- Generated CSV logs for run parameters, comments, and performance summaries.
- Generated figures in `plots/` or `results_*` directories.
- Trained model artifacts such as `.joblib`, `.pt`, or `.pth` files.

Large binary and generated files are intentionally ignored by `.gitignore` for future commits.

## Shower Parameters

The central derived-data product is a ROOT tree of shower variables. Typical variables include:

- threshold occupancy: `Thr1`, `Thr2`, `Thr3`, `N1`, `N2`, `N3`, `ratioThr23`;
- longitudinal development: `Begin`, `Zbary`, `Zrms`, `Nmax`, `z0_fit`, `Xmax`, `lambda`;
- transverse morphology: `Radius`, `Density`, `lambda1`, `lambda2`, `eccentricity3D`;
- clustering and topology: `NClusters`, `AvgClustSize`, `MaxClustSize`, `nTrackSegments`;
- timing summaries where available: `tMin`, `tMax`, `tMean`, `tSpread`.

These variables are extracted mainly through `ShowerAnalyzer/`.

## Reproduction Notes

To reproduce the full analysis in principle, a user would need:

- an SDHCAL-compatible simulation and digitization environment;
- ROOT and C++ build support;
- LCIO/Marlin conversion tools if starting from `.slcio`;
- Python packages for ML workflows;
- local replacements for the absolute input paths currently present in scripts;
- enough storage for ROOT files and generated model artifacts.

No commands in this documentation update recompiled code, reran training, or regenerated scientific results.

## Path Hygiene for Future Work

For a more reusable release, the next step would be to replace hard-coded paths with one of:

- a `config.yaml` file;
- command-line arguments;
- environment variables such as `SDHCAL_DATA_DIR`;
- a small loader module shared by PID and energy scripts.
