# Repository Map

The layout follows the analysis chain: one numbered directory per stage under
`analysis/`, the shared Python library under `src/sdhcal/`, the shared C++ under
`cpp/`. Data locations come from `SDHCAL_DATA_DIR`, `SDHCAL_MODEL_DIR` and
`SDHCAL_OUT_DIR`, resolved by `src/sdhcal/paths.py` on the Python side and
`cpp/include/SdhcalPaths.h` on the C++ side.

See [`../analysis/README.md`](../analysis/README.md) for the chain diagram and
the conventions every stage shares.

## Current Layout

```text
.
|-- src/sdhcal/                      shared Python library, installed with `pip install -e .`
|                                      paths  rootio  smearing  splits  metrics  target  runlog  calib
|-- cpp/
|   |-- include/                     ShowerAnalyzer.h, SdhcalPaths.h
|   `-- src/                         ShowerAnalyzer.cpp
|
|-- analysis/                        the chain, one directory per stage
|   |-- 1_shower_features/           computeParams, computeParams_parallel, run_parallel.sh
|   |-- 2_variable_study/            compare*.C and their comparison PDFs
|   |-- 3_pid/
|   |   |-- bdt/                     LightGBM classifier (maintained), interpretation scripts, logs
|   |   |-- mlp/                     scikit-learn MLP classifiers and logs
|   |   |-- randomforest/            tree-based baseline
|   |   |-- gnn/                     exploratory PyTorch Geometric studies (six variants)
|   |   `-- cnn/                     exploratory convolutional study
|   |-- 4_energy/
|   |   |-- ml_bdt/                  LightGBM regressors (maintained), plots, logs
|   |   |-- ml_mlp/                  MLP regressors, plots, logs
|   |   `-- chi2_tminuit/            weighted least-squares calibration, per-particle subfolders
|   `-- 5_pid_energy/                pid_energy_coupling.py (--pid none|true|bdt|mlp), figures.py
|
|-- calibration/                     chi2 coefficients, read by both the C++ and the Python side
|-- tools/                           held-out sample builder, calibration merge, run-log repair
|-- tests/                           test suite for the shared library
|-- results/                         curated figures referenced by the report
|-- report/                          LaTeX report and compiled PDF, self-contained
|-- docs/                            documentation
|-- Makefile  pyproject.toml  requirements.txt
|-- CITATION.cff
`-- LICENSE.md
```

## Consolidated Scripts

Several families of near-identical scripts were merged; the originals are in the
git history. See [corrections.md](corrections.md).

| Was | Is now |
|---|---|
| `PID_RECONSTRUCTION/{pi-_proton,kaon_pi-_proton}/{BDT,MLP,noPID}_{BDT,MLP,chi2}.py`, `hadron_pid_reco_linear.py` (13 files) | `analysis/5_pid_energy/pid_energy_coupling.py --pid ... --reco ...` |
| `Energy_reconstruction_ml/BDT/{pion,kaon,proton}_energy_reco_lgbm.py`, `pion_proton_Ereco_lgbm.py` | `analysis/4_energy/ml_bdt/hadron_energy_reco_lgbm.py --particles pi\|kaon\|proton\|mixed\|mix-pi-proton` |
| `Energy_reconstruction_ml/MLP/MLP_Energy_reconstruction_{pion,kaon,proton,all}.py` (identical and non-functional) | `analysis/4_energy/ml_mlp/mlp_energy_reco.py --particles ...` |
| `energy_reconstruction_Tminuit/{kaon,pion-,proton}/EnergyReco.C` | `analysis/4_energy/chi2_tminuit/EnergyReco.C("<particle>")` |

## Directory Renames

Every move was a `git mv`, so file history is preserved.

| Before | After |
|---|---|
| `sdhcal/` | `src/sdhcal/` |
| `include/SdhcalPaths.h` | `cpp/include/SdhcalPaths.h` |
| `ShowerAnalyzer/ShowerAnalyzer.{h,cpp}` | `cpp/{include,src}/` |
| `ShowerAnalyzer/` (drivers) | `analysis/1_shower_features/` |
| `compare_parameters/` | `analysis/2_variable_study/` |
| `PID/{BDT,MLP,GNN,CNN,RandomForest}/` | `analysis/3_pid/{bdt,mlp,gnn,cnn,randomforest}/` |
| `Energy_reconstruction_ml/{BDT,MLP}/` | `analysis/4_energy/{ml_bdt,ml_mlp}/` |
| `energy_reconstruction_Tminuit/` | `analysis/4_energy/chi2_tminuit/` |
| `PID_RECONSTRUCTION/` | `analysis/5_pid_energy/` |

## Where Outputs Go

Run outputs are written under `$SDHCAL_OUT_DIR` (default `./output`, not
versioned), one directory per study, each holding `plots/`, `tables/`,
`arrays/`, `models/`, the parameter and performance CSVs, and
`runs/<name>_<id>.json` with the full configuration of that run.

Figures worth keeping are copied by hand into `results/` and committed.

## Result Folders Still Beside Their Scripts

These predate the output convention and hold the figures produced during the
internship. They are kept as a record; new runs no longer write here.

- `analysis/3_pid/bdt/results_with_time/`
- `analysis/3_pid/mlp/results/`, `analysis/3_pid/mlp/results_with_time/`
- `analysis/3_pid/gnn/plots/`
- `analysis/3_pid/randomforest/processed_data/`
- `analysis/4_energy/ml_bdt/plots/`, `analysis/4_energy/ml_bdt/results_*_energy_reco/`
- `analysis/4_energy/ml_mlp/results_*_energy_reco/`
- `analysis/4_energy/chi2_tminuit/{kaon,pion-,proton}/plots/`
- `analysis/5_pid_energy/{pi-_proton,kaon_pi-_proton}/plots/`

## Earlier Migration of Curated Results

An earlier reorganisation moved selected figures into `results/`. Recorded here
for traceability; the left column holds the paths as they were then.

| Previous path | Current path |
|---|---|
| `compare_parameters/plots/` | `results/shower-variables/plots/` |
| `PID/BDT/lgbm_viz/` | `results/pid/bdt/feature-interpretation/` |
| `energy_reconstruction_Tminuit/plots/` | `results/energy-reconstruction/chi2-tminuit/summary-plots/` |
| `PID_RECONSTRUCTION/confusion_matrix_pid_LGBM.png` | `results/pid-energy-coupling/summary/confusion_matrix_pid_LGBM.png` |
| `PID_RECONSTRUCTION/confusion_matrix_pid_param_LGBM.png` | `results/pid-energy-coupling/summary/confusion_matrix_pid_param_LGBM.png` |
