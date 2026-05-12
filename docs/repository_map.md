# Repository Map

This map documents the current conservative layout after the first low-risk results migration. Source folders are intentionally kept in their original locations because several scripts still contain hard-coded paths from the internship environment.

## Current Layout

```text
.
|-- ShowerAnalyzer/                  ROOT/C++ shower feature extraction source
|-- PID/                             PID source scripts and remaining run folders
|   |-- BDT/                         LightGBM PID source scripts and logs
|   |-- MLP/                         MLP PID source scripts and logs
|   |-- GNN/                         GNN PID source scripts and plots not yet migrated
|   |-- CNN/                         exploratory CNN source
|   `-- RandomForest/                random-forest baseline source and remaining outputs
|-- Energy_reconstruction_ml/        ML energy reconstruction source and remaining run folders
|-- energy_reconstruction_Tminuit/   ROOT/TMinuit source macros and particle subfolders
|-- compare_parameters/              ROOT macros and earlier top-level comparison PDFs
|-- PID_RECONSTRUCTION/              PID-energy coupling source scripts and remaining run folders
|-- tools/                           utility scripts and shower visual examples
|-- results/                         selected low-risk result folders moved with git mv
|-- docs/                            documentation
|-- CITATION.cff
`-- LICENSE.md
```

## Results Migrated So Far

| Previous path | Current path | Notes |
|---|---|---|
| `compare_parameters/plots/` | `results/shower-variables/plots/` | Shower-variable comparison overlays and figures. |
| `PID/BDT/lgbm_viz/` | `results/pid/bdt/feature-interpretation/` | LightGBM feature-importance, SHAP-style, and tree-visualization figures. |
| `energy_reconstruction_Tminuit/plots/` | `results/energy-reconstruction/chi2-tminuit/summary-plots/` | Summary chi2/TMinuit linearity, deviation, and resolution figures. |
| `PID_RECONSTRUCTION/confusion_matrix_pid_LGBM.png` | `results/pid-energy-coupling/summary/confusion_matrix_pid_LGBM.png` | Summary PID confusion matrix used in coupling studies. |
| `PID_RECONSTRUCTION/confusion_matrix_pid_param_LGBM.png` | `results/pid-energy-coupling/summary/confusion_matrix_pid_param_LGBM.png` | Summary parameter-based PID confusion matrix used in coupling studies. |

No large run folders were moved in this conservative migration.

## Remaining Original Result Locations

Some result folders remain beside source scripts for now:

- `PID/BDT/results_with_time/`
- `PID/MLP/results/`
- `PID/MLP/results_with_time/`
- `PID/GNN/plots/`
- `PID/RandomForest/processed_data/`
- `Energy_reconstruction_ml/BDT/plots/`
- `Energy_reconstruction_ml/BDT/results_*_energy_reco/`
- `Energy_reconstruction_ml/MLP/results_*_energy_reco/`
- `energy_reconstruction_Tminuit/{kaon,pion-,proton}/plots/`
- `PID_RECONSTRUCTION/*/plots/`

These were left in place to avoid breaking scripts that read or write local paths.

## Planned Future Structure

After hard-coded paths are replaced by configuration files or command-line arguments, the repository can move toward:

```text
src/
|-- shower-features/
|-- pid/
|   |-- bdt/
|   |-- mlp/
|   |-- gnn/
|   |-- cnn/
|   `-- random-forest/
|-- energy-reconstruction/
|   |-- ml/
|   |   |-- bdt/
|   |   `-- mlp/
|   `-- chi2-tminuit/
|-- pid-energy-coupling/
`-- tools/

results/
|-- shower-variables/
|-- pid/
|-- energy-reconstruction/
|-- pid-energy-coupling/
`-- archive/
```

The future migration should be done in small commits with `git mv`, preserving all scientific outputs and updating documentation links after each step.
