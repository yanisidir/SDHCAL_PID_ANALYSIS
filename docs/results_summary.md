# Results Summary

The final M2 report in `report/` is the reference for the numerical claims kept
from the internship. Training, validation and final evaluation samples were
distinct for the DRD6 results.

## Results Retained in the Final Report

### Particle identification

The multiclass LightGBM classifier reaches a balanced accuracy of **67.8 %**.
The class recalls are:

| True species | Recall |
|---|---:|
| pi- | 48.8 % |
| proton | 68.6 % |
| K0L | 86.0 % |

The dominant limitation is the pion/proton overlap: 47.9 % of pions are
classified as protons. The binary pion/proton classifier used in the coupled
energy study reaches an accuracy close to 61 %, with recalls of 49.1 % for
pions and 73.0 % for protons.

The GNN work is retained as an exploratory proof of feasibility. No quantitative
GNN performance is used as a headline result.

### PID-energy coupling

The following biases were recalculated from the event-level prediction tables.
Values are percentages, rounded to 0.1 percentage point.

| Scenario | pi- 20 GeV | pi- 80 GeV | K0L 20 GeV | K0L 80 GeV | p 20 GeV | p 80 GeV |
|---|---:|---:|---:|---:|---:|---:|
| pion parametric calibration applied to all | -0.7 | -4.9 | -6.4 | -5.6 | -6.8 | -4.8 |
| BDT PID + parametric reconstruction | +2.8 | -4.7 | -2.9 | -5.2 | -1.7 | -4.3 |
| BDT PID + BDT energy reconstruction | +7.1 | +0.9 | +6.3 | 0.0 | +3.0 | 0.0 |

The main conclusion is therefore about the energy scale: species-dependent
routing can reduce some calibration biases, and the BDT PID + BDT energy chain
keeps the bias below 1 % in magnitude for all three species at 80 GeV.

The historical width curves were computed from a truncated core and are used
only for qualitative comparisons. The final report does not claim a numerical
gain in absolute energy resolution.

## Representative Outputs

Useful PID locations:

- `results/pid/bdt/feature-interpretation/`
- `analysis/archive/results/3_pid/bdt/results_with_time/`
- `analysis/archive/results/3_pid/mlp/results/`
- `analysis/archive/results/3_pid/gnn/plots/`
- `results/pid-energy-coupling/summary/`

Useful energy-reconstruction locations:

- `analysis/4_energy/ml_bdt/plots/`
- `analysis/archive/results/4_energy/ml_bdt/results_all_energy_reco/plots/`
- `results/energy-reconstruction/chi2-tminuit/summary-plots/`
- `analysis/archive/results/kaon_pi-_proton/csv/`

Useful shower-variable locations:

- `analysis/2_variable_study/`
- `results/shower-variables/plots/`

## Interpretation of Historical Files

Some stored outputs predate the current methodology. In particular:

- files named `noPID_*` route events using the true particle identity and thus
  represent perfect PID, not a particle-blind baseline;
- the historical quantity named `sigma_over_E` mixes bias and spread;
- historical width curves use an uncorrected truncated core;
- the old per-particle MLP energy outputs must not be used;
- old timing-ablation and GNN numbers are not retained as final results.

These files remain useful for documenting the analysis workflow. New numerical
claims should follow the definitions and validation procedures implemented in
`src/sdhcal/` and summarized in the final report.
