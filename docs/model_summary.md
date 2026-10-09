# Model Summary

This document summarizes the modeling approaches present in the repository. It is descriptive rather than a benchmark claim.

## Shared Inputs

Most PID and ML energy-reconstruction scripts use engineered shower variables extracted from ROOT trees. Common inputs include threshold counts, shower start, density, radius, barycentre variables, clustering summaries, timing summaries, and fitted longitudinal-profile quantities.

Some GNN scripts use hit-level or flattened event representations and build graph-like data structures for PyTorch Geometric.

## PID Models

### BDT / LightGBM

Location: `analysis/3_pid/bdt/`

The BDT studies use LightGBM classifiers for hadron identification. Scripts include multiclass PID for pi-, K0, and proton, as well as pairwise studies such as pion-proton separation. Supporting scripts compute feature importances, permutation importances, correlations, and tree visualizations.

Typical outputs:

- confusion matrices;
- training curves;
- model artifacts;
- feature-importance plots;
- run parameter and performance CSV files.

### MLP

Location: `analysis/3_pid/mlp/`

The MLP studies use scikit-learn neural-network classifiers on engineered shower features. The scripts include data cleaning, train/test splitting, standardization, optional balancing, and confusion-matrix outputs.

### GNN

Location: `analysis/3_pid/gnn/`

The GNN studies are exploratory PyTorch Geometric workflows using graph neural-network layers such as GraphConv, TAGConv, and GATConv in different scripts. These studies investigate whether hit-level or graph representations can complement hand-engineered shower variables.

## Energy Reconstruction Models

### ML Regression

Location: `analysis/4_energy/`

The ML energy reconstruction studies use LightGBM and MLP regressors, one script
each, selecting the sample with `--particles pi|kaon|proton|mixed|all`. Both
train on `log E` and correct the back-transform with Duan's smearing factor,
fitted on the validation set: exponentiating a log-space fit otherwise yields a
geometric mean, a few per cent below the arithmetic mean.

Bias and resolution are reported separately, and the resolution is corrected for
the truncation used to reject tails:

| Quantity | Definition |
|---|---|
| bias | mean of `(E_reco - E_true)/E_true`, i.e. the energy-scale error |
| resolution | standard deviation of the same residual around its own mean, estimated on the distribution core and divided by the truncation factor (0.7893 for the central 90 % of a Gaussian) |
| `rms_rel_about_zero` | `sqrt(mean(r^2))` - the historical `sigma_over_E`, equal to `sqrt(bias^2 + resolution^2)`, kept only so older log entries stay comparable |

The estimators live in `src/sdhcal/metrics.py` and are covered by the test suite.

### chi2/TMinuit

Location: `analysis/4_energy/chi2_tminuit/`

The TMinuit approach reconstructs energy using semi-digital hit counts. A typical parameterization uses N1, N2, and N3 with coefficients depending on total hit count, then minimizes a chi2-like objective against the true beam energy.

This approach is useful as a physics-motivated comparison to ML regression because it keeps the semi-digital calorimeter response explicit.

## PID-Energy Coupling

Location: `analysis/5_pid_energy/`

`analysis/5_pid_energy/pid_energy_coupling.py` compares three cases, which belong
in the same table:

| `--pid` | Routing | Reading |
|---|---|---|
| `none` | one inclusive regressor trained on the mixture | the PID-agnostic baseline |
| `true` | the true PDG code | the best any PID could achieve |
| `bdt`, `mlp` | the classifier's decision | what a real detector would get |

The earlier `noPID_*.py` scripts routed by the true PDG, so they measured
perfect PID, not the absence of PID; the baseline was never run. See
[corrections.md](corrections.md), section 2.3.

## Validation Caveats

- Stored figures and CSV files reflect the specific samples and configurations used at the time.
- The repository does not yet provide a single consolidated benchmark table.
- Some scripts are exploratory and should be treated as analysis notebooks in script form.
- Performance claims should be made only after rerunning with documented datasets, splits, and software versions.
