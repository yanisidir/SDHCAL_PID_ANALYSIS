# SDHCAL Particle Identification and Energy Reconstruction

This repository contains my M2 internship research software for studying particle identification and energy reconstruction in a Semi-Digital Hadronic Calorimeter (SDHCAL). It combines ROOT/C++ shower-parameter extraction with machine-learning and chi2/TMinuit reconstruction studies for simulated hadronic showers, with an emphasis on clear analysis workflows rather than a production-ready experiment framework.

**Skills demonstrated**

- Particle-physics data analysis with ROOT/C++, LCIO-derived detector outputs, and shower observables.
- Supervised learning for hadron PID using LightGBM BDTs, scikit-learn MLPs, and exploratory graph neural networks.
- Energy reconstruction with both ML regressors and a semi-digital chi2/TMinuit calibration approach.
- Reproducible research organization: seeded splits, per-run configuration records, performance tables, and method comparisons.
- Scientific communication through documented pipelines, result summaries, and portfolio-oriented repository structure.
- Critical review of one's own analysis: the defects found in this code and what was done about them are documented in
  [docs/corrections.md](docs/corrections.md).

## Project Overview

The SDHCAL is a highly granular hadronic calorimeter concept using semi-digital readout: each cell records threshold information rather than a fully analog energy deposit. This readout is well suited to imaging hadronic showers, but it also makes reconstruction dependent on shower topology, threshold occupancies, and calibration choices.

The project investigates two connected tasks:

1. **Particle identification (PID)** for simulated pi-, K0, and proton showers.
2. **Energy reconstruction** using shower features, semi-digital hit counts, and particle-dependent or particle-aware reconstruction strategies.

The repository is meant to show the analysis logic, code organization, and representative outputs from the internship. Large datasets and trained artifacts are not versioned.

## Physics Motivation

Hadronic calorimetry is challenging because hadron showers fluctuate strongly event by event. In a semi-digital detector, the information available per hit is compressed into threshold levels, so reconstruction relies on both counting information and spatial shower development. Useful PID and energy estimators can exploit:

- longitudinal shower development, such as first active layers, barycentres, RMS, and fitted profile quantities;
- transverse morphology, such as density, radius, eccentricity, and clustering;
- semi-digital threshold content, such as N1, N2, N3, Thr1, Thr2, Thr3, and threshold ratios;
- timing or timing-derived features where available;
- correlations between the particle hypothesis and the optimal energy reconstruction.

## Detector and Data Context

The analysis is based on simulated SDHCAL-like samples. The full upstream chain is external to this repository:

```text
SDHCAL simulation (.slcio)
        -> digitization (.slcio)
        -> LCIO to ROOT conversion
        -> shower-parameter extraction
        -> PID and energy reconstruction studies
```

The repository assumes ROOT trees containing either per-hit detector information or derived shower parameters. Many scripts still contain local research-path defaults from the internship environment; for reuse, those paths should be changed to local data locations.

## Main Objectives

- Extract physically interpretable shower variables from digitized calorimeter hits.
- Compare hadronic shower variables across particle species.
- Train and evaluate PID models for pi-, K0, and proton separation.
- Reconstruct incident energy using ML regression and chi2/TMinuit semi-digital calibration.
- Study how PID decisions affect downstream energy reconstruction.

## Methods

**Shower feature extraction**

- ROOT/C++ event loop over digitized hits.
- Computation of threshold counts, shower start, barycentre, RMS, density, radius, clustering, longitudinal-profile quantities, timing summaries, and topology features.
- Also written per event: `eventNumber`, so a held-out sample can be proved disjoint from the training events, and `touchesLastLayer`, so the machine-learning chain can apply the same containment cut as the chi2 chain.
- Main code: `analysis/1_shower_features/`.

**Particle identification**

- LightGBM BDT classifiers using engineered shower variables: `analysis/3_pid/bdt/LGBM_classifier_PID.py`.
- Class imbalance handled by SMOTE on the training split only, or by LightGBM class weights; when hyper-parameters are searched, the resampling runs inside each cross-validation fold.
- scikit-learn MLP baselines with standardization: `analysis/3_pid/mlp/`.
- Exploratory PyTorch Geometric GNN models using hit-level or graph-like shower representations.
- Main code: `analysis/3_pid/`.

**Energy reconstruction**

- LightGBM and MLP regressors trained on derived shower parameters, in log(E) with a bias-corrected back-transform: `hadron_energy_reco_lgbm.py` and `mlp_energy_reco.py`, both taking `--particles pi|kaon|proton|mixed|all`.
- Semi-digital weighted least-squares calibration with TMinuit on the threshold multiplicities N1, N2, N3, fitted on a training split and evaluated on a held-out one.
- Main code: `analysis/4_energy/` and `analysis/4_energy/chi2_tminuit/`.

**PID-energy coupling**

- Three cases, quoted together: no PID at all (one inclusive regressor), perfect PID (routing by the true particle) and real PID (routing by the classifier).
- Main code: `analysis/5_pid_energy/pid_energy_coupling.py`, driven by `--pid` and `--reco`. See [its README](analysis/5_pid_energy/README.md).

## Repository Structure

```text
.
├── src/sdhcal/              Shared Python library, installed with `pip install -e .`
│                              paths  rootio  smearing  splits  metrics  target  runlog  calib
├── cpp/                     Shared C++ used by the ROOT stages
│   ├── include/                ShowerAnalyzer.h, SdhcalPaths.h
│   └── src/                    ShowerAnalyzer.cpp
│
├── analysis/                THE CHAIN, one directory per stage
│   ├── 1_shower_features/      digitized hits -> shower variables  (C++/ROOT)
│   ├── 2_variable_study/       how the variables differ by species (ROOT macros)
│   ├── 3_pid/                  bdt/ mlp/ randomforest/ gnn/ cnn/
│   ├── 4_energy/               ml_bdt/ ml_mlp/ chi2_tminuit/
│   └── 5_pid_energy/           does identifying first help?
│
├── calibration/             chi2 coefficients: one file, read by C++ and Python
├── tools/                   utilities outside the chain (held-out sample, log repair)
├── tests/                   test suite for the shared library
├── docs/                    documentation, including the record of corrections
├── results/                 curated figures referenced by the report
├── report/                  M2 internship LaTeX report and compiled PDF (self-contained)
├── data/                    detector data, not versioned
│
├── Makefile                 make check, and one target per stage
├── pyproject.toml           package metadata, pytest and ruff configuration
├── requirements.txt         Python dependencies
├── CITATION.cff             Suggested citation metadata
└── LICENSE.md               Non-binding license selection note
```

The numbering of `analysis/` gives the reading order of the analysis chain.
Stages 3 and 4 are independent branches off stage 1 and stage 5 joins them; see
[analysis/README.md](analysis/README.md) for the diagram and the conventions
every stage shares.


See also:

- [Project overview](docs/project_overview.md)
- [Data pipeline](docs/data_pipeline.md)
- [Model summary](docs/model_summary.md)
- [Results summary](docs/results_summary.md)
- [Repository map](docs/repository_map.md)
- [Corrections](docs/corrections.md) - defects found in this code, and what was done

## Internship Report

The M2 internship report is included in [report/](report/) as a LaTeX project with its compiled PDF, [report/main.pdf](report/main.pdf). The report provides the longer scientific narrative behind the code: SDHCAL detector context, semi-digital calorimetry, hadronic shower variables, PID methods, energy reconstruction, and PID-energy coupling studies.

See [report/README.md](report/README.md) for the report structure and compilation instructions.

## Data Availability

The raw, digitized, converted ROOT files, generated CSV tables, trained models, and large intermediate arrays are not included in this repository. They are too large for GitHub and depend on external simulation and reconstruction environments.

In principle, the data chain depends on external tools such as SDHCAL simulation, digitization, LCIO/Marlin processing, ROOT, and local storage paths. Lightweight figures and selected performance summaries are kept in the repository when useful for review.

## Getting Started

```bash
make venv                 # create .venv, install the dependencies and the sdhcal package
source .venv/bin/activate
make check                # compile every source, run the tests, check the C++
```

Data locations come from the environment rather than from paths written into the
scripts:

```bash
export SDHCAL_DATA_DIR=/path/to/root/files      # inputs   (default: ./data)
export SDHCAL_MODEL_DIR=/path/to/models         # models   (default: ./models)
export SDHCAL_OUT_DIR=/path/to/output           # outputs  (default: ./output)
```

A missing dataset raises an error naming the variable to set, instead of failing
inside uproot. The expected layout under `SDHCAL_DATA_DIR` is declared in
[`src/sdhcal/paths.py`](src/sdhcal/paths.py).

## Running the Analyses

One target per stage of the chain, in order:

```bash
make features     # 1. digitized hits -> shower variables           (C++/ROOT)
make variables    # 2. compare the variables between species        (ROOT macros)
make pid          # 3. hadron identification                        (LightGBM)
make energy       # 4. energy reconstruction, ML and chi2
make coupling     # 5. does identifying first help? the three cases
```

`make help` lists every target. Arguments can be overridden without editing
anything:

```bash
make pid PID_ARGS="--no-timing"
make energy ENERGY_ARGS="--particles pi --seeds 10 --require-contained"
```

Each target simply calls the stage's own script, so you can always call it
directly:

```bash
# 3. identification, with and without the timing variables
python analysis/3_pid/bdt/LGBM_classifier_PID.py --sigma-time 0.1 --seeds 5
python analysis/3_pid/bdt/LGBM_classifier_PID.py --no-timing

# 4. energy, per particle and with one inclusive regressor
python analysis/4_energy/ml_bdt/hadron_energy_reco_lgbm.py --particles all --seeds 5
python analysis/4_energy/ml_bdt/hadron_energy_reco_lgbm.py --particles mixed
root -l -b -q 'analysis/4_energy/chi2_tminuit/EnergyReco.C("pi")' && make calibration

# 5. the coupling study: these three numbers belong in the same table
python analysis/5_pid_energy/pid_energy_coupling.py --pid none --reco bdt   # no PID
python analysis/5_pid_energy/pid_energy_coupling.py --pid true --reco bdt   # perfect PID
python analysis/5_pid_energy/pid_energy_coupling.py --pid bdt  --reco bdt   # real PID
```

Every run writes its full configuration - feature list, smearing width, seeds,
git commit, package versions - to `$SDHCAL_OUT_DIR/<study>/runs/*.json`, so a
figure can be traced back to the configuration that produced it. See
[analysis/README.md](analysis/README.md) for the conventions every stage shares.

## Representative Results and Figures

The figures below are examples of outputs currently stored in the repository. They are visual evidence of the analysis workflow, not a claim of publication-level performance, and they predate the corrections in [docs/corrections.md](docs/corrections.md) - see [Limitations](#limitations).

<p align="center">
  <img src="results/pid-energy-coupling/summary/confusion_matrix_pid_LGBM.png" width="31%" alt="Example PID confusion matrix">
  <img src="analysis/4_energy/ml_bdt/plots/Lin_n_Dev_all_LGBM.png" width="31%" alt="Example ML energy reconstruction linearity and deviation">
  <img src="results/energy-reconstruction/chi2-tminuit/summary-plots/Lin_n_Dev_all_chi2.png" width="31%" alt="Example chi2/TMinuit energy reconstruction linearity and deviation">
</p>

Additional useful figure locations:

- `results/pid/bdt/feature-interpretation/` - LightGBM feature-importance, SHAP-style summaries, and tree visualizations.
- `analysis/3_pid/gnn/plots/` - GNN training and confusion-matrix plots.
- `analysis/4_energy/ml_bdt/plots/` - ML energy linearity, relative deviation, and resolution plots.
- `analysis/4_energy/ml_mlp/results_*_energy_reco/plots/` - MLP regression curves. The per-particle folders are not trustworthy: the three scripts that wrote them were identical and all read the pion file.
- `results/energy-reconstruction/chi2-tminuit/summary-plots/` - chi2/TMinuit reconstruction profiles and diagnostics.
- `results/shower-variables/plots/` - shower-variable overlays and comparisons.
- `results/pid-energy-coupling/summary/` - PID-energy coupling summary confusion matrices.
- `analysis/5_pid_energy/*/plots/` - coupling comparisons. What is labelled "no PID" there is in fact perfect PID; see [analysis/5_pid_energy/README.md](analysis/5_pid_energy/README.md).

## Limitations

Read these before quoting any number from this repository.

- **The stored figures predate the corrections.** Everything under `results/`
  and in `report/` was produced by the code as it stood before the review
  recorded in [docs/corrections.md](docs/corrections.md). Several of those
  corrections change a quoted quantity: in particular the resolution curves
  understate sigma by about 21 %, because the 90 %-truncated RMS was never
  corrected for the truncation, and the chi2 coefficients were fitted and
  evaluated on the same events.
- **The chi2-versus-ML comparison is not yet like-for-like.** The chi2 fit now
  splits train and test, but it has to be re-run; until then the committed
  coefficients are flagged `fitted_on_train_split: false`.
- **The PID-agnostic baseline has never been measured.** The scripts named
  `noPID_*` routed events by the true particle, so what was compared was real
  PID against perfect PID. `--pid none` now exists and needs to be run.
- **The held-out sample has not been verified.** Whether `val_set.root` shares
  events with the training files was never established. `tools/build_holdout.py
  --check` answers the question; if it fails, the end-to-end numbers are
  optimistic by an unknown amount.
- **Per-particle MLP energy results should be disregarded.** The three scripts
  that were supposed to produce them were byte-identical, read the pion file,
  and did not run at all.
- **No figure in the repository can be attributed to a configuration**, because
  the feature list and the smearing width were never logged and the run logs
  were column-shifted. The logs are repaired and new runs record everything, but
  the older figures cannot be reconstructed after the fact.
- The large input datasets are not included, so direct reruns need the
  simulation chain.
- The GNN and CNN studies are exploratory and no result in the report depends on
  them.

## Future Improvements

- Re-run every study with the corrected estimators and publish a single
  benchmark table with uncertainties from the multi-seed runs.
- Add a small synthetic fixture so the pipelines, not just the shared package,
  can be smoke-tested in CI.
- Scan the timing resolution instead of fixing sigma = 0.1 ns, now that the
  smearing is independent per variable and configurable.
- Record a container or lock file for the ROOT and Python versions.
- Treat `Begin = 48` as a categorical "not found" flag for the neural models
  rather than an extreme numerical value.

## Citation

If this repository is useful for orientation or comparison, please cite it using the metadata in [CITATION.cff](CITATION.cff). The repository represents internship research software, not a peer-reviewed publication.

## License

No final license has been selected yet. See [LICENSE.md](LICENSE.md) for a non-binding recommendation. A permissive code license such as MIT or BSD-3-Clause would usually be appropriate for portfolio research code, but dataset rights, collaboration rules, and figure ownership should be checked before choosing one.

## Contact

**Author:** IDIR Mohamed Yanis

**Email:** yanis.idr@outlook.fr
