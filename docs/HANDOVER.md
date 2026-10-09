# Project Handover

This document is the starting point for a student continuing the SDHCAL PID and
energy-reconstruction study. It describes the current state of the repository,
the external material that is still needed, and the safest order in which to
resume the analysis.

## Read This First

1. Read the [root README](../README.md) for the physics goal and headline
   results.
2. Read the [analysis-chain guide](../analysis/README.md) for the five stages.
3. Read [data_pipeline.md](data_pipeline.md) for the expected ROOT files and
   branch conventions.
4. Use [results_summary.md](results_summary.md) and the
   [final report](../report/main.pdf) as the reference for numerical claims.

The report is a scientific account of the internship. The repository also
contains historical scripts and figures, which are useful for traceability but
are not all valid benchmark outputs.

## Current Scientific Baseline

- The final DRD6 PID results were evaluated on samples distinct from the
  training data.
- The retained multiclass LightGBM balanced accuracy is **67.8 %**, with recalls
  of **48.8 %** for pi-, **68.6 %** for protons and **86.0 %** for K0L.
- In the coupled analysis, the main supported conclusion concerns the energy
  scale bias. The historical width curves are qualitative and must not be used
  to claim a numerical gain in absolute resolution.
- The GNN and CNN directories contain exploratory studies, not the reference
  PID pipeline.

## What Is and Is Not Versioned

The source code, selected figures, calibration file and final report are in the
repository. The large simulation products, ROOT inputs, trained models and full
run outputs are not.

To reproduce the analysis, obtain the following from the previous student,
supervisor or collaboration storage:

```text
$SDHCAL_DATA_DIR/
|-- params/130k_<particle>_E1to130_params.root
|-- merged_primaryEnergy/130k_<particle>_E1to130_params_merged.root
|-- digitized/130k_<particle>_E1to130_digitized.root
|-- raw/130k_<particle>_E1to130.root
`-- val_set.root
```

Here `<particle>` is `pi`, `proton` or `kaon`. If an existing evaluation sample
is recovered, also recover its manifest or verify it with
`tools/build_holdout.py --check` before using it.

## First Setup

Requirements:

- Python 3.10 or newer;
- ROOT with `root`, `root-config` and C++17 support;
- enough local or shared storage for the ROOT samples.

From the repository root:

```bash
make venv
source .venv/bin/activate

export SDHCAL_DATA_DIR=/path/to/root/files
export SDHCAL_MODEL_DIR=/path/to/models
export SDHCAL_OUT_DIR=/path/to/output

make check
```

`make check` does not require the physics datasets. It checks the Python code,
the shared-library tests and, when ROOT is available, the C++ sources.

## Recommended Execution Order

Run one stage at a time and inspect its output before continuing:

```bash
make features
make variables
make pid
make energy
make coupling
```

The stage-specific README files document their inputs and outputs. Do not start
with `make coupling`: it requires a trained classifier, three particle-specific
energy regressors and an inclusive regressor.

New outputs belong under `$SDHCAL_OUT_DIR`; do not overwrite the committed
historical figures. Each maintained pipeline writes a JSON run record containing
the feature list, seed, software versions and artifact paths.

## Model Artifacts Used by the Coupled Analysis

Stage 5 accepts explicit model paths. For a stable local setup, publish the
selected artifacts as:

```text
$SDHCAL_MODEL_DIR/pid_bdt.joblib
$SDHCAL_MODEL_DIR/energy_bdt_pi.joblib
$SDHCAL_MODEL_DIR/energy_bdt_proton.joblib
$SDHCAL_MODEL_DIR/energy_bdt_kaon.joblib
$SDHCAL_MODEL_DIR/energy_bdt_mixed.joblib
```

The `mixed` model is essential: it is the genuinely PID-independent baseline.
The old files named `noPID_*` used the true particle identity and therefore
represent perfect PID, not an analysis without PID.

## Known Limitations

- The ROOT inputs used for the final presentation are not committed, so their
  provenance cannot be reconstructed from Git alone.
- The committed semi-digital calibration coefficients predate the corrected
  train/test procedure and are marked `fitted_on_train_split: false`. Refit them
  before making a quantitative chi2-versus-ML comparison.
- Historical `sigma_over_E` values combine bias and spread. Use the estimators
  in `src/sdhcal/metrics.py`, which report them separately.
- Historical width plots use a truncated core without the correction required
  for an absolute resolution measurement.
- Old per-particle MLP energy outputs must not be reused: the original scripts
  did not represent three valid independent reconstructions.
- Several GNN/CNN and legacy analysis scripts are retained as exploratory work;
  maintained entry points are identified in the stage README files and Makefile.

## Priority Work for the Next Student

1. Recover and document the simulation datasets and their production settings.
2. Build or verify a held-out sample and keep its event-level manifest.
3. Refit the semi-digital calibration on its training partition.
4. Retrain PID and energy models with several seeds and retain their run JSONs.
5. Run the coupled study for `--pid none`, `--pid true` and `--pid bdt` on the
   same events.
6. Produce one benchmark table containing bias, resolution and statistical
   uncertainties for all three cases.
7. Validate the simulation-based conclusions with beam-test data when such data
   become available.

## Rules That Protect the Comparison

- Never tune a model on the final evaluation sample.
- Apply resampling such as SMOTE only inside the training partition.
- Compare reconstruction methods on the same event selection; use
  `touchesLastLayer` to align the containment cut.
- Keep `eventNumber` in derived trees so sample disjointness can be checked.
- Keep bias and resolution separate.
- Record the data version, features, seeds, time smearing and software versions
  for every result that may be quoted.

