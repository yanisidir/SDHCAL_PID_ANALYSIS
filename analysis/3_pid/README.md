# Stage 3 - Particle identification

Which hadron produced this shower: pi-, p or K0_L? The integer class labels are
fixed by the order in `sdhcal.paths.PARTICLES`, i.e. 0 = pi-, 1 = p, 2 = K0_L,
and every model and every confusion matrix in the repository uses that order.

```bash
make pid                                                 # default: 5 seeds, timing on
python bdt/LGBM_classifier_PID.py --no-timing            # the timing comparison
python bdt/LGBM_classifier_PID.py --sigma-time 0.05      # another timing resolution
python bdt/LGBM_classifier_PID.py --no-smote             # class weights instead of SMOTE
```

**Reads** `$SDHCAL_DATA_DIR/params/130k_<particle>_E1to130_params.root`.

**Writes** `$SDHCAL_OUT_DIR/pid_lgbm/`: confusion matrix, training curve, score
histograms per class, feature importance, the model with its feature list, and
the run record.

## Contents

| Directory | Status |
|---|---|
| `bdt/` | **The maintained classifier.** LightGBM, with SMOTE applied to the training split only, early stopping on validation, and multi-seed uncertainties. |
| `mlp/` | scikit-learn MLP baselines on the same engineered variables. |
| `randomforest/` | tree-based baseline, for comparison. |
| `gnn/` | exploratory PyTorch Geometric studies on hit-level graphs. Six variants kept as written during the internship; no result in the report depends on them. Needs `requirements-gnn.txt`. |
| `cnn/` | exploratory convolutional approach, same status. |

## What to be careful about

- **The class imbalance is handled on the training split only.** Oversampling
  before a cross-validation split leaks: SMOTE interpolates between neighbours,
  so a synthetic point built from a training-fold event can land in the
  validation fold. When `--grid-search` is used, the resampling runs inside each
  fold, through an `imblearn` pipeline.
- **Accuracy alone is misleading** on a test split that keeps the natural class
  proportions, so balanced accuracy, and per-class efficiency and purity, are
  reported next to it.
- **The timing variables are smeared** before use, to represent a finite
  detector time resolution. The width is `--sigma-time` and is recorded in the
  run log; it used to be hard-coded at 0.1 ns in five places. The smearing is
  independent per variable and applied exactly once - neither was true before,
  see [`../../docs/corrections.md`](../../docs/corrections.md) section 2.8.
