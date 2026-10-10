# Stage 3 - Particle identification

Which hadron produced this shower? The maintained two-class study uses
0 = pi- and 1 = proton. The older three-class outputs remain in the repository
only for traceability.

```bash
make pid                                                 # default: 5 seeds, timing on
python bdt/train.py --no-timing            # the timing comparison
python bdt/train.py --sigma-time 0.05      # another timing resolution
python bdt/train.py --no-smote             # class weights instead of SMOTE
```

**Reads** `$SDHCAL_DATA_DIR/params/130k_<particle>_E1to130_params.root`.

**Writes** `$SDHCAL_OUT_DIR/pid_lgbm/`: confusion matrix, training curve, score
histograms per class, feature importance, the model with its feature list, and
the run record.

## Contents

| Directory | Status |
|---|---|
| `bdt/` | **The maintained classifier.** LightGBM, with SMOTE applied to the training split only, early stopping on validation, and multi-seed uncertainties. |
| `mlp/` | Maintained binary scikit-learn MLP baseline on the same engineered variables. |
| `gnn/` | Binary pion/proton PyTorch Geometric workflow on hit-level graphs. `train.py` is maintained; the other variants are archived internship prototypes. Needs `requirements-gnn.txt`. |
| `cnn/` | Binary pion/proton hit-level baseline using permutation-invariant point-wise convolutions and masked pooling. |

### Main scripts

- `bdt/train.py`: maintained multiclass classifier.
- `bdt/interpret.py`: correlation and permutation-importance diagnostics.
- `mlp/train.py`: maintained binary pion/proton MLP.
- `mlp/evaluate.py`: evaluation of a saved model on the discrete sample.
- `mlp/importance.py`: permutation importance on that external validation.
- `gnn/train.py`: maintained binary pion/proton graph workflow.
- `cnn/train.py`: maintained binary pion/proton convolutional hit-set workflow.

The older binary BDT, AdaBoost, random-forest and experimental GNN variants are
under [`../archive/code/3_pid/`](../archive/code/3_pid/). Their historical run
outputs are under [`../archive/results/3_pid/`](../archive/results/3_pid/).

New LightGBM runs write plots, tables, arrays, models and their JSON run record
under `$SDHCAL_OUT_DIR/pid_lgbm/`. Historical outputs remain beside the scripts
for traceability. The hit-level GNN and CNN studies require:

```bash
pip install -r ../../requirements-gnn.txt
```

The hit-level CSVs must contain `eventNumber`, `x_coords`, `y_coords`,
`z_coords`, `time` and `thr`. Run the binary workflows from anywhere with:

```bash
python analysis/3_pid/cnn/train.py --pion-csv /path/pi.csv --proton-csv /path/proton.csv
python analysis/3_pid/gnn/train.py --pion-csv /path/pi.csv --proton-csv /path/proton.csv
```

Both scripts make event-disjoint stratified splits, fit preprocessing on train
only, apply timing smearing exactly once to all events, save the best-validation
checkpoint, and report AUC, accuracy, balanced accuracy, class efficiencies and
purities under `$SDHCAL_OUT_DIR`.

The shower-variable MLP reads the parameter shards derived from `new_data`
under `$SDHCAL_SHARDS_DIR` (default: `<repo>/new_data_params`). It trains and
calibrates only on the continuous `200k` production and evaluates only on the
independent `100k` discrete production:

```bash
python analysis/3_pid/mlp/train.py
python analysis/3_pid/mlp/train.py \
  --max-shards-per-particle 1 --max-events-per-particle 1000 --max-epochs 2
```

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
  independent per variable and applied exactly once.
