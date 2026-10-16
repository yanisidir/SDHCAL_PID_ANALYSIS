# Stage 4 - Energy reconstruction

What was the energy of the incident hadron? Two independent methods, which the
report compares:

| Directory | Method |
|---|---|
| `ml_bdt/` | **LightGBM regressors** on the shower variables. The maintained pipeline. |
| `ml_mlp/` | scikit-learn MLP regressors, same inputs, for comparison. |
| `gnn/` | Hit-level PyTorch Geometric regressor for pion, proton, or a PID-agnostic pion+proton mixture. |
| `chi2_tminuit/` | **Semi-digital calibration**: a weighted least-squares fit with TMinuit on the threshold multiplicities N1, N2, N3. |

```bash
make energy                                      # both methods
make energy-ml                                   # LightGBM, per particle and inclusive
make energy-chi2                                 # the chi2 fit, then publish its coefficients

python ml_bdt/train.py --particles pi --seeds 5
python ml_bdt/train.py --particles mixed     # one inclusive regressor
python ml_bdt/train.py --particles mix-pi-proton
```

**Reads** the configured consolidated files for LightGBM, the parameter shards
derived from `data/lots/pi-proton/hits` for the MLP, the hit shards in `data/lots/pi-proton/hits` for the GNN,
and the digitized plus raw files for the chi2 fit.

**Writes** `$SDHCAL_OUT_DIR/energy_reco_{lgbm,mlp,gnn,chi2}/`.

## Hit-level GNN (pion/proton)

`gnn/train.py` reads the correct sharded productions directly from
`$SDHCAL_HIT_SHARDS_DIR` (default: `<repo>/data/lots/pi-proton/hits`):

- `200k_pi-` and `200k_proton`, continuous from 1 to 130 GeV, are used for
  training and internal calibration (early stopping and target correction);
- `100k_pi-_discret` and `100k_proton_discret`, independent productions at
  fixed energies from 5 to 100 GeV, are kept entirely as external validation.

Each ROOT event already contains `energy`, `I/J/K` (or `x/y/z`), `time` and
`thr`; no separate generator file is paired by row. The default model is one
PID-agnostic pion+proton regressor:

```bash
python analysis/4_energy/gnn/train.py --particles mix-pi-proton
```

Particle-specific runs and a small smoke run are available with:

```bash
python analysis/4_energy/gnn/train.py --particles pi
python analysis/4_energy/gnn/train.py --particles proton
python analysis/4_energy/gnn/train.py \
  --particles mix-pi-proton --max-shards-per-particle 1 --max-events-per-shard 100
```

The network predicts `log(E)`, includes shower hit multiplicity explicitly, and
fits the back-transform correction on the internal 200k calibration split only.
Graphs are generated lazily per batch rather than retaining hundreds of
thousands of graphs in RAM. The 100k validation sample never affects training.
Results, checkpoints, preprocessing, predictions, linearity, bias and
resolution are written under `$SDHCAL_OUT_DIR/energy_reco_gnn/`.

## Shower-variable MLP (pion/proton)

`ml_mlp/train.py` uses the parameter shards derived from `data/lots/pi-proton/hits`
under `$SDHCAL_SHARDS_DIR` (default: `<repo>/data/lots/pi-proton/params`). As for the GNN,
the continuous `200k` lots provide training and internal calibration, while the
entire independent `100k` discrete lots are reserved for external validation.

```bash
python analysis/4_energy/ml_mlp/train.py --particles all
python analysis/4_energy/ml_mlp/train.py --particles mix-pi-proton
python analysis/4_energy/ml_mlp/train.py --particles pi \
  --max-shards-per-particle 1 --max-events-per-particle 1000 --max-epochs 2
```

## The LightGBM `mixed` mode matters

`--particles mixed` trains **one** regressor on all three species. It is the
PID-agnostic baseline that stage 5 needs, and it did not exist in the original
code - which is why the coupling study never had a real baseline to compare
against.

## How bias and resolution are defined

Reported separately, because a single number conflates them:

| Quantity | Definition |
|---|---|
| bias | mean of `(E_reco - E_true)/E_true`: the energy-scale error |
| resolution | spread of the same residual around its own mean, estimated on the distribution core and corrected for the truncation (divide by 0.7893 for the central 90 % of a Gaussian) |
| `rms_rel_about_zero` | `sqrt(mean(r^2))`, the historical `sigma_over_E`. Equal to `sqrt(bias^2 + resolution^2)`, so not a resolution; kept only so older log entries stay comparable |

The regressors train on `log E` and correct the back-transform with Duan's
smearing factor, fitted on a held-out calibration split (called validation in
the older pipelines): exponentiating a log-space fit otherwise returns a
geometric mean, a few per cent below the arithmetic mean.
`--no-bias-correction` reproduces the old, uncorrected behaviour.

## Comparing the two methods fairly

Three conditions, none of which held in the original code:

1. **The chi2 fit must be evaluated out-of-sample.** It now splits 80/20 and
   draws every figure from the test part. The coefficients currently committed
   in [`../../calibration/`](../../calibration/) are flagged
   `fitted_on_train_split: false` and must be regenerated before the comparison
   is quoted.
2. **Both must run on the same events.** The chi2 chain rejects showers reaching
   the last layer; pass `--require-contained` to the ML pipeline to apply the
   same cut.
3. **The coefficients must be the same on both sides.** They live in one JSON
   file, written by the fit and read by `sdhcal.calib`; the C++ and Python copies
   had already diverged when they were separate.
