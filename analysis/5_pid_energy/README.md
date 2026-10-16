# Stage 5 - PID and energy reconstruction, coupled

Does identifying the hadron first improve the reconstruction of its energy?

Everything is driven by one script:

```bash
python study.py --pid bdt  --reco bdt    # realistic chain
python study.py --pid true --reco bdt    # perfect-PID upper bound
python study.py --pid none --reco bdt    # no PID at all: the baseline
python study.py --pid bdt  --reco chi2   # semi-digital calibration
```

The three `--pid` modes are the point of the study and must be quoted together:

| `--pid` | Routing | Reading |
|---|---|---|
| `none` | a single inclusive regressor, trained on the mixture | the PID-agnostic baseline |
| `true` | the true PDG code | the best any PID could achieve |
| `bdt`, `mlp` | the classifier's decision | what a real detector would get |

The gap between `none` and `true` is how much particle identification could
possibly buy; the gap between `none` and `bdt` is how much it actually buys.

**What was measured: nothing.** The PID-agnostic baseline and the routed chain
land on the same resolution, 0.1199 against 0.1199, and perfect PID does not
separate from either. Routing buys nothing here, which is the result the stage
reports -- not a negative to be explained away. Two reasons stand behind it: the
shower variables the regressors read already carry whatever the classifier could
tell them, and the pion/proton separation itself is close to its physical limit
(see [`../3_pid/`](../3_pid/)). The figure to quote is the three modes side by
side, not the routed chain alone.

`--pid none` needs an inclusive regressor, trained with:

```bash
python ../4_energy/ml_bdt/train.py --particles mixed
```

## Pointing at the models

Training writes a run-scoped artifact, because several runs and several seeds
coexist:

```text
$SDHCAL_OUT_DIR/pid_lgbm/models/lgbm_pid_<run>_seed<seed>.joblib
$SDHCAL_OUT_DIR/energy_reco_lgbm/<mode>/models/lgbm_regressor_<mode>_<run>_seed<seed>.joblib
```

Which run feeds the coupling study is a decision, so it is not guessed. Either
pass the paths:

```bash
python study.py --pid bdt --reco bdt \
  --classifier "$SDHCAL_OUT_DIR/pid_lgbm/models/lgbm_pid_7_seed42.joblib" \
  --regressor pi=".../lgbm_regressor_pi_3_seed42.joblib" \
  --regressor proton=".../lgbm_regressor_proton_3_seed42.joblib" \
  --regressor kaon=".../lgbm_regressor_kaon_3_seed42.joblib"
```

or publish the chosen ones under the default convention, so the run records a
stable identity:

```text
$SDHCAL_MODEL_DIR/pid_bdt.joblib
$SDHCAL_MODEL_DIR/energy_bdt_<pi|proton|kaon|mixed>.joblib
```

Whichever is used, the resolved paths are written into the run's JSON record. A
model saved by an older script, without its feature list, is rejected rather
than fed columns in an unchecked order.

## Outputs

Per run, under `$SDHCAL_OUT_DIR/pid_energy_coupling/pid-<mode>_reco-<mode>/`:

* `tables/events_*.csv` - one row per event: true and reconstructed energy, true
  and routed label, class scores;
* `plots/bias_resolution_*.png` - bias and resolution versus true energy, with
  uncertainties, the two kept apart;
* `plots/confusion_matrix_*.png` - the PID confusion matrix, when a classifier
  is used;
* `coupling_parameters.csv`, `coupling_performances.csv` and `runs/*.json` - the
  configuration and the metrics of each run.

Earlier summary confusion matrices are kept in
[`../../results/pid-energy-coupling/summary/`](../../results/pid-energy-coupling/summary/).

These are exploratory internship studies, not final detector-performance claims.

## Historical files

The CSVs and figures here predate the current methodology, and **their names do
not say what they are**:

* `no_pid_BDT.csv`, `no_pid_chi2.csv`, `no_pid_BDT_pi_all.csv`,
  `no_pid_chi2_pion_all.csv` and `confusion_matrix_pid_param_LGBM_pi_p.png` were
  produced by the old `noPID_*.py` scripts, which routed events by the **true**
  PDG code. They are therefore **perfect-PID** results, not no-PID results.
* No genuine PID-agnostic baseline exists in this directory, because no
  inclusive regressor had been trained.

They are kept under [`../archive/results/`](../archive/results/) as part of the
internship record, but any comparison drawn from them should be labelled "real
PID versus perfect PID".

The archived plotting scripts expect the same historical CSV layout.

## The fifteen scripts that used to be here

`BDT_BDT.py`, `BDT_chi2.py`, `MLP_BDT.py`, `MLP_MLP.py`, `MLP_chi2.py`,
`noPID_BDT.py`, `noPID_MLP.py`, `noPID_chi2.py` and `hadron_pid_reco_linear.py`,
across both `pi-_proton/` and `kaon_pi-_proton/`, were copies differing by a few
literals. They are in the git history; `study.py` covers all of
their combinations and fixes three defects they shared: the mislabelled
baseline, the unverified held-out sample, and `tMin` being smeared twice.
