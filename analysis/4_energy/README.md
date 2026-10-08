# Stage 4 - Energy reconstruction

What was the energy of the incident hadron? Two independent methods, which the
report compares:

| Directory | Method |
|---|---|
| `ml_bdt/` | **LightGBM regressors** on the shower variables. The maintained pipeline. |
| `ml_mlp/` | scikit-learn MLP regressors, same inputs, for comparison. |
| `chi2_tminuit/` | **Semi-digital calibration**: a weighted least-squares fit with TMinuit on the threshold multiplicities N1, N2, N3. |

```bash
make energy                                      # both methods
make energy-ml                                   # LightGBM, per particle and inclusive
make energy-chi2                                 # the chi2 fit, then publish its coefficients

python ml_bdt/hadron_energy_reco_lgbm.py --particles pi --seeds 5
python ml_bdt/hadron_energy_reco_lgbm.py --particles mixed     # one inclusive regressor
python ml_bdt/hadron_energy_reco_lgbm.py --particles mix-pi-proton
```

**Reads** `$SDHCAL_DATA_DIR/merged_primaryEnergy/...` for the ML regressors, and
the digitized plus raw files for the chi2 fit.

**Writes** `$SDHCAL_OUT_DIR/energy_reco_{lgbm,mlp,chi2}/`.

## The `mixed` mode matters

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
smearing factor, fitted on the validation set: exponentiating a log-space fit
otherwise returns a geometric mean, a few per cent below the arithmetic mean.
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
