# Semi-digital energy reconstruction (TMinuit)

Reconstructs the incident energy from the threshold multiplicities alone, with no
machine learning. It is the physics-motivated counterpart to the regressors in
[`../ml_bdt/`](../ml_bdt/), and it keeps the semi-digital response explicit.

```bash
make energy-chi2                                              # the three fits, then publish
root -l -b -q 'fit.C("pi")'                            # one particle
root -l -b -q 'fit.C("kaon", 0.2, 42, true)'           # test fraction, seed, containment cut
python ../../../tools/update_calibration.py                   # merge the fit fragments
```

**Reads** `$SDHCAL_DATA_DIR/digitized/...` for the hits and
`$SDHCAL_DATA_DIR/raw/...` for the true energy.

**Writes** figures to `$SDHCAL_OUT_DIR/energy_reco_chi2/plots/` and the fitted
coefficients to `../../../calibration/fit_<particle>.json`.

## Model

| Symbol | Meaning |
|---|---|
| `N1`, `N2`, `N3` | hits above threshold 1, 2 and 3 |
| `Nhit` | `N1 + N2 + N3` |
| `alpha(Nhit)` | `a0 + a1*Nhit + a2*Nhit^2`, and likewise `beta`, `gamma` |
| `Ereco` | `alpha*N1 + beta*N2 + gamma*N3` |

The quadratic dependence on the total multiplicity lets the per-threshold
weights soften as the shower becomes denser and saturation sets in.

## The minimised quantity is not a chi2

```text
S = sum_i (Ereco_i - Etrue_i)^2 / Etrue_i
```

The denominator is an energy, not a variance, so this is a weighted least
squares. The macro reports `S/ndf` and says so explicitly: the number that used
to be printed as `chi2/ndf` was never a goodness-of-fit test and should not be
quoted as one.

## Comparing this with the regressors

The macro now splits the sample 80/20 with a recorded seed, fits on the training
part and draws every figure and number from the test part. Before, the fit and
the figures used the same events, so the method was quoted in-sample while the
regressors were quoted on a held-out set.

Two conditions remain for a fair comparison:

1. **Re-run the fit.** The coefficients committed in
   [`../../../calibration/`](../../../calibration/) are the ones recovered from
   the internship code and carry `"fitted_on_train_split": false`. `sdhcal.calib`
   warns when they are used.
2. **Match the event selection.** This chain rejects showers reaching the last
   layer (`requireContained`, on by default). Pass `--require-contained` to the
   ML pipeline to apply the identical cut.

## Files

| File | Role |
|---|---|
| `fit.C` | The fit. Takes the particle, the test fraction, the split seed and the containment flag. |
| `plots.C` | Resolution and linearity summary plots. |
| `validate.C` | Input-file sanity check. |
| [`../../archive/code/4_energy/chi2_tminuit/`](../../archive/code/4_energy/chi2_tminuit/) | Superseded coupled and per-particle variants, with their historical logs and figures. |

The active `fit.C` takes the particle as an argument, so the per-particle copies
are not needed by the maintained workflow.

Summary figures promoted for the report are in
[`../../../results/energy-reconstruction/chi2-tminuit/summary-plots/`](../../../results/energy-reconstruction/chi2-tminuit/summary-plots/).
