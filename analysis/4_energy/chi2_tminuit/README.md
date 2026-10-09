# Semi-digital energy reconstruction (TMinuit)

Reconstructs the incident energy from the threshold multiplicities alone, with no
machine learning. It is the physics-motivated counterpart to the regressors in
[`../ml_bdt/`](../ml_bdt/), and it keeps the semi-digital response explicit.

```bash
make energy-chi2                                              # the three fits, then publish
root -l -b -q 'EnergyReco.C("pi")'                            # one particle
root -l -b -q 'EnergyReco.C("kaon", 0.2, 42, true)'           # test fraction, seed, containment cut
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
| `EnergyReco.C` | The fit. Takes the particle, the test fraction, the split seed and the containment flag. |
| `ResoAndLin.C` | Resolution and linearity summary plots. |
| `pion_proton_EnergyReco.C` | Coupled pion/proton study. |
| `validate_root_file.C` | Input-file sanity check. |
| `kaon/`, `pion-/`, `proton/` | Linear-model variants (`EnergyReco_linear.C`, `EnergyReco2.C`), the coefficient-evolution macro `plot_coeff.C`, and the logs and figures of the internship runs. |

The per-particle copies of the quadratic fit that used to live in those three
subdirectories were removed: `EnergyReco.C` takes the particle as an argument.
The linear variants are a genuine alternative model and were kept.

Summary figures promoted for the report are in
[`../../../results/energy-reconstruction/chi2-tminuit/summary-plots/`](../../../results/energy-reconstruction/chi2-tminuit/summary-plots/).
