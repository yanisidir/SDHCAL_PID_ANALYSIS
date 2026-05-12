# chi2/TMinuit Energy Reconstruction

This directory contains ROOT/C++ studies of semi-digital energy reconstruction using TMinuit.

## Method

The main approach uses threshold hit counts:

- `N1`: number of hits above threshold 1;
- `N2`: number of hits above threshold 2;
- `N3`: number of hits above threshold 3.

A reconstructed energy is built from threshold-count terms whose coefficients can depend on the total hit count. TMinuit is then used to minimize a chi2-like objective against true beam energy.

## Main Files

- `EnergyReco.C`: combined energy reconstruction macro.
- `pion_proton_EnergyReco.C`: coupled pion/proton study.
- `ResoAndLin.C`: plotting macro for resolution and linearity summaries.
- `validate_root_file.C`: ROOT-file validation helper.
- `pion-/`, `kaon/`, `proton/`: particle-specific macros and logs.
- `../results/energy-reconstruction/chi2-tminuit/summary-plots/`: generated linearity, deviation, and resolution figures moved out of this source/macro folder.

## Notes

This method is useful as a physics-motivated comparison to ML regression because it keeps the semi-digital response explicit. Existing macros contain local data paths and should be configured before reuse.
