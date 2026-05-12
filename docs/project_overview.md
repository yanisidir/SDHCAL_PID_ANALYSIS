# Project Overview

This repository documents M2 internship research software for SDHCAL particle identification and energy reconstruction. The work is organized around simulated hadronic showers and focuses on how semi-digital calorimeter information can be transformed into interpretable variables, PID decisions, and reconstructed energy estimates.

## Scientific Context

The SDHCAL concept uses highly granular readout and multiple threshold levels per cell. This gives access to detailed shower shapes, but the energy information is not directly analog. Reconstruction therefore depends on a combination of threshold occupancies, spatial topology, longitudinal development, and calibration strategy.

The analysis targets three hadronic categories used throughout the repository:

- pi-
- K0
- proton

The code was written during an internship workflow, so several scripts retain local input paths and experiment-specific assumptions. The repository is intended for inspection, adaptation, and portfolio review rather than immediate turnkey rerunning.

## Main Questions

- Which shower variables are informative for distinguishing hadron species?
- How well can BDT, MLP, and GNN-style models separate particle categories from reconstructed observables or hit-level information?
- How does ML-based energy regression compare with a semi-digital chi2/TMinuit reconstruction?
- Does using PID information improve or change the energy reconstruction strategy?

## Main Components

- `ShowerAnalyzer/`: ROOT/C++ extraction of shower observables from digitized hits.
- `compare_parameters/`: comparison plots for selected shower variables.
- `PID/`: classification studies with BDT, MLP, GNN, and other exploratory baselines.
- `Energy_reconstruction_ml/`: ML regression studies for energy reconstruction.
- `energy_reconstruction_Tminuit/`: chi2/TMinuit reconstruction using semi-digital hit counts.
- `PID_RECONSTRUCTION/`: studies coupling PID decisions to energy reconstruction.

## Repository Status

This is an internship research repository. It contains scientific scripts, analysis outputs, and documentation, but it does not include the large datasets or a complete environment snapshot. The documentation aims to make the work understandable for physics supervisors, PhD application reviewers, and data-analysis or ML reviewers.
