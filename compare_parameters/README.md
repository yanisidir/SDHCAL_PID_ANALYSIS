# Shower-Parameter Comparisons

This directory contains ROOT macros and stored figures comparing selected shower variables across samples or particle categories.

## Contents

- `compareRadius.C`, `compareDensity.C`, `compareBegin.C`, `compareThr3.C`, `compareEmFraction.C`: ROOT comparison macros.
- top-level `.pdf` figures: earlier comparison outputs.
- `plots/`: additional PNG/PDF overlays and comparison figures.

## Purpose

These plots help inspect whether engineered shower variables show physically meaningful separation between hadron categories. They are useful before and after PID training because they connect model inputs back to detector observables.

## Notes

The stored figures are retained as lightweight analysis outputs. The macros may require local ROOT files and path adjustments before rerunning.
