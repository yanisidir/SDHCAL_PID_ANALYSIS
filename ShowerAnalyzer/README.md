# Shower Analyzer

This directory contains the ROOT/C++ code used to extract shower parameters from digitized SDHCAL hit information.

## Main Files

- `ShowerAnalyzer.h` and `ShowerAnalyzer.cpp`: implementation of shower-observable calculations.
- `computeParams.cpp`: event-loop driver for producing parameter ROOT trees.
- `computeParams_parallel.cpp`: parallelized variant of the parameter extraction driver.
- `run_parallel.sh`: shell helper for parallel execution.

## Example Observables

The analyzer computes variables used downstream for PID and energy reconstruction, including:

- threshold counts and ratios;
- shower start and longitudinal barycentre;
- transverse radius and density;
- clustering summaries;
- timing summaries;
- fitted longitudinal-profile quantities;
- topology-related variables such as track-segment counts.

## Outputs

The expected output is a ROOT file containing a `paramsTree` with derived shower variables. Compiled objects and binaries are generated artifacts and should not be treated as source code.
