# Data Pipeline

This repository does not include the large input datasets. The expected workflow starts from simulated SDHCAL events and ends with ROOT parameter trees, CSV summaries, trained models, and plots.

## Pipeline Stages

```text
Simulation
  -> digitization
  -> LCIO to ROOT conversion
  -> shower parameter extraction
  -> PID studies
  -> energy reconstruction studies
  -> PID-energy coupling comparisons
```

## Expected Data Products

The scripts assume several types of data products:

- `.slcio` files from detector simulation and digitization stages.
- ROOT files containing per-hit information, usually read from a `tree`.
- ROOT files containing derived shower parameters, usually read from `paramsTree`.
- Generated CSV logs for run parameters, comments, and performance summaries.
- Generated figures in `plots/` or `results_*` directories.
- Trained model artifacts such as `.joblib`, `.pt`, or `.pth` files.

Large binary and generated files are intentionally ignored by `.gitignore` for future commits.

## Shower Parameters

The central derived-data product is a ROOT tree of shower variables. Typical variables include:

- threshold occupancy: `Thr1`, `Thr2`, `Thr3`, `N1`, `N2`, `N3`, `ratioThr23`;
- longitudinal development: `Begin`, `Zbary`, `Zrms`, `Nmax`, `z0_fit`, `Xmax`, `lambda`;
- transverse morphology: `Radius`, `Density`, `lambda1`, `lambda2`, `eccentricity3D`;
- clustering and topology: `NClusters`, `AvgClustSize`, `MaxClustSize`, `nTrackSegments`;
- timing summaries where available: `tMin`, `tMax`, `tMean`, `tSpread`.

These variables are extracted mainly through `analysis/1_shower_features/`.

## Per-event bookkeeping branches

Besides the shower observables, `ShowerAnalyzer` writes two branches that exist
for the integrity of the analysis rather than for the physics:

| Branch | Why it exists |
|---|---|
| `eventNumber` | Lets a held-out sample be proved disjoint from the training events. Without it the coupling study could not tell generalisation from training performance. Taken from the input tree when present, otherwise the row index, with a warning. |
| `touchesLastLayer` | Flags showers reaching the last active layer (`K == 47`). The chi2 chain rejects them; exposing the flag lets the ML chain apply the identical cut, so the two methods are compared on the same events. |

`Begin` carries the sentinel value 48 when no shower start is found - one past
the last layer, deliberately outside the physical range. It is a "not found"
marker, not a depth.

## Reproduction Notes

To reproduce the full analysis, a user would need:

- an SDHCAL-compatible simulation and digitization environment;
- ROOT and C++ build support (`make shower-analyzer`);
- LCIO/Marlin conversion tools if starting from `.slcio`;
- the Python packages in `requirements.txt` (`make venv`);
- enough storage for ROOT files and generated model artifacts.

Data locations are no longer written into the scripts. Set:

```bash
export SDHCAL_DATA_DIR=/path/to/root/files      # inputs   (default: ./data)
export SDHCAL_MODEL_DIR=/path/to/models         # models   (default: ./models)
export SDHCAL_OUT_DIR=/path/to/output           # outputs  (default: ./output)
```

The expected layout under `SDHCAL_DATA_DIR` is declared once, in
`src/sdhcal/paths.py`:

```text
$SDHCAL_DATA_DIR/
|-- params/130k_<particle>_E1to130_params.root
|-- merged_primaryEnergy/130k_<particle>_E1to130_params_merged.root
|-- digitized/130k_<particle>_E1to130_digitized.root
|-- raw/130k_<particle>_E1to130.root
`-- val_set.root                                 built by tools/build_holdout.py
```

A missing file raises an error naming the variable to set, instead of failing
inside uproot or returning an empty frame.

## Building the Held-Out Sample

The evaluation sample is not a leftover of a training split: it is carved out
first, with a recorded seed, and the two halves are asserted disjoint.

```bash
python tools/build_holdout.py --fraction 0.15 --seed 20240601 --write-train-pool
python tools/build_holdout.py --check      # verify an existing val_set.root
```

This writes `val_set_manifest.json` next to the data, recording the fraction,
the seed, the per-particle counts and a digest of the held-out event numbers.
