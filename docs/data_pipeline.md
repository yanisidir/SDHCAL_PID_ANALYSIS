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

## Corrections to the shower variables

A review of `cpp/src/ShowerAnalyzer.cpp` found four defects that changed what the
variables mean. Parameter files produced before these fixes carry the old values.

**`eccentricity3D` was inverted.** `TMatrixDSymEigen::GetEigenValues()` returns
eigenvalues in *descending* order, so `eig(2)` is the smallest, not the largest.
The variable computed the smallest eigenvalue over the trace - a sphericity,
bounded by 1/3 - while its definition (annex 1) is the largest over the trace,
tending to 1 for an elongated shower. Measured on 200 000 events: median 0.060,
max 0.295 before; median 0.843 after. Any physical reading of the old values is
backwards.

**The Gaisser-Hillas fit was not fitting.** Four causes, now fixed:

| Cause | Effect before |
|---|---|
| no parameter bounds, and `(x-z0)^k` is NaN for `x < z0` | `lambda` stayed at its seed 10.0 in **97.4 %** of events |
| the fit status was never checked, and failures wrote the seed | `z0_fit` was 0 in 21.8 %, the other three in 4.1 % |
| `MAX_LAYER = 29`, against 48 layers in the detector | `Xmax` capped at 290 mm in 99.4 % |
| `hitsByLayer[base+j]` inserted empty layers *while iterating the map* | zero-signal points injected into the profile |

The depth also came from `layer * 10 mm`, a hard-coded pitch against the SDHCAL's
~26 mm; the measured `vz` is used instead, so `z0_fit`, `Xmax` and `lambda` are
now in millimetres. On failure the four parameters are **NaN** rather than a seed
value, because a failed fit is not a measurement - `rootio.clean` then drops the
event instead of feeding a fabricated number to a model.

**`primaryEnergy` is copied** from the input when it carries one, which the
sharded lots do. That removes the separate merge step for them.

**Three variables are exactly redundant**, verified on 200 000 events:

```text
ratioThr23  = (100 - Thr1) / Thr1        (Thr1+Thr2+Thr3 = 100 exactly)
tSpread     = tMax - tMin                (tMin std 0.3 ns against tMax std 315 ns)
AvgClustSize = nHitsTotal / NClusters
nHitsTotal  = N1+N2+N3 ;  sumThrTotal = N1+2*N2+3*N3
```

Only the first two are dropped from the PID feature list. A decision tree splits
on one column at a time and is invariant to a monotone transform of it, so a
feature that is such a transform of another can never yield a split the other
could not. The rest are *combinations* of several columns, which a tree cannot
build from splits, so they earn their place. All of them stay in the files.

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
- ROOT and C++ build support (`make cpp` verifies the sources);
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

## The Two-Class Study (pion/proton)

A later delivery, now under `data/lots/pi-proton/`, holds **two independent
Geant4 productions** with two roles:

| Role | Lot | Events | Energies | Why |
|---|---|---|---|---|
| `train` | `200k_pi-`, `200k_proton` | 199 971 / 199 975 | continuous 1-130 GeV | trains the models, calibrates the chi2 |
| `val` | `100k_pi-_discret`, `100k_proton_discret` | 100 000 each | 20 fixed points, 5-100 GeV in 5 GeV steps, 5 000 events per point | the evaluation sample, and the natural binning of the resolution and linearity curves |

Because they are separate productions, **the evaluation sample is not carved out
of the training one**. Nothing has to be held out: `tools/build_holdout.py` plays
no part in this study, and the training stages must not be given
`--exclude-holdout`. The train/val separation rests on how the samples were
simulated, which no event key can prove.

The lots carry **no kaon**, so this study separates two species.

They already hold the shower variables, so the C++ extraction does not run, and
they carry the true energy, so nothing has to be merged. Three things do have to
be repaired, and `src/sdhcal/shards.py` does all three once:

| Problem in the lots | What is done |
|---|---|
| `particlePDG` and `particle1PDG` are `0` in every event | the label is taken from the lot a file belongs to |
| `eventNumber` restarts at 0 in every shard, so the 200k lots hold 5 000 distinct values for 200 000 events | `shardIndex * 100000 + localEventNumber`, with the shard id read off the *file name* so the key survives a re-listing |
| the two lots' raw shard ids overlap (0..39 against 2..41) | a per-lot offset, `LOT_ID_BASE`, so a training event and a validation event can never answer to the same key |
| `touchesLastLayer` was never written | recomputed as `any(K == 47)`; the lots kept their hit branches |

Their tree is `tree` and their energy branch `energy`; both are translated to
`paramsTree` and `primaryEnergy`.

The last two rows matter more than they look. `splits.event_keys` keys on
`(particlePDG, eventNumber)`. Used as delivered - a constant PDG on top of an
event number that restarts - every key would collapse onto one of 5 000 values;
and without the per-lot offset, 95 000 training events would share a key with a
validation event.

### Running it

```bash
python tools/build_2class_params.py          # both roles
export SDHCAL_PARAMS_SET=2class              # redirects every stage
make pid energy
```

This writes, under `<data>/params_2class/`:

```text
pi_params.root        training sample, pion
proton_params.root    training sample, proton
val_set.root          evaluation sample, both species in one tree
manifest.json         roles, lots, energy points, events per point, derivations
```

`SDHCAL_PARAMS_SET=2class` is what points `params`, `params_merged` and `holdout`
at that directory; no stage script holds a second set of paths.
`python tools/build_2class_params.py --check` re-verifies the files, and
`--roles val` rebuilds only the evaluation sample.

`$SDHCAL_SHARDS_DIR` points at the lots themselves; it defaults to
`<repo>/data/lots/pi-proton/params`.

### Feature interpretation

`analysis/3_pid/bdt/interpret.py` produces the correlation
matrix and the permutation importances. It replaces
`legacy_importance.py`, which could not run on either sample:
it filtered on a `primaryID` branch present in no parameter file, listed two
features (`Zbary_thr2`, `Zbary_thr3`) present in neither, named the kaon lot
unconditionally, and loaded one hard-coded legacy artifact.

The feature list, the species and the time smearing are taken **from the model
artifact**, so the figures describe the model that was trained rather than a list
kept in parallel; and the importances are measured on the evaluation lot, so no
event informing them was ever seen in training.

```bash
make interpret-2class
```

**Reading the importance plot.** The script warns about near-duplicate feature
pairs, and there are two:

| Pair | r | Why |
|---|---|---|
| `tMax`, `tSpread` | +1.0000 | `tSpread = tMax - tMin`, and `tMin` is essentially constant (std 0.04 ns) |
| `Thr1`, `ratioThr23` | -0.9972 | `Thr1 + Thr2 + Thr3 = 100`, so the three fractions carry two degrees of freedom |

Permutation importance shuffles one column at a time, so when two columns carry
the same information, shuffling either leaves the other intact and both score
near zero however useful that information is. A low bar on one of these is a
reason to distrust the measurement, not evidence that the variable does not
matter.

### The chi2 calibration, fitted out of sample

`analysis/4_energy/chi2_tminuit/fit.C` reads the digitized hits and the
generator-level files of the three-class sample, and it fitted its coefficients
and drew its curves on the same events - which is why every coefficient in
`calibration/chi2_coefficients.json` is flagged `fitted_in_sample`.

The two roles make the honest version possible, and no C++ has to run:

```bash
python tools/fit_chi2_2class.py        # fit on train, evaluate on val
```

Two things make this simple. The model depends only on `N1`, `N2`, `N3`, which the
consolidated files already carry, so the hits are not needed. And it is **linear
in its nine coefficients** - expanding `alpha(N) N1 + beta(N) N2 + gamma(N) N3`
gives one column per coefficient - so the weighted least squares
`sum (Ereco - Etrue)^2 / Etrue` has a closed-form solution. `calib.fit_wls`
returns the exact global minimum, with no starting values and no convergence to
check; the TMinuit fit approximated that same minimum from a hand-chosen seed.

The coefficients go to `calibration/chi2_coefficients_2class.json`:
`calib.coefficients_path()` follows `$SDHCAL_PARAMS_SET` just as `paths.dataset`
does, so the two studies cannot end up sharing a calibration, and a stage reading
coefficients cannot end up on a different sample from the stage reading
parameters.

#### The containment cut goes to every arm or to none

The chi2 chain rejects showers reaching the last active layer, so its
coefficients are only valid under that cut. Evaluating them on the full sample
would take the method outside the domain it was fitted on, and comparing it with a
regressor trained on everything would not be like-for-like. Hence:

```bash
make energy-2class       # regressors trained with --require-contained
make chi2-2class         # calibration fitted under the same cut
make coupling-2class     # six scenarios, all with --require-contained
```

The cut keeps about half the events (100 744 of 200 000 in the evaluation lot).
Whichever choice was made is recorded in the calibration metadata and in each
run's parameter log.

### Excluding held-out events (the three-class study only)

The training stages take `--exclude-holdout`, which drops the events of the
held-out file before splitting. It exists for a sample *carved* out of the
parameter files, where the training scripts would otherwise be fitted on the very
events the coupling study evaluates on. The two-class study needs no such thing,
since its evaluation sample is a separate production. The flag is recorded in each
run's parameter log, so a run that used it is distinguishable afterwards.

## Building the Held-Out Sample

The evaluation sample is not a leftover of a training split: it is carved out
first, with a recorded seed, and the two halves are asserted disjoint.

```bash
python tools/build_holdout.py --fraction 0.15 --seed 20240601 --write-train-pool
python tools/build_holdout.py --check      # verify an existing val_set.root
```

`--check` compares the held-out sample against `train_pool_<particle>.root`, not
against the parameter file: the held-out events are carved *out of* that file, so
they are all still in it, and comparing with it would report a 100 % overlap for a
correctly built sample. The pool is the complement the training stages read.

This writes `val_set_manifest.json` next to the data, recording the fraction,
the seed, the per-particle counts and a digest of the held-out event numbers.

## What is actually on disk, and what reads it

`data/` is not versioned, so nothing above says which of its thirteen
directories are still inputs and which are sediment. Measured on 2026-10-16:
**50 GB in total, of which about 30 GB is read by nothing.** The inventory is
here because the question was asked once and the answer took an hour of
cross-checking `paths.py`, the macros and the Makefile against the filesystem.

### Read by the maintained analysis

| Directory | Size | Read by |
| --- | --- | --- |
| `lots/pi-proton/{hits,params}` | 3.5 GB + 54 MB | everything in the two-class study, through `$SDHCAL_HIT_SHARDS_DIR` and `$SDHCAL_SHARDS_DIR` |
| `params_2class/` | 53 MB | `paths.dataset(...)` under `SDHCAL_PARAMS_SET=2class`; built by `tools/build_2class_params.py` |
| `hits_v2/` | 1.3 GB | `scripts/20_pid_hits.sh` |
| `digitized/` | 2.5 GB | stage 1 and the three-class chi2 fit |
| `raw/130k_*` | 9.7 GB | `fit.C` and `plots.C`, through `sdhcal::raw()`, for the true energy |
| `params/130k_*` | ~60 MB | `paths.dataset("params")` in the three-class layout |
| `merged_primaryEnergy/200k_*` | 2.5 GB | the five stage-2 macros -- see the caveat below |
| `merged_primaryEnergy/130k_*` | 36 MB | `paths.dataset("params_merged")`, three-class layout |

### Read by nothing

| Directory | Size | Why it is still there |
| --- | --- | --- |
| `lots/pi-proton/params-superseded/` | 3.5 GB | the shards from before the `ShowerAnalyzer` corrections. Median `eccentricity3D` is 0.058 against 0.843 in the corrected lot. It reads without error and yields plausible wrong numbers; that is the trap `scripts/common.sh` documents. |
| `raw/200k_*`, `raw/100k_*_discret` | 14 GB | generator-level files of the two-class lots. The lots carry the true energy themselves, so no stage opens these. |
| `raw/100k_*_E1to100`, `raw/10k_*` | 6.3 GB | in no layout. |
| `params/200k_*`, `params/100k_*_discret` | 3.6 GB | inputs to `tools/data_prep/merge_primary_energy.py`, a step already consumed. |
| `merged_primaryEnergy/` others | ~1.1 GB | in no layout. |
| `legacy/` | 1.7 GB | no reference anywhere. `paths.legacy()` is a path translator for the internship machine, not a reader of this directory -- the name coincidence misleads. |
| `hits/` | 293 MB | superseded by `hits_v2/`, same filenames, four times smaller. |
| `split/`, `split_pi/`, `split_pr/` | 3.6 MB | thirty `partN.root` of 115 kB, no reference. |
| `data_1k/` | 13 MB | no reference. |

Nothing here is deleted. There are 301 GB free on the machine and **no second
copy of these lots exists on it**, so deleting the generator-level files would
close the door on ever re-digitising the two-class lots. Naming them as dead is
the point; removing them is a separate decision.

### Three traps this inventory exposes

**`tools/export_hits.py` writes where nothing reads.** Its default output is
`data/hits/`, while `scripts/20_pid_hits.sh` reads `data/hits_v2/`. Re-running
the exporter does not refresh what the PID study trains on.

**The stage-2 macros read pre-correction parameters.** `begin.C`, `density.C`,
`em_fraction.C`, `radius.C` and `thr3.C` open
`merged_primaryEnergy/200k_*_params_merged.root`, whose `eccentricity3D` median
is 0.058 -- the superseded generation. Checked, and the committed figures are
safe: `Begin`, `Density`, `Radius` and `Thr3` are bit-identical between the two
generations on the same shard, and `emFraction` is a simulation truth branch
that `ShowerAnalyzer` does not compute. The caveat is for the next variable
added to those macros, which may not be.

**Four source files were living under `data/`.** `clean_csv.py`,
`analyse_csv.py`, `root_to_csv.py` and `repack_tree.C` sat in `data/data_1k/`
and `data/params/`, a git-ignored tree, so they existed only on this disk. They
are now in `tools/data_prep/`, beside the other preparation utilities;
`repack_tree.C` is the provenance of `params/kaon_params_repacked.root`.
