# Corrections

A review of this repository found defects ranging from files that could not be
parsed to a comparison whose conclusion could not be supported. This document
records what was wrong, why it mattered, and what was done, so that the figures
and numbers already produced can be judged on the right basis.

Nothing here was re-run: the ROOT inputs are not versioned and were not
available. **The figures currently stored under `results/` and in `report/` were
produced by the code as it was before these corrections.** Where a correction
changes a quoted quantity, that is stated below.

---

## 1. Files that could not run

Five Python files raised `SyntaxError` or `IndentationError` and so could never
have executed in the state they were committed in:

| File | Error |
|---|---|
| `Energy_reconstruction_ml/MLP/MLP_Energy_reconstruction_{kaon,pion,proton}.py` | `"file_path": config.file_path),` - one closing parenthesis too many |
| `Energy_reconstruction_ml/BDT/kaon_energy_reco_lgbm.py` | a comment whose `#` had been lost, leaving a bare `Input features` line |
| `PID/BDT/pion_proton_classifier.py` | an orphan `auc_csv` line at module indentation |

All five are fixed, and `make compile` now byte-compiles every Python file in
the repository so this class of defect cannot return unnoticed.

### The three MLP energy scripts were the same file

`MLP_Energy_reconstruction_kaon.py`, `_pion.py` and `_proton.py` were
**byte-for-byte identical** (identical md5): all three read the pion parameter
file and all three wrote into the same `results_energy_reco/` directory. Any
"kaon MLP" or "proton MLP" energy number is therefore either a pion number or
comes from a version of the script that is not in the repository.

They also shared a second, fatal defect: `load_root` was called with a `str`
where it iterated over a list of paths, so it looped over the *characters* of
the filename, found no file, and raised `RuntimeError: No valid ROOT files
found`.

They are replaced by one parameterised script,
`Energy_reconstruction_ml/MLP/mlp_energy_reco.py`, which takes the particle on
the command line.

---

## 2. Defects that affect a result

### 2.1 The chi2 method was evaluated in-sample

`energy_reconstruction_Tminuit/EnergyReco.C` fitted its nine coefficients on
every event and then drew the linearity and resolution curves on **those same
events**, while the machine-learning models were evaluated on a held-out test
set. The comparison was therefore not like-for-like, and it favoured the chi2.

The macro now splits the sample 80/20 with a recorded seed, fits on the training
part only, and produces every figure and number from the test part.

> The chi2 coefficients currently stored in
> `calibration/chi2_coefficients.json` are the ones recovered from the old code
> and are flagged `"fitted_on_train_split": false`. Re-run the macro to replace
> them before quoting a chi2-versus-ML comparison.

### 2.2 The two chains did not run on the same events

The chi2 chain rejected every shower with a hit in the last layer (`K == 47`), a
longitudinal-containment cut that preferentially removes high-energy showers.
`ShowerAnalyzer` applied no equivalent cut, so the machine-learning chain saw a
different sample and the two resolution curves were not superposable.

`ShowerAnalyzer` now writes a `touchesLastLayer` flag, and the energy pipeline
accepts `--require-contained` to apply exactly the same selection.

### 2.3 "No PID" was in fact perfect PID

`noPID_BDT.py`, `noPID_MLP.py` and `noPID_chi2.py` routed each event to a
particle-specific estimator using the **true** PDG code:

```python
mask = (label_true_int == lbl_int)        # noPID_BDT.py:167
```

That is a perfect-PID upper bound, not a PID-agnostic baseline. The study
therefore compared realistic PID against perfect PID, and the question it set
out to answer - is a particle-blind reconstruction worse? - was never asked.

`PID_RECONSTRUCTION/pid_energy_coupling.py` now offers three explicit modes:

| `--pid` | Meaning |
|---|---|
| `none` | one inclusive regressor trained on the mixture: the genuine baseline |
| `true` | routing by the true PDG: the perfect-PID upper bound |
| `bdt` / `mlp` | routing by the classifier: the realistic case |

The inclusive regressor needed for `--pid none` is trained with
`--particles mixed`, a mode that did not previously exist.

### 2.4 The held-out sample was never verified

Inference ran on `val_set.root` while the classifier and the regressors had been
trained on `130k_*_params.root` with their own internal random splits. Nothing
in the repository recorded how `val_set.root` had been built, so there was no
way to know whether the end-to-end numbers were measured partly on events the
models had already seen.

* `tools/build_holdout.py` builds a held-out sample with a recorded fraction and
  seed, writes a manifest, and asserts that the two halves are disjoint.
* `sdhcal.splits.assert_disjoint` performs the check event by event.
* `pid_energy_coupling.py` runs that check before reporting anything, and
  refuses to continue if it fails. `--skip-disjointness-check` exists but logs a
  warning saying the numbers may be training performance.

This requires an `eventNumber` branch, which `ShowerAnalyzer` now writes.

### 2.5 sigma(E)/E was not a resolution

The reported quantity was

```python
sigma_over_E = sqrt(mean(((E_pred - E_true) / E_true) ** 2))
```

the RMS of the relative residual **about zero**, i.e. `sqrt(bias^2 + sigma^2)`.
A perfectly stable estimator that is 10 % low scores the same as an unbiased one
with a 10 % spread.

The plotted resolution curve had a second problem: it used a 90 %-truncated RMS
with **no truncation correction**. For a Gaussian, the RMS of the central 90 %
is 0.7893 sigma, so the curve understated the resolution by about 21 %.

`sdhcal.metrics` now reports bias and resolution separately, corrects the
truncation, and attaches uncertainties. The historical quantity is still logged,
under the name `rms_rel_about_zero`, so older entries stay comparable.

### 2.6 Training on log(E) left an uncorrected scale error

A least-squares fit on `log E` estimates the conditional mean of the logarithm,
so exponentiating it yields the conditional *geometric* mean, which is below the
arithmetic mean by roughly `exp(-sigma^2/2)`. With a 20 % log-space spread the
reconstructed energy scale is about 2 % low - a shift that lands directly on the
linearity and relative-deviation curves.

`sdhcal.target.LogTarget` applies Duan's smearing estimator, a single
multiplicative factor measured on the **validation** set, so the energy scale is
corrected without touching the resolution. `--no-bias-correction` reproduces the
old behaviour.

One sign convention is worth keeping in mind, because it is counter-intuitive:
the mean of the relative residual `(E_pred - E_true)/E_true` moves the *other*
way, by `+sigma^2/2`, since dividing by a fluctuating `E_true` over-weights the
events where it is small. The quantity corrected here is the ratio of means,
which is what a linearity profile shows.

### 2.7 SMOTE leaked into the hyper-parameter search

In `LGBM_classifier_PID.py` the training set was oversampled *before*
`GridSearchCV`, so every validation fold contained synthetic points interpolated
from events in the corresponding training folds, and the cross-validation score
was optimistic. Resampling now happens inside an `imblearn.pipeline.Pipeline`,
hence inside each fold.

### 2.8 The time smearing was wrong in three different ways

This is the operation the whole timing study rests on.

1. **The noise was identical across variables.** `smear_column_gaussian`
   constructed a fresh generator from the same seed on every call, so `tMin`,
   `tMax` and `tSpread` all received the *same* noise vector: the smearing was
   100 % correlated instead of independent.
2. **`tMin` was smeared twice at inference.** `time_cols_clf` contained `tMin`,
   `time_cols_reg` contained it too, and both calls used `seed=42`, so `tMin`
   was shifted by exactly twice the same noise and no longer matched what the
   regressor had been trained on.
3. **One script used the global `np.random` with no seed**, so its runs were not
   reproducible.

`sdhcal.smearing` draws one generator per `(seed, column)` pair from a stable
hash of the column name, and records what it has smeared on the DataFrame so a
second application of the same column raises `SmearingAlreadyApplied`.

The width itself, sigma = 0.1 ns, was hard-coded in five places with no stated
justification. It is now a command-line argument and is recorded in the run log,
so a sensitivity scan is a loop rather than an edit.

### 2.9 "chi2" was a misnomer, and Minuit was mis-configured

The minimised quantity was `sum (Ereco - Etrue)^2 / Etrue`. The denominator is
an energy, not a variance, so it is a weighted least squares and the `chi2/ndf`
that was printed was not a goodness-of-fit test. It is now reported as `S/ndf`
with that caveat stated in the output.

Three further problems in the same fit:

* **`c0` was frozen.** It was declared with limits `[0, 1e-10]`, pinning it at a
  value far below the other scales, with the comment "comme dans ton code". All
  nine parameters are now free.
* **The steps did not match the parameter scales.** `b2` and `c2` are of order
  `1e-8` and `1e-7` but were given a step of `1e-5`, a hundred to a thousand
  times their own value, which makes the numerical gradient meaningless. Each
  step is now a fixed fraction of its starting value.
* **Convergence was never checked**, although the starting values were the output
  of a previous fit - the fit was seeded with its own answer. The Migrad return
  code and the covariance status are now checked and reported, HESSE is run, and
  the starting values are a scale-free guess.

### 2.10 The two trees were paired by row number

`EnergyReco.C` read the hit tree and the energy tree at the same entry index and
only *warned* when their entry counts differed, so a mismatch would silently
attach each shower to another event's energy. Pairing is now by `eventNumber`
when both trees carry it, and a count mismatch without that branch is fatal.

### 2.11 The resolution fit was biased at high energy

Every energy bin filled the same fixed +-20 GeV histogram of `E_reco - E_true`,
so above roughly 40 GeV the Gaussian core was truncated by the histogram range
itself. The window now scales with the energy, the fit status is checked with a
documented fallback, and the bias is reported next to the resolution instead of
being absorbed into it. Events above 100 GeV were also dropped from the figures
while the fit used them; the range now follows the data, to 130 GeV.

### 2.12 The training curve of the regression was never produced

`fit` was given a single `eval_set` but two `eval_names`:

```python
eval_set=[(X_val, y_val_z)],
eval_names=["train", "valid"],
```

LightGBM stored the validation metric under the key `"train"` and left
`"valid"` empty, so the guard `if len(train_l2) and len(valid_l2)` was always
false and no curve was ever written. Had it been written, the curve labelled
"train" would have been the validation set. Both sets are now registered.

---

## 3. Reproducibility

### 3.1 The run logs were corrupted

The logger wrote the CSV header once, at file creation, then appended rows built
from the configuration's current keys. When `val_size` was added to the
configuration, every later row gained a field the header did not have, so **all
columns after it shifted by one**. In `PID/BDT/run_parameters.csv`, run 9 - the
run behind `lgbm_model_9.joblib`, the classifier used by the coupling study -
appeared to have `random_state=0.2` and `learning_rate=42`.

Five files were affected, not one:

| File | Shifted rows |
|---|---|
| `PID/BDT/run_parameters.csv` | 24 |
| `PID/MLP/run_parameters.csv` | 1 |
| `Energy_reconstruction_ml/BDT/parameters/run_parameters_regression_pi.csv` | 23 |
| `Energy_reconstruction_ml/BDT/parameters/run_parameters_regression_proton.csv` | 19 |
| `Energy_reconstruction_ml/BDT/parameters/run_parameters_regression_kaon.csv` | 9 |

`tools/repair_run_logs.py` realigned them, inserting the missing `val_size`
column and padding the older rows. Run 9 now reads `learning_rate=0.01`,
`n_estimators=1000`, `num_leaves=63`, `random_state=42`, `class_weight=None`.

New runs are written by `sdhcal.runlog`, which widens the header instead of
shifting columns, and `make check` fails if any log is column-shifted again.

### 3.2 What was being scanned was not being logged

Neither the feature list nor the smearing width appeared in the run parameters,
although those were precisely the quantities varied between runs. The
consequence is visible in the logs: PID accuracy ranges from 0.897 (run 1) to
0.613 (runs 29-31), and `sigma_over_E` for the pion from 0.256 to 1.12, with no
record of what changed. **No figure in the repository can be attributed to a
specific configuration.**

`sdhcal.runlog.RunLogger` now records the feature list, the smearing
configuration, the git commit, the package versions and the artifact paths, in
the CSV and in one JSON file per run.

### 3.3 Everything else

* **No dependency file.** `requirements.txt`, `requirements-gnn.txt` and
  `pyproject.toml` were added. The optional torch dependencies are separated,
  since no result in the report depends on them.
* **175 hard-coded `/gridgroup/...` paths**, including one inside an `#include`
  directive. Data locations now come from `SDHCAL_DATA_DIR`,
  `SDHCAL_MODEL_DIR` and `SDHCAL_OUT_DIR`:
  * the maintained pipelines resolve their inputs through `sdhcal.paths.dataset`,
    which raises an actionable error naming the variable to set;
  * the ROOT macros use `include/SdhcalPaths.h`, which exposes the same layout
    to C++;
  * the auxiliary scripts - permutation importance, tree plotting, the
    exploratory GNN and CNN studies - keep their original literal wrapped in
    `sdhcal.paths.legacy()`, which maps the obsolete prefix onto the configured
    directory at run time. Rewriting them by hand would have meant editing code
    that is not otherwise maintained, and the literal still documents where the
    file used to live.

  One side effect was worth catching: `PID/MLP/hadron_classifier_MLP.py` read
  three per-particle directories that all held a file of the same name, so a
  blind translation would have made it read one file three times. It now names
  the three datasets explicitly.
* **No tests, no CI.** 64 tests cover the shared package, and
  `.github/workflows/ci.yml` runs them along with the Python and C++ compile
  checks on every push.
* **Single-split numbers with no uncertainty.** `--seeds n` repeats the whole
  experiment and reports mean +- standard deviation, so a difference between two
  methods can be judged against its own spread.
* **Models shipped without their own metadata.** A saved model now carries its
  feature list, its label mapping and its target transform, and inference
  refuses a bare estimator rather than trusting the call site to feed the right
  columns in the right order.

---

## 4. Structure

* **Fifteen near-identical coupling scripts** in two directories became one,
  `PID_RECONSTRUCTION/pid_energy_coupling.py`, driven by `--pid` and `--reco`.
  Five per-particle energy scripts became one with `--particles`. Duplication at
  that scale is why a defect fixed in one copy never reached the others - and
  how three files ended up identical and broken.
* **The chi2 coefficients were retyped by hand** from C++ into Python, where
  they had already drifted: the kaon `c0` read `1.22381e-14` in `EnergyReco.C`
  and `9.83127e-15` in `BDT_chi2.py`, so the two reconstructions were not
  computing the same energy. There is now one file,
  `calibration/chi2_coefficients.json`, written by the fit and read by
  `sdhcal.calib`.
* **A StandardScaler was fitted and shipped alongside every LightGBM model**,
  which needs none - the code's own comment said so. Removed from the tree
  pipelines; kept, and saved with the model, for the MLP, which genuinely needs
  it.
* **Several hundred lines of commented-out code** were removed from the energy
  pipeline. Two of the five syntax errors were created by editing inside such
  blocks.
* **`Begin = 48` is a sentinel**, not a depth: it marks "no shower start found"
  and sits one past the last layer on purpose. It is now documented both at the
  computation and on the output branch.
* **Two files differed only by case.** `compare_parameters/Density.pdf` and
  `compare_parameters/density.pdf` were both tracked with different content, so
  on macOS or Windows one silently overwrote the other and the checkout was
  never clean. `Density.pdf` is kept - it is what `compareDensity.C` writes, and
  it matches `Begin.pdf`, `Radius.pdf`, `Thr3.pdf` - and the stale lowercase
  copy was removed from the index; its content stays in the git history.
* **Three per-particle copies of the chi2 macro** (`kaon/EnergyReco.C`,
  `pion-/EnergyReco.C`, `proton/EnergyReco.C`) were removed: the top-level
  `EnergyReco.C` takes the particle as an argument. The linear-model variants
  are a genuine study variant and were kept.
* **An absolute path inside an `#include`.** `computeParams.cpp` included
  `"/gridgroup/ilc/midir/analyse/ShowerAnalyzer/ShowerAnalyzer.h"`, so the file
  could not be compiled anywhere else. It now includes `"ShowerAnalyzer.h"` and
  takes its input and output on the command line.
* **A gaussian fit whose result was discarded.** The resolution fit was written
  `hres.Fit("gaus", "QN0", ...)`; the `N` option skips storing the function, so
  the following `GetFunction("gaus")` would have returned a null pointer. The
  option is now `"Q0"`.

---

## 4b. Layout

The directories had grown four different naming conventions, two of which were
near-homonyms for different things: `PID/` did the identification while
`PID_RECONSTRUCTION/` did the coupling, and `Energy_reconstruction_ml/` sat next
to `energy_reconstruction_Tminuit/`. Nothing said in which order the stages ran.

The repository now follows the analysis chain:

```text
src/sdhcal/            shared Python library, installed with `pip install -e .`
cpp/{include,src}/     shared C++ used by the ROOT stages
analysis/
  1_shower_features/   digitized hits -> shower variables
  2_variable_study/    how the variables differ between species
  3_pid/               bdt/ mlp/ randomforest/ gnn/ cnn/
  4_energy/            ml_bdt/ ml_mlp/ chi2_tminuit/
  5_pid_energy/        does identifying first help?
```

Every move was a `git mv`, so each file keeps its history. Two consequences
worth noting:

* **The `src/` layout removed 17 `sys.path.insert` bootstraps.** The package is
  installed (`pip install -e .`, done by `make venv`), and the Makefile also
  exports `PYTHONPATH=src` so a fresh checkout works before any install.
* **`repo_root()` had to learn to find the checkout.** With the package at
  `src/sdhcal/`, walking up two levels lands in `src/`, which silently made
  `calib.coefficients_path()` look for `calibration/` in the wrong place. It now
  searches upwards for a marker file, and a test asserts it.

The Makefile gained one target per stage (`make features`, `make pid`,
`make energy`, `make coupling`), and each stage carries a README saying what it
reads, what it writes and what to be careful about.

---

## 5. What still needs the data

These cannot be settled without re-running:

1. **Re-run the chi2 fit** so its coefficients come from a training split, then
   `make calibration`. Until then the chi2-versus-ML comparison is not
   like-for-like.
2. **Rebuild the held-out sample** with `tools/build_holdout.py`, or verify the
   existing one with `--check`, and re-measure the coupling study. If the check
   fails, the published end-to-end numbers are optimistic by an unknown amount.
3. **Re-measure bias and resolution** with the corrected estimators. The
   resolution figures in `report/` understate sigma by about 21 % from the
   truncation alone, in a direction that flatters the detector.
4. **Re-run the per-particle MLP energy reconstruction**, which never ran as
   advertised.
5. **Quote the PID-agnostic baseline** (`--pid none`) next to perfect PID
   (`--pid true`), so the coupling study answers the question it poses.
6. **Scan the timing resolution** rather than fixing sigma = 0.1 ns, now that
   the smearing is correct and independent per variable.
