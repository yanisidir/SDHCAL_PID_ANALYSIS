# The analysis chain

One directory per stage, numbered in the order the data flows through them. Each
stage has its own README saying what it reads, what it writes, and how to run it.

```text
digitized hits (.root)
        |
        v
1_shower_features/     ROOT/C++  -> one row of shower variables per event
        |
        +--------------------------------+
        |                                |
        v                                v
2_variable_study/              3_pid/            4_energy/
how the variables differ       pi- / p / K0      E from the shower variables
between species                classifiers       or the hits (LightGBM, MLP,
                                                 DeepSets, chi2/TMinuit)
                                       |                |
                                       +-------+--------+
                                               |
                                               v
                                       5_pid_energy/
                                       does identifying first help?
                                       (measured: no)
```

Stages 3 and 4 are independent branches off stage 1; stage 5 joins them. The
numbering gives a reading order, not a strict dependency chain.

| Stage | What it answers | Language |
|---|---|---|
| [1_shower_features](1_shower_features/) | What does each shower look like, as numbers? | C++/ROOT |
| [2_variable_study](2_variable_study/) | Which variables separate the species at all? | ROOT macros |
| [3_pid](3_pid/) | Which hadron produced this shower? | Python |
| [4_energy](4_energy/) | What was its energy? | Python + C++/ROOT |
| [5_pid_energy](5_pid_energy/) | Does identifying it first improve the energy? (no) | Python |

## Running a stage

From the repository root:

```bash
make features     # 1
make variables    # 2
make pid          # 3
make energy       # 4  (both the ML and the chi2 reconstructions)
make coupling     # 5  (the three --pid modes, which belong in one table)
```

`make help` lists every target. Each target just calls the stage's own script,
so you can always run that script directly with your own arguments.

## File names

Source files use short role-based names consistently across methods:

- `train.py` trains a model;
- `evaluate.py`, `predict.py`, `interpret.py` and `importance.py` perform the
  corresponding post-training task;
- `fit.C`, `linear.C`, `plots.C` and `validate.C` are ROOT macros;
- `extract.cpp` and `extract_parallel.cpp` build the shower variables.

Historical plots, tables and logs keep their original run-specific names so
their provenance is not lost.

Superseded code and historical run outputs have been collected under
[`archive/`](archive/). The archive is not called by the Makefile and is kept
only for provenance; maintained stages must not import from it.

## Conventions shared by every stage

**Inputs** come from `$SDHCAL_DATA_DIR`, never from a path written into a
script. A missing file raises an error naming the variable to set.

**Outputs** go to `$SDHCAL_OUT_DIR/<study>/` (default `output/`), which is not
versioned, with the same subdirectories everywhere:

```text
output/<study>/
├── plots/            figures
├── tables/           CSV summaries
├── arrays/           .npz of y_true / y_pred, for later plotting
├── models/           trained models, each carrying its own feature list
├── runs/<id>.json    the full configuration of that run
├── *_parameters.csv  one row per run
└── *_performances.csv
```

Figures that end up in the report are copied by `make report-figures`, which
renames the 22 benchmark plots to what the LaTeX expects and writes them to
[`../report/Fig/`](../report/Fig/). That is the maintained path out of
`output/`, and it is scripted so no figure in the report is a file somebody
copied by hand.

[`../results/`](../results/) is **not** that path. It is the frozen archive of
the internship figures -- 88 files that predate run logging, which no current
run reproduces: not one of its filenames exists anywhere under `output/`. It is
kept because the root README displays some of them and because they are the
record of what was produced, not as a place to promote new figures to.

**Every run is identified.** The JSON record holds the feature list, the
smearing width, the seeds, the git commit and the package versions, so a figure
can be traced back to the configuration that produced it. This was not true of
the original code, and it is why the figures in `results/` cannot be attributed
to an exact run; the limitation is stated in the root README.

**Shared code** lives in [`../src/sdhcal/`](../src/sdhcal/) (Python) and
[`../cpp/`](../cpp/) (C++): paths, time smearing, bias and resolution
estimators, splits, run logging, chi2 coefficients. Nothing analysis-specific
belongs there, and nothing duplicated across stages belongs anywhere else.
