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
between species                classifiers       (LightGBM, MLP, chi2/TMinuit)
                                       |                |
                                       +-------+--------+
                                               |
                                               v
                                       5_pid_energy/
                                       does identifying first help?
```

Stages 3 and 4 are independent branches off stage 1; stage 5 joins them. The
numbering gives a reading order, not a strict dependency chain.

| Stage | What it answers | Language |
|---|---|---|
| [1_shower_features](1_shower_features/) | What does each shower look like, as numbers? | C++/ROOT |
| [2_variable_study](2_variable_study/) | Which variables separate the species at all? | ROOT macros |
| [3_pid](3_pid/) | Which hadron produced this shower? | Python |
| [4_energy](4_energy/) | What was its energy? | Python + C++/ROOT |
| [5_pid_energy](5_pid_energy/) | Does identifying it first improve the energy? | Python |

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

Figures worth keeping are copied by hand into [`../results/`](../results/) and
committed; everything else stays in `output/`.

**Every run is identified.** The JSON record holds the feature list, the
smearing width, the seeds, the git commit and the package versions, so a figure
can be traced back to the configuration that produced it. This was not true of
the original code, and it is why the figures already in `results/` cannot be
attributed to an exact run; the limitation is stated in the root README.

**Shared code** lives in [`../src/sdhcal/`](../src/sdhcal/) (Python) and
[`../cpp/`](../cpp/) (C++): paths, time smearing, bias and resolution
estimators, splits, run logging, chi2 coefficients. Nothing analysis-specific
belongs there, and nothing duplicated across stages belongs anywhere else.
