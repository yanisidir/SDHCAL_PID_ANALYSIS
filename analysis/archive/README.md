# Analysis archive

This directory contains material that is not part of the maintained analysis
pipeline. Nothing here is called by the Makefile.

| Directory | Contents |
|---|---|
| `code/` | Superseded prototypes, hard-coded diagnostics and old three-class variants. |
| `results/` | Historical plots, tables, model dumps and logs produced beside the old scripts. |
| `build/` | Rebuildable local binaries; ignored by Git. |
| `cache/` | Python bytecode caches; ignored by Git. |

The relative stage/method layout is preserved below each category so files with
the same name do not collide and their origin remains identifiable. Historical
results are retained for traceability only: they are not evidence from the
current pion/proton pipeline and should not be used for new performance claims.

Current runs write to `$SDHCAL_OUT_DIR`; selected publication figures belong in
the repository-level `results/` directory, not beside executable scripts.
