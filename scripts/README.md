# Recipes for the two-class pion/proton analysis

The commands that produce the results quoted in the report. They existed only as
shell history until now, which meant the headline numbers — the clustered PID
networks and the graph energy correction — could not be reproduced from the
repository.

Run them in order. Each is independent once the one before it has finished.

| Script | What it builds | Rough cost |
|---|---|---|
| [`10_tabular.sh`](10_tabular.sh) | BDT and MLP on shower descriptors, χ² calibration | minutes |
| [`20_pid_hits.sh`](20_pid_hits.sh) | CNN and graph network for PID, on clusters | ~2 h |
| [`30_energy_gnn.sh`](30_energy_gnn.sh) | graph network for energy: ratio target + control | ~5 h |
| [`40_benchmark.sh`](40_benchmark.sh) | comparison tables, ROOT figures, copy into `report/Fig/` | minutes |

The last one delegates to the Makefile (`predictions-2class`, `benchmark-2class`,
`report-figures`): the model lists and the figure names live there, beside the
rest of the pipeline, rather than in two places that can drift apart. `make
hits-2class` runs the two long scripts above it.

```bash
scripts/10_tabular.sh && scripts/20_pid_hits.sh && scripts/30_energy_gnn.sh && scripts/40_benchmark.sh
```

## The environment

[`common.sh`](common.sh) pins it and every script sources it. Three variables
matter, and none of them announce themselves when wrong — the run reads a
different sample and reports numbers that look entirely reasonable:

- `SDHCAL_SHARDS_DIR` — **the one to watch.** The Python library defaults to
  `new_data_params`, the lot produced *before* the `ShowerAnalyzer` corrections;
  these recipes override it with `new_data_params_v2`. Training without it set
  means training on the inverted eccentricity and the unbounded Gaisser–Hillas
  fit. This already happened once, to the MLP, and was caught only because its
  AUC came out at 0.599 instead of 0.677.
- `SDHCAL_HIT_SHARDS_DIR` — raw hit shards, defaults to `new_data`.
- `SDHCAL_PARAMS_SET=2class` — pion/proton layout and `chi2_coefficients_2class.json`.

`PYTHON` vaut `python3` par défaut et peut viser n'importe quel environnement
activé ou interpréteur explicite, sans activation automatique de Conda :

```bash
PYTHON=/chemin/vers/python scripts/10_tabular.sh
```

`10_tabular.sh` publishes the seed-42 artifacts under `models/`, after keeping
all five seeds in `output/` for the uncertainty estimate. The benchmark can
therefore run immediately after the four scripts above.

Pour éviter le choix historique par date de modification, fournir les variables
`PID_BDT_RUN_ID`, `PID_MLP_RUN_ID`, `ENERGY_BDT_PI_RUN_ID`,
`ENERGY_BDT_PROTON_RUN_ID`, `ENERGY_BDT_MIXED_RUN_ID` et
`ENERGY_BDT_RATIO_RUN_ID`. Les deux imports GNN acceptent de même
`ENERGY_GNN_RATIO_RUN_ID` et `ENERGY_GNN_RUN_ID`. Une variable absente conserve
le repli historique « fichier le plus récent » avec un avertissement ; un
identifiant fourni doit correspondre exactement à un fichier existant.

`N_TRAIN` reste surchargeable pour compatibilité. Il représente le budget
historique demandé dans les sidecars récents ; les tailles exactes sont lues du
`run.json` lorsqu'elles y sont disponibles. Voir
[la documentation de reproductibilité](../docs/reproducibility.md).

## Why the control runs are not optional

Two of the runs exist to interpret another one rather than to produce a number
of their own, and dropping them leaves the conclusions unsupported:

- `30_energy_gnn.sh` trains the same network on the plain energy target. It
  collapses to σ/E = 0.203 with the response drifting 1.21 → 0.81 across the
  range. Without it, the ratio result reads as a property of the target.
- `10_tabular.sh` trains the BDT on that same ratio target, where it changes
  nothing (53.7 % against 53.6 %). Together the two establish that the gain
  comes from the information the network reads, not from how the target is
  written.

## What these scripts do not cover

Producing the parameter shards themselves — `make params-2class` and the C++
step before it. These start from the shards as delivered.
