# Exécution, environnements et releases reproductibles

Ce document décrit la couche R10. Elle organise l'exécution et la traçabilité ;
elle ne change aucune formule, sélection, architecture de modèle, graine,
hyperparamètre, métrique ni convention de sortie scientifique.

## Environnements

Le Makefile et les scripts utilisent `python3` par défaut et acceptent un
interpréteur explicite, sans activer Conda ni installer de paquet pendant un
workflow :

```bash
make test PYTHON=/chemin/vers/python
PYTHON=/chemin/vers/python scripts/10_tabular.sh
make cpp ROOT_CONFIG=/chemin/vers/root-config
make variables ROOT=/chemin/vers/root
```

Les chemins contenant des espaces sont acceptés pour `PYTHON`, `ROOT_CONFIG`,
`ROOT`, les jeux de données et les destinations manipulées par les nouveaux
outils. Une dépendance absente provoque une erreur au point d'utilisation. Le
Makefile conserve le mode local historique : sans ROOT, `make cpp` et
`make cpp-test` signalent l'indisponibilité et passent. La CI exige ROOT avec
`REQUIRE_ROOT=1`, ce qui transforme cette situation en échec.

L'environnement tabulaire reste défini par `requirements.txt`. Les dépendances
Torch/PyG restent isolées dans `requirements-gnn.txt`. Les fichiers
`constraints/reference-py311-*.txt` figent uniquement les dépendances directes
de l'environnement Linux/Python 3.11 effectivement validé pendant R10 ; ce ne
sont pas des lockfiles transitifs ni une promesse de reproductibilité sur une
autre plateforme. Leur usage est optionnel :

```bash
python3.11 -m pip install -r requirements.txt \
  -c constraints/reference-py311-core.txt
python3.11 -m pip install -r requirements-gnn.txt \
  -c constraints/reference-py311-core.txt \
  -c constraints/reference-py311-gnn.txt
```

ROOT n'est pas installé par ce projet. La CI compile avec l'image déclarée dans
`.github/workflows/ci.yml`; un environnement local doit fournir des commandes
`root-config` et `root` cohérentes.

## Workflows et sélection d'artefact

Les cibles Make et les quatre recettes `scripts/*.sh` conservent leur ordre,
leurs budgets, leurs graines et leurs chemins de sortie. `scripts/common.sh`
expose maintenant deux modes de publication :

- `publish_selected` construit un nom exact à partir d'un `run_id` et échoue si
  ce fichier n'existe pas ;
- en l'absence d'identifiant, le repli `publish_latest` conserve la sélection
  historique par date de modification et émet un avertissement.

Pour rendre `scripts/10_tabular.sh` déterministe, fournir les identifiants connus :

```bash
PID_BDT_RUN_ID=12 PID_MLP_RUN_ID=9 \
ENERGY_BDT_PI_RUN_ID=7 ENERGY_BDT_PROTON_RUN_ID=7 \
ENERGY_BDT_MIXED_RUN_ID=8 ENERGY_BDT_RATIO_RUN_ID=9 \
scripts/10_tabular.sh
```

Les identifiants pi et proton sont indépendants : il faut enregistrer ceux
affichés par chaque logger, même s'ils sont souvent égaux. Pour les prédictions
GNN d'énergie, `ENERGY_GNN_RATIO_RUN_ID` et `ENERGY_GNN_RUN_ID` remplacent le
glob par un fichier `..._run<id>.npz` exact. Sans ces variables, le glob et son
comportement historique restent actifs.

Une publication ponctuelle peut être réalisée sans exécuter l'entraînement :

```bash
python tools/publish_artifact.py \
  --directory output/pid_lgbm/models \
  --template 'lgbm_pid_{run_id}_seed42.joblib' --run-id 12 \
  --destination models/pid_bdt.joblib
```

L'ancien choix par date doit désormais être demandé explicitement avec
`--latest --pattern`. Aucun fichier historique n'est renommé.

## Manifeste d'une release scientifique

`sdhcal.manifest` complète les configurations résolues R2, les bundles R1 et
les enregistrements `RunLogger`; il ne les duplique pas. Un manifeste version 1
contient :

- `release_id`, `experiment_type` et la commande exécutée ;
- l'empreinte logicielle existante (`git_commit`, `git_dirty`, Python,
  plateforme et versions disponibles) ;
- une référence hachée vers la configuration résolue, si elle existe ;
- les graines et neuf comptes d'événements à sémantique distincte ;
- des références groupées vers datasets, calibrations, modèles, prédictions et
  autres sorties.

Tous les chemins sont relatifs au dossier du manifeste (`path_base=manifest`).
Les fichiers générés et petites calibrations reçoivent un SHA-256 ; cet hash
sert aussi d'identifiant stable `sha256:<digest>` aux artefacts. Les datasets ne
sont pas hachés par défaut afin de ne pas lire implicitement de gros ROOT ;
`--hash-datasets` est une décision explicite.

Le schéma refuse les références manquantes lors de l'écriture. La lecture peut
également vérifier les fichiers et hashes. Les métadonnées absentes restent
`null`, elles ne sont jamais reconstruites à partir d'un budget approximatif.
`--run-record` reprend les comptes exacts déjà écrits par un trainer et référence
ce record par son hash ; une valeur CLI contradictoire est refusée.

Exemple après un run existant :

```bash
python tools/create_release.py \
  --release-id pid-bdt-run12 \
  --experiment-type pid-bdt \
  --command 'python analysis/3_pid/bdt/train.py --seeds 5' \
  --config output/pid_lgbm/runs/pid_12_resolved_config.json \
  --run-record output/pid_lgbm/runs/pid_12.json \
  --model pid=output/pid_lgbm/models/lgbm_pid_12_seed42.joblib \
  --seed 42 --seed 43 --seed 44 --seed 45 --seed 46
```

La destination par défaut est `results/<release-id>/manifest.json`. Le dossier
n'est créé qu'à la demande. Les gros artefacts restent dans `output/` ou
`models/` et sont référencés, pas copiés. Les chemins historiques restent donc
intacts. Une future release peut ajouter de petits sous-dossiers `tables/` ou
`figures/` seulement lorsqu'elle possède réellement ces fichiers.

## Comptes d'événements

Les champs suivants ne sont pas interchangeables : budget demandé, entrée,
après sélection, entraînement optimiseur, validation, test interne, calibration,
évaluation externe et intersection du benchmark. Le manifeste conserve chacun
séparément et écrit `null` lorsque la valeur exacte est indisponible.

Les CNN/GNN PID enregistrent déjà les tailles exactes dans `run.json`. Le GNN
d'énergie enregistre maintenant les tailles post-sélection exactes via
`RunLogger.log_event_counts`, sans modifier les indices. Lors de l'import NPZ,
ces valeurs sont reprises si le `run.json` correspondant existe. Le paramètre
historique `--n-train` et la clé `n_train` restent inchangés pour les anciens
tableaux ; dans la nouvelle section `event_counts`, ils sont explicitement
classés comme `requested_training_budget`. Ils ne sont jamais présentés comme
un comptage optimiseur observé sans enregistrement du trainer.

## CI, lint et contrôles locaux

La CI sépare : Python cœur sur 3.10/3.12, environnement Torch/PyG optionnel,
Ruff, ShellCheck et C++/ROOT. Aucun job ne télécharge les jeux scientifiques ni
ne lance un entraînement complet. Le job ROOT exécute la compilation, la
caractérisation C++ et le test d'intégration R9, y compris son skip documenté.

Le périmètre Ruff maintenu est : `src/`, `tools/`, `tests/`, les scripts BDT/MLP
actifs et `analysis/5_pid_energy/`. `analysis/archive/`, le rapport et les deux
points d'entrée CNN/GNN restent exclus d'une réécriture stylistique globale ;
leurs modules partagés sont néanmoins couverts via `src/` et les tests Torch.

Contrôles légers locaux :

```bash
make compile
make test
make lint
shellcheck scripts/*.sh analysis/1_shower_features/parallel.sh
make cpp cpp-test ROOT_CONFIG=/chemin/vers/root-config
git diff --check
```

## Limites connues

- La sélection par mtime reste disponible pour compatibilité et n'est pas
  reproductible en présence de runs concurrents ; les nouvelles releases
  doivent fournir un `run_id`.
- Les anciens runs ne possèdent pas tous une configuration résolue ni des
  comptes exacts. R10 les laisse absents au lieu de les inventer.
- Le manifeste référence les artefacts ; il ne garantit pas la disponibilité
  future d'un chemin externe si le fichier n'est pas archivé avec la release.
- Le round-trip ROOT 6.40.04/uproot 5.7.7 de R9 reste ignoré après timeout. La
  compilation et la régression numérique C++ restent valides, mais cette
  compatibilité Python n'est pas déclarée résolue.
- Les quatre `xfail` R9 sur collision `eventNumber`, branche absente, type
  incorrect et ré-enrichissement restent des défauts connus. R10 ne modifie ni
  schéma ROOT ni algorithme de génération.
