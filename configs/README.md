# Configurations d'expériences

Ces fichiers JSON sont des surcouches optionnelles aux valeurs définies dans le
code. Ils ne remplacent pas les variables d'environnement qui sélectionnent les
données et les répertoires.

La précédence est déterministe :

1. valeurs par défaut de la dataclass `Config` et du script ;
2. section du workflow dans le fichier JSON ;
3. options explicitement présentes sur la ligne de commande.

Un défaut `argparse` absent de la ligne de commande n'écrase donc jamais une
valeur du fichier. Les clés inconnues, les types incompatibles, une version de
schéma non supportée et un profil incompatible avec `SDHCAL_PARAMS_SET` sont
refusés.

## Profils disponibles

- `two_class.json` décrit les productions indépendantes pion/proton : `200k`
  pour l'entraînement/calibration et `discret` pour la validation. Il faut
  conserver `SDHCAL_PARAMS_SET=2class` et, comme dans `scripts/common.sh`, faire
  pointer `SDHCAL_SHARDS_DIR` vers les shards corrigés.
- `three_class.json` décrit le flux historique pion/proton/kaon basé sur les
  fichiers `130k` et son holdout. Il ne contient pas les MLP à deux classes,
  puisqu'ils lisent exclusivement les shards pion/proton.

Les profils restent volontairement minimaux : ils fixent uniquement ce qui
distingue réellement les jeux de données. Tous les hyperparamètres effectifs,
features ordonnées, graines et paramètres d'évaluation sont néanmoins fusionnés
avec les valeurs du code et écrits dans un fichier
`runs/<workflow>_<run>_resolved_config.json` à chaque nouveau run.

## Exemples réels

```bash
export SDHCAL_PARAMS_SET=2class
export SDHCAL_SHARDS_DIR="$PWD/data/lots/pi-proton/params"

python analysis/3_pid/bdt/train.py --config configs/two_class.json --seeds 5
python analysis/3_pid/mlp/train.py --config configs/two_class.json --seed 42
python analysis/4_energy/ml_bdt/train.py --config configs/two_class.json \
  --particles mixed --seeds 5
python analysis/4_energy/ml_mlp/train.py --config configs/two_class.json \
  --particles mixed
```

Dans le troisième exemple, `--particles mixed` et `--seeds 5` remplacent
explicitement les valeurs du JSON. Sans `--config`, les commandes conservent
exactement leurs anciens défauts.

## Clés supportées

Chaque section accepte les champs de sa dataclass `Config`. Les paramètres de
pilotage propres au script sont également acceptés :

- `pid_bdt` : `seeds` ;
- `pid_mlp` : `output_dir`, `comment` ;
- `energy_bdt` : `particles`, `seeds` ;
- `energy_mlp` : `particles`, `comment`.

Les noms sont ceux des valeurs effectives, par exemple `feature_cols`,
`random_state`, `smear_sigma_ns`, `correct_target_bias` et `use_smote`. Les
options CLI historiques gardent leurs noms (`--features`, `--seed`,
`--sigma-time`, `--no-bias-correction`, etc.) et sont traduites vers ces champs.

Le fichier résolu enregistre aussi le profil, la source des overrides CLI et les
valeurs effectives de `SDHCAL_DATA_DIR`, `SDHCAL_MODEL_DIR`, `SDHCAL_OUT_DIR`,
`SDHCAL_SHARDS_DIR`, `SDHCAL_HIT_SHARDS_DIR` et `SDHCAL_PARAMS_SET`. Il est
également inclus dans le champ `configuration` des nouveaux artefacts Joblib.
