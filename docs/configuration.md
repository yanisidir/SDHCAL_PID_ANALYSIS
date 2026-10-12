# Configuration des expériences

## Sources existantes et responsabilités

Avant R2, chaque entraîneur construisait directement sa dataclass `Config` à
partir des défauts `argparse`. Les emplacements étaient déjà résolus au moment de
l'exécution par `sdhcal.paths`. Les workflows d'ensemble ajoutaient leurs
overrides sans modifier Python : variables Make (`PID_ARGS`, `ENERGY_ARGS`,
`MLP_SEEDS`) et tableaux Bash (`COMMON`, `SAMPLING`).

R2 conserve cette séparation :

| Source | Responsabilité | Priorité |
|---|---|---:|
| dataclass et défauts du parseur | comportement historique sans configuration | 1 |
| `configs/*.json` | surcouche versionnée d'un profil scientifique | 2 |
| option CLI explicitement écrite | override d'une invocation | 3 |
| `SDHCAL_*` | chemins, lot actif et répertoires de sortie | indépendante |

Les scripts Shell et le Makefile conservent leurs commandes, budgets, ordre des
arguments et conventions de sortie. Ils restent donc compatibles et leurs
valeurs explicites continuent à avoir la priorité si `--config` est ajouté à
leurs variables d'arguments.

## Format JSON version 1

Un document contient exactement `schema_version`, `profile`, `dataset` et
`workflows`. `dataset.params_set` est contrôlé contre la valeur effective de
`SDHCAL_PARAMS_SET` ; un profil deux-classes ne peut ainsi pas lire
silencieusement la disposition trois-classes. Les champs `training_sample` et
`validation_sample` documentent le rôle des productions sans redéfinir leurs
chemins.

Chaque section de `workflows` est une surcouche partielle. Les clés autorisées
sont les champs de la dataclass correspondante, plus les quelques paramètres de
pilotage documentés dans [configs/README.md](../configs/README.md). Les clés
inconnues et types incompatibles sont des erreurs. Les listes restent ordonnées
et `null` n'est accepté que pour les valeurs réellement optionnelles.

Les workflows configurables en R2 sont les quatre entraîneurs tabulaires
maintenus : `pid_bdt`, `pid_mlp`, `energy_bdt` et `energy_mlp`. Les configurations
Torch existantes continuent d'être enregistrées par leurs scripts ; leurs
checkpoints et schémas NPZ ne sont pas migrés pendant cette étape.

## Configuration résolue

Chaque run tabulaire écrit sous son dossier `runs/` un JSON contenant :

- la totalité des valeurs effectives, y compris features, hyperparamètres,
  graines et options d'évaluation ;
- le profil et le fichier JSON éventuels ;
- la liste des paramètres explicitement remplacés en CLI ;
- les valeurs effectives des variables `SDHCAL_*` et l'indication
  `environment` ou `code_default` pour leur origine ;
- la description du dataset portée par le profil.

Ce même document alimente `configuration` dans les nouveaux artefacts Joblib du
schéma R1. Un artefact historique reste lisible et aucune information absente
n'est reconstruite.

Pour reproduire une invocation, restaurer d'abord les variables `SDHCAL_*`
enregistrées, puis réutiliser le profil indiqué et les overrides CLI. Les jeux de
données et calibrations externes restent nécessaires : le JSON résolu ne les
copie pas et ne garantit donc pas à lui seul une reproduction complète.
