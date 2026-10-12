# Architecture des BDT tabulaires

R5 sépare la logique de modèle LightGBM de l’orchestration des analyses, sans
modifier les algorithmes ni les interfaces en ligne de commande.

## Modules stables

- `sdhcal.pid_bdt` contient `Config`, `build_classifier`,
  `search_hyperparameters`, `train` et `evaluate` pour l’identification de
  particules binaire ou multiclasse.
- `sdhcal.energy_bdt` contient `Config`, `mode_particles`, `build_regressor`,
  `train` et `evaluate` pour la reconstruction d’énergie.

Les scripts historiques
`analysis/3_pid/bdt/train.py` et `analysis/4_energy/ml_bdt/train.py` réexportent
ces symboles. Ils restent responsables de la CLI, de la résolution de la
configuration, des entrées ROOT, des sélections d’événements, des découpages,
des graphiques, de la journalisation et de l’écriture des artefacts. Le script
d’interprétation PID continue de charger les modèles par l’interface stable
`sdhcal.artifacts`.

`sdhcal.tabular` reste la couche commune pour l’initialisation des graines et le
traitement tabulaire partagé. `sdhcal.config` résout la configuration dans les
scripts avant la construction des `Config` spécialisées. `sdhcal.artifacts`
reste l’unique interface de publication et de lecture des bundles ; les nouveaux
modules de modèle ne dépendent ni de la CLI ni de la sérialisation.

## Contrats scientifiques préservés

Pour le PID, l’ordre de `Config.particles` fixe toujours les labels entiers et
l’ordre des colonnes de `Config.feature_cols` est inchangé. SMOTE reste placé à
l’intérieur de chaque pli pendant la recherche d’hyperparamètres et il est
appliqué uniquement au sous-échantillon d’entraînement pendant l’ajustement
final. Les objectifs LightGBM (`binary` ou `multiclass`), les métriques, les
callbacks, les espaces de recherche, les graines et l’ordre des probabilités ne
changent pas.

Pour l’énergie, les transformations de cible, la correction de biais sur la
validation, le mode ratio, les métriques et les espaces de recherche sont
inchangés. La logique de calibration et les graphiques restent dans le script
d’analyse.

## Limitation historique explicitement conservée

Les deux configurations fixent `subsample=0.8`, mais ne fixent pas
`subsample_freq`. LightGBM conserve donc sa valeur par défaut `0` et le
sous-échantillonnage des lignes est inactif. R5 documente et teste ce comportement
sans le corriger, car l’activer modifierait l’entraînement et les résultats
numériques.

## Reproductibilité

Les valeurs `random_state`, `n_jobs=-1`, les callbacks d’arrêt anticipé et les
jeux d’évaluation restent ceux des scripts historiques. Les tests de régression
utilisent de petits tableaux synthétiques et imposent `n_jobs=1` uniquement dans
les tests de répétabilité ; le comportement de production n’est pas modifié.
Les artefacts Joblib historiques restent chargés par `sdhcal.artifacts` et leur
schéma n’est pas migré par R5.
