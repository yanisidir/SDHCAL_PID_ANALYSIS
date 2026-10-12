# Architecture des MLP tabulaires

R4 sépare l'implémentation scientifique des MLP de leur orchestration en ligne
de commande, sans réunir la classification PID et la régression d'énergie dans
un entraîneur générique.

## Modules scientifiques

`sdhcal.pid_mlp` contient la configuration du MLP pion/proton, sa construction
scikit-learn, sa grille d'hyperparamètres, la boucle `partial_fit`, l'arrêt
anticipé, le calcul des probabilités et les métriques de classification. Le
module reste explicitement binaire : les classes transmises au premier appel
`partial_fit` sont `[0, 1]` et correspondent à pion puis proton.

`sdhcal.energy_mlp` contient la configuration de la régression, la sélection des
espèces par mode, la construction du modèle, la boucle d'entraînement, la
transformation de cible, la correction de back-transform, les bins de validation
et l'évaluation en énergie. Les transformations sérialisées restent définies
dans `sdhcal.target` et les biais/résolutions restent calculés par
`sdhcal.metrics`.

Les deux modules délèguent à `sdhcal.tabular` la lecture ordonnée des shards,
l'échantillonnage, le nettoyage, les graines Python/NumPy et les indices de
calibration. Les labels PID, la coupe `primaryEnergy > 0` et les traitements de
cible restent séparés.

## Orchestration et artefacts

Les scripts `analysis/3_pid/mlp/train.py` et
`analysis/4_energy/ml_mlp/train.py` conservent les interfaces CLI, la résolution
de configuration par `sdhcal.config`, le scaling dans son ordre historique, les
tracés, les chemins de sortie, le `RunLogger` et la publication par
`sdhcal.artifacts`. Ils réexportent les anciens noms `Config`, `base_model`,
`fit_model`, `evaluate` et les helpers de données afin de préserver les imports
historiques.

Les scripts PID `evaluate.py` et `importance.py` importent désormais
`sdhcal.pid_mlp` au lieu du fichier voisin `train.py`. Ils continuent de lire les
mêmes clés d'artefact et d'appliquer le scaler, les caractéristiques et le
smearing enregistrés.

## Compatibilité et limites

Les artefacts PID MLP existants contiennent des objets scikit-learn et des types
natifs ; aucun objet `Config` du script n'y a été observé. Les classes de cible
énergie conservent leur chemin `sdhcal.target`. Le schéma Joblib et les fichiers
existants ne sont ni modifiés ni migrés.

La grille de recherche continue de recevoir les matrices déjà standardisées par
le script. Le risque connu de fuite de prétraitement en validation croisée est
donc volontairement conservé : sa correction serait un changement de méthode
scientifique hors R4. La reproductibilité complète de LightGBM, les métriques,
les architectures, l'ordre des appels aléatoires et les procédures de sélection
ne sont pas modifiés.
