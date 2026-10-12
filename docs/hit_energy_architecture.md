# Architecture de la reconstruction d’énergie au niveau des impacts

Le point d’entrée historique reste `analysis/4_energy/gnn/train.py`. Il résout les arguments et les chemins, construit les jeux et les chargeurs, orchestre l’entraînement, sélectionne le meilleur checkpoint et publie exactement les mêmes artefacts. Le nom du répertoire `gnn` est conservé même lorsque `--architecture deepsets` est utilisé.

## Modules

- `sdhcal.hit_energy` porte `ShowerEvent`, le chargement ROOT, les sélections, le prétraitement, la séparation entraînement/calibration, le jeu PyG paresseux et les métriques finales. Les imports `uproot` et `awkward` restent locaux au chargeur ROOT. Ce module ne connaît ni la CLI, ni les répertoires de sortie, ni Matplotlib.
- `sdhcal.hit_energy_models` contient séparément `EnergyGNN` et `EnergyDeepSets`, ainsi que la cible de régression et les boucles d’une époque d’entraînement ou d’inférence. Il dépend explicitement de Torch et PyG.
- `sdhcal.hit_energy_plots` isole les quatre fonctions de visualisation et la dépendance Matplotlib.

`sdhcal.graph` reste l’unique implémentation de la construction kNN. `EnergyGraphDataset` l’appelle avec les options causale, non dirigée et de fenêtrage existantes. En mode DeepSets, aucun arc n’est construit : l’architecture agrège directement l’ensemble des nœuds. Le GNN conserve ses convolutions GAT/GraphConv et la connectivité historique. Ces deux calculs ne sont pas assimilés.

`sdhcal.clustering` reste responsable des labels et résumés de clusters. `sdhcal.calib` fournit, sans changement, l’estimateur semi-numérique employé par la cible en ratio. `sdhcal.target` conserve la transformation logarithmique, son inverse et la correction de biais. `sdhcal.metrics` conserve les définitions du biais tronqué, de la résolution et des profils en énergie.

## Flux reproductible

Le chargeur lit les branches `x/y/z` ou `I/J/K`, `time`, `thr`, `energy` ou `primaryEnergy`, et éventuellement `eventNumber`. Il conserve l’ordre des fichiers, événements et impacts, les identifiants `{rôle}:{particule}:{fichier}:{événement}`, la sélection de confinement, le smearing temporel et la construction optionnelle des clusters. Le préprocesseur ajuste un `RobustScaler` spatial et un `QuantileTransformer` temporel sur l’échantillon déterministe d’impacts d’entraînement. Les 200k alimentent entraînement et calibration interne ; les 100k discrets restent la validation externe.

Le script conserve AdamW, le scheduler, la Smooth L1 en espace logarithmique, l’écrêtage des gradients, l’arrêt anticipé, la reconstruction directe ou par ratio au chi2, la correction de biais, les métriques, les noms de fichiers et le schéma NPZ. Les checkpoints restent des `state_dict` ; aucun nom de paramètre n’a changé.

## Compatibilité et dépendances optionnelles

Les préprocesseurs historiques ont été sérialisés par Joblib sous `__main__.ArrayHitPreprocessor`. Une sous-classe locale minimale portant ce nom est donc conservée dans le script ; son état et son comportement proviennent de `sdhcal.hit_energy.ArrayHitPreprocessor`. Aucune migration silencieuse d’artefact n’est effectuée. Les transformations de cible historiques continuent d’utiliser `sdhcal.target.LogTarget`.

Le paquet de base `sdhcal` n’importe pas ces modules automatiquement : Torch, PyG, uproot et awkward ne deviennent donc pas obligatoires pour les chaînes tabulaires. L’import du module de données hit-level requiert Torch/PyG pour son jeu de données, tandis que ROOT n’est requis qu’à l’appel du chargeur.

## Limites scientifiques inchangées

Cette extraction ne valide ni ne corrige la méthode physique. Les calculs de cœur tronqué, les coefficients de calibration, le choix DeepSets/GNN, la direction des graphes, le smearing, les cibles, les corrections et les sélections sont seulement caractérisés comme comportement historique. Une éventuelle correction scientifique ou un renommage du répertoire relève d’un chantier séparé.
