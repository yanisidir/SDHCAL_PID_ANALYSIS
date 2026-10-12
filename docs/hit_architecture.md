# Architecture du PID au niveau des hits

R6 sépare les composants réutilisables des deux workflows pion/proton sans
modifier leurs représentations, architectures ou procédures d’entraînement.

## Représentations distinctes

Le CNN et le GNN partagent le chargement, les identifiants d’événements, le
découpage et le prétraitement de `sdhcal.hit_pid`, mais ils ne consomment pas les
mêmes lots :

- `sdhcal.hit_cnn` matérialise un tableau NumPy par événement. `collate_hits`
  conserve l’ordre des hits, complète les événements avec des zéros et fournit
  un masque booléen. `ParticleCNN` applique les convolutions `1x1` historiques,
  puis les agrégations maximum et moyenne masquées.
- `sdhcal.hit_gnn` transforme chaque événement en objet PyG. Les variables de
  nœuds viennent du même `HitPreprocessor`, et les arêtes sont produites sans
  intermédiaire par `sdhcal.graph.build_knn_graph`. `ParticleGNN` conserve les
  couches GAT, GraphConv, GraphNorm, le résidu et les deux agrégations globales.

`sdhcal.clustering` reste une étape optionnelle exécutée par les scripts avant
la création des datasets. Elle n’a pas été déplacée ou modifiée.

## Entraînement et évaluation

Les boucles propres aux lots CNN et PyG restent dans `sdhcal.hit_cnn` et
`sdhcal.hit_gnn`. Elles conservent les appels forward, l’entropie croisée,
l’ordre des lots, les probabilités proton et les métriques de
`sdhcal.hit_pid.binary_metrics`.

`sdhcal.hit_training` ne contient que les éléments réellement identiques : le
calcul historique des poids de classes, le schéma de l’historique et la règle
d’arrêt anticipé fondée sur une amélioration strictement positive de l’AUC.
Il n’importe ni PyTorch ni PyG.

`sdhcal.hit_plots` écrit les deux mêmes PNG, avec les mêmes axes, histogrammes,
normalisations, libellés et noms de fichiers. `sdhcal.hit_eval` conserve le
schéma des prédictions externes, l’ordre des événements et la conversion de la
probabilité proton en score pion.

Les scripts `analysis/3_pid/cnn/train.py` et
`analysis/3_pid/gnn/train.py` restent responsables de la CLI, des chemins, des
données, du prétraitement, des DataLoaders, des optimiseurs, du scheduler, des
checkpoints et de la publication des résultats. Ils réexportent les symboles
historiques importés depuis les nouveaux modules.

## Checkpoints et dépendances optionnelles

Les checkpoints PID existants contiennent uniquement un `state_dict`, pas une
instance Python sérialisée. Les noms et formes des paramètres restent donc le
contrat de compatibilité ; R6 ne change ni le format `best_model.pt`, ni
`preprocessor.joblib`, ni les noms de sorties.

PyTorch reste requis uniquement pour `sdhcal.hit_cnn` et PyTorch Geometric
uniquement pour `sdhcal.hit_gnn`. Importer `sdhcal` ou les workflows tabulaires
ne charge aucune de ces dépendances optionnelles.

## Limitation scientifique conservée

Le GNN peut demander une construction causale puis convertir les arêtes en graphe
non dirigé. Cette combinaison potentiellement contradictoire est conservée
exactement : `build_knn_graph`, `to_undirected`, les fenêtres temporelles, le
voisinage et les boucles propres ne sont pas modifiés. Leur validation
scientifique appartient à un chantier séparé.
