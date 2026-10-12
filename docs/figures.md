# Figures du rapport : d'où elles viennent

Chaque figure tracée à partir de données est produite par une macro ROOT dans un
style unique — en-tête `SDHCAL Simulation`, graduations sur les quatre côtés,
légendes transparentes. `make benchmark-2class` les régénère toutes et
`make report-figures` les copie dans `report/Fig/` sous les noms que le LaTeX
attend.

## Régénérables

| Figure | Producteur |
| --- | --- |
| `bench_pid_rejection`, `bench_pid_efficiency` | `plots.C("pid")` |
| `bench_energy_resolution`, `bench_energy_linearity` | `plots.C("energy")` |
| `bench_species_*` (6) | `plots.C("coupling")`, vue par espèce |
| `bench_coupling_*` (5) | `plots.C("coupling")`, vue par source d'étiquette |
| `bench_confusion_matrix`, `bench_correlation_matrix` | `interpret_plots.C` |

Les courbes de couplage passent d'abord par `tools/coupling_curves.py`, qui
recalcule réponse et résolution par tranche d'énergie à partir des tables par
événement écrites par `analysis/5_pid_energy/study.py`. Les figures et les
nombres du texte viennent donc du même run.

## Non régénérables en l'état

Trois figures restent dans leur style d'origine parce que leurs entrées ne sont
plus dans `output/`. Les redessiner demanderait de relancer l'étude à trois
espèces, ce qui déplacerait les nombres publiés — c'est un changement
scientifique, pas un changement de style.

| Figure | Ce qu'elle montre | Ce qui manque |
| --- | --- | --- |
| `permutation_importances.png` | importances par **gain** LightGBM, ancienne liste de variables | les gains du run 3 classes ; le nom du fichier est trompeur, ce ne sont pas des importances par permutation |
| `prob_hists_9.png` | histogrammes de probabilité à trois classes | `arrays/` ne contient plus que des runs à deux classes |
| `training_curve_9.png` | courbe d'entraînement du run 9 | idem |

`tools/coupling_curves.py` et `analysis/6_benchmark/interpret_plots.C` existent ;
il ne manque que les données d'entrée.

## Figures qui ne sont pas des tracés

`sdhcal_prototype.jpeg` et `img/ms.png` sont des photographies et un schéma ; le
style de tracé ne s'y applique pas.
