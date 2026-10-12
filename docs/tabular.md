# Préparation tabulaire partagée

Le module `sdhcal.tabular` centralise uniquement les opérations qui avaient le
même comportement dans les chaînes tabulaires : initialisation des générateurs
Python/NumPy, lecture ordonnée de plusieurs espèces depuis les shards,
échantillonnage déterministe par `particlePDG`, nettoyage des lignes et
construction des indices d'entraînement/calibration stratifiés en énergie.

## Limites de responsabilité

- `rootio.py` lit et valide les arbres ROOT consolidés.
- `shards.py` découvre et lit les shards, construit leurs métadonnées et les
  identifiants d'événement globaux.
- `splits.py` fournit le partitionnement générique entraînement/validation/test
  et les primitives liées à l'identité et à la disjonction des événements.
- `tabular.py` orchestre les opérations tabulaires partagées par les MLP sans
  connaître les modèles, les cibles ou les sélections physiques.

Les conversions PDG→classe restent dans le PID MLP. La coupe
`primaryEnergy > 0`, les cibles et transformations de régression restent dans le
MLP énergie. Les BDT continuent d'utiliser `rootio.py` et `splits.py`, car leurs
fichiers, sélections, contrôles d'erreur et partitions ne sont pas équivalents à
ceux des MLP. Seule leur initialisation Python/NumPy commune est déléguée à
`tabular.py`.

## Contrats de compatibilité

`read_shard_particles` concatène les espèces dans l'ordre fourni et laisse
`shards.read_sample` déterminer l'ordre des shards et des événements. Les
colonnes de caractéristiques sont transmises sans tri ni déduplication.

`sample_per_particle` conserve l'ordre de première apparition des groupes
(`groupby(..., sort=False)`). Chaque groupe reçoit séparément le même
`random_state`, comme dans les scripts historiques, puis les résultats sont
concaténés avec un nouvel index. Cette sémantique ne doit pas être remplacée par
un générateur partagé sans validation scientifique.

`clean_event_rows` remplace `+inf` et `-inf` par `NaN`, puis supprime uniquement
les lignes incomplètes dans les colonnes explicitement requises. La
réindexation est explicite afin de préserver la différence historique : le PID
réindexe immédiatement, tandis que l'énergie réindexe après sa coupe d'énergie
positive. Le DataFrame d'entrée n'est pas modifié en place.

`stratified_energy_split_indices` reproduit la stratification historique : pour
chaque groupe, l'énergie est discrétisée selon des quantiles ; le nombre de
classes d'énergie est essayé de 10 à 1 jusqu'à ce que
`sklearn.model_selection.train_test_split` accepte la partition. Les strates,
indices, proportions, graines et règles de repli restent inchangés.

`seed_python_numpy` initialise `random` et le générateur global historique
`numpy.random` dans cet ordre. Il ne contrôle pas le parallélisme LightGBM ni les
générateurs propres au smearing, qui conservent leur gestion existante.

Le module n'impose aucune hypothèse supplémentaire sur les unités, les labels,
les observables ou la validité physique des événements. Il ne dépend ni de
LightGBM, ni de PyTorch, ni du code de tracé ou d'entraînement.
