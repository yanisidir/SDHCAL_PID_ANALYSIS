# Artefacts de modèles et provenance

Les modèles tabulaires PID et de reconstruction d'énergie utilisent
`sdhcal.artifacts`. Les checkpoints Torch et les fichiers de prédictions NPZ
gardent leurs formats actuels et ne relèvent pas de ce contrat.

## Versions du schéma

- **Version 0 (legacy)** : dictionnaire Joblib sans `schema_version`. Il doit au
  minimum contenir `model` et la liste ordonnée `features`. Le chargeur le rend
  sans ajouter ni inventer de métadonnées.
- **Version 1** : conserve toutes les clés historiques utilisées à l'inférence
  et ajoute `schema_version=1`, `kind` (`pid` ou `energy`) et `model_family`.
  Les sections optionnelles sont `preprocessing`, `training_counts`,
  `configuration`, `dependencies` et `provenance`. Elles ne sont écrites que
  lorsque le script d'entraînement peut les renseigner correctement.

`load_bundle(path, kind=...)` charge et valide les deux versions.
`save_bundle(path, payload, kind=..., model_family=...)` écrit une version 1
sans modifier le dictionnaire fourni. `artifact_metadata(payload)` produit une
vue JSON-compatible des métadonnées, sans sérialiser l'estimateur.

Joblib repose sur pickle : seuls des artefacts provenant d'une source de
confiance doivent être chargés. Les objets Python sérialisés gardent par ailleurs
une dépendance à leur chemin de module historique ; R1 ne déplace aucune de ces
classes.

## Provenance

`sdhcal.provenance.environment_fingerprint()` enregistre l'horodatage UTC, le
commit et l'état Git, Python, la plateforme, les versions disponibles des
principales dépendances et les variables d'environnement `SDHCAL_*` pertinentes.
L'ancienne interface `sdhcal.runlog.environment_fingerprint` reste disponible.

Une référence de fichier contient son chemin et sa taille. Son SHA-256 n'est
calculé que si `file_reference(..., include_sha256=True)` le demande
explicitement : les gros fichiers ROOT ne sont jamais hachés automatiquement.
Les nouveaux régresseurs basés sur `chi2_pi` enregistrent ainsi le chemin et le
hash du petit fichier de calibration utilisé pendant l'entraînement.

Cette dépendance reste chargée dynamiquement à l'inférence, comme auparavant.
Le chargeur n'impose pas encore que son hash corresponde à celui enregistré ;
le fichier de calibration et les jeux de données externes restent donc requis
pour reproduire entièrement un résultat.

## Usage pour les futurs modules

Les nouveaux bundles tabulaires doivent passer par `save_bundle` et renseigner
uniquement les métadonnées connues. Les consommateurs doivent utiliser
`load_bundle`, préciser `kind` lorsqu'il est connu et continuer à lire les clés
historiques (`scaler`, `target`, `smearing`, `baseline`, etc.) nécessaires à
l'inférence. Les schémas NPZ et les checkpoints Torch ne doivent pas être
transformés implicitement en bundles Joblib.

## Sélection et releases

La sélection d'un fichier parmi plusieurs runs relève de
`sdhcal.artifact_selection`, pas du schéma Joblib. Une sélection explicite exige
un modèle de nom contenant `{run_id}` et vérifie l'existence du fichier exact.
La sélection historique par mtime reste disponible comme repli annoncé.

`sdhcal.manifest` référence les artefacts R1, la configuration résolue R2 et les
sorties sans recopier leur contenu. Les artefacts générés reçoivent un identifiant
stable fondé sur SHA-256. Le format, les commandes et la politique de chemins
sont documentés dans [reproducibility.md](reproducibility.md).
