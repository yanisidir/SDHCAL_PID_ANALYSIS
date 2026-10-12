# Architecture des prédictions et du couplage PID–énergie

R8 sépare les calculs réutilisables des quatre points d’entrée historiques. Les commandes, options, chemins et noms de sorties de `tools/predict.py`, `tools/import_predictions.py`, `tools/compare_models.py` et `analysis/5_pid_energy/study.py` restent inchangés.

## Services

- `sdhcal.prediction` construit les matrices dans l’ordre exact déclaré par l’artefact, applique son scaler et son smearing, calcule les probabilités PID, inverse les cibles énergie et applique les calibrations chi2 historiques. Les coefficients restent chargés par `sdhcal.calib` ; ils ne sont ni copiés ni incorporés aux modèles.
- `sdhcal.prediction_io` traduit les deux formes d’identifiants hit-level, convertit les NPZ PID/énergie vers les tables de benchmark, lit et écrit les CSV et leurs sidecars JSON. Les schémas restent ceux de `sdhcal.benchmark.PID_COLUMNS` et `ENERGY_COLUMNS`.
- `sdhcal.coupling` porte la configuration et les scénarios `none`, `true`, `bdt` et `mlp`, le routage, les estimateurs par espèce ou inclusif, les métriques et la table événementielle. Le chargement ROOT, le contrôle de disjonction, les tracés et la journalisation restent dans le script d’orchestration.

`sdhcal.artifacts.load_bundle` reste l’unique chargeur des bundles Joblib de schémas 0 et 1. `sdhcal.benchmark` conserve la validation, l’intersection sur `(particlePDG, eventNumber)`, le tri et les métriques de comparaison. `sdhcal.metrics` conserve les définitions de biais et résolution ; `sdhcal.target` reste responsable des transformations inverses présentes dans les bundles.

## Formats et alignement

Les CSV PID gardent `eventNumber, particlePDG, primaryEnergy, score`. Les CSV énergie gardent `eventNumber, particlePDG, primaryEnergy, energyPred`. Les sidecars conservent leurs clés historiques et les NPZ ne sont jamais réécrits.

Les identifiants à trois champs (`particle:shard:event`) et quatre champs (`role:particle:stem:event`) sont convertis avec `sdhcal.shards.global_event_number`. L’alignement reste une intersection de clés, suivie du tri par PDG puis numéro d’événement. Les doublons, prédictions non finies et intersections vides restent refusés.

## Compatibilité et limites scientifiques

Les artefacts historiques conservent leur ordre de variables, scaler, cible, mapping de labels et metadata. Les bundles énergie utilisant `baseline=chi2_pi` continuent de dépendre du fichier de calibration sélectionné par `SDHCAL_PARAMS_SET`. Les artefacts historiques ne portent pas tous un hash de ce fichier ; R8 ne rend donc aucune vérification nouvelle obligatoire et ne modifie aucun coefficient.

Deux divergences historiques sont intentionnellement préservées. Le prédicteur générique applique le smearing et le scaler enregistrés dans le bundle. L’étude de couplage applique sa configuration de smearing et transmet toujours les variables brutes au classifieur ; elle n’interprète pas non plus les baselines de ratio comme le prédicteur générique. Uniformiser ces comportements modifierait les résultats scientifiques et relève d’un chantier séparé.

Les services R8 ne dépendent ni de Torch, ni de PyG, ni des bibliothèques graphiques ROOT. Cette refactorisation caractérise le comportement existant sans valider les conventions PDG, l’oracle de calibration, la méthode de cœur tronqué ou la politique de smearing.
