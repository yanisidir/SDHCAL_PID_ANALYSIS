# R0 — Baseline de caractérisation

Cette baseline protège le comportement logiciel observé avant les refactorings
R1 à R10. Elle ne constitue pas une validation des choix physiques ou
statistiques.

## Provenance

La baseline a été créée sur le worktree courant, qui comportait déjà des
modifications non commitées. Les fichiers concernés incluaient notamment le
Makefile, les scripts de workflow, les modules de graphes et de PID hit-level,
des outils de construction/comparaison ainsi que le rapport. Aucun de ces
changements préexistants n'a été écrasé ou annulé par R0.

Les nouveaux contrats sont regroupés dans :

- tests/fixtures/r0_contracts.json ;
- tests/fixtures/r0_events.json ;
- tests/test_r0_cli_contracts.py ;
- tests/test_r0_data_contracts.py ;
- tests/test_r0_artifact_contracts.py ;
- tests/test_r0_scientific_baseline.py ;
- tests/test_r0_workflow_contracts.py.

## Comportements protégés

- options CLI et valeurs par défaut littérales ;
- ordre des features tabulaires, hit-level et cluster-level ;
- indices d'un petit split stratifié déterministe ;
- construction des clés événement et conversion des identifiants hit-level ;
- clés des bundles Joblib tabulaires ;
- colonnes CSV et clés NPZ des prédictions ;
- petites références numériques exactes pour la calibration, les métriques, la
  ROC et les transformations de cible ;
- points d'entrée Make et Shell de l'analyse deux classes.

## Problèmes scientifiques connus non validés par R0

Les tests de caractérisation ne doivent pas être interprétés comme une validation
des points suivants :

- le modèle énergie publié sous le nom GNN peut être exécuté en architecture
  DeepSets ;
- les options causale et non orientée des graphes peuvent être activées
  simultanément ;
- certaines réponses en énergie utilisent un cœur tronqué plutôt qu'une moyenne
  sur l'échantillon complet ;
- l'identité physique associée au code PDG 311 doit être vérifiée ;
- les procédures optionnelles de grid search, le subsampling LightGBM et la
  politique de smearing demandent une revue scientifique séparée ;
- la calibration oracle et les budgets d'entraînement doivent être explicitement
  qualifiés dans les comparaisons ;
- l'extracteur ROOT parallèle présente un risque de collision de branches ;
- certains artefacts dépendent encore de fichiers de calibration externes et la
  sélection de fichiers par date de modification reste fragile.

Toute correction de ces sujets doit modifier ou remplacer le contrat concerné
dans un changement scientifique explicite, et non être glissée dans un
refactoring architectural.
