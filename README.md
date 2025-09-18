Voici un **README.md** prêt à coller à la racine de ton dépôt.
Il est pensé pour : SDHCAL, ROOT + C++11, Python (BDT/MLP/GNN), et reconstruction d’énergie avec/sans PID.
J’ai gardé les chemins exacts de ton arborescence et évité d’inventer des options de scripts (tu pourras compléter si tu exposes une CLI).

---

# SDHCAL PID ANALYSIS

Identification de particules (π⁻/K⁰/proton) et reconstruction d’énergie dans un SDHCAL (Semi-Digital Hadronic CALorimeter) à partir de jeux simulés (1–130 GeV).
Le projet regroupe : extraction de paramètres de gerbe (C++/ROOT), modèles de PID (BDT/MLP/GNN), reconstruction d’énergie (BDT/MLP et méthodes χ²/TMinuit), et figures associées.

## Points clés

* **Données** : fichiers ROOT simulés (brut, digitized, paramètres, splits, sets de validation) sous `data/`.
* **Paramètres de gerbe** : calculés/soignés dans `ShowerAnalyzer/` et `data/scripts/` (C++/ROOT + Python utilitaires).
* **PID** : modèles **BDT**, **MLP**, **GNN** sous `PID/` (artefacts, courbes, matrices de confusion).
* **Reconstruction d’énergie** : variantes **BDT/MLP** (Python) et **χ²/TMinuit** (C++/ROOT) sous `Energy_reconstruction_ml/` et `energy_reconstruction_Tminuit/`.
* **Comparaisons** : scripts ROOT pour comparer densité/rayon/EM-fraction/Thr3 sous `compare_parameters/`.

---

## Table des matières

* [Pré-requis](#pré-requis)
* [Installation rapide](#installation-rapide)
* [Organisation du dépôt](#organisation-du-dépôt)
* [Pipelines typiques](#pipelines-typiques)

  * [1) Préparation & paramètres](#1-préparation--paramètres)
  * [2) PID (BDT/MLP/GNN)](#2-pid-bdtmlpgnn)
  * [3) Reconstruction d’énergie](#3-reconstruction-dénergie)
  * [4) PID → Énergie (couplage)](#4-pid--énergie-couplage)
* [Résultats & figures](#résultats--figures)
* [Conseils de reproductibilité](#conseils-de-reproductibilité)
* [Contribuer](#contribuer)
* [Citation](#citation)
* [Licence](#licence)

---

## Pré-requis

### Outils système

* **C++11** (compilation testée en C++11)
* **ROOT** (avec `root-config` dans le PATH)
* **Python 3.9+** recommandé
* (Optionnel) **conda** / **mamba** pour l’environnement Python

### Python (typique)

* numpy, pandas, scikit-learn, joblib
* lightgbm (pour LGBM)
* matplotlib
* (GNN) PyTorch + PyTorch Geometric (si tu utilises `PID/GNN/`)

> Les modèles entraînés/scaleurs (fichiers `.joblib` et `.pt/.pth`) sont déjà versionnés dans les sous-dossiers `results*/models` pour référence, mais **ne sont pas nécessaires** si tu réentraînes.

---

## Installation rapide

```bash
# 1) Cloner
git clone <URL_DU_REPO> SDHCAL_PID_ANALYSIS
cd SDHCAL_PID_ANALYSIS

# 2) (Optionnel) Créer l'environnement
conda create -n sdhcal python=3.10 -y
conda activate sdhcal

# 3) Installer les dépendances usuelles
pip install numpy pandas scikit-learn joblib lightgbm matplotlib

# (GNN uniquement)
# pip install torch torch_geometric  # à adapter selon ta plateforme CUDA/CPU
```

Côté C++/ROOT (exemple de compilation manuelle) :

```bash
# Exemple: compiler un binaire simple utilisant ROOT (adapter le .cpp)
g++ -std=c++11 ShowerAnalyzer/computeParams.cpp $(root-config --cflags --libs) -o computeParams
```

---

## Organisation du dépôt

* `data/` — **Jeux de données** et artefacts

  * `raw/`, `digitized/` — fichiers ROOT bruts/digitisés
  * `params/`, `merged_primaryEnergy/` — fichiers ROOT de **paramètres** (features) par particule/énergie
  * `split*/`, `validation_set_*.root`, `val_set_*.root` — splits & sets de validation
  * `scripts/` — utilitaires (merge, repair, visualisation, split, shuffle, etc.)
  * `data_1k/` — petit set + scripts `root_to_csv.py`, `clean_csv.py`, `analyse_csv.py`
* `ShowerAnalyzer/` — **Extraction de paramètres C++/ROOT** (computeParams, version parallèle, logs)
* `PID/` — **Identification de particules**

  * `BDT/`, `MLP/`, `GNN/` — scripts d’entraînement/inférence, artefacts (`models/`, `plots/`, CSVs)
* `Energy_reconstruction_ml/` — **Reconstruction d’énergie** par BDT/MLP

  * `BDT/` et `MLP/` avec scripts, paramètres, performances et **plots**
* `energy_reconstruction_Tminuit/` — **Méthodes χ²/TMinuit** (ROOT/C++), par espèce (kaon, proton, pion-) + plots globaux
* `compare_parameters/` — Comparaisons de variables (ROOT macros `.C` et figures)
* `PID_RECONSTRUCTION/` — **Études couplées** PID → reconstruction d’énergie (figures, CSVs, scripts)
* `tools/` — utilitaires (RANSAC tracks, visualisation de gerbes)

---

## Pipelines typiques

### 1) Préparation & paramètres

1. **A partir de `data/raw/`** → digitisation/params (déjà présents sous `data/digitized/` et `data/params/`).
2. **Extraction/clean/merge** via `data/scripts/` (ex. `merge_primary_energy.py`, `repair_params.py`, `rootspliter.py`).
3. (C++) **Extraction parallèle** possible via `ShowerAnalyzer/computeParams_parallel.cpp`.

> Si tu repars de RAW, assure-toi que ROOT voit bien tes includes et que tu compiles en **C++11**.

### 2) PID (BDT/MLP/GNN)

* **BDT** : `PID/BDT/`

  * Entraînement/visualisation : `LGBM_classifier_PID.py`, `feature_importance_with_permutation.py`, `plot_trees.py`
  * Inférence (ex.) : `identify_hadron.py`
  * Artefacts : `processed_data/` (matrices de confusion, importances, scaler, modèle)
* **MLP** : `PID/MLP/`

  * Entraînement : `hadron_classifier_MLP.py` ou `2_hadron_classifier_MLP.py`
  * Résultats : `results/` (modèle, scaler, courbes d’entraînement, importances)
* **GNN** : `PID/GNN/`

  * Scripts : `GNN_3_classes.py`, `GNN.py`, variantes de debug
  * Modèles : `models/best_model_*.pt` et `best_model.pth`
  * Plots : `plots/` (courbes loss/acc, matrices de confusion)

> Les chemins d’entrée attendent des **ROOT de paramètres** (ex. `data/params/130k_*_params.root`) ou des CSV dérivés (`data/data_1k/*.csv`).
> Si tu souhaites une **CLI** unifiée, ajoute ultérieurement des arguments (train/test/split/paths) et mets-les en lumière ici.

### 3) Reconstruction d’énergie

* **ML (BDT / MLP)** : `Energy_reconstruction_ml/`

  * BDT : `hadron_energy_reco_lgbm.py`, `pion_energy_reco_lgbm.py`, `proton_energy_reco_lgbm.py`, etc.
  * MLP : `MLP_Energy_reconstruction_*.py`
  * Sorties : `performances/`, `results_*_energy_reco/` (modèles, scalers, `arrays/test_and_pred_*.npz`) & **plots** (linéarité, déviation, résolution, training)
* **χ² / TMinuit (ROOT/C++)** : `energy_reconstruction_Tminuit/`

  * Scripts par espèce (`kaon/`, `pion-/`, `proton/`), plus versions « all » (ex. `pion_proton_EnergyReco.C`)
  * Figures globales sous `energy_reconstruction_Tminuit/plots/`

### 4) PID → Énergie (couplage)

* `PID_RECONSTRUCTION/` propose des scénarios **avec** et **sans** PID (par espèce ou global)

  * CSV de synthèse : `kaon_pi-_proton/csv/` (ex. `pid_energy_LGBM.csv`, `no_pid_energy_param.csv`, …)
  * Figures comparatives : linéarité, déviation, résolution, σ/E vs E (avec/sans PID; 100 GeV/130 GeV calib…)

---

## Résultats & figures

Tu trouveras des figures prêtes à l’emploi dans :

* `PID/*/plots/` (PID)
* `Energy_reconstruction_ml/*/plots/` et `*/results_*/*/plots/` (reco d’énergie ML)
* `energy_reconstruction_Tminuit/plots/` (χ²/TMinuit)
* `PID_RECONSTRUCTION/*/plots/` (couplage PID↔Ereco)
* `compare_parameters/plots/` (variables de gerbe)

Quelques noms parlants (exemples) :

* `PID/BDT/processed_data/confusion_matrix.png`
* `Energy_reconstruction_ml/BDT/results_all_energy_reco/plots/linearity_and_deviation_all.png`
* `energy_reconstruction_Tminuit/plots/Resolution_relative_all.png`
* `PID_RECONSTRUCTION/kaon_pi-_proton/plots/PID_Resolution_relative_LGBM.png`

---

## Conseils de reproductibilité

* **Seeds** : fixe les graines (numpy/sklearn/torch) si tu veux des courbes strictement reproductibles.
* **Splits** : conserve les splits (`data/split*/`) pour comparer « à jeu égal ».
* **Normalisation** : toujours sérialiser/recharger les **scalers** correspondants au modèle (`scaler_*.joblib`).
* **Versionning** : ROOT + compilo C++11 + versions Python lib → consigne-les dans `Energy_reconstruction_ml/*/parameters/run_parameters*.csv` (déjà présent pour l’auto-traçabilité).
* **.gitignore** : le dépôt ignore modèles volumineux, résultats, données brutes (sauf README/samples). Place des **échantillons** dans `data/samples/` si tu veux des run « out-of-the-box ».

---

## Contribuer

1. Fork → branche thématique `feat/…` ou `fix/…`
2. Respecter C++ **C++11** (contexte HPC/ROOT) et PEP8 côté Python
3. Ajouter une note **Reproductibilité** (seed, splits, versions) dans tes PR
4. Pour les figures, exporter en **.pdf** et **.png** si utile, et déposer dans le dossier `plots/` pertinent

---

## Contact

* Auteur : *à compléter*
* Questions/bugs : ouvre une **Issue** avec un lien vers le script, l’input (ou échantillon) et la figure attendue.




