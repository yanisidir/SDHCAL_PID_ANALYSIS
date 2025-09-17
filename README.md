Ce dépôt contient le **code** (PID: BDT/MLP/CNN/GNN, reco d'énergie: LGBM/MLP/TMinuit).
Les **données (.root)** et **modèles (.joblib/.pt)** ne sont **pas** versionnés ici.

## Lancer rapidement
- Crée un env: `conda create -n sdhcal python=3.10 -y && conda activate sdhcal`
- Installe: `pip install numpy pandas scikit-learn lightgbm uproot matplotlib`
- Utilise les scripts dans `PID/` et `Energy_reconstruction_ml/` (voir les fichiers `.py`).

## Données
Place des **échantillons** dans `data/samples/` si besoin. Les gros `.root` restent en local.
