#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import joblib
import uproot
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --- Configurations ---
scaler_path = '/gridgroup/ilc/midir/Timing/files/analyse/MLP/processed_data/scaler.joblib'
model_path  = '/gridgroup/ilc/midir/Timing/files/analyse/MLP/processed_data/mlp_classifier.joblib'
feature_cols = [
    "Begin","Thr3","Density","Radius","nClusters","lambda1","lambda2",
    "Thr3ShowerLength","nLayersThr3","planesWithClusmore2","avgClustSize",
    "maxClustSize","Rmean","Rrms","Rskew","RadiusThr3","pctHitsFirst10",
    "LayerMaxHits","Zrms","Zbary","ratioThr23","sumThrTotal","nHitsTotal",
    "Zbary_thr3","Zbary_thr2","N2","N3"
]
target_col = 'particlePDG'
tree_name  = 'tree'

# --- Sous-répertoires et mapping des labels ---
subdirs   = ['analyse_pi', 'analyse_kaon', 'analyse_proton']
label_map = {-211: 0, 2212: 1, 311: 2}
target_names = ["pi-", "proton", "K0"]

# --- Chargement scaler et modèle (une seule fois) ---
scaler = joblib.load(scaler_path)
model  = joblib.load(model_path)
print(f"Modèle chargé, attend {model.n_features_in_} features.\n")

# --- Boucle sur chaque partX ---
base_dir = '/gridgroup/ilc/midir/Timing/files/analyse/data/'
for i in range(1, 11):
    print(f"=== Évaluation pour part{i}.root ===")
    # 1) Charger et concaténer les 3 canaux
    dfs = []
    for subdir in subdirs:
        path = f"{base_dir}/{subdir}/part{i}.root"
        with uproot.open(path) as f:
            if tree_name not in f:
                raise ValueError(f"TTree '{tree_name}' absent dans {path}")
            df = f[tree_name].arrays(feature_cols + [target_col], library='pd')
            dfs.append(df)
    df_test = pd.concat(dfs, ignore_index=True)
    print(f"  -> Events bruts : {len(df_test)}")

    # 2) Nettoyage & filtrage PDG utile
    df_test = df_test.dropna().reset_index(drop=True)
    df_test = df_test[df_test[target_col].isin(label_map.keys())]
    print(f"  -> Après filtre PDG : {len(df_test)}")

    # 3) Préparation X / y
    X = df_test[feature_cols].values
    y = df_test[target_col].map(label_map)
    if y.isna().any():
        bad = df_test[target_col][y.isna()].unique()
        raise ValueError(f"PDG non mappés dans part{i} : {bad}")

    # 4) Scaling et prédiction
    X_scaled = scaler.transform(X)
    y_pred   = model.predict(X_scaled)

    # 5) Calcul des métriques
    acc = accuracy_score(y, y_pred)
    cm  = confusion_matrix(y, y_pred)
    print(f"  Accuracy : {acc:.4f}")
    print("  Matrice de confusion (vrai vs prédit) :")
    print(cm)
    print("  Rapport de classification :")
    print(classification_report(
        y, y_pred,
        labels=list(label_map.values()),
        target_names=target_names,
        zero_division=0
    ))
    print("\n")
