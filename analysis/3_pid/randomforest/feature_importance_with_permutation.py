#!/usr/bin/env python3
"""feature_importance_with_permutation.py

Script pour afficher et visualiser les importances des features
à partir du classifieur RandomForest entraîné, avec étiquettes lisibles,
calculer aussi les importances par permutation sans besoin de fichiers test pré-sauvegardés.

Entrées :
- processed_data/scaler.joblib
- processed_data/rf_classifier.joblib
- data/raw_dataset.csv      # fichier CSV contenant toutes les features et la cible

Sorties :
- feature_importances.csv : importances RF triées
- permutation_importances.csv : importances par permutation triées
- feature_importances.pdf : graphique RF horizontal
- permutation_importances.pdf : graphique permutation horizontal
"""
import os
import uproot
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance


def main():
    # Dossiers et fichiers
    base_dirs = [
        "/home/ilc/midir/Timing/files/analyse/analyse_pi/",
        # "/home/ilc/midir/Timing/files/analyse/analyse_kaon/",
        "/home/ilc/midir/Timing/files/analyse/analyse_proton/"
    ]
    root_filename = "Params.root"
    tree_name     = "paramsTree"
    out_dir       = "processed_data"
    os.makedirs(out_dir, exist_ok=True)

    model_path = os.path.join(out_dir, "rf_classifier.joblib")
    scaler_path = os.path.join(out_dir, "scaler.joblib")
    raw_data_path = os.path.join("data", "raw_dataset.csv")

    # 2. Chargement et concaténation
    dfs = []
    print("→ Lecture de tous les fichiers ROOT …")
    for d in base_dirs:
        path = os.path.join(d, root_filename)
        if not os.path.isfile(path):
            print(f"  !!  Fichier manquant: {path}")
            continue
        f = uproot.open(path)
        tree = f[tree_name]
        df = tree.arrays(
            [
                "Begin", "Thr3", "Thr2", "Thr1", "Density", "Radius", "nClusters", "particlePDG",
                "lambda1", "lambda2", "Thr3ShowerLength", "nLayersThr3", "planesWithClusmore2",
                "avgClustSize", "maxClustSize", "Rmean", "Rrms", "Rskew", "RadiusThr3",
                "pctHitsFirst10", "LayerMaxHits", "Zrms", "Zbary", "ratioThr23", "sumThrTotal", "nHitsTotal"
            ],
            library="pd"
        )
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    print(f"   Total d’événements chargés : {len(df)}")

    # 3. Nettoyage (suppression des valeurs infinies / manquantes)
    df = df.replace([np.inf, -np.inf], pd.NA).dropna().reset_index(drop=True)
    print(f"   Après nettoyage (drop NA/inf) : {len(df)}")

    # 4. Filtrage des particules cibles
    # df = df[df["particlePDG"].isin([-211, 2212, 311])].copy()
    df = df[df["particlePDG"].isin([-211, 2212])].copy()
    if df.empty:
        raise RuntimeError("Aucune entrée après filtrage des PDG codes -211/2212/311 !")
    # df["label"] = df["particlePDG"].map({-211: 0, 2212: 1, 311: 2})
    df["label"] = df["particlePDG"].map({-211: 0, 2212: 1})

    # 5. Séparation features / labels
    feature_cols = [
        "Begin", "Thr3", "Density", "Radius", "nClusters", "lambda1", "lambda2",
        "Thr3ShowerLength", "nLayersThr3", "planesWithClusmore2", "avgClustSize",
        "maxClustSize", "Rmean", "Rrms", "Rskew", "RadiusThr3", "pctHitsFirst10",
        "LayerMaxHits", "Zrms", "Zbary", "ratioThr23", "sumThrTotal", "nHitsTotal"
    ]
    X = df[feature_cols].values
    y = df["label"].values

    # Chargement du modèle et du scaler
    clf = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # Reconstruction du test set
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    # Standardisation
    X_test_scaled = scaler.transform(X_test)

    # 1) Importance native RF
    imp_rf = clf.feature_importances_
    df_rf = pd.DataFrame({'feature': feature_cols, 'importance': imp_rf})
    df_rf = df_rf.sort_values('importance', ascending=False)
    df_rf.to_csv(os.path.join(out_dir, 'feature_importances.csv'), index=False)
    print("Importances RF sauvegardées dans feature_importances.csv")

    # Visualisation RF
    plt.figure(figsize=(10, 8))
    plt.barh(df_rf['feature'][::-1], df_rf['importance'][::-1])
    plt.title('Importances des features (RandomForest)')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'feature_importances.pdf'), dpi=150)
    print("Graphique RF sauvegardé dans feature_importances.pdf")

    # 2) Importance par permutation
    perm_res = permutation_importance(
        clf, X_test_scaled, y_test,
        n_repeats=10,
        random_state=42,
        n_jobs=-1
    )
    imp_perm = perm_res.importances_mean

    df_perm = pd.DataFrame({'feature': feature_cols, 'perm_importance': imp_perm})
    df_perm = df_perm.sort_values('perm_importance', ascending=False)
    df_perm.to_csv(os.path.join(out_dir, 'permutation_importances_RF.csv'), index=False)
    print("Importances par permutation sauvegardées dans permutation_importances_RF.csv")

    # Visualisation permutation
    plt.figure(figsize=(10, 8))
    plt.barh(df_perm['feature'][::-1], df_perm['perm_importance'][::-1])
    plt.title('Importances des features (Permutation) RF')
    plt.xlabel('Importance par permutation')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'permutation_importances_RF.pdf'), dpi=150)
    print("Graphique permutation sauvegardé dans permutation_importances_RF.pdf")


if __name__ == "__main__":
    main()
