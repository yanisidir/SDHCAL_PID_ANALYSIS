#!/usr/bin/env python3
"""hadron_classifier_random_forest.py

Script pour entraîner un classifieur Random Forest capable de
classer une particule parmi trois espèces : pion négatif (-211), proton (2212) et kaon neutre (311),
à partir des grandeurs reconstruites stockées dans les arbres ROOT.

Sorties :
- scaler.joblib : StandardScaler adapté sur l’échantillon d’entraînement
- rf_classifier.joblib : classifieur Random Forest entraîné
- confusion_matrix.png : matrice de confusion sur le jeu de test
"""

import os
import uproot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
import joblib

def main():
    # 1. Configuration
    base_dirs = [
        "/home/ilc/midir/Timing/files/analyse/analyse_pi/",
        "/home/ilc/midir/Timing/files/analyse/analyse_kaon/",
        "/home/ilc/midir/Timing/files/analyse/analyse_proton/"
    ]
    root_filename = "Params.root"
    tree_name     = "paramsTree"
    out_dir       = "processed_data"
    os.makedirs(out_dir, exist_ok=True)

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
                "Begin", "Thr3", "Density", "Radius", "nClusters", "particlePDG", "lambda1", "lambda2",
                "Thr3ShowerLength", "nLayersThr3", "planesWithClusmore2", "avgClustSize",
                "maxClustSize", "Rmean", "Rrms", "Rskew", "RadiusThr3", "pctHitsFirst10",
                "LayerMaxHits", "Zrms", "Zbary", "ratioThr23", "sumThrTotal", "nHitsTotal",
                "Zbary_thr3","Zbary_thr2", "N2", "N3"
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
    df = df[df["particlePDG"].isin([-211, 2212, 311])].copy()
    # df = df[df["particlePDG"].isin([-211, 2212])].copy()
    if df.empty:
        raise RuntimeError("Aucune entrée après filtrage des PDG codes -211/2212/311 !")
    df["label"] = df["particlePDG"].map({-211: 0, 2212: 1, 311: 2})
    # df["label"] = df["particlePDG"].map({-211: 0, 2212: 1})

    # 5. Séparation features / labels
    feature_cols = [
        "Begin", "Thr3", "Density", "Radius", "nClusters", "lambda1", "lambda2",
        "Thr3ShowerLength", "nLayersThr3", "planesWithClusmore2", "avgClustSize",
        "maxClustSize", "Rmean", "Rrms", "Rskew", "RadiusThr3", "pctHitsFirst10",
        "LayerMaxHits", "Zrms", "Zbary", "ratioThr23", "sumThrTotal", "nHitsTotal",
        "Zbary_thr3","Zbary_thr2", "N2", "N3"
    ]
    X = df[feature_cols].values
    y = df["label"].values

    # 6. Train / Test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"   Taille train : {len(X_train)}, test : {len(X_test)}")

    # 7. Standardisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # 8. Rééquilibrage par SMOTE
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train_scaled, y_train)
    print(f"   Après SMOTE : {np.bincount(y_train_res)} samples par classe")

    # 9. Recherche des meilleurs hyperparamètres Random Forest
    # param_grid = {
    #     'n_estimators': [100, 300, 500],
    #     'max_depth': [None, 10, 20],
    #     'max_features': ['auto', 'sqrt', 'log2'],
    #     'min_samples_split': [2, 5, 10]
    # }
    # cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # rf = RandomForestClassifier(random_state=42, n_jobs= -1)  # n_jobs=-1 = utiliser tous les cœurs dispo.
    # grid = GridSearchCV(
    #     estimator=rf,
    #     param_grid=param_grid,
    #     cv=cv,
    #     scoring='f1_macro',
    #     n_jobs=1,              # ou -1    
    #     error_score='raise',   # opt
    #     verbose=2
    # )
    # grid.fit(X_train_res, y_train_res)
    # print("Meilleurs paramètres trouvés :", grid.best_params_)
    # clf = grid.best_estimator_

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        max_features='auto',
        min_samples_split=10,
        random_state=42,
        n_jobs=-1
    )

    # 10. Entraînement final sur l'ensemble rééquilibré
    print("→ Entraînement final du MLP …")
    clf.fit(X_train_res, y_train_res)
    
    # 10. Évaluation
    y_pred = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n===== Résultats =====")
    print(f"Accuracy : {acc:.4f}")
    print("Matrice de confusion :\n", cm)
    print("\nRapport de classification :\n",
          classification_report(y_test, y_pred, target_names=["pi-", "proton", "K0"]))
        #   classification_report(y_test, y_pred, target_names=["pi-", "proton"]))

    # 11. Visualisation
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["pi-", "proton", "K0"], yticklabels=["pi-", "proton", "K0"])
    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
    #             xticklabels=["pi-", "proton"], yticklabels=["pi-", "proton"])
    plt.title("Confusion Matrix - Random Forest")
    plt.xlabel("Prédit")
    plt.ylabel("Vrai")
    plt.tight_layout()
    cm_path = os.path.join(out_dir, "confusion_matrix_pion_proton.png")
    plt.savefig(cm_path, dpi=150)

    # 12. Sauvegarde
    joblib.dump(scaler, os.path.join(out_dir, "scaler.joblib"))
    joblib.dump(clf,    os.path.join(out_dir, "rf_classifier.joblib"))
    print(f"   → Modèle Random Forest et scaler sauvegardés dans {out_dir}")


if __name__ == "__main__":
    main()
