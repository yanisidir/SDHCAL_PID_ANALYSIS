#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""hadron_classifier.py

Script pour entraîner un réseau de neurones (MLP scikit‑learn) capable de
classer une particule parmi trois espèces : pion négatif (-211), proton (2212) et kaon neutre (311),
à partir des grandeurs reconstruites stockées dans les arbres ROOT.

Sorties :
- scaler.joblib : StandardScaler adapté sur l’échantillon d’entraînement
- mlp_classifier.joblib : classifieur MLP entraîné
- confusion_matrix.pdf : matrice de confusion sur le jeu de test
"""

import os
import uproot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
import joblib


def main():
    # 1. Configuration
    base_dirs = [
        "/gridgroup/ilc/midir/analyse/analyse_pi/data/",
        "/gridgroup/ilc/midir/analyse/analyse_kaon/data/",
        "/gridgroup/ilc/midir/analyse/analyse_proton/data/"
    ]

    # root_filename = "Params.root"
    root_filename = "130k_Params.root"
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
                "LayerMaxHits", "Zrms", "Zbary", "ratioThr23","sumThrTotal", "nHitsTotal",
                "Zbary_thr3","Zbary_thr2", "N2", "N3", "Thr2", "Thr1"
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
    
    # feature_cols = ["Begin", "Thr3", "Density", "Radius", "nClusters", "lambda1", "lambda2",
    # "Thr3ShowerLength", "nLayersThr3", "planesWithClusmore2", "avgClustSize",
    # "maxClustSize", "Rmean", "Rrms", "Rskew", "RadiusThr3", "pctHitsFirst10",
    # "LayerMaxHits", "Zrms", "Zbary", "ratioThr23", "sumThrTotal", "nHitsTotal",
    # "Zbary_thr3","Zbary_thr2", "N2", "N3"]    
    
    feature_cols = ["Begin", "Thr3", "Density", "Radius", "nClusters",
    "nLayersThr3", "planesWithClusmore2", "avgClustSize",
    "maxClustSize", "pctHitsFirst10",
    "LayerMaxHits", "Zrms", "Zbary", "ratioThr23",
    "Zbary_thr3","Zbary_thr2"]    

    # feature_cols = ["Begin", "Thr3", "Density", "Radius", "nClusters",
    # "Thr3ShowerLength", "nLayersThr3", "planesWithClusmore2", "avgClustSize",
    # "maxClustSize", "Rmean", "Rrms", "pctHitsFirst10",
    # "LayerMaxHits", "Zrms", "Zbary", "ratioThr23","Zbary_thr3", "N2"] 

    # feature_cols = ["Begin", "Thr3", "Density", "Radius", "nClusters", "pctHitsFirst10", "Zrms", "Zbary", "ratioThr23", "N2"] 

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

    # 9. Recherche des meilleurs hyperparamètres
    param_grid = {
        'hidden_layer_sizes': [(64,32), (128,64,32)],
        'alpha': [1e-5, 1e-4, 1e-3],
        'learning_rate_init': [1e-4, 5e-4, 1e-3],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    base_clf = MLPClassifier(
        activation='relu', solver='adam', batch_size='auto',
        max_iter=300, early_stopping=True, n_iter_no_change=15,
        random_state=42, verbose=False
    )
    grid = GridSearchCV(base_clf, param_grid, cv=cv, scoring='f1_macro', n_jobs=-1, verbose=2)
    grid.fit(X_train_res, y_train_res)
    print("Meilleurs paramètres trouvés :", grid.best_params_)
    clf = grid.best_estimator_

    # clf = MLPClassifier(
    #     hidden_layer_sizes=(128, 64, 32),
    #     alpha=0.0001,
    #     learning_rate_init=0.001,
    #     activation='relu',
    #     solver='adam',
    #     batch_size='auto',
    #     max_iter=300,
    #     early_stopping=True,
    #     n_iter_no_change=15,
    #     random_state=42,
    #     verbose=False
    # )

    clf = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),
    alpha=0.0001,
    learning_rate='adaptive',
    learning_rate_init=0.001,
    activation='relu',
    solver='adam',
    batch_size=32,
    max_iter=500,
    early_stopping=True,
    n_iter_no_change=15,
    random_state=42,
    verbose=True
    )
        
    # 10. Entraînement final sur l'ensemble rééquilibré
    print("→ Entraînement final du MLP …")
    clf.fit(X_train_res, y_train_res)

    
    # 11. Évaluation
    y_pred = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # Probabilités prédites pour chaque classe (shape = [n_samples, 3])
    proba_test = clf.predict_proba(X_test_scaled)

    # 1) Courbe de loss
    plt.figure(figsize=(6,4))
    plt.plot(clf.loss_curve_, label="loss train")
    # 2) Courbe de score validation interne
    plt.plot(1 - np.array(clf.validation_scores_), '--', label="1 - score valid")
    plt.xlabel("Itération")
    plt.ylabel("Loss / 1 - Accuracy")
    plt.legend()
    plt.title("Évolution training loss vs validation score")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_vs_validation.pdf"), dpi=150)    

    print(f"\n===== Résultats =====")
    print(f"Accuracy : {acc:.4f}")
    print("Matrice de confusion :\n", cm_percent)
    print("\nRapport de classification :\n",
          classification_report(y_test, y_pred, target_names=["pi-", "proton", "K0"]))
        #   classification_report(y_test, y_pred, target_names=["pi-", "proton"]))

    annot = np.array([[f"{val:.1f}%" for val in row] for row in cm_percent])
    
    # 11. Visualisation
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_percent, annot=annot, fmt='', cmap='Blues', xticklabels=["pi-", "proton", "K0"], yticklabels=["pi-", "proton", "K0"])
    # sns.heatmap(cm_percent, annot=annot, fmt='', cmap='Blues', xticklabels=["pi-", "proton"], yticklabels=["pi-", "proton"])
    plt.title("Confusion Matrix - MLP")
    plt.xlabel("Prédit")
    plt.ylabel("Vrai")
    plt.tight_layout()
    cm_path = os.path.join(out_dir, "confusion_matrix.pdf")
    plt.savefig(cm_path, dpi=150)

    # Figure à 3 sous-graphes : distribution de la proba pour la classe proton
    plt.figure(figsize=(15,4))

    # 1er sous-graphe : histogramme des proba de classe ‘proton’ selon la vraie étiquette
    plt.subplot(131)
    plt.hist([p[1] for p,l in zip(proba_test, y_test) if l==0],
            bins=50, alpha=0.5, label='Pion- (vrai)')
    plt.hist([p[1] for p,l in zip(proba_test, y_test) if l==1],
            bins=50, alpha=0.5, label='Proton (vrai)')
    plt.xlabel('Probabilité prédite de proton')
    plt.ylabel('Nbr d’événements')
    plt.legend()
    plt.title('Distributions de la probabilité proton')

    # 2ème sous-graphe : même chose pour la classe K0 avec pion-
    plt.subplot(132)
    plt.hist([p[2] for p,l in zip(proba_test, y_test) if l==0],
            bins=50, alpha=0.5, label='Pion- (vrai)')
    plt.hist([p[2] for p,l in zip(proba_test, y_test) if l==2],
            bins=50, alpha=0.5, label='K0 (vrai)')
    plt.xlabel('Probabilité prédite de K0')
    plt.legend()
    plt.title('Distributions de la probabilité K0\n(vrai Pion- vs vrai K0)')

    # 3ème sous-graphe : même chose pour la classe K0 avec proton
    plt.subplot(133)
    plt.hist([p[2] for p,l in zip(proba_test, y_test) if l==1],
            bins=50, alpha=0.5, label='Proton (vrai)')
    plt.hist([p[2] for p,l in zip(proba_test, y_test) if l==2],
            bins=50, alpha=0.5, label='K0 (vrai)')
    plt.xlabel('Probabilité prédite de K0')
    plt.legend()
    plt.title('Distributions de la probabilité K0\n(vrai Proton vs vrai K0)')

    plt.show()
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "probability_histograms.pdf"), dpi=150)

    # 13. Sauvegarde
    joblib.dump(scaler, os.path.join(out_dir, "scaler.joblib"))
    joblib.dump(clf,    os.path.join(out_dir, "mlp_classifier.joblib"))
    print(f"   → Modèle et scaler sauvegardés dans {out_dir}")


if __name__ == "__main__":
    main()
