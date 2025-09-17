#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hadron_pid_reco.py

Effectue :
  1) Chargement du modèle MLP + scaler
  2) Prédiction du type de hadron
  3) Reconstruction énergétique linéaire + incertitude
  4) Écriture des résultats dans ROOT et CSV

Usage : python3 hadron_pid_reco.py inputfile.root output.root  
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import joblib
import uproot
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- Chemins à ajuster ---
SCALER_PATH = '/gridgroup/ilc/midir/Timing/files/analyse/MLP/processed_data/scaler.joblib'
MODEL_PATH  = '/gridgroup/ilc/midir/Timing/files/analyse/MLP/processed_data/mlp_classifier.joblib'

# --- Config Reconstruction énergétique linéaire ---
# chaque particule a trois paramètres constants [alpha, beta, gamma]
PARAMS = {
    'pi':     [0.0248983, 0.0859911, 0.372354],
    'proton': [0.0332745, 0.0949049, 0.283482],
    'kaon':   [0.036302,  0.056489,  0.353166],
}

FEATURE_COLUMNS = [
    "Begin", "Thr3", "Density", "Radius", "nClusters",
    "nLayersThr3", "planesWithClusmore2", "avgClustSize",
    "maxClustSize", "pctHitsFirst10",
    "LayerMaxHits", "Zrms", "Zbary", "ratioThr23",
    "Zbary_thr3", "Zbary_thr2"
]

LABEL_MAP = {-211: 0, 2212: 1, 311: 2}  # PDG -> modèle
IDX_TO_PARTICLE = {0: 'pi', 1: 'proton', 2: 'kaon'}
TARGET_NAMES = ["pi-", "proton", "kaon"]
TARGET_COL = 'particlePDG'


def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')


def reconstruct_energy(N1, N2, N3, particle_type):
    """ Reconstruit E_reco = alpha*N1 + beta*N2 + gamma*N3 """
    alpha, beta, gamma = PARAMS[particle_type]
    return alpha * N1 + beta * N2 + gamma * N3


def reconstruct_energy_uncertainty(N1, N2, N3, particle_type):
    """
    Propagation d'erreur en supposant sigma_Ni = sqrt(Ni) et
    dE/dNi = paramètre correspondant.
    """
    alpha, beta, gamma = PARAMS[particle_type]
    s1, s2, s3 = np.sqrt(N1), np.sqrt(N2), np.sqrt(N3)
    var = (alpha * s1)**2 + (beta * s2)**2 + (gamma * s3)**2
    return np.sqrt(var)


def load_model_and_scaler():
    if not os.path.isfile(MODEL_PATH) or not os.path.isfile(SCALER_PATH):
        logging.error("Modèle ou scaler introuvable.")
        sys.exit(1)
    return joblib.load(MODEL_PATH), joblib.load(SCALER_PATH)


def read_root(path):
    if not os.path.isfile(path):
        logging.error(f"Fichier ROOT introuvable : {path}")
        sys.exit(1)
    tree = uproot.open(path)['paramsTree']
    df = tree.arrays(library='pd')
    # Si N1 manquant, on le recompose
    if 'N1' not in df.columns and {'nHitsTotal','N2','N3'}.issubset(df.columns):
        df['N1'] = (df['nHitsTotal'] - df['N2'] - df['N3']).astype(int)
        logging.info("Colonne N1 reconstruite.")
    return df


def write_root(df, path):
    arrays = {col: df[col].to_numpy() for col in df.columns}
    with uproot.recreate(path) as fout:
        fout['tree'] = arrays


def write_csv(df, path):
    df.to_csv(path, index=False)
    logging.info(f"CSV écrit : {path}")


def process_events(df, model, scaler):
    missing = set(FEATURE_COLUMNS) - set(df.columns)
    if missing:
        raise KeyError(f"Colonnes manquantes : {missing}")
    df = df.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)

    # 1) prédiction PID
    X = scaler.transform(df[FEATURE_COLUMNS])
    proba = model.predict_proba(X)
    idx_max = proba.argmax(axis=1)
    labels = model.classes_[idx_max]
    particles = [IDX_TO_PARTICLE[int(lbl)] for lbl in labels]

    # 2) reconstruction Energie & incertitude
    E_rec_vals = []
    E_unc_vals = []
    for i, ptype in enumerate(particles):
        N1 = int(df.at[i, 'N1'])
        N2 = int(df.at[i, 'N2'])
        N3 = int(df.at[i, 'N3'])
        Erec = reconstruct_energy(N1, N2, N3, ptype)
        Eunc = reconstruct_energy_uncertainty(N1, N2, N3, ptype)
        E_rec_vals.append(Erec)
        E_unc_vals.append(Eunc)

    df['particle'] = particles
    df['pid_conf'] = np.round(proba[np.arange(len(df)), idx_max], 3)
    df['E_reco']  = np.round(E_rec_vals, 3)
    df['E_unc']   = np.round(E_unc_vals, 3)

    return df, labels


def evaluate(df, predictions):
    if TARGET_COL not in df.columns:
        logging.warning("Colonne target absente : évaluation impossible.")
        return
    y_true = df[TARGET_COL].map(LABEL_MAP)
    mask = ~y_true.isna()
    y_true = y_true[mask].astype(int)
    y_pred = predictions[mask].astype(int)

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true, y_pred,
        labels=list(IDX_TO_PARTICLE.keys()),
        target_names=TARGET_NAMES,
        zero_division=0
    )
    print("\n--- Évaluation du modèle ---")
    print(f"Accuracy : {acc:.4f}")
    print("Matrice de confusion (vrai vs prédit) :\n", cm)
    print("Rapport de classification :\n", report)


def main():
    setup_logging()
    if len(sys.argv) != 3:
        print("Usage: python hadron_pid_reco.py input.root output.root")
        sys.exit(1)
    input_path, output_path = sys.argv[1], sys.argv[2]

    model, scaler = load_model_and_scaler()
    df = read_root(input_path)

    try:
        df_out, pred_labels = process_events(df, model, scaler)
        evaluate(df, pred_labels)
    except Exception as e:
        logging.error(f"Erreur de traitement : {e}")
        sys.exit(3)

    # écriture ROOT
    write_root(df_out, output_path)
    logging.info("Fichier ROOT écrit : %s", output_path)
    # écriture CSV dérivé
    csv_path = os.path.splitext(output_path)[0] + '_results_linear.csv'
    write_csv(df_out[['particle', 'pid_conf', 'E_reco', 'E_unc']], csv_path)
    logging.info("Traitement terminé avec succès.")


if __name__ == '__main__':
    main()
