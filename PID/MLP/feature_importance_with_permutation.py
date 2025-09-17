#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""feature_importance_mlp.py (arguments hard-codés)

Calcule et sauve :
  - la matrice de corrélation des features (CSV + PDF)
  - les importances par permutation du MLP entraîné (CSV + PDF)

Aligné sur 2_hadron_classifier_MLP.py (mêmes features, cibles, split/test).
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import uproot
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

# ========================== PARAMÈTRES EN DUR ===============================
# Si RUN_ID est None, on prendra automatiquement le dernier run présent
# dans run_parameters.csv. Sinon, mettre un entier (ex: 7).
RUN_ID: int | None = None

# Fichiers ROOT à charger (doit correspondre à l’entraînement)
FILE_PATHS: List[Path] = [
    Path("/gridgroup/ilc/midir/analyse/data/params/130k_pi_E1to130_params.root"),
    Path("/gridgroup/ilc/midir/analyse/data/params/130k_kaon_E1to130_params.root"),
    Path("/gridgroup/ilc/midir/analyse/data/params/130k_proton_E1to130_params.root"),
]

TREE_NAME = "paramsTree"
TARGETS = (-211, 2212, 311)
TEST_SIZE = 0.20
RANDOM_STATE = 42

# EXACTEMENT la même liste de features que dans 2_hadron_classifier_MLP.py
FEATURE_COLS: Sequence[str] = [
    "Thr1", "Thr2", "Thr3", "Begin", "Radius", "Density", "NClusters",
    "ratioThr23", "Zbary", "Zrms", "PctHitsFirst10", "PlanesWithClusmore2",
    "AvgClustSize", "MaxClustSize", "lambda1", "lambda2",
    "tMin", "tMax", "tMean", "tSpread", "Nmax", "z0_fit", "Xmax", "lambda",
    "nTrackSegments", "eccentricity3D"
]

# Importance par permutation
N_REPEATS = 10           # nombre de permutations
SKIP_CORR = False        # mettre True pour ne pas recalculer la corrélation

# Dossiers/fichiers d’E/S (alignés avec 2_hadron_classifier_MLP.py)
PARAMETERS_FILE = Path("run_parameters.csv")
RES_DIR         = Path("results_with_time")
MODELS_DIR      = RES_DIR / "models"
PLOTS_DIR       = RES_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ========================== LOGGING =========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("feat_importance")


# ========================== OUTILS ==========================================
def _latest_run_id() -> int:
    """Retourne le dernier run_id trouvé dans run_parameters.csv."""
    if not PARAMETERS_FILE.exists():
        raise RuntimeError("run_parameters.csv introuvable. Lance d'abord 2_hadron_classifier_MLP.py.")
    run_ids: List[int] = []
    with PARAMETERS_FILE.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                run_ids.append(int(row["run_id"]))
            except (KeyError, ValueError):
                continue
    if not run_ids:
        raise RuntimeError("Aucun run_id valide trouvé dans run_parameters.csv.")
    return max(run_ids)


def _model_scaler_paths(run_id: int) -> Tuple[Path, Path]:
    model = MODELS_DIR / f"mlp_model_{run_id}.joblib"
    scaler = MODELS_DIR / f"scaler_{run_id}.joblib"
    if not model.is_file():
        raise FileNotFoundError(f"Modèle manquant: {model}")
    if not scaler.is_file():
        raise FileNotFoundError(f"Scaler manquant: {scaler}")
    return model, scaler


def load_root_files_from_paths(files: Iterable[Path], tree: str,
                               columns: Sequence[str]) -> pd.DataFrame:
    frames = []
    logger.info("Lecture des fichiers ROOT …")
    for path in files:
        if not path.is_file():
            logger.warning("  !! Fichier manquant: %s", path)
            continue
        with uproot.open(path) as f:
            frames.append(f[tree].arrays(columns, library="pd"))
    if not frames:
        raise RuntimeError("Aucun fichier ROOT valide – abandon.")
    df = pd.concat(frames, ignore_index=True)
    logger.info("Événements chargés : %d", len(df))
    return df


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.replace([np.inf, -np.inf], pd.NA).dropna().reset_index(drop=True)
    logger.info("Après nettoyage : %d (-%d)", len(df), before - len(df))
    return df


def filter_targets(df: pd.DataFrame, targets: Sequence[int]) -> pd.DataFrame:
    df = df[df["particlePDG"].isin(targets)].copy()
    if df.empty:
        raise RuntimeError("Aucune entrée après filtrage PDG.")
    label_map = {pdg: idx for idx, pdg in enumerate(targets)}
    df["label"] = df["particlePDG"].map(label_map)
    logger.info("Répartition des classes : %s", np.bincount(df["label"].values))
    return df


# ========================== CŒUR DU SCRIPT ===================================
def compute_and_save_correlation(df: pd.DataFrame,
                                 feature_cols: Sequence[str],
                                 out_prefix: str) -> None:
    """Sauve CSV + PDF de la matrice de corrélation."""
    corr = df[feature_cols].corr()
    csv_path = PLOTS_DIR / f"{out_prefix}_correlation_matrix.csv"
    pdf_path = PLOTS_DIR / f"{out_prefix}_correlation_matrix.pdf"
    corr.to_csv(csv_path, index=True)
    logger.info("Matrice de corrélation (CSV) : %s", csv_path)

    plt.figure(figsize=(16, 13))
    sns.heatmap(corr, annot=False, fmt=".2f", cmap="coolwarm",
                xticklabels=feature_cols, yticklabels=feature_cols)
    plt.title("Correlation Matrix Between Features")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(pdf_path, dpi=150)
    plt.close()
    logger.info("Matrice de corrélation (PDF) : %s", pdf_path)


def permutation_feature_importance(clf, X_test_scaled, y_test,
                                   feature_cols: Sequence[str],
                                   n_repeats: int,
                                   out_prefix: str) -> None:
    """Calcule et sauve l'importance par permutation (CSV + PDF)."""
    logger.info("Permutation importance (n_repeats=%d) …", n_repeats)
    perm_res = permutation_importance(
        clf, X_test_scaled, y_test,
        n_repeats=n_repeats,
        random_state=42,
        n_jobs=-1
    )
    imp = perm_res.importances_mean
    std = perm_res.importances_std

    df_imp = pd.DataFrame({
        "feature": feature_cols,
        "importance_mean": imp,
        "importance_std": std
    }).sort_values("importance_mean", ascending=False)

    csv_path = PLOTS_DIR / f"{out_prefix}_permutation_importances_MLP.csv"
    pdf_path = PLOTS_DIR / f"{out_prefix}_permutation_importances_MLP.pdf"
    df_imp.to_csv(csv_path, index=False)
    logger.info("Importances (CSV) : %s", csv_path)

    plt.figure(figsize=(10, 8))
    plt.barh(df_imp["feature"][::-1], df_imp["importance_mean"][::-1],
             xerr=df_imp["importance_std"][::-1])
    plt.title("Feature Importances (permutation) – MLP")
    plt.xlabel("Permutation importance (Δscore)")
    plt.tight_layout()
    plt.savefig(pdf_path, dpi=150)
    plt.close()
    logger.info("Importances (PDF) : %s", pdf_path)


def main() -> None:
    # 1) Déterminer le run_id
    run_id = _latest_run_id() if RUN_ID is None else RUN_ID
    logger.info("run_id utilisé : %d", run_id)

    # 2) Charger modèle + scaler
    model_path, scaler_path = _model_scaler_paths(run_id)
    logger.info("Chargement modèle: %s", model_path)
    logger.info("Chargement scaler: %s", scaler_path)
    clf    = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # 3) Charger données avec les mêmes colonnes que l’entraînement
    columns = list(FEATURE_COLS) + ["particlePDG"]
    df = load_root_files_from_paths(FILE_PATHS, TREE_NAME, columns)
    df = clean_dataframe(df)
    df = filter_targets(df, TARGETS)

    # 4) Recréer le même split test (même seeds / stratify)
    X = df[FEATURE_COLS].values
    y = df["label"].values
    _, X_test, _, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    X_test_scaled = scaler.transform(X_test)

    out_prefix = f"run_{run_id}"

    # 5) Corrélation
    if not SKIP_CORR:
        compute_and_save_correlation(df, FEATURE_COLS, out_prefix)

    # 6) Importances par permutation
    permutation_feature_importance(
        clf, X_test_scaled, y_test, FEATURE_COLS, N_REPEATS, out_prefix
    )


if __name__ == "__main__":
    main()
