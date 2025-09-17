#!/usr/bin/env python3
# PID_Inference_with_Parametric_Energy_LGBM_pi_p.py
# PID binaire π vs p avec énergie paramétrique — kaons exclus

from __future__ import annotations
import logging
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
import uproot
import lightgbm as lgb  # utilisé pour typer le classif chargé
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Sequence  # ### CHANGEMENT

# ------------------------------ Logging --------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ------------------------------ Config ---------------------------------------
class InferenceCfg:
    tree_name = "tree"

    # Features utilisées pour la classification (doivent matcher l'entraînement)
    clf_features = [
        "Thr1", "Thr2", "Thr3", 
        "Begin", "meanRadius", "nMipCluster", "first5LayersRMS",
        "Density", "NClusters", "ratioThr23", "Zbary", "Zrms",
        "PctHitsFirst10", "AvgClustSize", "MaxClustSize", "lambda1", "lambda2",
        "tMin", "tMax", "tMean", "tSpread", "Nmax", "Xmax",
        "eccentricity3D", "transverseRatio", "nTrack"
    ]

    # Colonnes nécessaires pour l'énergie paramétrique
    energy_cols = ["N1", "N2", "N3", "nHitsTotal", "sumThrTotal"]

    # Colonnes vérité (présentes dans validation_set_pion_proton.root)
    energy_col = "primaryEnergy"
    pdg_col    = "primaryID"

    # ### CHANGEMENT: binaire π/p uniquement
    label_to_tag  = {0: "pi", 1: "proton"}
    label_to_name = {0: "π-", 1: "p"}
    pdg_to_label  = {-211: 0, 2212: 1}

    # --- Smearing temporel optionnel (pour reproduire la résolution) ---
    time_smear_sigma_clf: float | None = 0.1
    time_cols_clf = ["tMin", "tMax", "tSpread"]


# ---------------- Paramètres figés (Par9) ----------------
# Chaque vecteur P contient 9 coefficients: (a0,a1,a2, b0,b1,b2, g0,g1,g2)
PAR_PI = np.array([
    5.49538e-02, -6.62504e-05, 5.16959e-08,
    7.69174e-02,  3.30055e-05, -1.00986e-07,
    2.52556e-13,  7.26170e-04, -2.94994e-07
], dtype=np.float64)

PAR_PROTON = np.array([
    5.32756e-02, -2.63648e-05, 1.19380e-08,
    1.00773e-01, -5.11167e-05, 1.19408e-08,
    2.75654e-12,  4.93381e-04, -1.76195e-07
], dtype=np.float64)

# ### CHANGEMENT: dictionnaire sans entrée kaon
PAR_BY_TAG = {
    "pi": PAR_PI,
    "proton": PAR_PROTON,
}

# ------------------------------ Helpers --------------------------------------
def _require_exists(p: Path, what: str):
    if not p.exists():
        logger.error("%s introuvable: %s", what, p)
        sys.exit(1)

def _require_columns(df: pd.DataFrame, cols: list[str], where: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        logger.error("Colonnes manquantes (%s): %s", where, ", ".join(missing))
        sys.exit(1)

def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 2) -> np.ndarray:  # ### CHANGEMENT
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            cm[t, p] += 1
    return cm

def _sigma_over_E(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    r = (y_pred - y_true) / y_true
    return float(np.sqrt(np.mean(r**2)))

# Smearing utilitaire
def smear_columns_gaussian(df: pd.DataFrame, cols: list[str], sigma: float | None, seed: int = 42) -> pd.DataFrame:
    if sigma is None:
        return df
    rng = np.random.default_rng(seed)
    n = len(df)
    for c in cols:
        if c in df.columns:
            df[c] = df[c].to_numpy() + rng.normal(0.0, sigma, size=n)
        else:
            logger.warning("Colonne %s absente pour smearing (ignorée)", c)
    return df

# ------------------------------- I/O -----------------------------------------
def load_root_files(paths, tree, columns):
    frames = []
    for path in paths:
        p = Path(path)
        if not p.exists():
            logger.error("Fichier introuvable: %s", path)
            continue
        with uproot.open(p) as f:
            if tree not in f:
                logger.error("Tree '%s' absent dans %s", tree, p)
                sys.exit(1)
            frames.append(f[tree].arrays(columns, library="pd"))
    if not frames:
        logger.error("Aucun fichier ROOT valide n'a été chargé.")
        sys.exit(1)
    df = pd.concat(frames, ignore_index=True)
    before = len(df)
    df = df.replace([np.inf, -np.inf], pd.NA).dropna().reset_index(drop=True)
    logger.info("Chargé %d événements (après nettoyage, -%d)", len(df), before - len(df))
    return df

# ------------------------------ Energy via Par9 ------------------------------
def energy_parametric(N1: np.ndarray, N2: np.ndarray, N3: np.ndarray, P: np.ndarray) -> np.ndarray:
    N1 = N1.astype(np.float64, copy=False)
    N2 = N2.astype(np.float64, copy=False)
    N3 = N3.astype(np.float64, copy=False)
    N  = N1 + N2 + N3
    a0,a1,a2, b0,b1,b2, g0,g1,g2 = P
    alpha = a0 + a1*N + a2*(N**2)
    beta  = b0 + b1*N + b2*(N**2)
    gamma = g0 + g1*N + g2*(N**2)
    return (alpha*N1 + beta*N2 + gamma*N3).astype(np.float32)

# ------------------------------ Confusion matrix -----------------------------
def plot_confusion_matrix(cm: np.ndarray, labels: Sequence[str], out: Path):
    pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100.0
    annot = np.array([[f"{v:.1f}%" for v in row] for row in pct])

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        pct, annot=annot, fmt='', cmap='Blues',
        xticklabels=labels, yticklabels=labels,
        cbar_kws={'label': 'Pourcentage (%)'}
    )
    plt.xlabel("Prédit")
    plt.ylabel("Vrai")
    plt.title("Matrice de confusion PID (π vs p)")  # ### CHANGEMENT
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


# ------------------------------ Inference ------------------------------------
def run_inference(files, out_csv):
    cfg = InferenceCfg()

    # === chemins  ===
    clf_model_path  = Path("/gridgroup/ilc/midir/analyse/PID/BDT/results_with_time/models/lgbm_model_29.joblib")

    _require_exists(clf_model_path,  "Modèle classif LGBM")

    # Colonnes nécessaires = features classif + colonnes énergie paramétrique + vérité
    need_cols = sorted(set(cfg.clf_features).union(cfg.energy_cols).union([cfg.energy_col, cfg.pdg_col]))
    df_all = load_root_files(files, cfg.tree_name, need_cols)
    _require_columns(df_all, cfg.clf_features, "features classif")
    _require_columns(df_all, ["N1","N2","N3"], "énergie paramétrique (N1,N2,N3)")
    _require_columns(df_all, [cfg.energy_col, cfg.pdg_col], "vérité (E, PDG)")

    # ### CHANGEMENT: on retire les événements kaons
    df_all = df_all[df_all[cfg.pdg_col].isin([-211, 2212])].reset_index(drop=True)
    if len(df_all) == 0:
        logger.error("Après filtrage π/p, aucun événement disponible.")
        sys.exit(1)
    logger.info("Événements conservés (π/p uniquement): %d", len(df_all))

    # --- Smearing temporel optionnel (classification) ---
    df_all = smear_columns_gaussian(df_all, cfg.time_cols_clf, cfg.time_smear_sigma_clf, seed=42)

    # --- Classification (LightGBM) ---
    logger.info("Chargement classif LGBM (BDT)")
    clf: lgb.LGBMClassifier = joblib.load(clf_model_path)

    Xc = df_all[cfg.clf_features].to_numpy(dtype=np.float32, copy=False)
    proba_all = clf.predict_proba(Xc)  # shape (N, n_classes)
    classes = np.array(clf.classes_)

    # On veut forcer un problème binaire: pi=0, proton=1
    # -> récupérer les colonnes de proba correspondant à ces labels, ignorer le reste.
    # Vérifications de base:
    wanted = {0, 1}
    have = set(classes.tolist())
    if not wanted.issubset(have):
        logger.error("Le classif ne contient pas les deux classes nécessaires {0,1}. Classes présentes: %s", classes)
        sys.exit(1)

    # Indices des colonnes pour pi et proton (dans cet ordre!)
    idx_pi = int(np.where(classes == 0)[0][0])
    idx_p  = int(np.where(classes == 1)[0][0])

    # Constituer une matrice (N,2): [P(pi), P(proton)]
    y_proba_bin = np.stack([proba_all[:, idx_pi], proba_all[:, idx_p]], axis=1)

    # Prédiction binaire
    y_pred = np.argmax(y_proba_bin, axis=1)          # 0=π, 1=p
    class_name = np.vectorize(cfg.label_to_name.get)(y_pred)
    proba_max = y_proba_bin[np.arange(len(y_pred)), y_pred]

    # --- Énergie paramétrique (routage binaire par classe prédite) ---
    N1 = df_all["N1"].to_numpy()
    N2 = df_all["N2"].to_numpy()
    N3 = df_all["N3"].to_numpy()

    E_pred = np.full(len(df_all), np.nan, dtype=np.float32)
    for label_int, tag in cfg.label_to_tag.items():  # {0:"pi", 1:"proton"}
        mask = (y_pred == label_int)
        n = int(mask.sum())
        if n == 0:
            continue
        P = PAR_BY_TAG[tag]
        E_pred[mask] = energy_parametric(N1[mask], N2[mask], N3[mask], P)
        logger.info("Énergie paramétrique %s: %d événements", tag, n)

    # --- Vérité : labels & énergie ---
    pdg_true = df_all[cfg.pdg_col].to_numpy()
    E_true   = df_all[cfg.energy_col].to_numpy(dtype=np.float32, copy=False)
    label_true_int = np.vectorize(lambda pdg: cfg.pdg_to_label.get(int(pdg), -1))(pdg_true)
    label_true = np.vectorize(lambda k: cfg.label_to_name.get(int(k), "UNK"))(label_true_int)

    # --- Métriques classification (2 classes) ---
    valid_mask = (label_true_int >= 0)
    if np.any(valid_mask):
        acc = float((y_pred[valid_mask] == label_true_int[valid_mask]).mean())
        cm = _confusion_matrix(label_true_int[valid_mask], y_pred[valid_mask], n_classes=2)  # ### CHANGEMENT
        logger.info("Classification accuracy (π vs p): %.4f", acc)
        logger.info("Confusion matrix (rows=true, cols=pred):\n%s", cm)
        class_names = [cfg.label_to_name[i] for i in range(2)]  # ["π-", "p"]
        png_path = "confusion_matrix_pid_param_LGBM_pi_p.png"   # ### CHANGEMENT
        plot_confusion_matrix(cm, class_names, png_path)
        logger.info("Matrice de confusion sauvegardée: %s", png_path)
    else:
        logger.warning("Aucun label vrai reconnu pour la classification (PDG inattendus).")

    # --- Métriques énergie paramétrique (global + par vraie particule) ---
    if np.all(np.isfinite(E_pred)):
        rmse = mean_squared_error(E_true, E_pred, squared=False)
        mae  = mean_absolute_error(E_true, E_pred)
        r2   = r2_score(E_true, E_pred)
        sigma_rel = _sigma_over_E(E_true, E_pred)
        logger.info("Énergie paramétrique (global): RMSE=%.4f  MAE=%.4f  R²=%.4f  σ(E)/E=%.4f",
                    rmse, mae, r2, sigma_rel)

        for lbl_int, lbl_name in cfg.label_to_name.items():  # π, p
            m = (label_true_int == lbl_int)
            if np.any(m):
                rmse_c = mean_squared_error(E_true[m], E_pred[m], squared=False)
                mae_c  = mean_absolute_error(E_true[m], E_pred[m])
                r2_c   = r2_score(E_true[m], E_pred[m])
                s_c    = _sigma_over_E(E_true[m], E_pred[m])
                logger.info("Énergie paramétrique (true=%s): RMSE=%.4f  MAE=%.4f  R²=%.4f  σ(E)/E=%.4f",
                            lbl_name, rmse_c, mae_c, r2_c, s_c)
    else:
        logger.warning("Certaines prédictions d'énergie sont NaN (classes sans routeur ?).")
    
    # --- Résultat final ---
    out_df = pd.DataFrame({
        # vérité
        "PDG_true": pdg_true,
        "label_true_int": label_true_int,
        "label_true": label_true,
        "E_true_GeV": E_true,
        # classification (binaire)
        "label_pred_int": y_pred,
        "label_pred": class_name,
        "proba_pred": proba_max,
        "proba_pi":     y_proba_bin[:, 0],     # ### CHANGEMENT
        "proba_proton": y_proba_bin[:, 1],     # ### CHANGEMENT
        # énergie paramétrique
        "E_pred_GeV": E_pred,
        # debug utile
        "dE_over_E": (E_pred - E_true) / E_true,
    })

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    logger.info("Résultats écrits dans %s (%d lignes)", out_path, len(out_df))


def main():
    files = ["/gridgroup/ilc/midir/analyse/data/validation_set_pion_proton.root"]
    out_csv = "BDT_chi2.csv"  # ### CHANGEMENT
    run_inference(files, out_csv)


if __name__ == "__main__":
    main()
