#!/usr/bin/env python3
# PID_Inference_with_Parametric_Energy_LGBM.py
# Remplace le MLP par un classifieur LightGBM (BDT) et ajoute le smearing temporel optionnel.

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
# ------------------------------ Logging --------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ------------------------------ Config ---------------------------------------
class InferenceCfg:
    tree_name = "paramsTree"

    # Features utilisées pour la classification (doivent matcher l'entraînement)
    clf_features = [
        "Thr1","Thr2","Thr3","Begin","Radius","Density","NClusters","ratioThr23",
        "Zbary","Zrms","PctHitsFirst10","PlanesWithClusmore2","AvgClustSize",
        "MaxClustSize","lambda1","lambda2","tMin","tMax","tMean","tSpread",
        "Nmax","z0_fit","Xmax","lambda","nTrackSegments","eccentricity3D"
    ]

    # Colonnes nécessaires pour l'énergie paramétrique
    # (N1, N2, N3 sont requis; on garde aussi ces deux totaux si dispo)
    energy_cols = ["N1", "N2", "N3", "nHitsTotal", "sumThrTotal"]

    # Colonnes vérité (présentes dans val_set.root)
    energy_col = "primaryEnergy"
    pdg_col    = "particlePDG"

    # mapping label int -> nom + tag (conforme à l'entraînement: 0:π-, 1:p, 2:K0)
    label_to_tag  = {0: "pi", 1: "proton", 2: "kaon"}
    label_to_name = {0: "π-", 1: "p", 2: "K0"}

    # mapping PDG -> label int
    pdg_to_label  = {-211: 0, 2212: 1, 311: 2}

    # --- Smearing temporel optionnel (pour reproduire la résolution) ---
    # Mettre à None pour désactiver. Par défaut 0.1 (cohérent avec l'entraînement BDT précédent).
    time_smear_sigma_clf: float | None = 0.1
    time_cols_clf = ["tMin", "tMax", "tSpread"]


# ---------------- Paramètres figés (Par9) ----------------
# Chaque vecteur P contient 9 coefficients: (a0,a1,a2, b0,b1,b2, g0,g1,g2)
PAR_PI     = np.array([ 4.30851e-02,-3.50665e-05,1.94847e-08,
                        1.07201e-01,-6.36403e-05,1.20235e-08,
                        1.91862e-13,7.35489e-04,-3.00925e-07 ], dtype=np.float64)
PAR_KAON   = np.array([ 5.60298e-02,-4.58223e-05,1.90941e-08,
                        8.00358e-02,-7.76142e-05,4.58240e-08,
                        9.83127e-15,7.97643e-04,-3.56930e-07 ], dtype=np.float64)
PAR_PROTON = np.array([ 4.35267e-02,-1.89745e-05,1.07039e-08,
                        1.22286e-01,-4.93952e-05,5.45074e-09,
                        2.60494e-13,4.77430e-04,-1.68624e-07 ], dtype=np.float64)

# PAR_PI     = np.array([ 4.30851e-02,-3.50665e-05,1.94847e-08,
#                         1.07201e-01,-6.36403e-05,1.20235e-08,
#                         1.91862e-13,7.35489e-04,-3.00925e-07 ], dtype=np.float64)
# PAR_KAON   = np.array([ 4.30851e-02,-3.50665e-05,1.94847e-08,
#                         1.07201e-01,-6.36403e-05,1.20235e-08,
#                         1.91862e-13,7.35489e-04,-3.00925e-07 ], dtype=np.float64)
# PAR_PROTON = np.array([ 4.30851e-02,-3.50665e-05,1.94847e-08,
#                         1.07201e-01,-6.36403e-05,1.20235e-08,
#                         1.91862e-13,7.35489e-04,-3.00925e-07 ], dtype=np.float64)

PAR_BY_TAG = {
    "pi": PAR_PI,
    "kaon": PAR_KAON,
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

def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 3) -> np.ndarray:
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
    """Ajoute un bruit gaussien N(0, sigma) aux colonnes demandées (in-place).
    Si sigma est None, ne fait rien."""
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
    """Calcule E_reco = (a0+a1*N+a2*N^2)*N1 + (b0+b1*N+b2*N^2)*N2 + (g0+g1*N+g2*N^2)*N3,
    avec N = N1+N2+N3. Tout est vectorisé (dtype float64 pour la stabilité)."""
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
    # Normalisation ligne par ligne (%)
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
    plt.title("Matrice de confusion PID ")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


# ------------------------------ Inference ------------------------------------
def run_inference(files, out_csv):
    cfg = InferenceCfg()

    # === chemins  ===
    # Chemin vers le modèle LGBM entraîné pour le PID (multiclasse)
    clf_model_path  = Path("/gridgroup/ilc/midir/analyse/PID/BDT/results_with_time/models/lgbm_model_9.joblib")

    # Sanity checks (classification seulement)
    _require_exists(clf_model_path,  "Modèle classif LGBM")

    # Colonnes nécessaires = features classif + colonnes énergie paramétrique + vérité
    need_cols = sorted(set(cfg.clf_features).union(cfg.energy_cols).union([cfg.energy_col, cfg.pdg_col]))
    df_all = load_root_files(files, cfg.tree_name, need_cols)
    _require_columns(df_all, cfg.clf_features, "features classif")
    _require_columns(df_all, ["N1","N2","N3"], "énergie paramétrique (N1,N2,N3)")
    _require_columns(df_all, [cfg.energy_col, cfg.pdg_col], "vérité (E, PDG)")

    # --- Smearing temporel optionnel (classification) ---
    df_all = smear_columns_gaussian(df_all, cfg.time_cols_clf, cfg.time_smear_sigma_clf, seed=42)

    # --- Classification (LightGBM, pas de scaler) ---
    logger.info("Chargement classif LGBM (BDT)")
    clf: lgb.LGBMClassifier = joblib.load(clf_model_path)

    if hasattr(clf, "classes_"):
        classes = np.array(clf.classes_)
        if not (len(classes) == 3 and np.all(classes == np.array([0, 1, 2]))):
            logger.error("Ordre/valeurs des classes inattendues dans le classif LGBM: %s", classes)
            sys.exit(1)

    Xc = df_all[cfg.clf_features].to_numpy(dtype=np.float32, copy=False)
    # Les arbres n'ont pas besoin de scaling
    y_proba = clf.predict_proba(Xc)  # shape (N,3)
    y_pred = np.argmax(y_proba, axis=1)
    class_name = np.vectorize(cfg.label_to_name.get)(y_pred)
    proba_max = y_proba[np.arange(len(y_pred)), y_pred]

    # --- Énergie paramétrique (routage par classe prédite) ---
    N1 = df_all["N1"].to_numpy()
    N2 = df_all["N2"].to_numpy()
    N3 = df_all["N3"].to_numpy()

    E_pred = np.full(len(df_all), np.nan, dtype=np.float32)

    for label_int, tag in cfg.label_to_tag.items():
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

    # --- Métriques classification ---
    valid_mask = label_true_int >= 0
    if np.any(valid_mask):
        acc = float((y_pred[valid_mask] == label_true_int[valid_mask]).mean())
        cm = _confusion_matrix(label_true_int[valid_mask], y_pred[valid_mask], n_classes=3)
        logger.info("Classification accuracy (global): %.4f", acc)
        logger.info("Confusion matrix (rows=true, cols=pred):\n%s", cm)
        # Sauvegarde de la matrice de confusion en pourcentage (normalisée par ligne)
        class_names = [cfg.label_to_name[i] for i in range(3)]  # ["π-", "p", "K0"]
        png_path = "confusion_matrix_pid_param_LGBM.png"
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

        for lbl_int, lbl_name in cfg.label_to_name.items():
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
        # classification
        "label_pred_int": y_pred,
        "label_pred": class_name,
        "proba_pred": proba_max,
        "proba_pi":     y_proba[:, 0],
        "proba_proton": y_proba[:, 1],
        "proba_kaon":   y_proba[:, 2],
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
    # === chemins  ===
    files = [
        "/gridgroup/ilc/midir/analyse/data/val_set.root"
    ]

    out_csv = "pid_energy_param_LGBM.csv"

    run_inference(files, out_csv)


if __name__ == "__main__":
    main()
