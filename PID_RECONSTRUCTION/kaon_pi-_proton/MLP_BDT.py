#!/usr/bin/env python3
# NN_NN.py

from __future__ import annotations
import logging
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
import uproot
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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

    # Features utilisées pour le classif (doivent matcher l'entraînement)
    clf_features = [
        "Thr1","Thr2","Thr3","Begin","Radius","Density","NClusters","ratioThr23",
        "Zbary","Zrms","PctHitsFirst10","PlanesWithClusmore2","AvgClustSize",
        "MaxClustSize","lambda1","lambda2","tMin","tMax","tMean","tSpread",
        "Nmax","z0_fit","Xmax","lambda","nTrackSegments","eccentricity3D"
    ]
    # Features utilisées pour la régression (doivent matcher l'entraînement)
    reg_features = [
        "Density", "NClusters", "Zbary", "AvgClustSize", "N3",  "N2", "N1", "tMin","nHitsTotal", "sumThrTotal"
    ]

    # Colonnes vérité (présentes dans val_set.root)
    energy_col = "primaryEnergy"
    pdg_col    = "particlePDG"

    # mapping label int -> nom + tag (conforme à l'entraînement: 0:π-, 1:p, 2:K0)
    label_to_tag  = {0: "pi", 1: "proton", 2: "kaon"}
    label_to_name = {0: "π-", 1: "p", 2: "K0"}

    # mapping PDG -> label int
    pdg_to_label  = {-211: 0, 2212: 1, 311: 2}


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
    # σ_rel := RMS((E_pred - E_true)/E_true)
    r = (y_pred - y_true) / y_true
    return float(np.sqrt(np.mean(r**2)))


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

# ------------------------------ Inference ------------------------------------
def run_inference(files, out_csv):
    cfg = InferenceCfg()

    # === chemins hardcodés ===
    clf_model_path  = Path("/gridgroup/ilc/midir/analyse/PID/MLP/results_with_time/models/mlp_model_6.joblib")
    clf_scaler_path = Path("/gridgroup/ilc/midir/analyse/PID/MLP/results_with_time/models/scaler_6.joblib")

    reg_paths = {
        "pi":     (Path("/gridgroup/ilc/midir/analyse/Energy_Recostruction_ML/BDT/results_pi_energy_reco/models/lgbm_regressor_pi_4.joblib"),
                   Path("/gridgroup/ilc/midir/analyse/Energy_Recostruction_ML/BDT/results_pi_energy_reco/models/scaler_pi_4.joblib")),
        "proton": (Path("/gridgroup/ilc/midir/analyse/Energy_Recostruction_ML/BDT/results_proton_energy_reco/models/lgbm_regressor_proton_4.joblib"),
                   Path("/gridgroup/ilc/midir/analyse/Energy_Recostruction_ML/BDT/results_proton_energy_reco/models/scaler_proton_4.joblib")),
        "kaon":   (Path("/gridgroup/ilc/midir/analyse/Energy_Recostruction_ML/BDT/results_kaon_energy_reco/models/lgbm_regressor_kaon_4.joblib"),
                   Path("/gridgroup/ilc/midir/analyse/Energy_Recostruction_ML/BDT/results_kaon_energy_reco/models/scaler_kaon_4.joblib")),
    }

    # Sanity checks
    _require_exists(clf_model_path,  "Modèle classif")
    _require_exists(clf_scaler_path, "Scaler classif")
    for tag, (mp, sp) in reg_paths.items():
        _require_exists(mp, f"Modèle régression {tag}")
        _require_exists(sp, f"Scaler régression {tag}")

    # Colonnes nécessaires = features + vérité
    need_cols = sorted(set(cfg.clf_features).union(cfg.reg_features).union([cfg.energy_col, cfg.pdg_col]))
    df_all = load_root_files(files, cfg.tree_name, need_cols)
    _require_columns(df_all, cfg.clf_features, "features classif")
    _require_columns(df_all, cfg.reg_features, "features régression")
    _require_columns(df_all, [cfg.energy_col, cfg.pdg_col], "vérité (E, PDG)")

    # --- Classification ---
    logger.info("Chargement classif")
    clf = joblib.load(clf_model_path)
    clf_scaler = joblib.load(clf_scaler_path)

    if hasattr(clf, "classes_"):
        classes = np.array(clf.classes_)
        if not (len(classes) == 3 and np.all(classes == np.array([0, 1, 2]))):
            logger.error("Ordre/valeurs des classes inattendues dans le classif: %s", classes)
            sys.exit(1)

    Xc = df_all[cfg.clf_features].to_numpy(dtype=np.float32, copy=False)
    Xc = clf_scaler.transform(Xc)
    y_proba = clf.predict_proba(Xc)  # shape (N,3)
    y_pred = np.argmax(y_proba, axis=1)
    class_name = np.vectorize(cfg.label_to_name.get)(y_pred)
    proba_max = y_proba[np.arange(len(y_pred)), y_pred]

    # --- Précharge régressseurs/scalers ---
    regs: dict[str, tuple[object, object]] = {}
    for tag, (mpath, spath) in reg_paths.items():
        regs[tag] = (joblib.load(mpath), joblib.load(spath))

    # --- Régression énergie (routage) ---
    Xr_full = df_all[cfg.reg_features].to_numpy(dtype=np.float32, copy=False)
    E_pred = np.full(len(df_all), np.nan, dtype=np.float32)
    for label_int, tag in cfg.label_to_tag.items():
        mask = (y_pred == label_int)
        n = int(mask.sum())
        if n == 0:
            continue
        reg, sc = regs[tag]
        Xr = sc.transform(Xr_full[mask])
        E_pred[mask] = reg.predict(Xr).astype(np.float32)
        logger.info("Régression %s: %d événements", tag, n)

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
    else:
        logger.warning("Aucun label vrai reconnu pour la classification (PDG inattendus).")

    # --- Métriques régression (global + par vraie particule) ---
    if np.all(np.isfinite(E_pred)):
        rmse = mean_squared_error(E_true, E_pred, squared=False)
        mae  = mean_absolute_error(E_true, E_pred)
        r2   = r2_score(E_true, E_pred)
        sigma_rel = _sigma_over_E(E_true, E_pred)
        logger.info("Régression (global): RMSE=%.4f  MAE=%.4f  R²=%.4f  σ(E)/E=%.4f",
                    rmse, mae, r2, sigma_rel)

        for lbl_int, lbl_name in cfg.label_to_name.items():
            m = (label_true_int == lbl_int)
            if np.any(m):
                rmse_c = mean_squared_error(E_true[m], E_pred[m], squared=False)
                mae_c  = mean_absolute_error(E_true[m], E_pred[m])
                r2_c   = r2_score(E_true[m], E_pred[m])
                s_c    = _sigma_over_E(E_true[m], E_pred[m])
                logger.info("Régression (true=%s): RMSE=%.4f  MAE=%.4f  R²=%.4f  σ(E)/E=%.4f",
                            lbl_name, rmse_c, mae_c, r2_c, s_c)
    else:
        logger.warning("Certaines prédictions d'énergie sont NaN (classes sans routeur régression ?).")

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
        # régression
        "E_pred_GeV": E_pred,
        # petit debug utile
        "dE_over_E": (E_pred - E_true) / E_true,
    })

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    logger.info("Résultats écrits dans %s (%d lignes)", out_path, len(out_df))


def main():
    # === chemins hardcodés ===
    files = [
        "/gridgroup/ilc/midir/analyse/data/val_set.root"
    ]
    out_csv = "predictions_pid_energy_BDT.csv"

    run_inference(files, out_csv)


if __name__ == "__main__":
    main()
