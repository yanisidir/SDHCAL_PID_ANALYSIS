#!/usr/bin/env python3
# NN_NN_noPID.py
# Même sortie que la version avec PID, mais sans classif : routage des régressseurs par identité vraie (PDG).

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

    # Features utilisées pour la régression (doivent matcher l'entraînement)
    reg_features = [
        "Thr1","Thr2","Thr3","Begin","Radius","Density","NClusters","ratioThr23",
        "Zbary","Zrms","PctHitsFirst10","PlanesWithClusmore2","AvgClustSize",
        "MaxClustSize","lambda1","lambda2","N3","N2","N1","tMin","tMax","tMean",
        "tSpread","Nmax","z0_fit","Xmax","lambda","nTrackSegments","eccentricity3D",
        "nHitsTotal","sumThrTotal"
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


# ------------------------------ Inference (no PID) ---------------------------
def run_inference(files, out_csv):
    cfg = InferenceCfg()

    # === chemins hardcodés des régressseurs par particule ===
    reg_paths = {
        "pi":     (Path("/gridgroup/ilc/midir/analyse/Energy_Recostruction_ML/MLP/results_pi_energy_reco/models/mlp_regressor_pi_5.joblib"),
                   Path("/gridgroup/ilc/midir/analyse/Energy_Recostruction_ML/MLP/results_pi_energy_reco/models/scaler_pi_5.joblib")),
        "proton": (Path("/gridgroup/ilc/midir/analyse/Energy_Recostruction_ML/MLP/results_proton_energy_reco/models/mlp_regressor_proton_4.joblib"),
                   Path("/gridgroup/ilc/midir/analyse/Energy_Recostruction_ML/MLP/results_proton_energy_reco/models/scaler_proton_4.joblib")),
        "kaon":   (Path("/gridgroup/ilc/midir/analyse/Energy_Recostruction_ML/MLP/results_kaon_energy_reco/models/mlp_regressor_kaon_4.joblib"),
                   Path("/gridgroup/ilc/midir/analyse/Energy_Recostruction_ML/MLP/results_kaon_energy_reco/models/scaler_kaon_4.joblib")),
    }

    # Sanity checks (régression uniquement)
    for tag, (mp, sp) in reg_paths.items():
        _require_exists(mp, f"Modèle régression {tag}")
        _require_exists(sp, f"Scaler régression {tag}")

    # Colonnes nécessaires = features régression + vérité
    need_cols = sorted(set(cfg.reg_features).union([cfg.energy_col, cfg.pdg_col]))
    df_all = load_root_files(files, cfg.tree_name, need_cols)
    _require_columns(df_all, cfg.reg_features, "features régression")
    _require_columns(df_all, [cfg.energy_col, cfg.pdg_col], "vérité (E, PDG)")

    # --- Vérité : labels & énergie ---
    pdg_true = df_all[cfg.pdg_col].to_numpy()
    E_true   = df_all[cfg.energy_col].to_numpy(dtype=np.float32, copy=False)
    label_true_int = np.vectorize(lambda pdg: cfg.pdg_to_label.get(int(pdg), -1))(pdg_true)
    label_true = np.vectorize(lambda k: cfg.label_to_name.get(int(k), "UNK"))(label_true_int)

    # --- Précharge régressseurs/scalers ---
    regs: dict[str, tuple[object, object]] = {}
    for tag, (mpath, spath) in reg_paths.items():
        regs[tag] = (joblib.load(mpath), joblib.load(spath))

    # --- Régression énergie (routage par identité vraie) ---
    Xr_full = df_all[cfg.reg_features].to_numpy(dtype=np.float32, copy=False)
    E_pred = np.full(len(df_all), np.nan, dtype=np.float32)

    n_total_used = 0
    for lbl_int, tag in cfg.label_to_tag.items():
        mask = (label_true_int == lbl_int)
        n = int(mask.sum())
        if n == 0:
            continue
        reg, sc = regs[tag]
        Xr = sc.transform(Xr_full[mask])
        E_pred[mask] = reg.predict(Xr).astype(np.float32)
        n_total_used += n
        logger.info("Régression (routeur vérité=%s): %d événements", tag, n)

    if n_total_used < len(df_all):
        logger.warning("Certains événements ont un PDG non reconnu: %d", len(df_all) - n_total_used)

    # --- Colonnes "classification" simulées pour compatibilité (== vérité) ---
    y_pred = label_true_int.copy()
    class_name = label_true.copy()
    one_hot_pi     = (label_true_int == 0).astype(np.float32)
    one_hot_proton = (label_true_int == 1).astype(np.float32)
    one_hot_kaon   = (label_true_int == 2).astype(np.float32)
    proba_max = np.where(label_true_int >= 0, 1.0, np.nan).astype(np.float32)
    y_proba_pi, y_proba_proton, y_proba_kaon = one_hot_pi, one_hot_proton, one_hot_kaon

    # --- Métriques "classification" triviales (1.0 sur PDG reconnus) ---
    valid_mask = label_true_int >= 0
    if np.any(valid_mask):
        acc = float((y_pred[valid_mask] == label_true_int[valid_mask]).mean())
        cm = _confusion_matrix(label_true_int[valid_mask], y_pred[valid_mask], n_classes=3)
        logger.info("Classification (simulée, vérité): accuracy=%.4f", acc)
        logger.info("Confusion matrix (rows=true, cols=pred):\n%s", cm)
    else:
        logger.warning("Aucun label vrai reconnu (PDG inattendus).")

    # --- Métriques régression (global + par vraie particule) ---
    valid_E = np.isfinite(E_pred)
    if np.any(valid_E):
        rmse = mean_squared_error(E_true[valid_E], E_pred[valid_E], squared=False)
        mae  = mean_absolute_error(E_true[valid_E], E_pred[valid_E])
        r2   = r2_score(E_true[valid_E], E_pred[valid_E])
        sigma_rel = _sigma_over_E(E_true[valid_E], E_pred[valid_E])
        logger.info("Régression (global): RMSE=%.4f  MAE=%.4f  R²=%.4f  σ(E)/E=%.4f",
                    rmse, mae, r2, sigma_rel)

        for lbl_int, lbl_name in cfg.label_to_name.items():
            m = (label_true_int == lbl_int) & valid_E
            if np.any(m):
                rmse_c = mean_squared_error(E_true[m], E_pred[m], squared=False)
                mae_c  = mean_absolute_error(E_true[m], E_pred[m])
                r2_c   = r2_score(E_true[m], E_pred[m])
                s_c    = _sigma_over_E(E_true[m], E_pred[m])
                logger.info("Régression (true=%s): RMSE=%.4f  MAE=%.4f  R²=%.4f  σ(E)/E=%.4f",
                            lbl_name, rmse_c, mae_c, r2_c, s_c)
    else:
        logger.warning("Toutes les prédictions d'énergie sont NaN (PDG non reconnus ?).")

    # --- Résultat final (mêmes colonnes que la version avec PID) ---
    out_df = pd.DataFrame({
        # vérité
        "PDG_true": pdg_true,
        "label_true_int": label_true_int,
        "label_true": label_true,
        "E_true_GeV": E_true,
        # classification (simulée à partir de la vérité)
        "label_pred_int": y_pred,
        "label_pred": class_name,
        "proba_pred": proba_max,
        "proba_pi":     y_proba_pi,
        "proba_proton": y_proba_proton,
        "proba_kaon":   y_proba_kaon,
        # régression
        "E_pred_GeV": E_pred,
        # debug
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
    out_csv = "no_predictions_pid_energy.csv"

    run_inference(files, out_csv)


if __name__ == "__main__":
    main()
