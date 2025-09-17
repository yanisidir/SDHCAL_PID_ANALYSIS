#!/usr/bin/env python3
# noPID_BDT_pi_p.py
# Régression d'énergie via LGBM, routage par vérité PDG, étude binaire π vs p (kaons exclus)
# - Kaons retirés (mapping, modèles, sorties)
# - Filtrage des événements: PDG ∈ {-211, 2212}
# - Arguments hardcodés (plus d'argparse)

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
    tree_name = "tree"

    # Features utilisées pour la régression (doivent matcher l'entraînement)
    reg_features = [
        "nHough",
        "nCluster",
        "nMipCluster",
        "nTrack",
        "end",
        "density",
        "transverseRatio",
        "reconstructedCosTheta",
        "meanRadius",
        "first5LayersRMS",
        "propLastLayers",
        "nhitDensity",
        "nHitCustom",
        "nHit2Custom",
        "nHit3Custom",
        "Begin",
        "Radius",
        "Density",
        "ratioThr23",
        "Zbary",
        "Zrms",
        "PctHitsFirst10",
        "PlanesWithClusmore2",
        "MaxClustSize",
        "lambda1",
        "lambda2",
        "N3",
        "tMax",
        "tMean",
        "Nmax",
        "Xmax",
        "sumThrTotal",
        "eccentricity3D",
    ]

    # Colonnes vérité (présentes dans validation_set_pion_proton.root)
    energy_col = "primaryEnergy"
    pdg_col    = "primaryID"

    # ### CHANGEMENT: binaire π/p uniquement
    label_to_tag  = {0: "pi", 1: "proton"}
    label_to_name = {0: "π-", 1: "p"}
    pdg_to_label  = {-211: 0, 2212: 1}

    # --- Smearing temporel optionnel (régression) ---
    time_smear_sigma_reg: float | None = 0.1
    time_cols_reg = ["tMin"]

    # Clip de sécurité sur log(E) avant exp : [1e-6 GeV, 1e6 GeV]
    logE_min = np.log(1e-6)
    logE_max = np.log(1e6)


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

# ------------------------------ Inference (no PID) ---------------------------
def run_inference(files, out_csv):
    cfg = InferenceCfg()

    # === chemins hardcodés des régressseurs par particule ===
    # ### CHANGEMENT: kaon supprimé
    # reg_paths = {
    #     "pi":     (Path("/gridgroup/ilc/midir/analyse/Energy_reconstruction_ml/BDT/results_pi_energy_reco/models/lgbm_regressor_pi_32.joblib"),
    #                Path("/gridgroup/ilc/midir/analyse/Energy_reconstruction_ml/BDT/results_pi_energy_reco/models/scaler_pi_32.joblib")),
    #     "proton": (Path("/gridgroup/ilc/midir/analyse/Energy_reconstruction_ml/BDT/results_proton_energy_reco/models/lgbm_regressor_proton_24.joblib"),
    #                Path("/gridgroup/ilc/midir/analyse/Energy_reconstruction_ml/BDT/results_proton_energy_reco/models/scaler_proton_24.joblib")),
    # }

    reg_paths = {
        "pi":     (Path("/gridgroup/ilc/midir/analyse/Energy_reconstruction_ml/BDT/results_pi_energy_reco/models/lgbm_regressor_pi_32.joblib"),
                   Path("/gridgroup/ilc/midir/analyse/Energy_reconstruction_ml/BDT/results_pi_energy_reco/models/scaler_pi_32.joblib")),
        "proton": (Path("/gridgroup/ilc/midir/analyse/Energy_reconstruction_ml/BDT/results_pi_energy_reco/models/lgbm_regressor_pi_32.joblib"),
                   Path("/gridgroup/ilc/midir/analyse/Energy_reconstruction_ml/BDT/results_pi_energy_reco/models/scaler_pi_32.joblib")),
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

    # ### CHANGEMENT: ne garder que π/p
    df_all = df_all[df_all[cfg.pdg_col].isin([-211, 2212])].reset_index(drop=True)
    if len(df_all) == 0:
        logger.error("Après filtrage π/p, aucun événement disponible.")
        sys.exit(1)
    logger.info("Événements conservés (π/p uniquement): %d", len(df_all))

    # --- Smearing temporel optionnel (régression) ---
    df_all = smear_columns_gaussian(df_all, cfg.time_cols_reg, cfg.time_smear_sigma_reg, seed=42)

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
    for lbl_int, tag in cfg.label_to_tag.items():  # {0:"pi", 1:"proton"}
        mask = (label_true_int == lbl_int)
        n = int(mask.sum())
        if n == 0:
            continue
        reg, sc = regs[tag]
        Xr = sc.transform(Xr_full[mask])

        # SORTIE EN LOG(E) -> clip sécurité -> passage en espace linéaire
        logE = reg.predict(Xr)
        logE = np.clip(logE, cfg.logE_min, cfg.logE_max)
        E_pred[mask] = np.exp(logE).astype(np.float32)

        n_total_used += n
        logger.info("Régression (routeur vérité=%s): %d événements", tag, n)

    if n_total_used < len(df_all):
        logger.warning("Certains événements ont un PDG non reconnu: %d", len(df_all) - n_total_used)

    # --- Colonnes "classification" simulées pour compatibilité (== vérité), binaire ---
    y_pred = label_true_int.copy()
    class_name = label_true.copy()
    one_hot_pi     = (label_true_int == 0).astype(np.float32)
    one_hot_proton = (label_true_int == 1).astype(np.float32)
    proba_max = np.where(label_true_int >= 0, 1.0, np.nan).astype(np.float32)

    # --- Métriques "classification" triviales (2 classes) ---
    valid_mask = label_true_int >= 0
    if np.any(valid_mask):
        acc = float((y_pred[valid_mask] == label_true_int[valid_mask]).mean())
        cm = _confusion_matrix(label_true_int[valid_mask], y_pred[valid_mask], n_classes=2)  # ### CHANGEMENT
        logger.info("Classification simulée (π vs p): accuracy=%.4f", acc)
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

        for lbl_int, lbl_name in cfg.label_to_name.items():  # π, p
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

    # --- Résultat final (mêmes colonnes que la version avec PID mais sans kaon) ---
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
        "proba_pi":     one_hot_pi,
        "proba_proton": one_hot_proton,
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
    # ### CHANGEMENT: arguments hardcodés
    files = ["/gridgroup/ilc/midir/analyse/data/validation_set_pion_proton.root"]
    out_csv = "no_pid_BDT_pi_all.csv"
    run_inference(files, out_csv)


if __name__ == "__main__":
    main()
