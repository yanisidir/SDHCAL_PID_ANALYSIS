#!/usr/bin/env python3
# PID_Inference_with_Parametric_Energy_noPID.py
# Même sortie que la version PID, mais sans classif : on route par l'identité vraie (PDG).

from __future__ import annotations
import logging
from pathlib import Path
import sys
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

    # Colonnes nécessaires pour l'énergie paramétrique
    energy_cols = ["N1", "N2", "N3", "nHitsTotal", "sumThrTotal"]

    # Colonnes vérité (présentes dans val_set.root)
    energy_col = "primaryEnergy"
    pdg_col    = "particlePDG"

    # mapping label int -> nom + tag (0:π-, 1:p, 2:K0)
    label_to_tag  = {0: "pi", 1: "proton", 2: "kaon"}
    label_to_name = {0: "π-", 1: "p", 2: "K0"}

    # mapping PDG -> label int
    pdg_to_label  = {-211: 0, 2212: 1, 311: 2}

# ---------------- Paramètres figés (Par9) ----------------
# Chaque vecteur P contient 9 coefficients: (a0,a1,a2, b0,b1,b2, g0,g1,g2)

# PAR_PI     = np.array([ 4.30851e-02,-3.50665e-05,1.94847e-08,
#                         1.07201e-01,-6.36403e-05,1.20235e-08,
#                         1.91862e-13,7.35489e-04,-3.00925e-07 ], dtype=np.float64)
# PAR_KAON   = np.array([ 5.60298e-02,-4.58223e-05,1.90941e-08,
#                         8.00358e-02,-7.76142e-05,4.58240e-08,
#                         9.83127e-15,7.97643e-04,-3.56930e-07 ], dtype=np.float64)
# PAR_PROTON = np.array([ 4.35267e-02,-1.89745e-05,1.07039e-08,
#                         1.22286e-01,-4.93952e-05,5.45074e-09,
#                         2.60494e-13,4.77430e-04,-1.68624e-07 ], dtype=np.float64)

PAR_PI     = np.array([ 4.30851e-02,-3.50665e-05,1.94847e-08,
                        1.07201e-01,-6.36403e-05,1.20235e-08,
                        1.91862e-13,7.35489e-04,-3.00925e-07 ], dtype=np.float64)
PAR_KAON   = np.array([ 4.30851e-02,-3.50665e-05,1.94847e-08,
                        1.07201e-01,-6.36403e-05,1.20235e-08,
                        1.91862e-13,7.35489e-04,-3.00925e-07 ], dtype=np.float64)
PAR_PROTON = np.array([ 4.30851e-02,-3.50665e-05,1.94847e-08,
                        1.07201e-01,-6.36403e-05,1.20235e-08,
                        1.91862e-13,7.35489e-04,-3.00925e-07 ], dtype=np.float64)

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

def _sigma_over_E(y_true: np.ndarray, y_pred: np.ndarray) -> float:
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

# ------------------------------ Inference (no PID) ---------------------------
def run_inference(files, out_csv):
    cfg = InferenceCfg()

    # Colonnes nécessaires = énergie paramétrique + vérité
    need_cols = sorted(set(cfg.energy_cols).union([cfg.energy_col, cfg.pdg_col]))
    df_all = load_root_files(files, cfg.tree_name, need_cols)
    _require_columns(df_all, ["N1","N2","N3"], "énergie paramétrique (N1,N2,N3)")
    _require_columns(df_all, [cfg.energy_col, cfg.pdg_col], "vérité (E, PDG)")

    # --- Vérité : labels & énergie ---
    pdg_true = df_all[cfg.pdg_col].to_numpy()
    E_true   = df_all[cfg.energy_col].to_numpy(dtype=np.float32, copy=False)

    label_true_int = np.vectorize(lambda pdg: cfg.pdg_to_label.get(int(pdg), -1))(pdg_true)
    label_true = np.vectorize(lambda k: cfg.label_to_name.get(int(k), "UNK"))(label_true_int)

    # --- Énergie paramétrique (routage par identité vraie) ---
    N1 = df_all["N1"].to_numpy()
    N2 = df_all["N2"].to_numpy()
    N3 = df_all["N3"].to_numpy()

    E_pred = np.full(len(df_all), np.nan, dtype=np.float32)

    n_total_used = 0
    for lbl_int, tag in cfg.label_to_tag.items():
        mask = (label_true_int == lbl_int)
        n = int(mask.sum())
        if n == 0:
            continue
        P = PAR_BY_TAG[tag]
        E_pred[mask] = energy_parametric(N1[mask], N2[mask], N3[mask], P)
        n_total_used += n
        logger.info("Énergie paramétrique (routeur vérité=%s): %d événements", tag, n)

    if n_total_used < len(df_all):
        logger.warning("Certains événements ont un PDG non reconnu: %d", len(df_all) - n_total_used)

    # --- Métriques régression (global + par vraie particule) ---
    if np.any(np.isfinite(E_pred)):
        valid = np.isfinite(E_pred)
        rmse = mean_squared_error(E_true[valid], E_pred[valid], squared=False)
        mae  = mean_absolute_error(E_true[valid], E_pred[valid])
        r2   = r2_score(E_true[valid], E_pred[valid])
        sigma_rel = _sigma_over_E(E_true[valid], E_pred[valid])
        logger.info("Énergie paramétrique (global): RMSE=%.4f  MAE=%.4f  R²=%.4f  σ(E)/E=%.4f",
                    rmse, mae, r2, sigma_rel)

        for lbl_int, lbl_name in cfg.label_to_name.items():
            m = (label_true_int == lbl_int) & valid
            if np.any(m):
                rmse_c = mean_squared_error(E_true[m], E_pred[m], squared=False)
                mae_c  = mean_absolute_error(E_true[m], E_pred[m])
                r2_c   = r2_score(E_true[m], E_pred[m])
                s_c    = _sigma_over_E(E_true[m], E_pred[m])
                logger.info("Énergie paramétrique (true=%s): RMSE=%.4f  MAE=%.4f  R²=%.4f  σ(E)/E=%.4f",
                            lbl_name, rmse_c, mae_c, r2_c, s_c)
    else:
        logger.warning("Toutes les prédictions d'énergie sont NaN (PDG non reconnus ?).")

    # --- Colonnes de sortie (compatibles avec la version PID) ---
    # On remplit la "classification" avec la vérité (aucun classif utilisé).
    one_hot_pi     = (label_true_int == 0).astype(np.float32)
    one_hot_proton = (label_true_int == 1).astype(np.float32)
    one_hot_kaon   = (label_true_int == 2).astype(np.float32)

    out_df = pd.DataFrame({
        # vérité
        "PDG_true": pdg_true,
        "label_true_int": label_true_int,
        "label_true": label_true,
        "E_true_GeV": E_true,
        # "classification" simulée (== vérité)
        "label_pred_int": label_true_int,
        "label_pred": label_true,
        "proba_pred": np.where(label_true_int >= 0, 1.0, np.nan).astype(np.float32),
        "proba_pi":     one_hot_pi,
        "proba_proton": one_hot_proton,
        "proba_kaon":   one_hot_kaon,
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
    # === chemins (adapter si besoin) ===
    files = [
        "/gridgroup/ilc/midir/analyse/data/val_set.root"
    ]
    # out_csv = "no_pid_energy_param.csv"

    out_csv = "no_pid_energy_param_pion_to_all.csv"

    run_inference(files, out_csv)


if __name__ == "__main__":
    main()
