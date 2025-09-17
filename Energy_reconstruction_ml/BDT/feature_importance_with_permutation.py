#!/usr/bin/env python3
"""
feature_importance_LGBM_energy_hardcoded.py

Version sans argparse (arguments "hardcodés"):
 - PARTICLES : liste des particules à traiter ("pi", "kaon", "proton")
 - NB_PERM   : nombre de répétitions pour permutation_importance
 - MODEL_PATH / SCALER_PATH : chemins explicites (optionnels) ; si None,
   le script détecte automatiquement le dernier run via les CSV/globs.

Sorties :
 - Matrice de corrélation des features
 - Importances natives LightGBM
 - Importances par permutation

Tout est sauvegardé dans results_*_energy_reco/plots/.
"""

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
import uproot
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

# ---------------------------- "Arguments" figés -------------------------------
# Particules à traiter (équivalent de --particles all)
# PARTICLES: List[str] = ["pi", "kaon", "proton"]
PARTICLES: List[str] = ["pi", "proton"]

# Nombre de répétitions pour permutation_importance (équivalent de --nb-perm)
NB_PERM: int = 10

# Si vous souhaitez forcer un modèle/scaler précis (uniquement si une seule particule)
# indiquez ci-dessous des chemins valides ; sinon laissez None pour auto-détection.
MODEL_PATH: Optional[Path] = None
SCALER_PATH: Optional[Path] = None

# ---------------------------- Logging ----------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------- Dataclass (alignée) ----------------------------
@dataclass
class GlobalCfg:
    tree_name: str = "tree"
    energy_col: str = "primaryEnergy"

    # Même liste que dans hadron_energy_regressor_LGBM_all.py
    feature_cols: Sequence[str] = field(default_factory=lambda: [
        "nHough",
        "nHough1",
        "nHough2",
        "nHough3",
        "nLayer",
        "nInteractingLayer",
        "nCluster",
        "nMipCluster",
        "nTrack",
        "ntracksClusterSize",
        "ntracksClusterNumber",
        "begin",
        "end",
        "density",
        "transverseRatio",
        "reconstructedCosTheta",
        "meanRadius",
        "nlongiProfile",
        "nradiProfile",
        "first5LayersRMS",
        "propLastLayers",
        "nhitDensity",
        "nHitCustom",
        "nHit1Custom",
        "nHit2Custom",
        "nHit3Custom",
        "Thr1",
        "Thr2",
        "Thr3",
        "Begin",
        "Radius",
        "Density",
        "NClusters",
        "ratioThr23",
        "Zbary",
        "Zrms",
        "PctHitsFirst10",
        "PlanesWithClusmore2",
        "AvgClustSize",
        "MaxClustSize",
        "lambda1",
        "lambda2",
        "N3",
        "N2",
        "N1",
        "tMin",
        "tMax",
        "tMean",
        "tSpread",
        "Nmax",
        "z0_fit",
        "Xmax",
        "lambda",
        "nTrackSegments",
        "eccentricity3D",
        "nHitsTotal",
        "sumThrTotal",
    ])

    test_size: float = 0.20
    random_state: int = 42

# --------------------- Configs particules (alignées) -------------------------
PARTICLE_CFG: Dict[str, Dict[str, Path | str]] = {
    "pi": {
        "file_path": "/gridgroup/ilc/midir/analyse/data/merged_primaryEnergy/200k_pi-_1-130_params_merged.root",
        "res_dir":   Path("results_pi_energy_reco"),
        "param_csv": Path("run_parameters_regression_pi.csv"),
        "tag":       "pi",
    },
    # "kaon": {
    #     "file_path": "/gridgroup/ilc/midir/analyse/data/merged_primaryEnergy/130k_kaon_E1to130_params_merged.root",
    #     "res_dir":   Path("results_kaon_energy_reco"),
    #     "param_csv": Path("run_parameters_regression_kaon.csv"),
    #     "tag":       "kaon",
    # },
    "proton": {
        "file_path": "/gridgroup/ilc/midir/analyse/data/merged_primaryEnergy/200k_proton_1-130_params_merged.root",
        "res_dir":   Path("results_proton_energy_reco"),
        "param_csv": Path("run_parameters_regression_proton.csv"),
        "tag":       "proton",
    },
}

# ---------------------------- Utilitaires I/O --------------------------------
def load_root(file: str | Path, tree: str, cols: Sequence[str]) -> pd.DataFrame:
    path = Path(file)
    logger.info("Lecture ROOT: %s", path)
    if not path.is_file():
        raise FileNotFoundError(f"Fichier introuvable: {path}")
    with uproot.open(path) as f:
        available = set(f[tree].keys())
        need = [c for c in cols if c in available]
        missing = [c for c in cols if c not in available]
        if missing:
            logger.warning("Colonnes manquantes ignorées (%d): %s", len(missing), ", ".join(missing))
        df = f[tree].arrays(need, library="pd")
    logger.info("Événements chargés: %d", len(df))
    return df

def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.replace([np.inf, -np.inf], pd.NA).dropna().reset_index(drop=True)
    logger.info("Après nettoyage: %d (supprimé %d)", len(df), before - len(df))
    return df

def ensure_dirs(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

# ----------------- Détection du dernier run & chemins modèles ----------------
def last_run_id_from_csv(csv_path: Path) -> Optional[int]:
    if not csv_path.exists():
        return None
    try:
        with csv_path.open() as f:
            reader = csv.DictReader(f)
            run_ids = [int(r["run_id"]) for r in reader if r.get("run_id", "").isdigit()]
        return max(run_ids) if run_ids else None
    except Exception as e:
        logger.warning("Impossible de lire %s (%s).", csv_path, e)
        return None

def guess_paths_for_run(res_dir: Path, tag: str, run_id: int) -> Tuple[Path, Path]:
    model = res_dir / "models" / f"lgbm_regressor_{tag}_{run_id}.joblib"
    scaler = res_dir / "models" / f"scaler_{tag}_{run_id}.joblib"
    return model, scaler

def fallback_latest_by_glob(res_dir: Path, tag: str) -> Tuple[Optional[Path], Optional[Path], Optional[int]]:
    models = sorted((res_dir / "models").glob(f"lgbm_regressor_{tag}_*.joblib"))
    scalers = sorted((res_dir / "models").glob(f"scaler_{tag}_*.joblib"))
    if not models or not scalers:
        return None, None, None
    def extract_run_id(p: Path) -> Optional[int]:
        try:
            return int(p.stem.split("_")[-1])
        except Exception:
            return None
    ids = [(m, extract_run_id(m)) for m in models]
    ids = [t for t in ids if t[1] is not None]
    if not ids:
        return models[-1], scalers[-1], None
    best_m, best_id = max(ids, key=lambda t: t[1])
    best_s = res_dir / "models" / f"scaler_{tag}_{best_id}.joblib"
    if best_s.exists():
        return best_m, best_s, best_id
    return models[-1], scalers[-1], None

def resolve_model_and_scaler(
    particle: str,
    explicit_model: Optional[Path],
    explicit_scaler: Optional[Path],
) -> Tuple[Path, Path, Optional[int]]:
    pconf = PARTICLE_CFG[particle]
    res_dir: Path = pconf["res_dir"]  # type: ignore
    tag: str = pconf["tag"]           # type: ignore

    if explicit_model and explicit_scaler:
        return explicit_model, explicit_scaler, None

    run_id = last_run_id_from_csv(pconf["param_csv"])  # type: ignore
    if run_id is not None:
        model, scaler = guess_paths_for_run(res_dir, tag, run_id)
        if model.exists() and scaler.exists():
            return model, scaler, run_id

    model, scaler, rid = fallback_latest_by_glob(res_dir, tag)
    if model and scaler:
        return model, scaler, rid

    raise FileNotFoundError(
        f"Aucun modèle/scaler détecté pour '{particle}'. "
        f"Renseignez MODEL_PATH et SCALER_PATH pour forcer un run précis."
    )

# ------------------------------ Traces & plots -------------------------------
def plot_correlation(df: pd.DataFrame, features: Sequence[str], out_path: Path) -> None:
    corr = df[features].corr()
    plt.figure(figsize=(15, 12))
    sns.heatmap(corr, annot=False, cmap="coolwarm",
                xticklabels=features, yticklabels=features)
    plt.title("Feature Correlation Matrix")
    plt.xticks(rotation=90); plt.yticks(rotation=0)
    plt.tight_layout()
    ensure_dirs(out_path)
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info("Corrélation sauvegardée: %s", out_path)

def save_importances_barh(names: Sequence[str], values: np.ndarray, title: str, out_path: Path) -> None:
    order = np.argsort(values)
    plt.figure(figsize=(10, 8))
    plt.barh(np.array(names)[order], np.array(values)[order])
    plt.xlabel("Importance")
    plt.title(title)
    plt.tight_layout()
    ensure_dirs(out_path)
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info("Plot importances sauvegardé: %s", out_path)

# ------------------------------ Cœur pipeline --------------------------------
def run_for_particle(
    particle: str,
    cfg: GlobalCfg,
    model_path: Optional[Path],
    scaler_path: Optional[Path],
    nb_perm: int,
) -> None:
    pconf = PARTICLE_CFG[particle]
    res_dir: Path = pconf["res_dir"]  # type: ignore
    tag: str = pconf["tag"]           # type: ignore

    model_p, scaler_p, run_id = resolve_model_and_scaler(
        particle, model_path, scaler_path
    )
    logger.info("Modèle: %s", model_p)
    logger.info("Scaler: %s", scaler_p)

    cols = list(cfg.feature_cols) + [cfg.energy_col]
    df = load_root(pconf["file_path"], cfg.tree_name, cols)  # type: ignore
    df = clean_df(df)

    feat_present = [c for c in cfg.feature_cols if c in df.columns]
    if not feat_present:
        raise RuntimeError("Aucune feature disponible après lecture du ROOT.")
    if len(feat_present) < len(cfg.feature_cols):
        logger.warning("Seulement %d/%d features présentes.", len(feat_present), len(cfg.feature_cols))

    X = df[feat_present].values.astype(np.float32)
    y = df[cfg.energy_col].values.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state
    )

    scaler = joblib.load(scaler_p)
    reg = joblib.load(model_p)

    X_test_s = scaler.transform(X_test)

    suffix = f"_{run_id}" if run_id is not None else ""

    # corr_path = res_dir / "plots" / f"correlation_matrix{suffix}.png"
    # plot_correlation(df, feat_present, corr_path)

    if not hasattr(reg, "feature_importances_"):
        raise AttributeError("Le modèle chargé n'expose pas 'feature_importances_'.")
    native_imp = np.asarray(reg.feature_importances_, dtype=float)

    imp_df = pd.DataFrame({"feature": feat_present, "importance": native_imp}).sort_values(
        "importance", ascending=False
    )
    imp_csv = res_dir / "plots" / f"feature_importances{suffix}.csv"
    ensure_dirs(imp_csv)
    imp_df.to_csv(imp_csv, index=False)
    save_importances_barh(
        feat_present, native_imp, "LGBM Feature Importances",
        res_dir / "plots" / f"feature_importances{suffix}.png"
    )

    perm = permutation_importance(
        reg, X_test_s, y_test,
        n_repeats=nb_perm, random_state=cfg.random_state, n_jobs=-1
    )
    perm_mean = perm.importances_mean
    perm_df = pd.DataFrame({"feature": feat_present, "perm_importance": perm_mean}).sort_values(
        "perm_importance", ascending=False
    )
    perm_csv = res_dir / "plots" / f"permutation_importances{suffix}.csv"
    ensure_dirs(perm_csv)
    perm_df.to_csv(perm_csv, index=False)
    save_importances_barh(
        feat_present, perm_mean, "Permutation Importances (R² drop)",
        res_dir / "plots" / f"permutation_importances{suffix}.png"
    )

    logger.info("Terminé pour %s. Sorties écrites dans %s/plots", particle, res_dir)

# ----------------------------------- Main ------------------------------------
def main():
    cfg = GlobalCfg()

    parts = PARTICLES.copy()
    # Si plusieurs particules, les chemins explicites ne sont pas utilisés.
    explicit_model = MODEL_PATH if len(parts) == 1 else None
    explicit_scaler = SCALER_PATH if len(parts) == 1 else None
    if len(parts) > 1 and (MODEL_PATH or SCALER_PATH):
        logger.warning("Plusieurs particules : MODEL_PATH/SCALER_PATH explicites seront ignorés.")

    for part in parts:
        if part not in PARTICLE_CFG:
            logger.error("Particule inconnue '%s' – on passe.", part)
            continue
        logger.info("=== Importances pour %s ===", part)
        run_for_particle(
            particle=part,
            cfg=cfg,
            model_path=explicit_model,
            scaler_path=explicit_scaler,
            nb_perm=NB_PERM,
        )

if __name__ == "__main__":
    main()
