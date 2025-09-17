#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyse_corr_permutation.py

- Charge les fichiers ROOT et sélectionne les features.
- Calcule la matrice de corrélation (Pearson) et l'exporte (CSV + heatmap PNG).
- Entraîne un LightGBM simple (multiclasse) et calcule les importances par permutation
  sur un jeu de test (CSV + barplot PNG).

"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence, Iterable, Dict, Optional, Union

import numpy as np
import pandas as pd
import uproot
import joblib
import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score

import matplotlib
matplotlib.use("Agg")  # pour les environnements sans display
import matplotlib.pyplot as plt
import seaborn as sns


# =============================================================================
# Config
# =============================================================================

@dataclass
class Config:

    # I/O
    # file_paths: List[Path] = field(default_factory=lambda: [
    #     Path("/gridgroup/ilc/midir/analyse/data/params/130k_pi_E1to130_params.root"),
    #     Path("/gridgroup/ilc/midir/analyse/data/params/130k_kaon_E1to130_params.root"),
    #     Path("/gridgroup/ilc/midir/analyse/data/params/130k_proton_E1to130_params.root"),
    # ])
    # tree_name: str = "paramsTree"

    file_paths: List[Path] = field(default_factory=lambda: [
        Path("/gridgroup/ilc/midir/analyse/data/merged_primaryEnergy/200k_pi-_1-130_params_merged.root"),
        Path("/gridgroup/ilc/midir/analyse/data/merged_primaryEnergy/200k_proton_1-130_params_merged.root"),
    ])
    tree_name: str = "tree"

    # targets: Sequence[int] = field(default_factory=lambda: [-211, 2212, 311])  # pi-, proton, K0

    targets: Sequence[int] = field(default_factory=lambda: [-211, 2212])  # pi-, proton, K0
    
    # feature_cols: Sequence[str] = field(default_factory=lambda: [
    #     "Thr1", "Thr2", "Thr3", "Begin", "Radius", "Density", "NClusters", "ratioThr23", "Zbary", "Zrms",
    #     "PctHitsFirst10", "PlanesWithClusmore2", "AvgClustSize", "MaxClustSize", "lambda1", "lambda2",
    #     "tMin", "tMax", "tMean", "tSpread", "Nmax", "z0_fit", "Xmax", "lambda",
    #     "nTrackSegments", "eccentricity3D"
    # ])

    feature_cols: Sequence[str] = field(default_factory=lambda: [
        "Thr1", "Thr2", "Thr3", 
        "Begin", "meanRadius", "nMipCluster", "first5LayersRMS",
        "Density", "NClusters", "ratioThr23", "Zbary", "Zrms",
        "PctHitsFirst10", "AvgClustSize", "MaxClustSize", "lambda1", "lambda2",
        "tMin", "tMax", "tMean", "tSpread", "Nmax", "Xmax",
        "eccentricity3D", "transverseRatio", "nTrack"
    ])

    results_dir: Path = Path("results_with_time")
    random_state: int = 42
    test_size: float = 0.2

    # Modèle 
    learning_rate: float = 0.10         
    n_estimators: int = 500              
    num_leaves: int = 63
    max_depth: int = -1
    reg_alpha: float = 0.0
    reg_lambda: float = 0.0
    min_child_samples: int = 30
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    class_weight: Optional[Union[str, Dict[int, float]]] = None

    # Options de perf
    use_gpu: bool = False                # passe à True si LightGBM GPU dispo
    early_stopping_rounds: int = 50
    valid_size: float = 0.1              # part du train pour la validation

    # Permutation importance
    perm_n_repeats: int = 5              # 10 -> 5
    perm_test_subsample: int = 50000  


# =============================================================================
# Logging
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("corr_perm")


# =============================================================================
# Utils I/O
# =============================================================================

def ensure_dirs(base: Path) -> Dict[str, Path]:
    paths = {
        "plots": base / "plots",
        "tables": base / "tables",
        "models": base / "models",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


# =============================================================================
# Data
# =============================================================================

def load_root(files: Iterable[Path], tree: str, cols: Sequence[str]) -> pd.DataFrame:
    dfs = []
    logger.info("Lecture des ROOT...")
    for p in files:
        if not p.is_file():
            logger.warning("Fichier manquant: %s", p)
            continue
        with uproot.open(p) as f:
            dfs.append(f[tree].arrays(cols, library="pd"))
    if not dfs:
        raise RuntimeError("Aucun fichier ROOT valide trouvé.")
    df = pd.concat(dfs, ignore_index=True)
    logger.info("Total événements chargés: %d", len(df))
    return df


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    logger.info("Après nettoyage: %d (supprimé %d)", len(df), before - len(df))
    return df


def filter_targets(df: pd.DataFrame, targets: Sequence[int]) -> pd.DataFrame:
    df = df[df["primaryID"].isin(targets)].copy()
    if df.empty:
        raise RuntimeError("Aucune entrée après filtrage PDG.")
    label_map = {pdg: idx for idx, pdg in enumerate(targets)}
    df["label"] = df["primaryID"].map(label_map)
    binc = np.bincount(df["label"])
    logger.info("Répartition des classes: %s", binc)
    return df

def smear_column_gaussian(df: pd.DataFrame, col: str, sigma: float, random_state: Optional[int] = None) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    df[col] = df[col].to_numpy() + rng.normal(0, sigma, size=len(df))
    return df
    
# =============================================================================
# Corrélation
# =============================================================================

def compute_and_save_correlation(df_feats: pd.DataFrame, out_csv: Path, out_png: Path) -> None:
    corr = df_feats.corr(method="pearson")
    corr.to_csv(out_csv, index=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0, square=True, cbar=True)
    plt.title("Matrice de corrélation (Pearson)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    logger.info("Corrélation: CSV -> %s | Heatmap -> %s", out_csv, out_png)


# =============================================================================
# Modèle + importances par permutation
# =============================================================================

def build_lgbm(cfg: Config) -> lgb.LGBMClassifier:
    extra = {
        "force_col_wise": True,  # supprime l'overhead détecté
        "n_jobs": -1,
        "random_state": cfg.random_state,
    }
    if cfg.use_gpu:
        # Nécessite LightGBM compilé avec GPU
        extra.update({"device_type": "gpu"})
    return lgb.LGBMClassifier(
        objective="binary",
        learning_rate=cfg.learning_rate,
        n_estimators=cfg.n_estimators,
        num_leaves=cfg.num_leaves,
        max_depth=cfg.max_depth,
        reg_alpha=cfg.reg_alpha,
        reg_lambda=cfg.reg_lambda,
        min_child_samples=cfg.min_child_samples,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        class_weight=cfg.class_weight,
        **extra
    )


def train_and_perm_importance(
    X: np.ndarray, y: np.ndarray, feature_names: Sequence[str], cfg: Config,
    out_csv: Path, out_png: Path, model_out: Path
) -> None:
    # split train/test
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y
    )

    # split train/valid pour early stopping
    X_tr_in, X_val, y_tr_in, y_val = train_test_split(
        X_tr, y_tr, test_size=cfg.valid_size, random_state=cfg.random_state, stratify=y_tr
    )

    clf = build_lgbm(cfg)
    clf.fit(
        X_tr_in, y_tr_in,
        eval_set=[(X_val, y_val)],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(stopping_rounds=cfg.early_stopping_rounds, verbose=False)]
    )

    best_it = getattr(clf, "best_iteration_", None)
    if best_it is not None:
        logger.info("Early stopping: best_iteration = %d", best_it)

    # évalue et sauvegarde
    y_pred = clf.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    joblib.dump(clf, model_out)
    logger.info("Modèle entraîné. Accuracy test: %.4f | modèle -> %s", acc, model_out)

    # --- Importances LightGBM (rapides) ---
    gain_importances = clf.booster_.feature_importance(importance_type="gain")
    gain_df = pd.DataFrame({
        "feature": feature_names,
        "gain_importance": gain_importances
    }).sort_values("gain_importance", ascending=False)
    gain_csv = out_csv.with_name(out_csv.stem + "_lgbm_gain.csv")
    gain_png = out_png.with_name(out_png.stem + "_lgbm_gain.png")
    gain_df.to_csv(gain_csv, index=False)

    plt.figure(figsize=(8, max(4, 0.35 * len(feature_names))))
    plt.barh(gain_df["feature"], gain_df["gain_importance"])
    plt.gca().invert_yaxis()
    plt.xlabel("Importance (gain LightGBM)")
    plt.title("Importances (gain) — rapides")
    plt.tight_layout()
    plt.savefig(gain_png, dpi=150)
    plt.close()
    logger.info("Importances gain: CSV -> %s | Figure -> %s", gain_csv, gain_png)

    # --- Importances par permutation (plus lentes) ---
    logger.info("Calcul des importances par permutation (échantillon + %d répétitions)...", cfg.perm_n_repeats)

    # sous-échantillon du test pour accélérer
    if cfg.perm_test_subsample is not None and len(X_te) > cfg.perm_test_subsample:
        rng = np.random.default_rng(cfg.random_state)
        idx_sub = rng.choice(len(X_te), size=cfg.perm_test_subsample, replace=False)
        X_perm = X_te[idx_sub]
        y_perm = y_te[idx_sub]
    else:
        X_perm, y_perm = X_te, y_te

    result = permutation_importance(
        clf, X_perm, y_perm,
        scoring="accuracy",
        n_repeats=cfg.perm_n_repeats,
        random_state=cfg.random_state,
        n_jobs=-1
    )

    idx = np.argsort(result.importances_mean)[::-1]
    ordered_features = np.array(feature_names)[idx]
    mean_imp = result.importances_mean[idx]
    std_imp = result.importances_std[idx]

    imp_df = pd.DataFrame({
        "feature": ordered_features,
        "mean_importance": mean_imp,
        "std_importance": std_imp
    })
    imp_df.to_csv(out_csv, index=False)
    logger.info("Importances permutation: CSV -> %s", out_csv)

    plt.figure(figsize=(8, max(4, 0.35 * len(ordered_features))))
    plt.barh(ordered_features, mean_imp, xerr=std_imp)
    plt.gca().invert_yaxis()
    plt.xlabel("Importance par permutation (moyenne ± écart-type)")
    plt.title(f"Importances par permutation (accuracy, n_repeats={cfg.perm_n_repeats})")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    logger.info("Importances permutation: Figure -> %s", out_png)

# =============================================================================
# Main
# =============================================================================

def main():
    cfg = Config()
    paths = ensure_dirs(cfg.results_dir)

    # Chargement
    cols_to_read = list(cfg.feature_cols) + ["primaryID"]
    df = load_root(cfg.file_paths, cfg.tree_name, cols_to_read)
    df = clean_df(df)
    df = filter_targets(df, cfg.targets)
    df = smear_column_gaussian(df, "tMin", sigma=0.1, random_state=cfg.random_state)
    df = smear_column_gaussian(df, "tMax", sigma=0.1, random_state=cfg.random_state)
    df = smear_column_gaussian(df, "tSpread", sigma=0.1, random_state=cfg.random_state)

    # Corrélation sur toutes les features (toutes classes confondues)
    df_feats = df[cfg.feature_cols].copy()
    compute_and_save_correlation(
        df_feats,
        out_csv=paths["tables"] / "correlation_matrix_pp.csv",
        out_png=paths["plots"] / "correlation_matrix_pp.png",
    )

    # Importances par permutation (modèle entraîné rapidement)
    X = df_feats.to_numpy()
    y = df["label"].to_numpy()
    train_and_perm_importance(
        X, y, cfg.feature_cols, cfg,
        out_csv=paths["tables"] / "permutation_importances_pp.csv",
        out_png=paths["plots"] / "permutation_importances_pp.png",
        model_out=paths["models"] / "lgbm_for_permutation_pp.joblib",
    )

    logger.info("Terminé. Résultats dans: %s", cfg.results_dir.resolve())


if __name__ == "__main__":
    main()
