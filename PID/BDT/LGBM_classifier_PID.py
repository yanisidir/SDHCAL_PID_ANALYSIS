#!/usr/bin/env python3
"""
hadron_classifier_LGBM.py

Classification hadronique (π⁻, p, K⁰) avec LightGBM.
- Données lues depuis plusieurs fichiers ROOT (uproot) et concaténées
- Nettoyage (±inf → NA, dropna), filtrage des PDG cibles et mappage en labels
- Split stratifié train/test puis train/val (sur la partie train)
- SMOTE appliqué uniquement au train (pas de StandardScaler : modèles en arbres)
- Entraînement LightGBM multiclass avec early stopping (multi_logloss)
- Option GridSearchCV (StratifiedKFold) pour rechercher les hyperparamètres
- Évaluation sur test : accuracy, AUC (OVR), multi-logloss + matrice de confusion
- Figures : courbe train/valid (logloss), confusion matrix, histogrammes de probabilités
- Sauvegardes :
    * Modèle .joblib dans results_with_time/models/
    * Plots dans results_with_time/plots/
    * Logs CSV : paramètres (run_parameters.csv), performances (hadron_performances.csv), commentaires (run_comments.csv)
    * Dump de X_test (CSV) pour analyses SHAP ultérieures

Dépendances : numpy, pandas, uproot, lightgbm, scikit-learn, imbalanced-learn, joblib, matplotlib, seaborn.

Usage rapide :
    python hadron_classifier_LGBM.py
(les chemins d’entrée, features, et hyperparamètres se configurent dans la dataclass Config)
"""

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence, Tuple, Dict, Iterable, Union, Optional

import joblib
import numpy as np
import pandas as pd
import uproot
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_auc_score, log_loss
)
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Config:
    """Pipeline settings and hyper-parameters."""
    # I/O (hardcodé)
    file_paths: List[Path] = field(default_factory=lambda: [
        Path("/gridgroup/ilc/midir/analyse/data/params/130k_pi_E1to130_params.root"),
        Path("/gridgroup/ilc/midir/analyse/data/params/130k_kaon_E1to130_params.root"),
        Path("/gridgroup/ilc/midir/analyse/data/params/130k_proton_E1to130_params.root"),
    ])
    tree_name: str = "paramsTree"

    # Entraînement
    grid_search: bool = False     # mettre True pour activer la grille
    test_size: float = 0.20
    val_size: float = 0.20        # fraction du TRAIN (après split test) réservée à la validation
    random_state: int = 42

    # Cibles et features
    targets: Sequence[int] = field(default_factory=lambda: [-211, 2212, 311])  # pi-, proton, K0
    feature_cols: Sequence[str] = field(default_factory=lambda: [
        "Thr1", "Thr2", "Thr3", "Begin", "Radius", "Density", "NClusters", "ratioThr23", "Zbary", "Zrms",
        "PctHitsFirst10", "PlanesWithClusmore2", "AvgClustSize", "MaxClustSize", "lambda1", "lambda2",
        "tMin", "tMax", "tMean", "tSpread", "Nmax", "z0_fit", "Xmax", "lambda",
        "nTrackSegments", "eccentricity3D"
    ])

    # targets: Sequence[int] = field(default_factory=lambda: [-211, 2212, 311])  # pi-, proton, K0
    # feature_cols: Sequence[str] = field(default_factory=lambda: [
    #     "Thr1", "Thr2", "Thr3", "Begin", "Radius", "Density", "NClusters", "ratioThr23", "Zbary", "Zrms",
    #     "PctHitsFirst10", "PlanesWithClusmore2", "AvgClustSize", "MaxClustSize", "lambda1", "lambda2",
    #     "Nmax", "z0_fit", "Xmax", "lambda",
    #     "nTrackSegments", "eccentricity3D"
    # ])

    # LightGBM (bons defaults)
    learning_rate: float = 0.01
    n_estimators: int = 1000
    num_leaves: int = 63
    max_depth: int = -1
    reg_alpha: float = 0.1
    reg_lambda: float = 0.1
    min_child_samples: int = 30
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    class_weight: Optional[Union[str, Dict[int, float]]] = None  # SMOTE activé -> pas de class_weight

    # Grid-search (si activée)
    param_grid: Dict[str, List] = field(default_factory=lambda: {
        'num_leaves': [31, 63, 127],
        'max_depth': [-1, 10, 20],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [500, 1000]
    })


# =============================================================================
# Logging
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Paths & CSV logs
# =============================================================================

PARAMETERS_FILE = Path("run_parameters.csv")
COMMENTS_FILE   = Path("run_comments.csv")
PERF_FILE       = Path("hadron_performances.csv")
RES_DIR         = Path("results_with_time")

def _next_run_id(csv_path: Path) -> int:
    if not csv_path.exists():
        return 1
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        try:
            return max(int(r["run_id"]) for r in reader) + 1
        except ValueError:
            return 1

def _write_dict_row(csv_path: Path, row: Dict[str, Union[str, int, float]]) -> None:
    is_new = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if is_new:
            writer.writeheader()
        writer.writerow(row)

def _write_row(csv_path: Path, header: List[str], row: List) -> None:
    is_new = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(header)
        writer.writerow(row)

def _filter_cfg(cfg: Config) -> Dict[str, Union[str, int, float]]:
    return {
        "file_paths": ";".join(str(p) for p in cfg.file_paths),
        "tree_name": cfg.tree_name,
        "test_size": cfg.test_size,
        "val_size": cfg.val_size,
        "random_state": cfg.random_state,
        "learning_rate": cfg.learning_rate,
        "n_estimators": cfg.n_estimators,
        "num_leaves": cfg.num_leaves,
        "max_depth": cfg.max_depth,
        "reg_alpha": cfg.reg_alpha,
        "reg_lambda": cfg.reg_lambda,
        "min_child_samples": cfg.min_child_samples,
        "subsample": cfg.subsample,
        "colsample_bytree": cfg.colsample_bytree,
        "class_weight": str(cfg.class_weight),
        "grid_search": int(cfg.grid_search),
    }

def init_run(config: Config) -> Tuple[int, Dict[str, Path]]:
    run_id = _next_run_id(PARAMETERS_FILE)
    paths = {
        "model":      RES_DIR / "models" / f"lgbm_model_{run_id}.joblib",
        "train_plot": RES_DIR / "plots" / f"training_curve_{run_id}.png",
        "cm_plot":    RES_DIR / "plots" / f"confusion_matrix_{run_id}.png",
        "hist_plot":  RES_DIR / "plots" / f"prob_hists_{run_id}.png",
    }
    for p in paths.values():
        p.parent.mkdir(parents=True, exist_ok=True)
    _write_dict_row(PARAMETERS_FILE, {"run_id": run_id, **_filter_cfg(config)})
    return run_id, paths

def log_comment(run_id: int, comment: str) -> None:
    _write_row(COMMENTS_FILE, ["run_id", "comment"], [run_id, comment])

def log_performance(run_id: int, loss: float, acc: float, auc: float) -> None:
    _write_row(PERF_FILE, ["run_id", "test_multi_logloss", "test_acc", "test_auc"], [run_id, loss, acc, auc])


# =============================================================================
# Data I/O & preprocessing
# =============================================================================

def load_root(files: Iterable[Path], tree: str, cols: Sequence[str]) -> pd.DataFrame:
    """Lit plusieurs ROOT et concatène les colonnes demandées."""
    dfs = []
    logger.info("Reading ROOT files...")
    for p in files:
        if not p.is_file():
            logger.warning("Missing file: %s", p)
            continue
        with uproot.open(p) as f:
            dfs.append(f[tree].arrays(cols, library='pd'))
    if not dfs:
        raise RuntimeError("No valid ROOT files found.")
    df = pd.concat(dfs, ignore_index=True)
    logger.info("Total events loaded: %d", len(df))
    return df

def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Remplace ±inf par NA, dropna, réindexe."""
    before = len(df)
    df = df.replace([np.inf, -np.inf], pd.NA).dropna().reset_index(drop=True)
    logger.info("After cleaning: %d (removed %d)", len(df), before - len(df))
    return df

def filter_targets(df: pd.DataFrame, targets: Sequence[int]) -> pd.DataFrame:
    """Filtre les PDG voulus et crée la colonne 'label' selon l'ordre des targets."""
    df = df[df['particlePDG'].isin(targets)].copy()
    if df.empty:
        raise RuntimeError("No entries after PDG filtering.")
    label_map = {pdg: idx for idx, pdg in enumerate(targets)}
    df['label'] = df['particlePDG'].map(label_map)
    binc = np.bincount(df['label'])
    logger.info("Class distribution: %s", binc)
    return df

def split_and_smote(X: np.ndarray, y: np.ndarray, cfg: Config):
    """
    Split train/test, puis train/val sur le train.
    Applique SMOTE uniquement au train (sans scaler, demandé).
    """
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y
    )
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tr, y_tr, test_size=cfg.val_size, random_state=cfg.random_state, stratify=y_tr
    )
    sm = SMOTE(random_state=cfg.random_state)
    X_tr_r, y_tr_r = sm.fit_resample(X_tr, y_tr)
    logger.info("After SMOTE (train): %s", np.bincount(y_tr_r))
    return X_tr_r, y_tr_r, X_val, y_val, X_te, y_te

def smear_column_gaussian(df: pd.DataFrame, col: str, sigma: float, random_state: Optional[int] = None) -> pd.DataFrame:
    """Ajoute un bruit gaussien contrôlé pour la reproductibilité."""
    rng = np.random.default_rng(random_state)
    df[col] = df[col].to_numpy() + rng.normal(0, sigma, size=len(df))
    return df


# =============================================================================
# Modèle & entraînement
# =============================================================================

def build_lgbm(cfg: Config) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective='multiclass', num_class=len(cfg.targets),
        learning_rate=cfg.learning_rate, n_estimators=cfg.n_estimators,
        num_leaves=cfg.num_leaves, max_depth=cfg.max_depth,
        reg_alpha=cfg.reg_alpha, reg_lambda=cfg.reg_lambda,
        min_child_samples=cfg.min_child_samples,
        subsample=cfg.subsample, colsample_bytree=cfg.colsample_bytree,
        class_weight=cfg.class_weight, random_state=cfg.random_state,
        n_jobs=-1
    )

def train_model(X_tr, y_tr, X_val, y_val, cfg: Config):
    """Entraîne le modèle; si grid_search, cherche les hyperparams puis re-fit avec early stopping."""
    if not cfg.grid_search:
        clf = build_lgbm(cfg)
        clf.fit(
            X_tr, y_tr,
            eval_set=[(X_tr, y_tr), (X_val, y_val)],
            eval_names=['train', 'valid'], eval_metric='multi_logloss',
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(period=100)]
        )
        return clf, clf.evals_result_

    logger.info("Running GridSearchCV...")
    base = build_lgbm(cfg)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.random_state)
    grid = GridSearchCV(base, cfg.param_grid, cv=cv, scoring='f1_macro', n_jobs=-1, verbose=1)
    grid.fit(X_tr, y_tr)
    best = grid.best_params_
    logger.info("Best params: %s", best)

    clf = build_lgbm(cfg)
    clf.set_params(**best)
    clf.fit(
        X_tr, y_tr,
        eval_set=[(X_tr, y_tr), (X_val, y_val)],
        eval_names=['train', 'valid'], eval_metric='multi_logloss',
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(period=100)]
    )
    return clf, clf.evals_result_


# =============================================================================
# Évaluation & graphiques
# =============================================================================

def evaluate(clf, X_test, y_test) -> Tuple[float, np.ndarray, np.ndarray, float, float]:
    y_pred = clf.predict(X_test)
    proba  = clf.predict_proba(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm  = confusion_matrix(y_test, y_pred)
    auc = roc_auc_score(y_test, proba, multi_class='ovr')
    mll = log_loss(y_test, proba, labels=np.arange(proba.shape[1]))
    logger.info("Accuracy: %.4f | AUC (OVR): %.4f | Multi-logloss: %.4f", acc, auc, mll)
    logger.info("Classification report:\n%s", classification_report(y_test, y_pred))
    return acc, cm, proba, auc, mll

def plot_training_curve(train_loss: List[float], valid_loss: List[float], out: Path):
    plt.figure(figsize=(6, 4))
    plt.plot(train_loss, label='train loss')
    plt.plot(valid_loss, '--', label='valid loss')
    plt.xlabel('Iteration'); plt.ylabel('Multi logloss')
    plt.legend(); plt.title('Training vs validation logloss')
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()

def plot_confusion_matrix(cm: np.ndarray, labels: Sequence[str], out: Path):
    pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100.0
    annot = np.array([[f"{v:.1f}%" for v in row] for row in pct])
    plt.figure(figsize=(6, 5))
    sns.heatmap(pct, annot=annot, fmt='', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted'); plt.ylabel('True'); plt.title('Confusion Matrix')
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()

def plot_probability_histograms(proba: np.ndarray, y_true: np.ndarray, out: Path):
    fig = plt.figure(figsize=(15, 4))
    # Proton
    ax = fig.add_subplot(1, 3, 1)
    ax.hist(proba[y_true == 0, 1], bins=50, alpha=0.5, label='π- (true)')
    ax.hist(proba[y_true == 1, 1], bins=50, alpha=0.5, label='p (true)')
    ax.set_xlabel('P(predicted=p)'); ax.set_ylabel('Events'); ax.legend(); ax.set_title('Proton probability')
    # π- vs K0
    ax = fig.add_subplot(1, 3, 2)
    ax.hist(proba[y_true == 0, 2], bins=50, alpha=0.5, label='π-')
    ax.hist(proba[y_true == 2, 2], bins=50, alpha=0.5, label='K0')
    ax.set_xlabel('P(predicted=K0)'); ax.legend(); ax.set_title('π- vs K0')
    # p vs K0
    ax = fig.add_subplot(1, 3, 3)
    ax.hist(proba[y_true == 1, 2], bins=50, alpha=0.5, label='p')
    ax.hist(proba[y_true == 2, 2], bins=50, alpha=0.5, label='K0')
    ax.set_xlabel('P(predicted=K0)'); ax.legend(); ax.set_title('p vs K0')
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close()


# =============================================================================
# Pipeline principal
# =============================================================================

def run(cfg: Config):
    run_id, paths = init_run(cfg)

    # Chargement & prétraitement
    df = load_root(cfg.file_paths, cfg.tree_name, list(cfg.feature_cols) + ['particlePDG'])
    df = clean_df(df)
    df = filter_targets(df, cfg.targets)
    df = smear_column_gaussian(df, "tMin", sigma=0.1, random_state=cfg.random_state)
    df = smear_column_gaussian(df, "tMax", sigma=0.1, random_state=cfg.random_state)
    df = smear_column_gaussian(df, "tSpread", sigma=0.1, random_state=cfg.random_state)

    # Conserver les noms de features (DataFrame) puis SMOTE sur ndarray
    X = df[cfg.feature_cols].copy()      # garde les noms de colonnes
    y = df['label'].to_numpy()
    X_tr, y_tr, X_val, y_val, X_te, y_te = split_and_smote(X.to_numpy(), y, cfg)

    # Refaire des DataFrames avec les bons noms (SMOTE renvoie des ndarray)
    feature_cols = list(cfg.feature_cols)
    X_tr = pd.DataFrame(X_tr, columns=feature_cols)
    X_val = pd.DataFrame(X_val, columns=feature_cols)
    X_te  = pd.DataFrame(X_te,  columns=feature_cols)

    # Entraînement (les noms de colonnes seront conservés dans le modèle)
    clf, evals = train_model(X_tr, y_tr, X_val, y_val, cfg)

    # Évaluation
    acc, cm, proba, auc, mll = evaluate(clf, X_te, y_te)

    # Figures
    plot_training_curve(evals['train']['multi_logloss'], evals['valid']['multi_logloss'], paths['train_plot'])
    plot_confusion_matrix(cm, ['pi-', 'proton', 'K0'], paths['cm_plot'])
    plot_probability_histograms(proba, y_te, paths['hist_plot'])

    # --- Sauvegarde d'un CSV des features de test (pour SHAP) ---
    X_test_df = X_te.copy()  # déjà DataFrame avec les bons noms
    X_test_csv = RES_DIR / "data" / f"X_test_{run_id}.csv"
    X_test_csv.parent.mkdir(parents=True, exist_ok=True)
    X_test_df.to_csv(X_test_csv, index=False)
    logger.info("Saved X_test CSV for SHAP -> %s", X_test_csv)

    # Sauvegardes & logs
    joblib.dump(clf, paths['model'])
    logger.info("Artifacts saved to %s", RES_DIR)
    log_performance(run_id, mll, acc, auc)
    log_comment(run_id, f"Run {run_id}: acc={acc:.4f}, auc={auc:.4f}, mlogloss={mll:.4f}")

# =============================================================================
# Exécution
# =============================================================================

if __name__ == "__main__":
    run(Config())
