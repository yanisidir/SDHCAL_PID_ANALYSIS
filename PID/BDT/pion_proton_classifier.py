#!/usr/bin/env python3

"""
pi_vs_proton_lgbm_smear_sweep.py

Classification binaire π⁻ vs proton avec LightGBM + étude de robustesse au smearing temporel.

Fonctions clés :
- I/O : lecture de plusieurs ROOT (uproot) depuis tree, concaténation
- Prétraitement : nettoyage (±inf→NA, dropna), filtrage PDG {-211, 2212}, création du label {π⁻=0, p=1}
- Split stratifié : train/test puis train/val (pas de StandardScaler, modèles en arbres)
- Option SMOTE (désactivé par défaut ici) pour équilibrer le train
- Entraînement LightGBM (objective='binary') avec early stopping (binary_logloss)
- Évaluation test : accuracy, AUC, binary logloss, matrice de confusion
- Figures : courbe train/valid (logloss), confusion matrix (%), histogrammes de P(proton)
- Logs & artefacts :
    * Modèle .joblib → results_with_time/models/
    * Plots → results_with_time/plots/
    * CSV : paramètres (run_parameters.csv), perfs (hadron_performances.csv), commentaires (run_comments.csv)
    * Dump X_test (CSV) pour analyses SHAP

Étude de smearing :
- `sweep_smearing` balaye une grille de sigma appliqués à (tMin, tMax, tSpread)
- Moyenne optionnelle sur plusieurs seeds pour stabiliser (accuracy, AUC, logloss)
- Détection de seuils de dégradation (acc ≤ acc0 − Δ) et estimation du “coude” (pente négative max)
- Sauvegarde d’un plot Accuracy vs Smearing

Unités :
- Les valeurs de `smear_grid` sont en unités des colonnes t* (ex. ns). Exemples :
  0.0 → 0 ps, 0.02 → 20 ps, 0.1 → 100 ps (adapter à vos unités réelles).

Usage rapide :
    # run simple (entraînement + évaluation + figures)
    python pi_vs_proton_lgbm_smear_sweep.py

    # étude de smearing (déjà dans le bloc __main__)
    python pi_vs_proton_lgbm_smear_sweep.py

Dépendances : numpy, pandas, uproot, lightgbm, scikit-learn, imbalanced-learn, joblib, matplotlib, seaborn.
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
from sklearn.metrics import roc_auc_score
# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Config:
    """Pipeline settings and hyper-parameters."""
    # I/O (seulement π- et proton)
    file_paths: List[Path] = field(default_factory=lambda: [
        Path("/gridgroup/ilc/midir/analyse/data/merged_primaryEnergy/200k_pi-_1-130_params_merged.root"),
        Path("/gridgroup/ilc/midir/analyse/data/merged_primaryEnergy/200k_proton_1-130_params_merged.root"),
    ])
    tree_name: str = "tree"

    # Entraînement
    grid_search: bool = False
    test_size: float = 0.20
    val_size: float = 0.20
    random_state: int = 42

    energy_col: str = "primaryEnergy"             
    energy_bins: Sequence[float] = tuple(range(1,131,10))

    # Cibles et features (binaire: π- vs proton)
    targets: Sequence[int] = field(default_factory=lambda: [-211, 2212])  # 0->pi-, 1->proton
    feature_cols: Sequence[str] = field(default_factory=lambda: [
        "Thr1", "Thr2", "Thr3", 
        "Begin", "meanRadius", "nMipCluster", "first5LayersRMS",
        "Density", "NClusters", "ratioThr23", "Zbary", "Zrms",
        "PctHitsFirst10", "AvgClustSize", "MaxClustSize", "lambda1", "lambda2",
        "tMin", "tMax", "tMean", "tSpread", "Nmax", "Xmax",
        "eccentricity3D", "transverseRatio", "nTrack"
    ])

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
    class_weight: Optional[Union[str, Dict[int, float]]] = None

    # Grid-search (optionnel)
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

def init_run(config: Config):
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
    _write_row(PERF_FILE, ["run_id", "test_binary_logloss", "test_acc", "test_auc"], [run_id, loss, acc, auc])

# =============================================================================
# Data I/O & preprocessing
# =============================================================================

def load_root(files: Iterable[Path], tree: str, cols: Sequence[str]) -> pd.DataFrame:
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
    before = len(df)
    df = df.replace([np.inf, -np.inf], pd.NA).dropna().reset_index(drop=True)
    logger.info("After cleaning: %d (removed %d)", len(df), before - len(df))
    return df

def filter_targets(df: pd.DataFrame, targets: Sequence[int]) -> pd.DataFrame:
    df = df[df['primaryID'].isin(targets)].copy()
    if df.empty:
        raise RuntimeError("No entries after PDG filtering.")
    label_map = {pdg: idx for idx, pdg in enumerate(targets)}  # -211->0, 2212->1
    df['label'] = df['primaryID'].map(label_map)
    binc = np.bincount(df['label'])
    logger.info("Class distribution: %s", binc)
    return df

def split_and_smote(X: np.ndarray, y: np.ndarray, cfg: Config, use_smote: bool = True):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y
    )
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tr, y_tr, test_size=cfg.val_size, random_state=cfg.random_state, stratify=y_tr
    )
    if use_smote:
        sm = SMOTE(random_state=cfg.random_state)
        X_tr, y_tr = sm.fit_resample(X_tr, y_tr)
        logger.info("After SMOTE (train): %s", np.bincount(y_tr))
    return X_tr, y_tr, X_val, y_val, X_te, y_te

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
        objective='binary',
        learning_rate=cfg.learning_rate, n_estimators=cfg.n_estimators,
        num_leaves=cfg.num_leaves, max_depth=cfg.max_depth,
        reg_alpha=cfg.reg_alpha, reg_lambda=cfg.reg_lambda,
        min_child_samples=cfg.min_child_samples,
        subsample=cfg.subsample, colsample_bytree=cfg.colsample_bytree,
        class_weight=cfg.class_weight, random_state=cfg.random_state,
        n_jobs=-1
    )

def train_model(X_tr, y_tr, X_val, y_val, cfg: Config):
    if not cfg.grid_search:
        clf = build_lgbm(cfg)
        clf.fit(
            X_tr, y_tr,
            eval_set=[(X_tr, y_tr), (X_val, y_val)],
            eval_names=['train', 'valid'], eval_metric='binary_logloss',
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(period=100)]
        )
        return clf, clf.evals_result_

    logger.info("Running GridSearchCV...")
    base = build_lgbm(cfg)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.random_state)
    grid = GridSearchCV(base, cfg.param_grid, cv=cv, scoring='f1', n_jobs=-1, verbose=1)
    grid.fit(X_tr, y_tr)
    best = grid.best_params_
    logger.info("Best params: %s", best)

    clf = build_lgbm(cfg)
    clf.set_params(**best)
    clf.fit(
        X_tr, y_tr,
        eval_set=[(X_tr, y_tr), (X_val, y_val)],
        eval_names=['train', 'valid'], eval_metric='binary_logloss',
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(period=100)]
    )
    return clf, clf.evals_result_

def train_model_fast(X_tr, y_tr, X_val, y_val, cfg: Config):
    clf = lgb.LGBMClassifier(
        objective='binary',
        learning_rate=0.05,           # un peu plus rapide à converger
        n_estimators=300,              # ↓
        num_leaves=31,                 # ↓
        max_depth=-1,
        reg_alpha=cfg.reg_alpha, reg_lambda=cfg.reg_lambda,
        min_child_samples=cfg.min_child_samples,
        subsample=0.8, colsample_bytree=0.8,
        class_weight=cfg.class_weight,
        random_state=cfg.random_state,
        n_jobs=4                       # évite de saturer la machine
    )
    clf.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_names=['valid'], eval_metric='binary_logloss',
        callbacks=[lgb.early_stopping(20), lgb.log_evaluation(period=50)]
    )
    return clf

# =============================================================================
# Évaluation & graphiques
# =============================================================================

def evaluate(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    proba  = clf.predict_proba(X_test)   # shape (n, 2) -> colonne 1 = P(classe=1) = proton
    acc = accuracy_score(y_test, y_pred)
    cm  = confusion_matrix(y_test, y_pred)
    auc = roc_auc_score(y_test, proba[:, 1])
    mll = log_loss(y_test, proba[:, 1])
    logger.info("Accuracy: %.4f | AUC: %.4f | Binary logloss: %.4f", acc, auc, mll)
    logger.info("Classification report:\n%s", classification_report(y_test, y_pred, target_names=['pi-', 'proton']))
    return acc, cm, proba, auc, mll

def plot_training_curve(train_loss: List[float], valid_loss: List[float], out: Path):
    plt.figure(figsize=(6, 4))
    plt.plot(train_loss, label='train loss')
    plt.plot(valid_loss, '--', label='valid loss')
    plt.xlabel('Iteration'); plt.ylabel('Binary logloss')
    plt.legend(); plt.title('Training vs validation logloss')
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()

def plot_confusion_matrix(cm: np.ndarray, labels: Sequence[str], out: Path):
    pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100.0
    annot = np.array([[f"{v:.1f}%" for v in row] for row in pct])
    plt.figure(figsize=(5, 4))
    sns.heatmap(pct, annot=annot, fmt='', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted'); plt.ylabel('True'); plt.title('Confusion Matrix (%)')
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()

def plot_probability_histograms(proba: np.ndarray, y_true: np.ndarray, out: Path):
    plt.figure(figsize=(6, 4))
    # histogramme de P(proton) pour les deux vraies classes
    plt.hist(proba[y_true == 0, 1], bins=50, alpha=0.5, label='π- (true)')
    plt.hist(proba[y_true == 1, 1], bins=50, alpha=0.5, label='p (true)')
    plt.xlabel('P(predicted = proton)'); plt.ylabel('Events'); plt.legend()
    plt.title('Separation with P(proton)')
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()

def precision_by_energy(y_true: np.ndarray, y_pred: np.ndarray, energy: np.ndarray,
                        bins: Sequence[float]) -> pd.DataFrame:
    bins = np.asarray(bins, dtype=float)
    idx = np.digitize(energy, bins, right=False)  # bin i: [bins[i-1], bins[i])
    rows = []
    for i in range(1, len(bins)):
        m = idx == i
        n = int(m.sum())
        if n == 0:
            rows.append((bins[i-1], bins[i], 0, np.nan, np.nan))
            continue
        precs = []
        for c in (0, 1):  # 0=pi-, 1=proton
            tp = np.sum((y_pred[m] == c) & (y_true[m] == c))
            fp = np.sum((y_pred[m] == c) & (y_true[m] != c))
            prec = tp / (tp + fp) if (tp + fp) > 0 else np.nan
            precs.append(prec)
        rows.append((bins[i-1], bins[i], n, precs[0], precs[1]))
    return pd.DataFrame(rows, columns=["E_low", "E_high", "N", "precision_pi_minus", "precision_proton"])

def plot_precision_by_energy(df_prec: pd.DataFrame, out: Path):
    centers = 0.5 * (df_prec["E_low"].to_numpy() + df_prec["E_high"].to_numpy())
    plt.figure(figsize=(6,4))
    plt.plot(centers, df_prec["precision_pi_minus"].to_numpy(), marker='o', label='Precision π−')
    plt.plot(centers, df_prec["precision_proton"].to_numpy(), marker='s', label='Precision p')
    plt.xlabel('Énergie (GeV)'); plt.ylabel('Preision'); plt.ylim(0, 1)
    plt.title('Preision by species vs. energy (test set)')
    plt.grid(True, ls=':')
    plt.legend()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()

def auc_by_energy(y_true: np.ndarray, proba: np.ndarray, energy: np.ndarray,
                  bins: Sequence[float]) -> pd.DataFrame:
    """
    Calcule l’AUC binaire par bin d’énergie.
    `proba` est l’array (n, 2) renvoyé par `predict_proba` (colonne 1 = P(proton)).
    Renvoie un DataFrame avec AUC par intervalle [E_low, E_high).
    """
    bins = np.asarray(bins, dtype=float)
    idx = np.digitize(energy, bins, right=False)  # bin i: [bins[i-1], bins[i])
    rows = []
    p1 = proba[:, 1]  # P(classe=1 = proton)
    for i in range(1, len(bins)):
        m = idx == i
        n = int(m.sum())
        if n == 0:
            rows.append((bins[i-1], bins[i], 0, 0, 0, np.nan))
            continue
        yb = y_true[m]
        nb_pos = int((yb == 1).sum())
        nb_neg = int((yb == 0).sum())
        if nb_pos > 0 and nb_neg > 0:
            auc = roc_auc_score(yb, p1[m])
        else:
            auc = np.nan  # AUC non définie si un seul type de classe présent
        rows.append((bins[i-1], bins[i], n, nb_neg, nb_pos, auc))
    return pd.DataFrame(rows, columns=["E_low", "E_high", "N", "N_pi_minus", "N_proton", "AUC"])

def plot_auc_by_energy(df_auc: pd.DataFrame, out: Path):
    centers = 0.5 * (df_auc["E_low"].to_numpy() + df_auc["E_high"].to_numpy())
    plt.figure(figsize=(6,4))
    plt.plot(centers, df_auc["AUC"].to_numpy(), marker='o')
    plt.xlabel('Énergie (GeV)'); plt.ylabel('AUC'); plt.ylim(0.5, 1.0)
    plt.title('AUC vs énergie (jeu de test)')
    plt.grid(True, ls=':')
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
# =============================================================================
# Sweep du smearing & détection du seuil
# =============================================================================

def stratified_frac(df: pd.DataFrame, label_col: str, frac: float, seed: int) -> pd.DataFrame:
    if frac >= 1.0:
        return df
    parts = []
    for _, g in df.groupby(label_col, group_keys=False):
        parts.append(g.sample(frac=frac, random_state=seed))
    return pd.concat(parts, ignore_index=True)


def run_with_sigma(cfg: Config, sigma: float) -> Tuple[float, float, float]:
    """
    Exécute la pipeline pour un smearing de 'sigma' (même unité que tMin/Max/Spread).
    Retourne (acc, auc, logloss) sur le test.
    """
    # --- on recharge proprement les données pour ne pas cumuler le bruit ---
    df = load_root(cfg.file_paths, cfg.tree_name, list(cfg.feature_cols) + ['primaryID'])
    df = clean_df(df)
    df = filter_targets(df, cfg.targets)
    df = stratified_frac(df, 'label', frac=0.3, seed=cfg.random_state)

    # appliquer le même smearing aux 3 colonnes temporelles mais avec des seeds différentes
    base = int(cfg.random_state if cfg.random_state is not None else 0)
    for i, col in enumerate(("tMin", "tMax", "tSpread"), start=1):
        df = smear_column_gaussian(df, col, sigma=sigma, random_state=base + 100*i)

    X = df[cfg.feature_cols].copy()
    y = df['label'].to_numpy()

    X_tr, y_tr, X_val, y_val, X_te, y_te = split_and_smote(X.to_numpy(), y, cfg, use_smote=False)
    feature_cols = list(cfg.feature_cols)
    X_tr = pd.DataFrame(X_tr, columns=feature_cols)
    X_val = pd.DataFrame(X_val, columns=feature_cols)
    X_te  = pd.DataFrame(X_te,  columns=feature_cols)

    clf = train_model_fast(X_tr, y_tr, X_val, y_val, cfg)
    acc, _, _, auc, mll = evaluate(clf, X_te, y_te)
    return acc, auc, mll


def sweep_smearing(cfg: Config,
                   smear_values: Sequence[float],
                   seeds: Sequence[int] = (42,),
                   deltas: Sequence[float] = (0.01, 0.02, 0.05)) -> Dict[str, Union[List[float], Dict[str, Optional[float]]]]:
    """
    Balaye des valeurs de smearing et, si 'seeds' contient plusieurs entiers,
    moyenne les métriques sur plusieurs random_state pour stabiliser.
    Renvoie un dict contenant les listes et les seuils estimés.
    """
    accs_mean, accs_std = [], []
    aucs_mean, losses_mean = [], []

    original_state = cfg.random_state

    for sigma in smear_values:
        accs, aucs, losses = [], [], []
        for s in seeds:
            cfg.random_state = s
            acc, auc, mll = run_with_sigma(cfg, sigma)
            accs.append(acc); aucs.append(auc); losses.append(mll)
        accs = np.array(accs); aucs = np.array(aucs); losses = np.array(losses)
        accs_mean.append(float(accs.mean()))
        accs_std.append(float(accs.std(ddof=1) if len(accs) > 1 else 0.0))
        aucs_mean.append(float(aucs.mean()))
        losses_mean.append(float(losses.mean()))

    # restaurer le random_state d'origine
    cfg.random_state = original_state

    # seuils : point où l'accuracy descend sous (acc0 - delta)
    acc0 = accs_mean[0]  # valeur sans smearing si smear_values[0] == 0.0
    thresholds = {}
    for d in deltas:
        thr = None
        for sig, acc in zip(smear_values, accs_mean):
            if acc <= acc0 - d:
                thr = float(sig); break
        thresholds[f"delta_{int(d*100)}pp"] = thr  # ex: delta_1pp -> 1 point de pourcentage

    # estimation "coude" via la plus forte pente négative discrète (acc_i - acc_{i-1})
    diffs = np.diff(accs_mean)
    knee_idx = int(np.argmin(diffs)) + 1 if len(diffs) else 0
    knee_sigma = float(smear_values[knee_idx]) if smear_values else None

    return {
        "smear_values": list(map(float, smear_values)),
        "acc_mean": accs_mean,
        "acc_std": accs_std,
        "auc_mean": aucs_mean,
        "logloss_mean": losses_mean,
        "thresholds": thresholds,
        "knee_sigma": knee_sigma,
    }


def plot_acc_vs_smear(smear_values: Sequence[float], acc_mean: Sequence[float], acc_std: Sequence[float], out: Path):
    plt.figure(figsize=(6,4))
    plt.errorbar(smear_values, acc_mean, yerr=acc_std, fmt='-o', capsize=3, label='Accuracy')
    plt.xlabel('Smearing ns')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs smearing')
    plt.grid(True, ls=':')
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150); plt.close()

# =============================================================================
# Pipeline principal
# =============================================================================

def run(cfg: Config):
    run_id, paths = init_run(cfg)

    # Chargement & prétraitement
    df = load_root(cfg.file_paths, cfg.tree_name, list(cfg.feature_cols) + ['primaryID', cfg.energy_col])

    df = clean_df(df)
    df = filter_targets(df, cfg.targets)

    base = int(cfg.random_state if cfg.random_state is not None else 0)
    for i, col in enumerate(("tMin", "tMax", "tSpread"), start=1):
        df = smear_column_gaussian(df, col, sigma=0.1, random_state=base + 100*i)

    # --- features / labels / energy
    X = df[cfg.feature_cols].to_numpy()
    y = df['label'].to_numpy()
    E = df[cfg.energy_col].to_numpy()

    # --- split stratifiés (sans SMOTE)
    X_tr, X_te, y_tr, y_te, E_tr, E_te = train_test_split(
        X, y, E, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y
    )
    X_tr, X_val, y_tr, y_val, E_tr, E_val = train_test_split(
        X_tr, y_tr, E_tr, test_size=cfg.val_size, random_state=cfg.random_state, stratify=y_tr
    )

    feature_cols = list(cfg.feature_cols)
    X_tr = pd.DataFrame(X_tr, columns=feature_cols)
    X_val = pd.DataFrame(X_val, columns=feature_cols)
    X_te  = pd.DataFrame(X_te,  columns=feature_cols)

    clf, evals = train_model(X_tr, y_tr, X_val, y_val, cfg)

    acc, cm, proba, auc, mll = evaluate(clf, X_te, y_te)

    # y_pred du test pour les précisions par énergie
    y_pred = clf.predict(X_te)
    prec_df = precision_by_energy(y_true=y_te, y_pred=y_pred, energy=E_te, bins=cfg.energy_bins)

    # Sauvegardes
    prec_csv  = RES_DIR / "data" / f"precision_by_energy_{run_id}.csv"
    prec_plot = RES_DIR / "plots" / f"precision_vs_energy_{run_id}.png"
    prec_df.to_csv(prec_csv, index=False)
    plot_precision_by_energy(prec_df, prec_plot)
    logger.info("Saved per-energy precision CSV -> %s", prec_csv)
    logger.info("Saved per-energy precision plot -> %s", prec_plot)

    auc_df  = auc_by_energy(y_true=y_te, proba=proba, energy=E_te, bins=cfg.energy_bins)

    auc_csv  = RES_DIR / "data"  / f"auc_by_energy_{run_id}.csv"
    auc_plot = RES_DIR / "plots" / f"auc_vs_energy_{run_id}.png"

    auc_df.to_csv(auc_csv, index=False)
    plot_auc_by_energy(auc_df, auc_plot)
    logger.info("Saved per-energy AUC CSV -> %s", auc_csv)
    logger.info("Saved per-energy AUC plot -> %s", auc_plot)
auc_csv
    # Récupérer les pertes pour les courbes
    train_loss = evals['train']['binary_logloss']
    valid_loss = evals['valid']['binary_logloss']
    plot_training_curve(train_loss, valid_loss, paths['train_plot'])
    plot_confusion_matrix(cm, ['pi-', 'proton'], paths['cm_plot'])
    plot_probability_histograms(proba, y_te, paths['hist_plot'])

    # Sauvegarde X_test
    X_test_df = X_te.copy()
    X_test_csv = RES_DIR / "data" / f"X_test_{run_id}.csv"
    X_test_csv.parent.mkdir(parents=True, exist_ok=True)
    X_test_df.to_csv(X_test_csv, index=False)
    logger.info("Saved X_test CSV for SHAP -> %s", X_test_csv)

    # Sauvegardes & logs
    joblib.dump(clf, paths['model'])
    logger.info("Artifacts saved to %s", RES_DIR)
    log_performance(run_id, mll, acc, auc)
    log_comment(run_id, f"Run {run_id}: acc={acc:.4f}, auc={auc:.4f}, blogloss={mll:.4f}")

# =============================================================================
# Exécution
# =============================================================================

if __name__ == "__main__":
    cfg = Config()
    run(Config())

    # # Si tes temps sont en ns: 0.0 -> 0 ps, 0.02 -> 20 ps, 0.05 -> 50 ps, 0.1 -> 100 ps, etc.
    # smear_grid = [1.00e-06, 2.15e-06, 4.64e-06, 1.00e-05, 2.15e-05, 4.64e-05, 1.00e-04]

    # # Optionnel : moyenner sur plusieurs seeds pour lisser l'effet SMOTE / split
    # seeds = (41, 42, 43)

    # res = sweep_smearing(cfg, smear_grid, seeds=seeds, deltas=(0.01, 0.02, 0.05))

    # # Log texte rapide
    # print("Smear\tAcc_mean\tAcc_std")
    # for s, a, e in zip(res["smear_values"], res["acc_mean"], res["acc_std"]):
    #     print(f"{s:.3f}\t{a:.4f}\t{e:.4f}")

    # print("Seuils (chute depuis 0-smear):", res["thresholds"])
    # print("Coude estimé (max pente négative):", res["knee_sigma"])

    # # Plot
    # plot_acc_vs_smear(res["smear_values"], res["acc_mean"], res["acc_std"],
    #                   RES_DIR / "plots" / "accuracy_vs_smearing.png")
