#!/usr/bin/env python3
"""2_hadron_classifier_MLP.py

    ./2_hadron_classifier_MLP.py
ou 
    ./2_hadron_classifier_MLP.py --grid-search
    ./2_hadron_classifier_MLP.py --file-paths f1.root f2.root f3.root
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple, Dict

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import uproot
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# -----------------------------------------------------------------------------
# Configuration dataclass
# -----------------------------------------------------------------------------
@dataclass
class Config:
    """Pipeline settings and hyper‑parameters."""

    # I/O
    file_paths: List[Path] = field(default_factory=lambda: [
        Path("/gridgroup/ilc/midir/analyse/data/params/130k_pi_E1to130_params.root"),
        Path("/gridgroup/ilc/midir/analyse/data/params/130k_kaon_E1to130_params.root"),
        Path("/gridgroup/ilc/midir/analyse/data/params/130k_proton_E1to130_params.root"),
    ])            # ROOT file to open
    tree_name: str = "paramsTree"

    # Option Gridsearch
    grid_search: bool = False

    # ML setup
    targets: Sequence[int] = (-211, 2212, 311)
    test_size: float = 0.20
    random_state: int = 42

    feature_cols: Sequence[str] = field(default_factory=lambda: [
        "Thr1", "Thr2", "Thr3", "Begin", "Radius", "Density", "NClusters", "ratioThr23", "Zbary", "Zrms",
        "PctHitsFirst10", "PlanesWithClusmore2", "AvgClustSize", "MaxClustSize", "lambda1", "lambda2",
        "tMin", "tMax", "tMean", "tSpread", "Nmax", "z0_fit", "Xmax", "lambda",
        "nTrackSegments", "eccentricity3D"
    ])

    # MLP hyper‑parameters
    hidden_layer_sizes: Sequence[int] = (128, 64, 32)
    alpha: float = 1e-4
    batch_size: int = 32
    learning_rate_init: float = 1e-3
    max_iter: int = 500

    # Grid‑search ranges
    param_grid: dict = field(default_factory=lambda: {
        "hidden_layer_sizes": [(64, 32), (128, 64, 32)],
        "alpha": [1e-5, 1e-4, 1e-3],
        "learning_rate_init": [1e-4, 5e-4, 1e-3],
    })

# -----------------------------------------------------------------------------
# Logging setup
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# -------------------------- run‑id utilities -------------------------------
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


def _write_dict_row(csv_path: Path, row: Dict[str, str | int | float]) -> None:
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


def _filter_cfg(cfg: Config) -> Dict[str, str | int | float]:
    """Sélectionne les paramètres clefs du run pour le CSV."""
    return {
        "file_paths":      ";".join(str(p) for p in cfg.file_paths),
        "tree_name":       cfg.tree_name,
        "test_size":       cfg.test_size,
        "random_state":    cfg.random_state,
        "hidden_layer_sizes": cfg.hidden_layer_sizes,
        "alpha":           cfg.alpha,
        "batch_size":      cfg.batch_size,
        "learning_rate_init": cfg.learning_rate_init,
        "max_iter":        cfg.max_iter,
        "grid_search":      int(cfg.grid_search),  # bool to int for CSV
    }


def init_run(config: Config) -> Tuple[int, Dict[str, Path]]:
    """Create output directories and allocate new run‑id."""
    run_id = _next_run_id(PARAMETERS_FILE)
    paths = {
        "model":    RES_DIR / "models" / f"mlp_model_{run_id}.joblib",
        "scaler":   RES_DIR / "models" / f"scaler_{run_id}.joblib",
        "train_plot": RES_DIR / "plots" / f"training_curve_{run_id}.pdf",
        "cm_plot":    RES_DIR / "plots" / f"confusion_matrix_{run_id}.pdf",
        "hist_plot":  RES_DIR / "plots" / f"prob_hists_{run_id}.pdf",
    }
    for p in paths.values():
        p.parent.mkdir(parents=True, exist_ok=True)
    # Save run parameters
    _write_dict_row(PARAMETERS_FILE, {"run_id": run_id, **_filter_cfg(config)})
    return run_id, paths


def log_comment(run_id: int, comment: str) -> None:
    _write_row(COMMENTS_FILE, ["run_id", "comment"], [run_id, comment])


def log_performance(run_id: int, loss: float, acc: float, auc: float) -> None:
    _write_row(PERF_FILE, ["run_id", "test_loss", "test_acc", "test_auc"],
               [run_id, loss, acc, auc])

# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def load_root_files_from_paths(files: Iterable[Path], tree: str,
                               columns: Sequence[str]) -> pd.DataFrame:
    """Lire l’arbre `tree` de chaque fichier et concaténer."""
    frames = []
    logger.info("Reading ROOT files …")
    for path in files:
        if not path.is_file():
            logger.warning("Missing file: %s", path)
            continue
        with uproot.open(path) as f:
            frames.append(f[tree].arrays(columns, library="pd"))
    if not frames:
        raise RuntimeError("No ROOT file could be opened – aborting.")
    df = pd.concat(frames, ignore_index=True)
    logger.info("Total events loaded: %d", len(df))
    return df


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Drop NaNs or infinite values then reset the index."""
    before = len(df)
    df = df.replace([np.inf, -np.inf], pd.NA).dropna().reset_index(drop=True)
    logger.info("After cleaning: %d (‑%d) events", len(df), before - len(df))
    return df


def filter_targets(df: pd.DataFrame, targets: Sequence[int]) -> pd.DataFrame:
    """Keep only the requested PDG codes and map them to [0, …]."""
    df = df[df["particlePDG"].isin(targets)].copy()
    if df.empty:
        raise RuntimeError("No entry left after PDG filtering!")
    label_map = {pdg: idx for idx, pdg in enumerate(targets)}
    df["label"] = df["particlePDG"].map(label_map)
    logger.info("Class distribution after filter: %s",
                np.bincount(df["label"].values))
    return df


def split_scale_resample(X: np.ndarray, y: np.ndarray, cfg: Config):
    """Train/test split, scale then oversample minority classes with SMOTE."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y,
    )
    logger.info("Train size: %d, Test size: %d", len(y_train), len(y_test))
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    smote = SMOTE(random_state=cfg.random_state)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
    logger.info("After SMOTE resampling: %s", np.bincount(y_train_res))
    return X_train_res, y_train_res, X_test_scaled, y_test, scaler


def build_mlp(cfg: Config) -> MLPClassifier:
    """Create an *unfitted* MLPClassifier according to Config."""
    return MLPClassifier(
        hidden_layer_sizes=cfg.hidden_layer_sizes,
        alpha=cfg.alpha,
        learning_rate="adaptive",
        learning_rate_init=cfg.learning_rate_init,
        activation="relu",
        solver="adam",
        batch_size=cfg.batch_size,
        max_iter=cfg.max_iter,
        early_stopping=True,
        n_iter_no_change=15,
        random_state=cfg.random_state,
        verbose=False,
    )


def train_model(X_train: np.ndarray, y_train: np.ndarray, cfg: Config,
                grid_search: bool = False) -> MLPClassifier:
    """Fit the MLP – with or without grid‑search."""
    base_clf = build_mlp(cfg)
    if not grid_search:
        logger.info("Training fixed‑hyper‑params MLP …")
        base_clf.verbose = True
        base_clf.fit(X_train, y_train)
        return base_clf
    logger.info("Running GridSearchCV …")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.random_state)
    grid = GridSearchCV(
        base_clf,
        cfg.param_grid,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=2,
    )
    grid.fit(X_train, y_train)
    logger.info("Best params: %s", grid.best_params_)
    return grid.best_estimator_


def plot_training_curve(clf: MLPClassifier, out_path: Path):
    fig = plt.figure(figsize=(6, 4))
    plt.plot(clf.loss_curve_, label="loss train")
    plt.plot(1 - np.array(clf.validation_scores_), "--", label="1 - score valid")
    plt.xlabel("Iteration")
    plt.ylabel("Loss / 1 - Accuracy")
    plt.legend()
    plt.title("Training loss vs. validation score")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, class_labels: Sequence[str], out_path: Path):
    cm_percent = cm.astype(float) / cm.sum(axis=1)[:, None] * 100
    annot = np.vectorize(lambda v: f"{v:.1f}%")(cm_percent)
    fig = plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm_percent,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=class_labels,
        yticklabels=class_labels,
    )
    plt.title("Confusion Matrix – MLP")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_probability_histograms(proba: np.ndarray, y_true: np.ndarray, out_path: Path):
    fig = plt.figure(figsize=(15, 4))
    # Proton probability
    ax = fig.add_subplot(1, 3, 1)
    ax.hist(proba[y_true == 0, 1], bins=50, alpha=0.5, label="π‑ (true)")
    ax.hist(proba[y_true == 1, 1], bins=50, alpha=0.5, label="p (true)")
    ax.set_xlabel("P(predicted = p)")
    ax.set_ylabel("Events")
    ax.set_title("Proton probability distribution")
    ax.legend()
    # K0 probability π‑ vs K0
    ax = fig.add_subplot(1, 3, 2)
    ax.hist(proba[y_true == 0, 2], bins=50, alpha=0.5, label="π‑ (true)")
    ax.hist(proba[y_true == 2, 2], bins=50, alpha=0.5, label="K0 (true)")
    ax.set_xlabel("P(predicted = K0)")
    ax.set_title("K0 probability (π‑ vs K0)")
    ax.legend()
    # K0 probability p vs K0
    ax = fig.add_subplot(1, 3, 3)
    ax.hist(proba[y_true == 1, 2], bins=50, alpha=0.5, label="p (true)")
    ax.hist(proba[y_true == 2, 2], bins=50, alpha=0.5, label="K0 (true)")
    ax.set_xlabel("P(predicted = K0)")
    ax.set_title("K0 probability (p vs K0)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------

def run(cfg: Config, grid_search: bool = False):
    # Initialize run and get paths
    run_id, run_paths = init_run(cfg)


    # On donne directement les fichiers à loader
    columns = list(cfg.feature_cols) + ["particlePDG"]
    df = load_root_files_from_paths(cfg.file_paths, cfg.tree_name, columns)
    df = clean_dataframe(df)
    df = filter_targets(df, cfg.targets)

    X = df[cfg.feature_cols].values
    y = df["label"].values
    X_train, y_train, X_test, y_test, scaler = split_scale_resample(X, y, cfg)

    clf = train_model(X_train, y_train, cfg, grid_search=grid_search)

    # Evaluation
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    proba_test = clf.predict_proba(X_test)
    auc = roc_auc_score(y_test, proba_test, multi_class='ovr')
    loss = clf.loss_curve_[-1] if hasattr(clf, 'loss_curve_') else float('nan')
    logger.info("\nAccuracy = %.4f", acc)
    logger.info("\n%s", classification_report(
        y_test, y_pred, target_names=["π‑", "p", "K0"],
    ))

    # Plots
    plot_training_curve(clf, run_paths['train_plot'])
    plot_confusion_matrix(cm, ["π‑", "p", "K0"], run_paths['cm_plot'])
    plot_probability_histograms(proba_test, y_test, run_paths['hist_plot'])

    # Persist artefacts
    joblib.dump(scaler, run_paths['scaler'])
    joblib.dump(clf, run_paths['model'])
    logger.info("Model + scaler saved to %s", RES_DIR / "models")

    # Log performance and comment
    log_performance(run_id, loss, acc, auc)
    log_comment(run_id, f"Finished run {run_id} with acc={acc:.4f}, auc={auc:.4f}")

# -----------------------------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train an MLP hadron classifier avec run-id tracking."
    )
    # Chemins facultatifs : si non précisés, on utilisera ceux définis dans Config
    p.add_argument(
        "--file-paths", nargs="+", metavar="FILE",
        help=(
            "Liste des chemins complets vers les fichiers ROOT à fusionner. "
            "Si absent, on utilise les chemins par défaut du script."
        ),
    )
    p.add_argument(
        "--grid-search", action="store_true",
        help="Effectuer une GridSearchCV pour optimiser les hyper‑paramètres.",
    )
    return p

def main(argv: Sequence[str] | None = None):
    args = build_arg_parser().parse_args(argv)

    # On transmet aussi --grid-search dans le config
    if args.file_paths:
        files = [Path(fp) for fp in args.file_paths]
        cfg = Config(file_paths=files, grid_search=args.grid_search)
    else:
        cfg = Config(grid_search=args.grid_search)

    run(cfg, grid_search=args.grid_search)

if __name__ == "__main__":
    main()