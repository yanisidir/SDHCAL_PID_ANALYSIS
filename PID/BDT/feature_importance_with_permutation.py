#!/usr/bin/env python3
"""feature_importance_modular.py

Modular script to compute and visualize feature correlations and importances
from a trained LGBM classifier, including permutation importances.

Usage:
    python3 feature_importance_modular.py \
      [--file-paths f1.root f2.root f3.root] \
      [--model processed_data/lgbm_classifier.joblib] \
      [--scaler processed_data/scaler.joblib] \
      [--output-dir results] \
      [--tree-name paramsTree]"""

import argparse
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence

import joblib
import numpy as np
import pandas as pd
import uproot
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

# -----------------------------------------------------------------------------
# Configuration dataclass
# -----------------------------------------------------------------------------
@dataclass
class Config:
    """Settings and hyper-parameters."""
    # Input ROOT files
    file_paths: List[Path] = field(default_factory=lambda: [
        Path("../../analyse_pi/data/130k_pi_E1to130_params.root"),
        Path("../../analyse_kaon/data/130k_kaon_E1to130_params.root"),
        Path("../../analyse_proton/data/130k_proton_E1to130_params.root"),
    ])
    tree_name: str = "paramsTree"

    # Predefined features and PDG targets
    feature_cols: Sequence[str] = field(default_factory=lambda: ["Thr1", "Thr2", "Thr3",  "Begin", "Radius", "Density", 
    "NClusters", "ratioThr23", "Zbary", "Zrms",     "PctHitsFirst10", "PlanesWithClusmore2", "AvgClustSize", "MaxClustSize", 
    "Zbary_thr3", "Zbary_thr2", "tMin", "tMax", "tMean", "tSpread"])
    targets: Sequence[int] = field(default_factory=lambda: [-211, 2212, 311])

    # Model and scaler paths
    model_path: Path = Path("results_with_time/models/lgbm_model_5.joblib")
    scaler_path: Path = Path("results_with_time/models/scaler_5.joblib")

    # Output
    output_dir: Path = Path("results_with_time/plots")
    test_size: float = 0.20
    random_state: int = 42

# -----------------------------------------------------------------------------
# Logging setup
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Data loading and preprocessing
# -----------------------------------------------------------------------------
def load_and_concat_roots(paths: Sequence[Path], tree: str, cols: Sequence[str]) -> pd.DataFrame:
    """Load branches from multiple ROOT files and concatenate into a DataFrame."""
    dfs = []
    logger.info("Reading ROOT files...")
    for p in paths:
        if not p.is_file():
            logger.warning("Missing file: %s", p)
            continue
        with uproot.open(p) as f:
            dfs.append(f[tree].arrays(cols, library='pd'))
    if not dfs:
        raise FileNotFoundError("No valid ROOT files found.")
    df = pd.concat(dfs, ignore_index=True)
    logger.info("Total events loaded: %d", len(df))
    return df


def clean_and_filter(df: pd.DataFrame, targets: Sequence[int]) -> pd.DataFrame:
    """Clean NaN/inf values and filter target PDG codes, mapping to labels."""
    before = len(df)
    df = df.replace([np.inf, -np.inf], pd.NA).dropna().reset_index(drop=True)
    logger.info("After cleaning: %d (-%d) events", len(df), before - len(df))
    df = df[df['primaryID'].isin(targets)].copy()
    if df.empty:
        raise RuntimeError("No entries after PDG filtering.")
    label_map = {pdg: idx for idx, pdg in enumerate(targets)}
    df['label'] = df['primaryID'].map(label_map)
    logger.info("Class distribution: %s", np.bincount(df['label']))
    return df

# -----------------------------------------------------------------------------
# Correlation plotting
# -----------------------------------------------------------------------------
def plot_correlation(df: pd.DataFrame, features: Sequence[str], out_path: Path) -> None:
    """Plot and save feature correlation heatmap."""
    corr = df[features].corr()
    plt.figure(figsize=(15, 12))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
                xticklabels=features, yticklabels=features)
    plt.title('Feature Correlation Matrix')
    plt.xticks(rotation=90); plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info("Saved correlation matrix to %s", out_path)

# -----------------------------------------------------------------------------
# Feature importances
# -----------------------------------------------------------------------------
def compute_and_save_importances(
    clf, scaler, X: np.ndarray, y: np.ndarray,
    features: Sequence[str], cfg: Config
) -> None:
    """Generate native and permutation importances, save CSV and plots."""
    # split & scale test set
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=cfg.test_size,
        random_state=cfg.random_state, stratify=y
    )
    X_test_scaled = scaler.transform(X_test)

    # native importances
    imp = clf.feature_importances_
    df_imp = pd.DataFrame({'feature': features, 'importance': imp})
    df_imp = df_imp.sort_values('importance', ascending=False)
    csv_native = cfg.output_dir / 'feature_importances.csv'
    df_imp.to_csv(csv_native, index=False)
    logger.info("Saved native importances to %s", csv_native)
    # plot
    plt.figure(figsize=(10, 8))
    plt.barh(df_imp['feature'][::-1], df_imp['importance'][::-1])
    plt.xlabel('Importance'); plt.title('LGBM Feature Importances')
    plt.tight_layout()
    pdf_native = cfg.output_dir / 'feature_importances.pdf'
    plt.savefig(pdf_native, dpi=150); plt.close()
    logger.info("Saved native importance plot to %s", pdf_native)

    # permutation importances
    perm = permutation_importance(
        clf, X_test_scaled, y_test,
        n_repeats=10, random_state=cfg.random_state, n_jobs=-1
    )
    df_perm = pd.DataFrame({'feature': features, 'perm_importance': perm.importances_mean})
    df_perm = df_perm.sort_values('perm_importance', ascending=False)
    csv_perm = cfg.output_dir / 'permutation_importances.csv'
    df_perm.to_csv(csv_perm, index=False)
    logger.info("Saved permutation importances to %s", csv_perm)
    # plot
    plt.figure(figsize=(10, 8))
    plt.barh(df_perm['feature'][::-1], df_perm['perm_importance'][::-1])
    plt.xlabel('Permutation Importance'); plt.title('Permutation Importances')
    plt.tight_layout()
    pdf_perm = cfg.output_dir / 'permutation_importances.pdf'
    plt.savefig(pdf_perm, dpi=150); plt.close()
    logger.info("Saved permutation importance plot to %s", pdf_perm)

# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------
def run(cfg: Config) -> None:
    # ensure output directory
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    # load & preprocess data
    cols = list(cfg.feature_cols) + ['primaryID']
    df = load_and_concat_roots(cfg.file_paths, cfg.tree_name, cols)
    df = clean_and_filter(df, cfg.targets)

    # correlation
    corr_pdf = cfg.output_dir / 'correlation_matrix.pdf'
    plot_correlation(df, cfg.feature_cols, corr_pdf)

    # load model and scaler
    clf = joblib.load(cfg.model_path)
    scaler = joblib.load(cfg.scaler_path)

    # features/labels
    X = df[cfg.feature_cols].values
    y = df['label'].values

    # importances
    compute_and_save_importances(clf, scaler, X, y, cfg.feature_cols, cfg)

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compute feature correlations and importances.")
    p.add_argument(
        "--file-paths", nargs='+', metavar='FILE',
        help="Paths to ROOT files."
    )
    p.add_argument(
        "--model", type=Path, help="Path to trained LGBM model joblib."
    )
    p.add_argument(
        "--scaler", type=Path, help="Path to scaler joblib."
    )
    p.add_argument(
        "--output-dir", type=Path, help="Directory for outputs."
    )
    return p


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    cfg = Config(
        file_paths=[Path(p) for p in args.file_paths] if args.file_paths else Config().file_paths,
        model_path=args.model or Config().model_path,
        scaler_path=args.scaler or Config().scaler_path,
        output_dir=args.output_dir or Config().output_dir
    )
    run(cfg)

if __name__ == '__main__':
    main()
