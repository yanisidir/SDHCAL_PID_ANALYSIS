#!/usr/bin/env python3
"""hadron_energy_regressor_MLP.py

Train an MLPRegressor to reconstruct the energy of a hadronic shower
from its shower–shape parameters.

Usage (minimal):
    python3 hadron_energy_regressor_MLP.py \
        --energy-col trueEnergy \
        --file-paths params1.root params2.root

Optional arguments:
    --grid-search   Enable a hyper‑parameter grid‑search.

Notes
-----
* Expects a ROOT ntuple with a float column holding the **true (MC) energy**
  of the primary hadron. Use `--energy-col` to override (default: `trueEnergy`).
* Define your input features in `Config.feature_cols`.
* Saves model and scaler as joblib, plus plots (training curve & pred vs true).
"""

import argparse
import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence, Tuple, Dict, Iterable

import joblib
import numpy as np
import pandas as pd
import uproot
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# -----------------------------------------------------------------------------
# Configuration dataclass
# -----------------------------------------------------------------------------
@dataclass
class Config:
    """Pipeline settings and hyper‑parameters."""

    # I/O
    file_path: str = "../../analyse_pi/data/130k_pi_E1to130_params.root"
    tree_name: str = "paramsTree"
    energy_col: str = "primaryEnergy"  # target column in ROOT containing MC energy  # target column in ROOT

    # Grid‑search option
    grid_search: bool = False

    # Train/test split
    test_size: float = 0.20
    random_state: int = 42

    # Input features
    feature_cols: Sequence[str] = field(default_factory=lambda: [
        "Thr1", "Thr2", "Thr3", "Radius", "Density", "NClusters",
        "ratioThr23", "Zbary", "Zrms", "PctHitsFirst10",
        "PlanesWithClusmore2", "AvgClustSize", "MaxClustSize",
        "Zbary_thr3", "Zbary_thr2", "tMin", "tMax", "tMean",
        "tSpread", "N1", "N2", "N3"
    ])

    # MLP hyper‑parameters
    hidden_layer_sizes: Tuple[int, ...] = (128, 64, 32)
    alpha: float = 1e-4
    batch_size: int = 32
    learning_rate_init: float = 1e-3
    max_iter: int = 500

    # Grid‑search ranges
    param_grid: Dict[str, List] = field(default_factory=lambda: {
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

# Run‑id utilities
PARAMETERS_FILE = Path("run_parameters_regression.csv")
COMMENTS_FILE   = Path("run_comments_regression.csv")
PERF_FILE       = Path("hadron_energy_performances.csv")
RES_DIR         = Path("results_energy_reco")


def _next_run_id(csv_path: Path) -> int:
    if not csv_path.exists():
        return 1
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        try:
            return max(int(r["run_id"]) for r in reader) + 1
        except ValueError:
            return 1


def _write_dict_row(csv_path: Path, row: Dict[str, object]) -> None:
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


def init_run(config: Config) -> Tuple[int, Dict[str, Path]]:
    run_id = _next_run_id(PARAMETERS_FILE)
    paths = {
        "model":    RES_DIR / "models" / f"mlp_regressor_{run_id}.joblib",
        "scaler":   RES_DIR / "models" / f"scaler_{run_id}.joblib",
        "train_plot": RES_DIR / "plots" / f"training_curve_{run_id}.pdf",
        "scatter_plot": RES_DIR / "plots" / f"pred_vs_true_{run_id}.pdf",
    }
    for p in paths.values():
        p.parent.mkdir(parents=True, exist_ok=True)
    cfg_dict = {
        "file_path":         config.file_path),
        "tree_name":         config.tree_name,
        "energy_col":        config.energy_col,
        "test_size":         config.test_size,
        "random_state":      config.random_state,
        "hidden_layers":     config.hidden_layer_sizes,
        "alpha":             config.alpha,
        "batch_size":        config.batch_size,
        "learning_rate_init":config.learning_rate_init,
        "max_iter":          config.max_iter,
        "grid_search":       int(config.grid_search),
    }
    _write_dict_row(PARAMETERS_FILE, {"run_id": run_id, **cfg_dict})
    return run_id, paths

# Data loading & preprocessing

def load_root(files: Iterable[Path], tree: str, cols: List[str]) -> pd.DataFrame:
    dfs = []
    logger.info("Reading ROOT files…")
    for p in files:
        if not p.is_file():
            logger.warning("Missing file: %s", p)
            continue
        with uproot.open(p) as f:
            dfs.append(f[tree].arrays(cols, library="pd"))
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


def split_and_scale(X: np.ndarray, y: np.ndarray, cfg: Config):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state
    )
    logger.info("Train size: %d, Test size: %d", len(y_tr), len(y_te))
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    return X_tr_s, y_tr, X_te_s, y_te, scaler

# Model building & training

def build_mlp(cfg: Config) -> MLPRegressor:
    return MLPRegressor(
        hidden_layer_sizes=cfg.hidden_layer_sizes,
        alpha=cfg.alpha,
        batch_size=cfg.batch_size,
        learning_rate_init=cfg.learning_rate_init,
        learning_rate="adaptive",
        solver="adam",
        max_iter=cfg.max_iter,
        early_stopping=True,
        n_iter_no_change=15,
        random_state=cfg.random_state,
        verbose=False,
    )


def train_model(X_tr: np.ndarray, y_tr: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
                cfg: Config) -> Tuple[MLPRegressor, Dict[str, list]]:
    if not cfg.grid_search:
        reg = build_mlp(cfg)
        reg.verbose = True
        reg.fit(X_tr, y_tr)
    else:
        logger.info("Running GridSearchCV…")
        base = build_mlp(cfg)
        cv = KFold(n_splits=5, shuffle=True, random_state=cfg.random_state)
        grid = GridSearchCV(
            base, cfg.param_grid, cv=cv,
            scoring="neg_root_mean_squared_error", n_jobs=-1, verbose=2
        )
        grid.fit(X_tr, y_tr)
        logger.info("Best params: %s", grid.best_params_)
        reg = grid.best_estimator_
    # No direct evals_result_; we track loss_curve_
    return reg, {}

# Evaluation & plotting

def evaluate(reg: MLPRegressor, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[float, float, float, np.ndarray]:
    y_pred = reg.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    logger.info("RMSE: %.4f, MAE: %.4f, R²: %.4f", rmse, mae, r2)
    return rmse, mae, r2, y_pred


def plot_training_curve(reg: MLPRegressor, out: Path):
    plt.figure(figsize=(6, 4))
    plt.plot(reg.loss_curve_, label="train loss")
    plt.xlabel("Iteration"); plt.ylabel("MSE loss")
    plt.legend(); plt.title("Training loss curve")
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()


def plot_pred_vs_true(y_true: np.ndarray, y_pred: np.ndarray, out: Path):
    plt.figure(figsize=(5, 5))
    sns.scatterplot(x=y_true, y=y_pred, s=10)
    lim = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lim, lim, "r--")
    plt.xlabel("True energy"); plt.ylabel("Predicted energy")
    plt.title("Energy reconstruction (MLP)")
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()

# Main pipeline

def run(cfg: Config):
    run_id, paths = init_run(cfg)

    # Load & preprocess
    cols = list(cfg.feature_cols) + [cfg.energy_col]
    df = load_root(cfg.file_path, cfg.tree_name, cols)
    df = clean_df(df)
    X = df[cfg.feature_cols].values.astype(np.float32)
    y = df[cfg.energy_col].values.astype(np.float32)
    X_tr, y_tr, X_te, y_te, scaler = split_and_scale(X, y, cfg)

    # Train
    reg, _ = train_model(X_tr, y_tr, X_te, y_te, cfg)

    # Evaluate
    rmse, mae, r2, y_pred = evaluate(reg, X_te, y_te)

    # Plots
    plot_training_curve(reg, paths["train_plot"])
    plot_pred_vs_true(y_te, y_pred, paths["scatter_plot"])

    # Save artifacts
    joblib.dump(scaler, paths["scaler"])
    joblib.dump(reg, paths["model"])
    logger.info("Artifacts saved to %s", RES_DIR)

    # Log
    log_performance(run_id, rmse, mae, r2)
    log_comment(run_id, f"Run {run_id}: rmse={rmse:.4f}, mae={mae:.4f}, r2={r2:.4f}")

# CLI

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train an MLP regressor for hadron energy.")
    p.add_argument(
        "--grid-search",
        action="store_true",
        help="Enable hyperparameter grid-search."
    )
    return p


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    cfg = Config(grid_search=args.grid_search)
    run(cfg)

if __name__ == "__main__":
    main()
