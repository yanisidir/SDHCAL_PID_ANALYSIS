#!/usr/bin/env python3
"""hadron_energy_regressor_LGBM_refactor.py

Train a LightGBM regressor to reconstruct the energy of a hadronic shower
from shower–shape parameters.

"""
from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import uproot
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic, norm

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
#  run configuration
# =============================================================================
@dataclass
class Config:
    """Pipeline settings and hyper‑parameters ()."""
    # I/O
    file_path: str = (
        
        "/gridgroup/ilc/midir/analyse/data/merged_primaryEnergy/130k_proton_E1to130_params_merged.root"
    )
    tree_name: str = "paramsTree"
    energy_col: str = "primaryEnergy"  # MC truth energy

    # Train/test split
    test_size: float = 0.20
    random_state: int = 42

    # Input features
    feature_cols: Sequence[str] = field(
        default_factory=lambda: [
            "Density",
            "NClusters",
            "Zbary",
            "AvgClustSize",
            "N3",
            "N2",
            "N1",
            "tMin",
            "nHitsTotal",
            "sumThrTotal",
        ]
    )

    # LGBM hyperparameters
    learning_rate: float = 0.05
    n_estimators: int = 2000
    num_leaves: int = 127
    max_depth: int = -1
    reg_alpha: float = 0.0
    reg_lambda: float = 0.1
    min_child_samples: int = 20
    subsample: float = 0.8
    colsample_bytree: float = 0.8

    # Early stopping rounds
    early_stopping_rounds: int = 50

    # Grid‑search (stays  here; set to True to enable)
    grid_search: bool = False
    param_grid: Dict[str, List] = field(
        default_factory=lambda: {
            "num_leaves": [63, 127, 255],
            "max_depth": [-1, 8, 12],
            "learning_rate": [0.01, 0.05, 0.1],
            "n_estimators": [1000, 2000, 4000],
        }
    )

# Output locations (, per-particle flavor)
PARAMETERS_FILE = Path("run_parameters_regression_proton.csv")
COMMENTS_FILE = Path("run_comments_regression_proton.csv")
PERF_FILE = Path("hadron_energy_performances_proton.csv")
RES_DIR = Path("results_proton_energy_reco")

# Colors for multi‑plots (kept minimal; extend if needed)
PARTICLE_COLORS: Dict[str, str] = {
    "pi": "C0",
    "proton": "C1",
    "proton": "C2",
}

# =============================================================================
# Run id & CSV logging helpers
# =============================================================================

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
    return {
        "file_path": cfg.file_path,
        "tree_name": cfg.tree_name,
        "energy_col": cfg.energy_col,
        "test_size": cfg.test_size,
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
        "grid_search": int(cfg.grid_search),
        "early_stopping_rounds": cfg.early_stopping_rounds,
    }


def init_run(config: Config) -> tuple[int, Dict[str, Path]]:
    run_id = _next_run_id(PARAMETERS_FILE)
    paths = {
        "model": RES_DIR / "models" / f"lgbm_regressor_proton_{run_id}.joblib",
        "scaler": RES_DIR / "models" / f"scaler_proton_{run_id}.joblib",
        "train_plot": RES_DIR / "plots" / f"training_curve_{run_id}.png",
        "scatter": RES_DIR / "plots" / f"pred_vs_true_{run_id}.png",
        "linearity": RES_DIR / "plots" / f"linearity_profile_{run_id}.png",
        "resolution": RES_DIR / "plots" / f"resolution_{run_id}.png",
        "combo": RES_DIR / "plots" / f"linearity_dev_combo_{run_id}.png",
    }
    for p in paths.values():
        p.parent.mkdir(parents=True, exist_ok=True)
    _write_dict_row(PARAMETERS_FILE, {"run_id": run_id, **_filter_cfg(config)})
    return run_id, paths


def log_comment(run_id: int, comment: str) -> None:
    _write_row(COMMENTS_FILE, ["run_id", "comment"], [run_id, comment])


def log_performance(run_id: int, rmse: float, mae: float, r2: float, sigma_rel: float) -> None:
    _write_row(
        PERF_FILE,
        ["run_id", "test_rmse", "test_mae", "test_r2", "sigma_rel"],
        [run_id, rmse, mae, r2, sigma_rel],
    )

# =============================================================================
# Data IO & preprocessing
# =============================================================================

def load_root(file: str | Path, tree: str, cols: Sequence[str]) -> pd.DataFrame:
    """Load a single ROOT TTree into a pandas DataFrame."""
    path = Path(file)
    logger.info("Reading ROOT file: %s", path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    with uproot.open(path) as f:
        df = f[tree].arrays(cols, library="pd")
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

# =============================================================================
# Model
# =============================================================================

def build_lgbm(cfg: Config) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="regression",
        learning_rate=cfg.learning_rate,
        n_estimators=cfg.n_estimators,
        num_leaves=cfg.num_leaves,
        max_depth=cfg.max_depth,
        reg_alpha=cfg.reg_alpha,
        reg_lambda=cfg.reg_lambda,
        min_child_samples=cfg.min_child_samples,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        random_state=cfg.random_state,
        n_jobs=-1,
        metric="l2",
    )


def train_model(
    X_tr: np.ndarray, y_tr: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, cfg: Config
) -> tuple[lgb.LGBMRegressor, Dict[str, List[float]]]:
    if not cfg.grid_search:
        reg = build_lgbm(cfg)
        reg.fit(
            X_tr,
            y_tr,
            eval_set=[(X_tr, y_tr), (X_val, y_val)],
            eval_names=["train", "valid"],
            eval_metric="l2",
            callbacks=[
                lgb.early_stopping(cfg.early_stopping_rounds),
                lgb.log_evaluation(period=100),
            ],
        )
        evals = reg.evals_result_
    else:
        logger.info("Running GridSearchCV…")
        base = build_lgbm(cfg)
        cv = KFold(n_splits=5, shuffle=True, random_state=cfg.random_state)
        grid = GridSearchCV(
            base,
            cfg.param_grid,
            cv=cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
            verbose=1,
            refit=True,
        )
        grid.fit(X_tr, y_tr)
        best_params = grid.best_params_
        logger.info("Best params: %s", best_params)
        reg = build_lgbm(cfg)
        reg.set_params(**best_params)
        reg.fit(
            X_tr,
            y_tr,
            eval_set=[(X_tr, y_tr), (X_val, y_val)],
            eval_names=["train", "valid"],
            eval_metric="l2",
            callbacks=[
                lgb.early_stopping(cfg.early_stopping_rounds),
                lgb.log_evaluation(period=100),
            ],
        )
        evals = reg.evals_result_
    return reg, evals

# =============================================================================
# Metrics & binning
# =============================================================================

def compute_global_resolution(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    residual = (y_pred - y_true) / y_true
    sigma = np.sqrt(np.mean(residual ** 2))  # RMS
    return float(sigma)


def bin_resolution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    nbins: int = 13,
    e_min: float | None = None,
    e_max: float | None = None,
    use_gauss_fit: bool = False,
    frac: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute σ(E)/E vs E_true.

    - use_gauss_fit: gaussian fit on residuals; else RMS (or truncated RMS if frac<1)
    - frac: central fraction (0<frac<=1) for truncated RMS (e.g. 0.9)
    """
    e = y_true
    r = (y_pred - y_true) / y_true

    if e_min is None:
        e_min = float(e.min())
    if e_max is None:
        e_max = float(e.max())

    bins = np.linspace(e_min, e_max, nbins + 1)
    centers, sigmas = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (e >= lo) & (e < hi)
        if mask.sum() < 20:
            continue
        res_bin = r[mask]
        if use_gauss_fit:
            mu, sigma = norm.fit(res_bin)
        else:
            if frac < 1.0:
                q_lo, q_hi = np.quantile(res_bin, [(1 - frac) / 2, 1 - (1 - frac) / 2])
                res_bin = res_bin[(res_bin >= q_lo) & (res_bin <= q_hi)]
            sigma = np.sqrt(np.mean(res_bin ** 2))
        centers.append(0.5 * (lo + hi))
        sigmas.append(float(sigma))
    return np.array(centers), np.array(sigmas)

# =============================================================================
# Plotting
# =============================================================================

def _save_fig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_training_curve(train_loss: List[float], valid_loss: List[float], out: Path) -> None:
    plt.figure(figsize=(6, 4))
    plt.plot(train_loss, label="train l2")
    plt.plot(valid_loss, "--", label="valid l2")
    plt.xlabel("Iteration")
    plt.ylabel("L2 loss")
    plt.legend()
    plt.title("Training vs validation loss")
    _save_fig(out)


def plot_pred_vs_true(y_true: np.ndarray, y_pred: np.ndarray, out: Path) -> None:
    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, y_pred, s=10)
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    plt.plot([lo, hi], [lo, hi], "r--", linewidth=1)
    plt.xlim(lo, hi)
    plt.ylim(lo, hi)
    plt.xlabel("True energy [GeV]")
    plt.ylabel("Predicted energy [GeV]")
    plt.title("Energy reconstruction")
    _save_fig(out)


def plot_profile_linearity(y_true: np.ndarray, y_pred: np.ndarray, out: Path, nbins: int = 20) -> None:
    plt.figure(figsize=(6, 6))
    bins = np.linspace(y_true.min(), y_true.max(), nbins + 1)
    centers = 0.5 * (bins[1:] + bins[:-1])
    mean, _, _ = binned_statistic(y_true, y_pred, statistic="mean", bins=bins)
    std, _, _ = binned_statistic(y_true, y_pred, statistic="std", bins=bins)
    cnt, _, _ = binned_statistic(y_true, y_pred, statistic="count", bins=bins)
    err = np.divide(std, np.sqrt(np.maximum(cnt, 1)), where=cnt > 0)
    m = cnt > 0
    plt.errorbar(centers[m], mean[m], yerr=err[m], fmt="o", markersize=4, capsize=3, label="Profile")
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    plt.plot([lo, hi], [lo, hi], "r--", linewidth=1, label="Ideal")
    plt.xlim(lo, hi)
    plt.ylim(lo, hi)
    plt.xlabel("True energy [GeV]")
    plt.ylabel("Reconstructed energy [GeV]")
    plt.title("Linearity profile")
    plt.legend()
    _save_fig(out)


def plot_resolution_curve(y_true: np.ndarray, y_pred: np.ndarray, out: Path, nbins: int = 13) -> None:
    centers, sigmas = bin_resolution(y_true, y_pred, nbins=nbins, use_gauss_fit=False, frac=0.9)
    plt.figure(figsize=(6, 4))
    plt.plot(centers, sigmas, marker="o")
    plt.xlabel("True energy [GeV]")
    plt.ylabel("σ(E)/E")
    plt.title("Energy resolution")
    _save_fig(out)


def plot_linearity_and_deviation_combo(y_true: np.ndarray, y_pred: np.ndarray, out: Path, nbins: int = 20) -> None:
    import matplotlib.gridspec as gridspec

    bins = np.linspace(y_true.min(), y_true.max(), nbins + 1)
    centers = 0.5 * (bins[1:] + bins[:-1])
    mean, _, _ = binned_statistic(y_true, y_pred, statistic="mean", bins=bins)
    std, _, _ = binned_statistic(y_true, y_pred, statistic="std", bins=bins)
    cnt, _, _ = binned_statistic(y_true, y_pred, statistic="count", bins=bins)
    sem = np.divide(std, np.sqrt(np.maximum(cnt, 1)), where=cnt > 0)

    with np.errstate(divide="ignore", invalid="ignore"):
        dev_mean = (mean - centers) / centers
        dev_err = sem / centers

    m = cnt > 0
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))

    fig = plt.figure(figsize=(7, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[2.5, 1.0], hspace=0.05)

    # Top: linearity
    ax1 = fig.add_subplot(gs[0])
    ax1.errorbar(centers[m], mean[m], yerr=sem[m], fmt="o", markersize=4, capsize=3, label="Profile")
    ax1.plot([lo, hi], [lo, hi], "r--", linewidth=1, label="Ideal")
    ax1.set_ylabel("Reconstructed energy [GeV]")
    ax1.set_xlim(lo, hi)
    ax1.set_ylim(lo, hi)
    ax1.set_title("Linearity (top) & Relative deviation (bottom)")
    ax1.legend(loc="best")
    ax1.grid(True)

    # Bottom: relative deviation
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    mb = m & np.isfinite(dev_mean) & np.isfinite(dev_err)
    ax2.errorbar(centers[mb], dev_mean[mb], yerr=dev_err[mb], fmt="o", markersize=4, capsize=3)
    ax2.axhline(0.0, linestyle="--", linewidth=1)
    ax2.set_xlabel("True energy [GeV]")
    ax2.set_ylabel("(⟨E_pred⟩-E_true)/E_true")
    ax2.grid(True)

    _save_fig(out)

# =============================================================================
# Evaluation
# =============================================================================

def evaluate(reg: lgb.LGBMRegressor, X_test: np.ndarray, y_test: np.ndarray):
    y_pred = reg.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    sigma_rel = compute_global_resolution(y_test, y_pred)
    logger.info("RMSE: %.4f, MAE: %.4f, R²: %.4f, σ(E)/E: %.4f", rmse, mae, r2, sigma_rel)
    return rmse, mae, r2, sigma_rel, y_pred

# =============================================================================
# Main pipeline
# =============================================================================

def run(cfg: Config) -> None:
    run_id, paths = init_run(cfg)

    # Load & preprocess
    cols = list(cfg.feature_cols) + [cfg.energy_col]
    df = load_root(cfg.file_path, cfg.tree_name, cols)
    df = clean_df(df)

    X = df[cfg.feature_cols].to_numpy(dtype=np.float32)
    y = df[cfg.energy_col].to_numpy(dtype=np.float32)

    X_tr, y_tr, X_te, y_te, scaler = split_and_scale(X, y, cfg)

    # Train
    reg, evals = train_model(X_tr, y_tr, X_te, y_te, cfg)

    # Evaluate
    rmse, mae, r2, sigma_rel, y_pred = evaluate(reg, X_te, y_te)

    # Plots
    plot_training_curve(evals["train"]["l2"], evals["valid"]["l2"], paths["train_plot"])
    plot_pred_vs_true(y_te, y_pred, paths["scatter"])
    plot_profile_linearity(y_te, y_pred, paths["linearity"])  # useful summary
    plot_resolution_curve(y_te, y_pred, paths["resolution"])   # σ(E)/E vs E
    plot_linearity_and_deviation_combo(y_te, y_pred, paths["combo"])  # 2-in-1 figure

    # Save artifacts
    joblib.dump(scaler, paths["scaler"])
    joblib.dump(reg, paths["model"])
    logger.info("Artifacts saved to %s", RES_DIR)

    # Log
    log_performance(run_id, rmse, mae, r2, sigma_rel)
    log_comment(run_id, f"Run {run_id}: rmse={rmse:.4f}, mae={mae:.4f}, r2={r2:.4f}, sigma_rel={sigma_rel:.4f}")


def main() -> None:
    """ entrypoint (no CLI)."""
    cfg = Config()
    run(cfg)


if __name__ == "__main__":
    main()
