#!/usr/bin/env python3
"""
Pipeline d’entraînement et d’évaluation pour la reconstruction d’énergie hadronique
(π⁻, K⁰_L, p) avec LightGBM.

Fonctions clés :
- Chargement des données depuis un fichier ROOT (via uproot) et sélection des features
- Nettoyage des NaN/inf, split train/val/test, standardisation optionnelle
- Entraînement en log(E) avec early stopping et (optionnel) GridSearchCV
- Métriques de test : RMSE, MAE, R², σ(E)/E (résolution globale)
- Sauvegarde des artefacts : modèle .joblib, scaler, courbes d’entraînement, profils
  linéarité/résolution/biais, importances de variables
- Journalisation des paramètres et performances dans des CSV
- Export des couples (y_true, y_pred) en .npz pour tracés externes

Entrées/Sorties :
- Entrée : fichiers ROOT définis dans PARTICLE_CFG (paramsTree)
- Sorties par particule (répertoire `results_*_energy_reco/`) :
  - models/: `{lgbm_regressor_<tag>_<run_id>.joblib}`, `scaler_<tag>_<run_id>.joblib`
  - plots/: `training_curve_*.png`, `pred_vs_true_*.png`, `resolution_curve_*.png`,
            `relative_deviation_*.png`, `feat_importance_*.png`
  - arrays/: `test_and_pred_<run_id>.npz`
  - CSV : paramètres, performances, commentaires

CLI :
    python hadron_energy_reco_lgbm.py --particles pi kaon proton
    python hadron_energy_reco_lgbm.py --particles all
    python hadron_energy_reco_lgbm.py --particles pi --grid-search

"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import uproot
from scipy import stats
from scipy.stats import binned_statistic
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_FMT = "%(asctime)s | %(levelname)8s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT, datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Paths:
    file_path: str
    res_dir: Path
    param_csv: Path
    perf_csv: Path
    comm_csv: Path
    tag: str


@dataclass
class GlobalCfg:
    # Data
    tree_name: str = "paramsTree"
    energy_col: str = "primaryEnergy"
    feature_cols: Sequence[str] = field(
        default_factory=lambda: [
            "Density",
            "NClusters",
            "Zbary",
            "AvgClustSize",
            "N3",
            "N2",
            "N1",
            # "tMin",
            "nHitsTotal",
            "sumThrTotal",
        ]
    )

    # feature_cols: Sequence[str] = field(default_factory=lambda: [
    #     "Thr1", "Thr2", "Thr3", "Begin", "Radius", "Density", "NClusters", "ratioThr23", "Zbary", "Zrms",
    #     "PctHitsFirst10", "PlanesWithClusmore2", "AvgClustSize", "MaxClustSize", "lambda1", "lambda2",
    #     "tMin", "tMax", "tMean", "tSpread", "Nmax", "z0_fit", "Xmax", "lambda",
    #     "nTrackSegments", "eccentricity3D"
    # ])

    # Split
    test_size: float = 0.20
    random_state: int = 42

    # LightGBM defaults
    learning_rate: float = 0.05
    n_estimators: int = 2000
    num_leaves: int = 127
    max_depth: int = -1
    reg_alpha: float = 0.0
    reg_lambda: float = 0.1
    min_child_samples: int = 20
    subsample: float = 0.8
    colsample_bytree: float = 0.8

    # Early stopping
    early_stopping_rounds: int = 50

    # Grid-search
    grid_search: bool = False
    param_grid: Dict[str, List] = field(
        default_factory=lambda: {
            "num_leaves": [63, 127, 255],
            "max_depth": [-1, 8, 12],
            "learning_rate": [0.01, 0.05, 0.1],
            "n_estimators": [1000, 2000, 4000],
        }
    )


# Particle specific configuration (unchanged paths)
PARTICLE_CFG: Mapping[str, Paths] = {
    "pi": Paths(
        file_path="/gridgroup/ilc/midir/analyse/data/merged_primaryEnergy/130k_pi_E1to130_params_merged.root",
        res_dir=Path("results_pi_energy_reco"),
        param_csv=Path("parameters/run_parameters_regression_pi.csv"),
        perf_csv=Path("performances/hadron_energy_performances_pi.csv"),
        comm_csv=Path("comments/run_comments_regression_pi.csv"),
        tag="pi",
    ),
    "kaon": Paths(
        file_path="/gridgroup/ilc/midir/analyse/data/merged_primaryEnergy/130k_kaon_E1to130_params_merged.root",
        res_dir=Path("results_kaon_energy_reco"),
        param_csv=Path("parameters/run_parameters_regression_kaon.csv"),
        perf_csv=Path("performances/hadron_energy_performances_kaon.csv"),
        comm_csv=Path("comments/run_comments_regression_kaon.csv"),
        tag="kaon",
    ),
    "proton": Paths(
        file_path="/gridgroup/ilc/midir/analyse/data/merged_primaryEnergy/130k_proton_E1to130_params_merged.root",
        res_dir=Path("results_proton_energy_reco"),
        param_csv=Path("parameters/run_parameters_regression_proton.csv"),
        perf_csv=Path("performances/hadron_energy_performances_proton.csv"),
        comm_csv=Path("comments/run_comments_regression_proton.csv"),
        tag="proton",
    ),
}

# Colors for multi-plots
PARTICLE_COLORS: Mapping[str, str] = {"pi": "red", "proton": "blue", "kaon": "green"}


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_global_seeds(seed: int = 42) -> None:
    """Set deterministic seeds for Python, NumPy and hash seed.

    Note: LightGBM CPU with multithreading can still be slightly non-deterministic.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# Small CSV helpers
# ---------------------------------------------------------------------------

def _next_run_id(csv_path: Path) -> int:
    if not csv_path.exists():
        return 1
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        try:
            return max(int(r["run_id"]) for r in reader) + 1
        except ValueError:
            return 1


def _write_dict_row(csv_path: Path, row: MutableMapping[str, object]) -> None:
    is_new = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def _write_row(csv_path: Path, header: Sequence[str], row: Sequence[object]) -> None:
    is_new = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(list(header))
        writer.writerow(list(row))


def _filter_cfg_for_csv(cfg: GlobalCfg) -> Dict[str, object]:
    d = asdict(cfg).copy()
    d.pop("feature_cols", None)
    d.pop("param_grid", None)
    return d


# ---------------------------------------------------------------------------
# Data I/O & preparation
# ---------------------------------------------------------------------------

def load_root(file: str | Path, tree: str, cols: Sequence[str]) -> pd.DataFrame:
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


def smear_column_gaussian(df: pd.DataFrame, col: str, sigma: float) -> pd.DataFrame:
    """Apply stochastic Gaussian smearing to a numeric column (in-place)."""
    df[col] = df[col].to_numpy() + np.random.normal(0.0, sigma, size=len(df))
    return df


def split_val_test(X: np.ndarray, y: np.ndarray, cfg: GlobalCfg, valid_size: float = 0.1):
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state
    )
    # valid sur un morceau de X_train
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=valid_size, random_state=cfg.random_state
    )
    # NB: LightGBM n’a pas besoin de scaler, mais si tu tiens à le garder :
    scaler = StandardScaler()
    X_tr_s  = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val)
    X_te_s  = scaler.transform(X_tmp)
    return X_tr_s, y_tr, X_val_s, y_val, X_te_s, y_tmp, scaler


# ---------------------------------------------------------------------------
# Modeling
# ---------------------------------------------------------------------------

def build_lgbm(cfg: GlobalCfg) -> lgb.LGBMRegressor:
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


def rel_rmse_logspace(y_true_log: np.ndarray, y_pred_log: np.ndarray):
    """Relative RMSE evaluated in linear space while training in log-space.

    Returns the LightGBM custom metric tuple: (name, value, is_higher_better)
    """
    y = np.exp(y_true_log)
    yhat = np.exp(y_pred_log)
    r = (yhat - y) / (y + 1e-12)
    return ("rel_rmse", float(np.sqrt(np.mean(r * r))), False)


def train_model(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: GlobalCfg,
):
    # work in log(E)
    y_tr_z = np.log(y_tr + 1e-9)
    y_val_z = np.log(y_val + 1e-9)

    if not cfg.grid_search:
        reg = build_lgbm(cfg)
        reg.fit(
            X_tr, y_tr_z,
            eval_set=[(X_val, y_val_z)],
            eval_names=["train", "valid"],
            eval_metric=[rel_rmse_logspace, "l2"],
            callbacks=[
                lgb.early_stopping(cfg.early_stopping_rounds, first_metric_only=True),
                lgb.log_evaluation(period=100),
            ],
        )
        return reg, reg.evals_result_

    # GridSearchCV on log(E)
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
    grid.fit(X_tr, y_tr_z)
    best_params = grid.best_params_
    logger.info("Best params: %s", best_params)

    reg = build_lgbm(cfg)
    reg.set_params(**best_params)
    reg.fit(
        X_tr,
        y_tr_z,
        eval_set=[(X_tr, y_tr_z), (X_val, y_val_z)],
        eval_names=["train", "valid"],
        eval_metric=[rel_rmse_logspace, "l2"],
        callbacks=[
            lgb.early_stopping(cfg.early_stopping_rounds, first_metric_only=True),
            lgb.log_evaluation(period=100),
        ],
    )
    return reg, reg.evals_result_


# ---------------------------------------------------------------------------
# Evaluation metrics & binning utilities
# ---------------------------------------------------------------------------

def compute_global_resolution(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    residual = (y_pred - y_true) / y_true
    return float(np.sqrt(np.mean(residual ** 2)))


def bin_resolution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    nbins: int = 13,
    e_min: float | None = None,
    e_max: float | None = None,
    use_gauss_fit: bool = False,
    frac: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute σ(E)/E versus true energy using RMS (or optional Gaussian fit).

    If frac < 1, compute a truncated RMS keeping the central fraction.
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
            _, sigma = stats.norm.fit(res_bin)
        else:
            if frac < 1.0:
                q_lo, q_hi = np.quantile(res_bin, [(1 - frac) / 2, 1 - (1 - frac) / 2])
                res_bin = res_bin[(res_bin >= q_lo) & (res_bin <= q_hi)]
            sigma = np.sqrt(np.mean(res_bin ** 2))
        centers.append(0.5 * (lo + hi))
        sigmas.append(sigma)
    return np.asarray(centers), np.asarray(sigmas)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def plot_training_curve(train_loss: Iterable[float], valid_loss: Iterable[float], out: Path) -> None:
    _ensure_dir(out)
    plt.figure(figsize=(6, 4))
    plt.plot(list(train_loss), label="train l2")
    plt.plot(list(valid_loss), "--", label="valid l2")
    plt.xlabel("Iteration")
    plt.ylabel("L2 loss")
    plt.legend()
    plt.title("Training vs validation loss")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def plot_profile_linearity(y_true: np.ndarray, y_pred: np.ndarray, out: Path, nbins: int = 20) -> None:
    _ensure_dir(out)
    plt.figure(figsize=(6, 6))
    bins = np.linspace(y_true.min(), y_true.max(), nbins + 1)
    centers = 0.5 * (bins[1:] + bins[:-1])
    mean, _, _ = binned_statistic(y_true, y_pred, statistic="mean", bins=bins)
    std, _, _ = binned_statistic(y_true, y_pred, statistic="std", bins=bins)
    cnt, _, _ = binned_statistic(y_true, y_pred, statistic="count", bins=bins)
    err = np.divide(std, np.sqrt(np.maximum(cnt, 1)), where=cnt > 0)
    m = cnt > 0
    plt.errorbar(centers[m], mean[m], yerr=err[m], fmt="o", markersize=4, capsize=3, label="Profile")
    lim = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lim, lim, "r--", linewidth=1, label="Ideal")
    plt.xlim(lim)
    plt.ylim(lim)
    plt.xlabel("True energy [GeV]")
    plt.ylabel("Reconstructed energy [GeV]")
    plt.title("Linearity profile")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def plot_resolution_curve(y_true: np.ndarray, y_pred: np.ndarray, out: Path, nbins: int = 13) -> None:
    _ensure_dir(out)
    centers, sigmas = bin_resolution(y_true, y_pred, nbins=nbins, use_gauss_fit=False, frac=0.9)
    plt.figure(figsize=(6, 4))
    plt.plot(centers, sigmas, marker="o")
    plt.xlabel("True energy [GeV]")
    plt.ylabel("σ(E)/E")
    plt.title("Energy resolution")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def plot_relative_deviation(y_true: np.ndarray, y_pred: np.ndarray, out: Path, nbins: int = 20) -> None:
    _ensure_dir(out)
    plt.figure(figsize=(6, 4))
    bins = np.linspace(y_true.min(), y_true.max(), nbins + 1)
    centers = 0.5 * (bins[1:] + bins[:-1])

    mean, _, _ = binned_statistic(y_true, y_pred, statistic="mean", bins=bins)
    std, _, _ = binned_statistic(y_true, y_pred, statistic="std", bins=bins)
    cnt, _, _ = binned_statistic(y_true, y_pred, statistic="count", bins=bins)

    with np.errstate(divide="ignore", invalid="ignore"):
        sem = std / np.sqrt(np.maximum(cnt, 1))
        dev_mean = (mean - centers) / centers
        dev_err = sem / centers

    m = (cnt > 0) & np.isfinite(dev_mean) & np.isfinite(dev_err)
    plt.errorbar(centers[m], dev_mean[m], yerr=dev_err[m], fmt="o", markersize=4, capsize=3)
    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.xlabel("True energy [GeV]")
    plt.ylabel("( <E_pred> - E_true ) / E_true")
    plt.title("Relative deviation")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def plot_feature_importance(reg: lgb.LGBMRegressor, feature_names: Sequence[str], out: Path) -> pd.DataFrame:
    _ensure_dir(out)
    imp = pd.DataFrame(
        {"feature": feature_names, "gain": reg.booster_.feature_importance(importance_type="gain")}
    ).sort_values("gain", ascending=False)
    plt.figure(figsize=(7, max(3, 0.3 * len(feature_names))))
    plt.barh(imp["feature"], imp["gain"])
    plt.gca().invert_yaxis()
    plt.xlabel("Total gain")
    plt.title("LightGBM feature importance")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    return imp


# ------------------------- Multi-particle plots -----------------------------

# def _global_energy_bounds(results: Mapping[str, Tuple[np.ndarray, np.ndarray]]) -> Tuple[float, float]:
#     e_min = min(float(y_true.min()) for (y_true, _) in results.values())
#     e_max = max(float(y_true.max()) for (y_true, _) in results.values())
#     return e_min, e_max


# def plot_profile_linearity_multi(
#     results: Mapping[str, Tuple[np.ndarray, np.ndarray]], out: Path, nbins: int = 20
# ) -> None:
#     _ensure_dir(out)
#     plt.figure(figsize=(6, 6))
#     e_min, e_max = _global_energy_bounds(results)
#     bins = np.linspace(e_min, e_max, nbins + 1)
#     centers = 0.5 * (bins[1:] + bins[:-1])

#     for label, (y_true, y_pred) in results.items():
#         color = PARTICLE_COLORS.get(label)
#         mean, _, _ = binned_statistic(y_true, y_pred, statistic="mean", bins=bins)
#         std, _, _ = binned_statistic(y_true, y_pred, statistic="std", bins=bins)
#         cnt, _, _ = binned_statistic(y_true, y_pred, statistic="count", bins=bins)
#         err = std / np.sqrt(np.maximum(cnt, 1))
#         m = cnt > 0
#         plt.errorbar(centers[m], mean[m], yerr=err[m], fmt="o", markersize=4, capsize=3, label=label, color=color)

#     lim = [e_min, e_max]
#     plt.plot(lim, lim, "r--", linewidth=1, label="Ideal")
#     plt.xlim(lim)
#     plt.ylim(lim)
#     plt.xlabel("True energy [GeV]")
#     plt.ylabel("Reconstructed energy [GeV]")
#     plt.title("Linearity profile")
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(out, dpi=150)
#     plt.close()


# def plot_resolution_curve_multi(
#     results: Mapping[str, Tuple[np.ndarray, np.ndarray]], out: Path, nbins: int = 13
# ) -> None:
#     _ensure_dir(out)
#     e_min, e_max = _global_energy_bounds(results)
#     plt.figure(figsize=(6, 4))
#     for label, (y_true, y_pred) in results.items():
#         color = PARTICLE_COLORS.get(label)
#         centers, sigmas = bin_resolution(
#             y_true, y_pred, nbins=nbins, e_min=e_min, e_max=e_max, use_gauss_fit=False, frac=0.9
#         )
#         plt.plot(centers, sigmas, marker="o", label=label, color=color)
#     plt.xlabel("True energy [GeV]")
#     plt.ylabel("σ(E)/E")
#     plt.title("Energy resolution")
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(out, dpi=150)
#     plt.close()


# def plot_relative_deviation_multi(
#     results: Mapping[str, Tuple[np.ndarray, np.ndarray]], out: Path, nbins: int = 20
# ) -> None:
#     _ensure_dir(out)
#     e_min, e_max = _global_energy_bounds(results)
#     bins = np.linspace(e_min, e_max, nbins + 1)
#     centers = 0.5 * (bins[1:] + bins[:-1])

#     plt.figure(figsize=(6, 4))
#     for label, (y_true, y_pred) in results.items():
#         color = PARTICLE_COLORS.get(label)
#         mean, _, _ = binned_statistic(y_true, y_pred, statistic="mean", bins=bins)
#         std, _, _ = binned_statistic(y_true, y_pred, statistic="std", bins=bins)
#         cnt, _, _ = binned_statistic(y_true, y_pred, statistic="count", bins=bins)
#         with np.errstate(divide="ignore", invalid="ignore"):
#             sem = std / np.sqrt(np.maximum(cnt, 1))
#             dev_mean = (mean - centers) / centers
#             dev_err = sem / centers
#         m = (cnt > 0) & np.isfinite(dev_mean)
#         plt.errorbar(centers[m], dev_mean[m], yerr=dev_err[m], fmt="o", markersize=4, capsize=3, label=label, color=color)

#     plt.axhline(0.0, linestyle="--", linewidth=1)
#     plt.xlabel("True energy [GeV]")
#     plt.ylabel("( <E_pred> - E_true ) / E_true")
#     plt.title("Relative deviation")
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(out, dpi=150)
#     plt.close()


# def plot_linearity_and_deviation_combo(
#     y_true: np.ndarray, y_pred: np.ndarray, out: Path, nbins: int = 20
# ) -> None:
#     from matplotlib import gridspec

#     _ensure_dir(out)
#     bins = np.linspace(y_true.min(), y_true.max(), nbins + 1)
#     centers = 0.5 * (bins[1:] + bins[:-1])
#     mean, _, _ = binned_statistic(y_true, y_pred, statistic="mean", bins=bins)
#     std, _, _ = binned_statistic(y_true, y_pred, statistic="std", bins=bins)
#     cnt, _, _ = binned_statistic(y_true, y_pred, statistic="count", bins=bins)
#     sem = std / np.sqrt(np.maximum(cnt, 1))

#     with np.errstate(divide="ignore", invalid="ignore"):
#         dev_mean = (mean - centers) / centers
#         dev_err = sem / centers

#     m = cnt > 0
#     lim = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]

#     fig = plt.figure(figsize=(7, 8))
#     gs = gridspec.GridSpec(2, 1, height_ratios=[2.5, 1.0], hspace=0.05)

#     # Top: linearity
#     ax1 = fig.add_subplot(gs[0])
#     ax1.errorbar(centers[m], mean[m], yerr=sem[m], fmt="o", markersize=4, capsize=3, label="Profile")
#     ax1.plot(lim, lim, "r--", linewidth=1, label="Ideal")
#     ax1.set_ylabel("Reconstructed energy [GeV]")
#     ax1.set_xlim(lim)
#     ax1.set_ylim(lim)
#     ax1.set_title("Linearity (top) & Relative deviation (bottom)")
#     ax1.legend(loc="best")
#     ax1.grid(True)

#     # Bottom: deviation
#     ax2 = fig.add_subplot(gs[1], sharex=ax1)
#     mb = m & np.isfinite(dev_mean) & np.isfinite(dev_err)
#     ax2.errorbar(centers[mb], dev_mean[mb], yerr=dev_err[mb], fmt="o", markersize=4, capsize=3)
#     ax2.axhline(0.0, linestyle="--", linewidth=1)
#     ax2.set_xlabel("True energy [GeV]")
#     ax2.set_ylabel("(⟨E_pred⟩-E_true)/E_true")
#     ax2.grid(True)

#     plt.tight_layout()
#     plt.savefig(out, dpi=150)
#     plt.close()


# def plot_linearity_and_deviation_combo_multi(
#     results: Mapping[str, Tuple[np.ndarray, np.ndarray]], out: Path, nbins: int = 20
# ) -> None:
#     from matplotlib import gridspec

#     _ensure_dir(out)
#     e_min, e_max = _global_energy_bounds(results)
#     bins = np.linspace(e_min, e_max, nbins + 1)
#     centers = 0.5 * (bins[1:] + bins[:-1])

#     fig = plt.figure(figsize=(7, 8))
#     gs = gridspec.GridSpec(2, 1, height_ratios=[2.5, 1.0], hspace=0.05)
#     ax1 = fig.add_subplot(gs[0])
#     ax2 = fig.add_subplot(gs[1], sharex=ax1)
#     lim = [e_min, e_max]

#     # Top: linearity multi
#     for label, (y_true, y_pred) in results.items():
#         color = PARTICLE_COLORS.get(label)
#         mean, _, _ = binned_statistic(y_true, y_pred, statistic="mean", bins=bins)
#         std, _, _ = binned_statistic(y_true, y_pred, statistic="std", bins=bins)
#         cnt, _, _ = binned_statistic(y_true, y_pred, statistic="count", bins=bins)
#         sem = np.divide(std, np.sqrt(np.maximum(cnt, 1)), where=cnt > 0)
#         m = cnt > 0
#         ax1.errorbar(centers[m], mean[m], yerr=sem[m], fmt="o", markersize=4, capsize=3, label=label, color=color)

#     ax1.plot(lim, lim, "r--", linewidth=1, label="Ideal")
#     ax1.set_xlim(lim)
#     ax1.set_ylim(lim)
#     ax1.set_ylabel("Reconstructed energy [GeV]")
#     ax1.set_title("Linearity (top) & Relative deviation (bottom) — multi")
#     ax1.legend(loc="best")
#     ax1.grid(True)

#     # Bottom: relative deviation multi
#     for label, (y_true, y_pred) in results.items():
#         color = PARTICLE_COLORS.get(label)
#         mean, _, _ = binned_statistic(y_true, y_pred, statistic="mean", bins=bins)
#         std, _, _ = binned_statistic(y_true, y_pred, statistic="std", bins=bins)
#         cnt, _, _ = binned_statistic(y_true, y_pred, statistic="count", bins=bins)
#         sem = np.divide(std, np.sqrt(np.maximum(cnt, 1)), where=cnt > 0)
#         with np.errstate(divide="ignore", invalid="ignore"):
#             dev_mean = (mean - centers) / centers
#             dev_err = sem / centers
#         m = (cnt > 0) & np.isfinite(dev_mean) & np.isfinite(dev_err)
#         ax2.errorbar(centers[m], dev_mean[m], yerr=dev_err[m], fmt="o", markersize=4, capsize=3, label=label, color=color)

#     ax2.axhline(0.0, linestyle="--", linewidth=1)
#     ax2.set_xlabel("True energy [GeV]")
#     ax2.set_ylabel("(⟨E_pred⟩-E_true)/E_true")
#     ax2.grid(True)

#     plt.tight_layout()
#     plt.savefig(out, dpi=150)
#     plt.close()


# ---------------------------------------------------------------------------
# Pipeline for a single particle
# ---------------------------------------------------------------------------

def evaluate(reg: lgb.LGBMRegressor, X_test: np.ndarray, y_test: np.ndarray):
    y_pred_log = reg.predict(X_test)
    y_pred = np.exp(y_pred_log)

    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    sigma_rel = compute_global_resolution(y_test, y_pred)
    logger.info("RMSE: %.4f | MAE: %.4f | R²: %.4f | σ(E)/E: %.4f", rmse, mae, r2, sigma_rel)
    return rmse, mae, r2, sigma_rel, y_pred


def run_for_particle(particle: str, cfg: GlobalCfg) -> Tuple[np.ndarray, np.ndarray]:
    if particle not in PARTICLE_CFG:
        raise KeyError(f"Unknown particle '{particle}'")

    pconf = PARTICLE_CFG[particle]
    res_dir = pconf.res_dir
    res_dir.mkdir(parents=True, exist_ok=True)

    # Paths for artifacts
    run_id = _next_run_id(pconf.param_csv)
    model_path = res_dir / "models" / f"lgbm_regressor_{pconf.tag}_{run_id}.joblib"
    scaler_path = res_dir / "models" / f"scaler_{pconf.tag}_{run_id}.joblib"
    plots_dir = res_dir / "plots"
    train_plot = plots_dir / f"training_curve_{run_id}.png"
    scatter_plot = plots_dir / f"pred_vs_true_{run_id}.png"

    # Log parameters
    row = {"run_id": run_id, "particle": particle, "file_path": pconf.file_path, **_filter_cfg_for_csv(cfg)}
    _write_dict_row(pconf.param_csv, row)

    # Load & prep
    cols = list(cfg.feature_cols) + [cfg.energy_col]
    df = load_root(pconf.file_path, cfg.tree_name, cols)
    df = clean_df(df)
    # df = smear_column_gaussian(df, "tMin", sigma=0.1)

    X = df[cfg.feature_cols].to_numpy(dtype=np.float32)
    y = df[cfg.energy_col].to_numpy(dtype=np.float32)

    X_tr, y_tr, X_val, y_val, X_te, y_te, scaler = split_val_test(X, y, cfg, valid_size=0.125)

    # Train (validation ≠ test)
    reg, evals = train_model(X_tr, y_tr, X_val, y_val, cfg)

    # Evaluate sur le test seulement
    rmse, mae, r2, sigma_rel, y_pred = evaluate(reg, X_te, y_te)

    # Plots
    train_l2 = evals.get("train", {}).get("l2", [])
    valid_l2 = evals.get("valid", {}).get("l2", [])
    if len(train_l2) and len(valid_l2):
        plot_training_curve(train_l2, valid_l2, train_plot)

    plot_profile_linearity(y_te, y_pred, scatter_plot)
    plot_resolution_curve(y_te, y_pred, plots_dir / f"resolution_curve_{run_id}.png")
    plot_relative_deviation(y_te, y_pred, plots_dir / f"relative_deviation_{run_id}.png", nbins=20)
    # plot_linearity_and_deviation_combo(y_te, y_pred, plots_dir / f"linearity_and_deviation_{run_id}.png", nbins=20)
    plot_feature_importance(reg, list(cfg.feature_cols), plots_dir / f"feat_importance_{run_id}.png")

    # Persist artifacts
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_path)
    joblib.dump(reg, model_path)
    logger.info("Artifacts saved to %s", res_dir)

    # Perf CSVs
    _write_row(pconf.perf_csv, ["run_id", "test_rmse", "test_mae", "test_r2", "sigma_over_E"], [run_id, rmse, mae, r2, sigma_rel])
    _write_row(pconf.comm_csv, ["run_id", "comment"], [run_id, f"rmse={rmse:.4f}, mae={mae:.4f}, r2={r2:.4f}, sigma/E={sigma_rel:.4f}"])

    npz_dir = res_dir / "arrays"
    npz_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_dir / f"test_and_pred_{run_id}.npz",
                        y_true=y_te, y_pred=y_pred)    

    return y_te, y_pred


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train LightGBM regressors for hadron energy (π, K, p).")
    p.add_argument(
        "--particles",
        nargs="+",
        default=["all"],
        help="List of particles to process: pi kaon proton or 'all'",
    )
    p.add_argument("--grid-search", action="store_true", help="Enable hyperparameter grid-search.")
    return p


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    parts = list(PARTICLE_CFG.keys()) if "all" in args.particles else args.particles

    cfg = GlobalCfg(grid_search=args.grid_search)
    set_global_seeds(cfg.random_state)

    for part in parts:
        if part not in PARTICLE_CFG:
            logger.error("Unknown particle '%s' – skipping.", part)
            continue
        logger.info("=== Training for %s ===", part)
        run_for_particle(part, cfg)

    # results_multi: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    # for part in parts:
    #     if part not in PARTICLE_CFG:
    #         logger.error("Unknown particle '%s' – skipping.", part)
    #         continue
    #     logger.info("=== Training for %s ===", part)
    #     y_te, y_pred = run_for_particle(part, cfg)
    #     results_multi[part] = (y_te, y_pred)

    # # Combined plots if ≥2 particles
    # if len(results_multi) >= 2:
    #     out_dir = Path("results_all_energy_reco") / "plots"
    #     out_dir.mkdir(parents=True, exist_ok=True)
    #     plot_profile_linearity_multi(results_multi, out_dir / "profile_linearite_all.png", nbins=20)
    #     plot_resolution_curve_multi(results_multi, out_dir / "resolution_all.png", nbins=13)
    #     plot_relative_deviation_multi(results_multi, out_dir / "relative_deviation_all.png", nbins=20)
    #     plot_linearity_and_deviation_combo_multi(results_multi, out_dir / "linearity_and_deviation_all.png", nbins=20)
    #     logger.info("Combined plots saved to %s", out_dir)


if __name__ == "__main__":
    main()
