#!/usr/bin/env python3
# MLP_Energy_reconstruction_all.py
"""
Exemple d'appel

# Tous les hadrons
python3 MLP_Energy_reconstruction_all.py --particles all

# Un seul (ex : proton) + grid-search
python3 MLP_Energy_reconstruction_all.py --particles proton --grid-search

# Overrides MLP
python3 MLP_Energy_reconstruction_all.py \
  --particles pi \
  --hidden-layers 256,128,64 \
  --alpha 1e-4 --batch-size 64 --lr 5e-4 --max-iter 800
"""

import argparse
import csv
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Sequence, Tuple, Dict, Union

import joblib
import numpy as np
import pandas as pd
import uproot
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# seaborn is optional
try:
    import seaborn as sns  # type: ignore
    _HAS_SNS = True
except Exception:
    _HAS_SNS = False

import scipy.stats as stats
from scipy.stats import binned_statistic

# ---------------------------- Logging ----------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------- Dataclass --------------------------------------
@dataclass
class GlobalCfg:
    tree_name: str = "paramsTree"
    energy_col: str = "primaryEnergy"
    feature_cols: Sequence[str] = field(default_factory=lambda: [
        "Thr1", "Thr2", "Thr3", "Begin", "Radius", "Density", "NClusters", "ratioThr23", "Zbary", "Zrms",
        "PctHitsFirst10", "PlanesWithClusmore2", "AvgClustSize", "MaxClustSize", "lambda1", "lambda2",
        "N3",  "N2", "N1", "tMin", "tMax", "tMean", "tSpread", "Nmax", "z0_fit", "Xmax", "lambda",
        "nTrackSegments", "eccentricity3D", "nHitsTotal", "sumThrTotal"
    ])
    test_size: float = 0.20
    random_state: int = 42

    # MLP defaults
    hidden_layer_sizes: Tuple[int, ...] = (128, 64, 32)
    alpha: float = 1e-4
    batch_size: int = 32
    learning_rate_init: float = 1e-3
    max_iter: int = 500
    early_stopping: bool = True
    n_iter_no_change: int = 15
    validation_fraction: float = 0.1

    # Grid-search
    grid_search: bool = False
    param_grid: Dict[str, List] = field(default_factory=lambda: {
        "hidden_layer_sizes": [(64, 32), (128, 64, 32)],
        "alpha": [1e-5, 1e-4, 1e-3],
        "learning_rate_init": [1e-4, 5e-4, 1e-3],
    })

# --------------------- Particle-specific config ------------------------------
PARTICLE_CFG = {
    "pi": {
        "file_path": "/gridgroup/ilc/midir/analyse/data/params/130k_pi_E1to130_params_merged.root",
        "res_dir":   Path("results_pi_energy_reco"),
        "param_csv": Path("run_parameters_regression_pi.csv"),
        "perf_csv":  Path("hadron_energy_performances_pi.csv"),
        "comm_csv":  Path("run_comments_regression_pi.csv"),
        "tag":       "pi",
    },
    "kaon": {
        "file_path": "/gridgroup/ilc/midir/analyse/data/params/130k_kaon_E1to130_params_merged.root",
        "res_dir":   Path("results_kaon_energy_reco"),
        "param_csv": Path("run_parameters_regression_kaon.csv"),
        "perf_csv":  Path("hadron_energy_performances_kaon.csv"),
        "comm_csv":  Path("run_comments_regression_kaon.csv"),
        "tag":       "kaon",
    },
    "proton": {
        "file_path": "/gridgroup/ilc/midir/analyse/data/params/130k_proton_E1to130_params_merged.root",
        "res_dir":   Path("results_proton_energy_reco"),
        "param_csv": Path("run_parameters_regression_proton.csv"),
        "perf_csv":  Path("hadron_energy_performances_proton.csv"),
        "comm_csv":  Path("run_comments_regression_proton.csv"),
        "tag":       "proton",
    },
}

# ---------------------------- Utilities --------------------------------------
def next_run_id(csv_path: Path) -> int:
    if not csv_path.exists():
        return 1
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        try:
            return max(int(r["run_id"]) for r in reader) + 1
        except ValueError:
            return 1

def write_dict_row(csv_path: Path, row: Dict[str, Union[str, int, float]]) -> None:
    is_new = not csv_path.exists()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if is_new:
            writer.writeheader()
        writer.writerow(row)

def write_row(csv_path: Path, header: List[str], row: List) -> None:
    is_new = not csv_path.exists()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(header)
        writer.writerow(row)

def filter_cfg(cfg: GlobalCfg) -> Dict[str, Union[str, int, float]]:
    d = asdict(cfg).copy()
    d.pop("feature_cols", None)
    d.pop("param_grid", None)
    # pretty-print tuple
    d["hidden_layer_sizes"] = ",".join(map(str, cfg.hidden_layer_sizes))
    return d

# ----------------------------- I/O & Prep ------------------------------------
def load_root(file: str | Path, tree: str, cols: Sequence[str]) -> pd.DataFrame:
    path = Path(file)
    logger.info("Reading ROOT file: %s", path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    with uproot.open(path) as f:
        if tree not in f:
            raise KeyError(f"Tree '{tree}' not found in {path}")
        df = f[tree].arrays(cols, library="pd")
    logger.info("Total events loaded: %d", len(df))
    return df

def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.replace([np.inf, -np.inf], pd.NA).dropna().reset_index(drop=True)
    logger.info("After cleaning: %d (removed %d)", len(df), before - len(df))
    return df

def split_and_scale(X: np.ndarray, y: np.ndarray, cfg: GlobalCfg):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    return X_train_s, y_train, X_test_s, y_test, scaler

# ------------------------------ Model ----------------------------------------
def build_mlp(cfg: GlobalCfg) -> MLPRegressor:
    return MLPRegressor(
        hidden_layer_sizes=cfg.hidden_layer_sizes,
        alpha=cfg.alpha,
        batch_size=cfg.batch_size,
        learning_rate_init=cfg.learning_rate_init,
        learning_rate="adaptive",
        solver="adam",
        max_iter=cfg.max_iter,
        early_stopping=cfg.early_stopping,
        n_iter_no_change=cfg.n_iter_no_change,
        validation_fraction=cfg.validation_fraction,
        random_state=cfg.random_state,
        verbose=False,
    )


def train_model(X_tr, y_tr, X_val, y_val, cfg: GlobalCfg):
    """Train the MLP. If grid-search is enabled, tune hyper-parameters.

    Returns: (regressor, evals_result_dict)
      - evals_result_dict mimics LGBM: {"train": {"loss": [...]}, "valid": {"loss": [...] (optional)}}
    """
    if not cfg.grid_search:
        reg = build_mlp(cfg)
        reg.verbose = True
        reg.fit(X_tr, y_tr)
        evals = {"train": {"loss": list(getattr(reg, "loss_curve_", []))}}
        # sklearn may expose validation_scores_ (R^2). Convert to pseudo-loss if available
        val_scores = getattr(reg, "validation_scores_", None)
        if val_scores is not None:
            # Convert R^2 to a decreasing "loss-like" curve: loss = 1 - R^2
            evals["valid"] = {"loss": [1.0 - float(s) for s in val_scores]}
    else:
        logger.info("Running GridSearchCV…")
        base = build_mlp(cfg)
        cv = KFold(n_splits=5, shuffle=True, random_state=cfg.random_state)
        grid = GridSearchCV(
            base, cfg.param_grid, cv=cv,
            scoring="neg_root_mean_squared_error", n_jobs=-1, verbose=1
        )
        grid.fit(X_tr, y_tr)
        reg = grid.best_estimator_
        logger.info("Best params: %s", grid.best_params_)
        evals = {"train": {"loss": list(getattr(reg, "loss_curve_", []))}}
    return reg, evals

# --------------------------- Evaluation & plots ------------------------------
def compute_global_resolution(y_true, y_pred):
    residual = (y_pred - y_true) / y_true
    sigma = np.sqrt(np.mean(residual**2))  # RMS
    return sigma


def bin_resolution(y_true, y_pred, nbins=13, e_min=None, e_max=None, use_gauss_fit=False, frac=1.0):
    """
    Calcule σ(E)/E vs E_true.
    - nbins: nombre de bins
    - use_gauss_fit: fit gaussien sur le résidu, sinon RMS (ou RMS tronquée si frac<1)
    - frac: fraction centrale (0<frac<=1) pour RMS tronquée (ex: 0.9)
    """
    e = y_true
    r = (y_pred - y_true) / y_true

    if e_min is None: e_min = e.min()
    if e_max is None: e_max = e.max()

    bins = np.linspace(e_min, e_max, nbins+1)
    centers, sigmas = [], []

    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (e >= lo) & (e < hi)
        if mask.sum() < 20:
            continue
        res_bin = r[mask]
        if use_gauss_fit:
            mu, sigma = stats.norm.fit(res_bin)
        else:
            if frac < 1.0:
                # RMS tronquée
                q_lo, q_hi = np.quantile(res_bin, [(1-frac)/2, 1-(1-frac)/2])
                res_bin = res_bin[(res_bin>=q_lo) & (res_bin<=q_hi)]
            sigma = np.sqrt(np.mean(res_bin**2))
        centers.append(0.5*(lo+hi))
        sigmas.append(sigma)
    return np.array(centers), np.array(sigmas)


def evaluate(reg, X_test, y_test):
    y_pred = reg.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    sigma_rel = compute_global_resolution(y_test, y_pred)
    logger.info("RMSE: %.4f, MAE: %.4f, R²: %.4f, σ(E)/E: %.4f", rmse, mae, r2, sigma_rel)
    return rmse, mae, r2, sigma_rel, y_pred


def plot_training_curve(train_loss: List[float], valid_loss: List[float] | None, out: Path):
    plt.figure(figsize=(6, 4))
    if train_loss:
        plt.plot(train_loss, label="train loss")
    if valid_loss:
        # valid_loss is pseudo-loss (1-R2) if available
        plt.plot(valid_loss, "--", label="valid (1-R²)")
    plt.xlabel("Iteration"); plt.ylabel("Loss")
    plt.legend(); plt.title("Training curve")
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()

def plot_profile_linearite(y_true: np.ndarray, y_pred: np.ndarray, out: Path, nbins: int = 20):
    plt.figure(figsize=(6,6))
    bins = np.linspace(y_true.min(), y_true.max(), nbins+1)
    centers = 0.5*(bins[1:] + bins[:-1])
    mean, _, _ = binned_statistic(y_true, y_pred, statistic='mean', bins=bins)
    std,  _, _ = binned_statistic(y_true, y_pred, statistic='std',  bins=bins)
    cnt,  _, _ = binned_statistic(y_true, y_pred, statistic='count', bins=bins)
    err = std/np.sqrt(np.maximum(cnt, 1))
    m = cnt > 0
    plt.errorbar(centers[m], mean[m], yerr=err[m], fmt='o', markersize=4, capsize=3, label="Profil")
    lim = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lim, lim, "r--", linewidth=1, label="Idéal")
    plt.xlim(lim); plt.ylim(lim)
    plt.xlabel("True energy [GeV]"); plt.ylabel("Reconstructed energy [GeV]")
    plt.title("Profil linéarité"); plt.legend(); plt.tight_layout()
    plt.savefig(out, dpi=150); plt.close()

def plot_resolution_curve(y_true, y_pred, out: Path, nbins=13):
    centers, sigmas = bin_resolution(y_true, y_pred, nbins=nbins, use_gauss_fit=False, frac=0.9)
    plt.figure(figsize=(6,4))
    plt.plot(centers, sigmas, marker='o')
    plt.xlabel("True energy [GeV]"); plt.ylabel("σ(E)/E")
    plt.title("Energy resolution"); plt.tight_layout()
    plt.savefig(out, dpi=150); plt.close()
    
def plot_profile_linearite_multi(results: Dict[str, Tuple[np.ndarray, np.ndarray]],
                                 out: Path, nbins: int = 20):
    """
    results: dict { 'pi': (y_true, y_pred), 'kaon': (...), 'proton': (...) }
    Produit un seul graphe avec 3 courbes (profil ⟨E_pred⟩ vs E_true).
    """
    plt.figure(figsize=(6,6))

    # bornes globales communes pour binning
    e_min = min(y_true.min() for (y_true, _) in results.values())
    e_max = max(y_true.max() for (y_true, _) in results.values())
    bins = np.linspace(e_min, e_max, nbins+1)
    centers = 0.5*(bins[1:] + bins[:-1])

    for label, (y_true, y_pred) in results.items():
        mean, _, _  = binned_statistic(y_true, y_pred, statistic='mean',  bins=bins)
        std,  _, _  = binned_statistic(y_true, y_pred, statistic='std',   bins=bins)
        cnt,  _, _  = binned_statistic(y_true, y_pred, statistic='count', bins=bins)
        err = std/np.sqrt(np.maximum(cnt, 1))
        # masque pour bins vides
        m = cnt > 0
        plt.errorbar(centers[m], mean[m], yerr=err[m], fmt='o', markersize=4, capsize=3, label=label)

    lim = [e_min, e_max]
    plt.plot(lim, lim, "r--", linewidth=1, label="Idéal")
    plt.xlim(lim); plt.ylim(lim)
    plt.xlabel("True energy [GeV]")
    plt.ylabel("Reconstructed energy [GeV]")
    plt.title("Profil linéarité (comparatif)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def plot_resolution_curve_multi(results: Dict[str, Tuple[np.ndarray, np.ndarray]],
                                out: Path, nbins: int = 13):
    """
    results: dict { 'pi': (y_true, y_pred), ... }
    Produit un seul graphe avec 3 courbes σ(E)/E vs E_true.
    """
    # bornes communes pour le binning
    e_min = min(y_true.min() for (y_true, _) in results.values())
    e_max = max(y_true.max() for (y_true, _) in results.values())

    plt.figure(figsize=(6,4))
    for label, (y_true, y_pred) in results.items():
        centers, sigmas = bin_resolution(y_true, y_pred, nbins=nbins,
                                         e_min=e_min, e_max=e_max,
                                         use_gauss_fit=False, frac=0.9)
        plt.plot(centers, sigmas, marker='o', label=label)

    plt.xlabel("True energy [GeV]")
    plt.ylabel("σ(E)/E")
    plt.title("Energy resolution (comparatif)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()

# ------------------------------ Pipeline -------------------------------------

def run_for_particle(particle: str, cfg: GlobalCfg):
    pconf = PARTICLE_CFG[particle]
    param_csv, perf_csv, comm_csv = pconf["param_csv"], pconf["perf_csv"], pconf["comm_csv"]
    res_dir = pconf["res_dir"]; res_dir.mkdir(parents=True, exist_ok=True)

    run_id = next_run_id(param_csv)

    # Prepare paths
    paths = {
        "model":      res_dir / "models" / f"mlp_regressor_{pconf['tag']}_{run_id}.joblib",
        "scaler":     res_dir / "models" / f"scaler_{pconf['tag']}_{run_id}.joblib",
        "train_plot": res_dir / "plots"  / f"training_curve_{run_id}.pdf",
        "scatter":    res_dir / "plots"  / f"pred_vs_true_{run_id}.pdf",
        "res_curve":  res_dir / "plots"  / f"resolution_curve_{run_id}.pdf",
    }
    for p in paths.values():
        p.parent.mkdir(parents=True, exist_ok=True)

    # Log parameters
    row = {"run_id": run_id, "particle": particle,
           "file_path": pconf["file_path"], **filter_cfg(cfg)}
    write_dict_row(param_csv, row)

    # Load data
    cols = list(cfg.feature_cols) + [cfg.energy_col]
    df = load_root(pconf["file_path"], cfg.tree_name, cols)
    df = clean_df(df)

    X = df[cfg.feature_cols].values.astype(np.float32)
    y = df[cfg.energy_col].values.astype(np.float32)

    X_tr, y_tr, X_te, y_te, scaler = split_and_scale(X, y, cfg)

    # Train
    reg, evals = train_model(X_tr, y_tr, X_te, y_te, cfg)

    # Eval
    rmse, mae, r2, sigma_rel, y_pred = evaluate(reg, X_te, y_te)

    # Plots
    train_loss = evals.get("train", {}).get("loss", [])
    valid_loss = evals.get("valid", {}).get("loss", None)
    plot_training_curve(train_loss, valid_loss, paths["train_plot"])
    plot_profile_linearite(y_te, y_pred, paths["scatter"])
    plot_resolution_curve(y_te, y_pred, paths["res_curve"])

    # Save
    joblib.dump(scaler, paths["scaler"])
    joblib.dump(reg, paths["model"])
    logger.info("Artifacts saved to %s", res_dir)

    # Perf logs
    write_row(perf_csv,
              ["run_id", "test_rmse", "test_mae", "test_r2", "sigma_over_E"],
              [run_id, rmse, mae, r2, sigma_rel])
    write_row(comm_csv, ["run_id", "comment"],
              [run_id, f"rmse={rmse:.4f}, mae={mae:.4f}, r2={r2:.4f}, sigma/E={sigma_rel:.4f}"])
    return y_te, y_pred

# ------------------------------ CLI ------------------------------------------

def build_arg_parser():
    p = argparse.ArgumentParser(description="Train MLP regressors for hadron energy (π, K, p) with a LightGBM-like pipeline.")
    p.add_argument("--particles", nargs="+", default=["all"],
                   help="List of particles to process: pi kaon proton or 'all'")
    p.add_argument("--grid-search", action="store_true", help="Enable hyperparameter grid-search.")

    # Optional MLP overrides
    p.add_argument("--hidden-layers", type=str, help="Comma-separated, e.g. '128,64,32'")
    p.add_argument("--alpha", type=float)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--lr", dest="learning_rate_init", type=float)
    p.add_argument("--max-iter", type=int)
    return p


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    parts = list(PARTICLE_CFG.keys()) if "all" in args.particles else args.particles

    cfg = GlobalCfg(grid_search=args.grid_search)

    # apply optional overrides
    if args.hidden_layers:
        try:
            cfg.hidden_layer_sizes = tuple(int(x) for x in args.hidden_layers.split(',') if x.strip())  # type: ignore
        except Exception:
            raise SystemExit("--hidden-layers must be a comma-separated list of integers, e.g. 128,64,32")
    if args.alpha is not None: cfg.alpha = args.alpha
    if args.batch_size is not None: cfg.batch_size = args.batch_size
    if args.learning_rate_init is not None: cfg.learning_rate_init = args.learning_rate_init
    if args.max_iter is not None: cfg.max_iter = args.max_iter

    results_multi: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    for part in parts:
        if part not in PARTICLE_CFG:
            logger.error("Unknown particle '%s' – skipping.", part)
            continue
        logger.info("=== Training for %s ===", part)
        y_te, y_pred = run_for_particle(part, cfg)  # entraîne, fait les plots individuels, renvoie arrays
        results_multi[part] = (y_te, y_pred)

    # Graphes comparatifs (seulement si on a ≥ 2 particules)
    if len(results_multi) >= 2:
        out_dir = Path("results_all_energy_reco") / "plots"
        out_dir.mkdir(parents=True, exist_ok=True)
        plot_profile_linearite_multi(results_multi, out_dir / "profile_linearite_all.pdf", nbins=20)
        plot_resolution_curve_multi(results_multi, out_dir / "resolution_all.pdf", nbins=13)
        logger.info("Combined plots saved to %s", out_dir)


if __name__ == "__main__":
    main()







