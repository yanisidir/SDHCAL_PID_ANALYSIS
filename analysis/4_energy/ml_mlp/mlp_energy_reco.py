#!/usr/bin/env python3
"""MLP energy reconstruction for hadronic showers (pi-, K0_L, p).

Replaces ``MLP_Energy_reconstruction_{pion,kaon,proton,all}.py``. Those four
files could not run and could not have produced per-particle results:

* the ``pion``, ``kaon`` and ``proton`` versions were **byte-for-byte identical**
  (same md5), all reading the pion file and all writing into the same
  ``results_energy_reco/`` directory, so any "kaon MLP" or "proton MLP" number
  was in fact a pion number;
* all three carried a ``SyntaxError`` at line 140 (``config.file_path),``), so
  the interpreter refused them outright;
* ``load_root`` was handed a ``str`` where it iterated over a list of paths, so
  it looped over the *characters* of the filename, found no file and raised
  ``RuntimeError: No valid ROOT files found``;
* the test set was passed as the validation set, and the model was evaluated in
  linear energy space with no separation of bias and resolution.

This version takes the particle on the command line, keeps a real validation
split, trains on ``log E`` with the bias-corrected back-transform, and reports
bias and resolution separately. Unlike the LightGBM pipeline the scaler is kept:
a neural network genuinely needs standardised inputs, and it is saved with the
model so inference cannot forget it.

Examples
--------
::

    python mlp_energy_reco.py --particles pi
    python mlp_energy_reco.py --particles all --seeds 3
    python mlp_energy_reco.py --particles mixed        # PID-agnostic baseline
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


from sdhcal import metrics, paths, rootio, runlog, splits
from sdhcal.smearing import TimeSmearing, smear_time_columns
from sdhcal.target import make_target

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)8s | %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger("energy_reco_mlp")

MODES = (*paths.PARTICLES, "mixed")


@dataclass
class Config:
    """Everything that defines a run. Logged verbatim, feature list included."""

    tree_name: str = "paramsTree"
    energy_col: str = "primaryEnergy"

    feature_cols: Sequence[str] = field(
        default_factory=lambda: [
            "Thr1", "Thr2", "Thr3", "Radius", "Density", "NClusters", "ratioThr23",
            "Zbary", "Zrms", "PctHitsFirst10", "PlanesWithClusmore2", "AvgClustSize",
            "MaxClustSize", "N1", "N2", "N3",
        ]
    )

    test_size: float = 0.20
    val_size: float = 0.10
    random_state: int = 42

    target: str = "log"
    correct_target_bias: bool = True

    smear_sigma_ns: Optional[float] = None
    smear_columns: Sequence[str] = ("tMin", "tMax", "tMean", "tSpread")

    hidden_layer_sizes: Tuple[int, ...] = (128, 64, 32)
    alpha: float = 1e-4
    batch_size: int = 256
    learning_rate_init: float = 1e-3
    max_iter: int = 500
    n_iter_no_change: int = 15

    n_bins: int = 13
    log_bins: bool = False
    core_method: str = "truncated_rms"
    core_frac: float = 0.9

    grid_search: bool = False
    param_grid: Dict[str, List] = field(
        default_factory=lambda: {
            "hidden_layer_sizes": [(64, 32), (128, 64, 32)],
            "alpha": [1e-5, 1e-4, 1e-3],
            "learning_rate_init": [1e-4, 5e-4, 1e-3],
        }
    )

    def smearing(self) -> TimeSmearing:
        return TimeSmearing(
            sigma_ns=self.smear_sigma_ns, columns=tuple(self.smear_columns), seed=self.random_state
        )

    def loggable(self, mode: str) -> Dict[str, object]:
        out: Dict[str, object] = {
            "mode": mode,
            "tree_name": self.tree_name,
            "energy_col": self.energy_col,
            "test_size": self.test_size,
            "val_size": self.val_size,
            "random_state": self.random_state,
            "target": self.target,
            "correct_target_bias": int(self.correct_target_bias),
            "hidden_layer_sizes": "x".join(str(n) for n in self.hidden_layer_sizes),
            "alpha": self.alpha,
            "batch_size": self.batch_size,
            "learning_rate_init": self.learning_rate_init,
            "max_iter": self.max_iter,
            "n_iter_no_change": self.n_iter_no_change,
            "core_method": self.core_method,
            "core_frac": self.core_frac,
            "n_bins": self.n_bins,
            "grid_search": int(self.grid_search),
        }
        out.update(self.smearing().as_dict())
        return out


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def input_files(mode: str) -> List[Path]:
    wanted = paths.PARTICLES if mode == "mixed" else (mode,)
    return [paths.dataset("params_merged", particle) for particle in wanted]


def load_dataset(mode: str, cfg: Config) -> pd.DataFrame:
    needed = list(dict.fromkeys([*cfg.feature_cols, cfg.energy_col]))
    frames = [rootio.load_trees(path, cfg.tree_name, needed) for path in input_files(mode)]
    df = rootio.clean(pd.concat(frames, ignore_index=True), columns=needed)
    df = df[df[cfg.energy_col] > 0].reset_index(drop=True)
    logger.info("Mode %-7s %8d events", mode, len(df))
    return smear_time_columns(df, cfg.smearing())


def build_mlp(cfg: Config, seed: int) -> MLPRegressor:
    return MLPRegressor(
        hidden_layer_sizes=cfg.hidden_layer_sizes,
        alpha=cfg.alpha,
        batch_size=cfg.batch_size,
        learning_rate_init=cfg.learning_rate_init,
        learning_rate="adaptive",
        solver="adam",
        max_iter=cfg.max_iter,
        early_stopping=True,
        n_iter_no_change=cfg.n_iter_no_change,
        random_state=seed,
        verbose=False,
    )


def train(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    cfg: Config,
    seed: int,
):
    """Fit on the training split; calibrate the back-transform on validation."""
    target = make_target(cfg.target, correct_bias=cfg.correct_target_bias)
    z_train = target.forward(y_train)

    best: Dict[str, object] = {}
    if cfg.grid_search:
        logger.info("Running GridSearchCV on the training split")
        grid = GridSearchCV(
            build_mlp(cfg, seed),
            cfg.param_grid,
            cv=KFold(n_splits=5, shuffle=True, random_state=seed),
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
            verbose=1,
        )
        grid.fit(x_train, z_train)
        best = grid.best_params_
        logger.info("Best parameters: %s", best)

    model = build_mlp(cfg, seed)
    if best:
        model.set_params(**best)
    model.fit(x_train, z_train)
    logger.info("Converged after %d iterations (loss=%.5f)", model.n_iter_, model.loss_)

    factor = target.fit_correction(y_val, model.predict(x_val))
    logger.info("Back-transform bias correction: %.5f [%+.2f %%]", factor, 100.0 * (factor - 1.0))
    return model, target, best


def evaluate(model, target, x_test: np.ndarray, y_test: np.ndarray, cfg: Config):
    y_pred = target.inverse(model.predict(x_test))
    core = metrics.core_bias_resolution(
        metrics.relative_residual(y_test, y_pred), method=cfg.core_method, frac=cfg.core_frac
    )
    results = {
        "rmse_gev": float(mean_squared_error(y_test, y_pred) ** 0.5),
        "mae_gev": float(mean_absolute_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred)),
        "bias": core.bias,
        "bias_err": core.bias_err,
        "resolution": core.sigma,
        "resolution_err": core.sigma_err,
        "energy_scale": float(np.mean(y_pred) / np.mean(y_test)),
        "rms_rel_about_zero": metrics.rms_relative_about_zero(y_test, y_pred),
        "n_test": int(y_test.size),
        "target_bias_correction": float(getattr(target, "factor", 1.0)),
    }
    logger.info(
        "Test: RMSE=%.3f GeV  MAE=%.3f GeV  R2=%.4f | bias=%+.4f  resolution=%.4f  scale=%.4f",
        results["rmse_gev"], results["mae_gev"], results["r2"],
        results["bias"], results["resolution"], results["energy_scale"],
    )
    return results, y_pred


def plot_loss_curve(model: MLPRegressor, out: Path) -> None:
    plt.figure(figsize=(6, 4))
    plt.plot(model.loss_curve_, label="training loss")
    if getattr(model, "validation_scores_", None):
        twin = plt.gca().twinx()
        twin.plot(model.validation_scores_, "--", color="tab:orange", label="internal validation R2")
        twin.set_ylabel("internal validation $R^2$")
    plt.xlabel("Epoch")
    plt.ylabel("MSE on the transformed target")
    plt.title("MLP training")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def plot_linearity(y_true: np.ndarray, y_pred: np.ndarray, out: Path, cfg: Config) -> None:
    bins = metrics.energy_bins(float(y_true.min()), float(y_true.max()), cfg.n_bins, log=cfg.log_bins)
    centers, means, errors = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_true >= lo) & (y_true < hi)
        if mask.sum() < 20:
            continue
        centers.append(float(np.mean(y_true[mask])))
        means.append(float(np.mean(y_pred[mask])))
        errors.append(float(np.std(y_pred[mask], ddof=1) / np.sqrt(mask.sum())))
    plt.figure(figsize=(6, 6))
    plt.errorbar(centers, means, yerr=errors, fmt="o", markersize=4, capsize=3, label="Profile")
    lim = [float(y_true.min()), float(y_true.max())]
    plt.plot(lim, lim, "r--", linewidth=1, label="Ideal")
    plt.xlabel("True energy [GeV]")
    plt.ylabel(r"$\langle E_{\rm reco}\rangle$ [GeV]")
    plt.title("Linearity (MLP)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def plot_bias_and_resolution(y_true: np.ndarray, y_pred: np.ndarray, out: Path, cfg: Config):
    binned = metrics.binned_bias_resolution(
        y_true, y_pred, n_bins=cfg.n_bins, log_bins=cfg.log_bins,
        method=cfg.core_method, frac=cfg.core_frac,
    )
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(6.5, 7), sharex=True)
    top.errorbar(binned["center"], binned["bias"], yerr=binned["bias_err"], fmt="o", capsize=3)
    top.axhline(0.0, linestyle="--", linewidth=1, color="0.4")
    top.set_ylabel(r"bias $\langle \Delta E/E \rangle$")
    top.set_title("Bias and resolution versus true energy (MLP)")
    top.grid(alpha=0.3)
    bottom.errorbar(
        binned["center"], binned["sigma"], yerr=binned["sigma_err"], fmt="o", capsize=3, color="tab:purple"
    )
    bottom.set_xlabel("True energy [GeV]")
    bottom.set_ylabel(r"resolution $\sigma(\Delta E/E)$")
    bottom.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return binned


def run_single_seed(mode: str, cfg: Config, seed: int, log: runlog.RunLogger) -> Dict[str, float]:
    set_global_seeds(seed)
    seeded = replace(cfg, random_state=seed)

    df = load_dataset(mode, seeded)
    x = df[list(seeded.feature_cols)].to_numpy(dtype=np.float32)
    y = df[seeded.energy_col].to_numpy(dtype=np.float64)

    split = splits.split_indices(len(df), test_size=seeded.test_size, val_size=seeded.val_size, seed=seed)
    logger.info("Split sizes: %s", split.sizes())

    # The scaler is fitted on the training split only.
    scaler = StandardScaler().fit(x[split.train])
    x_train, x_val, x_test = (scaler.transform(x[part]) for part in (split.train, split.val, split.test))

    model, target, best = train(x_train, y[split.train], x_val, y[split.val], seeded, seed)
    results, y_pred = evaluate(model, target, x_test, y[split.test], seeded)
    results["seed"] = seed
    results["n_iter"] = int(model.n_iter_)
    if best:
        results["best_params"] = ";".join(f"{k}={v}" for k, v in best.items())

    tag = f"{mode}_{log.run_id}_seed{seed}"
    model_path = log.path("models", f"mlp_regressor_{tag}.joblib")
    joblib.dump(
        {"model": model, "scaler": scaler, "target": target, "features": list(seeded.feature_cols)},
        model_path,
    )

    plot_loss_curve(model, log.path("plots", f"training_curve_{tag}.png"))
    plot_linearity(y[split.test], y_pred, log.path("plots", f"linearity_{tag}.png"), seeded)
    binned = plot_bias_and_resolution(
        y[split.test], y_pred, log.path("plots", f"bias_resolution_{tag}.png"), seeded
    )
    arrays = log.path("arrays", f"test_and_pred_{tag}.npz")
    np.savez_compressed(
        arrays,
        y_true=y[split.test],
        y_pred=y_pred,
        test_index=split.test,
        bin_center=binned["center"],
        bin_bias=binned["bias"],
        bin_sigma=binned["sigma"],
    )
    log.log_artifacts(**{f"model_seed{seed}": model_path, f"arrays_seed{seed}": arrays})
    return results


def run_mode(mode: str, cfg: Config, n_seeds: int) -> Dict[str, float]:
    out_dir = paths.output_dir() / "energy_reco_mlp" / mode
    log = runlog.RunLogger(out_dir, name="energy_reco")
    log.start(cfg.loggable(mode), features=list(cfg.feature_cols))
    logger.info("=== mode %s | run %d | %d seed(s) ===", mode, log.run_id, n_seeds)

    per_seed = [run_single_seed(mode, cfg, cfg.random_state + i, log) for i in range(n_seeds)]
    pd.DataFrame(per_seed).to_csv(log.path("tables", f"per_seed_{mode}_{log.run_id}.csv"), index=False)

    summary: Dict[str, float] = {"n_seeds": n_seeds}
    for key in ("rmse_gev", "mae_gev", "r2", "bias", "resolution", "energy_scale", "rms_rel_about_zero"):
        values = np.array([r[key] for r in per_seed], dtype=float)
        summary[key] = float(values.mean())
        summary[f"{key}_spread"] = float(values.std(ddof=1)) if n_seeds > 1 else float("nan")

    log.log_metrics(**summary)
    log.log_comment(
        f"mode={mode} bias={summary['bias']:+.4f} resolution={summary['resolution']:.4f} "
        f"over {n_seeds} seed(s)"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--particles", nargs="+", default=["all"], metavar="MODE",
                   help="pi, kaon, proton, mixed or all")
    p.add_argument("--target", choices=("log", "linear"), default="log")
    p.add_argument("--no-bias-correction", action="store_true")
    p.add_argument("--features", nargs="+", default=None)
    p.add_argument("--sigma-time", type=float, default=None, metavar="NS")
    p.add_argument("--seeds", type=int, default=1, metavar="N")
    p.add_argument("--n-bins", type=int, default=13)
    p.add_argument("--log-bins", action="store_true")
    p.add_argument("--grid-search", action="store_true")
    return p


def resolve_modes(requested: Sequence[str]) -> List[str]:
    if "all" in requested:
        return list(paths.PARTICLES)
    unknown = [m for m in requested if m not in MODES]
    if unknown:
        raise SystemExit(f"unknown mode(s): {', '.join(unknown)}; expected among {MODES} or 'all'")
    return list(dict.fromkeys(requested))


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.seeds < 1:
        raise SystemExit("--seeds must be at least 1")

    cfg = Config(
        target=args.target,
        correct_target_bias=not args.no_bias_correction,
        smear_sigma_ns=args.sigma_time,
        n_bins=args.n_bins,
        log_bins=args.log_bins,
        grid_search=args.grid_search,
    )
    if args.features:
        cfg = replace(cfg, feature_cols=list(args.features))

    for mode in resolve_modes(args.particles):
        try:
            run_mode(mode, cfg, args.seeds)
        except (paths.DatasetNotFound, FileNotFoundError) as exc:
            logger.error("%s: %s", mode, exc)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
