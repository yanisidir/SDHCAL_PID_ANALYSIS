#!/usr/bin/env python3
"""LightGBM energy reconstruction for hadronic showers (pi-, K0_L, p).

Trains one regressor per particle, or a single *inclusive* regressor on the
mixture of all three. The inclusive mode is what the PID-energy coupling study
needs as a genuinely PID-agnostic baseline.

What changed with respect to the first version of this script
-------------------------------------------------------------
* **The training curve existed only on paper.** ``fit`` received a single
  ``eval_set`` but two ``eval_names``, so LightGBM stored the *validation* metric
  under the key ``"train"`` and left ``"valid"`` empty; the guard
  ``if len(train_l2) and len(valid_l2)`` was then always false and no curve was
  ever written. Both sets are now passed, in the order the names declare.
* **sigma(E)/E was not a resolution.** It was the RMS of the relative residual
  about zero, which is ``sqrt(bias^2 + sigma^2)``, and the plotted curve used a
  90 %-truncated RMS with no truncation correction (~21 % low for a Gaussian).
  Bias and resolution are now reported separately by :mod:`sdhcal.metrics`, with
  the truncation corrected and uncertainties attached. The old quantity is still
  logged, as ``rms_rel_about_zero``, so earlier entries stay comparable.
* **Training on log(E) left a scale error.** ``exp(prediction)`` is a geometric
  mean, about ``exp(-sigma^2/2)`` below the arithmetic mean. The back-transform
  now carries Duan's smearing factor, fitted on the validation set only
  (:mod:`sdhcal.target`).
* **A StandardScaler was fitted and shipped for a tree model**, which needs none;
  it only added an artifact that had to travel with the model. Removed.
* **Hard-coded ``/gridgroup`` paths** are resolved through :mod:`sdhcal.paths`.
* **Split fractions were inconsistent** (``test_size`` of the whole sample,
  ``valid_size`` of the remainder). Both are now fractions of the whole sample.
* **Single-split numbers had no uncertainty.** ``--seeds n`` repeats the whole
  fit with n seeds and reports mean +- standard deviation.
* **The run log could not be read back.** It is now written by
  :class:`sdhcal.runlog.RunLogger`, which widens the CSV header instead of
  shifting columns, records the feature list and the smearing, and writes one
  JSON record per run.

Examples
--------
::

    python train.py --particles pi
    python train.py --particles all --seeds 5
    python train.py --particles mixed    # PID-agnostic baseline
    python train.py --particles pi --target linear
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
import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # the backend must be selected before this import

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold


from sdhcal import calib, metrics, paths, rootio, runlog, splits
from sdhcal.smearing import TimeSmearing, smear_time_columns
from sdhcal.target import make_target

LOG_FORMAT = "%(asctime)s | %(levelname)8s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt="%H:%M:%S")
logger = logging.getLogger("energy_reco_lgbm")

#: Particle keys accepted on the command line, plus the inclusive mixture.
PARTICLE_COLORS: Dict[str, str] = {"pi": "tab:red", "proton": "tab:blue", "kaon": "tab:green", "mixed": "0.3"}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class Config:
    """Everything that defines a run. Logged verbatim, feature list included."""

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
            "nHitsTotal",
            "sumThrTotal",
        ]
    )

    # Split: both fractions are of the whole sample.
    test_size: float = 0.20
    val_size: float = 0.10
    random_state: int = 42

    # Target transform.
    target: str = "log"
    #: Predict log(E_true / E_chi2) instead of log(E_true). The regressor then
    #: corrects the semi-digital calibration rather than replacing it, which
    #: separates two explanations of the graph network's gain: the formulation
    #: of the target, or the hit-level information it sees.
    ratio_to_chi2: bool = False
    correct_target_bias: bool = True

    # Time smearing. sigma_ns=None means "ideal timing".
    smear_sigma_ns: Optional[float] = None
    smear_columns: Sequence[str] = ("tMin", "tMax", "tMean", "tSpread")

    # Event selection. The chi2 chain rejects showers that reach the last layer;
    # applying the same cut here is what makes the two methods comparable.
    require_contained: bool = False
    last_layer: int = 47

    #: Drop the events of the held-out file before splitting, so the coupling
    #: study evaluates the regressor on events it never saw.
    exclude_holdout: bool = False

    # LightGBM.
    learning_rate: float = 0.05
    n_estimators: int = 2000
    num_leaves: int = 127
    max_depth: int = -1
    reg_alpha: float = 0.0
    reg_lambda: float = 0.1
    min_child_samples: int = 20
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    early_stopping_rounds: int = 50

    # Evaluation.
    n_bins: int = 13
    log_bins: bool = False
    core_method: str = "truncated_rms"
    core_frac: float = 0.9

    grid_search: bool = False
    param_grid: Dict[str, List] = field(
        default_factory=lambda: {
            "num_leaves": [63, 127, 255],
            "max_depth": [-1, 8, 12],
            "learning_rate": [0.01, 0.05, 0.1],
            "n_estimators": [1000, 2000],
        }
    )

    def smearing(self) -> TimeSmearing:
        return TimeSmearing(
            sigma_ns=self.smear_sigma_ns,
            columns=tuple(self.smear_columns),
            seed=self.random_state,
        )

    def loggable(self, mode: str) -> Dict[str, object]:
        """Flat description of the run for the parameter log."""
        out: Dict[str, object] = {
            "mode": mode,
            "tree_name": self.tree_name,
            "energy_col": self.energy_col,
            "test_size": self.test_size,
            "val_size": self.val_size,
            "random_state": self.random_state,
            "target": self.target,
            "correct_target_bias": int(self.correct_target_bias),
            "require_contained": int(self.require_contained),
            "exclude_holdout": int(self.exclude_holdout),
            "learning_rate": self.learning_rate,
            "n_estimators": self.n_estimators,
            "num_leaves": self.num_leaves,
            "max_depth": self.max_depth,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "min_child_samples": self.min_child_samples,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "early_stopping_rounds": self.early_stopping_rounds,
            "core_method": self.core_method,
            "core_frac": self.core_frac,
            "n_bins": self.n_bins,
            "log_bins": int(self.log_bins),
            "grid_search": int(self.grid_search),
        }
        out.update(self.smearing().as_dict())
        return out


def set_global_seeds(seed: int) -> None:
    """Seed Python and NumPy. LightGBM on several threads stays slightly jittery."""
    random.seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def mode_particles(mode: str) -> Tuple[str, ...]:
    """Particles feeding a mode.

    ``pi`` trains on pions alone; ``mixed`` trains a single inclusive regressor on
    all three species; ``mix-pi-proton`` trains one inclusive regressor on the
    listed subset, which is what the old ``pion_proton_Ereco_lgbm.py`` did.
    """
    if mode == "mixed":
        # Every species of the active study, which is two in the pion/proton
        # study and three in the original one.
        return tuple(paths.active_particles())
    if mode.startswith("mix-"):
        wanted = tuple(mode.split("-")[1:])
        unknown = [p for p in wanted if p not in paths.active_particles()]
        if unknown or len(wanted) < 2:
            raise SystemExit(
                f"bad mixture {mode!r}: expected mix-<particle>-<particle>[...] "
                f"with particles among {paths.active_particles()}"
            )
        return wanted
    return (mode,)


def input_files(mode: str) -> List[Path]:
    """ROOT files feeding a mode."""
    return [paths.dataset("params_merged", particle) for particle in mode_particles(mode)]


def load_dataset(mode: str, cfg: Config) -> pd.DataFrame:
    """Load, select, clean and smear the events of one mode."""
    needed = list(dict.fromkeys([*cfg.feature_cols, cfg.energy_col]))
    optional = ["particlePDG", "eventNumber"]
    if cfg.require_contained:
        optional.append("touchesLastLayer")

    frames = []
    for path in input_files(mode):
        available = _branches(path, cfg.tree_name)
        extra = [c for c in optional if c in available]
        frames.append(rootio.load_trees(path, cfg.tree_name, needed + extra))
    df = pd.concat(frames, ignore_index=True)
    df = rootio.clean(df, columns=needed)

    if cfg.exclude_holdout:
        holdout = rootio.load_trees(
            paths.dataset("holdout"), cfg.tree_name, ["particlePDG", "eventNumber"]
        )
        df, dropped = splits.exclude_keys(df, splits.event_keys(holdout))
        logger.info("Mode %-7s excluded %d held-out events", mode, dropped)
        if not dropped:
            raise RuntimeError(
                "--exclude-holdout removed no event: the held-out file shares no event key "
                "with the parameter files, so one of the two was not built from the other."
            )

    if cfg.require_contained:
        df = apply_containment(df, cfg)

    df = df[df[cfg.energy_col] > 0].reset_index(drop=True)
    logger.info("Mode %-7s %8d events after selection", mode, len(df))
    return smear_time_columns(df, cfg.smearing())


def _branches(path: Path, tree: str) -> set:
    import uproot

    with uproot.open(path) as handle:
        return set(handle[tree].keys())


def apply_containment(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Keep only showers that do not reach the last active layer.

    This is the cut the chi2 chain applies (``K == 47``). ShowerAnalyzer now
    writes a ``touchesLastLayer`` flag so the machine-learning chain can apply
    exactly the same selection; without that branch the cut cannot be reproduced
    and the run must stop rather than silently compare different samples.
    """
    if "touchesLastLayer" not in df.columns:
        raise KeyError(
            "--require-contained needs the 'touchesLastLayer' branch, which is written by "
            "the current ShowerAnalyzer. Re-run the parameter extraction, or drop the flag "
            "and state in the comparison that the chi2 and ML selections differ."
        )
    before = len(df)
    out = df[df["touchesLastLayer"] == 0].reset_index(drop=True)
    logger.info(
        "Containment cut (no hit in layer %d): kept %d of %d events (%.2f %%)",
        cfg.last_layer,
        len(out),
        before,
        100.0 * len(out) / max(before, 1),
    )
    return out


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_regressor(cfg: Config, seed: int) -> lgb.LGBMRegressor:
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
        random_state=seed,
        n_jobs=-1,
        metric="l2",
        verbose=-1,
    )


def train(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_val: pd.DataFrame,
    y_val: np.ndarray,
    cfg: Config,
    seed: int,
):
    """Fit the regressor on the transformed target, early-stopping on validation.

    Both evaluation sets are registered, in the order their names declare, so
    ``evals_result_`` really contains a train curve and a validation curve.
    """
    target = make_target(cfg.target, correct_bias=cfg.correct_target_bias)
    z_train, z_val = target.forward(y_train), target.forward(y_val)

    params = {}
    if cfg.grid_search:
        logger.info("Running GridSearchCV on the training split")
        grid = GridSearchCV(
            build_regressor(cfg, seed),
            cfg.param_grid,
            cv=KFold(n_splits=5, shuffle=True, random_state=seed),
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
            verbose=1,
        )
        grid.fit(x_train, z_train)
        params = grid.best_params_
        logger.info("Best parameters: %s", params)

    model = build_regressor(cfg, seed)
    if params:
        model.set_params(**params)
    model.fit(
        x_train,
        z_train,
        eval_set=[(x_train, z_train), (x_val, z_val)],
        eval_names=["train", "valid"],
        eval_metric="l2",
        callbacks=[
            lgb.early_stopping(cfg.early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=200),
        ],
    )

    # The bias correction of the back-transform is measured on validation data,
    # never on the test set.
    factor = target.fit_correction(y_val, model.predict(x_val))
    logger.info(
        "Back-transform bias correction (Duan smearing factor): %.5f  [%+.2f %% on the energy scale]",
        factor,
        100.0 * (factor - 1.0),
    )
    return model, target, params


def evaluate(
    model: lgb.LGBMRegressor,
    target,
    x_test: pd.DataFrame,
    y_test: np.ndarray,
    cfg: Config,
    scale: np.ndarray | None = None,
) -> Tuple[Dict[str, float], np.ndarray]:
    """Compute the test metrics, keeping bias and resolution apart.

    ``scale`` puts both the truth and the prediction back into GeV when the
    regressor was trained on a ratio. A resolution quoted on a ratio is not
    comparable with anything else in the benchmark, so the conversion happens
    here rather than being left to the caller.
    """
    y_pred = target.inverse(model.predict(x_test))
    if scale is not None:
        y_pred = y_pred * scale
        y_test = y_test * scale

    core = metrics.core_bias_resolution(
        metrics.relative_residual(y_test, y_pred),
        method=cfg.core_method,
        frac=cfg.core_frac,
    )
    out = {
        "rmse_gev": float(mean_squared_error(y_test, y_pred) ** 0.5),
        "mae_gev": float(mean_absolute_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred)),
        "bias": core.bias,
        "bias_err": core.bias_err,
        "resolution": core.sigma,
        "resolution_err": core.sigma_err,
        "energy_scale": float(np.mean(y_pred) / np.mean(y_test)),
        # Historical definition, kept so older rows stay comparable.
        "rms_rel_about_zero": metrics.rms_relative_about_zero(y_test, y_pred),
        "n_test": int(y_test.size),
        "best_iteration": int(getattr(model, "best_iteration_", 0) or 0),
        "target_bias_correction": float(getattr(target, "factor", 1.0)),
    }
    logger.info(
        "Test: RMSE=%.3f GeV  MAE=%.3f GeV  R2=%.4f | bias=%+.4f+-%.4f  resolution=%.4f+-%.4f "
        "| scale=%.4f  (historical rms=%.4f)",
        out["rmse_gev"],
        out["mae_gev"],
        out["r2"],
        out["bias"],
        out["bias_err"],
        out["resolution"],
        out["resolution_err"],
        out["energy_scale"],
        out["rms_rel_about_zero"],
    )
    return out, y_pred


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_training_curve(evals: Dict[str, Dict[str, List[float]]], out: Path) -> None:
    train_curve = evals.get("train", {}).get("l2", [])
    valid_curve = evals.get("valid", {}).get("l2", [])
    if not train_curve or not valid_curve:
        logger.warning("No training curve available (train=%d, valid=%d points)", len(train_curve), len(valid_curve))
        return
    plt.figure(figsize=(6, 4))
    plt.plot(train_curve, label="train")
    plt.plot(valid_curve, "--", label="validation")
    plt.xlabel("Boosting iteration")
    plt.ylabel("L2 loss on the transformed target")
    plt.title("Training and validation loss")
    plt.legend()
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
    lim = [float(min(y_true.min(), np.min(means))), float(max(y_true.max(), np.max(means)))]
    plt.plot(lim, lim, "r--", linewidth=1, label="Ideal")
    plt.xlim(lim)
    plt.ylim(lim)
    plt.xlabel("True energy [GeV]")
    plt.ylabel(r"$\langle E_{\rm reco}\rangle$ [GeV]")
    plt.title("Linearity")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def plot_bias_and_resolution(y_true: np.ndarray, y_pred: np.ndarray, out: Path, cfg: Config) -> Dict[str, np.ndarray]:
    """Bias and resolution versus true energy, on one figure, with error bars."""
    binned = metrics.binned_bias_resolution(
        y_true,
        y_pred,
        n_bins=cfg.n_bins,
        log_bins=cfg.log_bins,
        method=cfg.core_method,
        frac=cfg.core_frac,
    )
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(6.5, 7), sharex=True)

    top.errorbar(binned["center"], binned["bias"], yerr=binned["bias_err"], fmt="o", capsize=3)
    top.axhline(0.0, linestyle="--", linewidth=1, color="0.4")
    top.set_ylabel(r"bias $\langle \Delta E/E \rangle$")
    top.set_title("Bias and resolution versus true energy")
    top.grid(alpha=0.3)

    bottom.errorbar(
        binned["center"], binned["sigma"], yerr=binned["sigma_err"], fmt="o", capsize=3, color="tab:purple"
    )
    bottom.set_xlabel("True energy [GeV]")
    bottom.set_ylabel(r"resolution $\sigma(\Delta E/E)$")
    bottom.grid(alpha=0.3)
    if cfg.log_bins:
        bottom.set_xscale("log")

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return binned


def plot_feature_importance(model: lgb.LGBMRegressor, features: Sequence[str], out: Path) -> pd.DataFrame:
    importance = pd.DataFrame(
        {"feature": list(features), "gain": model.booster_.feature_importance(importance_type="gain")}
    ).sort_values("gain", ascending=False)
    plt.figure(figsize=(7, max(3.0, 0.32 * len(features))))
    plt.barh(importance["feature"], importance["gain"])
    plt.gca().invert_yaxis()
    plt.xlabel("Total gain")
    plt.title("LightGBM feature importance")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    return importance


# ---------------------------------------------------------------------------
# One mode, one or several seeds
# ---------------------------------------------------------------------------
def run_single_seed(mode: str, cfg: Config, seed: int, log: runlog.RunLogger) -> Dict[str, float]:
    """Train and evaluate once, write the artifacts, return the test metrics."""
    set_global_seeds(seed)
    seeded = replace(cfg, random_state=seed)

    df = load_dataset(mode, seeded)
    x = df[list(seeded.feature_cols)].astype(np.float32)
    y = df[seeded.energy_col].to_numpy(dtype=np.float64)

    baseline = None
    if seeded.ratio_to_chi2:
        # The pion coefficients for every event: a baseline that needs no
        # particle identity, so it is reproducible at inference.
        coefficients = calib.load("pi")
        baseline = calib.ereco(
            df["N1"].to_numpy(), df["N2"].to_numpy(), df["N3"].to_numpy(), coefficients.values
        )
        usable = np.isfinite(baseline) & (baseline > 0)
        if not usable.all():
            logger.info("Dropping %d event(s) with no chi2 baseline", int((~usable).sum()))
            df, x, y, baseline = df[usable], x[usable], y[usable], baseline[usable]
        logger.info(
            "Target is log(E_true / E_chi2); baseline from the pi coefficients (fitted %s)",
            "in-sample" if coefficients.fitted_in_sample else "out-of-sample",
        )
        y = y / baseline

    split = splits.split_indices(
        len(df), test_size=seeded.test_size, val_size=seeded.val_size, seed=seed
    )
    logger.info("Split sizes: %s", split.sizes())

    model, target, best_params = train(
        x.iloc[split.train], y[split.train], x.iloc[split.val], y[split.val], seeded, seed
    )
    results, y_pred = evaluate(
        model,
        target,
        x.iloc[split.test],
        y[split.test],
        seeded,
        scale=None if baseline is None else baseline[split.test],
    )
    results["seed"] = seed
    if best_params:
        results["best_params"] = ";".join(f"{k}={v}" for k, v in best_params.items())

    tag = f"{mode}_{log.run_id}_seed{seed}"
    plots = log.base_dir / "plots"
    model_path = log.path("models", f"lgbm_regressor_{tag}.joblib")
    joblib.dump(
        {
            "model": model,
            "target": target,
            "features": list(seeded.feature_cols),
            # Read by tools/predict.py: without it the prediction is a bare
            # correction factor and nothing downstream would notice.
            "baseline": "chi2_pi" if seeded.ratio_to_chi2 else None,
        },
        model_path,
    )

    plot_training_curve(model.evals_result_, log.path("plots", f"training_curve_{tag}.png"))
    plot_linearity(y[split.test], y_pred, log.path("plots", f"linearity_{tag}.png"), seeded)
    binned = plot_bias_and_resolution(
        y[split.test], y_pred, log.path("plots", f"bias_resolution_{tag}.png"), seeded
    )
    importance = plot_feature_importance(
        model, list(seeded.feature_cols), log.path("plots", f"feature_importance_{tag}.png")
    )
    importance.to_csv(log.path("tables", f"feature_importance_{tag}.csv"), index=False)

    arrays = log.path("arrays", f"test_and_pred_{tag}.npz")
    np.savez_compressed(
        arrays,
        y_true=y[split.test],
        y_pred=y_pred,
        test_index=split.test,
        bin_center=binned["center"],
        bin_bias=binned["bias"],
        bin_bias_err=binned["bias_err"],
        bin_sigma=binned["sigma"],
        bin_sigma_err=binned["sigma_err"],
    )
    log.log_artifacts(**{f"model_seed{seed}": model_path, f"arrays_seed{seed}": arrays, "plots": plots})
    return results


def run_mode(mode: str, cfg: Config, n_seeds: int) -> Dict[str, float]:
    """Run one mode with ``n_seeds`` independent splits and summarise."""
    out_dir = paths.output_dir() / "energy_reco_lgbm" / mode
    log = runlog.RunLogger(out_dir, name="energy_reco")
    log.start(cfg.loggable(mode), features=list(cfg.feature_cols))
    logger.info("=== mode %s | run %d | %d seed(s) ===", mode, log.run_id, n_seeds)

    per_seed = [run_single_seed(mode, cfg, cfg.random_state + i, log) for i in range(n_seeds)]
    pd.DataFrame(per_seed).to_csv(log.path("tables", f"per_seed_{mode}_{log.run_id}.csv"), index=False)

    summary: Dict[str, float] = {"n_seeds": n_seeds}
    for key in ("rmse_gev", "mae_gev", "r2", "bias", "resolution", "energy_scale", "rms_rel_about_zero"):
        values = np.array([r[key] for r in per_seed], dtype=float)
        summary[key] = float(values.mean())
        # With a single seed the spread is unknown; report the fit uncertainty instead.
        summary[f"{key}_spread"] = float(values.std(ddof=1)) if n_seeds > 1 else float("nan")

    log.log_metrics(**summary)
    log.log_comment(
        f"mode={mode} target={cfg.target} smear={cfg.smear_sigma_ns} "
        f"bias={summary['bias']:+.4f} resolution={summary['resolution']:.4f} "
        f"(+-{summary['resolution_spread']:.4f} over {n_seeds} seeds)"
    )
    logger.info(
        "Summary %s: bias=%+.4f  resolution=%.4f +- %.4f  scale=%.4f",
        mode,
        summary["bias"],
        summary["resolution"],
        summary["resolution_spread"],
        summary["energy_scale"],
    )
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--particles",
        nargs="+",
        default=["all"],
        metavar="MODE",
        help="pi, kaon, proton, mixed (one inclusive regressor on all three), "
        "mix-pi-proton (inclusive on a subset), or all (the three single-particle modes)",
    )
    p.add_argument("--target", choices=("log", "linear"), default="log", help="regression target")
    p.add_argument(
        "--ratio-to-chi2",
        action="store_true",
        help="predict log(E_true / E_chi2) rather than the energy: the regressor "
             "corrects the semi-digital calibration instead of replacing it",
    )
    p.add_argument(
        "--no-bias-correction",
        action="store_true",
        help="skip Duan's smearing factor on the log back-transform (reproduces the old scale error)",
    )
    p.add_argument("--features", nargs="+", default=None, help="override the feature list")
    p.add_argument(
        "--exclude-holdout", action="store_true",
        help="drop the held-out events before splitting, so the coupling study evaluates "
             "on events this regressor never saw",
    )
    p.add_argument(
        "--sigma-time",
        type=float,
        default=None,
        metavar="NS",
        help="width of the Gaussian time smearing, in ns; omitted means ideal timing",
    )
    p.add_argument(
        "--require-contained",
        action="store_true",
        help="keep only showers with no hit in the last layer, matching the chi2 selection",
    )
    p.add_argument("--seeds", type=int, default=1, metavar="N", help="number of independent splits")
    p.add_argument("--n-bins", type=int, default=13, help="energy bins for the bias/resolution curves")
    p.add_argument("--log-bins", action="store_true", help="use logarithmic energy bins")
    p.add_argument(
        "--core-method",
        choices=("truncated_rms", "iterative_gauss", "full_rms"),
        default="truncated_rms",
        help="resolution estimator",
    )
    p.add_argument("--grid-search", action="store_true", help="search hyper-parameters first")
    return p


def resolve_modes(requested: Sequence[str]) -> List[str]:
    if "all" in requested:
        return list(paths.active_particles())
    allowed = (*paths.active_particles(), "mixed")
    unknown = [m for m in requested if m not in allowed and not m.startswith("mix-")]
    if unknown:
        raise SystemExit(
            f"unknown mode(s): {', '.join(unknown)}; expected among {allowed}, 'all', "
            "or a mixture such as mix-pi-proton"
        )
    for mode in requested:
        mode_particles(mode)  # validate the mixtures now rather than mid-run
    return list(dict.fromkeys(requested))


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.seeds < 1:
        raise SystemExit("--seeds must be at least 1")

    cfg = Config(
        target=args.target,
        ratio_to_chi2=args.ratio_to_chi2,
        correct_target_bias=not args.no_bias_correction,
        smear_sigma_ns=args.sigma_time,
        require_contained=args.require_contained,
        exclude_holdout=args.exclude_holdout,
        n_bins=args.n_bins,
        log_bins=args.log_bins,
        core_method=args.core_method,
        grid_search=args.grid_search,
    )
    if args.features:
        cfg = replace(cfg, feature_cols=list(args.features))

    summaries = {}
    for mode in resolve_modes(args.particles):
        try:
            summaries[mode] = run_mode(mode, cfg, args.seeds)
        except (paths.DatasetNotFound, FileNotFoundError) as exc:
            logger.error("%s: %s", mode, exc)
            return 1

    if len(summaries) > 1:
        logger.info("--- comparison ---")
        for mode, summary in summaries.items():
            logger.info(
                "%-7s bias=%+.4f  resolution=%.4f  scale=%.4f",
                mode,
                summary["bias"],
                summary["resolution"],
                summary["energy_scale"],
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
