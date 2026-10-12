"""Energy LightGBM configuration, training, and evaluation logic.

ROOT I/O, event selection, plots, artifact writing, and CLI orchestration remain
in the analysis entry point.  ``subsample`` is intentionally preserved with
LightGBM's default ``subsample_freq=0``; row subsampling is therefore inactive in
the historical workflow and this module does not silently enable it.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold

from . import metrics, paths, tabular
from .smearing import TimeSmearing
from .target import make_target

LOGGER = logging.getLogger("energy_reco_lgbm")

__all__ = [
    "Config",
    "build_regressor",
    "evaluate",
    "mode_particles",
    "set_global_seeds",
    "train",
]


@dataclass
class Config:
    """Everything that defines a historical energy BDT run."""

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
    test_size: float = 0.20
    val_size: float = 0.10
    random_state: int = 42
    target: str = "log"
    ratio_to_chi2: bool = False
    correct_target_bias: bool = True
    smear_sigma_ns: float | None = None
    smear_columns: Sequence[str] = ("tMin", "tMax", "tMean", "tSpread")
    require_contained: bool = False
    last_layer: int = 47
    exclude_holdout: bool = False
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
    n_bins: int = 13
    log_bins: bool = False
    core_method: str = "truncated_rms"
    core_frac: float = 0.9
    grid_search: bool = False
    param_grid: dict[str, list] = field(
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

    def loggable(self, mode: str) -> dict[str, object]:
        out: dict[str, object] = {
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
    tabular.seed_python_numpy(seed)


def mode_particles(mode: str) -> tuple[str, ...]:
    """Return particles feeding a historical single or inclusive mode."""
    if mode == "mixed":
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
    """Fit on the transformed target and calibrate its inverse on validation."""
    target = make_target(cfg.target, correct_bias=cfg.correct_target_bias)
    z_train, z_val = target.forward(y_train), target.forward(y_val)
    params = {}
    if cfg.grid_search:
        LOGGER.info("Running GridSearchCV on the training split")
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
        LOGGER.info("Best parameters: %s", params)
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
    factor = target.fit_correction(y_val, model.predict(x_val))
    LOGGER.info(
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
) -> tuple[dict[str, float], np.ndarray]:
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
        "rms_rel_about_zero": metrics.rms_relative_about_zero(y_test, y_pred),
        "n_test": int(y_test.size),
        "best_iteration": int(getattr(model, "best_iteration_", 0) or 0),
        "target_bias_correction": float(getattr(target, "factor", 1.0)),
    }
    LOGGER.info(
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
