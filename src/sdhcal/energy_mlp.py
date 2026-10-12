"""Tabular MLP model, training, and evaluation for energy reconstruction.

The regression target, calibration correction, metrics, and early-stopping
sequence are intentionally kept distinct from the PID classifier.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neural_network import MLPRegressor

from . import metrics, paths, tabular
from .smearing import TimeSmearing, smear_time_columns
from .target import make_target

LOGGER = logging.getLogger("energy_mlp_train")
MODES = ("pi", "proton", "mix-pi-proton")

__all__ = [
    "MODES",
    "Config",
    "base_model",
    "choose_hyperparameters",
    "evaluate",
    "fit_model",
    "hyperparameter_grid",
    "load_sample",
    "particles_for_mode",
    "train_calibration_indices",
    "validation_bins",
]


@dataclass
class Config:
    """Configuration of the historical tabular energy MLP workflow."""

    feature_cols: Sequence[str] = field(
        default_factory=lambda: [
            "Thr1",
            "Thr2",
            "Thr3",
            "Radius",
            "Density",
            "NClusters",
            "ratioThr23",
            "Zbary",
            "Zrms",
            "PctHitsFirst10",
            "PlanesWithClusmore2",
            "AvgClustSize",
            "MaxClustSize",
            "N1",
            "N2",
            "N3",
        ]
    )
    calibration_size: float = 0.10
    random_state: int = 42
    target: str = "log"
    correct_target_bias: bool = True
    smear_sigma_ns: float | None = None
    hidden_layer_sizes: tuple[int, ...] = (128, 64, 32)
    alpha: float = 1e-4
    batch_size: int = 256
    learning_rate_init: float = 1e-3
    max_epochs: int = 500
    patience: int = 20
    n_bins: int = 13
    core_method: str = "truncated_rms"
    core_frac: float = 0.9
    grid_search: bool = False
    max_shards_per_particle: int | None = None
    max_events_per_particle: int | None = None

    def smearing(self, seed: int) -> TimeSmearing:
        return TimeSmearing(
            sigma_ns=self.smear_sigma_ns,
            columns=("tMin", "tMax", "tMean", "tSpread"),
            seed=seed,
        )

    def loggable(self, mode: str) -> dict[str, object]:
        return {
            "mode": mode,
            "shards_dir": str(paths.shards_dir()),
            "train_sample": "200k",
            "validation_sample": "discret",
            "calibration_size": self.calibration_size,
            "random_state": self.random_state,
            "target": self.target,
            "correct_target_bias": int(self.correct_target_bias),
            "smear_sigma_ns": self.smear_sigma_ns,
            "hidden_layer_sizes": "x".join(map(str, self.hidden_layer_sizes)),
            "alpha": self.alpha,
            "batch_size": self.batch_size,
            "learning_rate_init": self.learning_rate_init,
            "max_epochs": self.max_epochs,
            "patience": self.patience,
            "grid_search": int(self.grid_search),
            "max_shards_per_particle": self.max_shards_per_particle,
            "max_events_per_particle": self.max_events_per_particle,
        }


def particles_for_mode(mode: str) -> tuple[str, ...]:
    """Return the ordered particle inputs of one energy mode."""
    return ("pi", "proton") if mode == "mix-pi-proton" else (mode,)


def load_sample(sample: str, mode: str, cfg: Config, seed: int) -> pd.DataFrame:
    """Load and prepare one energy sample while retaining its historical cuts."""
    frame = tabular.read_shard_particles(
        sample,
        particles_for_mode(mode),
        features=cfg.feature_cols,
        limit=cfg.max_shards_per_particle,
    )
    if cfg.max_events_per_particle is not None:
        frame = tabular.sample_per_particle(frame, cfg.max_events_per_particle, seed)
    needed = [*cfg.feature_cols, "primaryEnergy", "particlePDG", "eventNumber"]
    frame = tabular.clean_event_rows(frame, needed, reset_index=False)
    frame = frame[frame["primaryEnergy"] > 0].reset_index(drop=True)
    frame = smear_time_columns(frame, cfg.smearing(seed))
    if frame.empty:
        raise RuntimeError(f"No usable event in {sample}/{mode}")
    LOGGER.info("%s/%s: %d events", sample, mode, len(frame))
    return frame


def train_calibration_indices(frame: pd.DataFrame, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """Return the historical particle-and-energy stratified calibration split."""
    return tabular.stratified_energy_split_indices(
        frame["primaryEnergy"].to_numpy(),
        frame["particlePDG"].to_numpy(),
        test_size=cfg.calibration_size,
        random_state=cfg.random_state,
    )


def base_model(cfg: Config, **overrides) -> MLPRegressor:
    """Build the historical scikit-learn energy regressor."""
    parameters = {
        "hidden_layer_sizes": cfg.hidden_layer_sizes,
        "alpha": cfg.alpha,
        "batch_size": cfg.batch_size,
        "learning_rate_init": cfg.learning_rate_init,
        "activation": "relu",
        "solver": "adam",
        "shuffle": True,
        "random_state": cfg.random_state,
    }
    parameters.update(overrides)
    return MLPRegressor(**parameters)


def hyperparameter_grid() -> dict[str, list]:
    """Return a fresh copy of the historical energy grid-search space."""
    return {
        "hidden_layer_sizes": [(64, 32), (128, 64, 32)],
        "alpha": [1e-5, 1e-4, 1e-3],
        "learning_rate_init": [5e-4, 1e-3],
    }


def choose_hyperparameters(x: np.ndarray, z: np.ndarray, cfg: Config) -> dict[str, object]:
    """Run the unchanged optional energy-regression grid search."""
    if not cfg.grid_search:
        return {}
    search = GridSearchCV(
        base_model(cfg, early_stopping=True, max_iter=cfg.max_epochs),
        hyperparameter_grid(),
        scoring="neg_root_mean_squared_error",
        cv=KFold(5, shuffle=True, random_state=cfg.random_state),
        n_jobs=-1,
    )
    search.fit(x, z)
    LOGGER.info("Best grid-search parameters: %s", search.best_params_)
    return search.best_params_


def fit_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_calibration: np.ndarray,
    y_calibration: np.ndarray,
    cfg: Config,
):
    """Fit with the historical target, epoch order, and early stopping."""
    target = make_target(cfg.target, correct_bias=cfg.correct_target_bias)
    z_train = target.forward(y_train)
    z_calibration = target.forward(y_calibration)
    best_parameters = choose_hyperparameters(x_train, z_train, cfg)
    model = base_model(cfg, **best_parameters)
    history = {"train_loss": [], "calibration_loss": []}
    best_model = None
    best_loss = np.inf
    stale = 0
    rng = np.random.default_rng(cfg.random_state)
    for epoch in range(1, cfg.max_epochs + 1):
        order = rng.permutation(len(z_train))
        model.partial_fit(x_train[order], z_train[order])
        train_loss = float(model.loss_)
        calibration_prediction = model.predict(x_calibration)
        calibration_loss = float(mean_squared_error(z_calibration, calibration_prediction))
        history["train_loss"].append(train_loss)
        history["calibration_loss"].append(calibration_loss)
        if epoch == 1 or epoch % 10 == 0:
            LOGGER.info(
                "epoch=%03d train_mse=%.6f calibration_mse=%.6f",
                epoch,
                train_loss,
                calibration_loss,
            )
        if calibration_loss < best_loss - 1e-7:
            best_loss = calibration_loss
            best_model = copy.deepcopy(model)
            stale = 0
        else:
            stale += 1
            if stale >= cfg.patience:
                LOGGER.info("Early stopping after %d epochs", epoch)
                break
    if best_model is None:
        raise RuntimeError("MLP training did not produce a checkpoint")
    correction = target.fit_correction(y_calibration, best_model.predict(x_calibration))
    LOGGER.info("Back-transform correction fitted on 200k calibration: %.6f", correction)
    return best_model, target, history, best_parameters, best_loss


def validation_bins(y_true: np.ndarray, fallback_bins: int) -> np.ndarray:
    """Construct the historical external-validation energy bins."""
    points = np.unique(np.asarray(y_true, dtype=float))
    if len(points) == 1:
        margin = max(0.5, 0.05 * abs(float(points[0])))
        return np.asarray([points[0] - margin, points[0] + margin])
    if len(points) <= 100:
        middle = 0.5 * (points[:-1] + points[1:])
        return np.concatenate(
            [
                [points[0] - 0.5 * (points[1] - points[0])],
                middle,
                [points[-1] + 0.5 * (points[-1] - points[-2])],
            ]
        )
    return metrics.energy_bins(float(points.min()), float(points.max()), fallback_bins)


def evaluate(model, target, x: np.ndarray, y: np.ndarray, cfg: Config):
    """Evaluate reconstructed energy with the historical scientific metrics."""
    prediction = target.inverse(model.predict(x))
    core = metrics.core_bias_resolution(
        metrics.relative_residual(y, prediction),
        method=cfg.core_method,
        frac=cfg.core_frac,
    )
    r2 = float("nan") if np.ptp(y) == 0 else float(r2_score(y, prediction))
    result = {
        "rmse_gev": float(mean_squared_error(y, prediction) ** 0.5),
        "mae_gev": float(mean_absolute_error(y, prediction)),
        "r2": r2,
        "bias": core.bias,
        "bias_err": core.bias_err,
        "resolution": core.sigma,
        "resolution_err": core.sigma_err,
        "energy_scale": float(np.mean(prediction) / np.mean(y)),
        "rms_rel_about_zero": metrics.rms_relative_about_zero(y, prediction),
        "n_validation": int(len(y)),
        "target_bias_correction": float(getattr(target, "factor", 1.0)),
    }
    LOGGER.info(
        "External validation: RMSE=%.3f GeV MAE=%.3f GeV bias=%+.4f resolution=%.4f",
        result["rmse_gev"],
        result["mae_gev"],
        result["bias"],
        result["resolution"],
    )
    return result, prediction
