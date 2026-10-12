"""Binary pion/proton tabular MLP model, training, and evaluation logic.

Data-source and command-line orchestration stay in the analysis entry points.
The sample helper here only applies the PID-specific labeling and smearing
around the shared operations from :mod:`sdhcal.tabular`.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier

from . import paths, tabular
from .smearing import TimeSmearing, smear_time_columns

LOGGER = logging.getLogger("pid_mlp_train")
PARTICLES = ("pi", "proton")
CLASS_NAMES = (r"$\pi^-$", "p")

__all__ = [
    "CLASS_NAMES",
    "PARTICLES",
    "Config",
    "base_model",
    "choose_hyperparameters",
    "evaluate",
    "fit_model",
    "hyperparameter_grid",
    "load_sample",
    "train_calibration_indices",
]


@dataclass
class Config:
    """Configuration of the historical binary PID MLP workflow."""

    feature_cols: Sequence[str] = field(
        default_factory=lambda: [
            "Thr1",
            "Thr2",
            "Thr3",
            "Begin",
            "Radius",
            "Density",
            "NClusters",
            "Zbary",
            "Zrms",
            "PctHitsFirst10",
            "PlanesWithClusmore2",
            "AvgClustSize",
            "MaxClustSize",
            "lambda1",
            "lambda2",
            "tMin",
            "tMax",
            "tMean",
            "Nmax",
            "z0_fit",
            "Xmax",
            "lambda",
            "nTrackSegments",
            "eccentricity3D",
        ]
    )
    calibration_size: float = 0.10
    random_state: int = 42
    smear_sigma_ns: float | None = 0.1
    hidden_layer_sizes: tuple[int, ...] = (128, 64, 32)
    alpha: float = 1e-4
    batch_size: int = 256
    learning_rate_init: float = 1e-3
    max_epochs: int = 300
    patience: int = 20
    grid_search: bool = False
    max_shards_per_particle: int | None = None
    max_events_per_particle: int | None = None

    def smearing(self, seed: int) -> TimeSmearing:
        return TimeSmearing(
            sigma_ns=self.smear_sigma_ns,
            columns=("tMin", "tMax", "tMean"),
            seed=seed,
        )

    def loggable(self) -> dict[str, object]:
        return {
            "particles": ";".join(PARTICLES),
            "shards_dir": str(paths.shards_dir()),
            "train_sample": "200k",
            "validation_sample": "discret",
            "calibration_size": self.calibration_size,
            "random_state": self.random_state,
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


def load_sample(sample: str, cfg: Config, seed: int) -> pd.DataFrame:
    """Load and prepare one pion/proton shard sample in historical row order."""
    frame = tabular.read_shard_particles(
        sample,
        PARTICLES,
        features=cfg.feature_cols,
        limit=cfg.max_shards_per_particle,
    )
    if cfg.max_events_per_particle is not None:
        frame = tabular.sample_per_particle(frame, cfg.max_events_per_particle, seed)
    needed = [*cfg.feature_cols, "particlePDG", "primaryEnergy", "eventNumber"]
    frame = tabular.clean_event_rows(frame, needed)
    labels = {paths.PARTICLE_PDG["pi"]: 0, paths.PARTICLE_PDG["proton"]: 1}
    frame["label"] = frame["particlePDG"].map(labels).astype(int)
    frame = smear_time_columns(frame, cfg.smearing(seed))
    LOGGER.info("%s sample: %d events", sample, len(frame))
    return frame


def train_calibration_indices(frame: pd.DataFrame, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """Return the historical label-and-energy stratified calibration split."""
    return tabular.stratified_energy_split_indices(
        frame["primaryEnergy"].to_numpy(),
        frame["label"].to_numpy(),
        test_size=cfg.calibration_size,
        random_state=cfg.random_state,
    )


def base_model(cfg: Config, **overrides) -> MLPClassifier:
    """Build the historical scikit-learn PID classifier."""
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
    return MLPClassifier(**parameters)


def hyperparameter_grid() -> dict[str, list]:
    """Return a fresh copy of the historical PID grid-search space."""
    return {
        "hidden_layer_sizes": [(64, 32), (128, 64, 32)],
        "alpha": [1e-5, 1e-4, 1e-3],
        "learning_rate_init": [5e-4, 1e-3],
    }


def choose_hyperparameters(x: np.ndarray, y: np.ndarray, cfg: Config) -> dict[str, object]:
    """Run the unchanged optional PID grid search."""
    if not cfg.grid_search:
        return {}
    search = GridSearchCV(
        base_model(cfg, early_stopping=True, max_iter=cfg.max_epochs),
        hyperparameter_grid(),
        scoring="balanced_accuracy",
        cv=StratifiedKFold(5, shuffle=True, random_state=cfg.random_state),
        n_jobs=-1,
    )
    search.fit(x, y)
    LOGGER.info("Best grid-search parameters: %s", search.best_params_)
    return search.best_params_


def fit_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_calibration: np.ndarray,
    y_calibration: np.ndarray,
    cfg: Config,
) -> tuple[MLPClassifier, dict[str, list[float]], dict[str, object]]:
    """Fit with the historical epoch order, checkpointing, and early stopping."""
    best_parameters = choose_hyperparameters(x_train, y_train, cfg)
    model = base_model(cfg, **best_parameters)
    history = {"train_loss": [], "calibration_loss": [], "calibration_auc": []}
    best_model = None
    best_loss = np.inf
    stale = 0
    rng = np.random.default_rng(cfg.random_state)
    for epoch in range(1, cfg.max_epochs + 1):
        order = rng.permutation(len(y_train))
        if epoch == 1:
            model.partial_fit(x_train[order], y_train[order], classes=np.array([0, 1]))
        else:
            model.partial_fit(x_train[order], y_train[order])
        calibration_probability = model.predict_proba(x_calibration)
        train_loss = float(model.loss_)
        calibration_loss = float(log_loss(y_calibration, calibration_probability, labels=[0, 1]))
        calibration_auc = float(roc_auc_score(y_calibration, calibration_probability[:, 1]))
        history["train_loss"].append(train_loss)
        history["calibration_loss"].append(calibration_loss)
        history["calibration_auc"].append(calibration_auc)
        if epoch == 1 or epoch % 10 == 0:
            LOGGER.info(
                "epoch=%03d train_loss=%.5f calibration_loss=%.5f calibration_auc=%.4f",
                epoch,
                train_loss,
                calibration_loss,
                calibration_auc,
            )
        if calibration_loss < best_loss - 1e-6:
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
    return best_model, history, best_parameters


def evaluate(model: MLPClassifier, x: np.ndarray, y: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray]:
    """Evaluate binary pion/proton probabilities with the historical metrics."""
    probability = model.predict_proba(x)
    prediction = np.argmax(probability, axis=1)
    matrix = confusion_matrix(y, prediction, labels=[0, 1])
    results = {
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "auc": float(roc_auc_score(y, probability[:, 1])),
        "log_loss": float(log_loss(y, probability, labels=[0, 1])),
        "n_validation": int(len(y)),
    }
    for index, particle in enumerate(PARTICLES):
        results[f"efficiency_{particle}"] = float(matrix[index, index] / max(matrix[index].sum(), 1))
        results[f"purity_{particle}"] = float(matrix[index, index] / max(matrix[:, index].sum(), 1))
    LOGGER.info(
        "External validation:\n%s",
        classification_report(y, prediction, target_names=PARTICLES),
    )
    return results, probability, matrix
