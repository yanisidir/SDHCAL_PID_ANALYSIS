#!/usr/bin/env python3
"""Binary pion/proton MLP PID on shower-level variables.

The continuous 200k productions provide training and an internal calibration
split.  The independent 100k fixed-energy productions are used only for the
final external validation.
"""

from __future__ import annotations

import argparse
import copy
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from sdhcal import paths, runlog, shards
from sdhcal.smearing import TimeSmearing, smear_time_columns


LOGGER = logging.getLogger("pid_mlp_train")
PARTICLES = ("pi", "proton")
CLASS_NAMES = (r"$\pi^-$", "p")


@dataclass
class Config:
    feature_cols: Sequence[str] = field(
        default_factory=lambda: [
            "Thr1", "Thr2", "Thr3", "Begin", "Radius", "Density", "NClusters",
            "Zbary", "Zrms", "PctHitsFirst10", "PlanesWithClusmore2", "AvgClustSize",
            "MaxClustSize", "lambda1", "lambda2", "tMin", "tMax", "tMean",
            "Nmax", "z0_fit", "Xmax", "lambda", "nTrackSegments", "eccentricity3D",
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


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_sample(sample: str, cfg: Config, seed: int) -> pd.DataFrame:
    frames = [
        shards.read_sample(
            sample,
            particle,
            features=cfg.feature_cols,
            limit=cfg.max_shards_per_particle,
        )
        for particle in PARTICLES
    ]
    frame = pd.concat(frames, ignore_index=True)
    if cfg.max_events_per_particle is not None:
        frame = pd.concat(
            [
                group.sample(
                    n=min(len(group), cfg.max_events_per_particle),
                    random_state=seed,
                )
                for _, group in frame.groupby("particlePDG", sort=False)
            ],
            ignore_index=True,
        )
    needed = [*cfg.feature_cols, "particlePDG", "primaryEnergy", "eventNumber"]
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=needed).reset_index(drop=True)
    labels = {paths.PARTICLE_PDG["pi"]: 0, paths.PARTICLE_PDG["proton"]: 1}
    frame["label"] = frame["particlePDG"].map(labels).astype(int)
    frame = smear_time_columns(frame, cfg.smearing(seed))
    LOGGER.info("%s sample: %d events", sample, len(frame))
    return frame


def train_calibration_indices(frame: pd.DataFrame, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    if not 0 < cfg.calibration_size < 1:
        raise ValueError("calibration-size must lie strictly between zero and one")
    energy = frame["primaryEnergy"].to_numpy()
    label = frame["label"].to_numpy()
    last_error = None
    for n_bins in range(10, 0, -1):
        quantile = np.zeros(len(frame), dtype=int)
        for value in np.unique(label):
            mask = label == value
            edges = np.unique(np.quantile(energy[mask], np.linspace(0, 1, n_bins + 1)))
            quantile[mask] = np.digitize(energy[mask], edges[1:-1])
        strata = np.asarray(
            [f"{kind}:{bin_}" for kind, bin_ in zip(label, quantile, strict=True)]
        )
        try:
            return train_test_split(
                np.arange(len(frame)),
                test_size=cfg.calibration_size,
                random_state=cfg.random_state,
                stratify=strata,
            )
        except ValueError as error:
            last_error = error
    raise ValueError(f"Cannot create the train/calibration split: {last_error}")


def base_model(cfg: Config, **overrides) -> MLPClassifier:
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


def choose_hyperparameters(x: np.ndarray, y: np.ndarray, cfg: Config) -> dict[str, object]:
    if not cfg.grid_search:
        return {}
    search = GridSearchCV(
        base_model(cfg, early_stopping=True, max_iter=cfg.max_epochs),
        {
            "hidden_layer_sizes": [(64, 32), (128, 64, 32)],
            "alpha": [1e-5, 1e-4, 1e-3],
            "learning_rate_init": [5e-4, 1e-3],
        },
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
        calibration_loss = float(
            log_loss(y_calibration, calibration_probability, labels=[0, 1])
        )
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
    LOGGER.info("External validation:\n%s", classification_report(y, prediction, target_names=PARTICLES))
    return results, probability, matrix


def save_plots(history: dict, probability: np.ndarray, y: np.ndarray, matrix: np.ndarray, log) -> None:
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["calibration_loss"], label="internal calibration")
    axes[0].set(xlabel="Epoch", ylabel="Log loss")
    axes[0].legend()
    axes[1].plot(epochs, history["calibration_auc"])
    axes[1].set(xlabel="Epoch", ylabel="Calibration AUC", ylim=(0, 1))
    figure.tight_layout()
    figure.savefig(log.path("plots", f"training_{log.run_id}.png"), dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(probability[y == 0, 1], bins=50, alpha=0.6, label="true pion")
    axes[0].hist(probability[y == 1, 1], bins=50, alpha=0.6, label="true proton")
    axes[0].set(xlabel="P(proton)", ylabel="Events")
    axes[0].legend()
    percent = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    axes[1].imshow(percent, vmin=0, vmax=1, cmap="Blues")
    for row in range(2):
        for column in range(2):
            axes[1].text(column, row, f"{percent[row, column]:.1%}", ha="center", va="center")
    axes[1].set(xticks=[0, 1], yticks=[0, 1], xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                xlabel="Predicted", ylabel="True")
    figure.tight_layout()
    figure.savefig(log.path("plots", f"validation_{log.run_id}.png"), dpi=160)
    plt.close(figure)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=paths.output_dir() / "pid_mlp")
    parser.add_argument("--sigma-time", type=float, default=0.1)
    parser.add_argument("--no-timing", action="store_true")
    parser.add_argument("--calibration-size", type=float, default=0.10)
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grid-search", action="store_true")
    parser.add_argument("--max-shards-per-particle", type=int)
    parser.add_argument("--max-events-per-particle", type=int)
    parser.add_argument("--comment", default="")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.max_epochs < 1 or args.patience < 1 or args.batch_size < 1:
        raise SystemExit("epochs, patience and batch size must be positive")
    for name in ("max_shards_per_particle", "max_events_per_particle"):
        value = getattr(args, name)
        if value is not None and value < 1:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    cfg = Config(
        calibration_size=args.calibration_size,
        random_state=args.seed,
        smear_sigma_ns=None if args.no_timing else args.sigma_time,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        grid_search=args.grid_search,
        max_shards_per_particle=args.max_shards_per_particle,
        max_events_per_particle=args.max_events_per_particle,
    )
    set_seeds(args.seed)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)8s | %(message)s")
    log = runlog.RunLogger(args.output_dir, name="pid_mlp")
    log.start(cfg.loggable(), features=list(cfg.feature_cols))
    log.log_comment(args.comment)

    continuous = load_sample("200k", cfg, args.seed)
    validation = load_sample("discret", cfg, args.seed + 10_000)
    train_index, calibration_index = train_calibration_indices(continuous, cfg)
    scaler = StandardScaler().fit(continuous.iloc[train_index][list(cfg.feature_cols)])
    x_train = scaler.transform(continuous.iloc[train_index][list(cfg.feature_cols)])
    x_calibration = scaler.transform(continuous.iloc[calibration_index][list(cfg.feature_cols)])
    x_validation = scaler.transform(validation[list(cfg.feature_cols)])
    y_train = continuous.iloc[train_index]["label"].to_numpy()
    y_calibration = continuous.iloc[calibration_index]["label"].to_numpy()
    y_validation = validation["label"].to_numpy()

    model, history, best = fit_model(x_train, y_train, x_calibration, y_calibration, cfg)
    results, probability, matrix = evaluate(model, x_validation, y_validation)
    results["best_parameters"] = ";".join(f"{k}={v}" for k, v in best.items())
    results["best_epoch"] = int(np.argmin(history["calibration_loss"]) + 1)
    model_path = log.path("models", f"pid_mlp_{log.run_id}.joblib")
    joblib.dump(
        {
            "model": model,
            "scaler": scaler,
            "features": list(cfg.feature_cols),
            "particles": list(PARTICLES),
            "label_to_pdg": {0: paths.PARTICLE_PDG["pi"], 1: paths.PARTICLE_PDG["proton"]},
            "smearing": cfg.smearing(args.seed).as_dict(),
        },
        model_path,
    )
    arrays_path = log.path("arrays", f"validation_{log.run_id}.npz")
    np.savez_compressed(
        arrays_path,
        y_true=y_validation,
        probability_proton=probability[:, 1],
        event_number=validation["eventNumber"].to_numpy(),
        primary_energy=validation["primaryEnergy"].to_numpy(),
        train_event_number=continuous.iloc[train_index]["eventNumber"].to_numpy(),
        train_particle_pdg=continuous.iloc[train_index]["particlePDG"].to_numpy(),
        calibration_event_number=continuous.iloc[calibration_index]["eventNumber"].to_numpy(),
        calibration_particle_pdg=continuous.iloc[calibration_index]["particlePDG"].to_numpy(),
        confusion_matrix=matrix,
        train_loss=np.asarray(history["train_loss"]),
        calibration_loss=np.asarray(history["calibration_loss"]),
    )
    save_plots(history, probability, y_validation, matrix, log)
    log.log_metrics(**results)
    log.log_artifacts(model=model_path, predictions=arrays_path)
    print(f"validation_auc={results['auc']:.4f} validation_bacc={results['balanced_accuracy']:.4f}")
    print(f"results: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
