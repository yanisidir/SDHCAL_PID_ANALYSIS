#!/usr/bin/env python3
"""MLP energy reconstruction for pion and proton showers.

The continuous 200k production is split into fit and internal calibration
subsets.  The independent 100k fixed-energy production is never used during
training and provides the external validation curves and metrics.
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

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from sdhcal import metrics, paths, runlog, shards
from sdhcal.smearing import TimeSmearing, smear_time_columns
from sdhcal.target import make_target


LOGGER = logging.getLogger("energy_mlp_train")
MODES = ("pi", "proton", "mix-pi-proton")


@dataclass
class Config:
    feature_cols: Sequence[str] = field(
        default_factory=lambda: [
            "Thr1", "Thr2", "Thr3", "Radius", "Density", "NClusters",
            "ratioThr23", "Zbary", "Zrms", "PctHitsFirst10",
            "PlanesWithClusmore2", "AvgClustSize", "MaxClustSize", "N1", "N2", "N3",
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
    return ("pi", "proton") if mode == "mix-pi-proton" else (mode,)


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_sample(sample: str, mode: str, cfg: Config, seed: int) -> pd.DataFrame:
    frames = [
        shards.read_sample(
            sample,
            particle,
            features=cfg.feature_cols,
            limit=cfg.max_shards_per_particle,
        )
        for particle in particles_for_mode(mode)
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
    needed = [*cfg.feature_cols, "primaryEnergy", "particlePDG", "eventNumber"]
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=needed)
    frame = frame[frame["primaryEnergy"] > 0].reset_index(drop=True)
    frame = smear_time_columns(frame, cfg.smearing(seed))
    if frame.empty:
        raise RuntimeError(f"No usable event in {sample}/{mode}")
    LOGGER.info("%s/%s: %d events", sample, mode, len(frame))
    return frame


def train_calibration_indices(frame: pd.DataFrame, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    if not 0 < cfg.calibration_size < 1:
        raise ValueError("calibration-size must lie strictly between zero and one")
    particle = frame["particlePDG"].to_numpy()
    energy = frame["primaryEnergy"].to_numpy()
    last_error = None
    for n_bins in range(10, 0, -1):
        quantile = np.zeros(len(frame), dtype=int)
        for value in np.unique(particle):
            selected = particle == value
            edges = np.unique(np.quantile(energy[selected], np.linspace(0, 1, n_bins + 1)))
            quantile[selected] = np.digitize(energy[selected], edges[1:-1])
        strata = np.asarray(
            [f"{kind}:{bin_}" for kind, bin_ in zip(particle, quantile, strict=True)]
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


def base_model(cfg: Config, **overrides) -> MLPRegressor:
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


def choose_hyperparameters(x: np.ndarray, z: np.ndarray, cfg: Config) -> dict[str, object]:
    if not cfg.grid_search:
        return {}
    search = GridSearchCV(
        base_model(cfg, early_stopping=True, max_iter=cfg.max_epochs),
        {
            "hidden_layer_sizes": [(64, 32), (128, 64, 32)],
            "alpha": [1e-5, 1e-4, 1e-3],
            "learning_rate_init": [5e-4, 1e-3],
        },
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
    points = np.unique(np.asarray(y_true, dtype=float))
    if len(points) == 1:
        margin = max(0.5, 0.05 * abs(float(points[0])))
        return np.asarray([points[0] - margin, points[0] + margin])
    if len(points) <= 100:
        middle = 0.5 * (points[:-1] + points[1:])
        return np.concatenate(
            [[points[0] - 0.5 * (points[1] - points[0])], middle,
             [points[-1] + 0.5 * (points[-1] - points[-2])]]
        )
    return metrics.energy_bins(float(points.min()), float(points.max()), fallback_bins)


def evaluate(model, target, x: np.ndarray, y: np.ndarray, cfg: Config):
    prediction = target.inverse(model.predict(x))
    core = metrics.core_bias_resolution(
        metrics.relative_residual(y, prediction), method=cfg.core_method, frac=cfg.core_frac
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
        result["rmse_gev"], result["mae_gev"], result["bias"], result["resolution"],
    )
    return result, prediction


def save_plots(history: dict, y: np.ndarray, prediction: np.ndarray, bins: np.ndarray, log, tag: str):
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, history["train_loss"], label="train")
    plt.plot(epochs, history["calibration_loss"], label="internal calibration")
    plt.xlabel("Epoch")
    plt.ylabel("Training objective / calibration MSE")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(log.path("plots", f"training_{tag}.png"), dpi=160)
    plt.close()

    centers, means, errors = [], [], []
    for low, high in zip(bins[:-1], bins[1:], strict=True):
        selected = (y >= low) & (y < high)
        if selected.sum() < 20:
            continue
        centers.append(float(np.mean(y[selected])))
        means.append(float(np.mean(prediction[selected])))
        errors.append(float(np.std(prediction[selected], ddof=1) / np.sqrt(selected.sum())))
    lower, upper = float(y.min()), float(y.max())
    if lower == upper:
        margin = max(0.5, 0.05 * abs(lower))
        lower, upper = lower - margin, upper + margin
    plt.figure(figsize=(6, 6))
    plt.errorbar(centers, means, yerr=errors, fmt="o", capsize=3, label="MLP")
    plt.plot([lower, upper], [lower, upper], "r--", label="ideal")
    plt.xlabel("True energy [GeV]")
    plt.ylabel("Mean reconstructed energy [GeV]")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(log.path("plots", f"linearity_{tag}.png"), dpi=160)
    plt.close()

    binned = metrics.binned_bias_resolution(
        y, prediction, bins=bins, method="truncated_rms", frac=0.9
    )
    figure, (top, bottom) = plt.subplots(2, 1, figsize=(6.5, 7), sharex=True)
    top.errorbar(binned["center"], binned["bias"], yerr=binned["bias_err"], fmt="o")
    top.axhline(0, color="0.4", linestyle="--")
    top.set_ylabel("Bias")
    top.grid(alpha=0.3)
    bottom.errorbar(binned["center"], binned["sigma"], yerr=binned["sigma_err"], fmt="o")
    bottom.set(xlabel="True energy [GeV]", ylabel="Resolution")
    bottom.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(log.path("plots", f"bias_resolution_{tag}.png"), dpi=160)
    plt.close(figure)
    return binned


def run_mode(mode: str, cfg: Config, comment: str) -> dict[str, float]:
    set_seeds(cfg.random_state)
    output = paths.output_dir() / "energy_reco_mlp" / mode
    log = runlog.RunLogger(output, name="energy_mlp")
    log.start(cfg.loggable(mode), features=list(cfg.feature_cols))
    log.log_comment(comment)
    continuous = load_sample("200k", mode, cfg, cfg.random_state)
    validation = load_sample("discret", mode, cfg, cfg.random_state + 10_000)
    train_index, calibration_index = train_calibration_indices(continuous, cfg)
    scaler = StandardScaler().fit(continuous.iloc[train_index][list(cfg.feature_cols)])
    x_train = scaler.transform(continuous.iloc[train_index][list(cfg.feature_cols)])
    x_calibration = scaler.transform(continuous.iloc[calibration_index][list(cfg.feature_cols)])
    x_validation = scaler.transform(validation[list(cfg.feature_cols)])
    y_train = continuous.iloc[train_index]["primaryEnergy"].to_numpy(dtype=float)
    y_calibration = continuous.iloc[calibration_index]["primaryEnergy"].to_numpy(dtype=float)
    y_validation = validation["primaryEnergy"].to_numpy(dtype=float)
    model, target, history, best, best_loss = fit_model(
        x_train, y_train, x_calibration, y_calibration, cfg
    )
    result, prediction = evaluate(model, target, x_validation, y_validation, cfg)
    result["best_calibration_loss"] = float(best_loss)
    result["best_epoch"] = int(np.argmin(history["calibration_loss"]) + 1)
    result["best_parameters"] = ";".join(f"{k}={v}" for k, v in best.items())
    tag = f"{mode}_{log.run_id}"
    model_path = log.path("models", f"mlp_energy_{tag}.joblib")
    joblib.dump(
        {"model": model, "scaler": scaler, "target": target, "features": list(cfg.feature_cols),
         "particles": list(particles_for_mode(mode)),
         "smearing": cfg.smearing(cfg.random_state).as_dict()},
        model_path,
    )
    bins = validation_bins(y_validation, cfg.n_bins)
    binned = save_plots(history, y_validation, prediction, bins, log, tag)
    arrays_path = log.path("arrays", f"validation_{tag}.npz")
    np.savez_compressed(
        arrays_path,
        y_true=y_validation,
        y_pred=prediction,
        particle_pdg=validation["particlePDG"].to_numpy(),
        event_number=validation["eventNumber"].to_numpy(),
        train_event_number=continuous.iloc[train_index]["eventNumber"].to_numpy(),
        train_particle_pdg=continuous.iloc[train_index]["particlePDG"].to_numpy(),
        calibration_event_number=continuous.iloc[calibration_index]["eventNumber"].to_numpy(),
        calibration_particle_pdg=continuous.iloc[calibration_index]["particlePDG"].to_numpy(),
        bin_center=binned["center"],
        bin_bias=binned["bias"],
        bin_resolution=binned["sigma"],
        train_loss=np.asarray(history["train_loss"]),
        calibration_loss=np.asarray(history["calibration_loss"]),
    )
    log.log_metrics(**result)
    log.log_artifacts(model=model_path, predictions=arrays_path)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--particles", nargs="+", default=["all"], metavar="MODE")
    parser.add_argument("--target", choices=("log", "linear"), default="log")
    parser.add_argument("--no-bias-correction", action="store_true")
    parser.add_argument("--features", nargs="+")
    parser.add_argument("--sigma-time", type=float)
    parser.add_argument("--calibration-size", type=float, default=0.10)
    parser.add_argument("--max-epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-bins", type=int, default=13)
    parser.add_argument("--grid-search", action="store_true")
    parser.add_argument("--max-shards-per-particle", type=int)
    parser.add_argument("--max-events-per-particle", type=int)
    parser.add_argument("--comment", default="")
    return parser.parse_args(argv)


def resolve_modes(requested: Sequence[str]) -> list[str]:
    aliases = {"mixed": "mix-pi-proton"}
    normalized = [aliases.get(value, value) for value in requested]
    if "all" in normalized:
        return ["pi", "proton"]
    unknown = [value for value in normalized if value not in MODES]
    if unknown:
        raise SystemExit(f"unknown mode(s): {', '.join(unknown)}; expected {MODES} or all")
    return list(dict.fromkeys(normalized))


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.max_epochs < 1 or args.patience < 1 or args.batch_size < 1 or args.n_bins < 1:
        raise SystemExit("epochs, patience, batch size and n-bins must be positive")
    for name in ("max_shards_per_particle", "max_events_per_particle"):
        value = getattr(args, name)
        if value is not None and value < 1:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    cfg = Config(
        calibration_size=args.calibration_size,
        random_state=args.seed,
        target=args.target,
        correct_target_bias=not args.no_bias_correction,
        smear_sigma_ns=args.sigma_time,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        n_bins=args.n_bins,
        grid_search=args.grid_search,
        max_shards_per_particle=args.max_shards_per_particle,
        max_events_per_particle=args.max_events_per_particle,
    )
    if args.features:
        cfg.feature_cols = args.features
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)8s | %(message)s")
    for mode in resolve_modes(args.particles):
        run_mode(mode, cfg, args.comment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
