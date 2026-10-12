#!/usr/bin/env python3
"""Binary pion/proton MLP PID on shower-level variables.

The continuous 200k productions provide training and an internal calibration
split.  The independent 100k fixed-energy productions are used only for the
final external validation.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from sdhcal import artifacts, paths, pid_mlp, provenance, runlog, tabular
from sdhcal import config as sdconfig

LOGGER = pid_mlp.LOGGER
PARTICLES = pid_mlp.PARTICLES
CLASS_NAMES = pid_mlp.CLASS_NAMES
Config = pid_mlp.Config
load_sample = pid_mlp.load_sample
train_calibration_indices = pid_mlp.train_calibration_indices
base_model = pid_mlp.base_model
choose_hyperparameters = pid_mlp.choose_hyperparameters
fit_model = pid_mlp.fit_model
evaluate = pid_mlp.evaluate


def set_seeds(seed: int) -> None:
    tabular.seed_python_numpy(seed)


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


def build_parser() -> argparse.ArgumentParser:
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
    parser.add_argument("--config", type=Path, default=None, help="optional versioned JSON configuration")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def resolve_configuration(
    argv: Sequence[str] | None = None,
) -> tuple[Config, Path, str, dict]:
    args, explicit = sdconfig.parse_cli(build_parser(), argv)
    defaults = asdict(Config())
    defaults.update(output_dir=paths.output_dir() / "pid_mlp", comment="")

    document = None
    file_values = {}
    if args.config is not None:
        document = sdconfig.load_config(args.config)
        file_values = sdconfig.workflow_values(
            document, "pid_mlp", params_set=paths.params_set()
        )

    cli_values = {}
    mapping = {
        "output_dir": "output_dir",
        "sigma_time": "smear_sigma_ns",
        "calibration_size": "calibration_size",
        "max_epochs": "max_epochs",
        "patience": "patience",
        "batch_size": "batch_size",
        "seed": "random_state",
        "grid_search": "grid_search",
        "max_shards_per_particle": "max_shards_per_particle",
        "max_events_per_particle": "max_events_per_particle",
        "comment": "comment",
    }
    for cli_name, config_name in mapping.items():
        if cli_name in explicit:
            cli_values[config_name] = explicit[cli_name]

    effective = sdconfig.merge_config(
        defaults,
        file_values,
        cli_values,
        nullable_types={
            "smear_sigma_ns": (int, float),
            "max_shards_per_particle": int,
            "max_events_per_particle": int,
        },
    )
    config_keys = set(asdict(Config()))
    cfg = Config(**{key: effective[key] for key in config_keys})
    if explicit.get("no_timing"):
        cfg = replace(cfg, smear_sigma_ns=None)
    output_dir = Path(effective["output_dir"])
    comment = effective["comment"]
    resolved = sdconfig.build_resolved_config(
        "pid_mlp",
        {**asdict(cfg), "output_dir": output_dir, "comment": comment},
        config_path=args.config,
        cli_overrides=[key for key in explicit if key != "config"],
        profile=document.get("profile") if document else None,
        dataset=document.get("dataset") if document else None,
    )
    return cfg, output_dir, comment, resolved


def main(argv: Sequence[str] | None = None) -> int:
    try:
        cfg, output_dir, comment, resolved = resolve_configuration(argv)
    except sdconfig.ConfigError as error:
        raise SystemExit(f"configuration error: {error}") from error
    if cfg.max_epochs < 1 or cfg.patience < 1 or cfg.batch_size < 1:
        raise SystemExit("epochs, patience and batch size must be positive")
    for name in ("max_shards_per_particle", "max_events_per_particle"):
        value = getattr(cfg, name)
        if value is not None and value < 1:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    set_seeds(cfg.random_state)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)8s | %(message)s")
    log = runlog.RunLogger(output_dir, name="pid_mlp")
    log.start(cfg.loggable(), features=list(cfg.feature_cols))
    log.log_comment(comment)
    resolved_path = sdconfig.write_resolved_config(
        log.path("runs", f"pid_mlp_{log.run_id}_resolved_config.json"), resolved
    )
    log.log_artifacts(resolved_config=resolved_path)

    continuous = load_sample("200k", cfg, cfg.random_state)
    validation = load_sample("discret", cfg, cfg.random_state + 10_000)
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
    artifacts.save_bundle(
        model_path,
        {
            "model": model,
            "scaler": scaler,
            "features": list(cfg.feature_cols),
            "particles": list(PARTICLES),
            "label_to_pdg": {0: paths.PARTICLE_PDG["pi"], 1: paths.PARTICLE_PDG["proton"]},
            "smearing": cfg.smearing(cfg.random_state).as_dict(),
        },
        kind="pid",
        model_family="sklearn_mlp",
        preprocessing={
            "scaler": type(scaler).__name__,
            "smearing": cfg.smearing(cfg.random_state).as_dict(),
        },
        training_counts={
            "train": int(len(train_index)),
            "calibration": int(len(calibration_index)),
            "external_validation": int(len(validation)),
        },
        configuration=resolved,
        provenance=provenance.environment_fingerprint(),
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
    print(f"results: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
