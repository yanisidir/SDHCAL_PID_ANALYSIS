#!/usr/bin/env python3
"""MLP energy reconstruction for pion and proton showers.

The continuous 200k production is split into fit and internal calibration
subsets.  The independent 100k fixed-energy production is never used during
training and provides the external validation curves and metrics.
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

from sdhcal import (
    artifacts,
    energy_mlp,
    metrics,
    paths,
    provenance,
    runlog,
    tabular,
)
from sdhcal import config as sdconfig

LOGGER = energy_mlp.LOGGER
MODES = energy_mlp.MODES
Config = energy_mlp.Config
particles_for_mode = energy_mlp.particles_for_mode
load_sample = energy_mlp.load_sample
train_calibration_indices = energy_mlp.train_calibration_indices
base_model = energy_mlp.base_model
choose_hyperparameters = energy_mlp.choose_hyperparameters
fit_model = energy_mlp.fit_model
validation_bins = energy_mlp.validation_bins
evaluate = energy_mlp.evaluate


def set_seeds(seed: int) -> None:
    tabular.seed_python_numpy(seed)


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


def run_mode(
    mode: str,
    cfg: Config,
    comment: str,
    resolved_config: dict | None = None,
) -> dict[str, float]:
    set_seeds(cfg.random_state)
    output = paths.output_dir() / "energy_reco_mlp" / mode
    log = runlog.RunLogger(output, name="energy_mlp")
    log.start(cfg.loggable(mode), features=list(cfg.feature_cols))
    log.log_comment(comment)
    mode_config = (
        sdconfig.update_resolved_values(resolved_config, mode=mode)
        if resolved_config is not None
        else None
    )
    if mode_config is not None:
        resolved_path = sdconfig.write_resolved_config(
            log.path("runs", f"energy_mlp_{log.run_id}_resolved_config.json"),
            mode_config,
        )
        log.log_artifacts(resolved_config=resolved_path)
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
    artifacts.save_bundle(
        model_path,
        {
            "model": model,
            "scaler": scaler,
            "target": target,
            "features": list(cfg.feature_cols),
            "particles": list(particles_for_mode(mode)),
            "smearing": cfg.smearing(cfg.random_state).as_dict(),
        },
        kind="energy",
        model_family="sklearn_mlp",
        preprocessing={
            "scaler": type(scaler).__name__,
            "target": target.as_dict(),
            "smearing": cfg.smearing(cfg.random_state).as_dict(),
        },
        training_counts={
            "train": int(len(train_index)),
            "calibration": int(len(calibration_index)),
            "external_validation": int(len(validation)),
        },
        configuration=mode_config if mode_config is not None else cfg.loggable(mode),
        provenance=provenance.environment_fingerprint(),
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


def build_parser() -> argparse.ArgumentParser:
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
    parser.add_argument("--config", type=Path, default=None, help="optional versioned JSON configuration")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def resolve_modes(requested: Sequence[str]) -> list[str]:
    aliases = {"mixed": "mix-pi-proton"}
    normalized = [aliases.get(value, value) for value in requested]
    if "all" in normalized:
        return ["pi", "proton"]
    unknown = [value for value in normalized if value not in MODES]
    if unknown:
        raise SystemExit(f"unknown mode(s): {', '.join(unknown)}; expected {MODES} or all")
    return list(dict.fromkeys(normalized))


def resolve_configuration(
    argv: Sequence[str] | None = None,
) -> tuple[Config, list[str], str, dict]:
    args, explicit = sdconfig.parse_cli(build_parser(), argv)
    defaults = asdict(Config())
    defaults.update(particles=["all"], comment="")

    document = None
    file_values = {}
    if args.config is not None:
        document = sdconfig.load_config(args.config)
        file_values = sdconfig.workflow_values(
            document, "energy_mlp", params_set=paths.params_set()
        )

    cli_values = {}
    mapping = {
        "particles": "particles",
        "target": "target",
        "features": "feature_cols",
        "sigma_time": "smear_sigma_ns",
        "calibration_size": "calibration_size",
        "max_epochs": "max_epochs",
        "patience": "patience",
        "batch_size": "batch_size",
        "seed": "random_state",
        "n_bins": "n_bins",
        "grid_search": "grid_search",
        "max_shards_per_particle": "max_shards_per_particle",
        "max_events_per_particle": "max_events_per_particle",
        "comment": "comment",
    }
    for cli_name, config_name in mapping.items():
        if cli_name in explicit:
            cli_values[config_name] = explicit[cli_name]
    if explicit.get("no_bias_correction"):
        cli_values["correct_target_bias"] = False

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
    requested = list(effective["particles"])
    comment = effective["comment"]
    resolved = sdconfig.build_resolved_config(
        "energy_mlp",
        {**asdict(cfg), "particles": requested, "comment": comment},
        config_path=args.config,
        cli_overrides=[key for key in explicit if key != "config"],
        profile=document.get("profile") if document else None,
        dataset=document.get("dataset") if document else None,
    )
    return cfg, requested, comment, resolved


def main(argv: Sequence[str] | None = None) -> int:
    try:
        cfg, requested, comment, resolved = resolve_configuration(argv)
    except sdconfig.ConfigError as error:
        raise SystemExit(f"configuration error: {error}") from error
    if cfg.max_epochs < 1 or cfg.patience < 1 or cfg.batch_size < 1 or cfg.n_bins < 1:
        raise SystemExit("epochs, patience, batch size and n-bins must be positive")
    for name in ("max_shards_per_particle", "max_events_per_particle"):
        value = getattr(cfg, name)
        if value is not None and value < 1:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)8s | %(message)s")
    for mode in resolve_modes(requested):
        run_mode(mode, cfg, comment, resolved)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
