#!/usr/bin/env python3
"""Hit-level GNN energy reconstruction for pion and proton showers.

The script reads the sharded ROOT files under ``data/lots/pi-proton/hits``. Each event already
contains its digitized hits and true ``energy``. The continuous 200k production
is used for training and internal calibration; the independent fixed-energy
100k production is kept intact as the external validation sample. The model can
be particle-specific or PID-agnostic on the combined pion+proton sample.
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import joblib
import numpy as np
import torch
from torch_geometric.loader import DataLoader

from sdhcal import calib, clustering, paths, runlog
from sdhcal.hit_energy import ArrayHitPreprocessor as _ArrayHitPreprocessor
from sdhcal.hit_energy import (
    EnergyGraphDataset,
    ShowerEvent,
    load_particle_events,
    regression_metrics,
    require_tree,
    split_train_calibration,
)
from sdhcal.hit_energy_models import (
    EnergyDeepSets,
    EnergyGNN,
    predict_forward,
    target_mode,  # noqa: F401 - historical import surface
    train_epoch,
    transformed_target,  # noqa: F401 - historical import surface
)
from sdhcal.hit_energy_plots import (
    plot_bias_resolution,
    plot_linearity,
    plot_training,
    validation_energy_bins,
)
from sdhcal.hit_pid import FEATURE_NAMES, seed_everything
from sdhcal.target import make_target

LOGGER = logging.getLogger("gnn_energy_reco")
PARTICLE_MODES = ("pi", "proton", "mix-pi-proton")


class ArrayHitPreprocessor(_ArrayHitPreprocessor):
    """Compatibility path for historical __main__ joblib artefacts."""


# Kept as a module-level alias for callers that imported the old helper.
_tree = require_tree


def selected_particles(mode: str) -> tuple[str, ...]:
    return ("pi", "proton") if mode == "mix-pi-proton" else (mode,)


LOT_DIRECTORIES = {
    "train": {"pi": "200k_pi-", "proton": "200k_proton"},
    "validation": {"pi": "100k_pi-_discret", "proton": "100k_proton_discret"},
}


def input_shards(args: argparse.Namespace, role: str, particle: str) -> list[Path]:
    custom = getattr(args, f"{particle}_{role}_dir")
    directory = Path(custom) if custom else paths.hit_shards_dir() / LOT_DIRECTORIES[role][particle]
    if not directory.is_dir():
        raise FileNotFoundError(
            f"Missing {role} directory {directory}. Set SDHCAL_HIT_SHARDS_DIR or pass "
            f"--{particle}-{role}-dir."
        )
    shards = sorted(
        directory.glob("*_digitized.root"),
        key=lambda path: [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path.name)],
    )
    if args.max_shards_per_particle:
        shards = shards[: args.max_shards_per_particle]
    if not shards:
        raise FileNotFoundError(f"No '*_digitized.root' file in {directory}")
    return shards


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--particles", choices=PARTICLE_MODES, default="mix-pi-proton")
    parser.add_argument("--pi-train-dir", type=Path)
    parser.add_argument("--pi-validation-dir", type=Path)
    parser.add_argument("--proton-train-dir", type=Path)
    parser.add_argument("--proton-validation-dir", type=Path)
    parser.add_argument("--tree-name", default="tree")
    parser.add_argument("--output-dir", type=Path, default=paths.output_dir() / "energy_reco_gnn")
    parser.add_argument("--comment", default="")
    parser.add_argument("--max-shards-per-particle", type=int)
    parser.add_argument("--max-events-per-shard", type=int)
    parser.add_argument("--calibration-size", type=float, default=0.1)
    parser.add_argument("--split-energy-bins", type=int, default=10)
    parser.add_argument("--sigma-time", type=float, default=0.1)
    parser.add_argument(
        "--no-timing",
        action="store_true",
        help="controlled ablation: blank the time feature, keeping the input width, the "
        "architecture and the parameter count identical to a timed run, so the comparison "
        "isolates the information the time carries",
    )
    parser.add_argument("--require-contained", action="store_true")
    parser.add_argument("--scaler-max-hits", type=int, default=500_000)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--time-window", type=float, default=2.0)
    parser.add_argument("--causal", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--undirected", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--knn-chunk-size", type=int, default=512)
    parser.add_argument(
        "--architecture",
        choices=("gat", "deepsets"),
        default="gat",
        help="deepsets drops the kNN graph entirely, which is where the CPU time goes",
    )
    parser.add_argument("--deepsets-blocks", type=int, default=6)
    parser.add_argument(
        "--target",
        choices=("energy", "ratio-chi2"),
        default="energy",
        help="ratio-chi2 trains on log(E_true / E_chi2): the network corrects the "
        "semi-digital calibration instead of replacing it",
    )
    parser.add_argument(
        "--baseline-coefficients",
        choices=("pi", "self"),
        default="pi",
        help="whose chi2 coefficients give the baseline. 'pi' applies the pion "
        "calibration to every event, which needs no particle identity and so "
        "is reproducible at inference; 'self' uses the true species, which is "
        "an oracle and must be compared only against oracle-routed chi2",
    )
    parser.add_argument(
        "--cluster-hits",
        action="store_true",
        help="group neighbouring fired pads within a layer and treat each group as one "
        "node; about four times fewer nodes, computed once at load rather than per epoch",
    )
    parser.add_argument("--bias-correction", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--n-energy-bins", type=int, default=13)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = {
        "batch-size": args.batch_size,
        "epochs": args.epochs,
        "patience": args.patience,
        "k": args.k,
        "knn-chunk-size": args.knn_chunk_size,
        "scaler-max-hits": args.scaler_max_hits,
        "split-energy-bins": args.split_energy_bins,
        "n-energy-bins": args.n_energy_bins,
    }
    bad = [name for name, value in positive.items() if value < 1]
    if bad:
        raise ValueError(f"These arguments must be positive: {', '.join(bad)}")
    if args.hidden_dim < 4 or args.hidden_dim % 4:
        raise ValueError("hidden-dim must be >= 4 and divisible by four")
    if args.sigma_time < 0:
        raise ValueError("sigma-time must be non-negative")
    if not 0 < args.calibration_size < 1:
        raise ValueError("calibration-size must lie strictly between zero and one")
    if args.learning_rate <= 0:
        raise ValueError("learning-rate must be positive")
    if not 0 <= args.dropout < 1:
        raise ValueError("dropout must be in [0, 1)")
    for name in ("max_shards_per_particle", "max_events_per_shard"):
        value = getattr(args, name)
        if value is not None and value < 1:
            raise ValueError(f"{name.replace('_', '-')} must be positive when provided")


def main() -> None:
    args = parse_args()
    validate_args(args)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)8s | %(message)s",
        datefmt="%H:%M:%S",
    )
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode_dir = args.output_dir / args.particles
    log = runlog.RunLogger(mode_dir, name="gnn_energy")
    configuration = vars(args).copy()
    configuration["device"] = str(device)
    configuration["input_shards"] = {
        f"{role}_{particle}": [str(path) for path in input_shards(args, role, particle)]
        for role in ("train", "validation")
        for particle in selected_particles(args.particles)
    }
    node_features = (
        [*clustering.CLUSTER_FEATURE_NAMES[:4], *clustering.CLUSTER_FEATURE_NAMES[5:], "log1p_size"]
        if args.cluster_hits
        else list(FEATURE_NAMES)
    )
    log.start(configuration, features=[*node_features, "log1p_n_nodes"])
    log.log_comment(args.comment)

    train_events: list[ShowerEvent] = []
    validation_events: list[ShowerEvent] = []
    use_baseline = args.target == "ratio-chi2"

    for offset, particle in enumerate(selected_particles(args.particles)):
        coefficients = None
        if use_baseline:
            which = "pi" if args.baseline_coefficients == "pi" else particle
            entry = calib.load(which)
            coefficients = entry.values
            LOGGER.info(
                "%s: chi2 baseline from the %s coefficients (fitted %s)",
                particle,
                which,
                "in-sample" if entry.fitted_in_sample else "out-of-sample",
            )
        train_events.extend(
            load_particle_events(
                particle,
                input_shards(args, "train", particle),
                sample_role="train",
                tree_name=args.tree_name,
                sigma_time=args.sigma_time,
                seed=args.seed + offset,
                max_events_per_shard=args.max_events_per_shard,
                require_contained=args.require_contained,
                cluster_hits=args.cluster_hits,
                baseline_coefficients=coefficients,
            )
        )
        validation_events.extend(
            load_particle_events(
                particle,
                input_shards(args, "validation", particle),
                sample_role="validation",
                tree_name=args.tree_name,
                sigma_time=args.sigma_time,
                seed=args.seed + 10_000 + offset,
                max_events_per_shard=args.max_events_per_shard,
                require_contained=args.require_contained,
                cluster_hits=args.cluster_hits,
                baseline_coefficients=coefficients,
            )
        )
    train_indices, calibration_indices = split_train_calibration(
        train_events,
        calibration_size=args.calibration_size,
        seed=args.seed,
        energy_bins=args.split_energy_bins,
    )
    events = [*train_events, *validation_events]
    split = {
        "train": train_indices,
        "calibration": calibration_indices,
        "validation": np.arange(len(train_events), len(events)),
    }
    LOGGER.info("Split sizes: %s", {name: len(indices) for name, indices in split.items()})
    # Workflow provenance only: these are the exact post-selection counts that
    # already drive the loaders below.  No budget or split is changed here.
    log.log_event_counts(
        selected_events=len(events),
        training_events=len(train_indices),
        calibration_events=len(calibration_indices),
        validation_events=len(validation_events),
    )

    preprocessor = ArrayHitPreprocessor(seed=args.seed, drop_timing=args.no_timing)
    preprocessor.fit(events, split["train"], args.scaler_max_hits)
    datasets = {
        name: EnergyGraphDataset(events, indices, preprocessor, args) for name, indices in split.items()
    }
    LOGGER.info(
        "Graphs are constructed lazily per batch; dataset sizes: %s",
        {name: len(dataset) for name, dataset in datasets.items()},
    )
    loaders = {
        name: DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=name == "train",
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        for name, dataset in datasets.items()
    }

    if args.architecture == "deepsets":
        model = EnergyDeepSets(
            len(node_features), args.hidden_dim, args.dropout, n_blocks=args.deepsets_blocks
        ).to(device)
    else:
        model = EnergyGNN(len(node_features), args.hidden_dim, args.dropout).to(device)
    model.use_baseline = use_baseline
    LOGGER.info(
        "Architecture: %s (%d trainable parameters)",
        args.architecture,
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    checkpoint = log.path("models", f"energy_gnn_{args.particles}_run{log.run_id}.pt")
    history: dict[str, list[float]] = {"train_loss": [], "calibration_loss": []}
    best_calibration_loss = np.inf
    stale_epochs = 0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, loaders["train"], optimizer, device)
        # This is the number early stopping and checkpoint selection run on, so
        # it must be measured against the same target the model is trained on;
        # see target_mode().
        calibration_loss, _, _, _ = predict_forward(model, loaders["calibration"], device)
        scheduler.step(calibration_loss)
        history["train_loss"].append(train_loss)
        history["calibration_loss"].append(calibration_loss)
        LOGGER.info(
            "epoch=%03d train_loss=%.5f calibration_loss=%.5f lr=%.2e",
            epoch,
            train_loss,
            calibration_loss,
            optimizer.param_groups[0]["lr"],
        )
        if calibration_loss < best_calibration_loss:
            best_calibration_loss = calibration_loss
            stale_epochs = 0
            torch.save(model.state_dict(), checkpoint)
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                LOGGER.info("Early stopping after %d epochs", epoch)
                break

    model.load_state_dict(torch.load(checkpoint, map_location=device))
    _, calibration_forward, calibration_true, calibration_refs = predict_forward(
        model, loaders["calibration"], device
    )
    target = make_target("log", correct_bias=args.bias_correction)
    # In ratio mode the network predicts a correction, so the bias correction is
    # fitted on the ratio too, not on the energy.
    calibration_reference = (
        calibration_true / np.asarray([events[int(i)].baseline for i in calibration_refs])
        if use_baseline
        else calibration_true
    )
    correction = target.fit_correction(calibration_reference, calibration_forward)
    validation_loss, validation_forward, validation_true, source_indices = predict_forward(
        model, loaders["validation"], device
    )
    validation_prediction = target.inverse(validation_forward)
    if use_baseline:
        validation_prediction = validation_prediction * np.asarray(
            [events[int(index)].baseline for index in source_indices]
        )
    results = regression_metrics(validation_true, validation_prediction)
    results.update(
        validation_log_loss=validation_loss,
        best_calibration_loss=float(best_calibration_loss),
        target_bias_correction=float(correction),
    )
    LOGGER.info(
        "External validation RMSE=%.3f GeV MAE=%.3f GeV R2=%.4f bias=%+.4f resolution=%.4f",
        results["rmse_gev"],
        results["mae_gev"],
        results["r2"],
        results["bias"],
        results["resolution"],
    )

    preprocessor_path = log.path("models", f"preprocessor_{args.particles}_run{log.run_id}.joblib")
    target_path = log.path("models", f"target_{args.particles}_run{log.run_id}.joblib")
    joblib.dump(preprocessor, preprocessor_path)
    joblib.dump(target, target_path)
    training_plot = log.path("plots", f"training_{args.particles}_run{log.run_id}.png")
    linearity_plot = log.path("plots", f"linearity_{args.particles}_run{log.run_id}.png")
    resolution_plot = log.path("plots", f"bias_resolution_{args.particles}_run{log.run_id}.png")
    plot_training(history, training_plot)
    energy_bins = validation_energy_bins(validation_true, args.n_energy_bins)
    plot_linearity(validation_true, validation_prediction, linearity_plot, energy_bins)
    binned = plot_bias_resolution(validation_true, validation_prediction, resolution_plot, energy_bins)
    prediction_path = log.path("arrays", f"validation_predictions_{args.particles}_run{log.run_id}.npz")
    np.savez_compressed(
        prediction_path,
        y_true=validation_true,
        y_pred=validation_prediction,
        source_index=source_indices,
        event_id=np.asarray([events[int(index)].event_id for index in source_indices]),
        particle=np.asarray([events[int(index)].particle for index in source_indices]),
        bin_center=binned["center"],
        bin_bias=binned["bias"],
        bin_resolution=binned["sigma"],
        train_loss=np.asarray(history["train_loss"]),
        calibration_loss=np.asarray(history["calibration_loss"]),
    )
    log.log_metrics(**results)
    log.log_artifacts(
        checkpoint=checkpoint,
        preprocessor=preprocessor_path,
        target=target_path,
        predictions=prediction_path,
        training_plot=training_plot,
        linearity_plot=linearity_plot,
        bias_resolution_plot=resolution_plot,
    )
    print(f"results: {mode_dir}")


if __name__ == "__main__":
    main()
