"""Permutation-invariant convolutional pion/proton classifier on SDHCAL hits.

The model uses shared 1x1 convolutions followed by masked max/mean pooling. It
therefore treats an event as a set of hits: predictions do not depend on CSV row
order or on padding introduced by other events in a batch.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from sdhcal import clustering
from sdhcal import paths as sdhcal_paths
from sdhcal.hit_cnn import ParticleCNN, ParticleDataset, collate_hits, evaluate, train_epoch
from sdhcal.hit_eval import event_energies, save_predictions
from sdhcal.hit_pid import (
    FEATURE_NAMES,
    HitPreprocessor,
    append_csv,
    binary_metrics,
    load_binary_hits,
    seed_everything,
    split_events,
    write_json,
)
from sdhcal.hit_plots import plot_results
from sdhcal.hit_training import BestAucEarlyStopping, balanced_class_weights, new_history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pion-csv",
        default=str(sdhcal_paths.legacy("/gridgroup/ilc/midir/Timing/files/csv/1k_pi_E1to100_flat.csv")),
    )
    parser.add_argument(
        "--proton-csv",
        default=str(sdhcal_paths.legacy("/gridgroup/ilc/midir/Timing/files/csv/1k_proton_E1to100_flat.csv")),
    )
    parser.add_argument("--output-dir", type=Path, default=sdhcal_paths.output_dir() / "pid_hit_cnn")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--comment", default="")
    parser.add_argument("--sigma-time", type=float, default=0.1, help="Gaussian timing resolution in ns")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--external-pion-csv",
        type=Path,
        default=None,
        help="hit CSV of the independent lot; with --external-proton-csv the best "
             "checkpoint is also scored on it, which is what makes this model "
             "comparable with the ones trained on shower variables",
    )
    parser.add_argument("--external-proton-csv", type=Path, default=None)
    parser.add_argument(
        "--cluster-hits",
        action="store_true",
        help="group neighbouring fired pads within a layer into one node; about "
             "four times fewer nodes, done once when the data is loaded",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1:
        raise ValueError("epochs, patience and batch-size must be positive")
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    frame = load_binary_hits(
        args.pion_csv,
        args.proton_csv,
        sigma_time=args.sigma_time,
        seed=args.seed,
    )
    if args.cluster_hits:
        before = len(frame)
        frame = clustering.cluster_hit_frame(frame)
        print(f"clustered {before} hits into {len(frame)} cluster rows")
    splits = split_events(frame, seed=args.seed)
    preprocessor = HitPreprocessor(seed=args.seed).fit(splits["train"])
    datasets = {name: ParticleDataset(split, preprocessor) for name, split in splits.items()}

    loader_options = {
        "batch_size": args.batch_size,
        "collate_fn": collate_hits,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    loaders = {
        name: DataLoader(dataset, shuffle=name == "train", **loader_options)
        for name, dataset in datasets.items()
    }

    train_labels = np.asarray([label for _, label in datasets["train"].samples])
    class_weights = balanced_class_weights(train_labels)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
    n_features = preprocessor.n_features(splits["train"])
    model = ParticleCNN(n_features, args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    run_name = args.run_name or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    checkpoint = run_dir / "best_model.pt"
    history = new_history()
    stopping = BestAucEarlyStopping(args.patience)
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, loaders["train"], optimizer, loss_fn, device)
        val_loss, val_metrics, _, _ = evaluate(model, loaders["val"], loss_fn, device)
        scheduler.step(val_loss)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_metrics["auc"])
        history["val_balanced_accuracy"].append(val_metrics["balanced_accuracy"])
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_auc={val_metrics['auc']:.4f} val_bacc={val_metrics['balanced_accuracy']:.4f}"
        )
        improved, should_stop = stopping.update(val_metrics["auc"])
        if improved:
            torch.save(model.state_dict(), checkpoint)
        elif should_stop:
            print(f"early stopping after {epoch} epochs")
            break

    model.load_state_dict(torch.load(checkpoint, map_location=device))
    test_loss, test_metrics, probabilities, labels = evaluate(model, loaders["test"], loss_fn, device)
    test_metrics.update(
        loss=test_loss,
        best_validation_auc=float(stopping.best_auc),
        events=len(datasets["test"]),
    )

    joblib.dump(preprocessor, run_dir / "preprocessor.joblib")
    np.savez(run_dir / "test_predictions.npz", probability_proton=probabilities, label=labels)

    if bool(args.external_pion_csv) != bool(args.external_proton_csv):
        raise SystemExit("--external-pion-csv and --external-proton-csv go together")
    if args.external_pion_csv:
        external_frame = load_binary_hits(
            args.external_pion_csv,
            args.external_proton_csv,
            sigma_time=args.sigma_time,
            seed=args.seed,
        )
        if args.cluster_hits:
            external_frame = clustering.cluster_hit_frame(external_frame)
        # The preprocessor is the one fitted on the training split: scaling the
        # external lot on its own statistics would leak it into the model.
        external = ParticleDataset(external_frame, preprocessor)
        external_loader = DataLoader(external, shuffle=False, **loader_options)
        _, external_metrics, external_probabilities, external_labels = evaluate(
            model, external_loader, loss_fn, device
        )
        test_metrics.update(
            external_auc=external_metrics["auc"],
            external_balanced_accuracy=external_metrics["balanced_accuracy"],
            external_events=len(external),
        )
        save_predictions(
            run_dir / "external_predictions.npz",
            event_ids=external.event_ids,
            probability_proton=external_probabilities,
            energies=event_energies(external_frame),
            labels=external_labels,
        )
        print(
            f"external_auc={external_metrics['auc']:.4f} "
            f"external_bacc={external_metrics['balanced_accuracy']:.4f} "
            f"({len(external)} events)"
        )
    plot_results(run_dir, history, probabilities, labels, test_metrics)
    configuration = vars(args).copy()
    configuration.update(
        model="permutation_invariant_cnn",
        feature_names=list(FEATURE_NAMES),
        device=str(device),
        split_events={name: len(dataset) for name, dataset in datasets.items()},
    )
    configuration = {
        key: str(value) if isinstance(value, Path) else value for key, value in configuration.items()
    }
    write_json(
        run_dir / "run.json",
        {"configuration": configuration, "history": history, "test_metrics": test_metrics},
    )
    append_csv(
        args.output_dir / "runs.csv",
        {
            "run": run_name,
            "seed": args.seed,
            "sigma_time": args.sigma_time,
            "test_auc": test_metrics["auc"],
            "test_balanced_accuracy": test_metrics["balanced_accuracy"],
            "test_accuracy": test_metrics["accuracy"],
            "comment": args.comment,
        },
    )
    print(f"test_auc={test_metrics['auc']:.4f} test_bacc={test_metrics['balanced_accuracy']:.4f}")
    print(f"results: {run_dir}")


if __name__ == "__main__":
    main()
