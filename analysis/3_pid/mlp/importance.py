#!/usr/bin/env python3
"""Permutation importance of a binary PID MLP on external validation data."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance

from train import Config, load_sample


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="artifact written by train.py")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--max-shards-per-particle", type=int)
    parser.add_argument("--max-events-per-particle", type=int)
    args = parser.parse_args()
    if args.repeats < 1:
        raise SystemExit("--repeats must be positive")

    artifact = joblib.load(args.model)
    if artifact.get("particles") != ["pi", "proton"]:
        raise ValueError("Permutation importance requires a binary pion/proton artifact")
    features = list(artifact["features"])
    smearing = artifact.get("smearing", {})
    sigma = smearing.get("smear_sigma_ns")
    if sigma == "none":
        sigma = None
    seed = int(smearing.get("smear_seed", 42))
    cfg = Config(
        feature_cols=features,
        random_state=seed,
        smear_sigma_ns=sigma,
        max_shards_per_particle=args.max_shards_per_particle,
        max_events_per_particle=args.max_events_per_particle,
    )
    validation = load_sample("discret", cfg, seed + 10_000)
    x = artifact["scaler"].transform(validation[features])
    y = validation["label"].to_numpy()
    result = permutation_importance(
        artifact["model"],
        x,
        y,
        scoring="balanced_accuracy",
        n_repeats=args.repeats,
        random_state=seed,
        n_jobs=-1,
    )
    importance = pd.DataFrame(
        {
            "feature": features,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    output = args.output_dir or args.model.parent.parent / "interpretation"
    output.mkdir(parents=True, exist_ok=True)
    importance.to_csv(output / "permutation_importance.csv", index=False)

    plt.figure(figsize=(8, max(4, 0.32 * len(features))))
    ordered = importance.iloc[::-1]
    plt.barh(ordered["feature"], ordered["importance_mean"], xerr=ordered["importance_std"])
    plt.xlabel("Decrease in balanced accuracy")
    plt.title("MLP permutation importance — external validation")
    plt.tight_layout()
    plt.savefig(output / "permutation_importance.png", dpi=160)
    plt.close()

    correlation = validation[features].corr()
    correlation.to_csv(output / "feature_correlation.csv")
    size = max(7, 0.35 * len(features))
    figure, axis = plt.subplots(figsize=(size, size))
    image = axis.imshow(correlation, vmin=-1, vmax=1, cmap="coolwarm")
    axis.set_xticks(range(len(features)), features, rotation=90, fontsize=7)
    axis.set_yticks(range(len(features)), features, fontsize=7)
    figure.colorbar(image, ax=axis, label="Pearson correlation")
    figure.tight_layout()
    figure.savefig(output / "feature_correlation.png", dpi=160)
    plt.close(figure)
    print(f"results: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
