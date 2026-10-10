#!/usr/bin/env python3
"""Evaluate a saved binary PID MLP on the independent discrete production."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
from sklearn.metrics import balanced_accuracy_score, classification_report, roc_auc_score

from train import Config, load_sample


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="joblib artifact written by train.py")
    parser.add_argument("--max-shards-per-particle", type=int)
    parser.add_argument("--max-events-per-particle", type=int)
    args = parser.parse_args()

    artifact = joblib.load(args.model)
    if artifact.get("particles") != ["pi", "proton"]:
        raise ValueError("This evaluator accepts only a pion/proton PID artifact")
    smearing = artifact.get("smearing", {})
    sigma = smearing.get("smear_sigma_ns")
    if sigma == "none":
        sigma = None
    seed = int(smearing.get("smear_seed", 42))
    cfg = Config(
        feature_cols=artifact["features"],
        random_state=seed,
        smear_sigma_ns=sigma,
        max_shards_per_particle=args.max_shards_per_particle,
        max_events_per_particle=args.max_events_per_particle,
    )
    frame = load_sample("discret", cfg, seed + 10_000)
    x = artifact["scaler"].transform(frame[list(cfg.feature_cols)])
    y = frame["label"].to_numpy()
    probability = artifact["model"].predict_proba(x)
    prediction = probability.argmax(axis=1)
    print(classification_report(y, prediction, target_names=["pi", "proton"], digits=4))
    print(f"balanced_accuracy={balanced_accuracy_score(y, prediction):.6f}")
    print(f"auc={roc_auc_score(y, probability[:, 1]):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
