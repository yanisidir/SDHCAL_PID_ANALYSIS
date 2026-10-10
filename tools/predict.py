#!/usr/bin/env python3
"""Apply a trained model to the evaluation lot and write one row per event.

The model comparison works by asking every model the same question about the same
events, so the models must not each evaluate themselves: each writes predictions
in one agreed format, and ``tools/compare_pid.py`` and ``tools/compare_energy.py``
do the rest. That also means a new model is comparable as soon as it can write
this table, whatever it is made of.

Nothing model-specific lives here. The artifacts written by the trainers already
declare their own feature list, their scaler when they need one, their time
smearing and their target transform, so this reads the bundle and applies what it
finds. The chi2 estimator has no artifact and is addressed by name.

Usage::

    export SDHCAL_PARAMS_SET=2class
    python tools/predict.py --kind pid    --model models/pid_bdt.joblib --name bdt
    python tools/predict.py --kind energy --model models/energy_bdt_mixed.joblib --name bdt
    python tools/predict.py --kind energy --chi2 --name chi2

Predictions land in ``$SDHCAL_OUT_DIR/benchmark/<kind>/<name>.csv``.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sdhcal import benchmark, calib, paths, rootio
from sdhcal.smearing import TimeSmearing, smear_time_columns

logging.basicConfig(level=logging.INFO, format="%(levelname)8s | %(message)s")
logger = logging.getLogger("predict")

TREE = "paramsTree"
CHI2_COLUMNS = ("N1", "N2", "N3")


def output_path(kind: str, name: str) -> Path:
    return paths.output_dir() / "benchmark" / kind / f"{name}.csv"


def load_bundle(path: Path) -> dict:
    bundle = joblib.load(path)
    if "model" not in bundle or "features" not in bundle:
        raise SystemExit(
            f"{path} is not a usable artifact: it must carry 'model' and 'features'. "
            "The trainers record the feature list beside the model precisely so that "
            "evaluation cannot drift from training."
        )
    return bundle


def load_sample(columns: list[str], *, require_contained: bool) -> pd.DataFrame:
    """Read the evaluation lot, with the selection applied to every model alike."""
    path = paths.dataset("holdout")
    needed = list(dict.fromkeys([*columns, "particlePDG", "eventNumber", "primaryEnergy"]))
    if require_contained:
        needed.append("touchesLastLayer")
    frame = rootio.load_and_clean(path, TREE, needed)
    frame = frame[frame["primaryEnergy"] > 0].reset_index(drop=True)
    if require_contained:
        before = len(frame)
        frame = frame[frame["touchesLastLayer"] == 0].reset_index(drop=True)
        logger.info("Containment cut: %d of %d events kept", len(frame), before)
    logger.info("Evaluation sample: %d events (%s)", len(frame), path.name)
    return frame


def apply_smearing(frame: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """Reapply the smearing the model was trained under, read off the artifact."""
    recorded = bundle.get("smearing") or {}
    sigma = recorded.get("smear_sigma_ns", None)
    sigma = None if sigma in (None, "none") else float(sigma)
    columns = recorded.get("smear_columns")
    smearing = TimeSmearing(
        sigma_ns=sigma,
        columns=tuple(columns.split(";")) if isinstance(columns, str) else TimeSmearing().columns,
        seed=int(recorded.get("smear_seed", 42)),
    )
    return smear_time_columns(frame, smearing)


def design(frame: pd.DataFrame, bundle: dict) -> np.ndarray:
    features = list(bundle["features"])
    missing = [c for c in features if c not in frame.columns]
    if missing:
        raise SystemExit(f"the evaluation lot lacks feature(s): {', '.join(missing)}")
    x = frame[features].to_numpy(dtype=np.float32)
    scaler = bundle.get("scaler")
    return scaler.transform(x) if scaler is not None else x


def predict_pid(bundle: dict, frame: pd.DataFrame, positive_pdg: int) -> np.ndarray:
    """Probability of ``positive_pdg``, whichever class index the model gave it."""
    label_to_pdg = {int(k): int(v) for k, v in (bundle.get("label_to_pdg") or {}).items()}
    if not label_to_pdg:
        label_to_pdg = dict(enumerate(paths.PARTICLE_PDG[p] for p in bundle["particles"]))
    if positive_pdg not in label_to_pdg.values():
        raise SystemExit(f"the model does not know PDG {positive_pdg}; it knows {sorted(label_to_pdg.values())}")
    index = next(label for label, pdg in label_to_pdg.items() if pdg == positive_pdg)

    proba = bundle["model"].predict_proba(design(frame, bundle))
    return np.asarray(proba)[:, index].astype(float)


def predict_energy(bundle: dict, frame: pd.DataFrame) -> np.ndarray:
    raw = bundle["model"].predict(design(frame, bundle))
    target = bundle.get("target")
    value = np.asarray(target.inverse(raw) if target is not None else raw, dtype=float)

    # A regressor trained on log(E_true / E_chi2) outputs a correction factor.
    # The artifact says so; without acting on it the prediction would be a
    # number around one and every metric would be quietly wrong.
    if bundle.get("baseline") == "chi2_pi":
        coefficients = calib.load("pi")
        value = value * calib.ereco(
            frame["N1"].to_numpy(), frame["N2"].to_numpy(), frame["N3"].to_numpy(), coefficients.values
        )
    elif bundle.get("baseline"):
        raise SystemExit(f"unknown baseline {bundle['baseline']!r} in the artifact")
    return value


def predict_chi2(frame: pd.DataFrame) -> np.ndarray:
    """Semi-digital calibration, applied per species with its own coefficients."""
    energy = np.full(len(frame), np.nan)
    for particle, pdg in paths.PARTICLE_PDG.items():
        mask = (frame["particlePDG"] == pdg).to_numpy()
        if not mask.any():
            continue
        try:
            coefficients = calib.load(particle)
        except (FileNotFoundError, KeyError) as exc:
            raise SystemExit(f"no chi2 coefficients for {particle}: {exc}") from exc
        if coefficients.fitted_in_sample:
            logger.warning(
                "%s coefficients are flagged fitted_in_sample: this estimator is being "
                "compared on events it was fitted on",
                particle,
            )
        subset = frame.loc[mask]
        energy[mask] = calib.ereco(
            subset["N1"].to_numpy(), subset["N2"].to_numpy(), subset["N3"].to_numpy(), coefficients.values
        )
    if np.isnan(energy).any():
        raise SystemExit(f"{int(np.isnan(energy).sum())} events got no chi2 estimate")
    return energy


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--kind", required=True, choices=("pid", "energy"))
    parser.add_argument("--model", type=Path, default=None, help="joblib artifact to apply")
    parser.add_argument("--chi2", action="store_true", help="the semi-digital calibration, which has no artifact")
    parser.add_argument("--name", required=True, help="short model name, used for the output file and the legends")
    parser.add_argument(
        "--n-train",
        type=int,
        default=None,
        help="events this model was trained on; recorded beside the predictions so a "
             "comparison cannot quote the number without the budget that produced it",
    )
    parser.add_argument(
        "--positive-pdg",
        type=int,
        default=paths.PARTICLE_PDG["pi"],
        help="PID only: the class the score refers to (default: the pion, the species the report loses)",
    )
    parser.add_argument(
        "--require-contained",
        action="store_true",
        help="keep only showers with no hit in the last layer; pass it to every model or to none",
    )
    args = parser.parse_args(argv)

    if args.chi2 == (args.model is not None):
        raise SystemExit("pass either --model or --chi2, not both and not neither")
    if args.chi2 and args.kind != "energy":
        raise SystemExit("--chi2 is an energy estimator")

    if args.chi2:
        frame = load_sample(list(CHI2_COLUMNS), require_contained=args.require_contained)
        value = predict_chi2(frame)
        column, extra = "energyPred", {"estimator": "chi2"}
    else:
        bundle = load_bundle(args.model)
        columns = list(bundle["features"])
        if bundle.get("baseline"):
            columns += [c for c in CHI2_COLUMNS if c not in columns]
        frame = load_sample(columns, require_contained=args.require_contained)
        frame = apply_smearing(frame, bundle)
        if args.kind == "pid":
            value = predict_pid(bundle, frame, args.positive_pdg)
            column, extra = "score", {"positive_pdg": args.positive_pdg}
        else:
            value = predict_energy(bundle, frame)
            column, extra = "energyPred", {}
        extra["artifact"] = str(args.model)

    table = pd.DataFrame(
        {
            "eventNumber": frame["eventNumber"].to_numpy(),
            "particlePDG": frame["particlePDG"].to_numpy(),
            "primaryEnergy": frame["primaryEnergy"].to_numpy(),
            column: value,
        }
    )
    benchmark.check_predictions(table, args.kind)

    out = output_path(args.kind, args.name)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    out.with_suffix(".json").write_text(
        json.dumps({"name": args.name, "kind": args.kind, "n_events": len(table), "n_train": args.n_train, **extra}, indent=2),
        encoding="utf-8",
    )
    logger.info("wrote %s (%d events)", out, len(table))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
