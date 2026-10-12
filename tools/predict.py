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
import logging
from pathlib import Path

import pandas as pd

from sdhcal import artifacts, paths, rootio
from sdhcal.manifest import EVENT_COUNT_FIELDS
from sdhcal.prediction import (
    CHI2_COLUMNS,
    apply_smearing,
    design,  # noqa: F401 - historical import surface
    predict_chi2,
    predict_energy,
    predict_pid,
)
from sdhcal.prediction_io import prediction_path as output_path
from sdhcal.prediction_io import write_prediction_table

logging.basicConfig(level=logging.INFO, format="%(levelname)8s | %(message)s")
logger = logging.getLogger("predict")

TREE = "paramsTree"


def load_bundle(path: Path, *, kind: str | None = None) -> dict:
    """Load a trusted artifact while preserving the historical CLI diagnostic."""
    try:
        return artifacts.load_bundle(path, kind=kind)
    except artifacts.ArtifactError as error:
        raise SystemExit(
            f"{path} is not a usable artifact: it must carry 'model' and 'features'. "
            "The trainers record the feature list beside the model precisely so that "
            f"evaluation cannot drift from training. ({error})"
        ) from error


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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--kind", required=True, choices=("pid", "energy"))
    parser.add_argument("--model", type=Path, default=None, help="joblib artifact to apply")
    parser.add_argument(
        "--chi2", action="store_true", help="the semi-digital calibration, which has no artifact"
    )
    parser.add_argument(
        "--name", required=True, help="short model name, used for the output file and the legends"
    )
    parser.add_argument(
        "--n-train",
        type=int,
        default=None,
        help="historical requested/recorded training budget; exact artifact counts are "
        "stored separately when available",
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
        recorded_counts = {}
    else:
        bundle = load_bundle(args.model, kind=args.kind)
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
        recorded_counts = bundle.get("training_counts", {})

    table = pd.DataFrame(
        {
            "eventNumber": frame["eventNumber"].to_numpy(),
            "particlePDG": frame["particlePDG"].to_numpy(),
            "primaryEnergy": frame["primaryEnergy"].to_numpy(),
            column: value,
        }
    )
    out = output_path(args.kind, args.name)
    event_counts = dict.fromkeys(EVENT_COUNT_FIELDS)
    event_counts["requested_training_budget"] = args.n_train
    event_counts["external_evaluation_events"] = len(table)
    if isinstance(recorded_counts, dict):
        if isinstance(recorded_counts.get("train"), int):
            event_counts["training_events"] = recorded_counts["train"]
        if isinstance(recorded_counts.get("train_before_resampling"), int):
            event_counts["selected_events"] = recorded_counts["train_before_resampling"]
        for source, destination in (
            ("calibration", "calibration_events"),
            ("validation", "validation_events"),
            ("test", "test_events"),
        ):
            if isinstance(recorded_counts.get(source), int):
                event_counts[destination] = recorded_counts[source]
    write_prediction_table(
        out,
        table,
        args.kind,
        {
            "name": args.name,
            "kind": args.kind,
            "n_events": len(table),
            "n_train": args.n_train,
            "event_counts": event_counts,
            **extra,
        },
    )
    logger.info("wrote %s (%d events)", out, len(table))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
