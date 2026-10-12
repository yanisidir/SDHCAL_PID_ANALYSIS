#!/usr/bin/env python3
"""Bring a neural network's own output into the benchmark format.

The torch studies write a compressed ``.npz`` of their validation predictions
rather than a table keyed the way :mod:`sdhcal.benchmark` expects, because they
read the digitized shards directly and never see the consolidated files. Rather
than make every network depend on the benchmark, this translates what they
already write.

The translation that matters is the event key. The networks identify an event by
``"<role>:<particle>:<shard stem>:<event number>"``, where the event number is
the one stored in the shard and so restarts at zero in every file. The rest of
the pipeline keys on ``(particlePDG, eventNumber)`` with an event number made
unique across a lot, so the shard stem is turned back into its id and folded in
exactly as :func:`sdhcal.shards.global_event_number` does. Without that step a
network's predictions would silently align with the wrong events.

Usage::

    python tools/import_predictions.py --kind energy \\
        --npz output/energy_reco_gnn/mix-pi-proton/arrays/validation_predictions_*.npz \\
        --name gnn
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import numpy as np

from sdhcal.manifest import EVENT_COUNT_FIELDS, event_counts_from_run_record
from sdhcal.prediction_io import (
    PARTICLE_ALIASES,  # noqa: F401 - historical import surface
    ROLE_TO_SAMPLE,  # noqa: F401 - historical import surface
    find_newest,
    npz_to_frame,
    parse_event_id,  # noqa: F401 - historical import surface
    prediction_path,
    write_prediction_table,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)8s | %(message)s")
logger = logging.getLogger("import_predictions")


def _run_record(source: Path) -> Path | None:
    """Locate metadata written beside known hit-level prediction formats."""
    direct = source.parent / "run.json"
    if direct.is_file():
        return direct
    match = re.search(r"_run([0-9]+)$", source.stem)
    if match:
        candidate = source.parent.parent / "runs" / f"gnn_energy_{match.group(1)}.json"
        if candidate.is_file():
            return candidate
    return None


def _recorded_counts(source: Path, n_train: int | None, n_evaluation: int) -> tuple[dict, Path | None]:
    """Keep requested budgets distinct from exact counts recorded by trainers."""
    record_path = _run_record(source)
    counts = dict.fromkeys(EVENT_COUNT_FIELDS)
    counts["requested_training_budget"] = n_train
    counts["external_evaluation_events"] = n_evaluation
    if record_path is not None:
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as error:
            logger.warning("cannot read run metadata %s: %s", record_path, error)
        else:
            for name, value in event_counts_from_run_record(record).items():
                if value is not None:
                    counts[name] = value
    return counts, record_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--kind", required=True, choices=("pid", "energy"))
    parser.add_argument("--npz", required=True, help="path or glob; the newest match is used")
    parser.add_argument("--name", required=True, help="short model name for the comparison")
    parser.add_argument(
        "--n-train",
        type=int,
        default=None,
        help="historical requested/recorded training budget; retained as n_train for "
        "benchmark compatibility and kept separate from exact run metadata",
    )
    args = parser.parse_args(argv)

    try:
        source = find_newest(args.npz)
    except FileNotFoundError as error:
        raise SystemExit(str(error)) from error
    payload = np.load(source, allow_pickle=True)
    logger.info("reading %s (%s)", source.name, ", ".join(sorted(payload.files)))

    try:
        table, roles = npz_to_frame(payload, args.kind)
    except KeyError as error:
        raise SystemExit(f"{source.name} {error.args[0]}") from error
    if roles != {"validation"} and roles != {"val"}:
        logger.warning(
            "predictions mix roles %s; the benchmark expects the validation lot only", sorted(roles)
        )

    out = prediction_path(args.kind, args.name)
    counts, record_path = _recorded_counts(source, args.n_train, len(table))
    write_prediction_table(
        out,
        table,
        args.kind,
        {
            "name": args.name,
            "kind": args.kind,
            "n_events": len(table),
            "n_train": args.n_train,
            "source": str(source),
            "run_record": str(record_path) if record_path is not None else None,
            "event_counts": counts,
        },
    )
    logger.info("wrote %s (%d events, %d energy points)", out, len(table), table["primaryEnergy"].nunique())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
