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
import glob
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from sdhcal import benchmark, paths, shards

logging.basicConfig(level=logging.INFO, format="%(levelname)8s | %(message)s")
logger = logging.getLogger("import_predictions")

#: Which lot a role name refers to, so the shard stem can be read with the right
#: naming convention.
ROLE_TO_SAMPLE = {"validation": "discret", "val": "discret", "train": "200k"}


#: The networks spell the pion differently from the rest of the pipeline.
PARTICLE_ALIASES = {"pion": "pi", "pi": "pi", "proton": "proton"}


def parse_event_id(event_id: str) -> tuple[str, str, int]:
    """Turn a network's own event identifier into the pipeline's global key.

    Two spellings are in use, because the two families read different files:

    ``role:particle:shard stem:event number``
        the energy GNN, which reads the digitized shards and knows their names.
    ``particle:shard index:event number``
        the hit-level PID networks, which read the exported CSVs where the shard
        id is already a column.

    Both end at the same place: the shard id folded into the event number exactly
    as :func:`sdhcal.shards.global_event_number` does, because the shard-local
    number alone repeats across files.
    """
    parts = event_id.split(":")
    if len(parts) == 4:
        role, particle, stem, number = parts
        sample = ROLE_TO_SAMPLE.get(role)
        if sample is None:
            raise ValueError(f"unknown sample role {role!r} in {event_id!r}; known: {sorted(ROLE_TO_SAMPLE)}")
        shard = shards.shard_id(sample, f"{stem}_params.root")
    elif len(parts) == 3:
        role, (particle, shard_text, number) = "validation", parts
        shard = int(shard_text)
    else:
        raise ValueError(f"cannot read an event id from {event_id!r}; expected three or four fields")

    if particle not in PARTICLE_ALIASES:
        raise ValueError(f"unknown particle {particle!r} in {event_id!r}")
    return role, PARTICLE_ALIASES[particle], int(shards.global_event_number(shard, np.array([int(number)]))[0])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--kind", required=True, choices=("pid", "energy"))
    parser.add_argument("--npz", required=True, help="path or glob; the newest match is used")
    parser.add_argument("--name", required=True, help="short model name for the comparison")
    parser.add_argument(
        "--n-train",
        type=int,
        default=None,
        help="events this model was trained on; recorded beside the predictions so a "
             "comparison cannot quote the number without the budget that produced it",
    )
    args = parser.parse_args(argv)

    matches = sorted(glob.glob(args.npz), key=lambda p: Path(p).stat().st_mtime)
    if not matches:
        raise SystemExit(f"no file matches {args.npz!r}")
    source = Path(matches[-1])
    payload = np.load(source, allow_pickle=True)
    logger.info("reading %s (%s)", source.name, ", ".join(sorted(payload.files)))

    required = {
        "energy": ("y_true", "y_pred", "event_id"),
        # PID carries the true energy rather than a truth label: the species is
        # already in the event id, and the comparison needs the energy to break
        # efficiency down by it.
        "pid": ("y_score", "event_id", "primary_energy"),
    }[args.kind]
    missing = [k for k in required if k not in payload.files]
    if missing:
        raise SystemExit(f"{source.name} lacks {', '.join(missing)}; it holds {sorted(payload.files)}")

    event_ids = [str(v) for v in payload["event_id"]]
    parsed = [parse_event_id(e) for e in event_ids]
    roles = {role for role, _, _ in parsed}
    if roles != {"validation"} and roles != {"val"}:
        logger.warning("predictions mix roles %s; the benchmark expects the validation lot only", sorted(roles))

    table = pd.DataFrame(
        {
            "eventNumber": [number for _, _, number in parsed],
            "particlePDG": [paths.PARTICLE_PDG[particle] for _, particle, _ in parsed],
            "primaryEnergy": np.asarray(
                payload["y_true" if args.kind == "energy" else "primary_energy"], dtype=float
            ),
        }
    )
    if args.kind == "energy":
        table["energyPred"] = np.asarray(payload["y_pred"], dtype=float)
    else:
        table["score"] = np.asarray(payload["y_score"], dtype=float)

    benchmark.check_predictions(table, args.kind)

    out = paths.output_dir() / "benchmark" / args.kind / f"{args.name}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    out.with_suffix(".json").write_text(
        json.dumps({"name": args.name, "kind": args.kind, "n_events": len(table), "n_train": args.n_train, "source": str(source)}, indent=2),
        encoding="utf-8",
    )
    logger.info("wrote %s (%d events, %d energy points)", out, len(table), table["primaryEnergy"].nunique())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
