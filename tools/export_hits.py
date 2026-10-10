#!/usr/bin/env python3
"""Export hit-level CSVs for the convolutional and graph PID studies.

Those two read one CSV row per calorimeter hit, and their defaults still point at
a thousand-event legacy sample from a production that no longer matches anything
else in the analysis. This writes the same format from the current lots, so they
can be compared with the shower-variable models on the same events.

Two things are deliberate.

*The event count is capped, and spread across shards.* A hit-level network costs
roughly 30 ms per event per epoch on CPU, so the sample size is a time budget,
not a detail: 100 000 events would be days. Events are taken from every shard
rather than from the first few, because in the fixed-energy lot each shard holds
a single energy - reading the first shard only would train and validate at 5 GeV
and nothing else.

*``shardIndex`` is written.* ``eventNumber`` restarts at zero in every shard, and
``sdhcal.hit_pid`` builds its event identifier from the pair when the column is
present. Without it, sixty events would share each identifier.

Usage::

    python tools/export_hits.py --role train --events-per-shard 50
    python tools/export_hits.py --role val   --events-per-shard 50
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from sdhcal import paths, shards

logging.basicConfig(level=logging.INFO, format="%(levelname)8s | %(message)s")
logger = logging.getLogger("export_hits")

#: Branch in the digitized shard -> column the networks read.
BRANCHES = {
    "x": "x_coords",
    "y": "y_coords",
    "z": "z_coords",
    "time": "time",
    "thr": "thr",
    # Cell indices, needed to group neighbouring pads into clusters. No model
    # reads them directly; sdhcal.clustering turns them into cluster membership.
    "I": "I",
    "J": "J",
    "K": "K",
}

#: Per-event, not per-hit. Carried so the PID comparison can break efficiency down
#: by true energy; sdhcal.hit_pid ignores columns it does not ask for.
EVENT_BRANCHES = {"energy": "primaryEnergy"}


def shard_files(sample: str, particle: str) -> list[Path]:
    """Digitized shards of one lot, ordered by the id the event key is built on."""
    directory = paths.hit_shards_dir() / paths.SHARD_SAMPLES[sample][particle]
    if not directory.is_dir():
        raise SystemExit(
            f"missing hit shards for {particle} of {sample}: {directory}\n"
            "Set SDHCAL_HIT_SHARDS_DIR to the directory holding the digitized lots."
        )
    found = sorted(directory.glob("*_digitized.root"), key=lambda p: shards.shard_id(sample, f"{p.stem}_params.root"))
    if not found:
        raise SystemExit(f"no '*_digitized.root' in {directory}")
    return found


def read_shard(path: Path, sample: str, limit: int | None) -> pd.DataFrame:
    """Flatten one shard to one row per hit, keeping at most ``limit`` events."""
    import uproot

    shard = shards.shard_id(sample, f"{path.stem}_params.root")
    with uproot.open(path) as handle:
        tree = handle["tree"]
        wanted = ["eventNumber", *BRANCHES, *EVENT_BRANCHES]
        # A uproot TTree is not a mapping: keys() is its branch listing.
        branch_names = tree.keys()
        available = {name.split(";")[0] for name in branch_names}
        missing = [name for name in wanted if name not in available]
        if missing:
            raise SystemExit(f"{path.name}: missing branch(es) {', '.join(missing)}")
        arrays = tree.arrays(wanted, library="np", entry_stop=limit)

    event_numbers = arrays["eventNumber"].astype("int64")
    counts = np.fromiter((len(np.asarray(v)) for v in arrays["thr"]), dtype="int64", count=len(event_numbers))
    keep = counts > 0
    if not keep.any():
        return pd.DataFrame()

    frame = pd.DataFrame(
        {
            "eventNumber": np.repeat(event_numbers[keep], counts[keep]),
            "shardIndex": np.int32(shard),
        }
    )
    for branch, column in BRANCHES.items():
        frame[column] = np.concatenate([np.asarray(v) for v, k in zip(arrays[branch], keep, strict=True) if k])
    for branch, column in EVENT_BRANCHES.items():
        frame[column] = np.repeat(np.asarray(arrays[branch], dtype=float)[keep], counts[keep])
    return frame


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--role", default="val", choices=sorted(shards.LOT_ROLES), help="train or val")
    parser.add_argument(
        "--events-per-shard",
        type=int,
        default=50,
        help="events kept from each shard; the total is this times the number of shards (40)",
    )
    parser.add_argument("--particles", nargs="+", default=list(paths.PARTICLES_2CLASS))
    parser.add_argument("--out-dir", type=Path, default=None, help="default: <data>/hits/<role>")
    args = parser.parse_args(argv)

    sample = shards.LOT_ROLES[args.role]
    out_dir = args.out_dir or (paths.data_dir() / "hits" / args.role)
    out_dir.mkdir(parents=True, exist_ok=True)

    for particle in args.particles:
        files = shard_files(sample, particle)
        frames = [read_shard(path, sample, args.events_per_shard) for path in files]
        frames = [f for f in frames if not f.empty]
        if not frames:
            raise SystemExit(f"no usable event for {particle}")
        table = pd.concat(frames, ignore_index=True)

        path = out_dir / f"{particle}_hits.csv"
        table.to_csv(path, index=False)
        n_events = table.groupby(["shardIndex", "eventNumber"], sort=False).ngroups
        logger.info(
            "%-7s %d shards -> %s: %d events, %d hits (%.0f hits/event), %.1f MB",
            particle,
            len(files),
            path.name,
            n_events,
            len(table),
            len(table) / n_events,
            path.stat().st_size / 1e6,
        )
    logger.info("hit CSVs in %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
