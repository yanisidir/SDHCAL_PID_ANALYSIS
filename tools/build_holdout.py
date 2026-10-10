#!/usr/bin/env python3
"""Build a held-out evaluation sample whose provenance is recorded.

The PID/energy coupling study evaluated on ``val_set.root`` while the classifier
and the regressors had been trained on the ``130k_*_params.root`` files with
their own internal random splits. Nothing in the repository said how
``val_set.root`` had been produced, so there was no way to know whether the
end-to-end numbers were measured on events the models had already seen. This
script removes the doubt: it carves a held-out fraction out of the parameter
files with a stated seed, writes the event identifiers it used, and refuses to
run if the inputs carry no event identifier at all.

The training scripts take their own train/validation/test split from the
*remaining* events by passing ``--exclude-holdout``; the complement is written
to ``<data>/train_pool_<particle>.root`` so both halves are explicit.

Usage::

    python tools/build_holdout.py --fraction 0.15 --seed 20240601
    python tools/build_holdout.py --check          # only verify an existing file
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from sdhcal import paths, rootio, splits

logging.basicConfig(level=logging.INFO, format="%(levelname)8s | %(message)s")
logger = logging.getLogger("build_holdout")

TREE = "paramsTree"
MANIFEST = "val_set_manifest.json"


def read_particle(particle: str) -> pd.DataFrame:
    """Read every branch of one particle's parameter file."""
    import uproot

    path = paths.dataset("params", particle)
    with uproot.open(path) as handle:
        tree = handle[TREE]
        branches = list(tree.keys())
        if "eventNumber" not in branches:
            raise KeyError(
                f"{path.name} has no 'eventNumber' branch. Without a per-event identifier the "
                "held-out sample cannot be proved disjoint from the training events. Add the "
                "branch in ShowerAnalyzer and re-run the parameter extraction."
            )
        df = tree.arrays(branches, library="pd")
    df["__source"] = particle
    logger.info("%-7s %8d events from %s", particle, len(df), path.name)
    return df


def write_tree(df: pd.DataFrame, path: Path) -> None:
    import uproot

    payload = {
        column: df[column].to_numpy()
        for column in df.columns
        if column != "__source" and not pd.api.types.is_object_dtype(df[column])
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with uproot.recreate(path) as handle:
        handle[TREE] = payload
    logger.info("wrote %s (%d events, %d branches)", path, len(df), len(payload))


def check_existing(particles: list[str]) -> int:
    """Verify that the held-out file shares no event with what the models train on.

    What to compare against depends on how the held-out sample came to be, and the
    two cases need opposite answers:

    *Carved out of the parameter files* by this script. Its events are then by
    construction still in those files, so comparing with them would report a 100 %
    overlap for a perfectly good sample. The meaningful counterpart is
    ``train_pool_<particle>.root``, the complement the training scripts read with
    ``--exclude-holdout``.

    *An independent production*, as in the two-class study, where the evaluation
    lot was simulated separately from the training lot. There is no pool, and the
    parameter files are exactly what the models trained on, so they are the right
    counterpart. Note that what the check then establishes is narrow: it catches
    both roles having been pointed at the same lot, but the real separation is a
    property of the simulations, which no event key can prove.

    The pool is therefore preferred when present, and the parameter file used
    otherwise.
    """
    holdout_path = paths.dataset("holdout")
    holdout = rootio.load_trees(holdout_path, TREE, ["particlePDG", "eventNumber"])
    holdout_keys = splits.event_keys(holdout)
    logger.info("held-out sample: %d events (%s)", len(holdout), holdout_path.name)

    failures = 0
    for particle in particles:
        pool_path = holdout_path.parent / f"train_pool_{particle}.root"
        if pool_path.is_file():
            train_path, kind = pool_path, "training pool"
        else:
            train_path, kind = paths.dataset("params", particle, require=False), "training file"
            if not train_path.is_file():
                logger.error(
                    "%-7s neither %s nor %s exists, so there is nothing to compare the "
                    "held-out sample against",
                    particle,
                    pool_path.name,
                    train_path.name,
                )
                failures += 1
                continue
        training = rootio.load_trees(train_path, TREE, ["particlePDG", "eventNumber"])
        try:
            report = splits.assert_disjoint(
                splits.event_keys(training),
                holdout_keys,
                name_a=f"{kind} for {particle} ({train_path.name})",
                name_b="held-out sample",
            )
            logger.info(
                "%-7s disjoint from the %s (%d training events checked)", particle, kind, report["n_a"]
            )
        except splits.NotDisjoint as exc:
            logger.error("%-7s %s", particle, exc)
            failures += 1
    if failures:
        logger.error(
            "The held-out sample overlaps what the models trained on, so any end-to-end number "
            "measured on it is training performance."
        )
    return 1 if failures else 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--particles", nargs="+", default=list(paths.active_particles()))
    parser.add_argument("--fraction", type=float, default=0.15, help="held-out fraction per particle")
    parser.add_argument("--seed", type=int, default=20240601, help="split seed, recorded in the manifest")
    parser.add_argument(
        "--check", action="store_true", help="only verify that the existing held-out file is disjoint"
    )
    parser.add_argument("--write-train-pool", action="store_true", help="also write the complement")
    args = parser.parse_args(argv)

    if args.check:
        return check_existing(args.particles)
    if not 0.0 < args.fraction < 1.0:
        raise SystemExit("--fraction must lie strictly between 0 and 1")

    holdout_frames, pool_frames = [], []
    manifest: dict[str, object] = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fraction": args.fraction,
        "seed": args.seed,
        "tree": TREE,
        "particles": {},
    }

    for particle in args.particles:
        df = read_particle(particle)
        # Hold out whole events, chosen by a seeded permutation of the event ids,
        # so the choice is reproducible and independent of the row order.
        split = splits.split_indices(len(df), test_size=args.fraction, val_size=0.0, seed=args.seed)
        held = df.iloc[split.test].reset_index(drop=True)
        pool = df.iloc[np.sort(np.concatenate([split.train, split.val]))].reset_index(drop=True)

        holdout_frames.append(held)
        pool_frames.append(pool)
        manifest["particles"][particle] = {
            "n_total": int(len(df)),
            "n_holdout": int(len(held)),
            "n_train_pool": int(len(pool)),
            "holdout_event_numbers_sha": _digest(held["eventNumber"].to_numpy()),
        }
        logger.info("%-7s held out %d of %d events", particle, len(held), len(df))

    holdout = pd.concat(holdout_frames, ignore_index=True)
    holdout_path = paths.dataset("holdout", require=False)
    write_tree(holdout, holdout_path)

    # The complement and the manifest live beside the held-out file rather than at
    # the top of the data directory: the two-class study writes its own
    # val_set.root, and a shared name would have one study overwrite the other's
    # provenance.
    beside = holdout_path.parent
    if args.write_train_pool:
        for particle, pool in zip(args.particles, pool_frames, strict=True):
            write_tree(pool, beside / f"train_pool_{particle}.root")

    manifest_path = beside / MANIFEST
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("manifest written to %s", manifest_path)

    # The guarantee is worth asserting rather than assuming.
    holdout_keys = set(splits.event_keys(holdout))
    for particle, pool in zip(args.particles, pool_frames, strict=True):
        overlap = holdout_keys & set(splits.event_keys(pool))
        if overlap:
            logger.error("%s: %d events are in both halves", particle, len(overlap))
            return 1
    logger.info("verified: the held-out sample and the training pool are disjoint")
    return 0


def _digest(values: np.ndarray) -> str:
    import hashlib

    return hashlib.sha256(np.sort(values).astype("int64").tobytes()).hexdigest()[:16]


if __name__ == "__main__":
    raise SystemExit(main())
