#!/usr/bin/env python3
"""Consolidate the sharded parameter lots into the files the pipeline reads.

The lots delivered as ``data/lots/pi-proton/params-superseded`` are two independent Geant4 productions
with two roles:

===========  ==========================  ==========================================
Role         Lot                         Why
===========  ==========================  ==========================================
``train``    ``200k_*`` (1-130 GeV)      continuous spectrum: trains the models and
                                         calibrates the chi2
``val``      ``100k_*_discret``          20 fixed energies, 5 000 events each: the
             (5-100 GeV, 5 GeV steps)    evaluation sample, and the natural binning
                                         of the resolution and linearity curves
===========  ==========================  ==========================================

Because they are separate productions, the evaluation sample is **not** carved out
of the training one: nothing has to be held out, ``tools/build_holdout.py`` plays
no part in this study, and the training stages must not be given
``--exclude-holdout``.

The lots already hold the shower variables, so the C++ extraction does not run
again, and they carry the true energy, so nothing has to be merged. What they lack
is a species label, a globally unique event identifier and the last-layer flag -
see :mod:`sdhcal.shards` for why each is needed. This script applies all of that
and writes, under ``<data>/params_2class/``:

* ``<particle>_params.root``  - the training sample, one file per particle;
* ``val_set.root``            - the evaluation sample, both particles in one tree,
  which is what ``paths.dataset("holdout")`` resolves to for this study.

Point the stages at the result by exporting the sample selector; no stage script
changes::

    python tools/build_2class_params.py
    export SDHCAL_PARAMS_SET=2class
    make pid

Usage::

    python tools/build_2class_params.py                 # both roles
    python tools/build_2class_params.py --roles val     # only the evaluation sample
    python tools/build_2class_params.py --check         # verify what exists
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from sdhcal import paths, rootio, shards, splits

logging.basicConfig(level=logging.INFO, format="%(levelname)8s | %(message)s")
logger = logging.getLogger("build_2class_params")

TREE = "paramsTree"
MANIFEST = "manifest.json"


def output_path(role: str, particle: str | None = None) -> Path:
    """Where one role's consolidated file goes, per the two-class layout."""
    directory = paths.data_dir() / "params_2class"
    if role == "val":
        return directory / "val_set.root"
    return directory / f"{particle}_params.root"


def manifest_path() -> Path:
    return paths.data_dir() / "params_2class" / MANIFEST


def write_tree(frame: pd.DataFrame, path: Path) -> None:
    import uproot

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {column: frame[column].to_numpy() for column in frame.columns}
    with uproot.recreate(path) as handle:
        handle[TREE] = payload
    logger.info("wrote %s (%d events, %d branches)", path, len(frame), len(payload))


def _read_frames(role: str, particles: list[str]) -> dict[str, pd.DataFrame]:
    sample = shards.LOT_ROLES[role]
    frames = {}
    for particle in particles:
        files = shards.shard_files(sample, particle)
        logger.info("%-5s %-7s %d shards from %s", role, particle, len(files), files[0].parent)
        frames[particle] = shards.read_sample(sample, particle)
    return frames


def _describe(frame: pd.DataFrame) -> dict[str, object]:
    energies = frame["primaryEnergy"]
    distinct = energies.round(3).nunique()
    out: dict[str, object] = {
        "n_events": int(len(frame)),
        "n_touching_last_layer": int(frame["touchesLastLayer"].sum()),
        "energy_min": float(energies.min()),
        "energy_max": float(energies.max()),
        "n_distinct_energies": int(distinct),
    }
    # The grid lot is meant to have a handful of exact energies with equal
    # statistics; recording them makes a silently mixed lot obvious.
    if distinct <= 32:
        counts = energies.round(3).value_counts().sort_index()
        out["energy_points"] = [float(e) for e in counts.index]
        out["events_per_point"] = sorted({int(c) for c in counts})
    return out


def build(roles: list[str], particles: list[str]) -> int:
    manifest: dict[str, object] = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tree": TREE,
        "source_tree": paths.SHARD_TREE,
        "shards_dir": str(paths.shards_dir()),
        "features": list(shards.FEATURES),
        "roles": {role: shards.LOT_ROLES[role] for role in roles},
        "derived": {
            "particlePDG": "from the lot; the files carry 0 for every event",
            "eventNumber": f"shardIndex * {shards.SHARD_STRIDE} + localEventNumber, "
            f"with shardIndex offset per lot ({shards.LOT_ID_BASE}) so the roles "
            "share no key",
            "touchesLastLayer": f"read from the branch, or any(K == {shards.LAST_LAYER}) "
            "for the delivered lots, which predate it",
            "primaryEnergy": "read from whichever of "
            + ", ".join(shards.ENERGY_BRANCHES)
            + " the shard carries",
        },
        "samples": {},
    }

    keys: dict[str, set[str]] = {}
    for role in roles:
        frames = _read_frames(role, particles)
        entry: dict[str, object] = {"lot": shards.LOT_ROLES[role], "particles": {}}

        for particle, frame in frames.items():
            particle_keys = splits.event_keys(frame)
            if particle_keys.nunique() != len(frame):
                logger.error(
                    "%s/%s: %d of %d event keys are duplicated",
                    role,
                    particle,
                    len(frame) - particle_keys.nunique(),
                    len(frame),
                )
                return 1
            keys[f"{role}/{particle}"] = set(particle_keys)
            entry["particles"][particle] = {
                "pdg": paths.PARTICLE_PDG[particle],
                "n_shards": len(shards.shard_files(shards.LOT_ROLES[role], particle)),
                **_describe(frame),
            }

        if role == "val":
            # Both species in one tree: the coupling stage reads a single
            # evaluation file and separates on particlePDG.
            merged = pd.concat([frames[p] for p in particles], ignore_index=True)
            write_tree(merged, output_path("val"))
        else:
            for particle, frame in frames.items():
                write_tree(frame, output_path(role, particle))

        manifest["samples"][role] = entry

    _report_overlaps(keys)
    manifest_path().parent.mkdir(parents=True, exist_ok=True)
    manifest_path().write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("manifest written to %s", manifest_path())
    logger.info("set SDHCAL_PARAMS_SET=2class to point the stages at these files")
    return 0


def _report_overlaps(keys: dict[str, set[str]]) -> None:
    """State what the key check does and does not establish.

    The two lots are separate productions, so disjointness is a property of how
    they were simulated, not something an event key can prove. The check below
    only catches a configuration error - both roles pointing at the same lot -
    which the per-lot id offset would otherwise hide.
    """
    names = sorted(keys)
    shared = [
        (a, b, len(keys[a] & keys[b]))
        for i, a in enumerate(names)
        for b in names[i + 1 :]
        if keys[a] & keys[b]
    ]
    for a, b, count in shared:
        logger.error("%s and %s share %d event keys", a, b, count)
    if not shared and len(names) > 1:
        logger.info(
            "no event key is shared between the %d sample(s) built; note that the "
            "train/val separation rests on them being independent productions, "
            "which no key can prove",
            len(names),
        )


def check(particles: list[str]) -> int:
    """Verify the consolidated files: keys unique, roles disjoint, branches complete."""
    existing = _read_manifest()
    if existing:
        logger.info("manifest: roles=%s built %s", existing.get("roles"), existing.get("created_utc"))

    needed = [*shards.FEATURES, "particlePDG", "eventNumber", "primaryEnergy", "touchesLastLayer"]
    failures = 0
    keys: dict[str, set[str]] = {}

    targets = [("train", particle) for particle in particles] + [("val", None)]
    for role, particle in targets:
        path = output_path(role, particle)
        if not path.is_file():
            logger.error("%-5s %-7s missing %s", role, particle or "", path)
            failures += 1
            continue
        frame = rootio.load_trees(path, TREE, needed)
        particle_keys = splits.event_keys(frame)
        duplicated = len(frame) - particle_keys.nunique()
        pdgs = sorted(frame["particlePDG"].unique().tolist())
        distinct = frame["primaryEnergy"].round(3).nunique()
        logger.info(
            "%-5s %-7s %7d events, %d duplicate key(s), PDG=%s, %d distinct energies, "
            "last-layer %.1f%%",
            role,
            particle or "both",
            len(frame),
            duplicated,
            pdgs,
            distinct,
            100.0 * frame["touchesLastLayer"].mean(),
        )
        if duplicated:
            failures += 1
        expected = (
            [paths.PARTICLE_PDG[p] for p in particles] if role == "val" else [paths.PARTICLE_PDG[particle]]
        )
        if pdgs != sorted(expected):
            logger.error("%-5s %-7s carries PDG %s, expected %s", role, particle, pdgs, sorted(expected))
            failures += 1
        keys[f"{role}/{particle or 'both'}"] = set(particle_keys)

    names = sorted(keys)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            overlap = keys[a] & keys[b]
            if overlap:
                logger.error("%s and %s share %d event keys", a, b, len(overlap))
                failures += 1
    if failures:
        return 1
    logger.info("verified: keys unique within each file and disjoint between train and val")
    return 0


def check_manifest() -> int:
    """Reject consolidated files built from a different shard directory."""
    existing = _read_manifest()
    if existing is None:
        logger.error("missing or unreadable manifest: %s", manifest_path())
        return 1

    recorded = existing.get("shards_dir")
    expected = paths.shards_dir().resolve()
    if not recorded or Path(str(recorded)).expanduser().resolve() != expected:
        logger.error(
            "consolidated files came from %s, expected %s; rebuild with make params-2class",
            recorded or "an unknown shard directory",
            expected,
        )
        return 1

    expected_roles = {"train": "200k", "val": "discret"}
    if existing.get("roles") != expected_roles:
        logger.error("manifest roles are %s, expected %s", existing.get("roles"), expected_roles)
        return 1

    missing = [
        output_path(role, particle)
        for role, particle in [("train", "pi"), ("train", "proton"), ("val", None)]
        if not output_path(role, particle).is_file()
    ]
    if missing:
        logger.error("manifest exists but consolidated file(s) are missing: %s", ", ".join(map(str, missing)))
        return 1

    logger.info("manifest matches corrected shards: %s", expected)
    return 0


def _read_manifest() -> dict | None:
    path = manifest_path()
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.warning("%s is not readable JSON; ignoring it", path)
        return None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--roles",
        nargs="+",
        default=sorted(shards.LOT_ROLES),
        choices=sorted(shards.LOT_ROLES),
        help="which roles to build (default: both; train=200k, val=100k discret)",
    )
    parser.add_argument(
        "--particles",
        nargs="+",
        default=list(paths.PARTICLES_2CLASS),
        help="particles to build (default: the two-class set)",
    )
    checks = parser.add_mutually_exclusive_group()
    checks.add_argument("--check", action="store_true", help="verify the existing files and their manifest")
    checks.add_argument(
        "--check-manifest",
        action="store_true",
        help="quickly verify that the consolidated files came from SDHCAL_SHARDS_DIR",
    )
    args = parser.parse_args(argv)

    unknown = [p for p in args.particles if p not in paths.PARTICLES_2CLASS]
    if unknown:
        raise SystemExit(
            f"unknown particle(s): {', '.join(unknown)}; the sharded lots hold "
            f"{', '.join(paths.PARTICLES_2CLASS)} and no kaon"
        )

    if args.check:
        return check_manifest() or check(args.particles)
    if args.check_manifest:
        return check_manifest()
    return build(args.roles, args.particles)


if __name__ == "__main__":
    raise SystemExit(main())
