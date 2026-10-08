#!/usr/bin/env python3
"""Merge the per-particle chi2 fit fragments into the shared calibration file.

``EnergyReco.C`` writes ``calibration/fit_<particle>.json`` after each fit. This
merges those fragments into ``calibration/chi2_coefficients.json``, the single
file that :mod:`sdhcal.calib` reads, so the coefficients are never retyped into
a Python script again.

Usage::

    python tools/update_calibration.py                 # merge every fragment
    python tools/update_calibration.py --particles pi   # only one
    python tools/update_calibration.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from sdhcal import calib, paths

logging.basicConfig(level=logging.INFO, format="%(levelname)8s | %(message)s")
logger = logging.getLogger("update_calibration")


def fragment_paths(particles: list[str]) -> dict[str, Path]:
    directory = paths.repo_root() / "calibration"
    found = {}
    for particle in particles:
        path = directory / f"fit_{particle}.json"
        if path.is_file():
            found[particle] = path
        else:
            logger.warning("no fragment for %s (%s)", particle, path.name)
    return found


def read_fragment(path: Path, particle: str) -> calib.Coefficients:
    blob = json.loads(path.read_text(encoding="utf-8"))
    missing = [name for name in calib.PARAM_NAMES if name not in blob]
    if missing:
        raise KeyError(f"{path.name} lacks coefficient(s): {', '.join(missing)}")

    values = np.array([float(blob[name]) for name in calib.PARAM_NAMES], dtype=float)
    errors = None
    if isinstance(blob.get("errors"), dict):
        errors = np.array([float(blob["errors"][name]) for name in calib.PARAM_NAMES], dtype=float)

    metadata = {
        k: v for k, v in blob.items() if k not in set(calib.PARAM_NAMES) | {"errors", "particle"}
    }
    if not metadata.get("fitted_on_train_split"):
        logger.warning(
            "%s is not flagged as fitted on a training split; its linearity and resolution "
            "cannot be compared with the held-out machine-learning numbers",
            particle,
        )
    if metadata.get("migrad_status", 0) != 0 or metadata.get("covariance_status", 3) < 3:
        logger.warning(
            "%s: the fit reported migrad_status=%s, covariance_status=%s - check convergence "
            "before using these coefficients",
            particle,
            metadata.get("migrad_status"),
            metadata.get("covariance_status"),
        )
    return calib.Coefficients(particle, values, errors, metadata)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--particles", nargs="+", default=list(paths.PARTICLES))
    parser.add_argument("--dry-run", action="store_true", help="show what would change")
    args = parser.parse_args(argv)

    fragments = fragment_paths(args.particles)
    if not fragments:
        logger.error(
            "no fragment found. Produce one first:\n"
            "    root -l -b -q 'energy_reconstruction_Tminuit/EnergyReco.C(\"pi\")'"
        )
        return 1

    entries = {particle: read_fragment(path, particle) for particle, path in fragments.items()}

    previous = {}
    try:
        previous = calib.load_all()
    except FileNotFoundError:
        logger.info("creating %s", calib.coefficients_path())

    for particle, coefficients in entries.items():
        if particle in previous:
            delta = np.abs(coefficients.values - previous[particle].values)
            scale = np.maximum(np.abs(previous[particle].values), 1e-30)
            logger.info(
                "%-7s largest relative change against the stored value: %.3g",
                particle,
                float(np.max(delta / scale)),
            )
        else:
            logger.info("%-7s new entry", particle)

    if args.dry_run:
        logger.info("dry run: nothing written")
        return 0

    written = calib.save(entries)
    logger.info("updated %s with: %s", written, ", ".join(sorted(entries)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
