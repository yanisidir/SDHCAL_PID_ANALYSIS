#!/usr/bin/env python3
"""Fit the semi-digital calibration on the training lot, evaluate it on the other.

``analysis/4_energy/chi2_tminuit/fit.C`` fits the same model, but it reads
the digitized hits and the generator-level files of the three-class sample, and it
fitted and drew its curves on the same events - which is why every committed
coefficient is flagged ``fitted_in_sample``. The two-class lots make the honest
version possible, because the sample that calibrates and the sample that evaluates
are two independent productions:

* fit on ``train`` (the continuous 1-130 GeV lot);
* evaluate on ``val`` (the 20 fixed energies), which is also the binning of the
  linearity and resolution curves.

Nothing has to be read from the hits: the model depends only on ``N1``, ``N2``,
``N3``, which the consolidated files already carry. And since it is linear in its
nine coefficients, the weighted least squares is solved in closed form by
:func:`sdhcal.calib.fit_wls` rather than searched with TMinuit - same objective,
exact minimum.

The selection matches the chi2 chain: a positive true energy, a non-empty shower,
and by default the containment cut that rejects showers reaching the last active
layer. ``--no-containment`` lifts it; whichever was used is recorded in the
calibration file, because the ML arm has to apply the identical cut for the
comparison to mean anything.

Usage::

    export SDHCAL_PARAMS_SET=2class
    python tools/fit_chi2_2class.py                 # fit, evaluate, write
    python tools/fit_chi2_2class.py --dry-run       # report without writing
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from sdhcal import calib, metrics, paths, rootio, shards

logging.basicConfig(level=logging.INFO, format="%(levelname)8s | %(message)s")
logger = logging.getLogger("fit_chi2_2class")

TREE = "paramsTree"
COLUMNS = ["N1", "N2", "N3", "primaryEnergy", "particlePDG", "touchesLastLayer"]


def select(frame: pd.DataFrame, *, require_contained: bool) -> pd.DataFrame:
    """The chi2 chain's event selection, applied identically to both lots."""
    keep = (frame["primaryEnergy"] > 0.0) & (frame[["N1", "N2", "N3"]].sum(axis=1) > 0)
    if require_contained:
        keep &= frame["touchesLastLayer"] == 0
    return frame.loc[keep].reset_index(drop=True)


def _columns(frame: pd.DataFrame) -> tuple[np.ndarray, ...]:
    return (
        frame["N1"].to_numpy(dtype=float),
        frame["N2"].to_numpy(dtype=float),
        frame["N3"].to_numpy(dtype=float),
        frame["primaryEnergy"].to_numpy(dtype=float),
    )


def assess(frame: pd.DataFrame, values: np.ndarray, label: str) -> dict[str, float]:
    """Objective per event, plus the bias and resolution of the core."""
    n1, n2, n3, e_true = _columns(frame)
    e_reco = calib.ereco(n1, n2, n3, values)
    objective = calib.wls_objective(n1, n2, n3, e_true, values)
    core = metrics.core_bias_resolution(metrics.relative_residual(e_true, e_reco))
    logger.info(
        "  %-22s n=%7d  S/n=%8.4f  bias=%+.4f  resolution=%.4f",
        label,
        len(frame),
        objective / len(frame),
        core.bias,
        core.sigma,
    )
    return {
        "n_events": int(len(frame)),
        "objective_per_event": float(objective / len(frame)),
        "bias": float(core.bias),
        "resolution": float(core.sigma),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--particles", nargs="+", default=list(paths.PARTICLES_2CLASS))
    parser.add_argument(
        "--no-containment",
        action="store_true",
        help="keep showers that reach the last layer (the chi2 chain rejects them)",
    )
    parser.add_argument("--dry-run", action="store_true", help="report without writing the coefficients")
    args = parser.parse_args(argv)

    if paths.params_set() != "2class":
        raise SystemExit(
            "this fit expects the two-class lots: export SDHCAL_PARAMS_SET=2class "
            "(the three-class chain is analysis/4_energy/chi2_tminuit/fit.C)"
        )
    require_contained = not args.no_containment

    validation = rootio.load_trees(paths.dataset("holdout"), TREE, COLUMNS)
    entries: dict[str, calib.Coefficients] = {}

    for particle in args.particles:
        pdg = paths.PARTICLE_PDG[particle]
        train = select(
            rootio.load_trees(paths.dataset("params", particle), TREE, COLUMNS),
            require_contained=require_contained,
        )
        val = select(
            validation.loc[validation["particlePDG"] == pdg], require_contained=require_contained
        )
        if train.empty or val.empty:
            logger.error("%s: empty sample after selection (train=%d, val=%d)", particle, len(train), len(val))
            return 1

        logger.info("%s: fitting on %d events of the %s lot", particle, len(train), shards.LOT_ROLES["train"])
        values = calib.fit_wls(*_columns(train))

        on_train = assess(train, values, "fit sample (train)")
        on_val = assess(val, values, "evaluation (val)")

        # What the honest fit is worth against the three-class coefficients already
        # in the repository, measured on the same events. Those were fitted on
        # every event of their own sample, so this is the comparison the
        # ``fitted_in_sample`` flag was introduced to make possible.
        reference_path = paths.repo_root() / "calibration/chi2_coefficients.json"
        try:
            reference = calib.load(particle, reference_path)
        except (FileNotFoundError, KeyError):
            reference = None
        if reference is not None:
            committed = assess(val, reference.values, "three-class committed")
            logger.info(
                "  -> resolution %+.4f, bias %+.4f against %s (fitted %s)",
                on_val["resolution"] - committed["resolution"],
                abs(on_val["bias"]) - abs(committed["bias"]),
                reference_path.name,
                "in-sample" if reference.fitted_in_sample else "out-of-sample",
            )

        entries[particle] = calib.Coefficients(
            particle=particle,
            values=values,
            metadata={
                "fitted_on_train_split": True,
                "fitted_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "fitted_by": "tools/fit_chi2_2class.py",
                "method": "closed-form weighted least squares, weight 1/Etrue",
                "fit_lot": shards.LOT_ROLES["train"],
                "eval_lot": shards.LOT_ROLES["val"],
                "selection": (
                    "Etrue > 0, N1+N2+N3 > 0"
                    + (f", no hit in layer {shards.LAST_LAYER}" if require_contained else "")
                ),
                "require_contained": bool(require_contained),
                "n_fit_events": on_train["n_events"],
                "objective_per_event_fit": on_train["objective_per_event"],
                "objective_per_event_eval": on_val["objective_per_event"],
                "bias_eval": on_val["bias"],
                "resolution_eval": on_val["resolution"],
            },
        )

    if args.dry_run:
        logger.info("--dry-run: nothing written")
        return 0

    path = calib.save(entries)
    logger.info("wrote %s", path)
    for particle, coeff in calib.load_all().items():
        logger.info("  %-7s in_sample=%s", particle, coeff.fitted_in_sample)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
