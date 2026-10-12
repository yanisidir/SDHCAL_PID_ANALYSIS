"""Response and resolution against energy, per coupling scenario and species.

The chapter-8 figures were drawn by an archived matplotlib script whose inputs
no longer exist in the shape it expected. This rebuilds them from the per-event
tables the coupling study writes, so the curves and the numbers in the text come
from the same run, and hands the result to the ROOT macro that draws every other
benchmark figure.

One row per (scenario, species, energy point). Species "all" is the mixture,
which is what the resolution quoted in the text refers to.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from sdhcal import paths
from sdhcal.metrics import core_bias_resolution

logger = logging.getLogger("coupling_curves")

#: The scenarios the report compares, in the order the legends should show them.
SCENARIOS = (
    "pid-none_reco-chi2",
    "pid-bdt_reco-chi2",
    "pid-true_reco-chi2",
    "pid-none_reco-bdt",
    "pid-bdt_reco-bdt",
    "pid-true_reco-bdt",
)


def latest_table(directory: Path) -> Path:
    """The highest-numbered event table, which is the newest run of a scenario."""
    tables = sorted((directory / "tables").glob("events_*.csv"))
    if not tables:
        raise FileNotFoundError(f"no event table in {directory / 'tables'}")
    return max(tables, key=lambda p: int(p.stem.rsplit("_", 1)[-1]))


def curve(frame: pd.DataFrame, *, method: str) -> pd.DataFrame:
    """Bias, response and resolution per true-energy point."""
    rows = []
    for energy, block in frame.groupby("E_true_GeV", sort=True):
        residual = block["dE_over_E"].to_numpy()
        estimate = core_bias_resolution(residual, method=method)
        if not np.isfinite(estimate.sigma):
            continue
        rows.append(
            {
                "energy": float(energy),
                "response": 1.0 + estimate.bias,
                "response_err": estimate.bias_err,
                "resolution": estimate.sigma,
                "resolution_err": estimate.sigma_err,
                "n": int(estimate.n_used),
            }
        )
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", default="truncated_rms", help="core estimator")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)8s | %(message)s")
    root = paths.output_dir() / "pid_energy_coupling"
    out = args.out_dir or (paths.output_dir() / "benchmark" / "coupling" / "tables")
    out.mkdir(parents=True, exist_ok=True)

    written = 0
    for scenario in SCENARIOS:
        directory = root / scenario
        if not directory.is_dir():
            logger.warning("%-22s absent, skipped", scenario)
            continue
        frame = pd.read_csv(latest_table(directory))
        for species, block in (("all", frame), *frame.groupby("label_true", sort=True)):
            table = curve(block, method=args.method)
            if table.empty:
                continue
            path = out / f"{scenario}__{species}.csv"
            table.to_csv(path, index=False)
            written += 1
        logger.info("%-22s %6d events, %2d energy points", scenario, len(frame), len(curve(frame, method=args.method)))

    logger.info("wrote %d curve table(s) to %s", written, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
