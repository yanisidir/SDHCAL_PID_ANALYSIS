"""Seed-to-seed spread of the cluster network, and what it does to the ablation.

Every network number in this study rests on one training run. The only measured
width is the BDT's, 0.0008 on sigma/E, and a neural net usually varies more than
a boosted tree. Without this number two things cannot be read: whether the
0.0027 per point that the timing ablation shows above 35 GeV is a result, and
whether a sigma_t scan separating configurations by 0.002 to 0.017 would mean
anything at all.

Reads the validation predictions each run writes, recomputes the resolution and
the two fit terms per seed, and reports the sample spread. With ``--ablation`` it
then expresses the measured timing difference in units of that spread, which is
the question the seeds were run to answer.

Usage::

    python tools/seed_spread.py
    python tools/seed_spread.py --ablation
"""

from __future__ import annotations

import argparse
import glob
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from sdhcal import paths
from sdhcal.benchmark import fit_resolution
from sdhcal.metrics import core_bias_resolution

logger = logging.getLogger("seed_spread")

#: Runs to collect: the published seed plus whatever the sweep produced.
DEFAULT_PATTERNS = ("energy_reco_gnn_ratio", "energy_reco_gnn_seed*")


def predictions(run_dir: Path) -> pd.DataFrame:
    """True and predicted energy of one run, from its validation arrays."""
    matches = sorted(run_dir.glob("**/arrays/validation_predictions_*.npz"))
    if not matches:
        raise FileNotFoundError(f"no validation predictions under {run_dir}")
    payload = np.load(max(matches, key=lambda p: p.stat().st_mtime), allow_pickle=True)
    missing = {"y_true", "y_pred"} - set(payload.files)
    if missing:
        raise KeyError(f"{run_dir.name}: the arrays lack {', '.join(sorted(missing))}")
    return pd.DataFrame({"true": payload["y_true"], "pred": payload["y_pred"]})


def summarise(frame: pd.DataFrame, *, min_entries: int = 500) -> dict[str, float]:
    """Global resolution and the two terms of the a + b fit, for one run."""
    residual = ((frame["pred"] - frame["true"]) / frame["true"]).to_numpy()
    overall = core_bias_resolution(residual)

    energies, widths, errors = [], [], []
    truth = frame["true"].to_numpy()
    for point in sorted(set(np.round(truth, 1))):
        mask = np.abs(truth - point) < 0.6
        if mask.sum() < min_entries:
            continue
        estimate = core_bias_resolution(residual[mask])
        if not np.isfinite(estimate.sigma):
            continue
        energies.append(point)
        widths.append(estimate.sigma)
        errors.append(estimate.sigma_err)

    fit = fit_resolution(np.array(energies), np.array(widths), np.array(errors))
    return {
        "resolution": overall.sigma,
        "bias": overall.bias,
        "a": fit.stochastic,
        "b": fit.constant,
        "n_points": len(energies),
    }


def seed_of(run_dir: Path) -> str:
    match = re.search(r"seed(\d+)", run_dir.name)
    return match.group(1) if match else "42 (published)"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patterns", nargs="+", default=list(DEFAULT_PATTERNS))
    parser.add_argument("--ablation", action="store_true", help="read the timing difference against the spread")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)8s | %(message)s")

    output = paths.output_dir()
    directories: list[Path] = []
    for pattern in args.patterns:
        directories.extend(Path(p) for p in sorted(glob.glob(str(output / pattern))))
    if not directories:
        raise SystemExit(f"no run directory matched {args.patterns} under {output}")

    rows = []
    for directory in directories:
        try:
            rows.append({"seed": seed_of(directory), **summarise(predictions(directory))})
        except (FileNotFoundError, KeyError) as error:
            logger.warning("%s skipped: %s", directory.name, error)
    if not rows:
        raise SystemExit("no run carried usable validation predictions")

    table = pd.DataFrame(rows).sort_values("seed")
    print(table.round(5).to_string(index=False))

    if len(table) < 2:
        print("\nOne run only: no spread can be measured yet.")
        return 0

    print(f"\n{'quantity':12} {'mean':>10} {'spread':>10}   (sample standard deviation over seeds)")
    spreads = {}
    for column in ("resolution", "a", "b"):
        mean, spread = table[column].mean(), table[column].std(ddof=1)
        spreads[column] = spread
        print(f"{column:12} {mean:10.5f} {spread:10.5f}")

    if args.ablation:
        # The published difference, taken from the ablation table so the two
        # analyses cannot drift apart.
        path = output / "benchmark" / "ablation" / "tables" / "timing.csv"
        if not path.is_file():
            logger.warning("no ablation table at %s; run tools/timing_ablation.py first", path)
            return 0
        ablation = pd.read_csv(path)
        high = ablation[ablation["energy"] >= 35.0]
        low = ablation[ablation["energy"] <= 15.0]
        # Two runs differ by sqrt(2) times the per-run spread.
        pair = spreads["resolution"] * np.sqrt(2.0)
        print(f"\nper-run spread on sigma/E : {spreads['resolution']:.5f}")
        print(f"expected spread of a difference between two runs : {pair:.5f}")
        for label, block in (("below 15 GeV", low), ("above 35 GeV", high)):
            mean_difference = float(block["difference"].mean())
            line = f"  {label:14} mean difference {mean_difference:+.5f}"
            if pair > 0:
                line += f"  = {abs(mean_difference) / pair:.1f} x that spread"
            print(line)
        print("\nA difference below about one spread is not separable from seed noise.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
