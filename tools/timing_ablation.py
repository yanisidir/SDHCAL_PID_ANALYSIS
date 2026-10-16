"""Resolution of the timed and untimed networks, point by point in energy.

The global resolutions differ by little, which reads as nothing. Resolved in
energy the picture is clear: the time helps below about 25 GeV and does nothing
above. That structure is the result, and it is invisible in any single number --
the same objection the report raises against quoting one sigma/E.

The timed side is averaged over seeds, because the training variance of this
network (0.00195 on sigma/E) is comparable to the differences being measured. A
single timed run against a single untimed run cannot separate the two.

Writes one row per energy point with both resolutions, their difference and its
error, for the ROOT macro to draw.

Usage::

    python tools/timing_ablation.py
    python tools/timing_ablation.py --timed gnn_ratio --untimed gnn_notime
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from sdhcal import paths
from sdhcal.metrics import core_bias_resolution

logger = logging.getLogger("timing_ablation")


def per_energy(frame: pd.DataFrame, *, min_entries: int = 500) -> pd.DataFrame:
    """Core resolution in each fixed-energy slice."""
    rows = []
    residual = ((frame["energyPred"] - frame["primaryEnergy"]) / frame["primaryEnergy"]).to_numpy()
    energy = frame["primaryEnergy"].to_numpy()
    for point in sorted(set(np.round(energy, 1))):
        mask = np.abs(energy - point) < 0.6
        if mask.sum() < min_entries:
            continue
        estimate = core_bias_resolution(residual[mask])
        rows.append(
            {
                "energy": float(point),
                "resolution": estimate.sigma,
                "resolution_err": estimate.sigma_err,
                "n": int(estimate.n_used),
            }
        )
    return pd.DataFrame(rows)


def from_run_dir(run_dir: Path) -> pd.DataFrame:
    """Per-energy resolution of one training run, read from its own arrays."""
    matches = sorted(run_dir.glob("**/arrays/validation_predictions_*.npz"))
    if not matches:
        raise FileNotFoundError(f"no validation predictions under {run_dir}")
    payload = np.load(max(matches, key=lambda p: p.stat().st_mtime), allow_pickle=True)
    frame = pd.DataFrame({"primaryEnergy": payload["y_true"], "energyPred": payload["y_pred"]})
    return per_energy(frame)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--timed-runs",
        nargs="+",
        default=["energy_reco_gnn_ratio", "energy_reco_gnn_seed43", "energy_reco_gnn_seed44", "energy_reco_gnn_seed45"],
        help="run directories under the output root that read the hit times; several means the "
        "comparison is made against their mean, with the seed spread as the error",
    )
    parser.add_argument("--untimed-run", default="energy_reco_gnn_notime", help="run with the time blanked")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)8s | %(message)s")

    root = paths.output_dir()
    timed = []
    for name in args.timed_runs:
        try:
            timed.append(from_run_dir(root / name))
        except FileNotFoundError as error:
            logger.warning("%s skipped: %s", name, error)
    if not timed:
        raise SystemExit("no timed run carried usable predictions")
    untimed = from_run_dir(root / args.untimed_run)
    logger.info("%d timed run(s) against one untimed run", len(timed))

    # The mean over seeds is what the untimed run must be compared with: a single
    # draw of the timed configuration carries the training variance that the seed
    # sweep exists to measure.
    wide = timed[0][["energy"]].copy()
    stack = pd.concat([t.set_index("energy")["resolution"] for t in timed], axis=1)
    wide["resolution_timed"] = stack.mean(axis=1).to_numpy()
    wide["spread_timed"] = stack.std(axis=1, ddof=1).to_numpy() if len(timed) > 1 else np.nan
    wide["n_seeds"] = len(timed)
    merged = wide.merge(untimed.rename(columns={"resolution": "resolution_untimed"})[
        ["energy", "resolution_untimed", "resolution_err"]], on="energy")
    merged = merged.rename(columns={"resolution_err": "resolution_err_untimed"})
    if merged.empty:
        raise SystemExit("the runs share no energy point")
    merged["difference"] = merged["resolution_timed"] - merged["resolution_untimed"]
    if len(timed) > 1:
        # Mean of n draws against a single draw: the spread of the difference is
        # sqrt(1/n + 1) times the per-run spread.
        merged["difference_err"] = merged["spread_timed"] * np.sqrt(1.0 / len(timed) + 1.0)
    else:
        merged["difference_err"] = merged["resolution_err_untimed"]

    # The weighted mean says whether anything survives averaging over energy; the
    # sign pattern says whether what is left is structured or scattered.
    weights = 1.0 / merged["difference_err"] ** 2
    mean = float((merged["difference"] * weights).sum() / weights.sum())
    mean_err = float(1.0 / np.sqrt(weights.sum()))
    negative = int((merged["difference"] < 0).sum())

    out = args.out_dir or (paths.output_dir() / "benchmark" / "ablation" / "tables")
    out.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out / "timing.csv", index=False)

    with pd.option_context("display.width", 200):
        print(merged.round(4).to_string(index=False))
    print(f"\nweighted mean difference : {mean:+.4f} +- {mean_err:.4f}")
    print(f"points where timing wins : {negative}/{len(merged)}")
    logger.info("wrote %s", out / "timing.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
