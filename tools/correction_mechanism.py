"""Why does the cluster network lower the stochastic term?

Chapter 7 attributes the gain to compensating, event by event, the share between
the electromagnetic and hadronic components of the shower. That is a mechanism,
not a measurement: nothing in the comparison establishes it.

The simulation carries the quantities the hypothesis is about -- ``emFraction``
and ``depositedEnergyNeutrons`` -- so it can be tested directly on the published
predictions, with no retraining.

Two corrections are compared against them:

``C_ideal = E_true / E_chi2``
    what the correction *should* have been, event by event. Its correlation with
    ``emFraction`` bounds the hypothesis: if the ideal correction does not track
    the electromagnetic fraction, no model can gain by following it, and the
    explanation in the report is wrong whatever the network learned.

``C_theta = E_pred / E_chi2``
    what the network actually learned. Its correlation says how much of that
    available structure was captured.

Energies are handled in fixed bins throughout. Pooling them would be meaningless
here: the neutron count correlates with the beam energy at +0.75, so any
correlation measured over the full range is mostly the energy itself.

Usage::

    python tools/correction_mechanism.py
    python tools/correction_mechanism.py --model gnn_ratio --out-dir output/mechanism
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import uproot

from sdhcal import paths, shards

logger = logging.getLogger("correction_mechanism")

#: Per-event truth the hypothesis is about, read from the hit shards.
TRUTH_BRANCHES = (
    "eventNumber",
    "emFraction",
    "depositedEnergy",
    "depositedEnergyNeutrons",
    "nNeutrons",
    "leakedEnergy",
)

#: Lot holding the evaluation sample; the predictions were made on it.
VAL_SAMPLE = "discret"


def hit_shard_files(sample: str, particle: str) -> list[Path]:
    """Every hit shard of one lot, ordered by :func:`shards.shard_id`.

    The truth branches this module needs live only in the hit shards: the
    parameter shards hold the twenty-four shower observables and nothing else,
    so ``shards.shard_files`` would find the wrong files.
    """
    lot = paths.SHARD_SAMPLES[sample][particle]
    directory = paths.hit_shards_dir() / lot
    if not directory.is_dir():
        raise FileNotFoundError(
            f"missing hit shard directory for {particle!r} of sample {sample!r}: {directory}\n"
            "Set SDHCAL_HIT_SHARDS_DIR to the directory holding the digitised lots."
        )
    found = sorted(directory.glob("*_digitized.root"), key=lambda p: hit_shard_id(sample, p))
    if not found:
        raise FileNotFoundError(f"no shard matching '*_digitized.root' in {directory}")
    return found


def hit_shard_id(sample: str, path: Path) -> int:
    """Shard id of a hit file, through the library's own mapping.

    ``shards.shard_id`` reads the parameter shards, whose names end in
    ``_digitized_params.root``; a hit shard is the same name without ``_params``.
    Restoring the suffix keeps one definition of the key instead of a second
    regex here that could drift away from it.
    """
    return shards.shard_id(sample, path.name.replace("_digitized.root", "_digitized_params.root"))


def read_truth(sample: str = VAL_SAMPLE) -> pd.DataFrame:
    """Per-event Monte-Carlo truth, keyed like the benchmark tables."""
    directory = paths.hit_shards_dir()
    frames = []
    for particle in paths.PARTICLES_2CLASS:
        for path in hit_shard_files(sample, particle):
            identifier = hit_shard_id(sample, path)
            with uproot.open(path) as handle:
                tree = handle[next(k for k in handle if "tree" in k.lower())]
                present = set(tree.keys())
                available = [b for b in TRUTH_BRANCHES if b in present]
                missing = set(TRUTH_BRANCHES) - present
                if missing:
                    raise KeyError(f"{path.name}: missing branch(es) {', '.join(sorted(missing))}")
                block = tree.arrays(available, library="pd")
            block["eventNumber"] = shards.global_event_number(identifier, block["eventNumber"].to_numpy())
            frames.append(block)
    if not frames:
        raise FileNotFoundError(f"no shard found for sample {sample!r} under {directory}")
    truth = pd.concat(frames, ignore_index=True)
    logger.info("truth: %d events from %s", len(truth), directory)
    return truth


def corrections(model: str, baseline: str = "chi2") -> pd.DataFrame:
    """``C_theta`` and ``C_ideal`` on the events both estimators predicted."""
    directory = paths.output_dir() / "benchmark" / "energy"
    learned = pd.read_csv(directory / f"{model}.csv")
    reference = pd.read_csv(directory / f"{baseline}.csv")
    merged = learned.merge(
        reference[["eventNumber", "energyPred"]],
        on="eventNumber",
        suffixes=("_model", "_baseline"),
    )
    if merged.empty:
        raise SystemExit(f"no event shared between {model} and {baseline}")
    merged["C_theta"] = merged["energyPred_model"] / merged["energyPred_baseline"]
    merged["C_ideal"] = merged["primaryEnergy"] / merged["energyPred_baseline"]
    logger.info("corrections: %d events shared by %s and %s", len(merged), model, baseline)
    return merged


def correlate(frame: pd.DataFrame, pairs: list[tuple[str, str]], *, min_entries: int = 200) -> pd.DataFrame:
    """Pearson correlation of each pair, within one energy point and species."""
    rows = []
    for (energy, pdg), block in frame.groupby(["primaryEnergy", "particlePDG"], sort=True):
        if len(block) < min_entries:
            continue
        row = {
            "energy": float(energy),
            "species": paths.PDG_TO_PARTICLE.get(int(pdg), str(pdg)),
            "n": len(block),
        }
        for left, right in pairs:
            a, b = block[left].to_numpy(), block[right].to_numpy()
            keep = np.isfinite(a) & np.isfinite(b)
            # A degenerate column has no correlation to report, not a zero one.
            if keep.sum() < min_entries or a[keep].std() == 0 or b[keep].std() == 0:
                row[f"{left}~{right}"] = np.nan
                continue
            row[f"{left}~{right}"] = float(np.corrcoef(a[keep], b[keep])[0, 1])
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gnn_ratio", help="learned estimator to test")
    parser.add_argument("--baseline", default="chi2", help="estimator the correction is relative to")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)8s | %(message)s")

    frame = corrections(args.model, args.baseline).merge(read_truth(), on="eventNumber", how="inner")
    logger.info("joined: %d events carry both a prediction and the truth", len(frame))
    if frame.empty:
        raise SystemExit("no event joined; the prediction keys and the shard keys disagree")

    # The neutron deposit is a tiny share of the visible energy, so the fraction
    # is what compares across energies, not the absolute value.
    frame["neutronFraction"] = frame["depositedEnergyNeutrons"] / frame["depositedEnergy"].replace(0, np.nan)

    pairs = [
        ("C_ideal", "emFraction"),
        ("C_theta", "emFraction"),
        ("C_ideal", "neutronFraction"),
        ("C_theta", "neutronFraction"),
        ("C_theta", "C_ideal"),
    ]
    table = correlate(frame, pairs)

    out = args.out_dir or (paths.output_dir() / "mechanism")
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / f"correlations_{args.model}.csv", index=False)
    frame.to_csv(out / f"events_{args.model}.csv", index=False)

    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(table.round(3).to_string(index=False))
    print()
    for species, block in table.groupby("species"):
        print(f"{species}: mean over energy points")
        for left, right in pairs:
            column = f"{left}~{right}"
            print(f"    {column:28} {block[column].mean():+.3f}")
    logger.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
