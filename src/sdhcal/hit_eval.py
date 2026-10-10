"""External evaluation of the hit-level PID networks.

The convolutional and graph studies split one hit CSV into their own
train/validation/test parts. That is right for choosing when to stop training,
but it means their test figures come from the same production they trained on,
while the shower-variable models are trained on the continuous 200k lot and
measured on the independent fixed-energy lot. Comparing the two as they stand
would compare protocols, not models.

So the networks keep their internal split and gain one more step: after training,
the best checkpoint is applied to a second, external CSV - the fixed-energy lot -
and those predictions are written where :mod:`sdhcal.benchmark` can read them.

Two conventions are fixed here rather than in each script.

*The score is the pion probability.* The networks are trained with the pion as
class 0 and the proton as class 1, so their natural output is the proton
probability; the benchmark quotes pion efficiency against proton rejection, so
the score is flipped once, here, instead of in each caller.

*The true energy travels with the prediction.* It is per event, not per hit, and
``hit_pid`` drops it because it is not a hit feature - but the comparison breaks
efficiency down by energy, so it is read back from the CSV by event.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["event_energies", "save_predictions"]

ENERGY_COLUMN = "primaryEnergy"


def event_energies(frame: pd.DataFrame) -> dict[str, float]:
    """True energy of each event, keyed by ``event_id``.

    Raises if an event carries more than one energy: that would mean hits from
    two events were merged under one identifier, which is exactly what the
    ``shardIndex`` column exists to prevent.
    """
    if ENERGY_COLUMN not in frame.columns:
        raise KeyError(
            f"the hit table has no {ENERGY_COLUMN!r} column, so the comparison could not break "
            "efficiency down by energy. Re-export it with tools/export_hits.py."
        )
    grouped = frame.groupby("event_id")[ENERGY_COLUMN]
    spread = grouped.nunique()
    if (spread > 1).any():
        offenders = spread[spread > 1].index.tolist()[:5]
        raise ValueError(f"{int((spread > 1).sum())} event(s) carry several energies, e.g. {offenders}")
    return grouped.first().to_dict()


def save_predictions(
    path: Path,
    *,
    event_ids: list[str],
    probability_proton: np.ndarray,
    energies: dict[str, float],
    labels: np.ndarray | None = None,
) -> Path:
    """Write one external prediction per event, for ``tools/import_predictions.py``.

    ``event_ids`` must be in the order the model produced ``probability_proton``;
    they are what ties a score back to an event, and the networks shuffle only
    their training loader, never the evaluation one.
    """
    probability_proton = np.asarray(probability_proton, dtype=float)
    if len(event_ids) != len(probability_proton):
        raise ValueError(
            f"{len(event_ids)} event ids for {len(probability_proton)} scores: the evaluation "
            "loader must not shuffle, or the two fall out of step"
        )
    missing = [e for e in event_ids if e not in energies]
    if missing:
        raise KeyError(f"{len(missing)} event(s) have no energy, e.g. {missing[:5]}")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        # The benchmark quotes pion efficiency, and the networks score the proton.
        "y_score": 1.0 - probability_proton,
        "event_id": np.asarray(event_ids),
        "primary_energy": np.asarray([energies[e] for e in event_ids], dtype=float),
    }
    if labels is not None:
        payload["label"] = np.asarray(labels)
    np.savez_compressed(path, **payload)
    return path
