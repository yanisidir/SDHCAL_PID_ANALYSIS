"""Detector time smearing, applied once, with independent noise per variable.

Three bugs in the original scripts all lived in this one operation, and all three
mattered because the timing variables are exactly what the study is about:

1. ``smear_column_gaussian`` re-seeded a fresh generator from the *same* seed on
   every call, so ``tMin``, ``tMax`` and ``tSpread`` all received the **identical**
   noise vector: the smearing was 100 % correlated across variables instead of
   independent.
2. The inference scripts called the smearing twice with overlapping column lists,
   so ``tMin`` ended up with twice the noise actually used at training time.
3. One script used the global ``np.random`` without a seed, so its runs were not
   reproducible.

:func:`smear_time_columns` fixes all three: one generator per (seed, column) pair
derived from a stable hash of the column name, and a marker in ``df.attrs`` that
makes a second application of the same column a hard error.
"""

from __future__ import annotations

import logging
import zlib
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["TimeSmearing", "SmearingAlreadyApplied", "smear_time_columns", "DEFAULT_TIME_COLUMNS"]

#: Time-like shower variables produced by ShowerAnalyzer, in nanoseconds.
DEFAULT_TIME_COLUMNS: tuple[str, ...] = ("tMin", "tMax", "tMean", "tSpread")

#: Key under which the applied smearing is recorded in ``DataFrame.attrs``.
_ATTR_KEY = "sdhcal_time_smearing"


class SmearingAlreadyApplied(RuntimeError):
    """Raised when a column would be smeared a second time within one pipeline."""


@dataclass(frozen=True)
class TimeSmearing:
    """Configuration of the Gaussian time smearing.

    Parameters
    ----------
    sigma_ns:
        Standard deviation of the added Gaussian noise, in nanoseconds. ``None``
        disables the smearing entirely, which is how an "ideal timing" run is
        requested.
    columns:
        Columns to smear. Only columns actually present in the frame are touched;
        a missing one is logged and skipped.
    seed:
        Base seed. The generator actually used for a column is derived from
        ``(seed, crc32(column))``, so each column gets independent noise while the
        result stays reproducible across machines and Python versions.
    """

    sigma_ns: float | None = 0.1
    columns: Sequence[str] = field(default=DEFAULT_TIME_COLUMNS)
    seed: int = 42

    def as_dict(self) -> dict[str, object]:
        """Flat description for the run log."""
        return {
            "smear_sigma_ns": "none" if self.sigma_ns is None else self.sigma_ns,
            "smear_columns": ";".join(self.columns),
            "smear_seed": self.seed,
        }


def _column_rng(seed: int, column: str) -> np.random.Generator:
    """Return a generator that is unique per (seed, column) and stable across runs.

    ``zlib.crc32`` is used rather than :func:`hash` because the built-in hash of a
    string is salted per interpreter process.
    """
    return np.random.default_rng([seed, zlib.crc32(column.encode("utf-8"))])


def smear_time_columns(
    df: pd.DataFrame,
    config: TimeSmearing,
    *,
    copy: bool = True,
) -> pd.DataFrame:
    """Add independent Gaussian noise to the configured time columns.

    Returns the frame (a copy by default). Raises :class:`SmearingAlreadyApplied`
    if any requested column was already smeared on this frame, which is what used
    to silently double the noise on ``tMin``.
    """
    out = df.copy() if copy else df

    if config.sigma_ns is None:
        logger.info("Time smearing disabled (sigma_ns=None)")
        out.attrs[_ATTR_KEY] = {}
        return out

    if config.sigma_ns < 0:
        raise ValueError(f"sigma_ns must be non-negative, got {config.sigma_ns}")

    already: dict[str, float] = dict(out.attrs.get(_ATTR_KEY, {}))
    requested = list(config.columns)
    clash = sorted(set(requested) & set(already))
    if clash:
        raise SmearingAlreadyApplied(
            "columns already smeared on this DataFrame: "
            + ", ".join(f"{c} (sigma={already[c]})" for c in clash)
            + ". Smear once, at load time, with a single TimeSmearing config."
        )

    n = len(out)
    applied = []
    for column in requested:
        if column not in out.columns:
            logger.warning("Column %s absent, smearing skipped", column)
            continue
        rng = _column_rng(config.seed, column)
        noise = rng.normal(0.0, config.sigma_ns, size=n)
        out[column] = out[column].to_numpy(dtype=float) + noise
        already[column] = config.sigma_ns
        applied.append(column)

    out.attrs[_ATTR_KEY] = already
    logger.info(
        "Applied time smearing sigma=%.4g ns (independent per column) to: %s",
        config.sigma_ns,
        ", ".join(applied) if applied else "nothing",
    )
    return out


def smeared_columns(df: pd.DataFrame) -> dict[str, float]:
    """Return the ``{column: sigma}`` smearing already recorded on a frame."""
    return dict(df.attrs.get(_ATTR_KEY, {}))
