"""Reading the shower-parameter ROOT trees.

Factored out of the five near-identical copies of ``load_root`` / ``clean_df``
that had drifted apart. One of those copies iterated over a ``str`` instead of a
list of paths, so it looped over the characters of the filename, found no file
and raised ``RuntimeError: No valid ROOT files found`` - the three MLP energy
scripts could never load anything.

``uproot`` is imported lazily so that the rest of :mod:`sdhcal` stays importable
(and testable) without it.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["load_trees", "clean", "load_and_clean"]

PathLike = str | Path


def _as_path_list(files: PathLike | Iterable[PathLike]) -> list[Path]:
    """Accept a single path or an iterable of paths; never iterate a string."""
    if isinstance(files, (str, Path)):
        return [Path(files)]
    return [Path(f) for f in files]


def load_trees(
    files: PathLike | Iterable[PathLike],
    tree: str,
    columns: Sequence[str],
    *,
    require_all: bool = True,
) -> pd.DataFrame:
    """Read ``columns`` from ``tree`` in each file and concatenate the results.

    Parameters
    ----------
    require_all:
        When true, a missing file or a missing branch is an error. The original
        code only logged a warning and carried on, so a typo in a path silently
        shrank the training sample.
    """
    import uproot  # imported here to keep sdhcal importable without uproot

    paths = _as_path_list(files)
    if not paths:
        raise ValueError("no input file given")

    frames = []
    for path in paths:
        if not path.is_file():
            message = f"missing ROOT file: {path}"
            if require_all:
                raise FileNotFoundError(message)
            logger.warning("%s (skipped)", message)
            continue
        with uproot.open(path) as handle:
            if tree not in handle:
                raise KeyError(f"tree {tree!r} not found in {path}")
            available = set(handle[tree].keys())
            missing = [c for c in columns if c not in available]
            if missing:
                raise KeyError(
                    f"{path.name}: branch(es) not found: {', '.join(missing)}. "
                    f"Available branches: {', '.join(sorted(available))}"
                )
            frames.append(handle[tree].arrays(list(columns), library="pd"))
        logger.info("Read %-28s %8d events", path.name, len(frames[-1]))

    if not frames:
        raise RuntimeError(f"no readable ROOT file among: {', '.join(map(str, paths))}")

    df = pd.concat(frames, ignore_index=True)
    logger.info("Total events loaded: %d", len(df))
    return df


def clean(df: pd.DataFrame, *, columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Replace +-inf by NA and drop incomplete rows.

    ``columns`` restricts the check to the columns that matter, so an unused
    branch full of NaN cannot empty the sample.
    """
    before = len(df)
    subset = list(columns) if columns is not None else None
    out = df.replace([np.inf, -np.inf], pd.NA).dropna(subset=subset).reset_index(drop=True)
    removed = before - len(out)
    if removed:
        logger.info(
            "Cleaning removed %d of %d events (%.3f %%)", removed, before, 100.0 * removed / max(before, 1)
        )
    if out.empty:
        raise RuntimeError("no event left after cleaning; check the branch selection")
    return out


def load_and_clean(
    files: PathLike | Iterable[PathLike],
    tree: str,
    columns: Sequence[str],
    *,
    require_all: bool = True,
) -> pd.DataFrame:
    """:func:`load_trees` followed by :func:`clean` on the same columns."""
    return clean(load_trees(files, tree, columns, require_all=require_all), columns=columns)
