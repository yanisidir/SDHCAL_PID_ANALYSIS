"""Reproducible splits, and the checks that make a held-out sample trustworthy.

Two problems are addressed here.

*The held-out sample was unverified.* The PID/energy coupling study ran inference
on ``val_set.root`` while the classifier and the regressors had been trained on
the ``130k_*_params.root`` files with their own internal random split. Nothing in
the repository established that the two were disjoint, so the end-to-end numbers
could have been measured partly on training events. :func:`assert_disjoint` and
:func:`overlap_report` turn that into a check that either passes or fails loudly,
and ``tools/build_holdout.py`` builds a held-out file whose provenance is
recorded.

*Splits were not auditable.* ``train_test_split`` was called with a bare
``random_state`` and the resulting indices were never kept, so a figure could not
be traced back to the events behind it. :func:`split_indices` returns indices,
which can be stored, re-used across models, and compared between studies.

Only numpy and pandas are required, so these helpers can be exercised by the test
suite without scikit-learn.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "Split",
    "split_indices",
    "event_keys",
    "overlap_report",
    "assert_disjoint",
    "exclude_keys",
    "NotDisjoint",
]


class NotDisjoint(AssertionError):
    """Raised when two event samples that must not overlap do overlap."""


@dataclass(frozen=True)
class Split:
    """Index arrays of a train/validation/test partition."""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray

    def sizes(self) -> dict[str, int]:
        return {"train": self.train.size, "val": self.val.size, "test": self.test.size}

    def check(self, n: int) -> None:
        """Verify the three parts are disjoint and cover exactly ``n`` entries."""
        allidx = np.concatenate([self.train, self.val, self.test])
        if allidx.size != n or np.unique(allidx).size != n:
            raise NotDisjoint(
                f"split does not partition {n} entries (got {allidx.size}, {np.unique(allidx).size} unique)"
            )


def split_indices(
    n: int,
    *,
    test_size: float = 0.2,
    val_size: float = 0.1,
    seed: int = 42,
    stratify: np.ndarray | None = None,
) -> Split:
    """Partition ``range(n)`` into train/validation/test index arrays.

    ``val_size`` and ``test_size`` are both fractions **of the whole sample**, not
    of the remainder: the original scripts mixed the two conventions, so that a
    nominal ``test_size=0.2, val_size=0.2`` actually left 64 %/16 %/20 %. Being
    explicit here keeps the three studies comparable.

    When ``stratify`` is given, the class proportions are preserved in each part.
    """
    if not 0.0 <= test_size < 1.0 or not 0.0 <= val_size < 1.0:
        raise ValueError("test_size and val_size must lie in [0, 1)")
    if test_size + val_size >= 1.0:
        raise ValueError(f"test_size + val_size must stay below 1 (got {test_size + val_size})")

    rng = np.random.default_rng(seed)
    if stratify is None:
        groups: Iterable[np.ndarray] = [rng.permutation(n)]
    else:
        stratify = np.asarray(stratify)
        if stratify.shape[0] != n:
            raise ValueError(f"stratify has length {stratify.shape[0]}, expected {n}")
        groups = [rng.permutation(np.flatnonzero(stratify == value)) for value in np.unique(stratify)]

    train_parts, val_parts, test_parts = [], [], []
    for idx in groups:
        m = idx.size
        n_test = int(round(test_size * m))
        n_val = int(round(val_size * m))
        if m - n_test - n_val <= 0:  # keep at least one training entry per class
            n_test = min(n_test, max(m - 1, 0))
            n_val = max(min(n_val, m - n_test - 1), 0)
        test_parts.append(idx[:n_test])
        val_parts.append(idx[n_test : n_test + n_val])
        train_parts.append(idx[n_test + n_val :])

    split = Split(
        train=np.sort(np.concatenate(train_parts)) if train_parts else np.array([], dtype=int),
        val=np.sort(np.concatenate(val_parts)) if val_parts else np.array([], dtype=int),
        test=np.sort(np.concatenate(test_parts)) if test_parts else np.array([], dtype=int),
    )
    split.check(n)
    return split


def event_keys(df: pd.DataFrame, key_cols: Sequence[str] = ("particlePDG", "eventNumber")) -> pd.Series:
    """Build a stable per-event key, e.g. ``"-211:12345"``.

    Raises ``KeyError`` naming the missing columns, because silently falling back
    to the row order is how the two-tree index pairing bug happened in the first
    place.
    """
    missing = [c for c in key_cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"cannot build an event key: missing column(s) {', '.join(missing)}. "
            "Add them to the branches read from the ROOT tree."
        )
    out = df[key_cols[0]].astype("int64").astype(str)
    for col in key_cols[1:]:
        out = out + ":" + df[col].astype("int64").astype(str)
    return out


def overlap_report(
    keys_a: Iterable[str],
    keys_b: Iterable[str],
    *,
    name_a: str = "A",
    name_b: str = "B",
) -> dict[str, object]:
    """Quantify the overlap between two sets of event keys."""
    set_a, set_b = set(keys_a), set(keys_b)
    common = set_a & set_b
    return {
        "name_a": name_a,
        "name_b": name_b,
        "n_a": len(set_a),
        "n_b": len(set_b),
        "n_common": len(common),
        "fraction_of_b": (len(common) / len(set_b)) if set_b else 0.0,
        "examples": sorted(common)[:10],
    }


def assert_disjoint(
    keys_a: Iterable[str],
    keys_b: Iterable[str],
    *,
    name_a: str = "training sample",
    name_b: str = "held-out sample",
) -> dict[str, object]:
    """Raise :class:`NotDisjoint` if the two samples share any event.

    Returns the report when they are disjoint, so a caller can log the sizes it
    verified.
    """
    report = overlap_report(keys_a, keys_b, name_a=name_a, name_b=name_b)
    if report["n_common"]:
        raise NotDisjoint(
            f"{report['n_common']} of the {report['n_b']} events in {name_b} "
            f"({100.0 * float(report['fraction_of_b']):.2f} %) also appear in {name_a}. "
            f"Examples: {', '.join(map(str, report['examples']))}. "
            "Evaluating on these events would report training performance."
        )
    return report


def exclude_keys(
    df: pd.DataFrame,
    keys: Iterable[str],
    *,
    key_cols: Sequence[str] = ("particlePDG", "eventNumber"),
) -> tuple[pd.DataFrame, int]:
    """Drop the rows of ``df`` whose event key is in ``keys``.

    This is what makes a held-out sample mean anything: ``build_holdout.py`` carves
    the sample out of the parameter files, but the training scripts read those same
    files, so without this the models would be fitted on the very events the
    coupling study evaluates on. Returns the filtered frame and how many rows went.
    """
    wanted = set(keys)
    if not wanted:
        return df, 0
    mask = ~event_keys(df, key_cols).isin(wanted)
    return df.loc[mask].reset_index(drop=True), int((~mask).sum())
