"""Shared, behavior-preserving preparation of tabular SDHCAL samples.

This module orchestrates operations that are identical in the tabular MLP
workflows: Python/NumPy seeding, ordered shard loading, deterministic sampling,
row cleaning, and energy-within-group calibration splitting. Particle labels,
physics selections, smearing, targets, and model preprocessing remain in their
workflow modules.

``shards`` remains responsible for shard discovery, ROOT access, metadata, and
event identifiers. ``rootio`` remains the low-level reader for consolidated
ROOT files, while ``splits`` retains the general train/validation/test and event
identity primitives. The functions here deliberately preserve the historical
MLP algorithms, including their ordering and random-state semantics.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import shards

__all__ = [
    "clean_event_rows",
    "read_shard_particles",
    "sample_per_particle",
    "seed_python_numpy",
    "stratified_energy_split_indices",
]


def seed_python_numpy(seed: int) -> None:
    """Seed the process-global Python and legacy NumPy random generators."""
    random.seed(seed)
    np.random.seed(seed)


def read_shard_particles(
    sample: str,
    particles: Sequence[str],
    *,
    features: Sequence[str],
    limit: int | None = None,
) -> pd.DataFrame:
    """Read particles in the supplied order and concatenate their shard rows."""
    frames = [shards.read_sample(sample, particle, features=features, limit=limit) for particle in particles]
    return pd.concat(frames, ignore_index=True)


def sample_per_particle(
    frame: pd.DataFrame,
    n: int,
    seed: int,
    *,
    particle_column: str = "particlePDG",
) -> pd.DataFrame:
    """Sample at most ``n`` rows per particle using the same seed per group.

    Groups retain their first-occurrence order. Each group uses pandas'
    historical ``random_state=seed`` behavior independently, and concatenation
    resets the index. These details are part of the workflow contract.
    """
    return pd.concat(
        [
            group.sample(n=min(len(group), n), random_state=seed)
            for _, group in frame.groupby(particle_column, sort=False)
        ],
        ignore_index=True,
    )


def clean_event_rows(
    frame: pd.DataFrame,
    required_columns: Sequence[str],
    *,
    reset_index: bool = True,
) -> pd.DataFrame:
    """Replace infinities by NaN and drop rows incomplete in required columns."""
    cleaned = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=list(required_columns))
    if reset_index:
        return cleaned.reset_index(drop=True)
    return cleaned


def stratified_energy_split_indices(
    energy: np.ndarray,
    groups: np.ndarray,
    *,
    test_size: float,
    random_state: int,
    max_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Split indices after stratifying quantile-binned energy within each group.

    The bin count is reduced from ``max_bins`` to one until scikit-learn can
    form the requested split. This is a mechanical extraction of the two MLP
    calibration split implementations; it intentionally retains their quantile,
    digitization, string-stratum, and fallback behavior.
    """
    if not 0 < test_size < 1:
        raise ValueError("calibration-size must lie strictly between zero and one")

    energy = np.asarray(energy)
    groups = np.asarray(groups)
    last_error = None
    for n_bins in range(max_bins, 0, -1):
        quantile = np.zeros(len(energy), dtype=int)
        for value in np.unique(groups):
            selected = groups == value
            edges = np.unique(np.quantile(energy[selected], np.linspace(0, 1, n_bins + 1)))
            quantile[selected] = np.digitize(energy[selected], edges[1:-1])
        strata = np.asarray([f"{kind}:{bin_}" for kind, bin_ in zip(groups, quantile, strict=True)])
        try:
            return train_test_split(
                np.arange(len(energy)),
                test_size=test_size,
                random_state=random_state,
                stratify=strata,
            )
        except ValueError as error:
            last_error = error
    raise ValueError(f"Cannot create the train/calibration split: {last_error}")
