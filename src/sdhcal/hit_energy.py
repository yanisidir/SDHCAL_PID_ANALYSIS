"""Hit-level event loading and preprocessing for energy reconstruction.

ROOT imports stay local to :func:`load_particle_events`.  PyTorch and PyG are
optional dependencies and are isolated in this hit-level module.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer, RobustScaler
from torch.utils.data import Dataset
from torch_geometric.data import Data

from . import calib, clustering, metrics
from .graph import build_knn_graph

if TYPE_CHECKING:
    import argparse

LOGGER = logging.getLogger("gnn_energy_reco")

__all__ = [
    "ArrayHitPreprocessor",
    "EnergyGraphDataset",
    "ShowerEvent",
    "load_particle_events",
    "regression_metrics",
    "require_tree",
    "split_train_calibration",
]


@dataclass
class ShowerEvent:
    coordinates: np.ndarray
    time: np.ndarray
    threshold: np.ndarray
    energy: float
    particle: str
    event_id: str
    composition: np.ndarray | None = None
    baseline: float = 0.0


@dataclass
class ArrayHitPreprocessor:
    """Historical spatial/time transformation fitted on training hits only.

    ``drop_timing`` is the controlled ablation: the time column is replaced by a
    constant rather than removed, so the input width, the architecture and the
    parameter count are identical to the timed run. Only the information differs,
    which is what an ablation is supposed to isolate; dropping the column would
    also shrink the first layer and confound the two.
    """

    seed: int
    spatial_scaler: RobustScaler | None = None
    time_scaler: QuantileTransformer | None = None
    drop_timing: bool = False

    def fit(self, events: list[ShowerEvent], indices: np.ndarray, max_hits: int) -> None:
        if max_hits < 1:
            raise ValueError("scaler-max-hits must be positive")
        rng = np.random.default_rng(self.seed)
        coordinate_parts: list[np.ndarray] = []
        time_parts: list[np.ndarray] = []
        remaining = max_hits
        for index in rng.permutation(indices):
            event = events[int(index)]
            take = min(len(event.time), remaining)
            if take < len(event.time):
                selected = rng.choice(len(event.time), size=take, replace=False)
            else:
                selected = slice(None)
            coordinate_parts.append(event.coordinates[selected])
            time_parts.append(event.time[selected])
            remaining -= take
            if remaining == 0:
                break
        if not coordinate_parts:
            raise ValueError("No training hits available to fit the preprocessor")
        coordinates = np.vstack(coordinate_parts)
        timing = np.concatenate(time_parts).reshape(-1, 1)
        self.spatial_scaler = RobustScaler().fit(coordinates)
        self.time_scaler = QuantileTransformer(
            n_quantiles=min(1000, len(timing)),
            output_distribution="normal",
            random_state=self.seed,
        ).fit(timing)

    def transform(self, event: ShowerEvent) -> np.ndarray:
        if self.spatial_scaler is None or self.time_scaler is None:
            raise RuntimeError("Preprocessor must be fitted before transform")
        spatial = self.spatial_scaler.transform(event.coordinates)
        if self.drop_timing:
            timing = np.zeros((len(event.time), 1), dtype=np.float64)
        else:
            timing = self.time_scaler.transform(event.time.reshape(-1, 1))
        if event.composition is not None:
            size = np.log1p(event.composition[:, 3:4])
            return np.hstack([spatial, timing, event.composition[:, :3], size]).astype(np.float32)
        if not np.isin(event.threshold, [1, 2, 3]).all():
            raise ValueError(f"Invalid threshold in event {event.event_id}")
        one_hot = np.eye(3, dtype=np.float32)[event.threshold.astype(np.int64) - 1]
        return np.hstack([spatial, timing, one_hot]).astype(np.float32)


def require_tree(handle, name: str, path: Path):
    """Return the requested ROOT tree or raise the historical diagnostic."""
    if name not in handle:
        raise KeyError(f"TTree {name!r} not found in {path}")
    return handle[name]


def load_particle_events(
    particle: str,
    shard_paths: list[Path],
    *,
    sample_role: str,
    tree_name: str,
    sigma_time: float,
    seed: int,
    max_events_per_shard: int | None,
    require_contained: bool,
    cluster_hits: bool = False,
    baseline_coefficients: np.ndarray | None = None,
) -> list[ShowerEvent]:
    """Read self-contained hit/energy shards from one simulation production."""
    import awkward as ak
    import uproot

    if not shard_paths:
        raise FileNotFoundError(f"No ROOT shard found for {sample_role}/{particle}")
    rng = np.random.default_rng(seed)
    events: list[ShowerEvent] = []
    rejected = {"invalid": 0, "empty": 0, "not_contained": 0}
    for shard_path in shard_paths:
        if not shard_path.is_file():
            raise FileNotFoundError(shard_path)
        with uproot.open(shard_path) as handle:
            tree = require_tree(handle, tree_name, shard_path)
            available = set(tree.keys())
            if {"x", "y", "z"} <= available:
                coordinate_columns = ("x", "y", "z")
            elif {"I", "J", "K"} <= available:
                coordinate_columns = ("I", "J", "K")
            else:
                raise KeyError(
                    f"{shard_path} needs vector branches x/y/z or I/J/K; available: {sorted(available)}"
                )
            energy_column = "energy" if "energy" in available else "primaryEnergy"
            if energy_column not in available:
                raise KeyError(f"{shard_path} has neither energy nor primaryEnergy")
            branches = [*coordinate_columns, "time", "thr", energy_column]
            if "eventNumber" in available:
                branches.append("eventNumber")
            if (require_contained or cluster_hits) and "K" not in branches:
                if "K" not in available:
                    raise KeyError("--require-contained and --cluster-hits need the K layer branch")
                branches.append("K")
            if cluster_hits:
                for name in ("I", "J"):
                    if name not in available:
                        raise KeyError(f"--cluster-hits needs the {name} cell-index branch")
                    if name not in branches:
                        branches.append(name)
            missing = [name for name in branches if name not in available]
            if missing:
                raise KeyError(f"{shard_path} missing branches: {missing}")
            entry_stop = max_events_per_shard if max_events_per_shard and max_events_per_shard > 0 else None
            arrays = tree.arrays(branches, entry_stop=entry_stop, library="ak", how=dict)

        n_events = len(arrays["thr"])
        energies = np.asarray(ak.to_numpy(arrays[energy_column]), dtype=np.float64)
        event_numbers = (
            np.asarray(ak.to_numpy(arrays["eventNumber"]), dtype=np.int64)
            if "eventNumber" in arrays
            else np.arange(n_events, dtype=np.int64)
        )
        for row in range(n_events):
            event_number = int(event_numbers[row])
            energy = float(energies[row])
            if not np.isfinite(energy) or energy <= 0:
                rejected["invalid"] += 1
                continue
            coordinates = np.column_stack(
                [np.asarray(ak.to_numpy(arrays[name][row]), dtype=np.float32) for name in coordinate_columns]
            )
            timing = np.asarray(ak.to_numpy(arrays["time"][row]), dtype=np.float32)
            threshold = np.asarray(ak.to_numpy(arrays["thr"][row]), dtype=np.int64)
            if len(timing) == 0:
                rejected["empty"] += 1
                continue
            if len(coordinates) != len(timing) or len(threshold) != len(timing):
                raise ValueError(f"Inconsistent jagged lengths in {shard_path.name}:{event_number}")
            if not np.isfinite(coordinates).all() or not np.isfinite(timing).all():
                rejected["invalid"] += 1
                continue
            if not np.isin(threshold, [1, 2, 3]).all():
                rejected["invalid"] += 1
                continue
            if require_contained and np.any(np.asarray(ak.to_numpy(arrays["K"][row]), dtype=np.int64) == 47):
                rejected["not_contained"] += 1
                continue
            if sigma_time:
                timing = timing + rng.normal(0.0, sigma_time, len(timing)).astype(np.float32)

            baseline = 0.0
            if baseline_coefficients is not None:
                counts = [float(np.count_nonzero(threshold == level)) for level in (1, 2, 3)]
                baseline = float(
                    calib.ereco(
                        np.array([counts[0]]),
                        np.array([counts[1]]),
                        np.array([counts[2]]),
                        baseline_coefficients,
                    )[0]
                )
                if not np.isfinite(baseline) or baseline <= 0.0:
                    rejected["no_baseline"] += 1
                    continue

            composition = None
            if cluster_hits:
                labels = clustering.cluster_labels(
                    np.asarray(ak.to_numpy(arrays["I"][row]), dtype=np.int64),
                    np.asarray(ak.to_numpy(arrays["J"][row]), dtype=np.int64),
                    np.asarray(ak.to_numpy(arrays["K"][row]), dtype=np.int64),
                )
                summary = clustering.cluster_features(
                    coordinates[:, 0],
                    coordinates[:, 1],
                    coordinates[:, 2],
                    timing,
                    threshold,
                    labels,
                )
                names = clustering.CLUSTER_FEATURE_NAMES
                coordinates = summary[:, [names.index("x"), names.index("y"), names.index("z")]]
                timing = summary[:, names.index("time")]
                composition = summary[
                    :,
                    [names.index(f"frac_thr{level}") for level in (1, 2, 3)] + [names.index("size")],
                ]
                threshold = np.ones(len(coordinates), dtype=np.int64)

            events.append(
                ShowerEvent(
                    coordinates=coordinates,
                    time=timing,
                    threshold=threshold,
                    composition=composition,
                    energy=energy,
                    particle=particle,
                    baseline=baseline,
                    event_id=f"{sample_role}:{particle}:{shard_path.stem}:{event_number}",
                )
            )
        LOGGER.info(
            "%s/%s: read %s (%d cumulative events)",
            sample_role,
            particle,
            shard_path.name,
            len(events),
        )
    LOGGER.info("%s/%s: loaded %d events; rejected %s", sample_role, particle, len(events), rejected)
    if not events:
        raise RuntimeError(f"No usable {particle} event was loaded")
    return events


def split_train_calibration(
    events: list[ShowerEvent], *, calibration_size: float, seed: int, energy_bins: int
) -> tuple[np.ndarray, np.ndarray]:
    """Split the training production into fit and internal-calibration subsets."""
    if not 0 < calibration_size < 1:
        raise ValueError("calibration-size must lie strictly between zero and one")
    indices = np.arange(len(events))
    particles = np.asarray([event.particle for event in events])
    energies = np.asarray([event.energy for event in events])
    last_error: Exception | None = None
    for n_bins in range(max(1, energy_bins), 0, -1):
        strata = np.empty(len(events), dtype=object)
        for particle in np.unique(particles):
            selected = particles == particle
            values = energies[selected]
            edges = np.unique(np.quantile(values, np.linspace(0, 1, n_bins + 1)))
            bins = np.digitize(values, edges[1:-1], right=False)
            strata[selected] = [f"{particle}:{value}" for value in bins]
        try:
            train, calibration = train_test_split(
                indices, test_size=calibration_size, random_state=seed, stratify=strata
            )
            LOGGER.info("Split stratified with %d energy quantile bins", n_bins)
            return train, calibration
        except ValueError as error:
            last_error = error
    raise ValueError(f"Cannot create a stratified train/calibration split: {last_error}")


#: Memory a dataset may spend caching transformed features before it gives up and
#: recomputes them. Clustered runs need about 0.4 GB; raw-hit runs about four
#: times that, which is why the budget exists rather than an unconditional cache.
FEATURE_CACHE_BUDGET_BYTES = 2 * 1024**3


class EnergyGraphDataset(Dataset):
    """Construct energy graphs lazily in the historical event-index order.

    The scalers are deterministic once fitted, so the transformed features of an
    event never change between epochs; recomputing them every time cost about
    12 % of a thirty-epoch run, measured at 782 us per event. They are cached on
    first use, under a memory budget: the cache is sized from the first event and
    switched off, with a line in the log, when it would not fit.
    """

    def __init__(
        self,
        events: list[ShowerEvent],
        indices: np.ndarray,
        preprocessor: ArrayHitPreprocessor,
        args: argparse.Namespace,
        *,
        cache_features: bool = True,
        cache_budget_bytes: int = FEATURE_CACHE_BUDGET_BYTES,
    ) -> None:
        self.events = events
        self.indices = np.asarray(indices, dtype=np.int64)
        self.preprocessor = preprocessor
        self.k = args.k
        self.time_window = args.time_window
        self.causal = args.causal
        self.undirected = args.undirected
        self.knn_chunk_size = args.knn_chunk_size
        self.build_edges = args.architecture != "deepsets"
        self._cache_enabled = bool(cache_features)
        self._cache_budget = int(cache_budget_bytes)
        self._cache: dict[int, np.ndarray] = {}

    def _features(self, index: int, event: ShowerEvent) -> np.ndarray:
        """Transformed features of one event, from the cache when it is kept."""
        if not self._cache_enabled:
            return self.preprocessor.transform(event)
        cached = self._cache.get(index)
        if cached is not None:
            return cached
        features = self.preprocessor.transform(event)
        if not self._cache:
            # Size the whole cache from the first event rather than discovering
            # the cost one event at a time.
            estimate = features.nbytes * len(self.indices)
            if estimate > self._cache_budget:
                LOGGER.info(
                    "feature cache off: %.1f GB needed, budget %.1f GB; features are recomputed each epoch",
                    estimate / 1024**3,
                    self._cache_budget / 1024**3,
                )
                self._cache_enabled = False
                return features
            LOGGER.info(
                "feature cache on: about %.2f GB for %d events", estimate / 1024**3, len(self.indices)
            )
        self._cache[index] = features
        return features

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> Data:
        index = int(self.indices[position])
        event = self.events[index]
        features = self._features(index, event)
        if self.build_edges:
            edge_index = build_knn_graph(
                features,
                event.time,
                k=self.k,
                time_window=self.time_window,
                causal=self.causal,
                undirected=self.undirected,
                chunk_size=self.knn_chunk_size,
            )
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        return Data(
            x=torch.from_numpy(features),
            edge_index=edge_index,
            y=torch.tensor([event.energy], dtype=torch.float32),
            baseline=torch.tensor([event.baseline], dtype=torch.float32),
            event_ref=torch.tensor([index], dtype=torch.long),
            num_nodes=len(features),
        )


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    """Return the unchanged external-validation energy metrics."""
    core = metrics.core_bias_resolution(metrics.relative_residual(y_true, y_pred))
    r2 = float("nan") if np.ptp(y_true) == 0 else float(r2_score(y_true, y_pred))
    return {
        "rmse_gev": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "mae_gev": float(mean_absolute_error(y_true, y_pred)),
        "r2": r2,
        "bias": core.bias,
        "bias_err": core.bias_err,
        "resolution": core.sigma,
        "resolution_err": core.sigma_err,
        "energy_scale": float(np.mean(y_pred) / np.mean(y_true)),
        "rms_rel_about_zero": metrics.rms_relative_about_zero(y_true, y_pred),
        "n_validation": int(len(y_true)),
    }
