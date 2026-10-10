"""Shared data preparation and reporting for hit-level pion/proton PID.

The GNN and convolutional studies operate on one CSV row per calorimeter hit.
This module keeps their event split, feature scaling, timing smearing and binary
metrics identical so that differences between the models are meaningful.
"""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer, RobustScaler


HIT_COLUMNS = ("eventNumber", "x_coords", "y_coords", "z_coords", "time", "thr")
FEATURE_NAMES = ("x", "y", "z", "time", "thr1", "thr2", "thr3")


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and, when installed, PyTorch."""

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _read_particle(path: str | Path, label: int, particle: str) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Hit CSV not found: {path}. Pass --pion-csv/--proton-csv or set SDHCAL_DATA_DIR."
        )

    frame = pd.read_csv(path)
    frame.columns = frame.columns.str.strip()
    missing = sorted(set(HIT_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")

    shard_column = next(
        (name for name in ("shardIndex", "fileIndex", "runNumber") if name in frame.columns),
        None,
    )
    # primaryEnergy is per event, not a hit feature, so it is not in HIT_COLUMNS
    # and no model reads it. It is carried through when present because the model
    # comparison breaks efficiency down by true energy, and deriving it again
    # afterwards would mean rebuilding event_id outside this function.
    # primaryEnergy is per event; I, J and K are the cell indices the clustering
    # needs. None is a model input: they are carried because dropping them here
    # would mean reading the CSV twice to get them back.
    passthrough = [
        name for name in ("primaryEnergy", "I", "J", "K") if name in frame.columns
    ]
    selected = [*HIT_COLUMNS, *([shard_column] if shard_column else []), *passthrough]
    frame = frame.loc[:, selected].copy()
    for column in HIT_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if frame.isna().any().any() or not np.isfinite(frame[list(HIT_COLUMNS)].to_numpy()).all():
        raise ValueError(f"{path} contains NaN or infinite hit values")

    event_numbers = frame["eventNumber"].to_numpy()
    if not np.all(event_numbers == event_numbers.astype(np.int64)):
        raise ValueError(f"{path} has non-integer eventNumber values")

    thresholds = frame["thr"].to_numpy()
    if not np.all(thresholds == thresholds.astype(np.int64)) or not np.isin(thresholds, [1, 2, 3]).all():
        values = np.unique(thresholds)
        raise ValueError(f"{path} has invalid thr values {values.tolist()}; expected integers 1, 2 or 3")

    # eventNumber is only unique within a shard in the large productions.  Use
    # an explicit shard-like column when one is supplied by the CSV.
    local_id = frame["eventNumber"].astype("int64").astype(str)
    if shard_column is not None:
        local_id = frame[shard_column].astype(str) + ":" + local_id
    frame["event_id"] = particle + ":" + local_id
    frame["label"] = label
    return frame


def load_binary_hits(
    pion_path: str | Path,
    proton_path: str | Path,
    *,
    sigma_time: float,
    seed: int,
) -> pd.DataFrame:
    """Load pion/proton hits and apply one reproducible detector-time smearing."""

    if sigma_time < 0:
        raise ValueError("sigma_time must be non-negative")
    full = pd.concat(
        [
            _read_particle(pion_path, 0, "pion"),
            _read_particle(proton_path, 1, "proton"),
        ],
        ignore_index=True,
    )
    if sigma_time:
        rng = np.random.default_rng(seed)
        full["time"] = full["time"].to_numpy() + rng.normal(0.0, sigma_time, len(full))
    return full


def split_events(
    frame: pd.DataFrame,
    *,
    test_size: float = 0.2,
    val_size: float = 0.1,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Make disjoint, class-stratified train/validation/test event splits."""

    if test_size <= 0 or val_size <= 0 or test_size + val_size >= 1:
        raise ValueError("test_size and val_size must be positive and sum to less than one")

    events = frame[["event_id", "label"]].drop_duplicates()
    if events["event_id"].duplicated().any():
        raise ValueError("At least one event_id is associated with multiple labels")
    counts = events["label"].value_counts()
    if set(counts.index) != {0, 1} or counts.min() < 3:
        raise ValueError(f"Need at least three events of each class, got {counts.to_dict()}")

    train_events, test_events = train_test_split(
        events,
        test_size=test_size,
        random_state=seed,
        stratify=events["label"],
    )
    relative_val_size = val_size / (1.0 - test_size)
    train_events, val_events = train_test_split(
        train_events,
        test_size=relative_val_size,
        random_state=seed,
        stratify=train_events["label"],
    )
    ids = {
        "train": set(train_events["event_id"]),
        "val": set(val_events["event_id"]),
        "test": set(test_events["event_id"]),
    }
    if ids["train"] & ids["val"] or ids["train"] & ids["test"] or ids["val"] & ids["test"]:
        raise RuntimeError("Event leakage detected between data splits")
    return {
        name: frame[frame["event_id"].isin(event_ids)].copy()
        for name, event_ids in ids.items()
    }


@dataclass
class HitPreprocessor:
    """Seven hit features fitted exclusively on training hits."""

    seed: int = 42
    spatial_scaler: RobustScaler | None = None
    time_scaler: QuantileTransformer | None = None

    def fit(self, train: pd.DataFrame) -> "HitPreprocessor":
        if train.empty:
            raise ValueError("Cannot fit the hit preprocessor on an empty training set")
        self.spatial_scaler = RobustScaler().fit(train[["x_coords", "y_coords", "z_coords"]])
        n_quantiles = min(1000, len(train))
        self.time_scaler = QuantileTransformer(
            n_quantiles=n_quantiles,
            output_distribution="normal",
            random_state=self.seed,
        ).fit(train[["time"]])
        return self

    #: Columns a clustered table carries that a per-pad one cannot.
    CLUSTER_COLUMNS = ("frac_thr1", "frac_thr2", "frac_thr3", "size")

    @staticmethod
    def is_clustered(table: pd.DataFrame) -> bool:
        return all(name in table.columns for name in HitPreprocessor.CLUSTER_COLUMNS)

    def n_features(self, table: pd.DataFrame) -> int:
        """Width of the vector transform() produces for this table."""
        return 8 if self.is_clustered(table) else len(FEATURE_NAMES)

    def transform(self, hits: pd.DataFrame) -> np.ndarray:
        if self.spatial_scaler is None or self.time_scaler is None:
            raise RuntimeError("HitPreprocessor.fit must be called before transform")
        spatial = self.spatial_scaler.transform(hits[["x_coords", "y_coords", "z_coords"]])
        timing = self.time_scaler.transform(hits[["time"]])
        if self.is_clustered(hits):
            # A cluster has no single threshold: it has a composition and a size.
            # log1p keeps a forty-pad cluster from dwarfing a two-pad one.
            composition = hits[["frac_thr1", "frac_thr2", "frac_thr3"]].to_numpy(dtype=np.float32)
            size = np.log1p(hits[["size"]].to_numpy(dtype=np.float32))
            return np.hstack([spatial, timing, composition, size]).astype(np.float32)
        thresholds = hits["thr"].to_numpy(dtype=np.int64)
        if not np.isin(thresholds, [1, 2, 3]).all():
            raise ValueError("thr must contain only 1, 2 or 3")
        one_hot = np.eye(3, dtype=np.float32)[thresholds - 1]
        return np.hstack([spatial, timing, one_hot]).astype(np.float32)


def binary_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    """Return scalar and per-class metrics at the fixed 0.5 threshold."""

    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = (probabilities >= 0.5).astype(np.int64)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    efficiencies = np.divide(
        np.diag(matrix),
        matrix.sum(axis=1),
        out=np.zeros(2, dtype=float),
        where=matrix.sum(axis=1) != 0,
    )
    purities = np.divide(
        np.diag(matrix),
        matrix.sum(axis=0),
        out=np.zeros(2, dtype=float),
        where=matrix.sum(axis=0) != 0,
    )
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "auc": float(roc_auc_score(labels, probabilities)),
        "confusion_matrix": matrix.tolist(),
        "pion_efficiency": float(efficiencies[0]),
        "proton_efficiency": float(efficiencies[1]),
        "pion_purity": float(purities[0]),
        "proton_purity": float(purities[1]),
    }


def append_csv(path: Path, row: dict[str, Any]) -> None:
    """Append a flat run record with a header on first use."""

    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
