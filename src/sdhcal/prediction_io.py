"""Prediction tables, sidecars and hit-level event identifier conversion."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from . import benchmark, paths, shards
from .artifact_selection import select_latest

__all__ = [
    "PARTICLE_ALIASES",
    "ROLE_TO_SAMPLE",
    "find_newest",
    "npz_to_frame",
    "parse_event_id",
    "prediction_path",
    "read_prediction_table",
    "read_sidecar",
    "write_prediction_table",
]

ROLE_TO_SAMPLE = {"validation": "discret", "val": "discret", "train": "200k"}
PARTICLE_ALIASES = {"pion": "pi", "pi": "pi", "proton": "proton"}


def prediction_path(kind: str, name: str) -> Path:
    """Return the historical benchmark CSV path for a named model."""
    return paths.output_dir() / "benchmark" / kind / f"{name}.csv"


def parse_event_id(event_id: str) -> tuple[str, str, int]:
    """Convert either historical hit-level identifier spelling to a global key."""
    parts = event_id.split(":")
    if len(parts) == 4:
        role, particle, stem, number = parts
        sample = ROLE_TO_SAMPLE.get(role)
        if sample is None:
            raise ValueError(f"unknown sample role {role!r} in {event_id!r}; known: {sorted(ROLE_TO_SAMPLE)}")
        shard = shards.shard_id(sample, f"{stem}_params.root")
    elif len(parts) == 3:
        role, (particle, shard_text, number) = "validation", parts
        shard = int(shard_text)
    else:
        raise ValueError(f"cannot read an event id from {event_id!r}; expected three or four fields")

    if particle not in PARTICLE_ALIASES:
        raise ValueError(f"unknown particle {particle!r} in {event_id!r}")
    global_number = int(shards.global_event_number(shard, np.array([int(number)]))[0])
    return role, PARTICLE_ALIASES[particle], global_number


def find_newest(pattern: str) -> Path:
    """Resolve the newest file matching the CLI's historical glob semantics."""
    return select_latest(pattern)


def npz_to_frame(payload, kind: str) -> tuple[pd.DataFrame, set[str]]:
    """Translate a supported hit-level NPZ payload to the benchmark table."""
    required = {
        "energy": ("y_true", "y_pred", "event_id"),
        "pid": ("y_score", "event_id", "primary_energy"),
    }[kind]
    missing = [key for key in required if key not in payload.files]
    if missing:
        raise KeyError(f"lacks {', '.join(missing)}; it holds {sorted(payload.files)}")

    event_ids = [str(value) for value in payload["event_id"]]
    parsed = [parse_event_id(event_id) for event_id in event_ids]
    roles = {role for role, _, _ in parsed}
    table = pd.DataFrame(
        {
            "eventNumber": [number for _, _, number in parsed],
            "particlePDG": [paths.PARTICLE_PDG[particle] for _, particle, _ in parsed],
            "primaryEnergy": np.asarray(
                payload["y_true" if kind == "energy" else "primary_energy"], dtype=float
            ),
        }
    )
    if kind == "energy":
        table["energyPred"] = np.asarray(payload["y_pred"], dtype=float)
    else:
        table["score"] = np.asarray(payload["y_score"], dtype=float)
    benchmark.check_predictions(table, kind)
    return table, roles


def write_prediction_table(
    path: str | Path,
    frame: pd.DataFrame,
    kind: str,
    metadata: Mapping[str, object],
) -> Path:
    """Write the historical CSV and adjacent JSON sidecar without schema changes."""
    destination = Path(path)
    benchmark.check_predictions(frame, kind)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    destination.with_suffix(".json").write_text(
        json.dumps(dict(metadata), indent=2),
        encoding="utf-8",
    )
    return destination


def read_prediction_table(path: str | Path, kind: str | None = None) -> pd.DataFrame:
    """Read a benchmark CSV, optionally validating it without reordering."""
    frame = pd.read_csv(path)
    return benchmark.check_predictions(frame, kind) if kind is not None else frame


def read_sidecar(path: str | Path) -> dict | None:
    """Read an optional prediction sidecar, preserving missing/invalid behavior."""
    sidecar = Path(path)
    if not sidecar.is_file():
        return None
    try:
        value = json.loads(sidecar.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None
