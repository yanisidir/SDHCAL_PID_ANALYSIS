"""Run bookkeeping that survives a change of schema.

The original loggers wrote the CSV header once, when the file was created, and
then appended rows built from whatever keys the config happened to have::

    def _write_dict_row(csv_path, row):
        is_new = not csv_path.exists()
        with csv_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if is_new:
                writer.writeheader()
            writer.writerow(row)

When ``val_size`` was later added to the config, every subsequent row gained a
field that the header did not have, so **all columns after it were shifted by
one**: in ``PID/BDT/run_parameters.csv`` run 9 appears to have
``random_state=0.2`` and ``learning_rate=42``. The hyper-parameters of the model
actually used downstream were no longer recoverable. On top of that, neither the
feature list nor the smearing configuration was logged, although those were the
variables being scanned.

:func:`append_row` therefore widens the header instead of silently misaligning
rows, and :class:`RunLogger` writes the complete configuration - features and
smearing included - to a per-run JSON file next to the CSV.
"""

from __future__ import annotations

import csv
import json
import logging
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .manifest import EVENT_COUNT_FIELDS
from .provenance import environment_fingerprint, to_jsonable

logger = logging.getLogger(__name__)

__all__ = ["append_row", "read_rows", "next_run_id", "RunLogger", "environment_fingerprint"]

#: Column written first in every CSV produced here.
_ID = "run_id"


def read_rows(csv_path: Path) -> list[dict[str, str]]:
    """Return the rows of a CSV as dicts, or ``[]`` if the file does not exist."""
    if not csv_path.is_file():
        return []
    with csv_path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def append_row(csv_path: Path, row: Mapping[str, Any]) -> None:
    """Append ``row`` to a CSV, widening the header when new keys appear.

    Values are written under their own column name whatever the order of the
    mapping. If ``row`` introduces keys the file does not have, the file is
    rewritten with the union of the columns and the existing rows are padded with
    empty strings, so a schema change can never shift the meaning of a column.
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    new_keys = list(row.keys())
    if not csv_path.is_file():
        header = ([_ID] if _ID in new_keys else []) + [k for k in new_keys if k != _ID]
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerow({k: row.get(k, "") for k in header})
        return

    existing = read_rows(csv_path)
    with csv_path.open(newline="", encoding="utf-8") as f:
        current_header = next(csv.reader(f), [])

    missing = [k for k in new_keys if k not in current_header]
    if not missing:
        with csv_path.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=current_header).writerow({k: row.get(k, "") for k in current_header})
        return

    header = current_header + missing
    logger.info("Widening %s with new columns: %s", csv_path.name, ", ".join(missing))
    tmp = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for old in existing:
            writer.writerow({k: old.get(k, "") for k in header})
        writer.writerow({k: row.get(k, "") for k in header})
    os.replace(tmp, csv_path)


def next_run_id(*csv_paths: Path) -> int:
    """Return one past the highest ``run_id`` found in the given CSVs."""
    highest = 0
    for path in csv_paths:
        for row in read_rows(Path(path)):
            try:
                highest = max(highest, int(float(row.get(_ID, 0) or 0)))
            except (TypeError, ValueError):
                continue
    return highest + 1


@dataclass
class RunLogger:
    """Allocate a run id and record its configuration, metrics and artifacts.

    Parameters
    ----------
    base_dir:
        Directory holding ``run_parameters.csv``, ``performances.csv``,
        ``comments.csv`` and the ``runs/`` folder of per-run JSON files.
    name:
        Prefix of the CSV files, so several studies can share a directory.
    """

    base_dir: Path
    name: str = "run"

    def __post_init__(self) -> None:
        self.base_dir = Path(self.base_dir)
        self.runs_dir = self.base_dir / "runs"
        self.params_csv = self.base_dir / f"{self.name}_parameters.csv"
        self.perf_csv = self.base_dir / f"{self.name}_performances.csv"
        self.comments_csv = self.base_dir / f"{self.name}_comments.csv"
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.run_id: int | None = None
        self._config: dict[str, Any] = {}

    def start(self, config: Mapping[str, Any], *, features: Sequence[str] | None = None) -> int:
        """Open a run: allocate an id, write the JSON record and the CSV row.

        ``features`` is stored explicitly because the feature set was the main
        quantity scanned between runs and was absent from the original logs.
        """
        self.run_id = next_run_id(self.params_csv, self.perf_csv)
        flat: dict[str, Any] = {_ID: self.run_id}
        flat.update({k: _flatten(v) for k, v in config.items()})
        if features is not None:
            flat["n_features"] = len(features)
            flat["features"] = ";".join(features)

        self._config = {
            _ID: self.run_id,
            "config": {k: _jsonable(v) for k, v in config.items()},
            "features": list(features) if features is not None else None,
            "environment": environment_fingerprint(),
        }
        self._write_json()
        append_row(self.params_csv, flat)
        logger.info("Run %d: configuration logged to %s", self.run_id, self.params_csv)
        return self.run_id

    def log_metrics(self, **metrics: Any) -> None:
        """Record test metrics for the current run, in the CSV and the JSON."""
        self._require_started()
        append_row(self.perf_csv, {_ID: self.run_id, **{k: _flatten(v) for k, v in metrics.items()}})
        self._config.setdefault("metrics", {}).update({k: _jsonable(v) for k, v in metrics.items()})
        self._write_json()

    def log_comment(self, comment: str) -> None:
        """Record a free-text note for the current run."""
        self._require_started()
        append_row(self.comments_csv, {_ID: self.run_id, "comment": comment})
        self._config["comment"] = comment
        self._write_json()

    def log_artifacts(self, **artifacts: Any) -> None:
        """Record the paths this run wrote, so a figure can be traced to a run."""
        self._require_started()
        self._config.setdefault("artifacts", {}).update({k: str(v) for k, v in artifacts.items()})
        self._write_json()

    def log_event_counts(self, **counts: int | None) -> None:
        """Record exact split counts, using null when a quantity is unavailable."""
        self._require_started()
        unknown = sorted(set(counts) - set(EVENT_COUNT_FIELDS))
        if unknown:
            raise ValueError(f"unknown event count field(s): {', '.join(unknown)}")
        for name, value in counts.items():
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
                raise ValueError(f"event count {name} must be a non-negative integer or null")
        self._config.setdefault("event_counts", {}).update(counts)
        self._write_json()

    def path(self, *parts: str) -> Path:
        """Build a run-scoped output path, creating the parent directory."""
        self._require_started()
        path = self.base_dir.joinpath(*parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def json_path(self) -> Path:
        self._require_started()
        return self.runs_dir / f"{self.name}_{self.run_id}.json"

    def _write_json(self) -> None:
        self.json_path.write_text(json.dumps(self._config, indent=2, default=str), encoding="utf-8")

    def _require_started(self) -> None:
        if self.run_id is None:
            raise RuntimeError("RunLogger.start() must be called before logging")


def _flatten(value: Any) -> Any:
    """Render a config value as a single CSV cell."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return ";".join(f"{k}={v}" for k, v in value.items())
    if isinstance(value, Iterable):
        return ";".join(str(v) for v in value)
    return str(value)


def _jsonable(value: Any) -> Any:
    return to_jsonable(value)
