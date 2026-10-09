"""Run logging: a schema change must widen the header, never shift a column."""

from __future__ import annotations

import csv
import json

import pytest

from sdhcal import runlog


def _read(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_appending_a_new_key_widens_the_header(tmp_path):
    """This is the exact corruption found in PID/BDT/run_parameters.csv.

    The old writer kept the header from the first row, so adding ``val_size``
    to the config shifted every later column by one: run 9 ended up recorded
    with ``random_state=0.2`` and ``learning_rate=42``.
    """
    csv_path = tmp_path / "params.csv"
    runlog.append_row(csv_path, {"run_id": 1, "test_size": 0.2, "random_state": 42, "lr": 0.01})
    runlog.append_row(
        csv_path,
        {"run_id": 2, "test_size": 0.2, "val_size": 0.2, "random_state": 42, "lr": 0.01},
    )

    rows = _read(csv_path)
    assert [r["run_id"] for r in rows] == ["1", "2"]
    # Every value stays under its own name in both rows.
    assert rows[0]["random_state"] == "42" and rows[0]["lr"] == "0.01"
    assert rows[1]["random_state"] == "42" and rows[1]["lr"] == "0.01"
    assert rows[1]["val_size"] == "0.2"
    # The pre-existing row is padded, not shifted.
    assert rows[0]["val_size"] == ""


def test_column_order_in_the_row_does_not_matter(tmp_path):
    csv_path = tmp_path / "params.csv"
    runlog.append_row(csv_path, {"run_id": 1, "a": 1, "b": 2})
    runlog.append_row(csv_path, {"b": 20, "run_id": 2, "a": 10})
    rows = _read(csv_path)
    assert rows[1] == {"run_id": "2", "a": "10", "b": "20"}


def test_run_id_is_first_column(tmp_path):
    csv_path = tmp_path / "params.csv"
    runlog.append_row(csv_path, {"alpha": 1, "run_id": 1})
    with csv_path.open(newline="", encoding="utf-8") as f:
        assert next(csv.reader(f))[0] == "run_id"


def test_next_run_id_spans_several_files(tmp_path):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    assert runlog.next_run_id(a, b) == 1
    runlog.append_row(a, {"run_id": 3, "x": 1})
    runlog.append_row(b, {"run_id": 7, "y": 1})
    assert runlog.next_run_id(a, b) == 8


def test_next_run_id_ignores_unparsable_ids(tmp_path):
    path = tmp_path / "a.csv"
    runlog.append_row(path, {"run_id": "nan", "x": 1})
    assert runlog.next_run_id(path) == 1


def test_runlogger_records_features_and_smearing(tmp_path):
    """The feature list and the smearing width were scanned but never logged."""
    log = runlog.RunLogger(tmp_path, name="pid")
    features = ["Thr1", "Thr2", "tMin"]
    run_id = log.start({"learning_rate": 0.01, "smear_sigma_ns": 0.1, "smear_seed": 42}, features=features)
    log.log_metrics(test_acc=0.91, test_auc=0.98)
    log.log_comment("baseline with timing")
    log.log_artifacts(confusion_matrix=tmp_path / "cm.png")

    assert run_id == 1
    row = _read(log.params_csv)[0]
    assert row["features"] == "Thr1;Thr2;tMin"
    assert row["n_features"] == "3"
    assert row["smear_sigma_ns"] == "0.1"

    record = json.loads(log.json_path.read_text(encoding="utf-8"))
    assert record["features"] == features
    assert record["metrics"]["test_acc"] == 0.91
    assert record["comment"] == "baseline with timing"
    assert record["artifacts"]["confusion_matrix"].endswith("cm.png")
    assert "python" in record["environment"]


def test_runlogger_allocates_increasing_ids(tmp_path):
    first = runlog.RunLogger(tmp_path, name="pid")
    first.start({"a": 1})
    second = runlog.RunLogger(tmp_path, name="pid")
    assert second.start({"a": 2}) == 2


def test_logging_before_start_is_an_error(tmp_path):
    log = runlog.RunLogger(tmp_path)
    with pytest.raises(RuntimeError):
        log.log_metrics(acc=1.0)


def test_sequences_and_paths_are_flattened(tmp_path):
    csv_path = tmp_path / "a.csv"
    runlog.append_row(csv_path, {"run_id": 1, "files": runlog._flatten(["x.root", "y.root"])})
    assert _read(csv_path)[0]["files"] == "x.root;y.root"
