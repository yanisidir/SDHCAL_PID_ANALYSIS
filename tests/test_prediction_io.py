"""R8 prediction format, identifier and alignment contracts."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sdhcal import benchmark, prediction_io

REPO = Path(__file__).resolve().parents[1]


def _load_importer():
    path = REPO / "tools/import_predictions.py"
    spec = importlib.util.spec_from_file_location("r8_import_predictions", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("event_id", "expected"),
    [
        ("pion:1002:7", ("validation", "pi", 100200007)),
        (
            "validation:pi:100k_pi-_5to100_discret_5GeV_part0_digitized:7",
            ("validation", "pi", 100200007),
        ),
        ("proton:1002:3", ("validation", "proton", 100200003)),
    ],
)
def test_event_identifier_conversion_is_unchanged(event_id, expected):
    assert prediction_io.parse_event_id(event_id) == expected


@pytest.mark.parametrize("kind", ["pid", "energy"])
def test_npz_conversion_preserves_schema_values_and_row_order(tmp_path, kind):
    source = tmp_path / f"{kind}.npz"
    common = {
        "event_id": np.array(["pion:1002:9", "proton:1002:4"]),
    }
    if kind == "pid":
        np.savez_compressed(source, **common, y_score=[0.8, 0.1], primary_energy=[10.0, 20.0])
        expected_columns = list(benchmark.PID_COLUMNS)
        expected_values = [0.8, 0.1]
        value_column = "score"
    else:
        np.savez_compressed(source, **common, y_true=[10.0, 20.0], y_pred=[9.5, 21.0])
        expected_columns = list(benchmark.ENERGY_COLUMNS)
        expected_values = [9.5, 21.0]
        value_column = "energyPred"
    with np.load(source, allow_pickle=True) as payload:
        table, roles = prediction_io.npz_to_frame(payload, kind)
    assert list(table) == expected_columns
    assert table["eventNumber"].tolist() == [100200009, 100200004]
    assert table["particlePDG"].tolist() == [-211, 2212]
    assert table[value_column].tolist() == expected_values
    assert roles == {"validation"}


def test_prediction_csv_and_sidecar_bytes_keep_the_historical_contract(tmp_path):
    table = pd.DataFrame(
        {
            "eventNumber": [2, 1],
            "particlePDG": [2212, -211],
            "primaryEnergy": [20.0, 10.0],
            "score": [0.1, 0.8],
        }
    )
    metadata = {"name": "model", "kind": "pid", "n_events": 2, "n_train": None}
    path = prediction_io.write_prediction_table(tmp_path / "model.csv", table, "pid", metadata)
    restored = prediction_io.read_prediction_table(path, "pid")
    pd.testing.assert_frame_equal(restored, table, check_dtype=False)
    assert path.with_suffix(".json").read_text(encoding="utf-8") == json.dumps(metadata, indent=2)
    assert prediction_io.read_sidecar(path.with_suffix(".json")) == metadata


def test_alignment_preserves_intersection_sorting_and_rejects_duplicates():
    first = pd.DataFrame(
        {
            "eventNumber": [3, 1, 2],
            "particlePDG": [-211, -211, 2212],
            "primaryEnergy": [30.0, 10.0, 20.0],
            "score": [0.3, 0.1, 0.2],
        }
    )
    second = first.iloc[[2, 0]].reset_index(drop=True)
    aligned = benchmark.align({"first": first, "second": second}, "pid")
    expected_keys = [(-211, 3), (2212, 2)]
    for frame in aligned.values():
        assert list(map(tuple, frame[["particlePDG", "eventNumber"]].to_numpy())) == expected_keys
    duplicate = pd.concat([first, first.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicated"):
        benchmark.align({"duplicate": duplicate, "second": second}, "pid")


def test_find_newest_preserves_mtime_selection(tmp_path):
    first = tmp_path / "prediction_1.npz"
    second = tmp_path / "prediction_2.npz"
    first.touch()
    second.touch()
    os.utime(first, (1, 1))
    os.utime(second, (2, 2))
    assert prediction_io.find_newest(str(tmp_path / "prediction_*.npz")) == second


def test_import_script_exports_the_central_identifier_parser():
    importer = _load_importer()
    assert importer.parse_event_id is prediction_io.parse_event_id


def test_importer_distinguishes_requested_budget_from_recorded_split(tmp_path):
    importer = _load_importer()
    run_dir = tmp_path / "clustered"
    run_dir.mkdir()
    source = run_dir / "external_predictions.npz"
    source.touch()
    (run_dir / "run.json").write_text(
        json.dumps({"configuration": {"split_events": {"train": 84, "val": 12, "test": 24}}}),
        encoding="utf-8",
    )

    counts, record = importer._recorded_counts(source, n_train=100, n_evaluation=50)

    assert record == run_dir / "run.json"
    assert counts["requested_training_budget"] == 100
    assert counts["training_events"] == 84
    assert counts["validation_events"] == 12
    assert counts["test_events"] == 24
    assert counts["external_evaluation_events"] == 50


@pytest.mark.parametrize(
    ("relative", "kind"),
    [
        ("output/benchmark/pid/cnn.csv", "pid"),
        ("output/benchmark/energy/bdt_mixed.csv", "energy"),
    ],
)
def test_available_historical_prediction_tables_and_sidecars_load_without_drift(relative, kind):
    path = REPO / relative
    if not path.is_file():
        pytest.skip(f"historical prediction absent: {path}")
    expected = pd.read_csv(path)
    actual = prediction_io.read_prediction_table(path, kind)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    metadata = prediction_io.read_sidecar(path.with_suffix(".json"))
    assert metadata is not None
    assert metadata["kind"] == kind
    assert metadata["n_events"] == len(actual)


def test_historical_pid_benchmark_metrics_are_unchanged_after_central_read():
    path = REPO / "output/benchmark/pid/cnn.csv"
    if not path.is_file():
        pytest.skip(f"historical prediction absent: {path}")
    direct = pd.read_csv(path)
    central = prediction_io.read_prediction_table(path, "pid")
    expected = benchmark.pid_summary(direct, positive_pdg=-211)
    actual = benchmark.pid_summary(central, positive_pdg=-211)
    assert actual["auc"] == expected["auc"]
    assert actual["accuracy_at_0p5"] == expected["accuracy_at_0p5"]
    assert actual["working_points"] == expected["working_points"]
