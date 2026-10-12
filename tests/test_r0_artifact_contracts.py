"""R0 characterization of model artifacts and prediction files."""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from sdhcal import benchmark, hit_eval

REPO = Path(__file__).resolve().parents[1]
CONTRACTS = json.loads((REPO / "tests/fixtures/r0_contracts.json").read_text(encoding="utf-8"))


def _load_script(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, REPO / relative)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {relative}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _literal_payloads_written(relative: str) -> list[set[str]]:
    tree = ast.parse((REPO / relative).read_text(encoding="utf-8"))
    found: list[set[str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        payload = None
        if (
            isinstance(node.func.value, ast.Name)
            and node.func.value.id == "joblib"
            and node.func.attr == "dump"
            and node.args
        ):
            payload = node.args[0]
        elif (
            isinstance(node.func.value, ast.Name)
            and node.func.value.id == "artifacts"
            and node.func.attr == "save_bundle"
            and len(node.args) >= 2
        ):
            payload = node.args[1]
        if not isinstance(payload, ast.Dict):
            continue
        found.append(
            {
                key.value
                for key in payload.keys
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            }
        )
    return found


@pytest.mark.parametrize(
    ("schema", "source"),
    [
        ("pid_bdt", "analysis/3_pid/bdt/train.py"),
        ("pid_mlp", "analysis/3_pid/mlp/train.py"),
        ("energy_bdt", "analysis/4_energy/ml_bdt/train.py"),
        ("energy_mlp", "analysis/4_energy/ml_mlp/train.py"),
    ],
)
def test_tabular_trainers_write_the_characterized_bundle_keys(schema: str, source: str):
    expected = set(CONTRACTS["artifact_schemas"][schema])
    assert expected in _literal_payloads_written(source)


def test_current_predictor_accepts_the_minimal_characterized_bundle(tmp_path: Path):
    predictor = _load_script("r0_predict", "tools/predict.py")
    artifact = tmp_path / "bundle.joblib"
    joblib.dump({"model": "synthetic", "features": ["Thr1"]}, artifact)
    assert predictor.load_bundle(artifact) == {"model": "synthetic", "features": ["Thr1"]}


def test_prediction_table_column_contracts_are_ordered():
    assert list(benchmark.PID_COLUMNS) == CONTRACTS["prediction_schemas"]["pid_csv"]
    assert list(benchmark.ENERGY_COLUMNS) == CONTRACTS["prediction_schemas"]["energy_csv"]

    pid = pd.DataFrame([[1, -211, 10.0, 0.75]], columns=CONTRACTS["prediction_schemas"]["pid_csv"])
    energy = pd.DataFrame([[1, -211, 10.0, 9.5]], columns=CONTRACTS["prediction_schemas"]["energy_csv"])
    assert benchmark.check_predictions(pid, "pid") is pid
    assert benchmark.check_predictions(energy, "energy") is energy


def test_external_pid_npz_schema_is_frozen(tmp_path: Path):
    path = hit_eval.save_predictions(
        tmp_path / "external.npz",
        event_ids=["pion:1002:0", "proton:1002:1"],
        probability_proton=np.array([0.2, 0.8]),
        energies={"pion:1002:0": 5.0, "proton:1002:1": 10.0},
        labels=np.array([0, 1]),
    )
    with np.load(path, allow_pickle=True) as payload:
        assert payload.files == CONTRACTS["prediction_schemas"]["pid_external_npz_with_labels"]
        assert payload["y_score"] == pytest.approx([0.8, 0.2])


def test_energy_hit_prediction_npz_keyword_schema_is_frozen():
    tree = ast.parse((REPO / "analysis/4_energy/gnn/train.py").read_text(encoding="utf-8"))
    schemas = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "np"
            and node.func.attr == "savez_compressed"
        ):
            schemas.append([keyword.arg for keyword in node.keywords])
    assert CONTRACTS["prediction_schemas"]["energy_hit_npz"] in schemas
