"""R0 characterization of feature order, splits and event identities."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sdhcal import clustering, hit_pid, shards, splits

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "tests/fixtures"
CONTRACTS = json.loads((FIXTURES / "r0_contracts.json").read_text(encoding="utf-8"))


def _load_script(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, REPO / relative)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {relative}")
    module = importlib.util.module_from_spec(spec)
    # Dataclasses resolve their module through sys.modules while the class body
    # is executed, so register this unique test-only name first.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("name", "path"),
    [
        ("pid_bdt", "analysis/3_pid/bdt/train.py"),
        ("pid_mlp", "analysis/3_pid/mlp/train.py"),
        ("energy_bdt", "analysis/4_energy/ml_bdt/train.py"),
        ("energy_mlp", "analysis/4_energy/ml_mlp/train.py"),
    ],
)
def test_tabular_model_feature_order_matches_the_r0_snapshot(name: str, path: str):
    module = _load_script(f"r0_{name}", path)
    assert list(module.Config().feature_cols) == CONTRACTS["feature_order"][name]


def test_hit_and_cluster_feature_order_matches_the_r0_snapshot():
    assert list(hit_pid.FEATURE_NAMES) == CONTRACTS["feature_order"]["hit_pid"]
    assert list(clustering.CLUSTER_FEATURE_NAMES) == CONTRACTS["feature_order"]["cluster"]


def test_small_stratified_split_indices_are_frozen():
    events = pd.read_json(FIXTURES / "r0_events.json")
    labels = (events["particlePDG"] == 2212).astype(int).to_numpy()
    split = splits.split_indices(len(events), test_size=0.25, val_size=0.25, seed=7, stratify=labels)
    assert split.train.tolist() == [1, 3, 8, 9]
    assert split.val.tolist() == [0, 4, 7, 10]
    assert split.test.tolist() == [2, 5, 6, 11]


def test_event_keys_keep_species_and_local_number_in_order():
    events = pd.read_json(FIXTURES / "r0_events.json")
    assert splits.event_keys(events).tolist() == [
        "-211:0",
        "-211:1",
        "-211:2",
        "-211:3",
        "-211:4",
        "-211:5",
        "2212:0",
        "2212:1",
        "2212:2",
        "2212:3",
        "2212:4",
        "2212:5",
    ]


def test_global_event_number_contract_is_an_integer_shard_fold():
    local = np.array([0, 7, 4999])
    assert shards.global_event_number(1002, local).tolist() == [100200000, 100200007, 100204999]


def test_both_hit_event_id_spellings_map_to_the_same_pipeline_key():
    importer = _load_script("r0_import_predictions", "tools/import_predictions.py")
    energy_id = "validation:pi:100k_pi-_5to100_discret_5GeV_part0_digitized:7"
    pid_id = "pion:1002:7"
    assert importer.parse_event_id(energy_id) == ("validation", "pi", 100200007)
    assert importer.parse_event_id(pid_id) == ("validation", "pi", 100200007)
