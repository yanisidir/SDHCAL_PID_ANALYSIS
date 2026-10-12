"""Unit tests for versioned configuration loading and resolution."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pytest

from sdhcal import artifacts, config

REPO = Path(__file__).resolve().parents[1]


def _document(**workflow_values):
    return {
        "schema_version": 1,
        "profile": "test",
        "dataset": {
            "params_set": "130k",
            "training_sample": "synthetic_train",
            "validation_sample": "synthetic_validation",
        },
        "workflows": {"pid_bdt": workflow_values},
    }


def test_load_valid_repository_configuration():
    document = config.load_config(REPO / "configs/three_class.json")
    values = config.workflow_values(document, "pid_bdt", params_set="130k")
    assert values["particles"] == ["pi", "proton", "kaon"]


def test_load_rejects_invalid_json(tmp_path: Path):
    path = tmp_path / "invalid.json"
    path.write_text('{"schema_version":', encoding="utf-8")
    with pytest.raises(config.ConfigError, match=r"invalid JSON.*line 1"):
        config.load_config(path)


def test_load_rejects_missing_file(tmp_path: Path):
    with pytest.raises(config.ConfigError, match="configuration file not found"):
        config.load_config(tmp_path / "missing.json")


def test_load_rejects_non_object_root(tmp_path: Path):
    path = tmp_path / "list.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(config.ConfigError, match="must be a JSON object"):
        config.load_config(path)


def test_workflow_document_rejects_unknown_keys():
    document = _document()
    document["typo"] = True
    with pytest.raises(config.ConfigError, match="unknown top-level configuration key"):
        config.workflow_values(document, "pid_bdt")


def test_workflow_document_rejects_dataset_environment_mismatch():
    with pytest.raises(config.ConfigError, match="expects SDHCAL_PARAMS_SET='130k'"):
        config.workflow_values(_document(), "pid_bdt", params_set="2class")


def test_merge_precedence_and_partial_configuration():
    defaults = {"seed": 42, "grid_search": False, "features": ["A", "B"]}
    merged = config.merge_config(
        defaults,
        {"seed": 7, "grid_search": True},
        {"seed": 11},
    )
    assert merged == {"seed": 11, "grid_search": True, "features": ["A", "B"]}


def test_merge_rejects_unknown_and_invalid_scientific_values():
    defaults = {"n_bins": 13, "learning_rate": 0.01, "grid_search": False}
    with pytest.raises(config.ConfigError, match="unknown configuration key.*n_bin"):
        config.merge_config(defaults, {"n_bin": 12})
    with pytest.raises(config.ConfigError, match="configuration.n_bins"):
        config.merge_config(defaults, {"n_bins": "13"})
    with pytest.raises(config.ConfigError, match="configuration.learning_rate"):
        config.merge_config(defaults, {"learning_rate": "0.1"})
    with pytest.raises(config.ConfigError, match="configuration.grid_search"):
        config.merge_config(defaults, {"grid_search": 1})


def test_merge_preserves_boolean_and_feature_order():
    defaults = {"grid_search": False, "feature_cols": ["A", "B"]}
    merged = config.merge_config(
        defaults,
        {"grid_search": True, "feature_cols": ["B", "A", "C"]},
    )
    assert merged["grid_search"] is True
    assert merged["feature_cols"] == ["B", "A", "C"]


def test_merge_handles_explicit_null_only_for_optional_values():
    defaults = {"smear_sigma_ns": 0.1, "n_bins": 13, "limit": None}
    merged = config.merge_config(
        defaults,
        {"smear_sigma_ns": None, "limit": 5},
        nullable_types={"smear_sigma_ns": (int, float), "limit": int},
    )
    assert merged["smear_sigma_ns"] is None
    assert merged["limit"] == 5
    with pytest.raises(config.ConfigError, match="invalid null value.*n_bins"):
        config.merge_config(defaults, {"n_bins": None})


def test_resolved_configuration_serializes_paths_numpy_and_tuples(tmp_path: Path):
    resolved = config.build_resolved_config(
        "pid_bdt",
        {
            "output": Path("output/run"),
            "seed": np.int64(42),
            "features": ("Thr1", "Density"),
            "smear_sigma_ns": None,
        },
        config_path=Path("configs/test.json"),
        cli_overrides=("seed",),
    )
    output = config.write_resolved_config(tmp_path / "resolved.json", resolved)
    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert loaded["values"] == {
        "output": "output/run",
        "seed": 42,
        "features": ["Thr1", "Density"],
        "smear_sigma_ns": None,
    }
    assert loaded["sources"]["config_file"] == "configs/test.json"
    assert loaded["sources"]["cli_overrides"] == ["seed"]


def test_resolved_configuration_is_valid_r1_artifact_metadata(tmp_path: Path):
    resolved = config.build_resolved_config("pid_bdt", {"features": ["Thr1"], "seed": 42})
    path = tmp_path / "model.joblib"
    artifacts.save_bundle(
        path,
        {"model": "synthetic", "features": ["Thr1"]},
        kind="pid",
        model_family="synthetic",
        configuration=resolved,
    )
    loaded = joblib.load(path)
    assert artifacts.artifact_metadata(loaded)["configuration"] == resolved
