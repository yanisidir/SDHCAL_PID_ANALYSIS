"""Focused tests for versioned and legacy model bundles."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from sdhcal import artifacts

REPO = Path(__file__).resolve().parents[1]


class LinearProbabilityModel:
    """Small deterministic estimator that remains importable after unpickling."""

    def predict_proba(self, values):
        probability = np.asarray(values, dtype=float)[:, 0] / 10.0
        return np.column_stack((1.0 - probability, probability))


class LinearEnergyModel:
    def predict(self, values):
        values = np.asarray(values, dtype=float)
        return values[:, 0] + 2.0 * values[:, 1]


class ShiftTarget:
    def inverse(self, values):
        return np.asarray(values, dtype=float) + 0.25


def _load_predictor():
    spec = importlib.util.spec_from_file_location("r1_predict", REPO / "tools/predict.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load tools/predict.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _legacy_payload() -> dict:
    return {
        "model": LinearProbabilityModel(),
        "features": ["Thr1", "Density"],
        "particles": ["pi", "proton"],
        "label_to_pdg": {0: -211, 1: 2212},
    }


def test_load_legacy_bundle_without_inventing_metadata(tmp_path: Path):
    path = tmp_path / "legacy.joblib"
    original = _legacy_payload()
    joblib.dump(original, path)

    loaded = artifacts.load_bundle(path, kind="pid")

    assert loaded.keys() == original.keys()
    assert loaded["features"] == ["Thr1", "Density"]
    assert "schema_version" not in loaded
    assert artifacts.artifact_metadata(loaded) == {
        "schema_version": artifacts.LEGACY_SCHEMA_VERSION,
        "legacy": True,
        "features": ["Thr1", "Density"],
        "n_features": 2,
    }


def test_versioned_round_trip_preserves_payload_and_feature_order(tmp_path: Path):
    path = tmp_path / "versioned.joblib"
    payload = _legacy_payload()
    original_keys = set(payload)

    written = artifacts.save_bundle(
        path,
        payload,
        kind="pid",
        model_family="synthetic",
        preprocessing={"scale": np.float64(2.0)},
        training_counts={"train": np.int64(12)},
        configuration={"seed": 42, "directory": Path("inputs")},
        dependencies={"calibration": {"path": Path("calibration.json")}},
        provenance={"git_commit": "abc123", "git_dirty": "False"},
    )
    loaded = artifacts.load_bundle(path, kind="pid")

    assert written == path
    assert set(payload) == original_keys
    assert "schema_version" not in payload
    assert original_keys < set(loaded)
    assert loaded["schema_version"] == artifacts.CURRENT_SCHEMA_VERSION
    assert loaded["kind"] == "pid"
    assert loaded["model_family"] == "synthetic"
    assert loaded["features"] == ["Thr1", "Density"]
    assert loaded["training_counts"]["train"] == 12
    assert loaded["configuration"]["directory"] == Path("inputs")

    metadata = artifacts.artifact_metadata(loaded)
    assert metadata["configuration"]["directory"] == "inputs"
    assert metadata["preprocessing"]["scale"] == 2.0
    assert metadata["legacy"] is False


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ("bare estimator", "must be a mapping"),
        ({"features": ["Thr1"]}, "missing required key"),
        ({"model": object()}, "missing required key"),
        ({"model": object(), "features": "Thr1"}, "ordered sequence"),
        ({"model": object(), "features": [], "schema_version": 99}, "unsupported"),
        (
            {"model": object(), "features": [], "schema_version": 1, "kind": "pid"},
            "model_family",
        ),
        (
            {
                "model": object(),
                "features": [],
                "schema_version": 1,
                "kind": "other",
                "model_family": "test",
            },
            "unknown kind",
        ),
    ],
)
def test_malformed_artifacts_are_rejected(tmp_path: Path, payload, message: str):
    path = tmp_path / "malformed.joblib"
    joblib.dump(payload, path)
    with pytest.raises(artifacts.ArtifactError, match=message):
        artifacts.load_bundle(path)


def test_requested_kind_must_match_versioned_bundle(tmp_path: Path):
    path = tmp_path / "energy.joblib"
    artifacts.save_bundle(
        path,
        {"model": object(), "features": ["N1"]},
        kind="energy",
        model_family="synthetic",
    )
    with pytest.raises(artifacts.ArtifactError, match="expected a pid artifact"):
        artifacts.load_bundle(path, kind="pid")


def test_predictor_accepts_legacy_and_versioned_bundles_without_prediction_drift(tmp_path: Path):
    predictor = _load_predictor()
    legacy_path = tmp_path / "legacy.joblib"
    versioned_path = tmp_path / "versioned.joblib"
    payload = _legacy_payload()
    joblib.dump(payload, legacy_path)
    artifacts.save_bundle(
        versioned_path,
        payload,
        kind="pid",
        model_family="synthetic",
    )
    frame = pd.DataFrame({"Thr1": [1.0, 5.0, 9.0], "Density": [3.0, 2.0, 1.0]})

    before = predictor.predict_pid(joblib.load(legacy_path), frame, positive_pdg=2212)
    legacy_after = predictor.predict_pid(
        predictor.load_bundle(legacy_path, kind="pid"), frame, positive_pdg=2212
    )
    versioned_after = predictor.predict_pid(
        predictor.load_bundle(versioned_path, kind="pid"), frame, positive_pdg=2212
    )

    np.testing.assert_allclose(legacy_after, before, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(versioned_after, before, rtol=0.0, atol=0.0)


def test_energy_prediction_is_identical_through_central_loader(tmp_path: Path):
    predictor = _load_predictor()
    legacy_path = tmp_path / "energy_legacy.joblib"
    versioned_path = tmp_path / "energy_versioned.joblib"
    payload = {
        "model": LinearEnergyModel(),
        "target": ShiftTarget(),
        "features": ["N1", "N2"],
        "baseline": None,
    }
    joblib.dump(payload, legacy_path)
    artifacts.save_bundle(
        versioned_path,
        payload,
        kind="energy",
        model_family="synthetic",
    )
    frame = pd.DataFrame({"N1": [1.0, 2.0], "N2": [3.0, 4.0]})

    before = predictor.predict_energy(joblib.load(legacy_path), frame)
    legacy_after = predictor.predict_energy(predictor.load_bundle(legacy_path, kind="energy"), frame)
    versioned_after = predictor.predict_energy(predictor.load_bundle(versioned_path, kind="energy"), frame)

    np.testing.assert_allclose(legacy_after, before, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(versioned_after, before, rtol=0.0, atol=0.0)
