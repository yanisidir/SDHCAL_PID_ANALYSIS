"""R8 regression tests for PID-dependent energy reconstruction."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sdhcal import calib, coupling, metrics

REPO = Path(__file__).resolve().parents[1]


class RoutingClassifier:
    def predict_proba(self, values):
        proton = np.asarray(values)[:, 0]
        return np.column_stack([1.0 - proton, proton])


class OffsetRegressor:
    def __init__(self, offset):
        self.offset = offset

    def predict(self, values):
        return np.asarray(values)[:, 0] + self.offset


class ScaleTarget:
    def __init__(self, scale):
        self.scale = scale

    def inverse(self, values):
        return np.asarray(values) * self.scale


def _frame(n=80):
    particle = np.where(np.arange(n) % 2 == 0, -211, 2212)
    # Every fourth event is deliberately routed to the wrong class.
    predicted_proton = np.where(np.arange(n) % 4 < 2, particle == 2212, particle == -211).astype(float)
    return pd.DataFrame(
        {
            "eventNumber": np.arange(n),
            "particlePDG": particle,
            "primaryEnergy": np.linspace(10.0, 50.0, n),
            "pid_feature": predicted_proton,
            "energy_feature": np.linspace(8.0, 48.0, n),
            "N1": np.arange(n) % 5 + 1.0,
            "N2": 0.0,
            "N3": 0.0,
        }
    )


def _classifier_bundle():
    return {
        "model": RoutingClassifier(),
        "features": ["pid_feature"],
        "particles": ["pi", "proton"],
    }


def _regressors():
    return {
        "mixed": {"model": OffsetRegressor(5.0), "features": ["energy_feature"]},
        "pi": {"model": OffsetRegressor(1.0), "features": ["energy_feature"]},
        "proton": {
            "model": OffsetRegressor(10.0),
            "target": ScaleTarget(2.0),
            "features": ["energy_feature"],
        },
    }


def _load_study():
    path = REPO / "analysis/5_pid_energy/study.py"
    spec = importlib.util.spec_from_file_location("r8_coupling_study", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_none_true_and_predicted_pid_routing_are_preserved():
    frame = _frame(8)
    none, none_probability = coupling.identify(
        frame, coupling.Config(particles=["pi", "proton"], pid="none"), None
    )
    true, true_probability = coupling.identify(
        frame, coupling.Config(particles=["pi", "proton"], pid="true"), None
    )
    predicted, probability = coupling.identify(
        frame, coupling.Config(particles=["pi", "proton"], pid="bdt"), _classifier_bundle()
    )
    np.testing.assert_array_equal(none, np.zeros(8, dtype=int))
    np.testing.assert_array_equal(true, [0, 1, 0, 1, 0, 1, 0, 1])
    np.testing.assert_array_equal(predicted, frame["pid_feature"].to_numpy(dtype=int))
    assert none_probability is true_probability is None
    np.testing.assert_array_equal(probability[:, 1], frame["pid_feature"])


def test_inclusive_true_and_misidentified_regressor_routing_are_exact():
    frame = _frame(8)
    regressors = _regressors()
    inclusive = coupling.reconstruct_energy(
        frame,
        np.zeros(8, dtype=int),
        coupling.Config(particles=["pi", "proton"], pid="none", reco="bdt"),
        regressors,
    )
    true_labels = (frame["particlePDG"] == 2212).to_numpy(dtype=int)
    oracle = coupling.reconstruct_energy(
        frame,
        true_labels,
        coupling.Config(particles=["pi", "proton"], pid="true", reco="bdt"),
        regressors,
    )
    routed = coupling.reconstruct_energy(
        frame,
        frame["pid_feature"].to_numpy(dtype=int),
        coupling.Config(particles=["pi", "proton"], pid="bdt", reco="bdt"),
        regressors,
    )
    base = frame[["energy_feature"]].to_numpy(dtype=np.float32)[:, 0]
    np.testing.assert_array_equal(inclusive, base + 5.0)
    np.testing.assert_array_equal(oracle, np.where(true_labels == 0, base + 1.0, (base + 10.0) * 2.0))
    expected_routed = np.where(frame["pid_feature"] == 0, base + 1.0, (base + 10.0) * 2.0)
    np.testing.assert_array_equal(routed, expected_routed)


def test_chi2_none_and_true_scenarios_keep_the_historical_coefficient_routing(monkeypatch):
    frame = _frame(8)
    coefficients = {
        "pi": calib.Coefficients(
            "pi", np.array([1.0, 0, 0, 0, 0, 0, 0, 0, 0]), metadata={"fitted_on_train_split": True}
        ),
        "proton": calib.Coefficients(
            "proton", np.array([2.0, 0, 0, 0, 0, 0, 0, 0, 0]), metadata={"fitted_on_train_split": True}
        ),
    }
    monkeypatch.setattr(calib, "load", coefficients.__getitem__)
    none = coupling.reconstruct_energy(
        frame,
        np.zeros(len(frame), dtype=int),
        coupling.Config(particles=["pi", "proton"], pid="none", reco="chi2"),
        {},
    )
    true_label = (frame["particlePDG"] == 2212).to_numpy(dtype=int)
    oracle = coupling.reconstruct_energy(
        frame,
        true_label,
        coupling.Config(particles=["pi", "proton"], pid="true", reco="chi2"),
        {},
    )
    np.testing.assert_array_equal(none, frame["N1"])
    np.testing.assert_array_equal(oracle, frame["N1"] * np.where(true_label == 0, 1.0, 2.0))


def test_summary_confusion_bias_and_resolution_match_existing_definitions():
    frame = _frame()
    config = coupling.Config(particles=["pi", "proton"], pid="bdt", reco="bdt", core_method="full_rms")
    routing = frame["pid_feature"].to_numpy(dtype=int)
    energy = frame["primaryEnergy"].to_numpy() * 1.03
    results, confusion = coupling.summarise(frame, config, routing, energy)
    true = (frame["particlePDG"] == 2212).to_numpy(dtype=int)
    expected_confusion = np.zeros((2, 2), dtype=np.int64)
    for actual, predicted in zip(true, routing, strict=True):
        expected_confusion[actual, predicted] += 1
    core = metrics.core_bias_resolution(
        metrics.relative_residual(frame["primaryEnergy"].to_numpy(), energy), method="full_rms"
    )
    np.testing.assert_array_equal(confusion, expected_confusion)
    assert results["pid_accuracy"] == np.mean(true == routing)
    assert results["bias"] == core.bias
    assert results["resolution"] == core.sigma
    assert results["energy_scale"] == pytest.approx(1.03)


@pytest.mark.parametrize("pid", ["none", "bdt"])
def test_per_event_output_schema_and_values_are_preserved(pid):
    frame = _frame(4)
    config = coupling.Config(particles=["pi", "proton"], pid=pid)
    routing = frame["pid_feature"].to_numpy(dtype=int)
    probabilities = (
        _classifier_bundle()["model"].predict_proba(frame[["pid_feature"]]) if pid == "bdt" else None
    )
    energy = frame["primaryEnergy"].to_numpy() * 1.1
    table = coupling.event_table(frame, config, routing, probabilities, energy)
    expected = [
        "eventNumber",
        "PDG_true",
        "label_true_int",
        "label_true",
        "E_true_GeV",
        "E_pred_GeV",
        "dE_over_E",
        "label_pred_int",
        "label_pred",
    ]
    if pid == "bdt":
        expected += ["proba_pi", "proba_proton", "proba_pred"]
    assert list(table) == expected
    np.testing.assert_allclose(table["dE_over_E"], 0.1)
    if pid == "none":
        assert table["label_pred_int"].tolist() == [-1] * 4
        assert table["label_pred"].tolist() == ["inclusive"] * 4


def test_missing_route_is_not_silently_dropped():
    frame = _frame(4)
    with pytest.raises(RuntimeError, match="received no energy estimate"):
        coupling.reconstruct_energy(
            frame,
            np.array([0, 1, 2, 0]),
            coupling.Config(particles=["pi", "proton"], pid="bdt", reco="bdt"),
            _regressors(),
        )


def test_study_exports_the_extracted_scientific_services():
    study = _load_study()
    assert study.Config is coupling.Config
    assert study.identify is coupling.identify
    assert study.reconstruct_energy is coupling.reconstruct_energy
    assert study.summarise is coupling.summarise
    assert study.required_regressors is coupling.required_regressors
    assert study.needed_columns is coupling.needed_columns


def test_all_published_artifacts_still_load_through_coupling_service():
    paths = sorted((REPO / "models").glob("*.joblib"))
    if not paths:
        pytest.skip("no published tabular artifacts are available")
    for path in paths:
        kind = "pid" if path.name.startswith("pid_") else "energy"
        bundle = coupling.load_artifact(path, path.name, kind=kind)
        assert "model" in bundle
        assert list(bundle["features"])
