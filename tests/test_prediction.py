"""R8 numerical contracts for reusable prediction services."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sdhcal import artifacts, calib, prediction
from sdhcal.smearing import TimeSmearing, smear_time_columns

REPO = Path(__file__).resolve().parents[1]


class AddScaler:
    def transform(self, values):
        return np.asarray(values) + 3.0


class ProbabilityModel:
    def predict_proba(self, values):
        values = np.asarray(values)
        probability = values[:, 0] / 10.0
        return np.column_stack([1.0 - probability, probability])


class EnergyModel:
    def predict(self, values):
        values = np.asarray(values)
        return values[:, 0] + 2.0 * values[:, 1]


class ShiftTarget:
    def inverse(self, values):
        return np.asarray(values) + 0.25


def _legacy_apply_smearing(frame, bundle):
    recorded = bundle.get("smearing") or {}
    sigma = recorded.get("smear_sigma_ns", None)
    sigma = None if sigma in (None, "none") else float(sigma)
    columns = recorded.get("smear_columns")
    smearing = TimeSmearing(
        sigma_ns=sigma,
        columns=tuple(columns.split(";")) if isinstance(columns, str) else TimeSmearing().columns,
        seed=int(recorded.get("smear_seed", 42)),
    )
    return smear_time_columns(frame, smearing)


def _legacy_design(frame, bundle):
    features = list(bundle["features"])
    missing = [column for column in features if column not in frame.columns]
    if missing:
        raise SystemExit(f"the evaluation lot lacks feature(s): {', '.join(missing)}")
    values = frame[features].to_numpy(dtype=np.float32)
    scaler = bundle.get("scaler")
    return scaler.transform(values) if scaler is not None else values


def _legacy_predict_pid(bundle, frame, positive_pdg):
    label_to_pdg = {int(key): int(value) for key, value in (bundle.get("label_to_pdg") or {}).items()}
    index = next(label for label, pdg in label_to_pdg.items() if pdg == positive_pdg)
    return np.asarray(bundle["model"].predict_proba(_legacy_design(frame, bundle)))[:, index].astype(float)


def _legacy_predict_energy(bundle, frame):
    raw = bundle["model"].predict(_legacy_design(frame, bundle))
    target = bundle.get("target")
    value = np.asarray(target.inverse(raw) if target is not None else raw, dtype=float)
    if bundle.get("baseline") == "chi2_pi":
        coefficients = calib.load("pi")
        value = value * calib.ereco(
            frame["N1"].to_numpy(),
            frame["N2"].to_numpy(),
            frame["N3"].to_numpy(),
            coefficients.values,
        )
    return value


def _load_predictor():
    path = REPO / "tools/predict.py"
    spec = importlib.util.spec_from_file_location("r8_predict", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_feature_order_scaling_and_dtype_match_pre_r8_exactly():
    frame = pd.DataFrame({"second": [1.0, 2.0], "first": [4.0, 5.0]})
    bundle = {"features": ["first", "second"], "scaler": AddScaler()}
    expected = _legacy_design(frame, bundle)
    actual = prediction.design(frame, bundle)
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)


def test_smearing_seed_columns_and_values_match_pre_r8_exactly():
    frame = pd.DataFrame({"tMin": np.linspace(0, 1, 8), "tMax": np.linspace(1, 2, 8), "x": 1.0})
    bundle = {"smearing": {"smear_sigma_ns": 0.1, "smear_columns": "tMax;tMin", "smear_seed": 17}}
    expected = _legacy_apply_smearing(frame.copy(), bundle)
    actual = prediction.apply_smearing(frame.copy(), bundle)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    assert actual.attrs == expected.attrs


def test_pid_probability_column_and_event_order_match_pre_r8_exactly():
    frame = pd.DataFrame({"feature": [1.0, 7.0, 4.0]})
    bundle = {
        "model": ProbabilityModel(),
        "features": ["feature"],
        "label_to_pdg": {0: -211, 1: 2212},
    }
    expected = _legacy_predict_pid(bundle, frame, 2212)
    actual = prediction.predict_pid(bundle, frame, 2212)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual, np.array([0.1, 0.7, 0.4], dtype=np.float32))


def test_energy_target_inverse_matches_pre_r8_exactly():
    frame = pd.DataFrame({"N2": [3.0, 4.0], "N1": [1.0, 2.0]})
    bundle = {
        "model": EnergyModel(),
        "target": ShiftTarget(),
        "features": ["N1", "N2"],
    }
    expected = _legacy_predict_energy(bundle, frame)
    actual = prediction.predict_energy(bundle, frame)
    np.testing.assert_array_equal(actual, expected)


def test_chi2_prediction_preserves_species_routing_formula_and_order(monkeypatch):
    coefficients = {
        "pi": calib.Coefficients(
            "pi", np.array([1.0, 0, 0, 0, 0, 0, 0, 0, 0]), metadata={"fitted_on_train_split": True}
        ),
        "proton": calib.Coefficients(
            "proton", np.array([2.0, 0, 0, 0, 0, 0, 0, 0, 0]), metadata={"fitted_on_train_split": True}
        ),
    }
    monkeypatch.setattr(calib, "load", coefficients.__getitem__)
    frame = pd.DataFrame(
        {
            "particlePDG": [-211, 2212, -211, 2212],
            "N1": [1.0, 2.0, 3.0, 4.0],
            "N2": 0.0,
            "N3": 0.0,
        }
    )
    np.testing.assert_array_equal(prediction.predict_chi2(frame), [1.0, 4.0, 3.0, 8.0])


def test_all_published_tabular_artifacts_preserve_historical_inference(monkeypatch):
    paths = sorted((REPO / "models").glob("*.joblib"))
    if not paths:
        pytest.skip("no published tabular artifacts are available")
    monkeypatch.setenv("SDHCAL_PARAMS_SET", "2class")
    for path in paths:
        bundle = artifacts.load_bundle(path)
        features = list(bundle["features"])
        frame = pd.DataFrame(
            {feature: np.linspace(index + 1.0, index + 2.0, 4) for index, feature in enumerate(features)}
        )
        for name, values in {
            "N1": [10.0, 20.0, 30.0, 40.0],
            "N2": [5.0, 6.0, 7.0, 8.0],
            "N3": [1.0, 2.0, 3.0, 4.0],
        }.items():
            if name not in frame:
                frame[name] = values
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names")
            if path.name.startswith("pid_"):
                expected = _legacy_predict_pid(bundle, frame, positive_pdg=-211)
                actual = prediction.predict_pid(bundle, frame, positive_pdg=-211)
            else:
                expected = _legacy_predict_energy(bundle, frame)
                actual = prediction.predict_energy(bundle, frame)
        np.testing.assert_array_equal(actual, expected, err_msg=path.name)


def test_predict_script_exports_the_central_inference_services():
    script = _load_predictor()
    assert script.apply_smearing is prediction.apply_smearing
    assert script.design is prediction.design
    assert script.predict_pid is prediction.predict_pid
    assert script.predict_energy is prediction.predict_energy
    assert script.predict_chi2 is prediction.predict_chi2


def test_prediction_services_import_without_deep_learning_dependencies(tmp_path):
    blocker = tmp_path / "block_deep_learning.py"
    blocker.write_text(
        """
import builtins
original = builtins.__import__
def blocked(name, *args, **kwargs):
    if name == 'torch' or name.startswith('torch.') or name == 'torch_geometric' or name.startswith('torch_geometric.'):
        raise ImportError('blocked optional dependency')
    return original(name, *args, **kwargs)
builtins.__import__ = blocked
from sdhcal import coupling, prediction, prediction_io
print('ok')
""".strip()
        + "\n",
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPO / "src")
    completed = subprocess.run(
        [sys.executable, str(blocker)],
        cwd=REPO,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "ok"
