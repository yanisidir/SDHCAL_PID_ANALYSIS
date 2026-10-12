"""R4 regression tests for the modular tabular MLP implementations."""

from __future__ import annotations

import copy
import importlib.util
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import log_loss, mean_squared_error, roc_auc_score
from sklearn.neural_network import MLPClassifier, MLPRegressor

from sdhcal import artifacts, energy_mlp, metrics, pid_mlp
from sdhcal.target import IdentityTarget, make_target

REPO = Path(__file__).resolve().parents[1]


def _load_script(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, REPO / relative)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {relative}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _legacy_pid_base(cfg: pid_mlp.Config, **overrides) -> MLPClassifier:
    parameters = {
        "hidden_layer_sizes": cfg.hidden_layer_sizes,
        "alpha": cfg.alpha,
        "batch_size": cfg.batch_size,
        "learning_rate_init": cfg.learning_rate_init,
        "activation": "relu",
        "solver": "adam",
        "shuffle": True,
        "random_state": cfg.random_state,
    }
    parameters.update(overrides)
    return MLPClassifier(**parameters)


def _legacy_energy_base(cfg: energy_mlp.Config, **overrides) -> MLPRegressor:
    parameters = {
        "hidden_layer_sizes": cfg.hidden_layer_sizes,
        "alpha": cfg.alpha,
        "batch_size": cfg.batch_size,
        "learning_rate_init": cfg.learning_rate_init,
        "activation": "relu",
        "solver": "adam",
        "shuffle": True,
        "random_state": cfg.random_state,
    }
    parameters.update(overrides)
    return MLPRegressor(**parameters)


def _legacy_pid_fit(x_train, y_train, x_calibration, y_calibration, cfg):
    model = _legacy_pid_base(cfg)
    history = {"train_loss": [], "calibration_loss": [], "calibration_auc": []}
    best_model = None
    best_loss = np.inf
    stale = 0
    rng = np.random.default_rng(cfg.random_state)
    for epoch in range(1, cfg.max_epochs + 1):
        order = rng.permutation(len(y_train))
        if epoch == 1:
            model.partial_fit(x_train[order], y_train[order], classes=np.array([0, 1]))
        else:
            model.partial_fit(x_train[order], y_train[order])
        probability = model.predict_proba(x_calibration)
        train_loss = float(model.loss_)
        calibration_loss = float(log_loss(y_calibration, probability, labels=[0, 1]))
        calibration_auc = float(roc_auc_score(y_calibration, probability[:, 1]))
        history["train_loss"].append(train_loss)
        history["calibration_loss"].append(calibration_loss)
        history["calibration_auc"].append(calibration_auc)
        if calibration_loss < best_loss - 1e-6:
            best_loss = calibration_loss
            best_model = copy.deepcopy(model)
            stale = 0
        else:
            stale += 1
            if stale >= cfg.patience:
                break
    return best_model, history


def _legacy_energy_fit(x_train, y_train, x_calibration, y_calibration, cfg):
    target = make_target(cfg.target, correct_bias=cfg.correct_target_bias)
    z_train = target.forward(y_train)
    z_calibration = target.forward(y_calibration)
    model = _legacy_energy_base(cfg)
    history = {"train_loss": [], "calibration_loss": []}
    best_model = None
    best_loss = np.inf
    stale = 0
    rng = np.random.default_rng(cfg.random_state)
    for _epoch in range(1, cfg.max_epochs + 1):
        order = rng.permutation(len(z_train))
        model.partial_fit(x_train[order], z_train[order])
        train_loss = float(model.loss_)
        prediction = model.predict(x_calibration)
        calibration_loss = float(mean_squared_error(z_calibration, prediction))
        history["train_loss"].append(train_loss)
        history["calibration_loss"].append(calibration_loss)
        if calibration_loss < best_loss - 1e-7:
            best_loss = calibration_loss
            best_model = copy.deepcopy(model)
            stale = 0
        else:
            stale += 1
            if stale >= cfg.patience:
                break
    target.fit_correction(y_calibration, best_model.predict(x_calibration))
    return best_model, target, history, best_loss


def _classification_data():
    rng = np.random.default_rng(120)
    x = rng.normal(size=(64, 4))
    y = (x[:, 0] + 0.4 * x[:, 1] > 0).astype(int)
    return x[:48], y[:48], x[48:], y[48:]


def _regression_data():
    rng = np.random.default_rng(230)
    x = rng.normal(size=(64, 4))
    y = 25.0 + 3.0 * x[:, 0] - 1.5 * x[:, 1]
    return x[:48], y[:48], x[48:], y[48:]


def _assert_network_parameters_equal(left, right):
    assert len(left.coefs_) == len(right.coefs_)
    for actual, expected in zip(left.coefs_, right.coefs_, strict=True):
        np.testing.assert_array_equal(actual, expected)
    for actual, expected in zip(left.intercepts_, right.intercepts_, strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_config_defaults_and_feature_order_are_preserved():
    pid = pid_mlp.Config()
    energy = energy_mlp.Config()

    assert asdict(pid)["hidden_layer_sizes"] == (128, 64, 32)
    assert asdict(pid)["max_epochs"] == 300
    assert list(pid.feature_cols)[:4] == ["Thr1", "Thr2", "Thr3", "Begin"]
    assert asdict(energy)["hidden_layer_sizes"] == (128, 64, 32)
    assert asdict(energy)["max_epochs"] == 500
    assert list(energy.feature_cols)[-3:] == ["N1", "N2", "N3"]


@pytest.mark.parametrize(
    ("module", "reference", "config"),
    [
        (pid_mlp, _legacy_pid_base, pid_mlp.Config()),
        (energy_mlp, _legacy_energy_base, energy_mlp.Config()),
    ],
)
def test_complete_model_parameters_match_the_historical_constructor(module, reference, config):
    assert module.base_model(config).get_params(deep=True) == reference(config).get_params(deep=True)


def test_hyperparameter_search_spaces_are_unchanged():
    expected = {
        "hidden_layer_sizes": [(64, 32), (128, 64, 32)],
        "alpha": [1e-5, 1e-4, 1e-3],
        "learning_rate_init": [5e-4, 1e-3],
    }
    assert expected == pid_mlp.hyperparameter_grid()
    assert expected == energy_mlp.hyperparameter_grid()
    assert pid_mlp.hyperparameter_grid() is not pid_mlp.hyperparameter_grid()
    assert energy_mlp.hyperparameter_grid() is not energy_mlp.hyperparameter_grid()


def test_historical_training_module_imports_are_compatibility_aliases():
    pid_train = _load_script("r4_pid_train", "analysis/3_pid/mlp/train.py")
    energy_train = _load_script("r4_energy_train", "analysis/4_energy/ml_mlp/train.py")

    assert pid_train.Config is pid_mlp.Config
    assert pid_train.base_model is pid_mlp.base_model
    assert pid_train.fit_model is pid_mlp.fit_model
    assert pid_train.evaluate is pid_mlp.evaluate
    assert energy_train.Config is energy_mlp.Config
    assert energy_train.base_model is energy_mlp.base_model
    assert energy_train.fit_model is energy_mlp.fit_model
    assert energy_train.evaluate is energy_mlp.evaluate


def test_pid_training_is_exactly_equivalent_on_deterministic_synthetic_data():
    x_train, y_train, x_calibration, y_calibration = _classification_data()
    cfg = pid_mlp.Config(
        hidden_layer_sizes=(7, 3),
        batch_size=8,
        max_epochs=5,
        patience=4,
        random_state=13,
    )

    expected_model, expected_history = _legacy_pid_fit(x_train, y_train, x_calibration, y_calibration, cfg)
    actual_model, actual_history, parameters = pid_mlp.fit_model(
        x_train, y_train, x_calibration, y_calibration, cfg
    )

    assert parameters == {}
    assert actual_history == expected_history
    _assert_network_parameters_equal(actual_model, expected_model)
    np.testing.assert_array_equal(
        actual_model.predict_proba(x_calibration),
        expected_model.predict_proba(x_calibration),
    )


def test_energy_training_is_exactly_equivalent_on_deterministic_synthetic_data():
    x_train, y_train, x_calibration, y_calibration = _regression_data()
    cfg = energy_mlp.Config(
        hidden_layer_sizes=(6, 3),
        batch_size=8,
        max_epochs=5,
        patience=4,
        random_state=17,
        target="log",
        correct_target_bias=True,
    )

    expected_model, expected_target, expected_history, expected_loss = _legacy_energy_fit(
        x_train, y_train, x_calibration, y_calibration, cfg
    )
    actual_model, actual_target, actual_history, parameters, actual_loss = energy_mlp.fit_model(
        x_train, y_train, x_calibration, y_calibration, cfg
    )

    assert parameters == {}
    assert actual_history == expected_history
    assert actual_loss == expected_loss
    assert actual_target.as_dict() == expected_target.as_dict()
    _assert_network_parameters_equal(actual_model, expected_model)
    np.testing.assert_array_equal(
        actual_target.inverse(actual_model.predict(x_calibration)),
        expected_target.inverse(expected_model.predict(x_calibration)),
    )


class _FlatClassifier:
    def __init__(self):
        self.loss_ = 1.0
        self.calls = 0
        self.classes = None

    def partial_fit(self, x, y, classes=None):
        self.calls += 1
        if classes is not None:
            self.classes = np.asarray(classes)
        return self

    def predict_proba(self, x):
        return np.tile([0.5, 0.5], (len(x), 1))


class _FlatRegressor:
    def __init__(self):
        self.loss_ = 1.0
        self.calls = 0

    def partial_fit(self, x, y):
        self.calls += 1
        return self

    def predict(self, x):
        return np.ones(len(x))


def test_pid_early_stopping_and_binary_class_contract(monkeypatch):
    monkeypatch.setattr(pid_mlp, "choose_hyperparameters", lambda *args: {})
    monkeypatch.setattr(pid_mlp, "base_model", lambda *args, **kwargs: _FlatClassifier())
    x = np.arange(16, dtype=float).reshape(8, 2)
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    cfg = pid_mlp.Config(max_epochs=10, patience=2)

    model, history, _ = pid_mlp.fit_model(x, y, x, y, cfg)

    assert len(history["train_loss"]) == 3
    assert model.calls == 1
    np.testing.assert_array_equal(model.classes, [0, 1])


def test_energy_early_stopping_contract(monkeypatch):
    monkeypatch.setattr(energy_mlp, "choose_hyperparameters", lambda *args: {})
    monkeypatch.setattr(energy_mlp, "base_model", lambda *args, **kwargs: _FlatRegressor())
    x = np.arange(16, dtype=float).reshape(8, 2)
    y = np.linspace(1.0, 8.0, 8)
    cfg = energy_mlp.Config(max_epochs=10, patience=2, target="linear", correct_target_bias=False)

    model, target, history, _, best_loss = energy_mlp.fit_model(x, y, x, y, cfg)

    assert len(history["train_loss"]) == 3
    assert model.calls == 1
    assert isinstance(target, IdentityTarget)
    assert best_loss == mean_squared_error(y, np.ones(len(y)))


class _FixedClassifier:
    def predict_proba(self, x):
        score = np.asarray(x[:, 0], dtype=float)
        return np.column_stack([1.0 - score, score])


class _FixedRegressor:
    def predict(self, x):
        return np.asarray(x[:, 0], dtype=float)


def test_pid_probabilities_and_metrics_preserve_prediction_order():
    x = np.array([[0.1], [0.8], [0.4], [0.7]])
    y = np.array([0, 1, 0, 1])

    result, probability, matrix = pid_mlp.evaluate(_FixedClassifier(), x, y)

    np.testing.assert_array_equal(probability[:, 1], x[:, 0])
    np.testing.assert_array_equal(matrix, [[2, 0], [0, 2]])
    assert result == {
        "accuracy": 1.0,
        "balanced_accuracy": 1.0,
        "auc": 1.0,
        "log_loss": pytest.approx(0.2990011586691898),
        "n_validation": 4,
        "efficiency_pi": 1.0,
        "purity_pi": 1.0,
        "efficiency_proton": 1.0,
        "purity_proton": 1.0,
    }


def test_energy_predictions_bias_and_resolution_match_existing_metrics():
    y = np.linspace(5.0, 100.0, 40)
    prediction = y * 1.04 + np.sin(np.arange(len(y)))
    x = prediction[:, None]
    cfg = energy_mlp.Config(core_method="full_rms")
    target = IdentityTarget()

    result, actual = energy_mlp.evaluate(_FixedRegressor(), target, x, y, cfg)
    expected_core = metrics.core_bias_resolution(
        metrics.relative_residual(y, prediction), method="full_rms", frac=cfg.core_frac
    )

    np.testing.assert_array_equal(actual, prediction)
    assert result["bias"] == expected_core.bias
    assert result["resolution"] == expected_core.sigma
    assert result["rms_rel_about_zero"] == metrics.rms_relative_about_zero(y, prediction)


@pytest.mark.parametrize("kind", ["linear", "log"])
def test_target_forward_inverse_contract_is_unchanged(kind):
    y = np.array([1.0, 10.0, 100.0])
    target = make_target(kind, correct_bias=False)
    target.fit_correction(y, target.forward(y))
    expected = y if kind == "linear" else y + target.epsilon
    np.testing.assert_allclose(target.inverse(target.forward(y)), expected, rtol=0, atol=1e-12)


def test_validation_bin_construction_is_preserved():
    np.testing.assert_array_equal(energy_mlp.validation_bins(np.array([10.0]), 13), [9.5, 10.5])
    np.testing.assert_array_equal(
        energy_mlp.validation_bins(np.array([5.0, 10.0, 20.0]), 13),
        [2.5, 7.5, 15.0, 25.0],
    )


def test_energy_artifact_round_trip_preserves_model_target_and_predictions(tmp_path):
    x_train, y_train, x_calibration, y_calibration = _regression_data()
    cfg = energy_mlp.Config(
        hidden_layer_sizes=(4,),
        batch_size=8,
        max_epochs=2,
        patience=2,
        random_state=29,
        target="log",
    )
    model, target, _, _, _ = energy_mlp.fit_model(x_train, y_train, x_calibration, y_calibration, cfg)
    path = tmp_path / "energy.joblib"
    artifacts.save_bundle(
        path,
        {"model": model, "target": target, "features": ["a", "b", "c", "d"]},
        kind="energy",
        model_family="sklearn_mlp",
    )

    loaded = artifacts.load_bundle(path, kind="energy")
    assert type(loaded["target"]).__module__ == "sdhcal.target"
    np.testing.assert_array_equal(
        loaded["target"].inverse(loaded["model"].predict(x_calibration)),
        target.inverse(model.predict(x_calibration)),
    )


def test_available_historical_pid_artifact_remains_readable_without_prediction_drift():
    path = REPO / "models/pid_mlp.joblib"
    if not path.exists():
        pytest.skip("repository-local historical PID MLP artifact is absent")
    direct = joblib.load(path)
    central = artifacts.load_bundle(path, kind="pid")
    features = list(direct["features"])
    frame = pd.DataFrame(
        np.linspace(0.0, 1.0, 3 * len(features)).reshape(3, len(features)),
        columns=features,
    )

    expected = direct["model"].predict_proba(direct["scaler"].transform(frame))
    actual = central["model"].predict_proba(central["scaler"].transform(frame))

    assert type(central["model"]).__module__.startswith("sklearn.")
    assert type(central["scaler"]).__module__.startswith("sklearn.")
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "relative",
    [
        "analysis/3_pid/mlp/train.py",
        "analysis/3_pid/mlp/evaluate.py",
        "analysis/3_pid/mlp/importance.py",
        "analysis/4_energy/ml_mlp/train.py",
    ],
)
@pytest.mark.parametrize("alternate_cwd", [False, True])
def test_mlp_cli_help_runs_from_repository_and_alternate_directory(relative, alternate_cwd, tmp_path):
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPO / "src")
    environment["MPLCONFIGDIR"] = str(tmp_path / "matplotlib")
    completed = subprocess.run(
        [sys.executable, str(REPO / relative), "--help"],
        cwd=tmp_path if alternate_cwd else REPO,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout.lower()
