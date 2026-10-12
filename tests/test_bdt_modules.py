"""R5 regression tests for the modular tabular BDT implementations."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import asdict, replace
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from sdhcal import artifacts, energy_bdt, pid_bdt
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


def _legacy_classifier(cfg: pid_bdt.Config, seed: int):
    multiclass = (
        {"objective": "multiclass", "num_class": len(cfg.particles)}
        if len(cfg.particles) > 2
        else {"objective": "binary"}
    )
    import lightgbm as lgb

    return lgb.LGBMClassifier(
        **multiclass,
        learning_rate=cfg.learning_rate,
        n_estimators=cfg.n_estimators,
        num_leaves=cfg.num_leaves,
        max_depth=cfg.max_depth,
        reg_alpha=cfg.reg_alpha,
        reg_lambda=cfg.reg_lambda,
        min_child_samples=cfg.min_child_samples,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        class_weight=cfg.class_weight,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )


def _legacy_regressor(cfg: energy_bdt.Config, seed: int):
    import lightgbm as lgb

    return lgb.LGBMRegressor(
        objective="regression",
        learning_rate=cfg.learning_rate,
        n_estimators=cfg.n_estimators,
        num_leaves=cfg.num_leaves,
        max_depth=cfg.max_depth,
        reg_alpha=cfg.reg_alpha,
        reg_lambda=cfg.reg_lambda,
        min_child_samples=cfg.min_child_samples,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        random_state=seed,
        n_jobs=-1,
        metric="l2",
        verbose=-1,
    )


def test_config_defaults_feature_order_and_grids_are_preserved():
    pid = pid_bdt.Config()
    energy = energy_bdt.Config()
    assert list(pid.feature_cols) == [
        "Thr1",
        "Thr2",
        "Thr3",
        "Begin",
        "Radius",
        "Density",
        "NClusters",
        "Zbary",
        "Zrms",
        "PctHitsFirst10",
        "PlanesWithClusmore2",
        "AvgClustSize",
        "MaxClustSize",
        "lambda1",
        "lambda2",
        "tMin",
        "tMax",
        "tMean",
        "Nmax",
        "z0_fit",
        "Xmax",
        "lambda",
        "nTrackSegments",
        "eccentricity3D",
    ]
    assert list(energy.feature_cols) == [
        "Density",
        "NClusters",
        "Zbary",
        "AvgClustSize",
        "N3",
        "N2",
        "N1",
        "nHitsTotal",
        "sumThrTotal",
    ]
    assert asdict(pid)["random_state"] == 42
    assert asdict(energy)["target"] == "log"
    assert pid.param_grid == pid_bdt.Config().param_grid
    assert pid.param_grid is not pid_bdt.Config().param_grid
    assert energy.param_grid is not energy_bdt.Config().param_grid


@pytest.mark.parametrize("particles", [("pi", "proton"), ("pi", "proton", "kaon")])
def test_complete_pid_lightgbm_parameters_match_historical_constructor(particles):
    cfg = pid_bdt.Config(particles=particles)
    actual = pid_bdt.build_classifier(cfg, 17).get_params(deep=True)
    expected = _legacy_classifier(cfg, 17).get_params(deep=True)
    assert actual == expected
    assert actual["objective"] == ("multiclass" if len(particles) > 2 else "binary")
    assert actual["subsample"] == 0.8
    assert actual["subsample_freq"] == 0


def test_complete_energy_lightgbm_parameters_match_historical_constructor():
    cfg = energy_bdt.Config()
    actual = energy_bdt.build_regressor(cfg, 19).get_params(deep=True)
    assert actual == _legacy_regressor(cfg, 19).get_params(deep=True)
    assert actual["subsample"] == 0.8
    assert actual["subsample_freq"] == 0


def test_smote_is_inside_pid_cross_validation(monkeypatch):
    captured = {}

    class Search:
        def __init__(self, estimator, parameters, **kwargs):
            captured.update(estimator=estimator, parameters=parameters, **kwargs)
            self.best_params_ = {"clf__num_leaves": 31}
            self.best_score_ = 0.75

        def fit(self, x, y):
            captured["fit"] = (x.copy(), y.copy())
            return self

    monkeypatch.setattr(pid_bdt, "GridSearchCV", Search)
    cfg = replace(pid_bdt.Config(), n_estimators=2)
    x = pd.DataFrame({"x": np.arange(12, dtype=float)})
    y = np.array([0] * 8 + [1] * 4)
    assert pid_bdt.search_hyperparameters(x, y, cfg, 23) == {"num_leaves": 31}
    assert list(captured["estimator"].named_steps) == ["smote", "clf"]
    assert captured["estimator"].named_steps["smote"].random_state == 23
    assert captured["scoring"] == "f1_macro"
    assert captured["cv"].n_splits == 5
    assert captured["cv"].shuffle is True
    assert captured["cv"].random_state == 23


def test_smote_resamples_training_only(monkeypatch):
    import imblearn.over_sampling

    seen = {}

    class Smote:
        def __init__(self, random_state):
            seen["seed"] = random_state

        def fit_resample(self, x, y):
            seen["resampled_rows"] = list(x.index)
            return x.to_numpy(), y

    class Model:
        def fit(self, x, y, **kwargs):
            seen["fit_x"] = x.copy()
            seen["fit_y"] = y.copy()
            seen["fit_kwargs"] = kwargs
            return self

    monkeypatch.setattr(imblearn.over_sampling, "SMOTE", Smote)
    monkeypatch.setattr(pid_bdt, "build_classifier", lambda cfg, seed: Model())
    x_train = pd.DataFrame({"f": [1.0, 2.0, 3.0]}, index=[10, 11, 12])
    y_train = np.array([0, 0, 1])
    x_val = pd.DataFrame({"f": [8.0, 9.0]}, index=[20, 21])
    y_val = np.array([0, 1])
    cfg = pid_bdt.Config(particles=("pi", "proton"))
    pid_bdt.train(x_train, y_train, x_val, y_val, cfg, 29)
    assert seen["seed"] == 29
    assert seen["resampled_rows"] == [10, 11, 12]
    assert seen["fit_kwargs"]["eval_set"][1][0] is x_val
    np.testing.assert_array_equal(seen["fit_kwargs"]["eval_set"][1][1], y_val)
    assert seen["fit_kwargs"]["eval_names"] == ["train", "valid"]
    assert seen["fit_kwargs"]["eval_metric"] == "binary_logloss"


def _classification_frame():
    rng = np.random.default_rng(310)
    x = pd.DataFrame(rng.normal(size=(96, 4)), columns=list("abcd"))
    y = (x["a"].to_numpy() + 0.3 * x["b"].to_numpy() > 0).astype(int)
    return x.iloc[:72], y[:72], x.iloc[72:], y[72:]


def _regression_frame():
    rng = np.random.default_rng(410)
    x = pd.DataFrame(rng.normal(size=(96, 4)), columns=list("abcd"))
    y = np.exp(3.0 + 0.15 * x["a"].to_numpy() - 0.05 * x["b"].to_numpy())
    return x.iloc[:72], y[:72], x.iloc[72:], y[72:]


def test_pid_training_matches_pre_extraction_reference_with_one_thread(monkeypatch):
    import lightgbm as lgb

    original = pid_bdt.build_classifier

    def one_thread(cfg, seed):
        return original(cfg, seed).set_params(n_jobs=1)

    monkeypatch.setattr(pid_bdt, "build_classifier", one_thread)
    x_train, y_train, x_val, y_val = _classification_frame()
    cfg = replace(
        pid_bdt.Config(particles=("pi", "proton")),
        use_smote=False,
        n_estimators=20,
        early_stopping_rounds=5,
    )
    actual, actual_best = pid_bdt.train(x_train, y_train, x_val, y_val, cfg, 31)
    expected = _legacy_classifier(cfg, 31).set_params(n_jobs=1)
    expected.fit(
        x_train,
        y_train,
        eval_set=[(x_train, y_train), (x_val, y_val)],
        eval_names=["train", "valid"],
        eval_metric="binary_logloss",
        callbacks=[
            lgb.early_stopping(cfg.early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=200),
        ],
    )
    assert actual_best == {}
    np.testing.assert_array_equal(actual.predict_proba(x_val), expected.predict_proba(x_val))
    assert actual.best_iteration_ == expected.best_iteration_


def test_energy_training_matches_pre_extraction_reference_with_one_thread(monkeypatch):
    import lightgbm as lgb

    original = energy_bdt.build_regressor

    def one_thread(cfg, seed):
        return original(cfg, seed).set_params(n_jobs=1)

    monkeypatch.setattr(energy_bdt, "build_regressor", one_thread)
    x_train, y_train, x_val, y_val = _regression_frame()
    cfg = replace(energy_bdt.Config(), n_estimators=20, early_stopping_rounds=5)
    actual, actual_target, actual_best = energy_bdt.train(x_train, y_train, x_val, y_val, cfg, 37)
    expected_target = make_target(cfg.target, correct_bias=cfg.correct_target_bias)
    z_train = expected_target.forward(y_train)
    z_val = expected_target.forward(y_val)
    expected = _legacy_regressor(cfg, 37).set_params(n_jobs=1)
    expected.fit(
        x_train,
        z_train,
        eval_set=[(x_train, z_train), (x_val, z_val)],
        eval_names=["train", "valid"],
        eval_metric="l2",
        callbacks=[
            lgb.early_stopping(cfg.early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=200),
        ],
    )
    expected_target.fit_correction(y_val, expected.predict(x_val))
    assert actual_best == {}
    np.testing.assert_array_equal(actual.predict(x_val), expected.predict(x_val))
    assert actual_target.factor == expected_target.factor
    np.testing.assert_array_equal(
        actual_target.inverse(actual.predict(x_val)), expected_target.inverse(expected.predict(x_val))
    )


@pytest.mark.parametrize(
    ("particles", "probability", "auc_key"),
    [
        (("pi", "proton"), [[0.8, 0.2], [0.1, 0.9], [0.6, 0.4], [0.2, 0.8]], "auc"),
        (
            ("pi", "proton", "kaon"),
            [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.2, 0.7], [0.7, 0.2, 0.1]],
            "auc_ovr_macro",
        ),
    ],
)
def test_pid_evaluation_preserves_binary_and_multiclass_probability_order(particles, probability, auc_key):
    class Model:
        best_iteration_ = 4

        def predict_proba(self, _frame):
            return np.asarray(probability)

    truth = np.arange(len(probability)) % len(particles)
    results, matrix, returned, prediction = pid_bdt.evaluate(
        Model(), pd.DataFrame({"x": truth}), truth, pid_bdt.Config(particles=particles)
    )
    np.testing.assert_array_equal(returned, probability)
    np.testing.assert_array_equal(prediction, np.argmax(probability, axis=1))
    assert matrix.shape == (len(particles), len(particles))
    assert auc_key in results
    assert results["best_iteration"] == 4


def test_energy_evaluation_preserves_ratio_scale_contract():
    class Model:
        best_iteration_ = 6

        def predict(self, _frame):
            return np.array([1.0, 1.1, 0.9, 1.05])

    target = IdentityTarget()
    truth_ratio = np.array([1.0, 1.0, 1.0, 1.0])
    scale = np.array([20.0, 30.0, 40.0, 50.0])
    results, prediction = energy_bdt.evaluate(
        Model(), target, pd.DataFrame({"x": truth_ratio}), truth_ratio, energy_bdt.Config(), scale=scale
    )
    np.testing.assert_array_equal(prediction, np.array([20.0, 33.0, 36.0, 52.5]))
    assert results["n_test"] == 4
    assert results["best_iteration"] == 6
    assert results["target_bias_correction"] == 1.0


@pytest.mark.parametrize(
    ("mode", "expected"),
    [("pi", ("pi",)), ("mix-pi-proton", ("pi", "proton"))],
)
def test_energy_particle_mode_contract(mode, expected):
    assert energy_bdt.mode_particles(mode) == expected


def test_historical_script_symbols_are_stable_aliases():
    pid_script = _load_script("r5_pid_bdt", "analysis/3_pid/bdt/train.py")
    energy_script = _load_script("r5_energy_bdt", "analysis/4_energy/ml_bdt/train.py")
    assert pid_script.Config is pid_bdt.Config
    assert pid_script.train is pid_bdt.train
    assert pid_script.evaluate is pid_bdt.evaluate
    assert energy_script.Config is energy_bdt.Config
    assert energy_script.train is energy_bdt.train
    assert energy_script.evaluate is energy_bdt.evaluate


@pytest.mark.parametrize(
    ("filename", "kind"),
    [
        ("pid_bdt.joblib", "pid"),
        ("energy_bdt_pi.joblib", "energy"),
        ("energy_bdt_proton.joblib", "energy"),
        ("energy_bdt_mixed.joblib", "energy"),
        ("energy_bdt_ratio.joblib", "energy"),
    ],
)
def test_historical_bdt_artifacts_load_unchanged_and_predict_identically(filename, kind):
    path = REPO / "models" / filename
    if not path.exists():
        pytest.skip(f"optional historical artifact absent: {filename}")
    direct = joblib.load(path)
    central = artifacts.load_bundle(path, kind=kind)
    assert list(central) == list(direct)
    assert list(central["features"]) == list(direct["features"])
    assert central["model"].get_params(deep=True) == direct["model"].get_params(deep=True)
    rng = np.random.default_rng(510)
    frame = pd.DataFrame(
        rng.uniform(1.0, 10.0, size=(7, len(direct["features"]))),
        columns=direct["features"],
    )
    if kind == "pid":
        np.testing.assert_array_equal(
            central["model"].predict_proba(frame), direct["model"].predict_proba(frame)
        )
    else:
        np.testing.assert_array_equal(central["model"].predict(frame), direct["model"].predict(frame))
