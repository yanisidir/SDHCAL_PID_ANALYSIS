"""Training in log space must not leave a systematic shortfall in energy."""

from __future__ import annotations

import numpy as np
import pytest

from sdhcal import metrics
from sdhcal.target import IdentityTarget, LogTarget, make_target


def _log_normal_sample(n=400_000, sigma=0.20, seed=0):
    """A model that is perfect in log space.

    The shower observables fix ``mu(x) = E[log E | x]``; the true energies
    compatible with those observables scatter log-normally around it, so
    ``E[E | x] = exp(mu + sigma^2 / 2)``. The log-trained model predicts ``mu``.
    """
    rng = np.random.default_rng(seed)
    mu = np.log(rng.uniform(1.0, 130.0, size=n))
    truth = np.exp(mu + rng.normal(0.0, sigma, size=n))
    return truth, mu, sigma


def test_naive_back_transform_underestimates_the_mean_energy():
    """exp(E[log E | x]) is the geometric mean: low by exp(-sigma^2/2)."""
    truth, pred_log, sigma = _log_normal_sample()
    naive = np.exp(pred_log)
    ratio = naive.mean() / truth.mean()
    assert ratio == pytest.approx(np.exp(-(sigma**2) / 2), rel=0.005)
    assert ratio < 0.985  # ~2 % too low, the offset seen on the linearity curve


def test_smearing_factor_restores_the_energy_scale():
    truth, pred_log, sigma = _log_normal_sample()
    target = LogTarget()
    factor = target.fit_correction(truth, pred_log)

    assert factor == pytest.approx(np.exp(sigma**2 / 2), rel=0.01)
    corrected = target.inverse(pred_log)
    assert corrected.mean() / truth.mean() == pytest.approx(1.0, abs=0.005)


def test_correction_is_a_pure_rescaling_so_resolution_is_untouched():
    """The factor multiplies every event, so the log-space spread is unchanged."""
    truth, pred_log, sigma = _log_normal_sample()
    target = LogTarget()
    target.fit_correction(truth, pred_log)
    corrected = target.inverse(pred_log)

    naive_spread = float(np.std(np.log(np.exp(pred_log) / truth)))
    corrected_spread = float(np.std(np.log(corrected / truth)))
    assert corrected_spread == pytest.approx(naive_spread, rel=1e-6)
    assert corrected_spread == pytest.approx(sigma, rel=0.02)


def test_relative_residual_bias_has_the_opposite_sign():
    """Documented nuance: <(E_pred-E_true)/E_true> moves the other way."""
    truth, pred_log, sigma = _log_normal_sample()
    naive = np.exp(pred_log)
    bias = metrics.core_bias_resolution(metrics.relative_residual(truth, naive), method="full_rms").bias
    assert bias == pytest.approx(np.exp(sigma**2 / 2) - 1, abs=0.005)
    assert bias > 0  # positive, while the ratio of means is below one


def test_inverse_without_fit_is_refused():
    """Forgetting the calibration step must fail loudly, not silently bias."""
    target = LogTarget()
    with pytest.raises(RuntimeError, match="fit_correction"):
        target.inverse(np.zeros(10))


def test_correction_can_be_disabled_explicitly():
    target = LogTarget(correct_bias=False)
    assert target.fit_correction(np.array([1.0, 2.0]), np.array([0.0, 0.7])) == 1.0
    assert np.allclose(target.inverse(np.array([0.0])), np.array([1.0]))


def test_forward_and_inverse_round_trip():
    target = LogTarget(correct_bias=False)
    y = np.array([1.0, 10.0, 130.0])
    assert np.allclose(target.inverse(target.forward(y)), y, rtol=1e-6)


def test_negative_energy_is_rejected():
    with pytest.raises(ValueError):
        LogTarget().forward(np.array([-1.0]))


def test_identity_target_is_a_no_op():
    target = IdentityTarget()
    y = np.array([1.0, 50.0])
    assert np.allclose(target.inverse(target.forward(y)), y)
    assert target.as_dict()["target"] == "linear"


def test_make_target_dispatch():
    assert isinstance(make_target("log"), LogTarget)
    assert isinstance(make_target("linear"), IdentityTarget)
    with pytest.raises(ValueError):
        make_target("sqrt")


def test_factor_is_logged():
    observed, pred_log, _ = _log_normal_sample(n=10_000)
    target = LogTarget()
    target.fit_correction(observed, pred_log)
    logged = target.as_dict()
    assert logged["target"] == "log" and logged["target_bias_correction"] > 1.0
