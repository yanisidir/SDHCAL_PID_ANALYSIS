"""Bias and resolution: the estimators must recover a known Gaussian."""

from __future__ import annotations

import math

import numpy as np
import pytest

from sdhcal import metrics


def test_truncation_correction_matches_monte_carlo():
    """The analytic factor must match the empirical RMS of a truncated Gaussian."""
    rng = np.random.default_rng(0)
    sample = rng.normal(0.0, 1.0, size=2_000_000)
    for frac in (0.68, 0.9, 0.95):
        lo, hi = np.quantile(sample, [(1 - frac) / 2, 1 - (1 - frac) / 2])
        core = sample[(sample >= lo) & (sample <= hi)]
        empirical = float(np.std(core, ddof=1))
        assert empirical == pytest.approx(metrics.truncation_correction(frac), rel=2e-3)


def test_truncation_correction_known_value():
    """frac=0.9 understates sigma by ~21 %, which is the bug being corrected."""
    assert metrics.truncation_correction(0.9) == pytest.approx(0.78934, rel=1e-4)
    assert metrics.truncation_correction(1.0) == 1.0


@pytest.mark.parametrize("method", ["truncated_rms", "iterative_gauss", "full_rms"])
def test_recovers_injected_bias_and_resolution(method):
    """A 5 % bias with a 12 % spread must come back as 5 % and 12 %."""
    rng = np.random.default_rng(1)
    truth = rng.uniform(1.0, 130.0, size=200_000)
    pred = truth * (1.0 + rng.normal(0.05, 0.12, size=truth.size))

    est = metrics.core_bias_resolution(metrics.relative_residual(truth, pred), method=method)
    assert est.bias == pytest.approx(0.05, abs=0.002)
    assert est.sigma == pytest.approx(0.12, rel=0.01)
    assert est.n_used <= est.n_total


def test_truncated_rms_is_robust_to_tails():
    """A heavy tail must not inflate the core resolution the way a full RMS does."""
    rng = np.random.default_rng(2)
    core = rng.normal(0.0, 0.10, size=100_000)
    tail = rng.normal(0.0, 1.50, size=2_000)
    residual = np.concatenate([core, tail])

    truncated = metrics.core_bias_resolution(residual, method="truncated_rms")
    full = metrics.core_bias_resolution(residual, method="full_rms")
    assert truncated.sigma == pytest.approx(0.10, rel=0.08)
    assert full.sigma > 1.5 * truncated.sigma


def test_rms_about_zero_conflates_bias_and_resolution():
    """The historical metric is sqrt(bias^2 + sigma^2), hence not a resolution."""
    rng = np.random.default_rng(3)
    truth = rng.uniform(1.0, 130.0, size=200_000)
    bias, sigma = 0.20, 0.10
    pred = truth * (1.0 + rng.normal(bias, sigma, size=truth.size))

    historical = metrics.rms_relative_about_zero(truth, pred)
    assert historical == pytest.approx(math.hypot(bias, sigma), rel=0.01)

    est = metrics.core_bias_resolution(metrics.relative_residual(truth, pred))
    assert est.sigma == pytest.approx(sigma, rel=0.02)
    assert historical > 2 * est.sigma


def test_core_estimate_is_nan_below_min_entries():
    est = metrics.core_bias_resolution(np.zeros(5), min_entries=20)
    assert math.isnan(est.sigma) and est.n_used == 0


def test_binned_bias_resolution_tracks_an_energy_dependent_sigma():
    """sigma(E)/E ~ 0.5/sqrt(E) must be recovered bin by bin."""
    rng = np.random.default_rng(4)
    truth = rng.uniform(5.0, 100.0, size=400_000)
    expected = 0.5 / np.sqrt(truth)
    pred = truth * (1.0 + rng.normal(0.0, 1.0, size=truth.size) * expected)

    out = metrics.binned_bias_resolution(truth, pred, n_bins=8)
    assert out["center"].size == 8
    predicted = 0.5 / np.sqrt(out["center"])
    assert np.allclose(out["sigma"], predicted, rtol=0.06)
    assert np.allclose(out["bias"], 0.0, atol=0.01)
    assert np.all(out["sigma"] > 0) and np.all(np.diff(out["sigma"]) < 0)


def test_log_bins_cover_the_full_range():
    edges = metrics.energy_bins(1.0, 130.0, 10, log=True)
    assert edges.size == 11
    assert edges[0] == pytest.approx(1.0) and edges[-1] == pytest.approx(130.0)
    with pytest.raises(ValueError):
        metrics.energy_bins(0.0, 130.0, 10, log=True)


def test_relative_residual_drops_non_positive_truth():
    r = metrics.relative_residual(np.array([0.0, 10.0]), np.array([5.0, 11.0]))
    assert r.size == 1 and r[0] == pytest.approx(0.1)
    with pytest.raises(ValueError):
        metrics.relative_residual(np.zeros(3), np.zeros(4))
