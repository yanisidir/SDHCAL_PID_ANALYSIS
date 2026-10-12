"""Small exact numerical baselines, not validations of the physics model.

Known scientific interpretation issues are documented in docs/r0_baseline.md.
These tests only detect accidental numerical changes during R1--R10.
"""

from __future__ import annotations

import numpy as np
import pytest

from sdhcal import benchmark, calib, metrics
from sdhcal.target import IdentityTarget, LogTarget


def test_quadratic_calibration_exact_small_example():
    coefficients = np.arange(1.0, 10.0)
    reconstructed = calib.ereco(
        np.array([1.0, 2.0]),
        np.array([3.0, 4.0]),
        np.array([5.0, 6.0]),
        coefficients,
    )
    assert reconstructed.tolist() == [5907.0, 13020.0]
    assert calib.design_matrix(
        np.array([1.0, 2.0]), np.array([3.0, 4.0]), np.array([5.0, 6.0])
    ) @ coefficients == pytest.approx(reconstructed)


def test_full_rms_core_estimate_exact_small_example():
    residual = np.tile(np.array([-0.2, -0.1, 0.0, 0.1, 0.2]), 4)
    estimate = metrics.core_bias_resolution(residual, method="full_rms")
    assert estimate.bias == pytest.approx(0.0, abs=1e-15)
    assert estimate.sigma == pytest.approx(0.14509525002200235, rel=1e-14)
    assert (estimate.n_used, estimate.n_total, estimate.method) == (20, 20, "full_rms")


def test_current_truncation_factor_is_frozen_without_endorsing_the_metric_semantics():
    assert metrics.truncation_correction(0.9) == pytest.approx(0.7893132990990862, rel=1e-14)


def test_current_roc_threshold_order_and_auc_are_frozen():
    curve = benchmark.roc_curve(
        np.array([1, 0, 1, 0]),
        np.array([0.9, 0.8, 0.2, 0.1]),
    )
    assert curve["threshold"] == pytest.approx([np.inf, 0.9, 0.8, 0.2, 0.1])
    assert curve["efficiency_signal"] == pytest.approx([0.0, 0.5, 0.5, 1.0, 1.0])
    assert curve["efficiency_background"] == pytest.approx([0.0, 0.0, 0.5, 0.5, 1.0])
    assert curve["auc"] == pytest.approx(0.75)


def test_target_transforms_keep_the_current_round_trip():
    energy = np.array([1.0, 10.0, 100.0])
    log_target = LogTarget(correct_bias=False)
    identity = IdentityTarget()
    expected_forward = np.log(energy + log_target.epsilon)
    assert log_target.forward(energy) == pytest.approx(expected_forward)
    # The current inverse does not subtract the epsilon added by forward().
    assert log_target.inverse(log_target.forward(energy)) == pytest.approx(energy + log_target.epsilon)
    assert identity.inverse(identity.forward(energy)) == pytest.approx(energy)
