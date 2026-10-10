"""Standard metrics for the model comparison.

The tests that matter here are the ones a wrong number would survive: a
resolution fit that does not recover the terms it was generated from, a rejection
quoted at the wrong efficiency, an alignment that silently compares two models on
two different event sets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sdhcal import benchmark


def _pid_frame(n=400, separation=1.0, seed=0):
    rng = np.random.default_rng(seed)
    pdg = np.where(np.arange(n) % 2 == 0, -211, 2212)
    score = rng.normal(np.where(pdg == 2212, separation, 0.0), 1.0)
    score = 1.0 / (1.0 + np.exp(-score))
    return pd.DataFrame(
        {
            "eventNumber": np.arange(n),
            "particlePDG": pdg,
            "primaryEnergy": rng.choice([10.0, 50.0, 100.0], n),
            "score": score,
        }
    )


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------
def test_a_missing_column_is_named():
    frame = _pid_frame().drop(columns="score")
    with pytest.raises(KeyError, match="lacks score"):
        benchmark.check_predictions(frame, "pid")


def test_duplicated_events_are_refused():
    frame = pd.concat([_pid_frame(4), _pid_frame(4)], ignore_index=True)
    with pytest.raises(ValueError, match="one per event"):
        benchmark.check_predictions(frame, "pid")


def test_non_finite_predictions_are_refused():
    frame = _pid_frame(10)
    frame.loc[0, "score"] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        benchmark.check_predictions(frame, "pid")


def test_align_restricts_to_the_shared_events():
    """Two models that dropped different events must not be compared as they are."""
    a = _pid_frame(100)
    b = _pid_frame(100).iloc[20:].reset_index(drop=True)
    out = benchmark.align({"a": a, "b": b}, "pid")
    assert len(out["a"]) == len(out["b"]) == 80
    assert out["a"]["eventNumber"].tolist() == out["b"]["eventNumber"].tolist()


def test_align_refuses_disjoint_samples():
    a = _pid_frame(10)
    b = _pid_frame(10)
    b["eventNumber"] += 1000
    with pytest.raises(ValueError, match="share no event"):
        benchmark.align({"a": a, "b": b}, "pid")


# ---------------------------------------------------------------------------
# Identification
# ---------------------------------------------------------------------------
def test_a_perfect_classifier_has_auc_one_and_infinite_rejection():
    labels = np.r_[np.zeros(50, dtype=int), np.ones(50, dtype=int)]
    curve = benchmark.roc_curve(labels, labels.astype(float))
    assert curve["auc"] == pytest.approx(1.0)
    assert np.isinf(benchmark.rejection_at_efficiency(curve, 1.0)["rejection"])


def test_a_coin_flip_has_auc_one_half():
    rng = np.random.default_rng(3)
    labels = rng.integers(0, 2, 4000)
    curve = benchmark.roc_curve(labels, rng.random(4000))
    assert curve["auc"] == pytest.approx(0.5, abs=0.03)


def test_rejection_is_quoted_at_the_requested_efficiency():
    frame = _pid_frame(4000, separation=1.5)
    labels = (frame["particlePDG"] == 2212).astype(int).to_numpy()
    curve = benchmark.roc_curve(labels, frame["score"].to_numpy())
    point = benchmark.rejection_at_efficiency(curve, 0.8)
    assert point["efficiency_signal"] >= 0.8 - 1e-9
    assert point["rejection"] == pytest.approx(1.0 / point["efficiency_background"])
    # A separated sample must reject better than a coin flip at the same point.
    assert point["rejection"] > 1.0


def test_roc_needs_both_classes():
    with pytest.raises(ValueError, match="both classes"):
        benchmark.roc_curve(np.ones(5, dtype=int), np.linspace(0, 1, 5))


def test_working_point_efficiency_and_purity():
    labels = np.array([1, 1, 1, 0, 0, 0])
    score = np.array([0.9, 0.8, 0.2, 0.7, 0.1, 0.05])
    point = benchmark.working_point(labels, score, 0.5)
    assert point["efficiency"] == pytest.approx(2 / 3)  # two of three signal kept
    assert point["purity"] == pytest.approx(2 / 3)  # two of three selected are signal


def test_efficiency_vs_energy_uses_the_fixed_energy_points():
    frame = _pid_frame(900, separation=2.0)
    table = benchmark.efficiency_vs_energy(frame, positive_pdg=2212, threshold=0.5)
    assert table["energy"].tolist() == [10.0, 50.0, 100.0]
    assert (table["n_signal"] > 0).all()
    assert ((table["efficiency_signal"] >= 0) & (table["efficiency_signal"] <= 1)).all()


# ---------------------------------------------------------------------------
# Energy
# ---------------------------------------------------------------------------
def test_resolution_fit_recovers_the_terms_it_was_generated_from():
    """The whole point of quoting a and b is that they are measurable."""
    energy = np.array([5.0, 10.0, 20.0, 40.0, 70.0, 100.0])
    a, b = 0.55, 0.08
    sigma = np.sqrt((a / np.sqrt(energy)) ** 2 + b**2)
    fit = benchmark.fit_resolution(energy, sigma)
    assert fit.stochastic == pytest.approx(a, rel=1e-6)
    assert fit.constant == pytest.approx(b, rel=1e-6)
    assert fit(energy) == pytest.approx(sigma, rel=1e-9)


def test_resolution_fit_handles_a_pure_stochastic_sample():
    energy = np.array([5.0, 10.0, 50.0, 100.0])
    sigma = 0.6 / np.sqrt(energy)
    fit = benchmark.fit_resolution(energy, sigma)
    assert fit.stochastic == pytest.approx(0.6, rel=1e-6)
    assert fit.constant == pytest.approx(0.0, abs=1e-6)


def test_resolution_fit_needs_two_points():
    with pytest.raises(ValueError, match="at least two"):
        benchmark.fit_resolution(np.array([10.0]), np.array([0.2]))


def _energy_frame(n_per_point=600, a=0.5, b=0.05, scale=1.0, seed=1):
    rng = np.random.default_rng(seed)
    points = np.array([5.0, 10.0, 20.0, 50.0, 100.0])
    truth, pred = [], []
    for e in points:
        sigma = np.sqrt((a / np.sqrt(e)) ** 2 + b**2)
        truth.append(np.full(n_per_point, e))
        pred.append(rng.normal(scale * e, sigma * e, n_per_point))
    return pd.DataFrame(
        {
            "eventNumber": np.arange(len(points) * n_per_point),
            "particlePDG": -211,
            "primaryEnergy": np.concatenate(truth),
            "energyPred": np.concatenate(pred),
        }
    )


def test_energy_summary_recovers_the_generated_terms():
    summary = benchmark.energy_summary(_energy_frame())
    assert summary.resolution.stochastic == pytest.approx(0.5, abs=0.06)
    assert summary.resolution.constant == pytest.approx(0.05, abs=0.03)
    assert summary.overall["bias"] == pytest.approx(0.0, abs=0.01)


def test_response_detects_a_miscalibrated_scale():
    """A 5 % scale error must show up as response 1.05, not hide in the width."""
    curve = benchmark.response_vs_energy(_energy_frame(scale=1.05))
    assert curve["response"].mean() == pytest.approx(1.05, abs=0.01)
    summary = benchmark.energy_summary(_energy_frame(scale=1.05))
    assert summary.overall["max_response_deviation"] > 0.03


def test_energy_summary_has_one_row_per_energy_point():
    summary = benchmark.energy_summary(_energy_frame())
    assert summary.curve["energy"].tolist() == [5.0, 10.0, 20.0, 50.0, 100.0]
