"""Time smearing: independent per column, reproducible, applied exactly once."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sdhcal.smearing import (
    SmearingAlreadyApplied,
    TimeSmearing,
    smear_time_columns,
    smeared_columns,
)


def _frame(n: int = 20_000) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    base = rng.uniform(0.0, 50.0, size=n)
    return pd.DataFrame({"tMin": base, "tMax": base + 5.0, "tSpread": base / 2.0, "Density": base})


def test_noise_is_independent_across_columns():
    """The original bug: the same seed per call gave all columns identical noise."""
    df = _frame()
    out = smear_time_columns(df, TimeSmearing(sigma_ns=0.1, columns=("tMin", "tMax", "tSpread")))

    d_min = out["tMin"] - df["tMin"]
    d_max = out["tMax"] - df["tMax"]
    d_spread = out["tSpread"] - df["tSpread"]

    # Identical noise would give exactly 1.0 for each of these correlations.
    assert abs(np.corrcoef(d_min, d_max)[0, 1]) < 0.03
    assert abs(np.corrcoef(d_min, d_spread)[0, 1]) < 0.03
    assert abs(np.corrcoef(d_max, d_spread)[0, 1]) < 0.03
    assert not np.allclose(d_min.to_numpy(), d_max.to_numpy())


def test_noise_has_the_requested_width_and_leaves_others_alone():
    df = _frame()
    out = smear_time_columns(df, TimeSmearing(sigma_ns=0.25, columns=("tMin",)))
    delta = (out["tMin"] - df["tMin"]).to_numpy()
    assert np.std(delta) == pytest.approx(0.25, rel=0.02)
    assert np.mean(delta) == pytest.approx(0.0, abs=0.01)
    assert np.array_equal(out["Density"].to_numpy(), df["Density"].to_numpy())


def test_reproducible_across_calls():
    df = _frame(500)
    cfg = TimeSmearing(sigma_ns=0.1, seed=1234)
    a = smear_time_columns(df, cfg)
    b = smear_time_columns(df, cfg)
    assert np.array_equal(a["tMin"].to_numpy(), b["tMin"].to_numpy())


def test_a_different_seed_gives_different_noise():
    df = _frame(500)
    a = smear_time_columns(df, TimeSmearing(sigma_ns=0.1, seed=1))
    b = smear_time_columns(df, TimeSmearing(sigma_ns=0.1, seed=2))
    assert not np.allclose(a["tMin"].to_numpy(), b["tMin"].to_numpy())


def test_second_application_is_refused():
    """The inference scripts smeared tMin twice, doubling the noise silently."""
    df = _frame(100)
    once = smear_time_columns(df, TimeSmearing(sigma_ns=0.1, columns=("tMin", "tMax", "tSpread")))
    with pytest.raises(SmearingAlreadyApplied, match="tMin"):
        smear_time_columns(once, TimeSmearing(sigma_ns=0.1, columns=("tMin",)), copy=False)


def test_disjoint_second_application_is_allowed():
    df = _frame(100)
    out = smear_time_columns(df, TimeSmearing(sigma_ns=0.1, columns=("tMin",)), copy=False)
    out = smear_time_columns(out, TimeSmearing(sigma_ns=0.1, columns=("tMax",)), copy=False)
    assert set(smeared_columns(out)) == {"tMin", "tMax"}


def test_disabled_smearing_is_a_no_op():
    df = _frame(100)
    out = smear_time_columns(df, TimeSmearing(sigma_ns=None))
    pd.testing.assert_frame_equal(out, df)
    assert smeared_columns(out) == {}


def test_missing_column_is_skipped_not_fatal():
    df = _frame(100).drop(columns=["tSpread"])
    out = smear_time_columns(df, TimeSmearing(sigma_ns=0.1, columns=("tMin", "tSpread")))
    assert set(smeared_columns(out)) == {"tMin"}


def test_negative_sigma_is_rejected():
    with pytest.raises(ValueError):
        smear_time_columns(_frame(10), TimeSmearing(sigma_ns=-0.1))


def test_config_is_loggable():
    cfg = TimeSmearing(sigma_ns=0.1, columns=("tMin", "tMax"), seed=7)
    assert cfg.as_dict() == {
        "smear_sigma_ns": 0.1,
        "smear_columns": "tMin;tMax",
        "smear_seed": 7,
    }
