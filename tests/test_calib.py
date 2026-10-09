"""Calibration coefficients: one file, read identically by C++ and Python."""

from __future__ import annotations

import json

import numpy as np
import pytest

from sdhcal import calib


def _coeff(particle="pi", scale=1.0):
    values = np.array([4.3e-2, -3.5e-5, 1.9e-8, 1.07e-1, -6.4e-5, 1.2e-8, 1.8e-13, 7.4e-4, -3.0e-7])
    return calib.Coefficients(particle, values * scale, metadata={"n_events": 1000})


def test_ereco_matches_the_written_formula():
    par = np.array([1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0])
    assert calib.ereco(np.array([10.0]), np.array([5.0]), np.array([2.0]), par)[0] == pytest.approx(
        1.0 * 10 + 2.0 * 5 + 3.0 * 2
    )


def test_ereco_uses_the_quadratic_dependence_on_nhit():
    par = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    n1, n2, n3 = np.array([2.0]), np.array([3.0]), np.array([5.0])
    # alpha = a1 * N = 1 * 10, so E = 10 * N1 = 20
    assert calib.ereco(n1, n2, n3, par)[0] == pytest.approx(20.0)


def test_ereco_is_vectorised():
    par = np.arange(9, dtype=float)
    out = calib.ereco(np.ones(4), np.ones(4), np.ones(4), par)
    assert out.shape == (4,) and np.all(out == out[0])


def test_ereco_rejects_a_wrong_number_of_coefficients():
    with pytest.raises(ValueError):
        calib.ereco(np.ones(3), np.ones(3), np.ones(3), np.ones(8))


def test_save_then_load_round_trips(tmp_path):
    path = tmp_path / "chi2.json"
    calib.save({"pi": _coeff("pi")}, path=path)
    loaded = calib.load("pi", path)
    assert np.allclose(loaded.values, _coeff("pi").values)
    assert loaded.metadata["n_events"] == 1000
    assert "formula" in loaded.metadata


def test_save_preserves_other_particles(tmp_path):
    path = tmp_path / "chi2.json"
    calib.save({"pi": _coeff("pi")}, path=path)
    calib.save({"kaon": _coeff("kaon", 2.0)}, path=path)
    allc = calib.load_all(path)
    assert set(allc) == {"pi", "kaon"}
    assert np.allclose(allc["kaon"].values, _coeff("pi").values * 2.0)


def test_coefficients_are_stored_in_a_named_order(tmp_path):
    path = tmp_path / "chi2.json"
    calib.save({"pi": _coeff("pi")}, path=path)
    entry = json.loads(path.read_text(encoding="utf-8"))["particles"]["pi"]
    assert [k for k in entry if k in calib.PARAM_NAMES] == list(calib.PARAM_NAMES)


def test_errors_round_trip(tmp_path):
    path = tmp_path / "chi2.json"
    coeff = calib.Coefficients("pi", _coeff().values, errors=np.full(9, 1e-6), metadata={})
    calib.save({"pi": coeff}, path=path)
    assert np.allclose(calib.load("pi", path).errors, 1e-6)


def test_in_sample_flag_defaults_to_true(tmp_path):
    """Coefficients without a recorded train split must be flagged as in-sample."""
    path = tmp_path / "chi2.json"
    calib.save({"pi": _coeff("pi")}, path=path)
    assert calib.load("pi", path).fitted_in_sample is True

    calib.save(
        {"kaon": calib.Coefficients("kaon", _coeff().values, metadata={"fitted_on_train_split": True})},
        path=path,
    )
    assert calib.load("kaon", path).fitted_in_sample is False


def test_missing_file_explains_how_to_produce_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="EnergyReco.C"):
        calib.load_all(tmp_path / "absent.json")


def test_unknown_particle_lists_the_available_ones(tmp_path):
    path = tmp_path / "chi2.json"
    calib.save({"pi": _coeff("pi")}, path=path)
    with pytest.raises(KeyError, match="pi"):
        calib.load("muon", path)


def test_repository_calibration_file_is_valid():
    """The committed calibration file must load and expose the three particles."""
    allc = calib.load_all()
    assert set(allc) >= {"pi", "kaon", "proton"}
    for _particle, coeff in allc.items():
        assert coeff.values.shape == (9,)
        assert np.all(np.isfinite(coeff.values))
        assert "source" in coeff.metadata
