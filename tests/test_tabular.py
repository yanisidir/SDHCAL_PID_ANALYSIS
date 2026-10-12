"""R3 characterization tests for shared tabular data preparation."""

from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from sdhcal import paths, tabular

REPO = Path(__file__).resolve().parents[1]


def _load_script(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, REPO / relative)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {relative}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _legacy_sample(frame: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    return pd.concat(
        [
            group.sample(n=min(len(group), n), random_state=seed)
            for _, group in frame.groupby("particlePDG", sort=False)
        ],
        ignore_index=True,
    )


def _legacy_clean(frame: pd.DataFrame, required: list[str], *, reset_index: bool) -> pd.DataFrame:
    cleaned = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=required)
    return cleaned.reset_index(drop=True) if reset_index else cleaned


def _legacy_split(
    energy: np.ndarray,
    groups: np.ndarray,
    *,
    test_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0 < test_size < 1:
        raise ValueError("calibration-size must lie strictly between zero and one")
    last_error = None
    for n_bins in range(10, 0, -1):
        quantile = np.zeros(len(energy), dtype=int)
        for value in np.unique(groups):
            selected = groups == value
            edges = np.unique(np.quantile(energy[selected], np.linspace(0, 1, n_bins + 1)))
            quantile[selected] = np.digitize(energy[selected], edges[1:-1])
        strata = np.asarray([f"{kind}:{bin_}" for kind, bin_ in zip(groups, quantile, strict=True)])
        try:
            return train_test_split(
                np.arange(len(energy)),
                test_size=test_size,
                random_state=random_state,
                stratify=strata,
            )
        except ValueError as error:
            last_error = error
    raise ValueError(f"Cannot create the train/calibration split: {last_error}")


def _particle_frame(counts: tuple[int, int]) -> pd.DataFrame:
    pdgs = np.repeat([paths.PARTICLE_PDG["pi"], paths.PARTICLE_PDG["proton"]], counts)
    return pd.DataFrame(
        {
            "feature_b": np.arange(len(pdgs), dtype=float) + 0.5,
            "feature_a": np.arange(len(pdgs), dtype=float),
            "particlePDG": pdgs,
            "primaryEnergy": np.arange(len(pdgs), dtype=float) + 1.0,
            "eventNumber": np.arange(100, 100 + len(pdgs)),
        }
    )


def test_seed_python_numpy_preserves_both_legacy_random_streams():
    random.seed(73)
    np.random.seed(73)
    expected_python = [random.random() for _ in range(4)]
    expected_numpy = np.random.random(4)

    tabular.seed_python_numpy(73)

    assert [random.random() for _ in range(4)] == expected_python
    np.testing.assert_array_equal(np.random.random(4), expected_numpy)


def test_read_shard_particles_preserves_calls_columns_and_event_order(monkeypatch):
    calls: list[tuple[object, ...]] = []

    def fake_read(sample, particle, *, features, limit):
        calls.append((sample, particle, tuple(features), limit))
        pdg = paths.PARTICLE_PDG[particle]
        base = 10 if particle == "pi" else 20
        return pd.DataFrame(
            {
                "feature_b": [base + 0.2, base + 1.2],
                "feature_a": [base, base + 1],
                "particlePDG": [pdg, pdg],
                "eventNumber": [base, base + 1],
            }
        )

    monkeypatch.setattr(tabular.shards, "read_sample", fake_read)
    actual = tabular.read_shard_particles(
        "200k", ("proton", "pi"), features=("feature_b", "feature_a"), limit=3
    )

    assert calls == [
        ("200k", "proton", ("feature_b", "feature_a"), 3),
        ("200k", "pi", ("feature_b", "feature_a"), 3),
    ]
    assert actual.columns.tolist() == [
        "feature_b",
        "feature_a",
        "particlePDG",
        "eventNumber",
    ]
    assert actual["eventNumber"].tolist() == [20, 21, 10, 11]
    assert len(actual) == 4


@pytest.mark.parametrize("counts,n", [((5, 5), 3), ((7, 2), 4), ((1, 2), 5)])
def test_sampling_is_exactly_the_historical_algorithm(counts, n):
    original = _particle_frame(counts)
    snapshot = original.copy(deep=True)

    expected = _legacy_sample(original, n, seed=17)
    actual = tabular.sample_per_particle(original, n, seed=17)

    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    pd.testing.assert_frame_equal(original, snapshot, check_exact=True)
    assert actual.groupby("particlePDG", sort=False).size().tolist() == [
        min(counts[0], n),
        min(counts[1], n),
    ]


def test_sampling_is_deterministic_and_retains_first_group_order():
    frame = pd.concat([_particle_frame((0, 4)), _particle_frame((5, 0))], ignore_index=True)
    first = tabular.sample_per_particle(frame, 3, seed=9)
    second = tabular.sample_per_particle(frame, 3, seed=9)

    pd.testing.assert_frame_equal(first, second, check_exact=True)
    assert first["particlePDG"].drop_duplicates().tolist() == [
        paths.PARTICLE_PDG["proton"],
        paths.PARTICLE_PDG["pi"],
    ]


@pytest.mark.parametrize("reset_index", [False, True])
def test_cleaning_matches_nan_infinity_masks_without_mutation(reset_index):
    frame = pd.DataFrame(
        {
            "feature_b": [1.0, np.inf, 3.0, 4.0, 5.0],
            "feature_a": [5.0, 6.0, np.nan, -np.inf, 9.0],
            "unused": [0.0, 1.0, 2.0, 3.0, np.nan],
            "eventNumber": [40, 41, 42, 43, 44],
        },
        index=[10, 11, 12, 13, 14],
    )
    snapshot = frame.copy(deep=True)
    required = ["feature_b", "feature_a", "eventNumber"]

    expected = _legacy_clean(frame, required, reset_index=reset_index)
    actual = tabular.clean_event_rows(frame, required, reset_index=reset_index)

    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    pd.testing.assert_frame_equal(frame, snapshot, check_exact=True)
    assert actual["eventNumber"].tolist() == [40, 44]
    assert actual.columns.tolist() == frame.columns.tolist()


@pytest.mark.parametrize(
    ("counts", "test_size"),
    [((40, 40), 0.25), ((45, 15), 0.20), ((12, 8), 0.30)],
)
def test_energy_group_split_indices_are_exactly_legacy(counts, test_size):
    groups = np.repeat([0, 1], counts)
    energy = np.concatenate([np.linspace(1.0, 130.0, counts[0]), np.linspace(3.0, 90.0, counts[1])])

    expected = _legacy_split(energy, groups, test_size=test_size, random_state=42)
    actual = tabular.stratified_energy_split_indices(energy, groups, test_size=test_size, random_state=42)

    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])


def test_energy_group_split_preserves_insufficient_strata_failure():
    energy = np.array([5.0, 10.0, 5.0, 10.0])
    groups = np.array([0, 0, 1, 1])
    with pytest.raises(ValueError) as legacy_error:
        _legacy_split(energy, groups, test_size=0.25, random_state=42)
    with pytest.raises(ValueError) as shared_error:
        tabular.stratified_energy_split_indices(energy, groups, test_size=0.25, random_state=42)
    assert str(shared_error.value) == str(legacy_error.value)


@pytest.fixture(scope="module")
def trainers():
    return {
        "pid_mlp": _load_script("r3_pid_mlp", "analysis/3_pid/mlp/train.py"),
        "energy_mlp": _load_script("r3_energy_mlp", "analysis/4_energy/ml_mlp/train.py"),
        "pid_bdt": _load_script("r3_pid_bdt", "analysis/3_pid/bdt/train.py"),
        "energy_bdt": _load_script("r3_energy_bdt", "analysis/4_energy/ml_bdt/train.py"),
    }


def test_pid_mlp_preparation_preserves_order_labels_and_filtering(trainers, monkeypatch):
    source = _particle_frame((5, 4))
    source.loc[1, "feature_a"] = np.inf
    source.loc[7, "feature_b"] = np.nan
    monkeypatch.setattr(tabular, "read_shard_particles", lambda *args, **kwargs: source.copy())
    module = trainers["pid_mlp"]
    cfg = module.Config(
        feature_cols=("feature_b", "feature_a"),
        max_events_per_particle=4,
        smear_sigma_ns=None,
    )

    expected = _legacy_sample(source, 4, 31)
    required = [
        "feature_b",
        "feature_a",
        "particlePDG",
        "primaryEnergy",
        "eventNumber",
    ]
    expected = _legacy_clean(expected, required, reset_index=True)
    expected["label"] = (
        expected["particlePDG"]
        .map({paths.PARTICLE_PDG["pi"]: 0, paths.PARTICLE_PDG["proton"]: 1})
        .astype(int)
    )

    actual = module.load_sample("200k", cfg, 31)

    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    assert actual[["feature_b", "feature_a"]].columns.tolist() == [
        "feature_b",
        "feature_a",
    ]


def test_energy_mlp_preparation_preserves_energy_cut_and_order(trainers, monkeypatch):
    source = _particle_frame((4, 4))
    source.loc[0, "primaryEnergy"] = 0.0
    source.loc[4, "primaryEnergy"] = -1.0
    source.loc[6, "feature_a"] = np.inf
    monkeypatch.setattr(tabular, "read_shard_particles", lambda *args, **kwargs: source.copy())
    module = trainers["energy_mlp"]
    cfg = module.Config(
        feature_cols=("feature_b", "feature_a"),
        max_events_per_particle=None,
        smear_sigma_ns=None,
    )

    required = [
        "feature_b",
        "feature_a",
        "primaryEnergy",
        "particlePDG",
        "eventNumber",
    ]
    expected = _legacy_clean(source, required, reset_index=False)
    expected = expected[expected["primaryEnergy"] > 0].reset_index(drop=True)

    actual = module.load_sample("discret", "mix-pi-proton", cfg, 19)

    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    assert actual["eventNumber"].tolist() == expected["eventNumber"].tolist()


@pytest.mark.parametrize("trainer", ["pid_mlp", "energy_mlp"])
def test_mlp_calibration_wrappers_preserve_exact_indices(trainers, trainer):
    module = trainers[trainer]
    counts = (40, 40)
    frame = _particle_frame(counts)
    cfg = module.Config(calibration_size=0.25, random_state=23)
    if trainer == "pid_mlp":
        frame["label"] = (frame["particlePDG"] == paths.PARTICLE_PDG["proton"]).astype(int)
        groups = frame["label"].to_numpy()
    else:
        groups = frame["particlePDG"].to_numpy()
    expected = _legacy_split(
        frame["primaryEnergy"].to_numpy(),
        groups,
        test_size=0.25,
        random_state=23,
    )

    actual = module.train_calibration_indices(frame, cfg)

    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])


@pytest.mark.parametrize("trainer", ["pid_bdt", "energy_bdt"])
def test_bdt_seed_wrapper_preserves_legacy_rng_behavior(trainers, trainer):
    module = trainers[trainer]
    random.seed(101)
    np.random.seed(101)
    expected = (random.random(), np.random.random())

    module.set_global_seeds(101)

    assert (random.random(), np.random.random()) == expected
