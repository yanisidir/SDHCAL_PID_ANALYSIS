"""Splits and the disjointness guarantee of the held-out sample."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sdhcal import splits


def test_split_partitions_the_sample_exactly():
    s = splits.split_indices(10_000, test_size=0.2, val_size=0.1, seed=0)
    assert s.sizes() == {"train": 7_000, "val": 1_000, "test": 2_000}
    s.check(10_000)
    assert not set(s.train) & set(s.test)
    assert not set(s.train) & set(s.val)
    assert not set(s.val) & set(s.test)


def test_fractions_are_of_the_whole_sample():
    """The old code mixed conventions: test=0.2 then val=0.2 of the remainder."""
    s = splits.split_indices(1_000, test_size=0.2, val_size=0.2, seed=0)
    assert s.test.size == 200 and s.val.size == 200 and s.train.size == 600


def test_split_is_reproducible_and_seed_dependent():
    a = splits.split_indices(1_000, seed=1)
    b = splits.split_indices(1_000, seed=1)
    c = splits.split_indices(1_000, seed=2)
    assert np.array_equal(a.test, b.test)
    assert not np.array_equal(a.test, c.test)


def test_stratification_preserves_class_proportions():
    labels = np.repeat([0, 1, 2], [6_000, 3_000, 1_000])
    s = splits.split_indices(labels.size, test_size=0.2, val_size=0.1, seed=3, stratify=labels)
    for part in (s.train, s.val, s.test):
        counts = np.bincount(labels[part], minlength=3) / part.size
        assert np.allclose(counts, [0.6, 0.3, 0.1], atol=0.01)


def test_impossible_fractions_are_rejected():
    with pytest.raises(ValueError):
        splits.split_indices(100, test_size=0.7, val_size=0.4)
    with pytest.raises(ValueError):
        splits.split_indices(100, test_size=-0.1)


def test_stratify_length_is_checked():
    with pytest.raises(ValueError):
        splits.split_indices(100, stratify=np.zeros(99))


def test_event_keys_are_unique_per_particle_and_event():
    df = pd.DataFrame({"particlePDG": [-211, 2212, -211], "eventNumber": [1, 1, 2]})
    keys = splits.event_keys(df)
    assert list(keys) == ["-211:1", "2212:1", "-211:2"]
    assert keys.is_unique


def test_event_keys_refuses_to_guess():
    with pytest.raises(KeyError, match="eventNumber"):
        splits.event_keys(pd.DataFrame({"particlePDG": [1]}))


def test_assert_disjoint_passes_on_disjoint_samples():
    report = splits.assert_disjoint(["a:1", "a:2"], ["a:3"])
    assert report["n_common"] == 0 and report["n_b"] == 1


def test_assert_disjoint_fails_and_quantifies_the_leak():
    """A held-out file overlapping the training sample must stop the study."""
    train = [f"-211:{i}" for i in range(1000)]
    holdout = [f"-211:{i}" for i in range(990, 1010)]
    with pytest.raises(splits.NotDisjoint) as excinfo:
        splits.assert_disjoint(train, holdout)
    message = str(excinfo.value)
    assert "10 of the 20 events" in message and "50.00 %" in message


def test_overlap_report_counts_unique_events():
    report = splits.overlap_report(["x", "x", "y"], ["y", "z"], name_a="train", name_b="val")
    assert report["n_a"] == 2 and report["n_b"] == 2 and report["n_common"] == 1
    assert report["fraction_of_b"] == pytest.approx(0.5)
