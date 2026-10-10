"""Ingestion of the sharded parameter lots.

The three things this module fixes are the three the lots get wrong, so each has
a test: the species label is absent from the files, ``eventNumber`` is only unique
inside a shard, and ``touchesLastLayer`` was never written.
"""

from __future__ import annotations

import numpy as np
import pytest

from sdhcal import paths, shards, splits

uproot = pytest.importorskip("uproot")
awkward = pytest.importorskip("awkward")


# ---------------------------------------------------------------------------
# Shard identity
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("200k_pi-_1-130_0_digitized_params.root", 0),
        ("200k_pi-_1-130_7_digitized_params.root", 7),
        ("200k_proton_1-130_39_digitized_params.root", 39),
    ],
)
def test_shard_id_of_the_continuous_lots_is_the_number_in_the_name(name, expected):
    assert shards.shard_id("200k", name) == expected


@pytest.mark.parametrize(
    ("name", "offset"),
    [
        ("100k_pi-_5to100_discret_5GeV_part0_digitized_params.root", 2),
        ("100k_pi-_5to100_discret_5GeV_part1_digitized_params.root", 3),
        ("100k_pi-_5to100_discret_100GeV_part0_digitized_params.root", 40),
        ("100k_pi-_5to100_discret_100GeV_part1_digitized_params.root", 41),
    ],
)
def test_shard_id_of_the_grid_lots_folds_energy_and_part(name, offset):
    assert shards.shard_id("discret", name) == shards.LOT_ID_BASE["discret"] + offset


def _grid_ids():
    return [
        shards.shard_id("discret", f"100k_pi-_5to100_discret_{energy}GeV_part{part}_digitized_params.root")
        for energy in range(5, 105, 5)
        for part in (0, 1)
    ]


def test_every_shard_of_a_grid_lot_gets_a_distinct_id():
    """The fold must not collide, or two shards would share an event key."""
    ids = _grid_ids()
    assert len(set(ids)) == len(ids) == 40
    assert max(ids) * shards.SHARD_STRIDE < 2**31  # must still fit in ROOT's int32


def test_the_two_lots_share_no_shard_id():
    """The regression that would have cost 95 000 training events.

    The grid lot's raw ids (2..41) overlap the continuous lot's (0..39). Without a
    per-lot offset, a training event and a validation event answer to the same
    ``(particlePDG, eventNumber)`` key, and anything that compares or subtracts the
    two samples acts on events that have nothing to do with each other.
    """
    train = {shards.shard_id("200k", f"200k_pi-_1-130_{i}_digitized_params.root") for i in range(40)}
    assert not train & set(_grid_ids())


def test_the_roles_name_the_two_lots():
    assert shards.LOT_ROLES == {"train": "200k", "val": "discret"}
    assert set(shards.LOT_ROLES.values()) == set(shards.LOT_ID_BASE)


def test_an_unrecognised_shard_name_is_refused_not_numbered_by_position():
    with pytest.raises(ValueError, match="cannot read a shard id"):
        shards.shard_id("200k", "some_other_file_params.root")


# ---------------------------------------------------------------------------
# The event key
# ---------------------------------------------------------------------------
def test_global_event_number_separates_shards_that_both_start_at_zero():
    local = np.arange(5)
    a = shards.global_event_number(0, local)
    b = shards.global_event_number(1, local)
    assert not set(a.tolist()) & set(b.tolist())


def test_global_event_number_refuses_a_local_number_that_would_collide():
    with pytest.raises(ValueError, match="would collide"):
        shards.global_event_number(0, np.array([shards.SHARD_STRIDE]))


# ---------------------------------------------------------------------------
# Reading a shard
# ---------------------------------------------------------------------------
def _write_shard(path, *, n=4, event_numbers=None, k_per_event=None):
    """Write a file shaped like a delivered shard: PDG zeroed, eventNumber local."""
    event_numbers = np.arange(n) if event_numbers is None else np.asarray(event_numbers)
    if k_per_event is None:
        # One event touching the last layer, the others not.
        k_per_event = [[1, 2], [3, shards.LAST_LAYER], [4], [5, 6]][:n]
    payload = {name: np.linspace(1.0, 2.0, n) for name in shards.FEATURES}
    payload["eventNumber"] = event_numbers.astype("int32")
    payload[paths.SHARD_ENERGY_BRANCH] = np.linspace(10.0, 100.0, n)
    payload["K"] = awkward.Array([list(map(int, hits)) for hits in k_per_event])
    # As delivered: no species label at all.
    payload["particlePDG"] = np.zeros(n, dtype="int32")
    with uproot.recreate(path) as handle:
        handle[paths.SHARD_TREE] = payload
    return path


def test_read_shard_takes_the_label_from_the_lot(tmp_path):
    """The files carry particlePDG == 0, so the label can only come from the lot."""
    path = _write_shard(tmp_path / "200k_pi-_1-130_3_digitized_params.root")
    frame = shards.read_shard(path, "200k", "pi")
    assert frame["particlePDG"].unique().tolist() == [paths.PARTICLE_PDG["pi"]]

    frame = shards.read_shard(path, "200k", "proton")
    assert frame["particlePDG"].unique().tolist() == [paths.PARTICLE_PDG["proton"]]


def test_read_shard_makes_the_event_number_unique_across_shards(tmp_path):
    a = shards.read_shard(_write_shard(tmp_path / "200k_pi-_1-130_0_digitized_params.root"), "200k", "pi")
    b = shards.read_shard(_write_shard(tmp_path / "200k_pi-_1-130_1_digitized_params.root"), "200k", "pi")
    assert a["localEventNumber"].tolist() == b["localEventNumber"].tolist()  # as delivered
    assert not set(a["eventNumber"]) & set(b["eventNumber"])  # after folding
    assert b["shardIndex"].unique().tolist() == [1]


def test_event_keys_of_two_shards_do_not_collide(tmp_path):
    """The regression this module exists for: keys collapsing onto one shard."""
    import pandas as pd

    frames = [
        shards.read_shard(_write_shard(tmp_path / f"200k_pi-_1-130_{i}_digitized_params.root"), "200k", "pi")
        for i in range(3)
    ]
    merged = pd.concat(frames, ignore_index=True)
    assert splits.event_keys(merged).nunique() == len(merged)


def test_read_shard_recomputes_the_last_layer_flag(tmp_path):
    path = _write_shard(
        tmp_path / "200k_pi-_1-130_0_digitized_params.root",
        n=3,
        k_per_event=[[1, 2], [shards.LAST_LAYER], [shards.LAST_LAYER - 1]],
    )
    frame = shards.read_shard(path, "200k", "pi")
    assert frame["touchesLastLayer"].tolist() == [0, 1, 0]


def test_read_shard_renames_the_energy_branch(tmp_path):
    path = _write_shard(tmp_path / "200k_pi-_1-130_0_digitized_params.root")
    frame = shards.read_shard(path, "200k", "pi")
    assert "primaryEnergy" in frame.columns
    assert paths.SHARD_ENERGY_BRANCH not in frame.columns
    assert frame["primaryEnergy"].iloc[0] == pytest.approx(10.0)


def test_read_shard_names_a_missing_branch(tmp_path):
    path = tmp_path / "200k_pi-_1-130_0_digitized_params.root"
    with uproot.recreate(path) as handle:
        handle["tree"] = {"eventNumber": np.arange(3, dtype="int32")}
    with pytest.raises(KeyError, match="no energy branch"):
        shards.read_shard(path, "200k", "pi")


def test_read_shard_accepts_either_lot_format(tmp_path):
    """The delivered lots and the regenerated ones differ in three ways.

    Delivered: tree "tree", energy branch "energy", hits kept and no last-layer
    flag. Regenerated by extract: tree "paramsTree", branch
    "primaryEnergy", the flag written and the hits dropped. Both must read.
    """
    payload = {name: np.linspace(1.0, 2.0, 3) for name in shards.FEATURES}
    payload["eventNumber"] = np.arange(3, dtype="int32")
    payload["primaryEnergy"] = np.array([10.0, 20.0, 30.0])
    payload["touchesLastLayer"] = np.array([0, 1, 0], dtype="int32")

    path = tmp_path / "200k_pi-_1-130_0_digitized_params.root"
    with uproot.recreate(path) as handle:
        handle["paramsTree"] = payload

    frame = shards.read_shard(path, "200k", "pi")
    assert frame["primaryEnergy"].tolist() == [10.0, 20.0, 30.0]
    # Read from the branch, not recomputed: there is no K to recompute it from.
    assert frame["touchesLastLayer"].tolist() == [0, 1, 0]
    assert frame["particlePDG"].unique().tolist() == [paths.PARTICLE_PDG["pi"]]


def test_read_shard_refuses_a_file_with_neither_flag_nor_hits(tmp_path):
    payload = {name: np.linspace(1.0, 2.0, 3) for name in shards.FEATURES}
    payload["eventNumber"] = np.arange(3, dtype="int32")
    payload["energy"] = np.array([10.0, 20.0, 30.0])
    path = tmp_path / "200k_pi-_1-130_0_digitized_params.root"
    with uproot.recreate(path) as handle:
        handle["tree"] = payload
    with pytest.raises(KeyError, match="neither 'touchesLastLayer' nor 'K'"):
        shards.read_shard(path, "200k", "pi")


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------
def test_shard_files_names_the_variable_to_set(tmp_path, monkeypatch):
    monkeypatch.setenv("SDHCAL_SHARDS_DIR", str(tmp_path / "absent"))
    with pytest.raises(FileNotFoundError, match="SDHCAL_SHARDS_DIR"):
        shards.shard_files("200k", "pi")


def test_shard_files_are_ordered_by_shard_id(tmp_path, monkeypatch):
    monkeypatch.setenv("SDHCAL_SHARDS_DIR", str(tmp_path))
    lot = tmp_path / paths.SHARD_SAMPLES["200k"]["pi"]
    lot.mkdir(parents=True)
    for i in (11, 2, 7):
        _write_shard(lot / f"200k_pi-_1-130_{i}_digitized_params.root")
    found = shards.shard_files("200k", "pi")
    assert [shards.shard_id("200k", p) for p in found] == [2, 7, 11]


def test_an_unknown_sample_or_particle_is_refused():
    with pytest.raises(KeyError, match="unknown sample"):
        shards.shard_files("nonexistent", "pi")
    with pytest.raises(ValueError, match="no 'kaon' lot"):
        shards.shard_files("200k", "kaon")
