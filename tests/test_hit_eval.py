"""External evaluation of the hit-level networks, and the event key it rests on.

A wrong event key does not raise: it quietly pairs one model's score with another
model's event, and every figure downstream stays plausible. These tests pin the
translation, both spellings of it, and the guards that make a silent mismatch
impossible.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sdhcal import hit_eval, paths, shards

REPO = Path(__file__).resolve().parents[1]


def _load_importer():
    """tools/ is not a package; load the script by path."""
    spec = importlib.util.spec_from_file_location("import_predictions", REPO / "tools/import_predictions.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# The event key
# ---------------------------------------------------------------------------
def test_the_four_field_id_of_the_energy_network_folds_the_shard_in():
    importer = _load_importer()
    role, particle, number = importer.parse_event_id(
        "validation:pi:100k_pi-_5to100_discret_5GeV_part0_digitized:7"
    )
    assert (role, particle) == ("validation", "pi")
    # 5 GeV part0 is shard 1002 once the per-lot offset is applied; the offset
    # lands on the shard id, not on the folded number.
    expected_shard = shards.LOT_ID_BASE["discret"] + 2
    assert number == expected_shard * shards.SHARD_STRIDE + 7


def test_the_three_field_id_of_the_pid_networks_uses_the_shard_column():
    importer = _load_importer()
    role, particle, number = importer.parse_event_id("pion:1002:7")
    assert (role, particle) == ("validation", "pi")
    assert number == 1002 * shards.SHARD_STRIDE + 7


def test_both_spellings_agree_on_the_same_event():
    """The two families must land on the same key or they cannot be compared."""
    importer = _load_importer()
    four = importer.parse_event_id("validation:pi:100k_pi-_5to100_discret_100GeV_part1_digitized:4999")
    three = importer.parse_event_id("pion:1041:4999")
    assert four[2] == three[2]


def test_the_pion_is_renamed_to_the_pipeline_spelling():
    importer = _load_importer()
    assert importer.parse_event_id("pion:1002:0")[1] == "pi"
    assert importer.parse_event_id("proton:1002:0")[1] == "proton"
    assert paths.PARTICLE_PDG[importer.parse_event_id("pion:1002:0")[1]] == -211


@pytest.mark.parametrize(
    "event_id",
    ["pi:1002", "a:b:c:d:e", "validation:pi:unrecognised_name:3", "martian:1002:0"],
)
def test_an_unreadable_id_is_refused(event_id):
    importer = _load_importer()
    with pytest.raises(ValueError):
        importer.parse_event_id(event_id)


# ---------------------------------------------------------------------------
# Energies and predictions
# ---------------------------------------------------------------------------
def _hits(n_events=3, hits_per_event=4):
    rows = []
    for event in range(n_events):
        for _ in range(hits_per_event):
            rows.append({"event_id": f"pion:1002:{event}", "primaryEnergy": 5.0 * (event + 1)})
    return pd.DataFrame(rows)


def test_event_energies_returns_one_value_per_event():
    energies = hit_eval.event_energies(_hits())
    assert energies == {"pion:1002:0": 5.0, "pion:1002:1": 10.0, "pion:1002:2": 15.0}


def test_two_energies_under_one_event_id_are_refused():
    """That would mean two events were merged under one identifier."""
    frame = _hits(n_events=1)
    frame.loc[0, "primaryEnergy"] = 99.0
    with pytest.raises(ValueError, match="several energies"):
        hit_eval.event_energies(frame)


def test_a_table_without_the_energy_column_says_how_to_fix_it():
    with pytest.raises(KeyError, match="export_hits"):
        hit_eval.event_energies(pd.DataFrame({"event_id": ["pion:1002:0"]}))


def test_save_predictions_flips_the_proton_score_to_the_pion(tmp_path):
    """The networks score the proton; the benchmark quotes pion efficiency."""
    path = hit_eval.save_predictions(
        tmp_path / "external.npz",
        event_ids=["pion:1002:0", "pion:1002:1"],
        probability_proton=np.array([0.2, 0.9]),
        energies={"pion:1002:0": 5.0, "pion:1002:1": 10.0},
    )
    payload = np.load(path, allow_pickle=True)
    assert payload["y_score"] == pytest.approx([0.8, 0.1])
    assert payload["primary_energy"] == pytest.approx([5.0, 10.0])


def test_scores_and_ids_falling_out_of_step_are_refused(tmp_path):
    with pytest.raises(ValueError, match="out of step"):
        hit_eval.save_predictions(
            tmp_path / "external.npz",
            event_ids=["pion:1002:0"],
            probability_proton=np.array([0.2, 0.9]),
            energies={"pion:1002:0": 5.0},
        )


def test_an_event_without_an_energy_is_refused(tmp_path):
    with pytest.raises(KeyError, match="no energy"):
        hit_eval.save_predictions(
            tmp_path / "external.npz",
            event_ids=["pion:1002:0"],
            probability_proton=np.array([0.2]),
            energies={},
        )
