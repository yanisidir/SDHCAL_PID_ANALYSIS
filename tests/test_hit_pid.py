from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sdhcal.hit_pid import HitPreprocessor, load_binary_hits, split_events


def _write_hits(path, *, bad_threshold: bool = False) -> None:
    rows = []
    for event in range(20):
        for hit in range(3):
            rows.append(
                {
                    "eventNumber": event,
                    "x_coords": event + hit,
                    "y_coords": event - hit,
                    "z_coords": hit,
                    "time": 0.1 * hit,
                    "thr": 4 if bad_threshold and event == 0 and hit == 0 else hit + 1,
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_binary_hit_pipeline_has_disjoint_events_and_seven_features(tmp_path):
    pion = tmp_path / "pion.csv"
    proton = tmp_path / "proton.csv"
    _write_hits(pion)
    _write_hits(proton)

    frame = load_binary_hits(pion, proton, sigma_time=0.0, seed=7)
    splits = split_events(frame, seed=7)
    event_ids = {name: set(split["event_id"]) for name, split in splits.items()}

    assert event_ids["train"].isdisjoint(event_ids["val"])
    assert event_ids["train"].isdisjoint(event_ids["test"])
    assert event_ids["val"].isdisjoint(event_ids["test"])
    assert {0, 1} == set(splits["train"]["label"])

    preprocessor = HitPreprocessor(seed=7).fit(splits["train"])
    transformed = preprocessor.transform(splits["val"].iloc[:5])
    assert transformed.shape == (5, 7)
    assert transformed.dtype == np.float32


def test_invalid_threshold_is_rejected(tmp_path):
    pion = tmp_path / "pion.csv"
    proton = tmp_path / "proton.csv"
    _write_hits(pion, bad_threshold=True)
    _write_hits(proton)

    with pytest.raises(ValueError, match="invalid thr"):
        load_binary_hits(pion, proton, sigma_time=0.0, seed=7)
