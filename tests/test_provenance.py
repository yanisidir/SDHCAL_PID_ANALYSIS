"""Tests for small, explicit provenance helpers."""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path

import numpy as np

from sdhcal import provenance, runlog


def test_sha256_file_matches_known_digest(tmp_path: Path):
    path = tmp_path / "small.txt"
    path.write_bytes(b"abc")
    assert provenance.sha256_file(path, chunk_size=2) == hashlib.sha256(b"abc").hexdigest()


def test_file_hashing_is_explicit(tmp_path: Path, monkeypatch):
    path = tmp_path / "input.dat"
    path.write_bytes(b"detector input")

    def unexpected_hash(_path):
        raise AssertionError("sha256_file must not run unless explicitly requested")

    monkeypatch.setattr(provenance, "sha256_file", unexpected_hash)
    assert provenance.file_reference(path) == {
        "path": str(path),
        "size_bytes": len(b"detector input"),
    }


def test_file_reference_can_include_requested_hash(tmp_path: Path):
    path = tmp_path / "calibration.json"
    path.write_bytes(b"{}")
    reference = provenance.file_reference(path, include_sha256=True)
    assert reference == {
        "path": str(path),
        "size_bytes": 2,
        "sha256": hashlib.sha256(b"{}").hexdigest(),
    }


def test_to_jsonable_handles_scientific_and_path_values():
    converted = provenance.to_jsonable(
        {
            "path": Path("data/sample.root"),
            "integer": np.int64(7),
            "float": np.float32(1.5),
            "array": np.array([1, 2]),
            "tuple": (Path("a"), np.int32(3)),
        }
    )
    assert converted == {
        "path": "data/sample.root",
        "integer": 7,
        "float": 1.5,
        "array": [1, 2],
        "tuple": ["a", 3],
    }


def test_environment_fingerprint_keeps_stable_contract(monkeypatch):
    monkeypatch.setattr(
        provenance,
        "git_fingerprint",
        lambda directory=None: {"git_commit": "abc123", "git_dirty": "False"},
    )
    monkeypatch.setenv("SDHCAL_PARAMS_SET", "2class")
    fingerprint = provenance.environment_fingerprint()

    assert fingerprint["git_commit"] == "abc123"
    assert fingerprint["git_dirty"] == "False"
    assert fingerprint["env_SDHCAL_PARAMS_SET"] == "2class"
    assert fingerprint["python"]
    assert fingerprint["python_executable"]
    assert fingerprint["platform"]
    assert fingerprint["version_numpy"]
    assert datetime.fromisoformat(fingerprint["timestamp_utc"]).tzinfo is not None


def test_runlog_preserves_environment_fingerprint_public_interface():
    assert runlog.environment_fingerprint is provenance.environment_fingerprint
