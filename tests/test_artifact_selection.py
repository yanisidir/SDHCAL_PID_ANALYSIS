"""R10 contracts for explicit and historical artifact selection."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from sdhcal import artifact_selection

REPO = Path(__file__).resolve().parents[1]


def test_explicit_run_selection_never_matches_a_run_id_substring(tmp_path: Path):
    run_one = tmp_path / "model_1_seed42.joblib"
    run_eleven = tmp_path / "model_11_seed42.joblib"
    run_one.write_bytes(b"one")
    run_eleven.write_bytes(b"eleven")

    selected = artifact_selection.select_run_artifact(tmp_path, "model_{run_id}_seed42.joblib", 1)

    assert selected == run_one.resolve()


def test_explicit_run_selection_rejects_missing_and_unsafe_requests(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="requested artifact"):
        artifact_selection.select_run_artifact(tmp_path, "model_{run_id}.joblib", 7)
    with pytest.raises(artifact_selection.ArtifactSelectionError, match="invalid run_id"):
        artifact_selection.select_run_artifact(tmp_path, "model_{run_id}.joblib", "../7")
    with pytest.raises(artifact_selection.ArtifactSelectionError, match="must contain"):
        artifact_selection.select_run_artifact(tmp_path, "model.joblib", 7)


def test_latest_selection_keeps_historical_mtime_semantics(tmp_path: Path):
    older = tmp_path / "model_1.joblib"
    newer = tmp_path / "model_2.joblib"
    older.write_bytes(b"old")
    newer.write_bytes(b"new")
    os.utime(older, (20, 20))
    os.utime(newer, (10, 10))

    assert artifact_selection.select_latest(str(tmp_path / "model_*.joblib")) == older


def test_publish_cli_supports_paths_with_spaces(tmp_path: Path):
    directory = tmp_path / "run artifacts"
    directory.mkdir()
    source = directory / "model_3.joblib"
    source.write_bytes(b"selected")
    destination = tmp_path / "published models" / "model.joblib"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPO / "src")

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools/publish_artifact.py"),
            "--directory",
            str(directory),
            "--template",
            "model_{run_id}.joblib",
            "--run-id",
            "3",
            "--destination",
            str(destination),
        ],
        cwd=REPO,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert destination.read_bytes() == b"selected"


def test_publish_cli_requires_latest_fallback_to_be_explicit(tmp_path: Path):
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPO / "src")
    completed = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools/publish_artifact.py"),
            "--directory",
            str(tmp_path),
            "--pattern",
            "*.joblib",
            "--destination",
            str(tmp_path / "published.joblib"),
        ],
        cwd=REPO,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "explicitly request --latest" in completed.stderr
