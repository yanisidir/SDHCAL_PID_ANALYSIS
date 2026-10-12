"""R10 scientific release manifest and event-count contracts."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from sdhcal import manifest, runlog

REPO = Path(__file__).resolve().parents[1]
PROVENANCE = {
    "timestamp_utc": "2026-10-11T00:00:00+00:00",
    "python": "3.11.0",
    "git_commit": "abc1234",
    "git_dirty": "False",
}


def _files(tmp_path: Path) -> tuple[Path, Path, Path]:
    config = tmp_path / "resolved.json"
    model = tmp_path / "model.joblib"
    dataset = tmp_path / "small.root"
    config.write_text('{"schema_version": 1}\n', encoding="utf-8")
    model.write_bytes(b"model")
    dataset.write_bytes(b"dataset")
    return config, model, dataset


def test_manifest_round_trip_uses_portable_paths_and_stable_artifact_ids(tmp_path: Path):
    config, model, dataset = _files(tmp_path)
    path = tmp_path / "results" / "release 1" / "manifest.json"
    payload = manifest.build_manifest(
        path,
        release_id="release-1",
        experiment_type="pid-bdt",
        command=["python", "analysis/3_pid/bdt/train.py", "--seeds", "5"],
        resolved_configuration=config,
        datasets={"training": dataset},
        models={"pid": model},
        random_seeds=[42, 43, 44, 45, 46],
        event_counts={"requested_training_budget": 100, "training_events": 84},
        provenance=PROVENANCE,
    )
    manifest.write_manifest(path, payload)
    loaded = manifest.load_manifest(path, require_files=True, verify_hashes=True)

    model_reference = loaded["references"]["models"]["pid"]
    assert model_reference["path_base"] == "manifest"
    assert not Path(model_reference["path"]).is_absolute()
    assert manifest.resolve_reference(path, model_reference) == model.resolve()
    digest = hashlib.sha256(b"model").hexdigest()
    assert model_reference["artifact_id"] == f"sha256:{digest}"
    assert manifest.artifact_identifier(model) == f"sha256:{digest}"
    assert "sha256" not in loaded["references"]["datasets"]["training"]
    assert loaded["resolved_configuration"]["sha256"] == hashlib.sha256(config.read_bytes()).hexdigest()


def test_manifest_detects_missing_and_modified_artifacts(tmp_path: Path):
    config, model, _ = _files(tmp_path)
    path = tmp_path / "release" / "manifest.json"
    payload = manifest.build_manifest(
        path,
        release_id="release-2",
        experiment_type="energy-gnn",
        command="python analysis/4_energy/gnn/train.py",
        resolved_configuration=config,
        models={"energy": model},
        provenance=PROVENANCE,
    )
    manifest.write_manifest(path, payload)
    model.write_bytes(b"changed")
    with pytest.raises(manifest.ManifestError, match="SHA-256 mismatch"):
        manifest.load_manifest(path, require_files=True, verify_hashes=True)
    model.unlink()
    with pytest.raises(FileNotFoundError, match="missing"):
        manifest.load_manifest(path, require_files=True)


def test_requested_and_actual_counts_remain_distinct_and_missing_is_null(tmp_path: Path):
    _, model, _ = _files(tmp_path)
    path = tmp_path / "manifest.json"
    payload = manifest.build_manifest(
        path,
        release_id="counts",
        experiment_type="pid-cnn",
        command="python train.py",
        models={"pid": model},
        event_counts={"requested_training_budget": 9000, "training_events": 8400},
        provenance=PROVENANCE,
    )

    assert payload["event_counts"]["requested_training_budget"] == 9000
    assert payload["event_counts"]["training_events"] == 8400
    assert payload["event_counts"]["input_events"] is None
    with pytest.raises(manifest.ManifestError, match="non-negative"):
        manifest.build_manifest(
            path,
            release_id="bad",
            experiment_type="pid",
            command="python train.py",
            event_counts={"training_events": -1},
            provenance=PROVENANCE,
        )
    with pytest.raises(manifest.ManifestError, match="path separators"):
        manifest.build_manifest(
            path,
            release_id="../bad",
            experiment_type="pid",
            command="python train.py",
            provenance=PROVENANCE,
        )


def test_existing_hit_run_metadata_yields_only_recorded_actual_counts():
    counts = manifest.event_counts_from_run_record(
        {"configuration": {"split_events": {"train": 8400, "val": 1200, "test": 2400}}}
    )
    assert counts["training_events"] == 8400
    assert counts["validation_events"] == 1200
    assert counts["test_events"] == 2400
    assert counts["requested_training_budget"] is None
    assert manifest.event_counts_from_run_record({})["training_events"] is None


def test_runlogger_records_counts_without_changing_configuration(tmp_path: Path):
    log = runlog.RunLogger(tmp_path, name="synthetic")
    log.start({"seed": 42})
    log.log_event_counts(training_events=8, validation_events=2, input_events=None)
    payload = json.loads(log.json_path.read_text(encoding="utf-8"))
    assert payload["config"] == {"seed": 42}
    assert payload["event_counts"] == {
        "training_events": 8,
        "validation_events": 2,
        "input_events": None,
    }


def test_release_cli_creates_only_the_requested_result_directory(tmp_path: Path):
    config, model, _ = _files(tmp_path)
    run_record = tmp_path / "run.json"
    run_record.write_text(
        json.dumps({"configuration": {"split_events": {"train": 84, "val": 12, "test": 24}}}),
        encoding="utf-8",
    )
    root = tmp_path / "results with spaces"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPO / "src")
    completed = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools/create_release.py"),
            "--release-id",
            "test-release",
            "--experiment-type",
            "pid-bdt",
            "--command",
            "python train.py --seeds 5",
            "--results-root",
            str(root),
            "--config",
            str(config),
            "--model",
            f"pid={model}",
            "--run-record",
            str(run_record),
            "--seed",
            "42",
        ],
        cwd=REPO,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    path = root / "test-release" / "manifest.json"
    assert completed.returncode == 0, completed.stderr
    assert path.is_file()
    assert sorted(item.name for item in path.parent.iterdir()) == ["manifest.json"]
    payload = manifest.load_manifest(path, require_files=True, verify_hashes=True)
    assert payload["event_counts"]["training_events"] == 84
    assert payload["event_counts"]["validation_events"] == 12
    assert payload["references"]["outputs"]["run_record"]["artifact_id"].startswith("sha256:")


def test_release_cli_rejects_count_conflicts_with_run_record(tmp_path: Path):
    record = tmp_path / "run.json"
    record.write_text(json.dumps({"configuration": {"split_events": {"train": 84}}}), encoding="utf-8")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPO / "src")
    completed = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools/create_release.py"),
            "--release-id",
            "conflict",
            "--experiment-type",
            "pid-gnn",
            "--command",
            "python train.py",
            "--results-root",
            str(tmp_path / "results"),
            "--run-record",
            str(record),
            "--training-events",
            "83",
        ],
        cwd=REPO,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "conflicts with the run record value 84" in completed.stderr
