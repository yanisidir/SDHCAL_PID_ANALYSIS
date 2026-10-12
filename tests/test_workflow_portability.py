"""Lightweight R10 integration checks for Make and Shell workflow portability."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _executable_script(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)
    return path


def test_common_shell_honours_python_override_with_spaces(tmp_path: Path):
    shards = tmp_path / "shards with spaces"
    shards.mkdir()
    interpreter = tmp_path / "python env" / "python"
    interpreter.parent.mkdir()
    interpreter.symlink_to(sys.executable)
    environment = os.environ.copy()
    environment.update(PYTHON=str(interpreter), SDHCAL_SHARDS_DIR=str(shards))
    completed = subprocess.run(
        ["bash", "-c", 'source scripts/common.sh; printf "%s" "$PYTHON"'],
        cwd=REPO,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == str(interpreter)


def test_make_python_override_is_quoted_and_missing_interpreter_fails(tmp_path: Path):
    interpreter = tmp_path / "python env" / "python"
    dry_run = subprocess.run(
        ["make", "-n", "test", f"PYTHON={interpreter}"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert dry_run.returncode == 0
    assert f'"{interpreter}" -m pytest' in dry_run.stdout

    missing = subprocess.run(
        ["make", "check-python", f"PYTHON={interpreter}"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert missing.returncode != 0
    assert "Python interpreter not found" in missing.stderr


def test_root_config_override_with_spaces_and_cpp_failure_propagation(tmp_path: Path):
    root_config = _executable_script(
        tmp_path / "ROOT install" / "bin" / "root-config",
        "#!/usr/bin/env bash\nif [ \"${1:-}\" = --cflags ]; then printf '%s\\n' '-I/tmp'; fi\n",
    )
    succeeds = subprocess.run(
        [
            "make",
            "cpp",
            f"ROOT_CONFIG={root_config}",
            "CXX=true",
            "REQUIRE_ROOT=1",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert succeeds.returncode == 0, succeeds.stderr

    fails = subprocess.run(
        [
            "make",
            "cpp",
            f"ROOT_CONFIG={root_config}",
            "CXX=false",
            "REQUIRE_ROOT=1",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert fails.returncode != 0


def test_shell_workflows_keep_historical_targets_and_parameters():
    tabular = (REPO / "scripts/10_tabular.sh").read_text(encoding="utf-8")
    pid_hits = (REPO / "scripts/20_pid_hits.sh").read_text(encoding="utf-8")
    energy = (REPO / "scripts/30_energy_gnn.sh").read_text(encoding="utf-8")
    benchmark = (REPO / "scripts/40_benchmark.sh").read_text(encoding="utf-8")

    assert "--seeds 5" in tabular
    assert "for seed in 42 43 44 45 46" in tabular
    assert "N_TRAIN:=8400" in pid_hits
    assert "cnn) epochs=40; patience=8" in pid_hits
    assert "gnn) epochs=30; patience=6" in pid_hits
    assert "N_TRAIN:=79451" in energy
    assert "--max-shards-per-particle 40 --max-events-per-shard 2500" in energy
    assert "make predictions-2class benchmark-2class" in benchmark
