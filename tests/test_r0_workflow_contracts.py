"""R0 characterization of the Make/Shell workflow entry points."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CONTRACTS = json.loads((REPO / "tests/fixtures/r0_contracts.json").read_text(encoding="utf-8"))


def test_two_class_make_targets_remain_public():
    makefile = (REPO / "Makefile").read_text(encoding="utf-8")
    targets = set(re.findall(r"^([A-Za-z0-9_.-]+):", makefile, re.MULTILINE))
    assert set(CONTRACTS["workflow"]["make_targets"]) <= targets


@pytest.mark.parametrize("relative", CONTRACTS["workflow"]["scripts"])
def test_workflow_scripts_exist_source_common_and_are_valid_bash(relative: str):
    path = REPO / relative
    source = path.read_text(encoding="utf-8")
    assert 'source "$(dirname "${BASH_SOURCE[0]}")/common.sh"' in source
    completed = subprocess.run(
        ["bash", "-n", str(path)], capture_output=True, text=True, timeout=10, check=False
    )
    assert completed.returncode == 0, completed.stderr


def test_common_environment_contract_is_explicit():
    source = (REPO / "scripts/common.sh").read_text(encoding="utf-8")
    exports = "\n".join(re.findall(r"^export .+$", source, re.MULTILINE))
    for variable in CONTRACTS["workflow"]["environment"]:
        assert f': "${{{variable}:=' in source
        assert variable in exports


def test_numbered_workflow_scripts_keep_their_execution_order():
    assert [Path(path).name for path in CONTRACTS["workflow"]["scripts"]] == [
        "10_tabular.sh",
        "20_pid_hits.sh",
        "30_energy_gnn.sh",
        "40_benchmark.sh",
    ]
