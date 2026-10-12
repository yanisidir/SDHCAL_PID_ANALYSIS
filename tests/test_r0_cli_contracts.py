"""R0 characterization of the command-line interfaces.

These tests freeze the public flags and literal defaults without importing the
optional Torch/PyG pipelines.  They describe the current interface; they do not
approve the scientific choices represented by those defaults.
"""

from __future__ import annotations

import ast
import contextlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CONTRACTS = json.loads((REPO / "tests/fixtures/r0_contracts.json").read_text(encoding="utf-8"))


def _arguments(path: Path) -> tuple[list[str], dict[str, object]]:
    """Return long options and defaults expressible as Python literals."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    flags: list[str] = []
    defaults: dict[str, object] = {}
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
        ):
            continue
        long_flags = [
            value
            for argument in node.args
            if isinstance(argument, ast.Constant)
            and isinstance((value := argument.value), str)
            and value.startswith("--")
        ]
        flags.extend(long_flags)
        if not long_flags:
            continue

        destination = max(long_flags, key=len).removeprefix("--").replace("-", "_")
        keywords = {keyword.arg: keyword.value for keyword in node.keywords}
        if "default" in keywords:
            with contextlib.suppress(ValueError, TypeError):
                defaults[destination] = ast.literal_eval(keywords["default"])
        elif "action" in keywords:
            try:
                action = ast.literal_eval(keywords["action"])
            except (ValueError, TypeError):
                action = None
            if action == "store_true":
                defaults[destination] = False
            elif action == "store_false":
                defaults[destination] = True
    return flags, defaults


@pytest.mark.parametrize("name", sorted(CONTRACTS["cli"]))
def test_cli_flags_and_literal_defaults_match_the_r0_snapshot(name: str):
    contract = CONTRACTS["cli"][name]
    flags, defaults = _arguments(REPO / contract["path"])
    assert flags == contract["flags"]
    for key, expected in contract["defaults"].items():
        assert defaults[key] == expected


# These CLIs use only the base dependencies.  The Torch CLIs are still covered
# statically above, so a base CI installation does not have to install PyG merely
# to protect their public flags.
RUNTIME_HELP = [
    "pid_bdt",
    "pid_mlp",
    "energy_bdt",
    "energy_mlp",
    "coupling",
    "predict",
    "import_predictions",
    "compare_models",
    "export_hits",
    "build_2class",
]


@pytest.mark.parametrize("name", RUNTIME_HELP)
def test_base_cli_help_is_executable_and_lists_the_contract(name: str, tmp_path: Path):
    contract = CONTRACTS["cli"][name]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPO / "src")
    environment["MPLCONFIGDIR"] = str(tmp_path / "matplotlib")
    completed = subprocess.run(
        [sys.executable, contract["path"], "--help"],
        cwd=REPO,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout.lower()
    for flag in contract["flags"]:
        assert flag in completed.stdout
