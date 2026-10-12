"""Integration contracts for configuration-aware tabular trainers."""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _load_script(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, REPO / relative)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {relative}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def trainers():
    return {
        "pid_bdt": _load_script("r2_pid_bdt", "analysis/3_pid/bdt/train.py"),
        "pid_mlp": _load_script("r2_pid_mlp", "analysis/3_pid/mlp/train.py"),
        "energy_bdt": _load_script("r2_energy_bdt", "analysis/4_energy/ml_bdt/train.py"),
        "energy_mlp": _load_script("r2_energy_mlp", "analysis/4_energy/ml_mlp/train.py"),
    }


def test_pid_bdt_without_config_preserves_previous_effective_defaults(trainers, monkeypatch):
    monkeypatch.delenv("SDHCAL_PARAMS_SET", raising=False)
    module = trainers["pid_bdt"]
    effective, seeds, resolved = module.resolve_configuration([])
    assert asdict(effective) == asdict(module.Config())
    assert seeds == 1
    assert resolved["profile"] is None
    assert resolved["values"]["seeds"] == 1


def test_pid_mlp_without_config_preserves_previous_effective_defaults(trainers, monkeypatch):
    monkeypatch.delenv("SDHCAL_PARAMS_SET", raising=False)
    module = trainers["pid_mlp"]
    effective, output_dir, comment, resolved = module.resolve_configuration([])
    assert asdict(effective) == asdict(module.Config())
    assert output_dir.name == "pid_mlp"
    assert comment == ""
    assert resolved["values"]["random_state"] == 42


def test_energy_bdt_without_config_preserves_previous_effective_defaults(trainers, monkeypatch):
    monkeypatch.delenv("SDHCAL_PARAMS_SET", raising=False)
    module = trainers["energy_bdt"]
    effective, particles, seeds, resolved = module.resolve_configuration([])
    assert asdict(effective) == asdict(module.Config())
    assert particles == ["all"]
    assert seeds == 1
    assert resolved["values"]["target"] == "log"


def test_energy_mlp_without_config_preserves_previous_effective_defaults(trainers, monkeypatch):
    monkeypatch.delenv("SDHCAL_PARAMS_SET", raising=False)
    module = trainers["energy_mlp"]
    effective, particles, comment, resolved = module.resolve_configuration([])
    assert asdict(effective) == asdict(module.Config())
    assert particles == ["all"]
    assert comment == ""
    assert resolved["values"]["n_bins"] == 13


def test_file_values_apply_but_explicit_cli_wins(trainers, tmp_path: Path, monkeypatch):
    monkeypatch.delenv("SDHCAL_PARAMS_SET", raising=False)
    path = tmp_path / "experiment.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "profile": "synthetic",
                "dataset": {
                    "params_set": "130k",
                    "training_sample": "train",
                    "validation_sample": "validation",
                },
                "workflows": {
                    "pid_bdt": {
                        "seeds": 3,
                        "grid_search": False,
                        "exclude_holdout": True,
                        "smear_sigma_ns": None,
                        "feature_cols": ["Density", "Thr1"],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    module = trainers["pid_bdt"]
    effective, seeds, resolved = module.resolve_configuration(
        ["--config", str(path), "--seeds", "7", "--grid-search"]
    )
    assert seeds == 7
    assert effective.grid_search is True
    assert effective.exclude_holdout is True
    assert effective.smear_sigma_ns is None
    assert list(effective.feature_cols) == ["Density", "Thr1"]
    assert resolved["sources"]["cli_overrides"] == ["seeds", "grid_search"]


def test_explicit_cli_feature_order_and_boolean_override_file(trainers, tmp_path: Path, monkeypatch):
    monkeypatch.delenv("SDHCAL_PARAMS_SET", raising=False)
    path = tmp_path / "energy.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "profile": "synthetic",
                "dataset": {
                    "params_set": "130k",
                    "training_sample": "train",
                    "validation_sample": "validation",
                },
                "workflows": {
                    "energy_bdt": {
                        "grid_search": False,
                        "feature_cols": ["N1", "N2"],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    module = trainers["energy_bdt"]
    effective, _, _, _ = module.resolve_configuration(
        ["--config", str(path), "--features", "N3", "N1", "--grid-search"]
    )
    assert list(effective.feature_cols) == ["N3", "N1"]
    assert effective.grid_search is True


def test_repository_two_class_profile_resolves_without_training(trainers, monkeypatch):
    monkeypatch.setenv("SDHCAL_PARAMS_SET", "2class")
    config_path = REPO / "configs/two_class.json"
    pid, seeds, _ = trainers["pid_bdt"].resolve_configuration(["--config", str(config_path)])
    energy, particles, _, _ = trainers["energy_bdt"].resolve_configuration(["--config", str(config_path)])
    assert list(pid.particles) == ["pi", "proton"]
    assert seeds == 1
    assert particles == ["all"]
    assert energy.require_contained is True
