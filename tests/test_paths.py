"""Dataset resolution and the translation of the internship machine's paths."""

from __future__ import annotations

from pathlib import Path

import pytest

from sdhcal import paths


@pytest.fixture
def configured(tmp_path, monkeypatch):
    data, models, output = tmp_path / "data", tmp_path / "models", tmp_path / "out"
    monkeypatch.setenv("SDHCAL_DATA_DIR", str(data))
    monkeypatch.setenv("SDHCAL_MODEL_DIR", str(models))
    monkeypatch.setenv("SDHCAL_OUT_DIR", str(output))
    return data, models, output


def test_directories_follow_the_environment(configured):
    data, models, output = configured
    assert paths.data_dir() == data
    assert paths.model_dir() == models
    assert paths.output_dir() == output


def test_directories_default_under_the_repository(monkeypatch):
    for var in ("SDHCAL_DATA_DIR", "SDHCAL_MODEL_DIR", "SDHCAL_OUT_DIR"):
        monkeypatch.delenv(var, raising=False)
    assert paths.data_dir() == paths.repo_root() / "data"
    assert paths.model_dir() == paths.repo_root() / "models"


def test_repo_root_is_the_checkout_not_the_src_directory():
    """The package lives at <repo>/src/sdhcal, so a naive parent.parent is wrong.

    This broke once, when the package moved into src/: calib.coefficients_path()
    started looking for calibration/ inside src/.
    """
    root = paths.repo_root()
    assert (root / "pyproject.toml").is_file()
    assert (root / "src" / "sdhcal" / "paths.py").is_file()
    assert (root / "calibration" / "chi2_coefficients.json").is_file()
    assert root.name != "src"


def test_dataset_builds_the_declared_layout(configured):
    data, _, _ = configured
    resolved = paths.dataset("params", "pi", require=False)
    assert resolved == data / "params/130k_pi_E1to130_params.root"
    assert paths.dataset("holdout", require=False) == data / "val_set.root"


def test_missing_dataset_names_the_variable_to_set(configured):
    with pytest.raises(paths.DatasetNotFound, match="SDHCAL_DATA_DIR"):
        paths.dataset("params", "pi")


def test_dataset_validates_its_arguments(configured):
    with pytest.raises(KeyError):
        paths.dataset("nonexistent", "pi", require=False)
    with pytest.raises(ValueError, match="requires a particle"):
        paths.dataset("params", require=False)
    with pytest.raises(ValueError, match="unknown particle"):
        paths.dataset("params", "muon", require=False)
    with pytest.raises(ValueError, match="takes no particle"):
        paths.dataset("holdout", "pi", require=False)


def test_particle_pdg_mapping_is_consistent():
    assert paths.PARTICLE_PDG == {"pi": -211, "proton": 2212, "kaon": 311}
    assert set(paths.PDG_TO_PARTICLE.values()) == set(paths.PARTICLES)
    # The order of PARTICLES fixes the integer class labels used everywhere.
    assert paths.PARTICLES == ("pi", "proton", "kaon")


@pytest.mark.parametrize(
    ("legacy", "expected"),
    [
        ("/gridgroup/ilc/midir/analyse/data/params/x.root", "data/params/x.root"),
        ("/gridgroup/ilc/midir/analyse/data/raw/130k_pi_E1to130.root", "data/raw/130k_pi_E1to130.root"),
        ("/gridgroup/ilc/midir/Timing/files/csv/1k_pi_E1to100_flat.csv", "data/csv/1k_pi_E1to100_flat.csv"),
        ("/gridgroup/ilc/midir/Timing/files/root/100k_pi.root", "data/root/100k_pi.root"),
        ("/gridgroup/ilc/midir/analyse/analyse_pi/data/p.root", "data/p.root"),
        ("../../analyse_kaon/data/p.root", "data/p.root"),
    ],
)
def test_legacy_data_paths_land_in_the_data_directory(configured, legacy, expected):
    tmp = configured[0].parent
    assert paths.legacy(legacy) == tmp / expected


@pytest.mark.parametrize(
    "legacy",
    [
        "/gridgroup/ilc/midir/analyse/PID/BDT/results_with_time/models/lgbm_model_26.joblib",
        "/gridgroup/ilc/midir/Timing/files/analyse/MLP/models/lgbm_model_26.joblib",
    ],
)
def test_legacy_model_paths_land_in_the_model_directory(configured, legacy):
    assert paths.legacy(legacy).parent.is_relative_to(configured[1])
    assert paths.legacy(legacy).name == "lgbm_model_26.joblib"


def test_a_longer_prefix_is_not_shadowed_by_a_shorter_one(configured):
    """`/Timing/files/analyse/` must win over `/Timing/files/`."""
    data, models, _ = configured
    resolved = paths.legacy("/gridgroup/ilc/midir/Timing/files/analyse/MLP/processed_data/scaler.joblib")
    assert resolved.is_relative_to(models)
    assert paths.legacy("/gridgroup/ilc/midir/Timing/files/analyse/data/x.root") == data / "x.root"


def test_an_unknown_path_is_returned_unchanged(configured):
    assert paths.legacy("already/fine.root") == Path("already/fine.root")
    assert paths.legacy("/absolute/elsewhere.root") == Path("/absolute/elsewhere.root")


def test_no_legacy_prefix_survives_in_the_maintained_pipelines():
    """The pipelines must not reintroduce an absolute internship path.

    Docstrings quote the old paths when explaining a fix, so only code lines are
    inspected.
    """
    maintained = [
        "sdhcal/smearing.py",
        "sdhcal/metrics.py",
        "sdhcal/runlog.py",
        "sdhcal/splits.py",
        "sdhcal/calib.py",
        "sdhcal/rootio.py",
        "sdhcal/target.py",
        "analysis/5_pid_energy/study.py",
        "analysis/4_energy/ml_mlp/train.py",
    ]
    offenders = []
    for relative in maintained:
        path = paths.repo_root() / relative
        if not path.is_file():
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith(("#", "*", '"', "'")):
                continue
            if "/gridgroup/" in stripped:
                offenders.append(f"{relative}:{number}")
    assert not offenders, "absolute internship paths reintroduced: " + ", ".join(offenders)
