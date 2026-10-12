"""R9 characterization of the C++/ROOT shower-feature applications."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]

OUTPUT_SCHEMA = [
    "eventNumber",
    "primaryEnergy",
    "particlePDG",
    "Thr1",
    "Thr2",
    "Thr3",
    "Begin",
    "Radius",
    "Density",
    "NClusters",
    "ratioThr23",
    "Zbary",
    "Zrms",
    "PctHitsFirst10",
    "PlanesWithClusmore2",
    "AvgClustSize",
    "MaxClustSize",
    "lambda1",
    "lambda2",
    "N3",
    "N2",
    "N1",
    "tMin",
    "tMax",
    "tMean",
    "tSpread",
    "Nmax",
    "z0_fit",
    "Xmax",
    "lambda",
    "nTrackSegments",
    "eccentricity3D",
    "nHitsTotal",
    "sumThrTotal",
    "touchesLastLayer",
]

INT_REFERENCE = {
    "eventNumber": [101, 205, 309],
    "particlePDG": [-211, 2212, -211],
    "Begin": [48, 48, 0],
    "NClusters": [0, 1, 21],
    "PlanesWithClusmore2": [0, 0, 5],
    "MaxClustSize": [0, 1, 1],
    "N3": [0, 0, 6],
    "N2": [0, 0, 5],
    "N1": [0, 1, 10],
    "nTrackSegments": [0, 0, 0],
    "nHitsTotal": [0, 1, 21],
    "touchesLastLayer": [0, 0, 1],
}

FLOAT_REFERENCE = {
    "primaryEnergy": [5.0, 10.0, 30.0],
    "Thr1": [0.0, 100.0, 47.619049072265625],
    "Thr2": [0.0, 0.0, 23.809524536132812],
    "Thr3": [0.0, 0.0, 28.571428298950195],
    "Radius": [0.0, 0.0, 5.395670413970947],
    "Density": [0.0, 1.0, 2.7321274280548096],
    "ratioThr23": [0.0, 0.0, 1.0999999046325684],
    "Zbary": [0.0, 0.0, 4.142857074737549],
    "Zrms": [0.0, 0.0, 9.682019233703613],
    "PctHitsFirst10": [0.0, 100.0, 95.23809814453125],
    "AvgClustSize": [0.0, 1.0, 1.0],
    "lambda1": [0.0, 0.0, 36.67307662963867],
    "lambda2": [0.0, 0.0, 1.2861078977584839],
    "tMin": [0.0, 4.0, 0.0],
    "tMax": [0.0, 4.0, 10.0],
    "tMean": [0.0, 4.0, 5.0],
    "tSpread": [0.0, 0.0, 10.0],
    "Nmax": [np.nan, np.nan, 7.088703632354736],
    "z0_fit": [np.nan, np.nan, -79.98625946044922],
    "Xmax": [np.nan, np.nan, 18.466054916381836],
    "lambda": [np.nan, np.nan, 79.99503326416016],
    "eccentricity3D": [0.0, 0.0, 0.9969367384910583],
    "sumThrTotal": [0.0, 1.0, 38.0],
}


def _working_root_config() -> Path | None:
    candidates: list[Path] = []
    configured = os.environ.get("ROOT_CONFIG")
    if configured:
        candidates.append(Path(configured))
    found = shutil.which("root-config")
    if found:
        candidates.append(Path(found))
    candidates.extend(Path.home().glob("miniconda3/envs/*/bin/root-config"))
    for candidate in dict.fromkeys(candidates):
        try:
            probe = subprocess.run(
                [str(candidate), "--version"],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if probe.returncode == 0:
            return candidate
    return None


@pytest.fixture(scope="session")
def cpp_tools(tmp_path_factory):
    root_config = _working_root_config()
    if root_config is None:
        pytest.skip("R9 ROOT integration blocked: no working root-config was found")

    build = tmp_path_factory.mktemp("r9_cpp")
    cflags = shlex.split(subprocess.check_output([root_config, "--cflags"], text=True))
    libraries = shlex.split(subprocess.check_output([root_config, "--libs"], text=True))
    root_python = root_config.parent / "python"
    if not root_python.is_file():
        pytest.skip(f"R9 ROOT integration blocked: PyROOT interpreter not found beside {root_config}")
    common = ["g++", "-std=c++17", "-O2", "-Wall", "-Wextra", *cflags, "-I", str(REPO / "cpp/include")]
    shared = [
        str(REPO / "cpp/src/RootShowerIO.cpp"),
        str(REPO / "cpp/src/ShowerFeatures.cpp"),
        str(REPO / "cpp/src/ShowerAnalyzer.cpp"),
    ]

    fixture = build / "create_fixture"
    extract = build / "extract"
    enrich = build / "extract_parallel"
    subprocess.run(
        [*common, str(REPO / "tests/cpp/create_root_fixture.cpp"), "-o", fixture, *libraries],
        check=True,
    )
    subprocess.run(
        [*common, str(REPO / "analysis/1_shower_features/extract.cpp"), *shared, "-o", extract, *libraries],
        check=True,
    )
    subprocess.run(
        [
            *common,
            str(REPO / "analysis/1_shower_features/extract_parallel.cpp"),
            *shared,
            "-o",
            enrich,
            *libraries,
        ],
        check=True,
    )
    return {
        "build": build,
        "fixture": fixture,
        "extract": extract,
        "enrich": enrich,
        "root_python": root_python,
    }


def _input(cpp_tools, name: str, mode: str = "normal") -> Path:
    path = cpp_tools["build"] / name
    subprocess.run([cpp_tools["fixture"], path, mode], check=True)
    return path


def _extract(cpp_tools, input_path: Path, name: str = "params.root") -> Path:
    output = cpp_tools["build"] / name
    subprocess.run([cpp_tools["extract"], input_path, "tree", output], check=True)
    return output


def _inspect(cpp_tools, path: Path, tree: str) -> dict:
    result = subprocess.run(
        [cpp_tools["root_python"], REPO / "tests/root_inspect.py", path, tree],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def test_historical_cli_exit_codes_are_preserved(cpp_tools):
    extract = subprocess.run([cpp_tools["extract"], "unexpected"], capture_output=True, text=True)
    enrich = subprocess.run([cpp_tools["enrich"]], capture_output=True, text=True)
    assert extract.returncode == 2
    assert "[<input.root> <treeName> <output.root>]" in extract.stderr
    assert enrich.returncode == 1
    assert "input.root output.root [treeName]" in enrich.stderr


def test_missing_file_and_tree_keep_historical_failures(cpp_tools):
    missing = subprocess.run(
        [cpp_tools["extract"], cpp_tools["build"] / "absent.root", "tree", cpp_tools["build"] / "out.root"],
        capture_output=True,
        text=True,
    )
    input_path = _input(cpp_tools, "wrong_tree.root")
    wrong_tree = subprocess.run(
        [cpp_tools["extract"], input_path, "absent", cpp_tools["build"] / "wrong_tree_out.root"],
        capture_output=True,
        text=True,
    )
    assert missing.returncode == 1
    assert "Impossible d'ouvrir" in missing.stderr
    assert wrong_tree.returncode == 1
    assert "TTree 'absent' introuvable" in wrong_tree.stderr


def test_fresh_output_schema_order_types_events_and_reference_values(cpp_tools):
    output = _extract(cpp_tools, _input(cpp_tools, "normal.root"))
    observed = _inspect(cpp_tools, output, "paramsTree")
    assert [branch["name"] for branch in observed["branches"]] == OUTPUT_SCHEMA
    assert observed["entries"] == 3
    titles = {branch["name"]: branch["title"] for branch in observed["branches"]}
    for name in INT_REFERENCE:
        assert titles[name] == f"{name}/I"
    for name in FLOAT_REFERENCE:
        assert titles[name] == f"{name}/F"
    for name, expected in INT_REFERENCE.items():
        np.testing.assert_array_equal([row[name] for row in observed["rows"]], expected)
    for name, expected in FLOAT_REFERENCE.items():
        tolerance = 1.0e-3 if name in {"Nmax", "z0_fit", "Xmax", "lambda"} else 1.0e-6
        actual = [np.nan if row[name] is None else row[name] for row in observed["rows"]]
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=tolerance, equal_nan=True)


def test_missing_event_number_keeps_row_index_fallback(cpp_tools):
    output = _extract(cpp_tools, _input(cpp_tools, "no_event.root", "no-event-number"), "fallback.root")
    observed = _inspect(cpp_tools, output, "paramsTree")
    assert [row["eventNumber"] for row in observed["rows"]] == [0, 1, 2]


def test_missing_optional_energy_keeps_historical_schema(cpp_tools):
    output = _extract(cpp_tools, _input(cpp_tools, "no_energy.root", "no-energy"), "no_energy_params.root")
    observed = _inspect(cpp_tools, output, "paramsTree")
    names = [branch["name"] for branch in observed["branches"]]
    assert "primaryEnergy" not in names
    assert names == [name for name in OUTPUT_SCHEMA if name != "primaryEnergy"]


def test_output_schema_contains_the_python_reader_contract(cpp_tools):
    output = _extract(cpp_tools, _input(cpp_tools, "python.root"), "python_params.root")
    columns = ["eventNumber", "particlePDG", "N1", "N2", "N3", "touchesLastLayer"]
    observed = _inspect(cpp_tools, output, "paramsTree")
    names = [branch["name"] for branch in observed["branches"]]
    assert all(column in names for column in columns)
    assert [row["eventNumber"] for row in observed["rows"]] == [101, 205, 309]
    assert [row["particlePDG"] for row in observed["rows"]] == [-211, 2212, -211]
    assert [row["touchesLastLayer"] for row in observed["rows"]] == [0, 0, 1]


def test_python_root_reader_round_trip_when_uproot_supports_root_output(cpp_tools):
    output = _extract(cpp_tools, _input(cpp_tools, "uproot.root"), "uproot_params.root")
    code = (
        "from sdhcal import rootio; import sys; "
        "frame=rootio.load_trees(sys.argv[1], 'paramsTree', ['eventNumber','particlePDG','N1']); "
        "assert frame.to_dict('list') == "
        "{'eventNumber':[101,205,309],'particlePDG':[-211,2212,-211],'N1':[0,1,10]}"
    )
    environment = os.environ.copy()
    current_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = str(REPO / "src") + (
        os.pathsep + current_pythonpath if current_pythonpath else ""
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", code, output],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
            env=environment,
        )
    except subprocess.TimeoutExpired:
        pytest.skip("uproot 5.7.7 blocks while reading TTrees produced by the available ROOT 6.40.04")
    assert result.returncode == 0, result.stderr


def test_enrichment_preserves_input_order_and_computed_values(cpp_tools):
    input_path = _input(cpp_tools, "enrich_input.root")
    output = cpp_tools["build"] / "enriched.root"
    subprocess.run([cpp_tools["enrich"], input_path, output, "tree"], check=True)
    observed = _inspect(cpp_tools, output, "tree")
    assert observed["entries"] == 3
    assert [branch["name"] for branch in observed["branches"][:11]] == [
        "eventNumber",
        "particle1PDG",
        "energy",
        "thr",
        "K",
        "I",
        "J",
        "x",
        "y",
        "z",
        "time",
    ]
    assert [row["particlePDG"] for row in observed["rows"]] == [-211, 2212, -211]
    assert [row["nHitsTotal"] for row in observed["rows"]] == [0, 1, 21]
    assert [row["touchesLastLayer"] for row in observed["rows"]] == [0, 0, 1]


@pytest.mark.xfail(
    strict=True, reason="known defect: cloning then adding eventNumber creates a duplicate branch"
)
def test_enrichment_event_number_branch_should_be_unique(cpp_tools):
    input_path = _input(cpp_tools, "collision_input.root")
    output = cpp_tools["build"] / "collision.root"
    subprocess.run([cpp_tools["enrich"], input_path, output, "tree"], check=True)
    observed = _inspect(cpp_tools, output, "tree")
    names = [branch["name"] for branch in observed["branches"]]
    assert names.count("eventNumber") == 1


@pytest.mark.xfail(
    strict=True, reason="known defect: required ROOT branches are not validated before dereference"
)
@pytest.mark.parametrize("mode", ["missing-time", "wrong-time-type"])
def test_invalid_input_should_fail_cleanly(cpp_tools, mode):
    input_path = _input(cpp_tools, f"{mode}.root", mode)
    output = cpp_tools["build"] / f"{mode}-out.root"
    result = subprocess.run(
        [cpp_tools["extract"], input_path, "tree", output], capture_output=True, text=True
    )
    assert result.returncode == 1
    assert "time" in result.stderr and ("missing" in result.stderr or "type" in result.stderr)


@pytest.mark.xfail(strict=True, reason="known defect: reprocessing appends duplicate derived branches")
def test_reprocessing_an_enriched_tree_should_not_duplicate_managed_branches(cpp_tools):
    input_path = _input(cpp_tools, "reprocess_input.root")
    first = cpp_tools["build"] / "enriched_once.root"
    second = cpp_tools["build"] / "enriched_twice.root"
    subprocess.run([cpp_tools["enrich"], input_path, first, "tree"], check=True)
    subprocess.run([cpp_tools["enrich"], first, second, "tree"], check=True)
    observed = _inspect(cpp_tools, second, "tree")
    names = [branch["name"] for branch in observed["branches"]]
    assert all(names.count(name) == 1 for name in OUTPUT_SCHEMA if name != "primaryEnergy")
