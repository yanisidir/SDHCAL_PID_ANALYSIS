"""R7 contracts for hit-level energy data processing."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pytest
from sklearn.preprocessing import QuantileTransformer, RobustScaler

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
ak = pytest.importorskip("awkward")
uproot = pytest.importorskip("uproot")

from sdhcal import clustering, metrics  # noqa: E402
from sdhcal.graph import build_knn_graph  # noqa: E402
from sdhcal.hit_energy import (  # noqa: E402
    ArrayHitPreprocessor,
    EnergyGraphDataset,
    ShowerEvent,
    load_particle_events,
    regression_metrics,
    split_train_calibration,
)
from sdhcal.hit_energy_models import EnergyDeepSets, EnergyGNN, predict_forward, train_epoch  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


class LegacyArrayHitPreprocessor:
    """Literal pre-R7 preprocessing behavior used as a numerical reference."""

    def __init__(self, seed: int):
        self.seed = seed
        self.spatial_scaler = None
        self.time_scaler = None

    def fit(self, events, indices, max_hits):
        if max_hits < 1:
            raise ValueError("scaler-max-hits must be positive")
        rng = np.random.default_rng(self.seed)
        coordinate_parts, time_parts = [], []
        remaining = max_hits
        for index in rng.permutation(indices):
            event = events[int(index)]
            take = min(len(event.time), remaining)
            selected = (
                rng.choice(len(event.time), size=take, replace=False)
                if take < len(event.time)
                else slice(None)
            )
            coordinate_parts.append(event.coordinates[selected])
            time_parts.append(event.time[selected])
            remaining -= take
            if remaining == 0:
                break
        if not coordinate_parts:
            raise ValueError("No training hits available to fit the preprocessor")
        coordinates = np.vstack(coordinate_parts)
        timing = np.concatenate(time_parts).reshape(-1, 1)
        self.spatial_scaler = RobustScaler().fit(coordinates)
        self.time_scaler = QuantileTransformer(
            n_quantiles=min(1000, len(timing)),
            output_distribution="normal",
            random_state=self.seed,
        ).fit(timing)

    def transform(self, event):
        spatial = self.spatial_scaler.transform(event.coordinates)
        timing = self.time_scaler.transform(event.time.reshape(-1, 1))
        if event.composition is not None:
            size = np.log1p(event.composition[:, 3:4])
            return np.hstack([spatial, timing, event.composition[:, :3], size]).astype(np.float32)
        one_hot = np.eye(3, dtype=np.float32)[event.threshold.astype(np.int64) - 1]
        return np.hstack([spatial, timing, one_hot]).astype(np.float32)


def _events() -> list[ShowerEvent]:
    return [
        ShowerEvent(
            coordinates=np.array([[i, 2 * i, 3 * i], [i + 1, 2 * i + 1, 3 * i + 1]], dtype=np.float32),
            time=np.array([0.1 * i, 0.1 * i + 0.05], dtype=np.float32),
            threshold=np.array([1, i % 3 + 1]),
            energy=float(5 + i),
            particle="pi" if i < 5 else "proton",
            event_id=f"train:event:{i}",
            baseline=float(4 + i),
        )
        for i in range(10)
    ]


def _write_root(path: Path) -> None:
    jagged = {
        "x": ak.Array([[1.0, 2.0, 3.0], [4.0, 5.0], []]),
        "y": ak.Array([[2.0, 3.0, 4.0], [5.0, 6.0], []]),
        "z": ak.Array([[3.0, 4.0, 5.0], [6.0, 7.0], []]),
        "I": ak.Array([[10, 11, 14], [20, 21], []]),
        "J": ak.Array([[20, 20, 25], [30, 31], []]),
        "K": ak.Array([[1, 1, 2], [46, 47], []]),
        "time": ak.Array([[0.1, 0.2, 0.4], [0.5, 0.6], []]),
        "thr": ak.Array([[1, 2, 3], [2, 1], []]),
    }
    with uproot.recreate(path) as output:
        output["tree"] = {
            **jagged,
            "energy": np.array([10.0, 20.0, 30.0]),
            "eventNumber": np.array([101, 205, 309], dtype=np.int64),
        }


def _load_script(name: str = "r7_energy_train"):
    path = REPO / "analysis/4_energy/gnn/train.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_shower_event_preserves_array_and_metadata_contracts():
    event = _events()[0]
    assert event.coordinates.shape == (2, 3)
    assert event.time.shape == event.threshold.shape == (2,)
    assert event.event_id == "train:event:0"
    assert event.composition is None
    assert event.baseline == 4.0


def test_root_loading_preserves_selection_order_hit_order_and_ids(tmp_path):
    shard = tmp_path / "tiny_digitized.root"
    _write_root(shard)
    events = load_particle_events(
        "pi",
        [shard],
        sample_role="validation",
        tree_name="tree",
        sigma_time=0.0,
        seed=42,
        max_events_per_shard=None,
        require_contained=True,
    )
    assert [event.event_id for event in events] == ["validation:pi:tiny_digitized:101"]
    assert [event.energy for event in events] == [10.0]
    np.testing.assert_array_equal(events[0].coordinates[:, 0], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(events[0].time, np.array([0.1, 0.2, 0.4], dtype=np.float32))
    np.testing.assert_array_equal(events[0].threshold, [1, 2, 3])


def test_root_loading_preserves_cluster_order_and_composition(tmp_path):
    shard = tmp_path / "cluster_digitized.root"
    _write_root(shard)
    event = load_particle_events(
        "proton",
        [shard],
        sample_role="train",
        tree_name="tree",
        sigma_time=0.0,
        seed=7,
        max_events_per_shard=1,
        require_contained=False,
        cluster_hits=True,
    )[0]
    labels = clustering.cluster_labels(np.array([10, 11, 14]), np.array([20, 20, 25]), np.array([1, 1, 2]))
    summary = clustering.cluster_features(
        np.array([1.0, 2.0, 3.0]),
        np.array([2.0, 3.0, 4.0]),
        np.array([3.0, 4.0, 5.0]),
        np.array([0.1, 0.2, 0.4]),
        np.array([1, 2, 3]),
        labels,
    )
    names = clustering.CLUSTER_FEATURE_NAMES
    expected_coordinates = summary[:, [names.index("x"), names.index("y"), names.index("z")]]
    expected_composition = summary[
        :, [names.index(f"frac_thr{level}") for level in (1, 2, 3)] + [names.index("size")]
    ]
    np.testing.assert_array_equal(event.coordinates, expected_coordinates)
    np.testing.assert_array_equal(event.composition, expected_composition)
    np.testing.assert_array_equal(event.time, summary[:, names.index("time")])


def test_preprocessor_fit_transform_matches_pre_r7_exactly_and_round_trips(tmp_path):
    events = _events()
    indices = np.array([0, 2, 4, 6, 8])
    expected = LegacyArrayHitPreprocessor(seed=19)
    actual = ArrayHitPreprocessor(seed=19)
    expected.fit(events, indices, max_hits=7)
    actual.fit(events, indices, max_hits=7)
    for event in events:
        np.testing.assert_array_equal(actual.transform(event), expected.transform(event))

    path = tmp_path / "preprocessor.joblib"
    joblib.dump(actual, path)
    restored = joblib.load(path)
    assert type(restored) is ArrayHitPreprocessor
    np.testing.assert_array_equal(restored.transform(events[3]), actual.transform(events[3]))


def test_split_is_deterministic_stratified_and_disjoint():
    events = [
        ShowerEvent(
            coordinates=np.zeros((1, 3), dtype=np.float32),
            time=np.zeros(1, dtype=np.float32),
            threshold=np.ones(1, dtype=np.int64),
            energy=float(5 + i % 5),
            particle="pi" if i < 20 else "proton",
            event_id=f"train:{i}",
        )
        for i in range(40)
    ]
    first = split_train_calibration(events, calibration_size=0.25, seed=42, energy_bins=2)
    second = split_train_calibration(events, calibration_size=0.25, seed=42, energy_bins=2)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    assert len(first[0]) == 30
    assert len(first[1]) == 10
    assert set(first[0]).isdisjoint(first[1])
    assert sorted(np.concatenate(first).tolist()) == list(range(40))


@pytest.mark.parametrize(("architecture", "with_edges"), [("gat", True), ("deepsets", False)])
def test_dataset_preserves_features_targets_references_and_graph_policy(architecture, with_edges):
    events = _events()
    preprocessor = ArrayHitPreprocessor(seed=3)
    preprocessor.fit(events, np.arange(len(events)), max_hits=100)
    args = SimpleNamespace(
        k=1,
        time_window=2.0,
        causal=True,
        undirected=True,
        knn_chunk_size=2,
        architecture=architecture,
    )
    dataset = EnergyGraphDataset(events, np.array([4, 1]), preprocessor, args)
    graph = dataset[0]
    expected_features = preprocessor.transform(events[4])
    np.testing.assert_array_equal(graph.x.numpy(), expected_features)
    assert graph.y.tolist() == [events[4].energy]
    assert graph.baseline.tolist() == [events[4].baseline]
    assert graph.event_ref.tolist() == [4]
    if with_edges:
        expected_edges = build_knn_graph(
            expected_features,
            events[4].time,
            k=1,
            time_window=2.0,
            causal=True,
            undirected=True,
            chunk_size=2,
        )
        torch.testing.assert_close(graph.edge_index, expected_edges, rtol=0, atol=0)
    else:
        assert graph.edge_index.shape == (2, 0)


def test_regression_metrics_preserve_energy_bias_and_resolution_definitions():
    y_true = np.linspace(5.0, 100.0, 100)
    y_pred = y_true * (1.0 + np.linspace(-0.08, 0.1, 100))
    actual = regression_metrics(y_true, y_pred)
    core = metrics.core_bias_resolution(metrics.relative_residual(y_true, y_pred))
    assert actual["bias"] == core.bias
    assert actual["bias_err"] == core.bias_err
    assert actual["resolution"] == core.sigma
    assert actual["resolution_err"] == core.sigma_err
    assert actual["energy_scale"] == np.mean(y_pred) / np.mean(y_true)
    assert actual["rms_rel_about_zero"] == metrics.rms_relative_about_zero(y_true, y_pred)
    assert actual["n_validation"] == 100


def test_script_keeps_minimal_historical_preprocessor_class_path_compatibility(monkeypatch):
    script = _load_script()
    assert issubclass(script.ArrayHitPreprocessor, ArrayHitPreprocessor)
    artifacts = sorted(REPO.glob("output/energy_reco_gnn*/**/preprocessor_*.joblib"))
    if not artifacts:
        pytest.skip("no historical hit-energy preprocessor is available")
    main_module = sys.modules["__main__"]
    monkeypatch.setattr(main_module, "ArrayHitPreprocessor", script.ArrayHitPreprocessor, raising=False)
    for path in artifacts:
        restored = joblib.load(path)
        assert isinstance(restored, script.ArrayHitPreprocessor)
        assert restored.spatial_scaler is not None
        assert restored.time_scaler is not None
        stable = ArrayHitPreprocessor(
            seed=restored.seed,
            spatial_scaler=restored.spatial_scaler,
            time_scaler=restored.time_scaler,
            # Part of the state since the timing ablation: an artefact saved by an
            # untimed run must rebuild as untimed, or this compares two different
            # preprocessors. Artefacts written before the field default to False.
            drop_timing=getattr(restored, "drop_timing", False),
        )
        event = _events()[2]
        np.testing.assert_array_equal(restored.transform(event), stable.transform(event))
        clustered = ShowerEvent(
            coordinates=event.coordinates,
            time=event.time,
            threshold=event.threshold,
            energy=event.energy,
            particle=event.particle,
            event_id=event.event_id,
            composition=np.array([[1.0, 0.0, 0.0, 2.0], [0.0, 0.5, 0.5, 4.0]]),
        )
        np.testing.assert_array_equal(restored.transform(clustered), stable.transform(clustered))


def test_script_exports_are_compatibility_aliases():
    script = _load_script("r7_energy_aliases")
    assert script.ShowerEvent is ShowerEvent
    assert script.EnergyGraphDataset is EnergyGraphDataset
    assert script.EnergyGNN is EnergyGNN
    assert script.EnergyDeepSets is EnergyDeepSets
    assert script.load_particle_events is load_particle_events
    assert script.split_train_calibration is split_train_calibration
    assert script.train_epoch is train_epoch
    assert script.predict_forward is predict_forward
    assert script.regression_metrics is regression_metrics


@pytest.mark.parametrize("working_directory", [REPO, REPO / "analysis/4_energy/gnn"])
def test_energy_cli_help_runs_from_root_and_script_directory(working_directory, tmp_path):
    script = "analysis/4_energy/gnn/train.py" if working_directory == REPO else "train.py"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPO / "src")
    environment["MPLCONFIGDIR"] = str(tmp_path / "matplotlib")
    completed = subprocess.run(
        [sys.executable, script, "--help"],
        cwd=working_directory,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--architecture" in completed.stdout
    assert "--target" in completed.stdout
    assert "--cluster-hits" in completed.stdout


def test_the_timing_ablation_blanks_the_column_without_changing_the_input():
    """The ablation must remove information, not capacity.

    Dropping the time column outright would also shrink the first layer, so a
    timed and an untimed run would differ in parameter count as well as in what
    they can see, and the comparison would not isolate the timing. Blanking the
    column keeps the architecture identical.
    """
    rng = np.random.default_rng(0)

    def event() -> ShowerEvent:
        n = 50
        return ShowerEvent(
            coordinates=rng.normal(size=(n, 3)).astype(np.float32),
            time=rng.uniform(0.0, 20.0, n).astype(np.float32),
            threshold=rng.integers(1, 4, n).astype(np.int64),
            energy=30.0,
            particle="pi",
            event_id="e",
        )

    events = [event() for _ in range(40)]
    indices = np.arange(len(events))

    timed = ArrayHitPreprocessor(seed=1)
    timed.fit(events, indices, max_hits=5000)
    untimed = ArrayHitPreprocessor(seed=1, drop_timing=True)
    untimed.fit(events, indices, max_hits=5000)

    with_time = timed.transform(events[0])
    without = untimed.transform(events[0])

    assert with_time.shape == without.shape, "the ablation must not change the input width"
    # Column 3 is the time, after the three spatial coordinates.
    assert with_time[:, 3].std() > 0.5, "the timed run should carry a varying time feature"
    assert np.all(without[:, 3] == 0.0), "the ablated run should carry a constant time feature"
    # Everything else is untouched.
    spatial_and_rest = [c for c in range(with_time.shape[1]) if c != 3]
    np.testing.assert_allclose(with_time[:, spatial_and_rest], without[:, spatial_and_rest])


def _cache_events(n_events: int = 60, n_nodes: int = 40) -> tuple[list, np.ndarray, ArrayHitPreprocessor]:
    rng = np.random.default_rng(3)
    events = [
        ShowerEvent(
            coordinates=rng.normal(size=(n_nodes, 3)).astype(np.float32),
            time=rng.uniform(0.0, 20.0, n_nodes).astype(np.float32),
            threshold=rng.integers(1, 4, n_nodes).astype(np.int64),
            energy=30.0,
            particle="pi",
            event_id=f"e{i}",
        )
        for i in range(n_events)
    ]
    indices = np.arange(len(events))
    preprocessor = ArrayHitPreprocessor(seed=1)
    preprocessor.fit(events, indices, max_hits=100000)
    return events, indices, preprocessor


def _dataset_args() -> SimpleNamespace:
    return SimpleNamespace(
        k=8, time_window=2.0, causal=True, undirected=True, knn_chunk_size=512, architecture="deepsets"
    )


def test_the_feature_cache_returns_exactly_what_recomputing_returns():
    """The cache is a speed-up, so it must not be a behaviour change.

    The scalers are deterministic once fitted, so a cached feature matrix has to
    equal the recomputed one bit for bit; anything else would make a run depend
    on whether the cache was on.
    """
    events, indices, preprocessor = _cache_events()
    args = _dataset_args()
    cold = EnergyGraphDataset(events, indices, preprocessor, args, cache_features=False)
    warm = EnergyGraphDataset(events, indices, preprocessor, args, cache_features=True)

    for position in range(len(indices)):
        np.testing.assert_array_equal(cold[position].x.numpy(), warm[position].x.numpy())
    # A second pass reads the cache rather than the preprocessor; same answer.
    for position in range(len(indices)):
        np.testing.assert_array_equal(cold[position].x.numpy(), warm[position].x.numpy())


def test_the_feature_cache_gives_up_rather_than_exhausting_memory():
    """A raw-hit run needs several GB; the budget turns the cache off instead."""
    events, indices, preprocessor = _cache_events()
    dataset = EnergyGraphDataset(
        events, indices, preprocessor, _dataset_args(), cache_features=True, cache_budget_bytes=16
    )
    first = dataset[0]
    assert dataset._cache_enabled is False
    assert dataset._cache == {}
    # Still correct, just slower.
    plain = EnergyGraphDataset(events, indices, preprocessor, _dataset_args(), cache_features=False)
    np.testing.assert_array_equal(first.x.numpy(), plain[0].x.numpy())
