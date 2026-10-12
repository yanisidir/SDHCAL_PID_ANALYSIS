"""R6 numerical and checkpoint tests for the extracted hit-level GNN."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
from torch import nn  # noqa: E402
from torch.nn import functional as F  # noqa: E402
from torch_geometric.data import Batch, Data  # noqa: E402
from torch_geometric.loader import DataLoader  # noqa: E402
from torch_geometric.nn import (  # noqa: E402
    GATConv,
    GraphConv,
    GraphNorm,
    global_max_pool,
    global_mean_pool,
)

from sdhcal import hit_gnn  # noqa: E402
from sdhcal.graph import build_knn_graph  # noqa: E402
from sdhcal.hit_pid import HitPreprocessor, binary_metrics  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


class LegacyParticleGNN(nn.Module):
    """Literal pre-R6 architecture used only as a numerical reference."""

    def __init__(self, in_channels: int, hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(in_channels, hidden_dim), nn.GELU())
        self.conv1 = GATConv(hidden_dim, hidden_dim // 4, heads=4, dropout=dropout)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.conv3 = GraphConv(hidden_dim, hidden_dim)
        self.norm1 = GraphNorm(hidden_dim)
        self.norm2 = GraphNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, data):
        x = self.encoder(data.x)
        first = F.gelu(self.norm1(self.conv1(x, data.edge_index), data.batch))
        second = F.gelu(self.norm2(self.conv2(self.dropout(first), data.edge_index), data.batch))
        x = F.gelu(self.conv3(self.dropout(second), data.edge_index)) + second
        pooled = torch.cat([global_max_pool(x, data.batch), global_mean_pool(x, data.batch)], dim=1)
        return self.classifier(pooled)


def _frame():
    rows = []
    for event_id, label, n_hits in [("pion:1:4", 0, 3), ("proton:1:2", 1, 3)]:
        for hit in range(n_hits):
            rows.append(
                {
                    "event_id": event_id,
                    "label": label,
                    "x_coords": float(hit + label),
                    "y_coords": float(hit * 2),
                    "z_coords": float(hit),
                    "time": float(hit) / 10,
                    "thr": hit + 1,
                }
            )
    return pd.DataFrame(rows)


def _graphs():
    frame = _frame()
    preprocessor = HitPreprocessor(seed=7).fit(frame)
    args = SimpleNamespace(k=2, time_window=2.0, causal=True, undirected=True, knn_chunk_size=2)
    return hit_gnn.create_graphs(frame, preprocessor, args), preprocessor, args


def _load_script():
    path = REPO / "analysis/3_pid/gnn/train.py"
    spec = importlib.util.spec_from_file_location("r6_pid_gnn", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _legacy_train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(batch), batch.y)
        loss.backward()
        optimizer.step()
        total += loss.item() * batch.num_graphs
    return total / len(loader.dataset)


def test_graph_creation_preserves_event_node_feature_and_edge_order_exactly():
    graphs, preprocessor, args = _graphs()
    frame = _frame()
    assert [graph.event_id for graph in graphs] == ["pion:1:4", "proton:1:2"]
    for graph, (event_id, hits) in zip(graphs, frame.groupby("event_id", sort=False), strict=True):
        expected_features = preprocessor.transform(hits)
        expected_edges = build_knn_graph(
            expected_features,
            hits["time"].to_numpy(),
            k=args.k,
            time_window=args.time_window,
            causal=args.causal,
            undirected=args.undirected,
            chunk_size=args.knn_chunk_size,
        )
        assert graph.event_id == event_id
        np.testing.assert_array_equal(graph.x.numpy(), expected_features)
        torch.testing.assert_close(graph.edge_index, expected_edges, rtol=0, atol=0)


def test_causal_then_undirected_edge_contract_is_frozen():
    features = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0]], dtype=np.float32)
    edges = build_knn_graph(
        features,
        np.array([0.0, 0.1, 0.2]),
        k=1,
        time_window=2.0,
        causal=True,
        undirected=True,
        chunk_size=2,
    )
    expected = torch.tensor([[0, 1, 1, 2, 0, 1, 2], [1, 0, 2, 1, 0, 1, 2]])
    torch.testing.assert_close(edges, expected, rtol=0, atol=0)


def test_graph_batching_and_architecture_forward_match_pre_r6_exactly():
    graphs, _, _ = _graphs()
    batch = Batch.from_data_list(graphs)
    torch.manual_seed(21)
    expected = LegacyParticleGNN(7, 8)
    torch.manual_seed(21)
    actual = hit_gnn.ParticleGNN(7, 8)
    assert list(actual.state_dict()) == list(expected.state_dict())
    for key in actual.state_dict():
        torch.testing.assert_close(actual.state_dict()[key], expected.state_dict()[key], rtol=0, atol=0)
    expected.eval()
    actual.eval()
    with torch.no_grad():
        expected_output = expected(batch)
        actual_output = actual(batch)
    assert batch.num_graphs == 2
    assert actual_output.shape == (2, 2)
    torch.testing.assert_close(actual_output, expected_output, rtol=0, atol=0)


def test_one_cpu_gnn_training_step_matches_pre_r6_exactly():
    graphs, _, _ = _graphs()
    loader = DataLoader(graphs, batch_size=2, shuffle=False)
    torch.manual_seed(23)
    expected = LegacyParticleGNN(7, 8)
    actual = hit_gnn.ParticleGNN(7, 8)
    actual.load_state_dict(expected.state_dict())
    expected_optimizer = torch.optim.AdamW(expected.parameters(), lr=1e-3, weight_decay=1e-4)
    actual_optimizer = torch.optim.AdamW(actual.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    torch.manual_seed(29)
    expected_loss = _legacy_train_epoch(expected, loader, expected_optimizer, loss_fn, torch.device("cpu"))
    torch.manual_seed(29)
    actual_loss = hit_gnn.train_epoch(actual, loader, actual_optimizer, loss_fn, torch.device("cpu"))
    assert actual_loss == expected_loss
    for key in actual.state_dict():
        torch.testing.assert_close(actual.state_dict()[key], expected.state_dict()[key], rtol=0, atol=0)


def test_gnn_evaluation_preserves_probability_and_label_order():
    graphs, _, _ = _graphs()
    loader = DataLoader(graphs, batch_size=2, shuffle=False)
    loss, metrics, probability, labels = hit_gnn.evaluate(
        hit_gnn.ParticleGNN(7, 8), loader, nn.CrossEntropyLoss(), torch.device("cpu")
    )
    assert np.isfinite(loss)
    assert labels.tolist() == [0, 1]
    assert metrics == binary_metrics(labels, probability)


def test_gnn_state_dict_checkpoint_round_trip_is_exact(tmp_path):
    model = hit_gnn.ParticleGNN(7, 8)
    checkpoint = tmp_path / "best_model.pt"
    torch.save(model.state_dict(), checkpoint)
    restored = hit_gnn.ParticleGNN(7, 8)
    restored.load_state_dict(torch.load(checkpoint, map_location="cpu"), strict=True)
    for key in model.state_dict():
        torch.testing.assert_close(restored.state_dict()[key], model.state_dict()[key], rtol=0, atol=0)


@pytest.mark.parametrize("variant", ["bench", "clustered"])
def test_available_historical_gnn_checkpoint_loads_strictly_and_runs(variant):
    path = REPO / "output/pid_hit_gnn" / variant / "best_model.pt"
    if not path.exists():
        pytest.skip(f"historical checkpoint absent: {path}")
    state = torch.load(path, map_location="cpu")
    encoder_weight = state["encoder.0.weight"]
    model = hit_gnn.ParticleGNN(encoder_weight.shape[1], encoder_weight.shape[0])
    legacy = LegacyParticleGNN(encoder_weight.shape[1], encoder_weight.shape[0])
    model.load_state_dict(state, strict=True)
    legacy.load_state_dict(state, strict=True)
    for key, value in state.items():
        torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
    graph = Data(
        x=torch.randn(3, encoder_weight.shape[1]),
        edge_index=torch.tensor([[0, 1, 2], [0, 1, 2]]),
        y=torch.tensor([0]),
    )
    with torch.no_grad():
        output = model.eval()(Batch.from_data_list([graph]))
        legacy_output = legacy.eval()(Batch.from_data_list([graph]))
    assert output.shape == (1, 2)
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output, legacy_output, rtol=0, atol=0)


def test_historical_gnn_script_exports_are_compatibility_aliases():
    script = _load_script()
    assert script.ParticleGNN is hit_gnn.ParticleGNN
    assert script.create_graphs is hit_gnn.create_graphs
    assert script.train_epoch is hit_gnn.train_epoch
    assert script.evaluate is hit_gnn.evaluate


@pytest.mark.parametrize("working_directory", [REPO, REPO / "analysis/3_pid/gnn"])
def test_gnn_cli_help_runs_from_root_and_script_directory(working_directory, tmp_path):
    script = "analysis/3_pid/gnn/train.py" if working_directory == REPO else "train.py"
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
    assert "--causal" in completed.stdout
    assert "--undirected" in completed.stdout
