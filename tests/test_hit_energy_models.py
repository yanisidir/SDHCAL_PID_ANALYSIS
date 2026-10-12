"""R7 numerical regression tests for hit-level energy models."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
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

from sdhcal.hit_energy_models import (  # noqa: E402
    EnergyDeepSets,
    EnergyGNN,
    predict_forward,
    target_mode,
    train_epoch,
    transformed_target,
)
from sdhcal.target import make_target  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
CHECKPOINTS = sorted(REPO.glob("output/energy_reco_gnn*/**/energy_gnn_*.pt"))
TARGETS = sorted(REPO.glob("output/energy_reco_gnn*/**/target_*.joblib"))


class LegacyEnergyGNN(nn.Module):
    """Literal pre-R7 graph architecture."""

    def __init__(self, in_channels: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(in_channels, hidden_dim), nn.GELU())
        self.conv1 = GATConv(hidden_dim, hidden_dim // 4, heads=4, dropout=dropout)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.conv3 = GraphConv(hidden_dim, hidden_dim)
        self.norm1 = GraphNorm(hidden_dim)
        self.norm2 = GraphNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, data):
        x = self.encoder(data.x)
        first = F.gelu(self.norm1(self.conv1(x, data.edge_index), data.batch))
        second = F.gelu(self.norm2(self.conv2(self.dropout(first), data.edge_index), data.batch))
        x = F.gelu(self.conv3(self.dropout(second), data.edge_index)) + second
        pooled = torch.cat([global_max_pool(x, data.batch), global_mean_pool(x, data.batch)], dim=1)
        counts = torch.bincount(data.batch, minlength=pooled.size(0)).float().log1p().unsqueeze(1)
        return self.regressor(torch.cat([pooled, counts], dim=1)).squeeze(1)


class LegacyEnergyDeepSets(nn.Module):
    """Literal pre-R7 set architecture."""

    def __init__(self, in_channels: int, hidden_dim: int, dropout: float, n_blocks: int = 6):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(in_channels, hidden_dim), nn.GELU())
        self.blocks = nn.ModuleList(
            nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(n_blocks)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(n_blocks))
        self.regressor = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, data):
        h = self.encoder(data.x)
        for block, norm in zip(self.blocks, self.norms, strict=True):
            summary = global_mean_pool(h, data.batch)
            h = norm(h + block(torch.cat([h, summary[data.batch]], dim=1)))
        pooled = torch.cat([global_max_pool(h, data.batch), global_mean_pool(h, data.batch)], dim=1)
        counts = torch.bincount(data.batch, minlength=pooled.size(0)).float().log1p().unsqueeze(1)
        return self.regressor(torch.cat([pooled, counts], dim=1)).squeeze(1)


def _batch(in_channels: int = 7) -> Batch:
    graphs = []
    for index, nodes in enumerate((3, 4)):
        generator = torch.Generator().manual_seed(100 + index + in_channels)
        source = torch.arange(nodes, dtype=torch.long)
        target = torch.roll(source, shifts=-1)
        graphs.append(
            Data(
                x=torch.randn(nodes, in_channels, generator=generator),
                edge_index=torch.stack([source, target]),
                y=torch.tensor([10.0 + 10 * index]),
                baseline=torch.tensor([9.0 + 10 * index]),
                event_ref=torch.tensor([index]),
                num_nodes=nodes,
            )
        )
    return Batch.from_data_list(graphs)


@pytest.mark.parametrize(
    ("legacy_class", "actual_class", "kwargs"),
    [
        (LegacyEnergyGNN, EnergyGNN, {}),
        (LegacyEnergyDeepSets, EnergyDeepSets, {"n_blocks": 3}),
    ],
)
def test_model_initialization_parameters_and_forward_match_pre_r7_exactly(legacy_class, actual_class, kwargs):
    torch.manual_seed(31)
    expected = legacy_class(7, 8, 0.0, **kwargs)
    torch.manual_seed(31)
    actual = actual_class(7, 8, 0.0, **kwargs)
    assert list(actual.state_dict()) == list(expected.state_dict())
    for key in actual.state_dict():
        torch.testing.assert_close(actual.state_dict()[key], expected.state_dict()[key], rtol=0, atol=0)
    expected.eval()
    actual.eval()
    with torch.no_grad():
        expected_output = expected(_batch())
        actual_output = actual(_batch())
    assert actual_output.shape == (2,)
    torch.testing.assert_close(actual_output, expected_output, rtol=0, atol=0)


def _legacy_train_epoch(model, loader, optimizer, device):
    model.train()
    total = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        energy = batch.y.clamp_min(1e-9)
        target = (
            torch.log(energy / batch.baseline.clamp_min(1e-9)) if model.use_baseline else torch.log(energy)
        )
        loss = F.smooth_l1_loss(model(batch), target)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total += loss.item() * batch.num_graphs
    return total / len(loader.dataset)


@pytest.mark.parametrize(
    ("legacy_class", "actual_class", "kwargs"),
    [
        (LegacyEnergyGNN, EnergyGNN, {}),
        (LegacyEnergyDeepSets, EnergyDeepSets, {"n_blocks": 2}),
    ],
)
def test_cpu_loss_gradients_and_optimizer_step_match_pre_r7_exactly(legacy_class, actual_class, kwargs):
    graphs = _batch().to_data_list()
    loader = DataLoader(graphs, batch_size=2, shuffle=False)
    torch.manual_seed(37)
    expected = legacy_class(7, 8, 0.0, **kwargs)
    actual = actual_class(7, 8, 0.0, **kwargs)
    actual.load_state_dict(expected.state_dict())
    expected.use_baseline = actual.use_baseline = True
    expected_optimizer = torch.optim.AdamW(expected.parameters(), lr=1e-3, weight_decay=1e-4)
    actual_optimizer = torch.optim.AdamW(actual.parameters(), lr=1e-3, weight_decay=1e-4)
    expected_loss = _legacy_train_epoch(expected, loader, expected_optimizer, torch.device("cpu"))
    actual_loss = train_epoch(actual, loader, actual_optimizer, torch.device("cpu"))
    assert actual_loss == expected_loss
    for (expected_name, expected_parameter), (actual_name, actual_parameter) in zip(
        expected.named_parameters(), actual.named_parameters(), strict=True
    ):
        assert actual_name == expected_name
        torch.testing.assert_close(actual_parameter.grad, expected_parameter.grad, rtol=0, atol=0)
        torch.testing.assert_close(actual_parameter, expected_parameter, rtol=0, atol=0)


def test_targets_prediction_order_loss_and_inverse_energy_are_preserved():
    batch = _batch()
    assert target_mode(nn.Linear(1, 1)) is False
    plain = transformed_target(batch, False)
    ratio = transformed_target(batch, True)
    torch.testing.assert_close(plain, torch.log(batch.y), rtol=0, atol=0)
    torch.testing.assert_close(ratio, torch.log(batch.y / batch.baseline), rtol=0, atol=0)

    class PerfectRatio(nn.Module):
        use_baseline = True

        def forward(self, data):
            return torch.log(data.y / data.baseline)

    loader = DataLoader(batch.to_data_list(), batch_size=1, shuffle=False)
    loss, prediction, energy, source = predict_forward(PerfectRatio(), loader, torch.device("cpu"))
    assert loss == 0.0
    np.testing.assert_array_equal(energy, [10.0, 20.0])
    np.testing.assert_array_equal(source, [0, 1])
    target = make_target("log", correct_bias=True)
    correction = target.fit_correction(energy / np.array([9.0, 19.0]), prediction)
    assert correction == pytest.approx(1.0)
    reconstructed = target.inverse(prediction) * np.array([9.0, 19.0])
    np.testing.assert_allclose(reconstructed, energy, rtol=1e-7, atol=0)


@pytest.mark.parametrize(("model_class", "kwargs"), [(EnergyGNN, {}), (EnergyDeepSets, {"n_blocks": 2})])
def test_state_dict_checkpoint_round_trip_is_exact(tmp_path, model_class, kwargs):
    model = model_class(7, 8, 0.0, **kwargs)
    checkpoint = tmp_path / "model.pt"
    torch.save(model.state_dict(), checkpoint)
    restored = model_class(7, 8, 0.0, **kwargs)
    restored.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    for key in model.state_dict():
        torch.testing.assert_close(restored.state_dict()[key], model.state_dict()[key], rtol=0, atol=0)


def test_every_available_historical_checkpoint_loads_strictly_and_matches_pre_r7_forward():
    if not CHECKPOINTS:
        pytest.skip("no historical hit-energy checkpoint is available")
    for checkpoint in CHECKPOINTS:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        in_channels = state["encoder.0.weight"].shape[1]
        hidden_dim = state["encoder.0.weight"].shape[0]
        if any(key.startswith("conv1.") for key in state):
            legacy = LegacyEnergyGNN(in_channels, hidden_dim, 0.0)
            actual = EnergyGNN(in_channels, hidden_dim, 0.0)
        else:
            n_blocks = len({int(key.split(".")[1]) for key in state if key.startswith("blocks.")})
            legacy = LegacyEnergyDeepSets(in_channels, hidden_dim, 0.0, n_blocks=n_blocks)
            actual = EnergyDeepSets(in_channels, hidden_dim, 0.0, n_blocks=n_blocks)
        legacy.load_state_dict(state, strict=True)
        actual.load_state_dict(state, strict=True)
        legacy.eval()
        actual.eval()
        batch = _batch(in_channels)
        with torch.no_grad():
            expected = legacy(batch)
            observed = actual(batch)
        torch.testing.assert_close(observed, expected, rtol=0, atol=0)


def test_every_available_historical_target_inverse_transformation_loads_and_runs():
    if not TARGETS:
        pytest.skip("no historical hit-energy target is available")
    values = np.array([-0.2, 0.0, 0.3])
    for path in TARGETS:
        target = joblib.load(path)
        prediction = target.inverse(values)
        assert prediction.shape == values.shape
        assert np.isfinite(prediction).all()
