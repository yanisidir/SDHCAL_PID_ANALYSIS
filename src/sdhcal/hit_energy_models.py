"""DeepSets and graph architectures for hit-level energy reconstruction."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GATConv, GraphConv, GraphNorm, global_max_pool, global_mean_pool

__all__ = [
    "EnergyDeepSets",
    "EnergyGNN",
    "predict_forward",
    "target_mode",
    "train_epoch",
    "transformed_target",
]


class EnergyGNN(nn.Module):
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

    def forward(self, data) -> torch.Tensor:
        x = self.encoder(data.x)
        first = F.gelu(self.norm1(self.conv1(x, data.edge_index), data.batch))
        second = F.gelu(self.norm2(self.conv2(self.dropout(first), data.edge_index), data.batch))
        x = F.gelu(self.conv3(self.dropout(second), data.edge_index)) + second
        pooled = torch.cat([global_max_pool(x, data.batch), global_mean_pool(x, data.batch)], dim=1)
        counts = torch.bincount(data.batch, minlength=pooled.size(0)).float().log1p().unsqueeze(1)
        return self.regressor(torch.cat([pooled, counts], dim=1)).squeeze(1)


class EnergyDeepSets(nn.Module):
    """Historical permutation-invariant energy regressor without graph edges."""

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

    def forward(self, data) -> torch.Tensor:
        h = self.encoder(data.x)
        for block, norm in zip(self.blocks, self.norms, strict=True):
            summary = global_mean_pool(h, data.batch)
            h = norm(h + block(torch.cat([h, summary[data.batch]], dim=1)))
        pooled = torch.cat([global_max_pool(h, data.batch), global_mean_pool(h, data.batch)], dim=1)
        counts = torch.bincount(data.batch, minlength=pooled.size(0)).float().log1p().unsqueeze(1)
        return self.regressor(torch.cat([pooled, counts], dim=1)).squeeze(1)


def target_mode(model) -> bool:
    """Whether the model predicts the correction to the chi2 estimate."""
    return bool(getattr(model, "use_baseline", False))


def transformed_target(batch, use_baseline: bool) -> torch.Tensor:
    """Return the historical log-energy or log-ratio training target."""
    energy = batch.y.clamp_min(1e-9)
    if use_baseline:
        return torch.log(energy / batch.baseline.clamp_min(1e-9))
    return torch.log(energy)


def train_epoch(model, loader, optimizer, device) -> float:
    model.train()
    total = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = F.smooth_l1_loss(model(batch), transformed_target(batch, target_mode(model)))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total += loss.item() * batch.num_graphs
    return total / len(loader.dataset)


def predict_forward(model, loader, device) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    total = 0.0
    predictions: list[float] = []
    energies: list[float] = []
    source_indices: list[int] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            output = model(batch)
            loss = F.smooth_l1_loss(output, transformed_target(batch, target_mode(model)))
            total += loss.item() * batch.num_graphs
            predictions.extend(output.cpu().tolist())
            energies.extend(batch.y.cpu().tolist())
            source_indices.extend(batch.event_ref.cpu().tolist())
    return (
        total / len(loader.dataset),
        np.asarray(predictions),
        np.asarray(energies),
        np.asarray(source_indices, dtype=np.int64),
    )
