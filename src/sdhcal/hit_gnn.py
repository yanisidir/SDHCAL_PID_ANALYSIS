"""Reusable graph PID components for binary pion/proton identification.

PyTorch and PyTorch Geometric remain optional and are imported only by this
hit-level module.  Graph construction delegates unchanged to
:func:`sdhcal.graph.build_knn_graph`.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GraphConv, GraphNorm, global_max_pool, global_mean_pool

from .graph import build_knn_graph
from .hit_pid import HitPreprocessor, binary_metrics

__all__ = ["ParticleGNN", "create_graphs", "evaluate", "train_epoch"]


def create_graphs(frame, preprocessor: HitPreprocessor, args) -> list[Data]:
    """Create graphs in event order with the historical node and edge conventions."""
    graphs: list[Data] = []
    for event_id, hits in frame.groupby("event_id", sort=False):
        features = preprocessor.transform(hits)
        edge_index = build_knn_graph(
            features,
            hits["time"].to_numpy(),
            k=args.k,
            time_window=args.time_window,
            causal=args.causal,
            undirected=args.undirected,
            chunk_size=args.knn_chunk_size,
        )
        graph = Data(
            x=torch.from_numpy(features),
            edge_index=edge_index,
            y=torch.tensor([int(hits["label"].iloc[0])], dtype=torch.long),
            num_nodes=len(features),
        )
        graph.event_id = str(event_id)
        graphs.append(graph)
    if not graphs:
        raise ValueError("No events were found in this data split")
    return graphs


class ParticleGNN(nn.Module):
    """Historical attention/graph-convolution pion-proton classifier."""

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

    def forward(self, data) -> torch.Tensor:
        x = self.encoder(data.x)
        first = F.gelu(self.norm1(self.conv1(x, data.edge_index), data.batch))
        second = F.gelu(self.norm2(self.conv2(self.dropout(first), data.edge_index), data.batch))
        x = F.gelu(self.conv3(self.dropout(second), data.edge_index)) + second
        pooled = torch.cat([global_max_pool(x, data.batch), global_mean_pool(x, data.batch)], dim=1)
        return self.classifier(pooled)


def train_epoch(model, loader, optimizer, loss_fn, device) -> float:
    """Run the unchanged GNN optimization loop for one epoch."""
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


def evaluate(model, loader, loss_fn, device):
    """Return historical loss, metrics, proton probabilities, and labels."""
    model.eval()
    total = 0.0
    probabilities: list[float] = []
    labels_out: list[int] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            total += loss_fn(logits, batch.y).item() * batch.num_graphs
            probabilities.extend(torch.softmax(logits, dim=1)[:, 1].cpu().tolist())
            labels_out.extend(batch.y.cpu().tolist())
    probabilities_array = np.asarray(probabilities)
    labels_array = np.asarray(labels_out)
    return (
        total / len(loader.dataset),
        binary_metrics(labels_array, probabilities_array),
        probabilities_array,
        labels_array,
    )
