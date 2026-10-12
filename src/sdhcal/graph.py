"""Graph construction shared by hit-level SDHCAL neural networks."""

from __future__ import annotations

import numpy as np


def build_knn_graph(
    features: np.ndarray,
    raw_time: np.ndarray,
    *,
    k: int,
    time_window: float,
    causal: bool,
    undirected: bool,
    chunk_size: int,
):
    """Build a bounded-memory spatial kNN graph on CPU.

    The return value is a PyTorch ``edge_index`` tensor. For every destination
    hit, candidate sources are ranked by spatial distance. In causal mode only
    hits at smaller z can be sources. A positive time window removes pairs that
    are too far apart in raw detector time.
    """

    import torch
    from torch_geometric.utils import add_self_loops, to_undirected

    n_hits = len(features)
    if n_hits == 0:
        raise ValueError("Cannot build a graph for an event without hits")
    if k < 1 or chunk_size < 1:
        raise ValueError("k and chunk_size must be positive")
    if n_hits == 1:
        return torch.tensor([[0], [0]], dtype=torch.long)

    coordinates = torch.as_tensor(features[:, :3], dtype=torch.float32)
    times = torch.as_tensor(raw_time, dtype=torch.float32).flatten()
    if len(times) != n_hits:
        raise ValueError(f"raw_time has {len(times)} entries for {n_hits} hits")
    neighbours = min(k, n_hits - 1)
    source_parts: list[torch.Tensor] = []
    destination_parts: list[torch.Tensor] = []

    for start in range(0, n_hits, chunk_size):
        stop = min(start + chunk_size, n_hits)
        destinations = torch.arange(start, stop)
        distances = torch.cdist(coordinates[start:stop], coordinates)
        distances[torch.arange(stop - start), destinations] = torch.inf

        if causal:
            prior_layer = coordinates[:, 2].unsqueeze(0) < coordinates[start:stop, 2].unsqueeze(1)
            distances.masked_fill_(~prior_layer, torch.inf)
        if time_window > 0:
            compatible_time = (times[start:stop].unsqueeze(1) - times.unsqueeze(0)).abs() <= time_window
            distances.masked_fill_(~compatible_time, torch.inf)

        values, indices = torch.topk(distances, neighbours, dim=1, largest=False)
        valid = torch.isfinite(values)
        if valid.any():
            source_parts.append(indices[valid])
            destination_parts.append(destinations.unsqueeze(1).expand_as(indices)[valid])

    if source_parts:
        edge_index = torch.stack([torch.cat(source_parts), torch.cat(destination_parts)])
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    if undirected:
        edge_index = to_undirected(edge_index, num_nodes=n_hits)
    edge_index, _ = add_self_loops(edge_index, num_nodes=n_hits)
    return edge_index.cpu()
