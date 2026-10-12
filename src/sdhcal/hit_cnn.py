"""Reusable hit-set CNN components for binary pion/proton identification.

The optional PyTorch dependency is intentionally isolated in this module.  Data
loading, splitting, CLI orchestration, checkpoint publication, and plotting stay
outside it.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from .hit_pid import HitPreprocessor, binary_metrics

__all__ = ["ParticleCNN", "ParticleDataset", "collate_hits", "evaluate", "train_epoch"]


class ParticleDataset(Dataset):
    """Materialized event samples in their original group order."""

    def __init__(self, frame, preprocessor: HitPreprocessor):
        self.samples: list[tuple[np.ndarray, int]] = []
        self.event_ids: list[str] = []
        for event_id, hits in frame.groupby("event_id", sort=False):
            features = preprocessor.transform(hits)
            if len(features):
                self.samples.append((features, int(hits["label"].iloc[0])))
                self.event_ids.append(str(event_id))
        if not self.samples:
            raise ValueError("No events were found in this data split")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[np.ndarray, int]:
        return self.samples[index]


def collate_hits(batch):
    """Pad a batch without changing hit order and return its validity mask."""
    features, labels = zip(*batch, strict=True)
    max_hits = max(len(event) for event in features)
    channels = features[0].shape[1]
    padded = torch.zeros(len(batch), channels, max_hits, dtype=torch.float32)
    valid = torch.zeros(len(batch), max_hits, dtype=torch.bool)
    for index, event in enumerate(features):
        n_hits = len(event)
        padded[index, :, :n_hits] = torch.from_numpy(event.T)
        valid[index, :n_hits] = True
    return padded, valid, torch.tensor(labels, dtype=torch.long)


class ParticleCNN(nn.Module):
    """Historical point-wise CNN with permutation-invariant masked pooling."""

    def __init__(self, in_channels: int, hidden_dim: int = 128):
        super().__init__()
        self.hit_encoder = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
            nn.GELU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, hits: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        encoded = self.hit_encoder(hits)
        mask = valid.unsqueeze(1)
        maximum = encoded.masked_fill(~mask, -torch.inf).amax(dim=2)
        mean = (encoded * mask).sum(dim=2) / mask.sum(dim=2).clamp_min(1)
        return self.classifier(torch.cat([maximum, mean], dim=1))


def train_epoch(model, loader, optimizer, loss_fn, device) -> float:
    """Run the unchanged CNN optimization loop for one epoch."""
    model.train()
    total = 0.0
    for hits, valid, labels in loader:
        hits, valid, labels = hits.to(device), valid.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(hits, valid), labels)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(labels)
    return total / len(loader.dataset)


def evaluate(model, loader, loss_fn, device):
    """Return historical loss, metrics, proton probabilities, and labels."""
    model.eval()
    total = 0.0
    probabilities: list[float] = []
    labels_out: list[int] = []
    with torch.no_grad():
        for hits, valid, labels in loader:
            hits, valid, labels = hits.to(device), valid.to(device), labels.to(device)
            logits = model(hits, valid)
            total += loss_fn(logits, labels).item() * len(labels)
            probabilities.extend(torch.softmax(logits, dim=1)[:, 1].cpu().tolist())
            labels_out.extend(labels.cpu().tolist())
    probabilities_array = np.asarray(probabilities)
    labels_array = np.asarray(labels_out)
    return (
        total / len(loader.dataset),
        binary_metrics(labels_array, probabilities_array),
        probabilities_array,
        labels_array,
    )
