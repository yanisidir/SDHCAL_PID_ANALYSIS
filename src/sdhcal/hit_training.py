"""Small shared utilities for the CNN and GNN PID training scripts.

This module deliberately contains no PyTorch import.  Model-specific batches,
forward calls, losses, and optimization loops remain in their model modules.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["BestAucEarlyStopping", "balanced_class_weights", "new_history"]


def balanced_class_weights(labels: np.ndarray) -> np.ndarray:
    """Return the exact two-class weights used by both historical workflows."""
    labels = np.asarray(labels)
    counts = np.bincount(labels, minlength=2)
    return len(labels) / (2 * np.maximum(counts, 1))


def new_history() -> dict[str, list[float]]:
    """Create the historical training-history schema."""
    return {"train_loss": [], "val_loss": [], "val_auc": [], "val_balanced_accuracy": []}


@dataclass
class BestAucEarlyStopping:
    """Track the strict-AUC improvement and patience rule used by both scripts."""

    patience: int
    best_auc: float = -np.inf
    stale_epochs: int = 0

    def update(self, auc: float) -> tuple[bool, bool]:
        """Return ``(improved, should_stop)`` after one validation epoch."""
        if auc > self.best_auc:
            self.best_auc = auc
            self.stale_epochs = 0
            return True, False
        self.stale_epochs += 1
        return False, self.stale_epochs >= self.patience
