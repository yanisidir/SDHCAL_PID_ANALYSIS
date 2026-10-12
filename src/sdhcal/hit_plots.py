"""Shared, behavior-preserving figures for hit-level PID workflows."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

__all__ = ["plot_results"]


def plot_results(run_dir: Path, history, probabilities, labels, metrics) -> None:
    """Write the historical training and test-performance PNG files."""
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="validation")
    axes[0].set(xlabel="Epoch", ylabel="Cross-entropy", title="Loss")
    axes[0].legend()
    axes[1].plot(epochs, history["val_auc"], label="AUC")
    axes[1].plot(epochs, history["val_balanced_accuracy"], label="Balanced accuracy")
    axes[1].set(xlabel="Epoch", ylabel="Metric", ylim=(0, 1), title="Validation")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(run_dir / "training.png", dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(probabilities[labels == 0], bins=40, alpha=0.6, label="pion")
    axes[0].hist(probabilities[labels == 1], bins=40, alpha=0.6, label="proton")
    axes[0].set(xlabel="P(proton)", ylabel="Events", title="Test scores")
    axes[0].legend()
    matrix = np.asarray(metrics["confusion_matrix"])
    normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    axes[1].imshow(normalized, vmin=0, vmax=1, cmap="Blues")
    for row in range(2):
        for column in range(2):
            axes[1].text(column, row, f"{normalized[row, column]:.1%}", ha="center", va="center")
    axes[1].set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["pion", "proton"],
        yticklabels=["pion", "proton"],
        xlabel="Predicted",
        ylabel="True",
        title="Test confusion matrix",
    )
    figure.tight_layout()
    figure.savefig(run_dir / "test_performance.png", dpi=160)
    plt.close(figure)
