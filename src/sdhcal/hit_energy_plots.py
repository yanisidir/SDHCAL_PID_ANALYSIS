"""Behavior-preserving plots for hit-level energy reconstruction."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import metrics  # noqa: E402

__all__ = [
    "plot_bias_resolution",
    "plot_linearity",
    "plot_training",
    "validation_energy_bins",
]


def plot_training(history: dict[str, list[float]], path: Path) -> None:
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, history["train_loss"], label="train")
    plt.plot(epochs, history["calibration_loss"], label="internal calibration")
    plt.xlabel("Epoch")
    plt.ylabel("Smooth L1 on log(E)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def validation_energy_bins(y_true: np.ndarray, fallback_bins: int) -> np.ndarray:
    """Use the fixed validation beam-energy points as their natural plot bins."""
    points = np.unique(np.asarray(y_true, dtype=float))
    if len(points) == 1:
        half_width = max(0.5, 0.05 * abs(float(points[0])))
        return np.asarray([points[0] - half_width, points[0] + half_width])
    if 1 < len(points) <= 100:
        middle = 0.5 * (points[:-1] + points[1:])
        return np.concatenate(
            [
                [points[0] - 0.5 * (points[1] - points[0])],
                middle,
                [points[-1] + 0.5 * (points[-1] - points[-2])],
            ]
        )
    return metrics.energy_bins(float(np.min(y_true)), float(np.max(y_true)), fallback_bins)


def plot_linearity(y_true: np.ndarray, y_pred: np.ndarray, path: Path, bins: np.ndarray) -> None:
    centers, means, errors = [], [], []
    for low, high in zip(bins[:-1], bins[1:], strict=True):
        selected = (y_true >= low) & (y_true < high)
        if selected.sum() < 20:
            continue
        centers.append(float(np.mean(y_true[selected])))
        means.append(float(np.mean(y_pred[selected])))
        errors.append(float(np.std(y_pred[selected], ddof=1) / np.sqrt(selected.sum())))
    lower, upper = float(y_true.min()), float(y_true.max())
    if lower == upper:
        margin = max(0.5, 0.05 * abs(lower))
        lower, upper = lower - margin, upper + margin
    limits = [lower, upper]
    plt.figure(figsize=(6, 6))
    plt.errorbar(centers, means, yerr=errors, fmt="o", capsize=3, label="GNN")
    plt.plot(limits, limits, "r--", label="ideal")
    plt.xlabel("True energy [GeV]")
    plt.ylabel("Mean reconstructed energy [GeV]")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_bias_resolution(y_true: np.ndarray, y_pred: np.ndarray, path: Path, bins: np.ndarray):
    binned = metrics.binned_bias_resolution(y_true, y_pred, bins=bins)
    figure, (top, bottom) = plt.subplots(2, 1, figsize=(6.5, 7), sharex=True)
    top.errorbar(binned["center"], binned["bias"], yerr=binned["bias_err"], fmt="o", capsize=3)
    top.axhline(0, color="0.4", linestyle="--")
    top.set_ylabel("Bias <(Ereco-Etrue)/Etrue>")
    top.grid(alpha=0.3)
    bottom.errorbar(binned["center"], binned["sigma"], yerr=binned["sigma_err"], fmt="o", capsize=3)
    bottom.set_xlabel("True energy [GeV]")
    bottom.set_ylabel("Resolution")
    bottom.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return binned
