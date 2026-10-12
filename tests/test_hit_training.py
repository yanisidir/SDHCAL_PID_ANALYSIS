"""R6 tests for dependency-light hit-level training and plotting utilities."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from sdhcal.hit_plots import plot_results
from sdhcal.hit_training import BestAucEarlyStopping, balanced_class_weights, new_history

REPO = Path(__file__).resolve().parents[1]


def test_balanced_class_weights_preserve_the_historical_formula():
    labels = np.array([0, 0, 0, 1])
    expected = len(labels) / (2 * np.maximum(np.bincount(labels, minlength=2), 1))
    np.testing.assert_array_equal(balanced_class_weights(labels), expected)


def test_history_schema_is_fresh_and_unchanged():
    first = new_history()
    second = new_history()
    assert list(first) == ["train_loss", "val_loss", "val_auc", "val_balanced_accuracy"]
    first["train_loss"].append(1.0)
    assert second["train_loss"] == []


def test_early_stopping_preserves_strict_auc_improvement_and_patience():
    stopping = BestAucEarlyStopping(patience=2)
    assert stopping.update(0.60) == (True, False)
    assert stopping.update(0.60) == (False, False)
    assert stopping.update(0.59) == (False, True)
    assert stopping.best_auc == 0.60
    assert stopping.stale_epochs == 2
    assert stopping.update(0.61) == (True, False)
    assert stopping.stale_epochs == 0


def test_shared_plotter_preserves_output_names(tmp_path):
    history = {
        "train_loss": [0.8, 0.7],
        "val_loss": [0.9, 0.75],
        "val_auc": [0.6, 0.7],
        "val_balanced_accuracy": [0.55, 0.65],
    }
    plot_results(
        tmp_path,
        history,
        np.array([0.2, 0.8]),
        np.array([0, 1]),
        {"confusion_matrix": [[1, 0], [0, 1]]},
    )
    assert (tmp_path / "training.png").is_file()
    assert (tmp_path / "test_performance.png").is_file()


def test_base_and_tabular_modules_import_when_torch_and_pyg_are_blocked(tmp_path):
    blocker = tmp_path / "block_optional.py"
    blocker.write_text(
        """
import builtins
original = builtins.__import__
def blocked(name, *args, **kwargs):
    if name == 'torch' or name.startswith('torch.') or name == 'torch_geometric' or name.startswith('torch_geometric.'):
        raise ImportError('blocked optional dependency')
    return original(name, *args, **kwargs)
builtins.__import__ = blocked
import sdhcal
from sdhcal import artifacts, config, metrics, tabular, pid_bdt, energy_bdt, pid_mlp, energy_mlp
print('ok')
""".strip()
        + "\n",
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPO / "src")
    completed = subprocess.run(
        [sys.executable, str(blocker)],
        cwd=REPO,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "ok"
