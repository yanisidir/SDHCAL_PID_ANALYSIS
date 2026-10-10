"""The target the energy network learns, and the one it is scored on.

These exist because of a real failure. The target mode was passed as an argument
to four functions and left off one of them, so the network learned
``log(E_true / E_chi2)`` while the loop scored it against ``log(E)``. Nothing
raised: the metric simply sat at the log of the mean energy, about 2.9, and never
moved for twenty-three epochs - and that metric is what early stopping and
checkpoint selection read, so the saved model was chosen at random.

The mode now lives on the model. These tests pin that, and pin the arithmetic
that made the symptom recognisable.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def energy_train():
    """Import analysis/4_energy/gnn/train.py as a module."""
    directory = REPO / "analysis/4_energy/gnn"
    sys.path.insert(0, str(directory))
    try:
        spec = importlib.util.spec_from_file_location("energy_gnn_train", directory / "train.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules["energy_gnn_train"] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(directory))


class _Batch:
    """The two fields transformed_target reads off a PyG batch."""

    def __init__(self, energy, baseline):
        self.y = torch.tensor(energy, dtype=torch.float32)
        self.baseline = torch.tensor(baseline, dtype=torch.float32)


def test_the_mode_defaults_to_the_plain_energy(energy_train):
    assert energy_train.target_mode(torch.nn.Linear(1, 1)) is False


def test_the_mode_is_read_off_the_model(energy_train):
    model = torch.nn.Linear(1, 1)
    model.use_baseline = True
    assert energy_train.target_mode(model) is True


def test_the_energy_target_is_the_log_energy(energy_train):
    batch = _Batch([10.0, 100.0], [9.0, 90.0])
    target = energy_train.transformed_target(batch, False)
    assert target.tolist() == pytest.approx([np.log(10.0), np.log(100.0)], rel=1e-6)


def test_the_ratio_target_is_the_log_of_the_correction(energy_train):
    batch = _Batch([10.0, 100.0], [9.0, 90.0])
    target = energy_train.transformed_target(batch, True)
    assert target.tolist() == pytest.approx([np.log(10 / 9), np.log(100 / 90)], rel=1e-6)


def test_the_ratio_target_sits_near_zero_whatever_the_energy(energy_train):
    """Why the ratio is better conditioned, and how the bug announced itself.

    The ratio target stays near zero across the whole range while the energy
    target grows with it. Scoring one against the other therefore produces an
    error of about log(E), which is exactly the constant 2.9 that was observed.
    """
    energies = [5.0, 20.0, 100.0]
    baselines = [4.8, 19.0, 97.0]
    batch = _Batch(energies, baselines)

    ratio = energy_train.transformed_target(batch, True)
    energy = energy_train.transformed_target(batch, False)

    assert abs(ratio).max() < 0.1
    assert energy.min() > 1.5
    # Mixing the two gives an error of order log(E): the signature of the bug.
    mismatch = (energy - ratio).abs()
    assert mismatch.min() > 1.5
    assert mismatch.tolist() == pytest.approx([np.log(e) for e in energies], abs=0.1)


def test_train_and_eval_agree_on_the_target(energy_train):
    """The regression itself: both paths must score the same quantity.

    A model that predicts the ratio must look good to both the training loss and
    the calibration loss. Before the fix the second disagreed with the first and
    reported a loss two orders of magnitude larger.
    """
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader

    rng = np.random.default_rng(0)
    graphs = []
    for _ in range(16):
        energy = float(rng.uniform(5.0, 100.0))
        baseline = energy * float(np.exp(rng.normal(0.0, 0.05)))
        n = int(rng.integers(3, 9))
        graphs.append(
            Data(
                x=torch.randn(n, 4),
                edge_index=torch.empty((2, 0), dtype=torch.long),
                y=torch.tensor([energy], dtype=torch.float32),
                baseline=torch.tensor([baseline], dtype=torch.float32),
                event_ref=torch.tensor([0], dtype=torch.long),
                num_nodes=n,
            )
        )
    loader = DataLoader(graphs, batch_size=8)

    class PerfectRatio(torch.nn.Module):
        """Predicts log(E/baseline) exactly, which is what ratio mode asks for."""

        use_baseline = True

        def forward(self, data):
            return torch.log(data.y / data.baseline)

    loss, _, _, _ = energy_train.predict_forward(PerfectRatio(), loader, torch.device("cpu"))
    # A model that is exactly right must score ~0. It scored ~2.9 before the fix.
    assert loss < 1e-6
