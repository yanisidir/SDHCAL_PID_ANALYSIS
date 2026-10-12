"""R6 numerical and compatibility tests for the extracted hit-level CNN."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from sdhcal import hit_cnn  # noqa: E402
from sdhcal.hit_pid import HitPreprocessor, binary_metrics  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


class LegacyParticleCNN(nn.Module):
    """Literal pre-R6 architecture used only as a numerical reference."""

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

    def forward(self, hits, valid):
        encoded = self.hit_encoder(hits)
        mask = valid.unsqueeze(1)
        maximum = encoded.masked_fill(~mask, -torch.inf).amax(dim=2)
        mean = (encoded * mask).sum(dim=2) / mask.sum(dim=2).clamp_min(1)
        return self.classifier(torch.cat([maximum, mean], dim=1))


def _frame():
    rows = []
    for event_id, label, n_hits in [("pion:1:4", 0, 3), ("proton:1:2", 1, 2)]:
        for hit in range(n_hits):
            rows.append(
                {
                    "event_id": event_id,
                    "label": label,
                    "x_coords": float(hit + 2 * label),
                    "y_coords": float(2 * hit - label),
                    "z_coords": float(hit),
                    "time": float(hit) / 10,
                    "thr": hit % 3 + 1,
                }
            )
    return pd.DataFrame(rows)


def _dataset():
    frame = _frame()
    preprocessor = HitPreprocessor(seed=7).fit(frame)
    return hit_cnn.ParticleDataset(frame, preprocessor), preprocessor


def _load_script():
    path = REPO / "analysis/3_pid/cnn/train.py"
    spec = importlib.util.spec_from_file_location("r6_pid_cnn", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _legacy_train_epoch(model, loader, optimizer, loss_fn, device):
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


def test_dataset_preserves_event_and_hit_order():
    dataset, preprocessor = _dataset()
    frame = _frame()
    assert dataset.event_ids == ["pion:1:4", "proton:1:2"]
    np.testing.assert_array_equal(
        dataset.samples[0][0], preprocessor.transform(frame[frame["event_id"] == "pion:1:4"])
    )
    assert [label for _, label in dataset.samples] == [0, 1]


def test_collation_preserves_padding_mask_labels_and_hit_order():
    dataset, _ = _dataset()
    padded, valid, labels = hit_cnn.collate_hits([dataset[0], dataset[1]])
    assert padded.shape == (2, 7, 3)
    assert valid.tolist() == [[True, True, True], [True, True, False]]
    assert labels.tolist() == [0, 1]
    np.testing.assert_array_equal(padded[0, :, :3].numpy().T, dataset.samples[0][0])
    np.testing.assert_array_equal(padded[1, :, :2].numpy().T, dataset.samples[1][0])
    np.testing.assert_array_equal(padded[1, :, 2].numpy(), np.zeros(7))


def test_architecture_parameters_and_forward_match_pre_r6_exactly():
    dataset, _ = _dataset()
    inputs = hit_cnn.collate_hits([dataset[0], dataset[1]])
    torch.manual_seed(11)
    expected = LegacyParticleCNN(7, 16)
    torch.manual_seed(11)
    actual = hit_cnn.ParticleCNN(7, 16)
    assert list(actual.state_dict()) == list(expected.state_dict())
    for key in actual.state_dict():
        torch.testing.assert_close(actual.state_dict()[key], expected.state_dict()[key], rtol=0, atol=0)
    expected.eval()
    actual.eval()
    with torch.no_grad():
        expected_output = expected(inputs[0], inputs[1])
        actual_output = actual(inputs[0], inputs[1])
    assert actual_output.shape == (2, 2)
    torch.testing.assert_close(actual_output, expected_output, rtol=0, atol=0)


def test_forward_is_invariant_to_hit_permutation_and_extra_padding():
    model = hit_cnn.ParticleCNN(7, 8).eval()
    hits = torch.randn(1, 7, 4)
    valid = torch.ones(1, 4, dtype=torch.bool)
    permutation = torch.tensor([2, 0, 3, 1])
    padded = torch.cat([hits, torch.randn(1, 7, 2)], dim=2)
    padded_valid = torch.tensor([[True, True, True, True, False, False]])
    with torch.no_grad():
        reference = model(hits, valid)
        permuted = model(hits[:, :, permutation], valid)
        with_padding = model(padded, padded_valid)
    torch.testing.assert_close(permuted, reference, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(with_padding, reference, rtol=1e-6, atol=1e-7)


def test_one_cpu_training_step_matches_pre_r6_exactly():
    dataset, _ = _dataset()
    loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=hit_cnn.collate_hits)
    torch.manual_seed(13)
    expected = LegacyParticleCNN(7, 8)
    actual = hit_cnn.ParticleCNN(7, 8)
    actual.load_state_dict(expected.state_dict())
    expected_optimizer = torch.optim.AdamW(expected.parameters(), lr=1e-3, weight_decay=1e-4)
    actual_optimizer = torch.optim.AdamW(actual.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    torch.manual_seed(17)
    expected_loss = _legacy_train_epoch(expected, loader, expected_optimizer, loss_fn, torch.device("cpu"))
    torch.manual_seed(17)
    actual_loss = hit_cnn.train_epoch(actual, loader, actual_optimizer, loss_fn, torch.device("cpu"))
    assert actual_loss == expected_loss
    for key in actual.state_dict():
        torch.testing.assert_close(actual.state_dict()[key], expected.state_dict()[key], rtol=0, atol=0)


def test_evaluation_preserves_loss_metrics_probability_and_label_order():
    dataset, _ = _dataset()
    loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=hit_cnn.collate_hits)
    model = hit_cnn.ParticleCNN(7, 8)
    loss, metrics, probability, labels = hit_cnn.evaluate(
        model, loader, nn.CrossEntropyLoss(), torch.device("cpu")
    )
    assert np.isfinite(loss)
    assert labels.tolist() == [0, 1]
    assert metrics == binary_metrics(labels, probability)


def test_cnn_state_dict_checkpoint_round_trip_is_exact(tmp_path):
    model = hit_cnn.ParticleCNN(7, 8)
    checkpoint = tmp_path / "best_model.pt"
    torch.save(model.state_dict(), checkpoint)
    restored = hit_cnn.ParticleCNN(7, 8)
    restored.load_state_dict(torch.load(checkpoint, map_location="cpu"), strict=True)
    for key in model.state_dict():
        torch.testing.assert_close(restored.state_dict()[key], model.state_dict()[key], rtol=0, atol=0)


@pytest.mark.parametrize("variant", ["bench", "clustered"])
def test_available_historical_cnn_checkpoint_loads_strictly_and_runs(variant):
    path = REPO / "output/pid_hit_cnn" / variant / "best_model.pt"
    if not path.exists():
        pytest.skip(f"historical checkpoint absent: {path}")
    state = torch.load(path, map_location="cpu")
    first_weight = state["hit_encoder.0.weight"]
    model = hit_cnn.ParticleCNN(first_weight.shape[1], first_weight.shape[0])
    legacy = LegacyParticleCNN(first_weight.shape[1], first_weight.shape[0])
    model.load_state_dict(state, strict=True)
    legacy.load_state_dict(state, strict=True)
    for key, value in state.items():
        torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
    hits = torch.randn(1, first_weight.shape[1], 3)
    with torch.no_grad():
        output = model.eval()(hits, torch.ones(1, 3, dtype=torch.bool))
        legacy_output = legacy.eval()(hits, torch.ones(1, 3, dtype=torch.bool))
    assert output.shape == (1, 2)
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output, legacy_output, rtol=0, atol=0)


def test_historical_script_exports_are_compatibility_aliases():
    script = _load_script()
    assert script.ParticleDataset is hit_cnn.ParticleDataset
    assert script.ParticleCNN is hit_cnn.ParticleCNN
    assert script.collate_hits is hit_cnn.collate_hits
    assert script.train_epoch is hit_cnn.train_epoch
    assert script.evaluate is hit_cnn.evaluate


@pytest.mark.parametrize("working_directory", [REPO, REPO / "analysis/3_pid/cnn"])
def test_cnn_cli_help_runs_from_root_and_script_directory(working_directory, tmp_path):
    script = "analysis/3_pid/cnn/train.py" if working_directory == REPO else "train.py"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPO / "src")
    environment["MPLCONFIGDIR"] = str(tmp_path / "matplotlib")
    completed = subprocess.run(
        [sys.executable, script, "--help"],
        cwd=working_directory,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--cluster-hits" in completed.stdout
    assert "--external-pion-csv" in completed.stdout
