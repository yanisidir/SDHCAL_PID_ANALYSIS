#!/usr/bin/env python3
"""Hit-level GNN energy reconstruction for pion and proton showers.

The script reads the sharded ROOT files under ``new_data``. Each event already
contains its digitized hits and true ``energy``. The continuous 200k production
is used for training and internal calibration; the independent fixed-energy
100k production is kept intact as the external validation sample. The model can
be particle-specific or PID-agnostic on the combined pion+proton sample.
"""

from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer, RobustScaler
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, GraphConv, GraphNorm, global_max_pool, global_mean_pool

from sdhcal import calib, clustering, metrics, paths, runlog
from sdhcal.graph import build_knn_graph
from sdhcal.hit_pid import FEATURE_NAMES, seed_everything
from sdhcal.target import make_target

LOGGER = logging.getLogger("gnn_energy_reco")
PARTICLE_MODES = ("pi", "proton", "mix-pi-proton")


@dataclass
class ShowerEvent:
    coordinates: np.ndarray
    time: np.ndarray
    threshold: np.ndarray
    energy: float
    particle: str
    event_id: str
    #: (n, 4) of [frac_thr1, frac_thr2, frac_thr3, size] when the hits were
    #: grouped into clusters; None when each node is a single pad.
    composition: np.ndarray | None = None
    #: Semi-digital chi2 estimate of this event's energy. The network can be
    #: asked to predict the correction to it rather than the energy itself.
    baseline: float = 0.0


@dataclass
class ArrayHitPreprocessor:
    seed: int
    spatial_scaler: RobustScaler | None = None
    time_scaler: QuantileTransformer | None = None

    def fit(self, events: list[ShowerEvent], indices: np.ndarray, max_hits: int) -> None:
        """Fit on a deterministic hit sample from training events only."""

        if max_hits < 1:
            raise ValueError("scaler-max-hits must be positive")
        rng = np.random.default_rng(self.seed)
        coordinate_parts: list[np.ndarray] = []
        time_parts: list[np.ndarray] = []
        remaining = max_hits
        for index in rng.permutation(indices):
            event = events[int(index)]
            take = min(len(event.time), remaining)
            if take < len(event.time):
                selected = rng.choice(len(event.time), size=take, replace=False)
            else:
                selected = slice(None)
            coordinate_parts.append(event.coordinates[selected])
            time_parts.append(event.time[selected])
            remaining -= take
            if remaining == 0:
                break
        if not coordinate_parts:
            raise ValueError("No training hits available to fit the preprocessor")
        coordinates = np.vstack(coordinate_parts)
        timing = np.concatenate(time_parts).reshape(-1, 1)
        self.spatial_scaler = RobustScaler().fit(coordinates)
        self.time_scaler = QuantileTransformer(
            n_quantiles=min(1000, len(timing)),
            output_distribution="normal",
            random_state=self.seed,
        ).fit(timing)

    def transform(self, event: ShowerEvent) -> np.ndarray:
        if self.spatial_scaler is None or self.time_scaler is None:
            raise RuntimeError("Preprocessor must be fitted before transform")
        spatial = self.spatial_scaler.transform(event.coordinates)
        timing = self.time_scaler.transform(event.time.reshape(-1, 1))
        if event.composition is not None:
            # A cluster has no single threshold: it has a composition and a size.
            # log1p keeps a 40-pad cluster from dwarfing a 2-pad one numerically.
            size = np.log1p(event.composition[:, 3:4])
            return np.hstack([spatial, timing, event.composition[:, :3], size]).astype(np.float32)
        if not np.isin(event.threshold, [1, 2, 3]).all():
            raise ValueError(f"Invalid threshold in event {event.event_id}")
        one_hot = np.eye(3, dtype=np.float32)[event.threshold.astype(np.int64) - 1]
        return np.hstack([spatial, timing, one_hot]).astype(np.float32)


def _tree(handle, name: str, path: Path):
    if name not in handle:
        raise KeyError(f"TTree {name!r} not found in {path}")
    return handle[name]


def load_particle_events(
    particle: str,
    shard_paths: list[Path],
    *,
    sample_role: str,
    tree_name: str,
    sigma_time: float,
    seed: int,
    max_events_per_shard: int | None,
    require_contained: bool,
    cluster_hits: bool = False,
    baseline_coefficients: np.ndarray | None = None,
) -> list[ShowerEvent]:
    """Read self-contained hit/energy shards from one simulation production."""

    import awkward as ak
    import uproot

    if not shard_paths:
        raise FileNotFoundError(f"No ROOT shard found for {sample_role}/{particle}")
    rng = np.random.default_rng(seed)
    events: list[ShowerEvent] = []
    rejected = {"invalid": 0, "empty": 0, "not_contained": 0}
    for shard_path in shard_paths:
        if not shard_path.is_file():
            raise FileNotFoundError(shard_path)
        with uproot.open(shard_path) as handle:
            tree = _tree(handle, tree_name, shard_path)
            available = set(tree.keys())
            if {"x", "y", "z"} <= available:
                coordinate_columns = ("x", "y", "z")
            elif {"I", "J", "K"} <= available:
                coordinate_columns = ("I", "J", "K")
            else:
                raise KeyError(
                    f"{shard_path} needs vector branches x/y/z or I/J/K; "
                    f"available: {sorted(available)}"
                )
            energy_column = "energy" if "energy" in available else "primaryEnergy"
            if energy_column not in available:
                raise KeyError(f"{shard_path} has neither energy nor primaryEnergy")
            branches = [*coordinate_columns, "time", "thr", energy_column]
            if "eventNumber" in available:
                branches.append("eventNumber")
            if (require_contained or cluster_hits) and "K" not in branches:
                if "K" not in available:
                    raise KeyError("--require-contained and --cluster-hits need the K layer branch")
                branches.append("K")
            if cluster_hits:
                for name in ("I", "J"):
                    if name not in available:
                        raise KeyError(f"--cluster-hits needs the {name} cell-index branch")
                    if name not in branches:
                        branches.append(name)
            missing = [name for name in branches if name not in available]
            if missing:
                raise KeyError(f"{shard_path} missing branches: {missing}")
            entry_stop = (
                max_events_per_shard
                if max_events_per_shard and max_events_per_shard > 0
                else None
            )
            arrays = tree.arrays(branches, entry_stop=entry_stop, library="ak", how=dict)

        n_events = len(arrays["thr"])
        energies = np.asarray(ak.to_numpy(arrays[energy_column]), dtype=np.float64)
        event_numbers = (
            np.asarray(ak.to_numpy(arrays["eventNumber"]), dtype=np.int64)
            if "eventNumber" in arrays
            else np.arange(n_events, dtype=np.int64)
        )
        for row in range(n_events):
            event_number = int(event_numbers[row])
            energy = float(energies[row])
            if not np.isfinite(energy) or energy <= 0:
                rejected["invalid"] += 1
                continue
            coordinates = np.column_stack(
                [
                    np.asarray(ak.to_numpy(arrays[name][row]), dtype=np.float32)
                    for name in coordinate_columns
                ]
            )
            timing = np.asarray(ak.to_numpy(arrays["time"][row]), dtype=np.float32)
            threshold = np.asarray(ak.to_numpy(arrays["thr"][row]), dtype=np.int64)
            if len(timing) == 0:
                rejected["empty"] += 1
                continue
            if len(coordinates) != len(timing) or len(threshold) != len(timing):
                raise ValueError(f"Inconsistent jagged lengths in {shard_path.name}:{event_number}")
            if not np.isfinite(coordinates).all() or not np.isfinite(timing).all():
                rejected["invalid"] += 1
                continue
            if not np.isin(threshold, [1, 2, 3]).all():
                rejected["invalid"] += 1
                continue
            if require_contained and np.any(
                np.asarray(ak.to_numpy(arrays["K"][row]), dtype=np.int64) == 47
            ):
                rejected["not_contained"] += 1
                continue
            if sigma_time:
                timing = timing + rng.normal(0.0, sigma_time, len(timing)).astype(np.float32)

            baseline = 0.0
            if baseline_coefficients is not None:
                counts = [float(np.count_nonzero(threshold == level)) for level in (1, 2, 3)]
                baseline = float(
                    calib.ereco(
                        np.array([counts[0]]), np.array([counts[1]]), np.array([counts[2]]),
                        baseline_coefficients,
                    )[0]
                )
                if not np.isfinite(baseline) or baseline <= 0.0:
                    rejected["no_baseline"] += 1
                    continue

            composition = None
            if cluster_hits:
                # Done once here, not once per epoch: that is the whole point.
                labels = clustering.cluster_labels(
                    np.asarray(ak.to_numpy(arrays["I"][row]), dtype=np.int64),
                    np.asarray(ak.to_numpy(arrays["J"][row]), dtype=np.int64),
                    np.asarray(ak.to_numpy(arrays["K"][row]), dtype=np.int64),
                )
                summary = clustering.cluster_features(
                    coordinates[:, 0], coordinates[:, 1], coordinates[:, 2],
                    timing, threshold, labels,
                )
                names = clustering.CLUSTER_FEATURE_NAMES
                coordinates = summary[:, [names.index("x"), names.index("y"), names.index("z")]]
                timing = summary[:, names.index("time")]
                composition = summary[
                    :, [names.index(f"frac_thr{level}") for level in (1, 2, 3)] + [names.index("size")]
                ]
                threshold = np.ones(len(coordinates), dtype=np.int64)  # unused once clustered

            events.append(
                ShowerEvent(
                    coordinates=coordinates,
                    time=timing,
                    threshold=threshold,
                    composition=composition,
                    energy=energy,
                    particle=particle,
                    baseline=baseline,
                    event_id=f"{sample_role}:{particle}:{shard_path.stem}:{event_number}",
                )
            )
        LOGGER.info("%s/%s: read %s (%d cumulative events)", sample_role, particle, shard_path.name, len(events))
    LOGGER.info("%s/%s: loaded %d events; rejected %s", sample_role, particle, len(events), rejected)
    if not events:
        raise RuntimeError(f"No usable {particle} event was loaded")
    return events


def split_train_calibration(
    events: list[ShowerEvent], *, calibration_size: float, seed: int, energy_bins: int
) -> tuple[np.ndarray, np.ndarray]:
    """Split the 200k production into fit and internal-calibration subsets."""

    if not 0 < calibration_size < 1:
        raise ValueError("calibration-size must lie strictly between zero and one")
    indices = np.arange(len(events))
    particles = np.asarray([event.particle for event in events])
    energies = np.asarray([event.energy for event in events])

    last_error: Exception | None = None
    for n_bins in range(max(1, energy_bins), 0, -1):
        strata = np.empty(len(events), dtype=object)
        for particle in np.unique(particles):
            selected = particles == particle
            values = energies[selected]
            edges = np.unique(np.quantile(values, np.linspace(0, 1, n_bins + 1)))
            bins = np.digitize(values, edges[1:-1], right=False)
            strata[selected] = [f"{particle}:{value}" for value in bins]
        try:
            train, calibration = train_test_split(
                indices, test_size=calibration_size, random_state=seed, stratify=strata
            )
            LOGGER.info("Split stratified with %d energy quantile bins", n_bins)
            return train, calibration
        except ValueError as error:
            last_error = error
    raise ValueError(f"Cannot create a stratified train/calibration split: {last_error}")


class EnergyGraphDataset(Dataset):
    """Construct graphs on demand so the full 600k-event set fits in memory."""

    def __init__(
        self,
        events: list[ShowerEvent],
        indices: np.ndarray,
        preprocessor: ArrayHitPreprocessor,
        args: argparse.Namespace,
    ) -> None:
        self.events = events
        self.indices = np.asarray(indices, dtype=np.int64)
        self.preprocessor = preprocessor
        self.k = args.k
        self.time_window = args.time_window
        self.causal = args.causal
        self.undirected = args.undirected
        self.knn_chunk_size = args.knn_chunk_size
        # DeepSets treats the shower as a point cloud, so no edges are needed.
        # Skipping them is not a detail: building the kNN graph costs about 31 ms
        # per event per epoch and dominates everything else on CPU.
        self.build_edges = args.architecture != "deepsets"

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> Data:
        index = int(self.indices[position])
        event = self.events[index]
        features = self.preprocessor.transform(event)
        if self.build_edges:
            edge_index = build_knn_graph(
                features,
                event.time,
                k=self.k,
                time_window=self.time_window,
                causal=self.causal,
                undirected=self.undirected,
                chunk_size=self.knn_chunk_size,
            )
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        return Data(
            x=torch.from_numpy(features),
            edge_index=edge_index,
            y=torch.tensor([event.energy], dtype=torch.float32),
            baseline=torch.tensor([event.baseline], dtype=torch.float32),
            event_ref=torch.tensor([index], dtype=torch.long),
            num_nodes=len(features),
        )


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
        pooled = torch.cat(
            [global_max_pool(x, data.batch), global_mean_pool(x, data.batch)], dim=1
        )
        # Global mean/max alone discard shower multiplicity, one of the strongest
        # energy observables. Add log(1 + N_hits) explicitly.
        counts = torch.bincount(data.batch, minlength=pooled.size(0)).float().log1p().unsqueeze(1)
        return self.regressor(torch.cat([pooled, counts], dim=1)).squeeze(1)


class EnergyDeepSets(nn.Module):
    """Permutation-invariant regressor over the hits, with no graph at all.

    Each node is updated from itself and from the current global summary, which
    is what makes the network aware of the whole shower without any edge. The
    DHCAL study (arXiv:2501.00128) found this both more accurate and far cheaper
    than attention over a graph: roughly a third of the compute and a tenth of
    the memory. Here the saving is larger still, because the edges themselves
    were the dominant cost.
    """

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
            summary = global_mean_pool(h, data.batch)          # one vector per shower
            h = norm(h + block(torch.cat([h, summary[data.batch]], dim=1)))
        pooled = torch.cat(
            [global_max_pool(h, data.batch), global_mean_pool(h, data.batch)], dim=1
        )
        # Same reasoning as the graph model: pooling alone discards multiplicity.
        counts = torch.bincount(data.batch, minlength=pooled.size(0)).float().log1p().unsqueeze(1)
        return self.regressor(torch.cat([pooled, counts], dim=1)).squeeze(1)


def target_mode(model) -> bool:
    """Whether this model predicts a correction to the chi2 estimate.

    Read off the model rather than passed to every function. It used to be an
    argument, and one of the four call sites was left without it: the loop
    scored the network against log(E) while it was learning log(E/E_chi2), so
    the metric that drives early stopping sat at the log of the mean energy and
    never moved. Binding the mode to the object makes that disagreement
    unrepresentable.
    """
    return bool(getattr(model, "use_baseline", False))


def transformed_target(batch, use_baseline: bool) -> torch.Tensor:
    """What the network is asked to predict.

    With ``use_baseline`` the target is ``log(E_true / E_chi2)``: the network
    corrects an existing estimator instead of rebuilding the energy from nothing.
    The CMS HGCAL study (arXiv:2406.11937) trains on exactly this ratio. It is a
    far better conditioned problem - the target sits near zero over the whole
    energy range, so the loss is not dominated by the high-energy events - and it
    starts from the semi-digital calibration rather than competing with it.
    """
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


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    core = metrics.core_bias_resolution(metrics.relative_residual(y_true, y_pred))
    r2 = float("nan") if np.ptp(y_true) == 0 else float(r2_score(y_true, y_pred))
    return {
        "rmse_gev": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "mae_gev": float(mean_absolute_error(y_true, y_pred)),
        "r2": r2,
        "bias": core.bias,
        "bias_err": core.bias_err,
        "resolution": core.sigma,
        "resolution_err": core.sigma_err,
        "energy_scale": float(np.mean(y_pred) / np.mean(y_true)),
        "rms_rel_about_zero": metrics.rms_relative_about_zero(y_true, y_pred),
        "n_validation": int(len(y_true)),
    }


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
    """Use the fixed 100k beam-energy points as their natural plot bins."""

    points = np.unique(np.asarray(y_true, dtype=float))
    if len(points) == 1:
        half_width = max(0.5, 0.05 * abs(float(points[0])))
        return np.asarray([points[0] - half_width, points[0] + half_width])
    if 1 < len(points) <= 100:
        middle = 0.5 * (points[:-1] + points[1:])
        return np.concatenate(
            [[points[0] - 0.5 * (points[1] - points[0])], middle,
             [points[-1] + 0.5 * (points[-1] - points[-2])]]
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
    top.errorbar(
        binned["center"], binned["bias"], yerr=binned["bias_err"], fmt="o", capsize=3
    )
    top.axhline(0, color="0.4", linestyle="--")
    top.set_ylabel("Bias <(Ereco-Etrue)/Etrue>")
    top.grid(alpha=0.3)
    bottom.errorbar(
        binned["center"], binned["sigma"], yerr=binned["sigma_err"], fmt="o", capsize=3
    )
    bottom.set_xlabel("True energy [GeV]")
    bottom.set_ylabel("Resolution")
    bottom.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return binned


def selected_particles(mode: str) -> tuple[str, ...]:
    return ("pi", "proton") if mode == "mix-pi-proton" else (mode,)


LOT_DIRECTORIES = {
    "train": {"pi": "200k_pi-", "proton": "200k_proton"},
    "validation": {"pi": "100k_pi-_discret", "proton": "100k_proton_discret"},
}


def input_shards(args: argparse.Namespace, role: str, particle: str) -> list[Path]:
    custom = getattr(args, f"{particle}_{role}_dir")
    directory = Path(custom) if custom else paths.hit_shards_dir() / LOT_DIRECTORIES[role][particle]
    if not directory.is_dir():
        raise FileNotFoundError(
            f"Missing {role} directory {directory}. Set SDHCAL_HIT_SHARDS_DIR or pass "
            f"--{particle}-{role}-dir."
        )
    shards = sorted(
        directory.glob("*_digitized.root"),
        key=lambda path: [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path.name)],
    )
    if args.max_shards_per_particle:
        shards = shards[: args.max_shards_per_particle]
    if not shards:
        raise FileNotFoundError(f"No '*_digitized.root' file in {directory}")
    return shards


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--particles", choices=PARTICLE_MODES, default="mix-pi-proton")
    parser.add_argument("--pi-train-dir", type=Path)
    parser.add_argument("--pi-validation-dir", type=Path)
    parser.add_argument("--proton-train-dir", type=Path)
    parser.add_argument("--proton-validation-dir", type=Path)
    parser.add_argument("--tree-name", default="tree")
    parser.add_argument("--output-dir", type=Path, default=paths.output_dir() / "energy_reco_gnn")
    parser.add_argument("--comment", default="")
    parser.add_argument("--max-shards-per-particle", type=int)
    parser.add_argument("--max-events-per-shard", type=int)
    parser.add_argument("--calibration-size", type=float, default=0.1)
    parser.add_argument("--split-energy-bins", type=int, default=10)
    parser.add_argument("--sigma-time", type=float, default=0.1)
    parser.add_argument("--require-contained", action="store_true")
    parser.add_argument("--scaler-max-hits", type=int, default=500_000)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--time-window", type=float, default=2.0)
    parser.add_argument("--causal", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--undirected", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--knn-chunk-size", type=int, default=512)
    parser.add_argument(
        "--architecture",
        choices=("gat", "deepsets"),
        default="gat",
        help="deepsets drops the kNN graph entirely, which is where the CPU time goes",
    )
    parser.add_argument("--deepsets-blocks", type=int, default=6)
    parser.add_argument(
        "--target",
        choices=("energy", "ratio-chi2"),
        default="energy",
        help="ratio-chi2 trains on log(E_true / E_chi2): the network corrects the "
             "semi-digital calibration instead of replacing it",
    )
    parser.add_argument(
        "--baseline-coefficients",
        choices=("pi", "self"),
        default="pi",
        help="whose chi2 coefficients give the baseline. 'pi' applies the pion "
             "calibration to every event, which needs no particle identity and so "
             "is reproducible at inference; 'self' uses the true species, which is "
             "an oracle and must be compared only against oracle-routed chi2",
    )
    parser.add_argument(
        "--cluster-hits",
        action="store_true",
        help="group neighbouring fired pads within a layer and treat each group as one "
             "node; about four times fewer nodes, computed once at load rather than per epoch",
    )
    parser.add_argument("--bias-correction", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--n-energy-bins", type=int, default=13)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = {
        "batch-size": args.batch_size,
        "epochs": args.epochs,
        "patience": args.patience,
        "k": args.k,
        "knn-chunk-size": args.knn_chunk_size,
        "scaler-max-hits": args.scaler_max_hits,
        "split-energy-bins": args.split_energy_bins,
        "n-energy-bins": args.n_energy_bins,
    }
    bad = [name for name, value in positive.items() if value < 1]
    if bad:
        raise ValueError(f"These arguments must be positive: {', '.join(bad)}")
    if args.hidden_dim < 4 or args.hidden_dim % 4:
        raise ValueError("hidden-dim must be >= 4 and divisible by four")
    if args.sigma_time < 0:
        raise ValueError("sigma-time must be non-negative")
    if not 0 < args.calibration_size < 1:
        raise ValueError("calibration-size must lie strictly between zero and one")
    if args.learning_rate <= 0:
        raise ValueError("learning-rate must be positive")
    if not 0 <= args.dropout < 1:
        raise ValueError("dropout must be in [0, 1)")
    for name in ("max_shards_per_particle", "max_events_per_shard"):
        value = getattr(args, name)
        if value is not None and value < 1:
            raise ValueError(f"{name.replace('_', '-')} must be positive when provided")


def main() -> None:
    args = parse_args()
    validate_args(args)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)8s | %(message)s",
        datefmt="%H:%M:%S",
    )
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode_dir = args.output_dir / args.particles
    log = runlog.RunLogger(mode_dir, name="gnn_energy")
    configuration = vars(args).copy()
    configuration["device"] = str(device)
    configuration["input_shards"] = {
        f"{role}_{particle}": [str(path) for path in input_shards(args, role, particle)]
        for role in ("train", "validation")
        for particle in selected_particles(args.particles)
    }
    node_features = (
        [*clustering.CLUSTER_FEATURE_NAMES[:4], *clustering.CLUSTER_FEATURE_NAMES[5:], "log1p_size"]
        if args.cluster_hits
        else list(FEATURE_NAMES)
    )
    log.start(configuration, features=[*node_features, "log1p_n_nodes"])
    log.log_comment(args.comment)

    train_events: list[ShowerEvent] = []
    validation_events: list[ShowerEvent] = []
    use_baseline = args.target == "ratio-chi2"

    for offset, particle in enumerate(selected_particles(args.particles)):
        coefficients = None
        if use_baseline:
            which = "pi" if args.baseline_coefficients == "pi" else particle
            entry = calib.load(which)
            coefficients = entry.values
            LOGGER.info(
                "%s: chi2 baseline from the %s coefficients (fitted %s)",
                particle,
                which,
                "in-sample" if entry.fitted_in_sample else "out-of-sample",
            )
        train_events.extend(
            load_particle_events(
                particle,
                input_shards(args, "train", particle),
                sample_role="train",
                tree_name=args.tree_name,
                sigma_time=args.sigma_time,
                seed=args.seed + offset,
                max_events_per_shard=args.max_events_per_shard,
                require_contained=args.require_contained,
                cluster_hits=args.cluster_hits,
                baseline_coefficients=coefficients,
            )
        )
        validation_events.extend(
            load_particle_events(
                particle,
                input_shards(args, "validation", particle),
                sample_role="validation",
                tree_name=args.tree_name,
                sigma_time=args.sigma_time,
                seed=args.seed + 10_000 + offset,
                max_events_per_shard=args.max_events_per_shard,
                require_contained=args.require_contained,
                cluster_hits=args.cluster_hits,
                baseline_coefficients=coefficients,
            )
        )
    train_indices, calibration_indices = split_train_calibration(
        train_events,
        calibration_size=args.calibration_size,
        seed=args.seed,
        energy_bins=args.split_energy_bins,
    )
    events = [*train_events, *validation_events]
    split = {
        "train": train_indices,
        "calibration": calibration_indices,
        "validation": np.arange(len(train_events), len(events)),
    }
    LOGGER.info("Split sizes: %s", {name: len(indices) for name, indices in split.items()})

    preprocessor = ArrayHitPreprocessor(seed=args.seed)
    preprocessor.fit(events, split["train"], args.scaler_max_hits)
    datasets = {
        name: EnergyGraphDataset(events, indices, preprocessor, args)
        for name, indices in split.items()
    }
    LOGGER.info(
        "Graphs are constructed lazily per batch; dataset sizes: %s",
        {name: len(dataset) for name, dataset in datasets.items()},
    )
    loaders = {
        name: DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=name == "train",
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        for name, dataset in datasets.items()
    }

    if args.architecture == "deepsets":
        model = EnergyDeepSets(
            len(node_features), args.hidden_dim, args.dropout, n_blocks=args.deepsets_blocks
        ).to(device)
    else:
        model = EnergyGNN(len(node_features), args.hidden_dim, args.dropout).to(device)
    model.use_baseline = use_baseline
    LOGGER.info(
        "Architecture: %s (%d trainable parameters)",
        args.architecture,
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    checkpoint = log.path("models", f"energy_gnn_{args.particles}_run{log.run_id}.pt")
    history: dict[str, list[float]] = {"train_loss": [], "calibration_loss": []}
    best_calibration_loss = np.inf
    stale_epochs = 0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, loaders["train"], optimizer, device)
        # This is the number early stopping and checkpoint selection run on, so
        # it must be measured against the same target the model is trained on;
        # see target_mode().
        calibration_loss, _, _, _ = predict_forward(model, loaders["calibration"], device)
        scheduler.step(calibration_loss)
        history["train_loss"].append(train_loss)
        history["calibration_loss"].append(calibration_loss)
        LOGGER.info(
            "epoch=%03d train_loss=%.5f calibration_loss=%.5f lr=%.2e",
            epoch,
            train_loss,
            calibration_loss,
            optimizer.param_groups[0]["lr"],
        )
        if calibration_loss < best_calibration_loss:
            best_calibration_loss = calibration_loss
            stale_epochs = 0
            torch.save(model.state_dict(), checkpoint)
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                LOGGER.info("Early stopping after %d epochs", epoch)
                break

    model.load_state_dict(torch.load(checkpoint, map_location=device))
    _, calibration_forward, calibration_true, calibration_refs = predict_forward(
        model, loaders["calibration"], device
    )
    target = make_target("log", correct_bias=args.bias_correction)
    # In ratio mode the network predicts a correction, so the bias correction is
    # fitted on the ratio too, not on the energy.
    calibration_reference = (
        calibration_true / np.asarray([events[int(i)].baseline for i in calibration_refs])
        if use_baseline
        else calibration_true
    )
    correction = target.fit_correction(calibration_reference, calibration_forward)
    validation_loss, validation_forward, validation_true, source_indices = predict_forward(
        model, loaders["validation"], device
    )
    validation_prediction = target.inverse(validation_forward)
    if use_baseline:
        validation_prediction = validation_prediction * np.asarray(
            [events[int(index)].baseline for index in source_indices]
        )
    results = regression_metrics(validation_true, validation_prediction)
    results.update(
        validation_log_loss=validation_loss,
        best_calibration_loss=float(best_calibration_loss),
        target_bias_correction=float(correction),
    )
    LOGGER.info(
        "External validation RMSE=%.3f GeV MAE=%.3f GeV R2=%.4f bias=%+.4f resolution=%.4f",
        results["rmse_gev"],
        results["mae_gev"],
        results["r2"],
        results["bias"],
        results["resolution"],
    )

    preprocessor_path = log.path("models", f"preprocessor_{args.particles}_run{log.run_id}.joblib")
    target_path = log.path("models", f"target_{args.particles}_run{log.run_id}.joblib")
    joblib.dump(preprocessor, preprocessor_path)
    joblib.dump(target, target_path)
    training_plot = log.path("plots", f"training_{args.particles}_run{log.run_id}.png")
    linearity_plot = log.path("plots", f"linearity_{args.particles}_run{log.run_id}.png")
    resolution_plot = log.path("plots", f"bias_resolution_{args.particles}_run{log.run_id}.png")
    plot_training(history, training_plot)
    energy_bins = validation_energy_bins(validation_true, args.n_energy_bins)
    plot_linearity(validation_true, validation_prediction, linearity_plot, energy_bins)
    binned = plot_bias_resolution(
        validation_true, validation_prediction, resolution_plot, energy_bins
    )
    prediction_path = log.path(
        "arrays", f"validation_predictions_{args.particles}_run{log.run_id}.npz"
    )
    np.savez_compressed(
        prediction_path,
        y_true=validation_true,
        y_pred=validation_prediction,
        source_index=source_indices,
        event_id=np.asarray([events[int(index)].event_id for index in source_indices]),
        particle=np.asarray([events[int(index)].particle for index in source_indices]),
        bin_center=binned["center"],
        bin_bias=binned["bias"],
        bin_resolution=binned["sigma"],
        train_loss=np.asarray(history["train_loss"]),
        calibration_loss=np.asarray(history["calibration_loss"]),
    )
    log.log_metrics(**results)
    log.log_artifacts(
        checkpoint=checkpoint,
        preprocessor=preprocessor_path,
        target=target_path,
        predictions=prediction_path,
        training_plot=training_plot,
        linearity_plot=linearity_plot,
        bias_resolution_plot=resolution_plot,
    )
    print(f"results: {mode_dir}")


if __name__ == "__main__":
    main()
