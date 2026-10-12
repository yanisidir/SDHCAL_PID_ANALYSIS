"""Reusable scientific logic for PID-dependent energy reconstruction.

This module preserves the distinct ``none``, ``true`` and classifier-routed
scenarios of the historical coupling study.  Dataset loading, plotting and CLI
handling remain in the study script.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import artifacts, calib, metrics, paths
from .prediction import CHI2_COLUMNS
from .smearing import TimeSmearing

__all__ = [
    "CHI2_COLUMNS",
    "CLASS_LABELS",
    "Config",
    "classifier_path",
    "event_table",
    "identify",
    "load_artifact",
    "needed_columns",
    "reconstruct_energy",
    "regressor_path",
    "required_regressors",
    "summarise",
]

logger = logging.getLogger("pid_energy_coupling")

CLASS_LABELS = {"pi": "pi-", "proton": "p", "kaon": "K0"}


@dataclass
class Config:
    tree_name: str = "paramsTree"
    energy_col: str = "primaryEnergy"
    pdg_col: str = "particlePDG"
    particles: Sequence[str] = field(default_factory=lambda: list(paths.active_particles()))
    pid: str = "bdt"
    reco: str = "bdt"
    smear_sigma_ns: float | None = 0.1
    smear_columns: Sequence[str] = ("tMin", "tMax", "tMean", "tSpread")
    require_contained: bool = False
    last_layer: int = 47
    check_disjoint: bool = True
    n_bins: int = 13
    core_method: str = "truncated_rms"
    classifier_path: Path | None = None
    regressor_paths: dict[str, Path] = field(default_factory=dict)

    def smearing(self, seed: int = 42) -> TimeSmearing:
        return TimeSmearing(
            sigma_ns=self.smear_sigma_ns,
            columns=tuple(self.smear_columns),
            seed=seed,
        )

    def loggable(self) -> dict[str, object]:
        output: dict[str, object] = {
            "pid": self.pid,
            "reco": self.reco,
            "particles": ";".join(self.particles),
            "check_disjoint": int(self.check_disjoint),
            "require_contained": int(self.require_contained),
            "n_bins": self.n_bins,
            "core_method": self.core_method,
            "classifier_path": str(self.classifier_path) if self.classifier_path else "",
            "regressor_paths": ";".join(
                f"{key}={value}" for key, value in sorted(self.regressor_paths.items())
            ),
        }
        output.update(self.smearing().as_dict())
        return output


def load_artifact(path: Path, what: str, *, kind: str | None = None) -> dict:
    """Load a model bundle and require its historical inference fields."""
    if not path.is_file():
        raise FileNotFoundError(
            f"{what} not found: {path}\n"
            "Set SDHCAL_MODEL_DIR, or train it first. The models are not versioned."
        )
    try:
        return artifacts.load_bundle(path, kind=kind)
    except artifacts.ArtifactError as error:
        raise ValueError(
            f"{what} at {path} is a bare estimator. Re-train it with the current scripts: the "
            "bundle must carry its feature list, so inference cannot silently feed the wrong "
            f"columns in the wrong order. ({error})"
        ) from error


def classifier_path(config: Config) -> Path:
    if config.classifier_path is not None:
        return config.classifier_path
    return paths.model_dir() / f"pid_{config.pid}.joblib"


def regressor_path(config: Config, mode: str) -> Path:
    override = config.regressor_paths.get(mode)
    if override is not None:
        return override
    return paths.model_dir() / f"energy_{config.reco}_{mode}.joblib"


def required_regressors(config: Config) -> list[str]:
    """Return the inclusive or per-species regressor buckets."""
    return ["mixed"] if config.pid == "none" else list(config.particles)


def needed_columns(config: Config, feature_sets: Sequence[Sequence[str]]) -> list[str]:
    columns = {config.energy_col, config.pdg_col, "eventNumber"}
    for features in feature_sets:
        columns.update(features)
    if config.reco == "chi2":
        columns.update(CHI2_COLUMNS)
    if config.require_contained:
        columns.add("touchesLastLayer")
    return sorted(columns)


def identify(
    frame: pd.DataFrame,
    config: Config,
    bundle: Mapping | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Return the exact historical routing labels and optional probabilities."""
    label_of = {paths.PARTICLE_PDG[particle]: index for index, particle in enumerate(config.particles)}
    true_label = frame[config.pdg_col].map(label_of).to_numpy()

    if config.pid == "none":
        return np.zeros(len(frame), dtype=int), None
    if config.pid == "true":
        logger.info("Routing by the TRUE particle: this is the perfect-PID upper bound")
        return true_label.astype(int), None

    assert bundle is not None
    model = bundle["model"]
    features = list(bundle["features"])
    declared = list(bundle.get("particles", config.particles))
    if declared != list(config.particles):
        raise ValueError(
            f"the classifier was trained on {declared} but this run targets {list(config.particles)}; "
            "the integer labels would not mean the same thing"
        )
    missing = [column for column in features if column not in frame.columns]
    if missing:
        raise KeyError(f"held-out sample lacks classifier feature(s): {', '.join(missing)}")

    probabilities = model.predict_proba(frame[features].to_numpy(dtype=np.float32))
    if probabilities.shape[1] != len(config.particles):
        raise ValueError(
            f"classifier returns {probabilities.shape[1]} classes, expected {len(config.particles)}"
        )
    return np.argmax(probabilities, axis=1).astype(int), probabilities


def reconstruct_energy(
    frame: pd.DataFrame,
    routing_label: np.ndarray,
    config: Config,
    regressors: Mapping[str, Mapping],
) -> np.ndarray:
    """Apply the historical estimator selected for each routed bucket."""
    energy = np.full(len(frame), np.nan, dtype=np.float64)
    buckets = {0: "mixed"} if config.pid == "none" else dict(enumerate(config.particles))

    for label, mode in buckets.items():
        mask = np.asarray(routing_label == label)
        if not mask.any():
            continue
        subset = frame.iloc[mask]

        if config.reco == "chi2":
            coefficients = calib.load("pi" if mode == "mixed" else mode)
            if mode == "mixed":
                logger.warning(
                    "chi2 with --pid none uses the pion coefficients for every event; "
                    "the semi-digital calibration has no particle-blind variant"
                )
            energy[mask] = calib.ereco(
                subset["N1"].to_numpy(),
                subset["N2"].to_numpy(),
                subset["N3"].to_numpy(),
                coefficients.values,
            )
            if coefficients.fitted_in_sample:
                logger.warning(
                    "chi2 coefficients for %s are flagged as fitted in-sample; the comparison "
                    "with the regressors is not like-for-like until fit.C is re-run",
                    mode,
                )
        else:
            bundle = regressors[mode]
            features = list(bundle["features"])
            missing = [column for column in features if column not in frame.columns]
            if missing:
                raise KeyError(f"held-out sample lacks regression feature(s): {', '.join(missing)}")
            values = subset[features].to_numpy(dtype=np.float32)
            if bundle.get("scaler") is not None:
                values = bundle["scaler"].transform(values)
            raw = bundle["model"].predict(values)
            target = bundle.get("target")
            energy[mask] = target.inverse(raw) if target is not None else raw

        logger.info("Routed %7d events through the %-6s estimator", int(mask.sum()), mode)

    if np.isnan(energy).any():
        raise RuntimeError(f"{int(np.isnan(energy).sum())} events received no energy estimate")
    return energy


def summarise(
    frame: pd.DataFrame,
    config: Config,
    routing_label: np.ndarray,
    energy: np.ndarray,
) -> tuple[dict[str, float], np.ndarray]:
    """Compute the historical per-species/global summary and confusion matrix."""
    label_of = {paths.PARTICLE_PDG[particle]: index for index, particle in enumerate(config.particles)}
    true_label = frame[config.pdg_col].map(label_of).to_numpy().astype(int)
    true_energy = frame[config.energy_col].to_numpy(dtype=np.float64)

    n_particles = len(config.particles)
    confusion = np.zeros((n_particles, n_particles), dtype=np.int64)
    if config.pid not in ("none",):
        for true, predicted in zip(true_label, routing_label):  # noqa: B905
            confusion[true, predicted] += 1

    results: dict[str, float] = {}
    if config.pid not in ("none", "true"):
        results["pid_accuracy"] = float((routing_label == true_label).mean())
        for index, particle in enumerate(config.particles):
            total = confusion[index].sum()
            results[f"pid_efficiency_{particle}"] = (
                float(confusion[index, index] / total) if total else float("nan")
            )

    core = metrics.core_bias_resolution(
        metrics.relative_residual(true_energy, energy), method=config.core_method
    )
    results.update(
        {
            "bias": core.bias,
            "bias_err": core.bias_err,
            "resolution": core.sigma,
            "resolution_err": core.sigma_err,
            "energy_scale": float(np.mean(energy) / np.mean(true_energy)),
            "rms_rel_about_zero": metrics.rms_relative_about_zero(true_energy, energy),
            "n_events": int(len(frame)),
        }
    )
    for index, particle in enumerate(config.particles):
        mask = true_label == index
        if not mask.any():
            continue
        estimate = metrics.core_bias_resolution(
            metrics.relative_residual(true_energy[mask], energy[mask]),
            method=config.core_method,
        )
        results[f"bias_{particle}"] = estimate.bias
        results[f"resolution_{particle}"] = estimate.sigma

    logger.info(
        "pid=%s reco=%s -> bias=%+.4f  resolution=%.4f  scale=%.4f%s",
        config.pid,
        config.reco,
        results["bias"],
        results["resolution"],
        results["energy_scale"],
        f"  PID accuracy={results['pid_accuracy']:.4f}" if "pid_accuracy" in results else "",
    )
    for particle in config.particles:
        if f"resolution_{particle}" in results:
            logger.info(
                "  true %-7s bias=%+.4f  resolution=%.4f",
                particle,
                results[f"bias_{particle}"],
                results[f"resolution_{particle}"],
            )
    return results, confusion


def event_table(
    frame: pd.DataFrame,
    config: Config,
    routing_label: np.ndarray,
    probabilities: np.ndarray | None,
    energy: np.ndarray,
) -> pd.DataFrame:
    """Build the per-event table consumed by the historical coupling plots."""
    label_of = {paths.PARTICLE_PDG[particle]: index for index, particle in enumerate(config.particles)}
    true_label = frame[config.pdg_col].map(label_of).to_numpy().astype(int)
    names = [CLASS_LABELS.get(particle, particle) for particle in config.particles]

    table = pd.DataFrame(
        {
            "PDG_true": frame[config.pdg_col].to_numpy(),
            "label_true_int": true_label,
            "label_true": [names[index] for index in true_label],
            "E_true_GeV": frame[config.energy_col].to_numpy(),
            "E_pred_GeV": energy,
            "dE_over_E": (energy - frame[config.energy_col].to_numpy()) / frame[config.energy_col].to_numpy(),
        }
    )
    if config.pid == "none":
        table["label_pred_int"] = -1
        table["label_pred"] = "inclusive"
    else:
        table["label_pred_int"] = routing_label
        table["label_pred"] = [names[index] for index in routing_label]
    if probabilities is not None:
        for index, particle in enumerate(config.particles):
            table[f"proba_{particle}"] = probabilities[:, index]
        table["proba_pred"] = probabilities[np.arange(len(probabilities)), routing_label]
    if "eventNumber" in frame.columns:
        table.insert(0, "eventNumber", frame["eventNumber"].to_numpy())
    return table
