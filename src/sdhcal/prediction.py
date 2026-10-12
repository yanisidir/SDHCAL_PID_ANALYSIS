"""Reusable tabular PID and energy inference operations.

The functions in this module deliberately preserve the inference conventions
of :mod:`tools.predict`: feature order comes from the artifact, preprocessing is
applied only when recorded, and calibration coefficients remain external.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

import numpy as np
import pandas as pd

from . import calib, paths
from .smearing import TimeSmearing, smear_time_columns

__all__ = [
    "CHI2_COLUMNS",
    "apply_smearing",
    "design",
    "predict_chi2",
    "predict_energy",
    "predict_pid",
]

logger = logging.getLogger("predict")

CHI2_COLUMNS = ("N1", "N2", "N3")


def apply_smearing(frame: pd.DataFrame, bundle: Mapping) -> pd.DataFrame:
    """Reapply the time smearing recorded by a model artifact."""
    recorded = bundle.get("smearing") or {}
    sigma = recorded.get("smear_sigma_ns", None)
    sigma = None if sigma in (None, "none") else float(sigma)
    columns = recorded.get("smear_columns")
    smearing = TimeSmearing(
        sigma_ns=sigma,
        columns=tuple(columns.split(";")) if isinstance(columns, str) else TimeSmearing().columns,
        seed=int(recorded.get("smear_seed", 42)),
    )
    return smear_time_columns(frame, smearing)


def design(frame: pd.DataFrame, bundle: Mapping) -> np.ndarray:
    """Build the model matrix in the artifact's exact feature order."""
    features = list(bundle["features"])
    missing = [column for column in features if column not in frame.columns]
    if missing:
        raise SystemExit(f"the evaluation lot lacks feature(s): {', '.join(missing)}")
    values = frame[features].to_numpy(dtype=np.float32)
    scaler = bundle.get("scaler")
    return scaler.transform(values) if scaler is not None else values


def predict_pid(bundle: Mapping, frame: pd.DataFrame, positive_pdg: int) -> np.ndarray:
    """Return the probability of ``positive_pdg`` using the recorded label map."""
    label_to_pdg = {int(key): int(value) for key, value in (bundle.get("label_to_pdg") or {}).items()}
    if not label_to_pdg:
        label_to_pdg = dict(enumerate(paths.PARTICLE_PDG[particle] for particle in bundle["particles"]))
    if positive_pdg not in label_to_pdg.values():
        raise SystemExit(
            f"the model does not know PDG {positive_pdg}; it knows {sorted(label_to_pdg.values())}"
        )
    index = next(label for label, pdg in label_to_pdg.items() if pdg == positive_pdg)

    probabilities = bundle["model"].predict_proba(design(frame, bundle))
    return np.asarray(probabilities)[:, index].astype(float)


def predict_energy(bundle: Mapping, frame: pd.DataFrame) -> np.ndarray:
    """Apply a regressor, its target inverse and its optional chi2 baseline."""
    raw = bundle["model"].predict(design(frame, bundle))
    target = bundle.get("target")
    value = np.asarray(target.inverse(raw) if target is not None else raw, dtype=float)

    if bundle.get("baseline") == "chi2_pi":
        coefficients = calib.load("pi")
        value = value * calib.ereco(
            frame["N1"].to_numpy(),
            frame["N2"].to_numpy(),
            frame["N3"].to_numpy(),
            coefficients.values,
        )
    elif bundle.get("baseline"):
        raise SystemExit(f"unknown baseline {bundle['baseline']!r} in the artifact")
    return value


def predict_chi2(frame: pd.DataFrame) -> np.ndarray:
    """Apply the historical semi-digital calibration per true species."""
    energy = np.full(len(frame), np.nan)
    for particle, pdg in paths.PARTICLE_PDG.items():
        mask = (frame["particlePDG"] == pdg).to_numpy()
        if not mask.any():
            continue
        try:
            coefficients = calib.load(particle)
        except (FileNotFoundError, KeyError) as error:
            raise SystemExit(f"no chi2 coefficients for {particle}: {error}") from error
        if coefficients.fitted_in_sample:
            logger.warning(
                "%s coefficients are flagged fitted_in_sample: this estimator is being "
                "compared on events it was fitted on",
                particle,
            )
        subset = frame.loc[mask]
        energy[mask] = calib.ereco(
            subset["N1"].to_numpy(),
            subset["N2"].to_numpy(),
            subset["N3"].to_numpy(),
            coefficients.values,
        )
    if np.isnan(energy).any():
        raise SystemExit(f"{int(np.isnan(energy).sum())} events got no chi2 estimate")
    return energy
