"""Standard metrics for comparing models on the same events.

Four PID models and four energy estimators cannot be compared by reading four
logs: they are trained on different inputs - shower variables for the BDT and the
MLP, raw hits for the CNN and the GNN - with different losses and different
epochs. What makes a comparison mean something is that every model is asked the
same question about the same events, and answered with the same figures of merit.

So training and evaluation are decoupled. A model writes one row per event of the
evaluation sample, and this module turns those rows into the numbers the
calorimetry literature quotes. Nothing here knows how a prediction was produced.

The join key is ``(particlePDG, eventNumber)``, globally unique since the shards
were given per-lot identifiers, so two models' predictions can be aligned event by
event rather than assumed to be in the same order.

**PID.** The figure of merit is background rejection at a fixed signal efficiency,
not accuracy: accuracy depends on the pion/proton mixture of the test sample,
which is an artefact of how the samples were generated, while a rejection quoted
at a stated efficiency does not. Working points are reported with binomial
uncertainties, and efficiency is also given per energy bin, because a classifier
that only works above 50 GeV is not the same tool as one that works everywhere.

**Energy.** The figures of merit are the response ``<E_rec>/E_true`` against
energy, and the resolution fitted to the usual

    sigma/E = a / sqrt(E)  (+)  b

where ``(+)`` is a quadratic sum, ``a`` is the stochastic term and ``b`` the
constant term that limits a calorimeter at high energy. Quoting ``a`` and ``b``
is what makes a number comparable with a published one; a single resolution at a
single energy is not.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import metrics

__all__ = [
    "PID_COLUMNS",
    "ENERGY_COLUMNS",
    "check_predictions",
    "align",
    "roc_curve",
    "rejection_at_efficiency",
    "working_point",
    "pid_summary",
    "efficiency_vs_energy",
    "ResolutionFit",
    "fit_resolution",
    "response_vs_energy",
    "energy_summary",
]

#: Columns a PID model must write. ``score`` is the probability of the positive
#: class, ``positive_pdg``; anything monotonically related to it works, since
#: every PID figure here is threshold-based.
PID_COLUMNS = ("eventNumber", "particlePDG", "primaryEnergy", "score")

#: Columns an energy estimator must write.
ENERGY_COLUMNS = ("eventNumber", "particlePDG", "primaryEnergy", "energyPred")


def check_predictions(frame: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Validate a prediction table against its contract.

    Raises rather than returning a partial frame: a missing column found here is
    a one-line fix in the model script, while the same column missing silently
    becomes a plot that is wrong in a way nobody notices.
    """
    required = {"pid": PID_COLUMNS, "energy": ENERGY_COLUMNS}[kind]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise KeyError(f"prediction table lacks {', '.join(missing)}; expected {', '.join(required)}")

    key = frame[["particlePDG", "eventNumber"]]
    if key.duplicated().any():
        n = int(key.duplicated().sum())
        raise ValueError(f"{n} duplicated (particlePDG, eventNumber) rows: predictions must be one per event")

    value = frame["score" if kind == "pid" else "energyPred"]
    if not np.isfinite(value.to_numpy(dtype=float)).all():
        raise ValueError("predictions contain NaN or infinity; a model must abstain explicitly, not silently")
    return frame


def align(tables: dict[str, pd.DataFrame], kind: str) -> dict[str, pd.DataFrame]:
    """Restrict every model to the events all of them predicted.

    Models drop events for their own reasons - a failed fit, a shower with too
    few hits for a graph. Comparing them on their own surviving subsets would
    reward whichever dropped the hardest events, so the comparison is made on the
    intersection, and the sizes are reported by the caller.
    """
    for name, frame in tables.items():
        try:
            check_predictions(frame, kind)
        except (KeyError, ValueError) as exc:
            raise type(exc)(f"{name}: {exc}") from exc

    common: set[tuple[int, int]] | None = None
    for frame in tables.values():
        keys = set(map(tuple, frame[["particlePDG", "eventNumber"]].to_numpy()))
        common = keys if common is None else (common & keys)
    if not common:
        raise ValueError("the models share no event; are they evaluated on the same sample?")

    out = {}
    for name, frame in tables.items():
        mask = [tuple(k) in common for k in frame[["particlePDG", "eventNumber"]].to_numpy()]
        out[name] = frame.loc[mask].sort_values(["particlePDG", "eventNumber"]).reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Identification
# ---------------------------------------------------------------------------
def roc_curve(labels: np.ndarray, score: np.ndarray) -> dict[str, np.ndarray]:
    """Signal efficiency and background efficiency at every threshold.

    Written out rather than taken from scikit-learn so that the same arrays feed
    the rejection curve and the working points without a second convention.
    """
    labels = np.asarray(labels, dtype=int)
    score = np.asarray(score, dtype=float)
    order = np.argsort(-score, kind="mergesort")
    labels, score = labels[order], score[order]

    signal = np.cumsum(labels == 1)
    background = np.cumsum(labels == 0)
    n_signal, n_background = int((labels == 1).sum()), int((labels == 0).sum())
    if not n_signal or not n_background:
        raise ValueError("a ROC needs both classes present")

    # Keep one point per distinct score, so a plateau does not become a staircase.
    last = np.r_[np.diff(score) != 0, True]
    eff_signal = np.r_[0.0, signal[last] / n_signal]
    eff_background = np.r_[0.0, background[last] / n_background]
    thresholds = np.r_[np.inf, score[last]]
    auc = float(np.trapezoid(eff_signal, eff_background))
    return {
        "efficiency_signal": eff_signal,
        "efficiency_background": eff_background,
        "threshold": thresholds,
        "auc": auc,
        "n_signal": n_signal,
        "n_background": n_background,
    }


def _binomial_error(k: float, n: int) -> float:
    """Standard error of an efficiency, floored so a 0 % or 100 % bin is not 0."""
    if n <= 0:
        return float("nan")
    p = min(max(k / n, 0.0), 1.0)
    return float(np.sqrt(max(p * (1.0 - p), 1.0 / (4.0 * n)) / n))


def rejection_at_efficiency(curve: dict[str, np.ndarray], target: float) -> dict[str, float]:
    """Background rejection ``1 / eff_background`` at a given signal efficiency.

    Rejection rather than background efficiency because it is what trigger and
    particle-flow studies quote, and because it separates good models: 1/0.5 and
    1/0.1 read as 2 and 10, not as 0.5 and 0.1.
    """
    eff_s = curve["efficiency_signal"]
    eff_b = curve["efficiency_background"]
    index = int(np.searchsorted(eff_s, target, side="left"))
    index = min(index, len(eff_s) - 1)
    background = float(eff_b[index])
    n_b = int(curve["n_background"])
    return {
        "target_efficiency": float(target),
        "efficiency_signal": float(eff_s[index]),
        "efficiency_background": background,
        "efficiency_background_err": _binomial_error(background * n_b, n_b),
        "rejection": float("inf") if background <= 0.0 else 1.0 / background,
        "threshold": float(curve["threshold"][index]),
    }


def working_point(labels: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, float]:
    """Efficiency, purity and counts at one threshold, with binomial errors."""
    labels = np.asarray(labels, dtype=int)
    selected = np.asarray(score, dtype=float) >= threshold
    true_positive = int(np.sum(selected & (labels == 1)))
    false_positive = int(np.sum(selected & (labels == 0)))
    n_signal = int(np.sum(labels == 1))
    n_selected = true_positive + false_positive

    efficiency = true_positive / n_signal if n_signal else float("nan")
    purity = true_positive / n_selected if n_selected else float("nan")
    return {
        "threshold": float(threshold),
        "efficiency": efficiency,
        "efficiency_err": _binomial_error(true_positive, n_signal),
        "purity": purity,
        "purity_err": _binomial_error(true_positive, n_selected),
        "n_selected": n_selected,
        "n_signal": n_signal,
    }


def pid_summary(
    frame: pd.DataFrame,
    *,
    positive_pdg: int,
    efficiencies: Sequence[float] = (0.5, 0.8, 0.9),
) -> dict[str, object]:
    """Every PID figure of merit for one model, on one event sample."""
    labels = (frame["particlePDG"].to_numpy() == positive_pdg).astype(int)
    score = frame["score"].to_numpy(dtype=float)
    curve = roc_curve(labels, score)

    out: dict[str, object] = {
        "auc": curve["auc"],
        "n_signal": curve["n_signal"],
        "n_background": curve["n_background"],
        # Accuracy at 0.5 is reported because readers expect it, with the caveat
        # that it moves with the class mixture while the rejections do not.
        "accuracy_at_0p5": float(np.mean((score >= 0.5).astype(int) == labels)),
        "working_points": {},
    }
    for target in efficiencies:
        out["working_points"][f"eff{int(round(target * 100))}"] = rejection_at_efficiency(curve, target)
    return out


def efficiency_vs_energy(
    frame: pd.DataFrame,
    *,
    positive_pdg: int,
    threshold: float,
    bins: np.ndarray | None = None,
) -> pd.DataFrame:
    """Signal efficiency and background efficiency per true-energy bin.

    The fixed-energy validation lot gives 20 clean points, so the default is to
    group by the distinct energies rather than impose arbitrary bins.
    """
    energy = frame["primaryEnergy"].to_numpy(dtype=float)
    labels = (frame["particlePDG"].to_numpy() == positive_pdg).astype(int)
    selected = frame["score"].to_numpy(dtype=float) >= threshold

    if bins is None:
        centres = np.unique(np.round(energy, 3))
        groups = [(c, np.isclose(energy, c)) for c in centres]
    else:
        index = np.digitize(energy, bins) - 1
        groups = [
            (0.5 * (bins[i] + bins[i + 1]), index == i) for i in range(len(bins) - 1) if np.any(index == i)
        ]

    rows = []
    for centre, mask in groups:
        sig, bkg = mask & (labels == 1), mask & (labels == 0)
        n_s, n_b = int(sig.sum()), int(bkg.sum())
        k_s, k_b = int((sig & selected).sum()), int((bkg & selected).sum())
        rows.append(
            {
                "energy": float(centre),
                "efficiency_signal": k_s / n_s if n_s else np.nan,
                "efficiency_signal_err": _binomial_error(k_s, n_s),
                "efficiency_background": k_b / n_b if n_b else np.nan,
                "efficiency_background_err": _binomial_error(k_b, n_b),
                "n_signal": n_s,
                "n_background": n_b,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Energy
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ResolutionFit:
    """Stochastic and constant terms of ``sigma/E = a/sqrt(E) (+) b``.

    ``a`` carries the sampling fluctuations and dominates at low energy; ``b`` is
    the term that does not improve with statistics - calibration spread,
    leakage, non-compensation - and sets the high-energy floor. A model that
    lowers ``b`` is better in a way that a single resolution figure hides.
    """

    stochastic: float
    constant: float
    stochastic_err: float = float("nan")
    constant_err: float = float("nan")
    n_points: int = 0
    rms_residual: float = float("nan")

    def __call__(self, energy: np.ndarray) -> np.ndarray:
        energy = np.asarray(energy, dtype=float)
        return np.sqrt((self.stochastic / np.sqrt(energy)) ** 2 + self.constant**2)

    def as_dict(self) -> dict[str, float]:
        return {
            "stochastic_term": self.stochastic,
            "stochastic_term_err": self.stochastic_err,
            "constant_term": self.constant,
            "constant_term_err": self.constant_err,
            "n_points": self.n_points,
            "rms_residual": self.rms_residual,
        }


def fit_resolution(
    energy: np.ndarray,
    sigma: np.ndarray,
    sigma_err: np.ndarray | None = None,
) -> ResolutionFit:
    """Fit ``sigma/E = a/sqrt(E) (+) b`` by linear least squares on the squares.

    Squaring makes the model linear in ``(a^2, b^2)``: ``sigma^2 = a^2/E + b^2``.
    That is solved exactly, with no starting values and no convergence to check -
    the same reasoning as the chi2 calibration in :mod:`sdhcal.calib`. The price
    is that the weights apply to ``sigma^2``, which is handled by propagating
    ``2 sigma d(sigma)``.

    A negative fitted square means the data do not support that term; it is
    clamped to zero and reported as such through ``rms_residual``.
    """
    energy = np.asarray(energy, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    good = np.isfinite(energy) & np.isfinite(sigma) & (energy > 0) & (sigma > 0)
    if good.sum() < 2:
        raise ValueError("a two-term resolution fit needs at least two usable points")

    energy, sigma = energy[good], sigma[good]
    design = np.column_stack([1.0 / energy, np.ones_like(energy)])
    target = sigma**2

    if sigma_err is not None:
        err = np.asarray(sigma_err, dtype=float)[good]
        weight = np.where(np.isfinite(err) & (err > 0), 1.0 / (2.0 * sigma * err), 1.0)
    else:
        weight = np.ones_like(sigma)

    weighted_design = design * weight[:, None]
    solution, *_ = np.linalg.lstsq(weighted_design, target * weight, rcond=None)
    a2, b2 = float(solution[0]), float(solution[1])

    residual = target - design @ solution
    rms = float(np.sqrt(np.mean(residual**2)))

    # The weights are 1/uncertainty on sigma^2 (propagated as 2 sigma d(sigma)),
    # so the inverse normal matrix is already the covariance. It is then inflated
    # by chi2/ndf when the two-term form does not describe the points: the per-bin
    # errors describe the sigmas, not the adequacy of the model, and quoting the
    # uninflated error would claim a precision the fit has not earned.
    try:
        covariance = np.linalg.inv(weighted_design.T @ weighted_design)
        ndf = max(len(energy) - 2, 1)
        chi2 = float(np.sum((weight * residual) ** 2))
        var = np.diag(covariance) * max(chi2 / ndf, 1.0)
    except np.linalg.LinAlgError:
        var = np.full(2, np.nan)

    a = float(np.sqrt(max(a2, 0.0)))
    b = float(np.sqrt(max(b2, 0.0)))
    a_err = float(np.sqrt(abs(var[0])) / (2.0 * a)) if a > 0 else float("nan")
    b_err = float(np.sqrt(abs(var[1])) / (2.0 * b)) if b > 0 else float("nan")
    return ResolutionFit(a, b, a_err, b_err, int(len(energy)), rms)


def response_vs_energy(
    frame: pd.DataFrame,
    *,
    method: str = "truncated_rms",
    bins: np.ndarray | None = None,
) -> pd.DataFrame:
    """Response, bias and resolution per true-energy point.

    ``response`` is ``<E_rec>/E_true``: a linearity curve. ``bias`` is the same
    thing minus one, kept separately because that is how the report quotes it.
    """
    truth = frame["primaryEnergy"].to_numpy(dtype=float)
    predicted = frame["energyPred"].to_numpy(dtype=float)

    if bins is None:
        centres = np.unique(np.round(truth, 3))
        groups = [(c, np.isclose(truth, c)) for c in centres]
    else:
        index = np.digitize(truth, bins) - 1
        groups = [
            (0.5 * (bins[i] + bins[i + 1]), index == i) for i in range(len(bins) - 1) if np.any(index == i)
        ]

    rows = []
    for centre, mask in groups:
        if mask.sum() < 20:
            continue
        core = metrics.core_bias_resolution(
            metrics.relative_residual(truth[mask], predicted[mask]), method=method
        )
        rows.append(
            {
                "energy": float(centre),
                "response": 1.0 + core.bias,
                "bias": core.bias,
                "bias_err": core.bias_err,
                "sigma": core.sigma,
                "sigma_err": core.sigma_err,
                "n_used": core.n_used,
                "n_total": core.n_total,
            }
        )
    return pd.DataFrame(rows)


@dataclass
class EnergySummary:
    """Per-energy curve plus the two terms that summarise it."""

    curve: pd.DataFrame
    resolution: ResolutionFit
    overall: dict[str, float] = field(default_factory=dict)

    def as_row(self) -> dict[str, float]:
        return {**self.overall, **self.resolution.as_dict()}


def energy_summary(frame: pd.DataFrame, *, method: str = "truncated_rms") -> EnergySummary:
    """Everything quoted for one energy estimator on one sample."""
    curve = response_vs_energy(frame, method=method)
    if curve.empty:
        raise ValueError("no energy point had enough events for a resolution estimate")
    fit = fit_resolution(curve["energy"].to_numpy(), curve["sigma"].to_numpy(), curve["sigma_err"].to_numpy())

    truth = frame["primaryEnergy"].to_numpy(dtype=float)
    predicted = frame["energyPred"].to_numpy(dtype=float)
    core = metrics.core_bias_resolution(metrics.relative_residual(truth, predicted), method=method)
    overall = {
        "n_events": int(len(frame)),
        "bias": core.bias,
        "bias_err": core.bias_err,
        "resolution": core.sigma,
        "resolution_err": core.sigma_err,
        # The largest departure from unity across the curve: a model can be
        # unbiased on average and badly non-linear point by point.
        "max_response_deviation": float(np.max(np.abs(curve["response"] - 1.0))),
    }
    return EnergySummary(curve=curve, resolution=fit, overall=overall)
