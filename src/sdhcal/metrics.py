"""Energy-reconstruction figures of merit, with bias and resolution kept apart.

The original code reported a single number::

    sigma_over_E = sqrt(mean(((E_pred - E_true) / E_true) ** 2))

which is the RMS of the relative residual *about zero*, so it mixes the
systematic bias with the event-by-event dispersion: a perfectly stable estimator
that is 10 % too low scores the same as an unbiased one with a 10 % spread. The
resolution curve then used a 90 %-truncated RMS with no truncation correction,
which understates a Gaussian sigma by ~21 %.

This module separates the two quantities and corrects the truncation:

* **bias** - mean of the relative residual, i.e. ``<E_reco>/E_true - 1``;
* **resolution** - standard deviation of the relative residual *around its own
  mean*, estimated on the distribution core and corrected back to a full-width
  Gaussian sigma.

:func:`rms_relative_about_zero` is kept, under a name that says what it is, so
older numbers remain reproducible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Literal

import numpy as np

__all__ = [
    "CoreEstimate",
    "relative_residual",
    "truncation_correction",
    "core_bias_resolution",
    "binned_bias_resolution",
    "rms_relative_about_zero",
    "energy_bins",
]

Method = Literal["truncated_rms", "iterative_gauss", "full_rms"]

_NORMAL = NormalDist()


@dataclass(frozen=True)
class CoreEstimate:
    """Bias and resolution of a relative-residual distribution.

    Attributes
    ----------
    bias, bias_err:
        Mean of ``(E_pred - E_true) / E_true`` and its standard error.
    sigma, sigma_err:
        Dispersion around that mean, corrected to a Gaussian sigma, and its
        standard error.
    n_used, n_total:
        Entries kept by the core selection, and entries supplied.
    method:
        Estimator used.
    """

    bias: float
    bias_err: float
    sigma: float
    sigma_err: float
    n_used: int
    n_total: int
    method: str


def relative_residual(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Return ``(y_pred - y_true) / y_true``, dropping non-positive true values."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    good = y_true > 0
    return (y_pred[good] - y_true[good]) / y_true[good]


def truncation_correction(frac: float) -> float:
    """Factor relating the RMS of a truncated Gaussian core to the full sigma.

    For a centred Gaussian restricted to the central fraction ``frac``, i.e. to
    ``[-z, z]`` with ``z = Phi^-1((1 + frac) / 2)``, the standard deviation of the
    truncated distribution is ``k * sigma`` with

    .. math:: k^2 = 1 - \\frac{2 z \\varphi(z)}{\\mathrm{frac}}

    so an uncorrected truncated RMS must be divided by ``k``. For ``frac = 0.9``,
    ``k ~= 0.7893``: the raw RMS90 underestimates sigma by about 21 %.
    """
    if not 0.0 < frac <= 1.0:
        raise ValueError(f"frac must lie in (0, 1], got {frac}")
    if frac == 1.0:
        return 1.0
    z = _NORMAL.inv_cdf(0.5 * (1.0 + frac))
    phi = _NORMAL.pdf(z)
    k2 = 1.0 - 2.0 * z * phi / frac
    if k2 <= 0:  # pragma: no cover - unreachable for frac in (0, 1]
        raise ValueError(f"degenerate truncation for frac={frac}")
    return math.sqrt(k2)


def _standard_errors(sigma: float, n: int) -> tuple[float, float]:
    """Standard errors on the mean and on the standard deviation of ``n`` draws."""
    if n < 2:
        return float("nan"), float("nan")
    return sigma / math.sqrt(n), sigma / math.sqrt(2.0 * (n - 1))


def core_bias_resolution(
    residual: np.ndarray,
    *,
    method: Method = "truncated_rms",
    frac: float = 0.9,
    n_sigma: float = 2.0,
    max_iter: int = 20,
    min_entries: int = 20,
) -> CoreEstimate:
    """Estimate the bias and the resolution of a relative-residual sample.

    Parameters
    ----------
    method:
        ``"truncated_rms"`` keeps the central ``frac`` of the sample by quantiles
        and divides the dispersion by :func:`truncation_correction`;
        ``"iterative_gauss"`` iterates mean/std within ``n_sigma`` until the
        selection is stable, the usual robust estimator of a Gaussian core with
        non-Gaussian tails; ``"full_rms"`` uses every entry with no correction.
    """
    residual = np.asarray(residual, dtype=float)
    residual = residual[np.isfinite(residual)]
    n_total = residual.size
    nan = float("nan")
    if n_total < min_entries:
        return CoreEstimate(nan, nan, nan, nan, 0, n_total, method)

    if method == "full_rms":
        core = residual
        bias = float(np.mean(core))
        sigma = float(np.std(core, ddof=1))
    elif method == "truncated_rms":
        lo, hi = np.quantile(residual, [(1.0 - frac) / 2.0, 1.0 - (1.0 - frac) / 2.0])
        core = residual[(residual >= lo) & (residual <= hi)]
        if core.size < min_entries:
            return CoreEstimate(nan, nan, nan, nan, int(core.size), n_total, method)
        bias = float(np.mean(core))
        sigma = float(np.std(core, ddof=1)) / truncation_correction(frac)
    elif method == "iterative_gauss":
        # The window must be built from the *corrected* sigma, otherwise the
        # iteration has no consistent fixed point: selecting within +-n_sigma of
        # the truncated standard deviation shrinks the window at every pass and
        # converges to an underestimate of sigma.
        kept = 2.0 * _NORMAL.cdf(n_sigma) - 1.0
        correction = truncation_correction(kept)
        bias, sigma = float(np.mean(residual)), float(np.std(residual, ddof=1))
        core = residual
        for _ in range(max_iter):
            if not math.isfinite(sigma) or sigma <= 0.0:
                break
            selected = residual[np.abs(residual - bias) <= n_sigma * sigma]
            if selected.size < min_entries:
                break
            core = selected
            new_bias = float(np.mean(core))
            new_sigma = float(np.std(core, ddof=1)) / correction
            converged = math.isclose(new_sigma, sigma, rel_tol=1e-4) and math.isclose(
                new_bias, bias, abs_tol=1e-6 + 1e-4 * abs(new_sigma)
            )
            bias, sigma = new_bias, new_sigma
            if converged:
                break
    else:
        raise ValueError(f"unknown method {method!r}")

    if core.size < min_entries or not math.isfinite(sigma):
        return CoreEstimate(nan, nan, nan, nan, int(core.size), n_total, method)

    bias_err, sigma_err = _standard_errors(sigma, int(core.size))
    return CoreEstimate(bias, bias_err, sigma, sigma_err, int(core.size), n_total, method)


def energy_bins(e_min: float, e_max: float, n_bins: int, *, log: bool = False) -> np.ndarray:
    """Return ``n_bins + 1`` bin edges, linear by default or logarithmic.

    The simulated spectrum spans 1-130 GeV, so logarithmic bins keep a comparable
    number of events per bin and a comparable relative width at both ends.
    """
    if e_min <= 0 and log:
        raise ValueError("logarithmic bins require e_min > 0")
    if log:
        return np.logspace(math.log10(e_min), math.log10(e_max), n_bins + 1)
    return np.linspace(e_min, e_max, n_bins + 1)


def binned_bias_resolution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    bins: np.ndarray | None = None,
    n_bins: int = 13,
    log_bins: bool = False,
    method: Method = "truncated_rms",
    frac: float = 0.9,
    min_entries: int = 20,
):
    """Bias and resolution of the relative residual versus true energy.

    Returns a dict of arrays with keys ``center``, ``bias``, ``bias_err``,
    ``sigma``, ``sigma_err``, ``count``, holding one entry per bin that reached
    ``min_entries``.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    good = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true > 0)
    y_true, y_pred = y_true[good], y_pred[good]

    if bins is None:
        bins = energy_bins(float(y_true.min()), float(y_true.max()), n_bins, log=log_bins)
    bins = np.asarray(bins, dtype=float)

    centers, biases, bias_errs, sigmas, sigma_errs, counts = [], [], [], [], [], []
    for lo, hi in zip(bins[:-1], bins[1:], strict=True):
        mask = (y_true >= lo) & (y_true < hi)
        if mask.sum() < min_entries:
            continue
        est = core_bias_resolution(
            relative_residual(y_true[mask], y_pred[mask]),
            method=method,
            frac=frac,
            min_entries=min_entries,
        )
        if not math.isfinite(est.sigma):
            continue
        centers.append(0.5 * (lo + hi))
        biases.append(est.bias)
        bias_errs.append(est.bias_err)
        sigmas.append(est.sigma)
        sigma_errs.append(est.sigma_err)
        counts.append(est.n_used)

    return {
        "center": np.asarray(centers),
        "bias": np.asarray(biases),
        "bias_err": np.asarray(bias_errs),
        "sigma": np.asarray(sigmas),
        "sigma_err": np.asarray(sigma_errs),
        "count": np.asarray(counts, dtype=int),
    }


def rms_relative_about_zero(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """RMS of the relative residual about zero: ``sqrt(mean(r**2))``.

    This is the quantity the original scripts logged as ``sigma_over_E``. It is
    ``sqrt(bias**2 + sigma**2)``, not a resolution, and is kept only so that the
    historical entries in the performance CSVs stay comparable.
    """
    r = relative_residual(y_true, y_pred)
    if r.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(r**2)))
