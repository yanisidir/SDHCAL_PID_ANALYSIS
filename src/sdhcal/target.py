"""Training on ``log E`` without inheriting its bias.

The regressors minimise a squared error on ``log E`` and the prediction is then
exponentiated. Training in log space is the right choice here - the relative
resolution is what matters, and the residuals are closer to Gaussian - but the
back-transform needs care.

A least-squares fit on ``log E`` estimates the conditional mean of the logarithm,
``mu(x) = E[log E | x]``, so exponentiating it gives the conditional *geometric*
mean. Because ``exp`` is convex, that always sits below the arithmetic mean the
calorimeter is supposed to report:

.. math:: \\exp(\\mathbb{E}[\\log E \\mid x]) = \\mathbb{E}[E \\mid x]\\, e^{-\\sigma^2/2} + O(\\sigma^3)

where ``sigma`` is the log-space spread of the energies compatible with the same
shower observables. With ``sigma = 0.20`` the reconstructed mean energy is about
2 % low - a scale error that lands squarely on the linearity and relative
deviation curves, where it is easy to mistake for a detector effect.

One sign convention is worth stating, because it is counter-intuitive and the
original code reported only the second quantity: the *mean of the relative
residual* ``(E_pred - E_true)/E_true`` moves the other way (``+sigma^2/2``),
since dividing by a fluctuating ``E_true`` over-weights the events where it is
small. The bias corrected here is the one on the ratio of means, i.e. what a
linearity profile shows.

:class:`LogTarget` applies Duan's smearing estimator: the back-transform is
multiplied by ``mean(exp(residual))`` measured on a held-out set, which corrects
the bias without assuming the residuals are exactly log-normal. The factor is
estimated on the validation set, never on the test set, and it is a single
multiplicative constant, so it rescales the energy without touching the
resolution.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["LogTarget", "IdentityTarget", "make_target"]


@dataclass
class LogTarget:
    """Forward/inverse transform of the regression target, with bias correction.

    Parameters
    ----------
    epsilon:
        Offset added before taking the logarithm, guarding against ``E = 0``.
    correct_bias:
        When true (the default), :meth:`fit_correction` must be called on a
        held-out set before :meth:`inverse`, and the back-transform includes
        Duan's smearing factor.
    """

    epsilon: float = 1e-9
    correct_bias: bool = True
    factor: float = field(default=1.0)
    _fitted: bool = field(default=False, repr=False)

    name = "log"

    def forward(self, y: np.ndarray) -> np.ndarray:
        """Map energies to the training target, ``log(E + epsilon)``."""
        y = np.asarray(y, dtype=float)
        if np.any(y < 0):
            raise ValueError("negative energies cannot be log-transformed")
        return np.log(y + self.epsilon)

    def fit_correction(self, y_true: np.ndarray, pred_forward: np.ndarray) -> float:
        """Estimate the smearing factor on a held-out set and store it.

        ``pred_forward`` are the model outputs in log space for the same events as
        ``y_true``. Returns the factor, which is 1 when the correction is disabled.
        """
        if not self.correct_bias:
            self.factor, self._fitted = 1.0, True
            return self.factor
        y_true = np.asarray(y_true, dtype=float)
        pred_forward = np.asarray(pred_forward, dtype=float)
        residual = self.forward(y_true) - pred_forward
        residual = residual[np.isfinite(residual)]
        if residual.size == 0:
            raise ValueError("no finite residual to estimate the smearing factor")
        self.factor = float(np.mean(np.exp(residual)))
        self._fitted = True
        return self.factor

    def inverse(self, pred_forward: np.ndarray) -> np.ndarray:
        """Map model outputs back to energies, applying the smearing factor."""
        if self.correct_bias and not self._fitted:
            raise RuntimeError(
                "fit_correction() must be called on a validation set before inverse(); "
                "without it the back-transform is biased low by exp(-sigma^2/2)"
            )
        return np.exp(np.asarray(pred_forward, dtype=float)) * self.factor

    def as_dict(self) -> dict:
        return {"target": self.name, "target_bias_correction": self.factor}


@dataclass
class IdentityTarget:
    """Train directly on the energy in GeV, for comparison with :class:`LogTarget`."""

    name = "linear"
    factor: float = 1.0

    def forward(self, y: np.ndarray) -> np.ndarray:
        return np.asarray(y, dtype=float)

    def fit_correction(self, y_true: np.ndarray, pred_forward: np.ndarray) -> float:  # noqa: ARG002
        return 1.0

    def inverse(self, pred_forward: np.ndarray) -> np.ndarray:
        return np.asarray(pred_forward, dtype=float)

    def as_dict(self) -> dict:
        return {"target": self.name, "target_bias_correction": 1.0}


def make_target(kind: str, *, correct_bias: bool = True):
    """Return the target transform named ``kind`` (``"log"`` or ``"linear"``)."""
    if kind == "log":
        return LogTarget(correct_bias=correct_bias)
    if kind == "linear":
        return IdentityTarget()
    raise ValueError(f"unknown target {kind!r}; expected 'log' or 'linear'")
