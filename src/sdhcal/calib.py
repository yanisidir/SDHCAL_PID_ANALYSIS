"""The semi-digital chi2 calibration coefficients, stored once.

The nine coefficients of the quadratic calibration were duplicated by hand: as
``startParams`` in ``energy_reconstruction_Tminuit/fit.C`` and as
``PAR_PI`` / ``PAR_KAON`` / ``PAR_PROTON`` literals in the Python coupling
scripts. The two copies had already diverged - the kaon ``c0`` read
``1.22381e-14`` in C++ and ``9.83127e-15`` in Python - so the C++ and Python
reconstructions were not computing the same energy.

There is now a single JSON file, ``calibration/chi2_coefficients.json``, written
by the fit and read by everything else. :func:`load` returns the coefficients
together with the metadata that says what they were fitted on.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import paths

__all__ = [
    "Coefficients",
    "coefficients_path",
    "load",
    "load_all",
    "save",
    "ereco",
    "design_matrix",
    "wls_objective",
    "fit_wls",
    "PARAM_NAMES",
]

#: Order in which the nine coefficients are stored everywhere.
PARAM_NAMES = ("a0", "a1", "a2", "b0", "b1", "b2", "c0", "c1", "c2")

_DEFAULT_RELATIVE = Path("calibration/chi2_coefficients.json")

#: Per-sample-set calibration files. The two-class study fits on its own lot, so
#: its coefficients must not overwrite those of the three-class study: the two
#: are different productions and are not interchangeable.
_RELATIVE_BY_SET = {
    "130k": _DEFAULT_RELATIVE,
    "2class": Path("calibration/chi2_coefficients_2class.json"),
}


def coefficients_path() -> Path:
    """Location of the calibration file of the active sample set.

    Follows ``$SDHCAL_PARAMS_SET`` the same way :func:`sdhcal.paths.dataset` does,
    so that a stage reading the coefficients and a stage reading the parameters
    cannot end up on two different samples.
    """
    return paths.repo_root() / _RELATIVE_BY_SET.get(paths.params_set(), _DEFAULT_RELATIVE)


@dataclass(frozen=True)
class Coefficients:
    """Nine calibration coefficients plus the provenance of the fit."""

    particle: str
    values: np.ndarray
    errors: np.ndarray | None = None
    metadata: Mapping[str, object] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.values.shape != (9,):
            raise ValueError(f"expected 9 coefficients, got shape {self.values.shape}")

    def as_dict(self) -> dict[str, float]:
        return {name: float(v) for name, v in zip(PARAM_NAMES, self.values, strict=True)}

    @property
    def fitted_in_sample(self) -> bool:
        """True when the metadata does not establish an out-of-sample evaluation.

        The original fit used every event and then drew its linearity and
        resolution curves on the same events, which flatters the method relative
        to the machine-learning models that were evaluated on a held-out test set.
        """
        meta = self.metadata or {}
        return not bool(meta.get("fitted_on_train_split", False))


def ereco(
    n1: np.ndarray,
    n2: np.ndarray,
    n3: np.ndarray,
    values: Sequence[float],
) -> np.ndarray:
    """Reconstructed energy of the semi-digital quadratic model.

    ``alpha(N) = a0 + a1 N + a2 N^2`` and likewise for ``beta``, ``gamma``, with
    ``N = N1 + N2 + N3``; then ``E = alpha N1 + beta N2 + gamma N3``.
    """
    par = np.asarray(values, dtype=float)
    if par.shape != (9,):
        raise ValueError(f"expected 9 coefficients, got shape {par.shape}")
    n1 = np.asarray(n1, dtype=float)
    n2 = np.asarray(n2, dtype=float)
    n3 = np.asarray(n3, dtype=float)
    n = n1 + n2 + n3
    alpha = par[0] + par[1] * n + par[2] * n * n
    beta = par[3] + par[4] * n + par[5] * n * n
    gamma = par[6] + par[7] * n + par[8] * n * n
    return alpha * n1 + beta * n2 + gamma * n3


def design_matrix(n1: np.ndarray, n2: np.ndarray, n3: np.ndarray) -> np.ndarray:
    """Columns such that ``design_matrix(...) @ values`` equals :func:`ereco`.

    The quadratic model looks non-linear but is linear in its nine coefficients:
    expanding ``alpha(N) N1 + beta(N) N2 + gamma(N) N3`` gives one column per
    coefficient, in the order of :data:`PARAM_NAMES`.
    """
    n1 = np.asarray(n1, dtype=float)
    n2 = np.asarray(n2, dtype=float)
    n3 = np.asarray(n3, dtype=float)
    n = n1 + n2 + n3
    return np.column_stack([n1, n * n1, n * n * n1, n2, n * n2, n * n * n2, n3, n * n3, n * n * n3])


def wls_objective(
    n1: np.ndarray,
    n2: np.ndarray,
    n3: np.ndarray,
    e_true: np.ndarray,
    values: Sequence[float],
) -> float:
    """``sum (Ereco - Etrue)^2 / Etrue``, the quantity fit.C minimises.

    Not a chi2 despite the file names: the denominator is an energy, not a
    variance, so its value per degree of freedom is not a goodness of fit.
    """
    e_true = np.asarray(e_true, dtype=float)
    residual = ereco(n1, n2, n3, values) - e_true
    return float(np.sum(residual * residual / e_true))


def fit_wls(
    n1: np.ndarray,
    n2: np.ndarray,
    n3: np.ndarray,
    e_true: np.ndarray,
) -> np.ndarray:
    """Exact minimiser of :func:`wls_objective`, in closed form.

    Because the model is linear in its coefficients, the weighted least squares
    has a solution rather than a search: this returns the global minimum, with no
    starting values and no convergence to check. The original TMinuit fit
    approximated this same minimum from a hand-chosen seed.

    The columns span many orders of magnitude - ``N1`` against ``N^2 N1`` - so
    they are normalised before solving and the scaling undone afterwards.
    """
    e_true = np.asarray(e_true, dtype=float)
    if e_true.size == 0:
        raise ValueError("cannot fit on an empty sample")
    if np.any(e_true <= 0.0):
        raise ValueError("every true energy must be positive: it is the fit weight")

    design = design_matrix(n1, n2, n3)
    root_weight = 1.0 / np.sqrt(e_true)
    a = design * root_weight[:, None]
    b = e_true * root_weight

    scale = np.linalg.norm(a, axis=0)
    scale[scale == 0.0] = 1.0
    solution, *_ = np.linalg.lstsq(a / scale, b, rcond=None)
    return solution / scale


def load_all(path: Path | None = None) -> dict[str, Coefficients]:
    """Load every particle's coefficients from the calibration file."""
    path = Path(path) if path is not None else coefficients_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"calibration file not found: {path}\n"
            "Produce it by running the chi2 fit:\n"
            "    root -l -b -q 'energy_reconstruction_Tminuit/fit.C(\"pi\")'"
        )
    blob = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, Coefficients] = {}
    for particle, entry in blob.get("particles", {}).items():
        values = np.array([float(entry[name]) for name in PARAM_NAMES], dtype=float)
        errors = (
            np.array([float(entry["errors"][name]) for name in PARAM_NAMES], dtype=float)
            if isinstance(entry.get("errors"), Mapping)
            else None
        )
        meta = {k: v for k, v in entry.items() if k not in set(PARAM_NAMES) | {"errors"}}
        meta.setdefault("model", blob.get("model"))
        meta.setdefault("formula", blob.get("formula"))
        out[particle] = Coefficients(particle, values, errors, meta)
    return out


def load(particle: str, path: Path | None = None) -> Coefficients:
    """Load one particle's coefficients."""
    allc = load_all(path)
    if particle not in allc:
        raise KeyError(f"no calibration for {particle!r}; available: {sorted(allc)}")
    return allc[particle]


def save(
    entries: Mapping[str, Coefficients],
    *,
    path: Path | None = None,
    model: str = "quadratic_in_nhit",
) -> Path:
    """Write (or update) the calibration file, preserving other particles."""
    path = Path(path) if path is not None else coefficients_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    blob["model"] = model
    blob["formula"] = "Ereco = (a0+a1*N+a2*N^2)*N1 + (b0+b1*N+b2*N^2)*N2 + (c0+c1*N+c2*N^2)*N3, N = N1+N2+N3"
    particles = blob.setdefault("particles", {})
    for particle, coeff in entries.items():
        entry: dict[str, object] = coeff.as_dict()
        if coeff.errors is not None:
            entry["errors"] = {n: float(e) for n, e in zip(PARAM_NAMES, coeff.errors, strict=True)}
        entry.update(coeff.metadata or {})
        particles[particle] = entry
    path.write_text(json.dumps(blob, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return path
