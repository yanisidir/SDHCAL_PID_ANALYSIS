"""Dataset and artifact locations, resolved at run time instead of hard-coded.

The original scripts carried absolute paths from the internship machine
(``/gridgroup/ilc/midir/...``), which made every one of them unusable elsewhere.
Paths are now resolved from environment variables, with sane defaults relative
to the repository root:

====================  ======================================  ==============================
Environment variable  Meaning                                 Default
====================  ======================================  ==============================
``SDHCAL_DATA_DIR``   root of the ROOT input files            ``<repo>/data``
``SDHCAL_MODEL_DIR``  trained models and scalers              ``<repo>/models``
``SDHCAL_OUT_DIR``    run outputs (plots, logs, arrays)       ``<repo>/output``
====================  ======================================  ==============================

Usage::

    from sdhcal import paths
    f = paths.dataset("params", "pi")        # <data>/params/130k_pi_E1to130_params.root
    f = paths.dataset("holdout")             # <data>/val_set.root

Missing files raise :class:`DatasetNotFound` with the environment variable to
set, rather than failing deep inside uproot.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "DatasetNotFound",
    "repo_root",
    "data_dir",
    "model_dir",
    "output_dir",
    "dataset",
    "legacy",
    "PARTICLES",
    "PARTICLE_PDG",
    "PDG_TO_PARTICLE",
]

#: Canonical particle keys used throughout the analysis.
PARTICLES = ("pi", "proton", "kaon")

#: PDG code of each simulated primary. The ordering of :data:`PARTICLES` defines
#: the integer class labels used by every classifier (0=pi-, 1=proton, 2=K0_L).
PARTICLE_PDG: dict[str, int] = {"pi": -211, "proton": 2212, "kaon": 311}

PDG_TO_PARTICLE: dict[int, str] = {v: k for k, v in PARTICLE_PDG.items()}

#: Filename templates, relative to :func:`data_dir`, keyed by dataset kind.
#: ``{particle}`` is substituted with one of :data:`PARTICLES`.
_LAYOUT: dict[str, str] = {
    # Shower parameters produced by ShowerAnalyzer (one row per event).
    "params": "params/130k_{particle}_E1to130_params.root",
    # Same, with the primary energy branch merged in.
    "params_merged": "merged_primaryEnergy/130k_{particle}_E1to130_params_merged.root",
    # Digitized hits (thr, I, J, K, time), one entry per event.
    "digitized": "digitized/130k_{particle}_E1to130_digitized.root",
    # Generator-level file holding primaryEnergy.
    "raw": "raw/130k_{particle}_E1to130.root",
    # Held-out evaluation sample, built by tools/build_holdout.py.
    "holdout": "val_set.root",
}


class DatasetNotFound(FileNotFoundError):
    """Raised when a declared dataset is absent from the configured data directory."""


def repo_root() -> Path:
    """Return the repository root.

    The package sits at ``<repo>/src/sdhcal``, so the root is three levels up.
    When the package is installed elsewhere - into a virtual environment's
    ``site-packages`` rather than in editable mode - that guess is wrong, so the
    result is validated against a marker file and the search walks upwards.
    """
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "pyproject.toml").is_file() and (candidate / "src" / "sdhcal").is_dir():
            return candidate
    # Installed outside a checkout: fall back to the current directory, so the
    # environment variables remain the way to point at the data.
    return Path.cwd()


def _dir_from_env(var: str, default_name: str) -> Path:
    value = os.environ.get(var)
    return Path(value).expanduser() if value else repo_root() / default_name


def data_dir() -> Path:
    """Root directory of the ROOT input files (``$SDHCAL_DATA_DIR``)."""
    return _dir_from_env("SDHCAL_DATA_DIR", "data")


def model_dir() -> Path:
    """Directory holding trained models and scalers (``$SDHCAL_MODEL_DIR``)."""
    return _dir_from_env("SDHCAL_MODEL_DIR", "models")


def output_dir() -> Path:
    """Directory for run outputs: plots, logs, arrays (``$SDHCAL_OUT_DIR``)."""
    return _dir_from_env("SDHCAL_OUT_DIR", "output")


def dataset(kind: str, particle: str | None = None, *, require: bool = True) -> Path:
    """Resolve the path of a declared dataset.

    Parameters
    ----------
    kind:
        One of the keys of :data:`_LAYOUT` (``params``, ``params_merged``,
        ``digitized``, ``raw``, ``holdout``).
    particle:
        Required for the per-particle datasets; must be one of :data:`PARTICLES`.
    require:
        When true (the default), raise :class:`DatasetNotFound` if the file is
        absent. Pass ``False`` to get the intended path of a file to be written.
    """
    if kind not in _LAYOUT:
        raise KeyError(f"unknown dataset kind {kind!r}; known kinds: {sorted(_LAYOUT)}")

    template = _LAYOUT[kind]
    if "{particle}" in template:
        if particle is None:
            raise ValueError(f"dataset kind {kind!r} requires a particle among {PARTICLES}")
        if particle not in PARTICLES:
            raise ValueError(f"unknown particle {particle!r}; expected one of {PARTICLES}")
        rel = template.format(particle=particle)
    else:
        if particle is not None:
            raise ValueError(f"dataset kind {kind!r} takes no particle argument")
        rel = template

    path = data_dir() / rel
    if require and not path.is_file():
        raise DatasetNotFound(
            f"missing dataset {kind!r}" + (f" for particle {particle!r}" if particle else "") + f": {path}\n"
            "Set SDHCAL_DATA_DIR to the directory holding the ROOT files, or copy the "
            "file to the path above. See docs/data_pipeline.md for how it is produced."
        )
    return path


# ---------------------------------------------------------------------------
# Legacy path translation
# ---------------------------------------------------------------------------
# The auxiliary scripts - permutation importance, tree plotting, the exploratory
# GNN and CNN studies, shower visualisation - carried absolute paths from the
# internship machine, 175 of them across the repository. Rewriting each by hand
# would have meant touching code that is not otherwise maintained, so those
# scripts keep their original literal and wrap it in :func:`legacy`, which maps
# the obsolete prefix onto the configured directory at run time. The literal
# stays visible in the source, which documents where the file used to live.
#
# Longest prefix wins; the table is sorted by prefix length below so that a more
# specific entry cannot be shadowed by a shorter one listed before it.
_LEGACY_PREFIX_TABLE: tuple[tuple[str, str], ...] = (
    # Per-particle working directories of the internship tree.
    ("/gridgroup/ilc/midir/analyse/analyse_pi/data/", "data"),
    ("/gridgroup/ilc/midir/analyse/analyse_kaon/data/", "data"),
    ("/gridgroup/ilc/midir/analyse/analyse_proton/data/", "data"),
    ("../../analyse_pi/data/", "data"),
    ("../../analyse_kaon/data/", "data"),
    ("../../analyse_proton/data/", "data"),
    # Data directories.
    ("/gridgroup/ilc/midir/analyse/data/", "data"),
    ("/gridgroup/ilc/midir/Timing/files/analyse/data/", "data"),
    ("/gridgroup/ilc/midir/Timing/files/analysisYanis_params/", "data"),
    ("/gridgroup/ilc/midir/Timing/files/analysisYanis/", "data"),
    ("/gridgroup/ilc/midir/Timing/files/", "data"),
    # Everything else under the old analysis root: models when the remainder
    # points into a models/ directory, otherwise the repository itself.
    ("/gridgroup/ilc/midir/Timing/files/analyse/", "model"),
    ("/gridgroup/ilc/midir/analyse/", "repo"),
)

#: Same table, longest prefix first.
_LEGACY_PREFIXES: tuple[tuple[str, str], ...] = tuple(
    sorted(_LEGACY_PREFIX_TABLE, key=lambda item: len(item[0]), reverse=True)
)


def legacy(path: str) -> Path:
    """Translate a path from the internship machine into a configured location.

    ``/gridgroup/ilc/midir/analyse/data/params/x.root`` becomes
    ``$SDHCAL_DATA_DIR/params/x.root``; a remainder pointing into a ``models/``
    directory is resolved under ``$SDHCAL_MODEL_DIR`` instead. A path that
    matches no known prefix is returned unchanged, so an already-correct path is
    safe to pass through.
    """
    text = str(path)
    for prefix, kind in _LEGACY_PREFIXES:
        if not text.startswith(prefix):
            continue
        remainder = text[len(prefix) :]
        if kind == "data":
            return data_dir() / remainder
        if kind == "model":
            _, _, after = remainder.partition("models/")
            return model_dir() / (after or remainder)
        # kind == "repo": keep the repository-relative layout, except for models.
        if "models/" in remainder:
            return model_dir() / remainder.split("models/", 1)[1]
        return repo_root() / remainder
    return Path(text)
