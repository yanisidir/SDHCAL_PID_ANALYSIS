"""Ingestion of the sharded parameter lots delivered as ``data/lots/pi-proton/params-superseded``.

Those lots already carry the 26 shower variables the PID stage trains on, so the
C++ extraction does not have to run again. Three things still stand between them
and the pipeline, and this module is where each is fixed once instead of in every
stage.

*The species label is missing.* ``particlePDG`` and ``particle1PDG`` are ``0`` in
every event of every lot, so nothing in the files says which particle produced a
shower. The label is taken from the lot a file belongs to, which is why
:func:`read_shard` demands the particle as an argument rather than guessing.

*``eventNumber`` is only unique inside a shard.* Every file restarts it at zero,
so the 200k lots hold 5 000 distinct values for 200 000 events. ``splits.event_keys``
keys on ``(particlePDG, eventNumber)``, so used as delivered - with a constant
PDG on top - every key would collapse onto one of 5 000 values and
``build_holdout.py`` would "prove" a disjointness that means nothing.
:func:`global_event_number` folds the shard id into the key to make it unique
again, and the id comes from the file *name* rather than from its position in a
directory listing, so the key survives a re-listing or an added file.

*``touchesLastLayer`` is absent.* The chi2 chain rejects showers reaching the last
active layer, and the ML chain must apply the identical cut for the two to be
compared on the same events. The lots kept their hit branches, so the flag is
recomputed here from ``K`` instead of re-running the extraction.

The true energy is already present, under the name ``energy``; it is renamed to
``primaryEnergy``, which is what the energy stages read. No merge step remains.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from . import paths

__all__ = [
    "FEATURES",
    "LOT_ROLES",
    "LOT_ID_BASE",
    "LAST_LAYER",
    "SHARD_STRIDE",
    "shard_files",
    "shard_id",
    "global_event_number",
    "read_shard",
    "read_sample",
]

#: The 26 shower variables of the original PID list, kept as the *stored* set.
#: The model's own list lives in train.py and is now shorter:
#: ``ratioThr23`` and ``tSpread`` were dropped there because each is a monotone
#: transform of another column and so cannot give a tree a split it did not
#: already have. They stay in the files, since storing them costs nothing and
#: removing them would foreclose any later study.
PID_FEATURES: tuple[str, ...] = (
    "Thr1",
    "Thr2",
    "Thr3",
    "Begin",
    "Radius",
    "Density",
    "NClusters",
    "ratioThr23",
    "Zbary",
    "Zrms",
    "PctHitsFirst10",
    "PlanesWithClusmore2",
    "AvgClustSize",
    "MaxClustSize",
    "lambda1",
    "lambda2",
    "tMin",
    "tMax",
    "tMean",
    "tSpread",
    "Nmax",
    "z0_fit",
    "Xmax",
    "lambda",
    "nTrackSegments",
    "eccentricity3D",
)

#: What the energy stages read: the regression features, plus the three threshold
#: counts the chi2 estimator needs (``CHI2_COLUMNS`` in the coupling script).
ENERGY_FEATURES: tuple[str, ...] = (
    "Density",
    "NClusters",
    "Zbary",
    "AvgClustSize",
    "N3",
    "N2",
    "N1",
    "nHitsTotal",
    "sumThrTotal",
)

#: Every variable carried into the consolidated files. Both studies read from one
#: file per particle, so the union is kept rather than one file per stage.
FEATURES: tuple[str, ...] = (
    *PID_FEATURES,
    *(name for name in ENERGY_FEATURES if name not in PID_FEATURES),
)

#: Index of the last active layer of the SDHCAL, as in extract.cpp.
LAST_LAYER = 47

#: Multiplier folding a shard id and a per-shard event number into one integer.
#: Larger than any shard (5 000 events), and the largest id in use times
#: SHARD_STRIDE still fits in the int32 that ROOT stores ``eventNumber`` as.
SHARD_STRIDE = 100_000

#: Role of each lot, as the simulations were produced: the continuous 1-130 GeV
#: spectrum trains the models and calibrates the chi2, and the fixed-energy grid
#: evaluates them. They are two independent Geant4 productions, so the evaluation
#: sample is not carved out of the training one and nothing has to be held out.
LOT_ROLES: dict[str, str] = {"train": "200k", "val": "discret"}

#: Added to a lot's shard ids so that two lots can never share an event key.
#: Without it the grid lot's ids (2..41) overlap the continuous lot's (0..39), and
#: the 95 000 training events sitting in the shared range would answer to the same
#: ``(particlePDG, eventNumber)`` key as a validation event - which any
#: disjointness check, or any attempt to exclude one sample from the other, would
#: act on.
LOT_ID_BASE: dict[str, int] = {"200k": 0, "discret": 1_000}

#: How a shard id is read off a file name, per sample. Deriving the id from the
#: name keeps the event key reproducible; deriving it from the position in a
#: sorted listing would not survive a lot gaining a file.
_SHARD_ID_PATTERNS: dict[str, re.Pattern[str]] = {
    "200k": re.compile(r"_1-130_(?P<shard>\d+)_digitized_params\.root$"),
    "discret": re.compile(r"_(?P<energy>\d+)GeV_part(?P<part>\d+)_digitized_params\.root$"),
}


#: Tree names a shard may use: the delivered lots append to the input tree, the
#: regenerated ones are written fresh by extract.
TREE_NAMES: tuple[str, ...] = ("tree", "paramsTree")

#: Likewise for the true energy, under its generator name or its canonical one.
ENERGY_BRANCHES: tuple[str, ...] = ("energy", "primaryEnergy")


def _first_present(available, candidates: Sequence[str], path: Path, what: str) -> str:
    """Pick the first candidate the file actually has, or say what it holds."""
    names = {str(name).split(";")[0] for name in available}
    for candidate in candidates:
        if candidate in names:
            return candidate
    raise KeyError(
        f"{Path(path).name}: no {what} among {', '.join(candidates)}; it holds {sorted(names)[:12]}"
    )


def shard_files(sample: str, particle: str) -> list[Path]:
    """Every shard of one lot, ordered by :func:`shard_id`.

    Raises ``FileNotFoundError`` naming the variable to set when the directory is
    absent or holds no shard, rather than returning an empty list that would show
    up much later as an empty frame.
    """
    if sample not in paths.SHARD_SAMPLES:
        raise KeyError(f"unknown sample {sample!r}; known samples: {sorted(paths.SHARD_SAMPLES)}")
    lot = paths.SHARD_SAMPLES[sample]
    if particle not in lot:
        raise ValueError(f"sample {sample!r} has no {particle!r} lot; it holds {sorted(lot)}")

    directory = paths.shards_dir() / lot[particle]
    if not directory.is_dir():
        raise FileNotFoundError(
            f"missing shard directory for {particle!r} of sample {sample!r}: {directory}\n"
            "Set SDHCAL_SHARDS_DIR to the directory holding the sharded parameter lots."
        )
    found = sorted(directory.glob("*_params.root"), key=lambda p: shard_id(sample, p))
    if not found:
        raise FileNotFoundError(f"no shard matching '*_params.root' in {directory}")
    return found


def shard_id(sample: str, path: Path | str) -> int:
    """Stable integer id of a shard, read off its file name.

    The ``200k`` lots number their shards directly. The ``discret`` lots identify
    one by an energy and a part, which are folded into a single id; the 5 GeV step
    of the grid makes ``energy // 5`` dense, so ids stay small enough for
    :data:`SHARD_STRIDE`.
    """
    name = Path(path).name
    match = _SHARD_ID_PATTERNS[sample].search(name)
    if match is None:
        raise ValueError(
            f"cannot read a shard id off {name!r} for sample {sample!r}. "
            "The event key is built from it, so an unrecognised name is refused "
            "rather than silently numbered by position."
        )
    groups = match.groupdict()
    base = LOT_ID_BASE[sample]
    if groups.get("shard") is not None:
        return base + int(groups["shard"])
    energy, part = int(groups["energy"]), int(groups["part"])
    if energy % 5:
        raise ValueError(f"{name!r}: energy {energy} is off the 5 GeV grid")
    return base + (energy // 5) * 2 + part


def global_event_number(shard: int, local: np.ndarray) -> np.ndarray:
    """Fold a shard id and its per-shard event numbers into unique integers."""
    local = np.asarray(local, dtype="int64")
    if local.size and (local.min() < 0 or local.max() >= SHARD_STRIDE):
        raise ValueError(
            f"shard {shard}: eventNumber out of [0, {SHARD_STRIDE}); the key would "
            "collide with the next shard."
        )
    return shard * SHARD_STRIDE + local


def read_shard(
    path: Path,
    sample: str,
    particle: str,
    *,
    features: Sequence[str] = FEATURES,
) -> pd.DataFrame:
    """Read one shard into the flat table the stages expect.

    Returns ``features`` plus ``particlePDG`` (from ``particle``, since the files
    carry none), ``eventNumber`` (unique across the lot), ``primaryEnergy`` (the
    lots' ``energy``), ``touchesLastLayer`` (recomputed from ``K``), and
    ``shardIndex``/``localEventNumber`` for provenance.
    """
    import uproot

    shard = shard_id(sample, path)
    with uproot.open(path) as handle:
        tree_name = _first_present(handle.keys(), TREE_NAMES, path, "tree")
        tree = handle[tree_name]
        # A uproot TTree is not a mapping: keys() is the branch listing, and the
        # names carry a ";<cycle>" suffix that has to come off before matching.
        branch_names = tree.keys()
        available = {name.split(";")[0] for name in branch_names}

        energy_branch = _first_present(available, ENERGY_BRANCHES, path, "energy branch")
        wanted = [*features, "eventNumber", energy_branch]

        # touchesLastLayer is written by the current extract; the delivered
        # lots predate it but kept their hits, so it is recomputed from K there.
        derive_last_layer = "touchesLastLayer" not in available
        if derive_last_layer:
            if "K" not in available:
                raise KeyError(
                    f"{path.name}: neither 'touchesLastLayer' nor 'K'; the last-layer "
                    "flag can be neither read nor recomputed."
                )
            wanted.append("K")
        else:
            wanted.append("touchesLastLayer")

        missing = [name for name in wanted if name not in available]
        if missing:
            raise KeyError(f"{path.name}: missing branch(es) {', '.join(missing)}")
        arrays = tree.arrays(wanted, library="np")

    local = arrays["eventNumber"].astype("int64")
    frame = pd.DataFrame({name: arrays[name] for name in features})
    # The lots carry particlePDG == 0 for every event, regenerated ones included,
    # because the digitized input has particle1PDG == 0. The lot is the only label.
    frame["particlePDG"] = np.int32(paths.PARTICLE_PDG[particle])
    frame["eventNumber"] = global_event_number(shard, local).astype("int64")
    frame["primaryEnergy"] = arrays[energy_branch].astype("float64")
    if derive_last_layer:
        frame["touchesLastLayer"] = np.fromiter(
            ((np.asarray(hits) == LAST_LAYER).any() for hits in arrays["K"]),
            dtype="int32",
            count=len(frame),
        )
    else:
        frame["touchesLastLayer"] = arrays["touchesLastLayer"].astype("int32")
    frame["shardIndex"] = np.int32(shard)
    frame["localEventNumber"] = local.astype("int32")
    return frame


def read_sample(
    sample: str,
    particle: str,
    *,
    features: Sequence[str] = FEATURES,
    limit: int | None = None,
) -> pd.DataFrame:
    """Concatenate every shard of one lot, shard by shard to bound the memory.

    ``limit`` stops after that many shards, which is what the tests use.
    """
    files = shard_files(sample, particle)
    if limit is not None:
        files = files[:limit]
    frames = [read_shard(path, sample, particle, features=features) for path in files]
    return pd.concat(frames, ignore_index=True)
