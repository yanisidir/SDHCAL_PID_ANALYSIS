"""Small, reusable provenance helpers for runs and model artifacts.

Large detector files are never hashed implicitly.  Callers must explicitly ask
for a SHA-256 digest when building a file reference; this is appropriate for
small configuration or calibration files, but not for multi-gigabyte ROOT lots.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
import subprocess
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "environment_fingerprint",
    "file_reference",
    "git_fingerprint",
    "sha256_file",
    "to_jsonable",
]


_PACKAGE_DISTRIBUTIONS = {
    "numpy": "numpy",
    "pandas": "pandas",
    "sklearn": "scikit-learn",
    "lightgbm": "lightgbm",
    "uproot": "uproot",
    "awkward": "awkward",
    "imblearn": "imbalanced-learn",
    "joblib": "joblib",
    "matplotlib": "matplotlib",
    "scipy": "scipy",
    "torch": "torch",
    "torch_geometric": "torch-geometric",
}

_RELEVANT_ENVIRONMENT = (
    "SDHCAL_DATA_DIR",
    "SDHCAL_MODEL_DIR",
    "SDHCAL_OUT_DIR",
    "SDHCAL_SHARDS_DIR",
    "SDHCAL_HIT_SHARDS_DIR",
    "SDHCAL_PARAMS_SET",
)


def git_fingerprint(directory: Path | None = None) -> dict[str, str]:
    """Return the current commit and dirty status, or an explicit unknown state."""
    cwd = Path(directory) if directory is not None else Path(__file__).resolve().parent
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=10,
                check=True,
            ).stdout.strip()
        )
        return {"git_commit": commit, "git_dirty": str(dirty)}
    except Exception:  # git may be absent, or the installed package may not be in a checkout
        return {"git_commit": "unknown", "git_dirty": "unknown"}


def environment_fingerprint() -> dict[str, str]:
    """Describe the software environment that produced a run or artifact.

    The historical keys from :mod:`sdhcal.runlog` are preserved.  Missing
    optional packages and unset SDHCAL environment variables are simply omitted.
    """
    info: dict[str, str] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        **git_fingerprint(),
    }
    for import_name, distribution in _PACKAGE_DISTRIBUTIONS.items():
        try:
            info[f"version_{import_name}"] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
    for name in _RELEVANT_ENVIRONMENT:
        value = os.environ.get(name)
        if value is not None:
            info[f"env_{name}"] = value
    return info


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash one explicitly requested file without loading it into memory."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def file_reference(path: str | Path, *, include_sha256: bool = False) -> dict[str, object]:
    """Describe an input file, hashing it only when explicitly requested."""
    source = Path(path)
    reference: dict[str, object] = {
        "path": str(source),
        "size_bytes": source.stat().st_size,
    }
    if include_sha256:
        reference["sha256"] = sha256_file(source)
    return reference


def to_jsonable(value: Any) -> Any:
    """Convert common scientific Python values to JSON-compatible objects."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, set):
        return [to_jsonable(item) for item in sorted(value, key=str)]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
