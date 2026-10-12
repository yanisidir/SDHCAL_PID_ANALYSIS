"""Deterministic selection and publication of workflow artifacts.

Explicit selection uses the run identifier embedded in a caller-provided
filename template.  The modification-time based selector is retained only as
the documented compatibility path for historical workflows.
"""

from __future__ import annotations

import glob
import shutil
from pathlib import Path

__all__ = [
    "ArtifactSelectionError",
    "publish_artifact",
    "select_latest",
    "select_run_artifact",
]


class ArtifactSelectionError(ValueError):
    """Raised when an artifact request is ambiguous or cannot be satisfied."""


def select_run_artifact(directory: str | Path, template: str, run_id: str | int) -> Path:
    """Select exactly ``template.format(run_id=run_id)`` below ``directory``.

    Requiring a filename template prevents substring matches such as run 1
    accidentally selecting run 11.  Templates may contain only the
    ``{run_id}`` field and must resolve to a direct child of ``directory``.
    """
    identifier = str(run_id)
    if not identifier or identifier in {".", ".."} or "/" in identifier or "\\" in identifier:
        raise ArtifactSelectionError(f"invalid run_id {identifier!r}")
    if template.count("{run_id}") != 1 or "{" in template.replace("{run_id}", ""):
        raise ArtifactSelectionError("artifact template must contain exactly one {run_id} field")
    filename = template.replace("{run_id}", identifier)

    base = Path(directory).resolve()
    candidate = (base / filename).resolve()
    if candidate.parent != base:
        raise ArtifactSelectionError("artifact template must name a direct child of its directory")
    if not candidate.is_file():
        raise FileNotFoundError(f"requested artifact does not exist: {candidate}")
    return candidate


def select_latest(pattern: str) -> Path:
    """Return the newest glob match, preserving the historical selector."""
    matches = sorted(glob.glob(pattern), key=lambda value: Path(value).stat().st_mtime)
    if not matches:
        raise FileNotFoundError(f"no file matches {pattern!r}")
    return Path(matches[-1])


def publish_artifact(source: str | Path, destination: str | Path) -> Path:
    """Copy one already selected artifact to its historical published path."""
    selected = Path(source)
    if not selected.is_file():
        raise FileNotFoundError(f"artifact does not exist: {selected}")
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(selected, target)
    return target
