"""Versioned Joblib bundles for tabular PID and energy models.

Schema version 0 denotes the legacy dictionaries written before this module
existed.  They have no explicit ``schema_version`` and are returned unchanged.
Schema version 1 adds descriptive metadata while preserving every historical
key used for inference.

Joblib uses pickle internally.  Only load artifacts from trusted sources.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import joblib

from .provenance import to_jsonable

__all__ = [
    "ArtifactError",
    "CURRENT_SCHEMA_VERSION",
    "LEGACY_SCHEMA_VERSION",
    "artifact_metadata",
    "load_bundle",
    "save_bundle",
    "validate_bundle",
]


LEGACY_SCHEMA_VERSION = 0
CURRENT_SCHEMA_VERSION = 1
SUPPORTED_KINDS = frozenset({"pid", "energy"})
_METADATA_FIELDS = (
    "schema_version",
    "kind",
    "model_family",
    "features",
    "preprocessing",
    "training_counts",
    "configuration",
    "dependencies",
    "provenance",
)


class ArtifactError(ValueError):
    """Raised when an object is not a supported SDHCAL model bundle."""


def _schema_version(payload: Mapping[str, Any]) -> int:
    if "schema_version" not in payload:
        return LEGACY_SCHEMA_VERSION
    version = payload["schema_version"]
    if isinstance(version, bool) or not isinstance(version, int):
        raise ArtifactError("artifact schema_version must be an integer")
    if version not in (LEGACY_SCHEMA_VERSION, CURRENT_SCHEMA_VERSION):
        raise ArtifactError(
            f"unsupported artifact schema_version {version}; "
            f"supported versions are {LEGACY_SCHEMA_VERSION} and {CURRENT_SCHEMA_VERSION}"
        )
    return version


def validate_bundle(payload: Mapping[str, Any], kind: str | None = None) -> Mapping[str, Any]:
    """Validate a legacy or versioned model bundle without modifying it."""
    if not isinstance(payload, Mapping):
        raise ArtifactError(f"model artifact must be a mapping, got {type(payload).__name__}")

    missing = [key for key in ("model", "features") if key not in payload]
    if missing:
        raise ArtifactError(f"model artifact is missing required key(s): {', '.join(missing)}")

    features = payload["features"]
    if isinstance(features, (str, bytes)) or not isinstance(features, Sequence):
        raise ArtifactError("artifact features must be an ordered sequence, not a string")

    version = _schema_version(payload)
    if kind is not None and kind not in SUPPORTED_KINDS:
        raise ArtifactError(f"unknown artifact kind {kind!r}; expected one of {sorted(SUPPORTED_KINDS)}")

    recorded_kind = payload.get("kind")
    if recorded_kind is not None and recorded_kind not in SUPPORTED_KINDS:
        raise ArtifactError(
            f"artifact records unknown kind {recorded_kind!r}; expected one of {sorted(SUPPORTED_KINDS)}"
        )
    if kind is not None and recorded_kind is not None and recorded_kind != kind:
        raise ArtifactError(f"expected a {kind} artifact, got {recorded_kind}")

    if version == CURRENT_SCHEMA_VERSION:
        required = [key for key in ("kind", "model_family") if key not in payload]
        if required:
            raise ArtifactError(f"schema {CURRENT_SCHEMA_VERSION} artifact is missing: {', '.join(required)}")
        if not isinstance(payload["model_family"], str) or not payload["model_family"]:
            raise ArtifactError("artifact model_family must be a non-empty string")
        for field in (
            "preprocessing",
            "training_counts",
            "configuration",
            "dependencies",
            "provenance",
        ):
            if field in payload and not isinstance(payload[field], Mapping):
                raise ArtifactError(f"artifact {field} must be a mapping")
    return payload


def load_bundle(path: str | Path, *, kind: str | None = None) -> dict[str, Any]:
    """Load and validate a trusted Joblib bundle, including legacy dictionaries."""
    artifact_path = Path(path)
    payload = joblib.load(artifact_path)
    validate_bundle(payload, kind=kind)
    # Existing callers expect a mutable dictionary.  Do not add inferred legacy
    # metadata here: loading an old artifact must preserve its keys exactly.
    return payload


def save_bundle(
    path: str | Path,
    payload: Mapping[str, Any],
    *,
    kind: str,
    model_family: str,
    preprocessing: Mapping[str, Any] | None = None,
    training_counts: Mapping[str, Any] | None = None,
    configuration: Mapping[str, Any] | None = None,
    dependencies: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> Path:
    """Write a schema-1 bundle while retaining all keys in ``payload``.

    Optional metadata is included only when the caller can supply it accurately.
    No dataset is inspected or hashed by this function.
    """
    bundle = dict(payload)
    version = bundle.get("schema_version", CURRENT_SCHEMA_VERSION)
    if version != CURRENT_SCHEMA_VERSION:
        raise ArtifactError(f"new bundles must use schema_version {CURRENT_SCHEMA_VERSION}, got {version!r}")
    bundle["schema_version"] = CURRENT_SCHEMA_VERSION
    bundle["kind"] = kind
    bundle["model_family"] = model_family
    optional = {
        "preprocessing": preprocessing,
        "training_counts": training_counts,
        "configuration": configuration,
        "dependencies": dependencies,
        "provenance": provenance,
    }
    for key, value in optional.items():
        if value is not None:
            bundle[key] = dict(value)

    validate_bundle(bundle, kind=kind)
    artifact_path = Path(path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, artifact_path)
    return artifact_path


def artifact_metadata(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return JSON-compatible descriptive metadata without the estimator itself."""
    validate_bundle(payload)
    version = _schema_version(payload)
    metadata = {
        "schema_version": version,
        "legacy": version == LEGACY_SCHEMA_VERSION,
    }
    for field in _METADATA_FIELDS[1:]:
        if field in payload:
            metadata[field] = to_jsonable(payload[field])
    metadata["n_features"] = len(payload["features"])
    return metadata
