"""Portable, versioned manifests for scientific runs and result releases.

The manifest references the resolved R2 configuration and R1 artifacts rather
than copying their metadata.  Paths are relative to the manifest location, so
a release directory can be moved as a unit.  Large input datasets are not
hashed unless a caller explicitly asks for it; generated artifacts are hashed
by default.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .provenance import environment_fingerprint, sha256_file, to_jsonable

__all__ = [
    "EVENT_COUNT_FIELDS",
    "MANIFEST_SCHEMA_VERSION",
    "ManifestError",
    "artifact_identifier",
    "build_manifest",
    "event_counts_from_run_record",
    "load_manifest",
    "make_reference",
    "resolve_reference",
    "validate_manifest",
    "write_manifest",
]

MANIFEST_SCHEMA_VERSION = 1
EVENT_COUNT_FIELDS = (
    "requested_training_budget",
    "input_events",
    "selected_events",
    "training_events",
    "validation_events",
    "test_events",
    "calibration_events",
    "external_evaluation_events",
    "common_benchmark_events",
)
_REFERENCE_GROUPS = ("datasets", "calibrations", "models", "predictions", "outputs")


class ManifestError(ValueError):
    """Raised when a manifest is incomplete, inconsistent, or unsupported."""


def artifact_identifier(path: str | Path) -> str:
    """Return the content-stable identifier used for generated artifacts."""
    return f"sha256:{sha256_file(path)}"


def make_reference(
    path: str | Path,
    manifest_path: str | Path,
    *,
    include_sha256: bool = False,
    artifact_id: bool = False,
) -> dict[str, object]:
    """Describe a file using a path relative to the manifest's directory."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"referenced file does not exist: {source}")
    manifest_dir = Path(manifest_path).expanduser().resolve().parent
    reference: dict[str, object] = {
        "path": os.path.relpath(source, manifest_dir),
        "path_base": "manifest",
        "size_bytes": source.stat().st_size,
    }
    if include_sha256 or artifact_id:
        digest = sha256_file(source)
        reference["sha256"] = digest
        if artifact_id:
            reference["artifact_id"] = f"sha256:{digest}"
    return reference


def resolve_reference(manifest_path: str | Path, reference: Mapping[str, object]) -> Path:
    """Resolve one manifest-relative reference without requiring it to exist."""
    if reference.get("path_base") != "manifest":
        raise ManifestError("reference path_base must be 'manifest'")
    value = reference.get("path")
    if not isinstance(value, str) or not value:
        raise ManifestError("reference path must be a non-empty string")
    return (Path(manifest_path).expanduser().resolve().parent / value).resolve()


def _event_counts(values: Mapping[str, int | None] | None) -> dict[str, int | None]:
    values = values or {}
    unknown = sorted(set(values) - set(EVENT_COUNT_FIELDS))
    if unknown:
        raise ManifestError(f"unknown event count field(s): {', '.join(unknown)}")
    counts: dict[str, int | None] = dict.fromkeys(EVENT_COUNT_FIELDS)
    for name, value in values.items():
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
            raise ManifestError(f"event count {name} must be a non-negative integer or null")
        counts[name] = value
    return counts


def event_counts_from_run_record(record: Mapping[str, Any]) -> dict[str, int | None]:
    """Extract only explicitly recorded counts from known R6/R7 run records."""
    explicit = record.get("event_counts")
    if isinstance(explicit, Mapping):
        return _event_counts(explicit)

    configuration = record.get("configuration")
    splits = configuration.get("split_events") if isinstance(configuration, Mapping) else None
    if not isinstance(splits, Mapping):
        return _event_counts(None)
    translated = {
        "training_events": splits.get("train"),
        "validation_events": splits.get("val"),
        "test_events": splits.get("test"),
    }
    return _event_counts({key: value for key, value in translated.items() if isinstance(value, int)})


def _references(
    values: Mapping[str, str | Path] | None,
    manifest_path: str | Path,
    *,
    generated: bool,
    include_sha256: bool,
) -> dict[str, dict[str, object]]:
    return {
        str(name): make_reference(
            path,
            manifest_path,
            include_sha256=include_sha256,
            artifact_id=generated and include_sha256,
        )
        for name, path in (values or {}).items()
    }


def build_manifest(
    manifest_path: str | Path,
    *,
    release_id: str,
    experiment_type: str,
    command: str | Sequence[str],
    resolved_configuration: str | Path | None = None,
    datasets: Mapping[str, str | Path] | None = None,
    calibrations: Mapping[str, str | Path] | None = None,
    models: Mapping[str, str | Path] | None = None,
    predictions: Mapping[str, str | Path] | None = None,
    outputs: Mapping[str, str | Path] | None = None,
    random_seeds: Sequence[int] = (),
    event_counts: Mapping[str, int | None] | None = None,
    provenance: Mapping[str, Any] | None = None,
    hash_datasets: bool = False,
    hash_generated: bool = True,
) -> dict[str, Any]:
    """Build a manifest without writing or copying any referenced artifact."""
    if not release_id.strip():
        raise ManifestError("release_id must be a non-empty string")
    if release_id in {".", ".."} or "/" in release_id or "\\" in release_id:
        raise ManifestError("release_id must not contain path separators")
    if not experiment_type.strip():
        raise ManifestError("experiment_type must be a non-empty string")
    if isinstance(command, str):
        if not command.strip():
            raise ManifestError("execution command must not be empty")
        execution_command: str | list[str] = command
    else:
        execution_command = [str(part) for part in command]
        if not execution_command:
            raise ManifestError("execution command must not be empty")
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in random_seeds):
        raise ManifestError("random_seeds must contain integers")

    config_reference = None
    if resolved_configuration is not None:
        config_reference = make_reference(
            resolved_configuration,
            manifest_path,
            include_sha256=True,
            artifact_id=False,
        )
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "release_id": release_id,
        "experiment_type": experiment_type,
        "provenance": to_jsonable(provenance or environment_fingerprint()),
        "resolved_configuration": config_reference,
        "execution": {"command": execution_command},
        "random_seeds": list(random_seeds),
        "event_counts": _event_counts(event_counts),
        "references": {
            "datasets": _references(datasets, manifest_path, generated=False, include_sha256=hash_datasets),
            "calibrations": _references(calibrations, manifest_path, generated=False, include_sha256=True),
            "models": _references(models, manifest_path, generated=True, include_sha256=hash_generated),
            "predictions": _references(
                predictions, manifest_path, generated=True, include_sha256=hash_generated
            ),
            "outputs": _references(outputs, manifest_path, generated=True, include_sha256=hash_generated),
        },
    }


def validate_manifest(
    payload: Mapping[str, Any],
    *,
    manifest_path: str | Path | None = None,
    require_files: bool = False,
    verify_hashes: bool = False,
) -> Mapping[str, Any]:
    """Validate schema and, optionally, referenced files and their hashes."""
    if not isinstance(payload, Mapping):
        raise ManifestError("manifest root must be an object")
    if payload.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ManifestError(
            f"unsupported manifest schema_version {payload.get('schema_version')!r}; "
            f"expected {MANIFEST_SCHEMA_VERSION}"
        )
    for field in ("release_id", "experiment_type"):
        if not isinstance(payload.get(field), str) or not payload[field]:
            raise ManifestError(f"manifest {field} must be a non-empty string")
    if not isinstance(payload.get("provenance"), Mapping):
        raise ManifestError("manifest provenance must be an object")
    missing_provenance = [
        name
        for name in ("timestamp_utc", "python", "git_commit", "git_dirty")
        if name not in payload["provenance"]
    ]
    if missing_provenance:
        raise ManifestError(f"manifest provenance is missing: {', '.join(missing_provenance)}")
    execution = payload.get("execution")
    if not isinstance(execution, Mapping) or not execution.get("command"):
        raise ManifestError("manifest execution.command is required")
    _event_counts(payload.get("event_counts"))

    references = payload.get("references")
    if not isinstance(references, Mapping) or set(references) != set(_REFERENCE_GROUPS):
        raise ManifestError(f"manifest references must contain {', '.join(_REFERENCE_GROUPS)}")
    all_references: list[Mapping[str, object]] = []
    config_reference = payload.get("resolved_configuration")
    if config_reference is not None:
        if not isinstance(config_reference, Mapping):
            raise ManifestError("resolved_configuration must be a file reference or null")
        all_references.append(config_reference)
    for group in _REFERENCE_GROUPS:
        entries = references[group]
        if not isinstance(entries, Mapping):
            raise ManifestError(f"manifest references.{group} must be an object")
        for name, reference in entries.items():
            if not isinstance(name, str) or not isinstance(reference, Mapping):
                raise ManifestError(f"invalid reference in {group}")
            all_references.append(reference)

    for reference in all_references:
        if manifest_path is None:
            if require_files or verify_hashes:
                raise ManifestError("manifest_path is required to validate referenced files")
            if reference.get("path_base") != "manifest" or not isinstance(reference.get("path"), str):
                raise ManifestError("invalid manifest-relative file reference")
            continue
        path = resolve_reference(manifest_path, reference)
        if require_files and not path.is_file():
            raise FileNotFoundError(f"manifest reference is missing: {path}")
        if verify_hashes and "sha256" in reference:
            if not path.is_file():
                raise FileNotFoundError(f"cannot verify missing manifest reference: {path}")
            actual = sha256_file(path)
            if actual != reference["sha256"]:
                raise ManifestError(f"SHA-256 mismatch for {path}")
    return payload


def write_manifest(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Validate and atomically write a deterministic manifest JSON file."""
    destination = Path(path)
    validate_manifest(payload, manifest_path=destination, require_files=True, verify_hashes=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=destination.parent, prefix=f".{destination.name}.", delete=False
    ) as stream:
        temporary = Path(stream.name)
        json.dump(to_jsonable(payload), stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, destination)
    return destination


def load_manifest(
    path: str | Path, *, require_files: bool = False, verify_hashes: bool = False
) -> dict[str, Any]:
    """Load and validate a manifest with errors tied to its location."""
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ManifestError(f"manifest not found: {source}") from error
    except json.JSONDecodeError as error:
        raise ManifestError(
            f"invalid JSON in {source} at line {error.lineno}, column {error.colno}: {error.msg}"
        ) from error
    validate_manifest(
        payload,
        manifest_path=source,
        require_files=require_files,
        verify_hashes=verify_hashes,
    )
    return dict(payload)
