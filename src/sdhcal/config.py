"""Minimal JSON configuration helpers for reproducible SDHCAL runs.

Configuration files are optional overlays.  Code defaults remain authoritative
when no file is supplied, and explicit command-line values always win.  Dataset
and output locations remain controlled by the existing ``SDHCAL_*`` environment
variables; their effective values are recorded in resolved configurations.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .provenance import to_jsonable

__all__ = [
    "CONFIG_SCHEMA_VERSION",
    "ConfigError",
    "build_resolved_config",
    "explicit_cli_values",
    "load_config",
    "merge_config",
    "parse_cli",
    "relevant_environment",
    "update_resolved_values",
    "validate_known_keys",
    "workflow_values",
    "write_resolved_config",
]


CONFIG_SCHEMA_VERSION = 1
SUPPORTED_WORKFLOWS = frozenset({"pid_bdt", "pid_mlp", "energy_bdt", "energy_mlp"})
_DOCUMENT_KEYS = frozenset({"schema_version", "profile", "dataset", "workflows"})
_DATASET_KEYS = frozenset({"params_set", "training_sample", "validation_sample"})


class ConfigError(ValueError):
    """Raised when a configuration cannot be interpreted without ambiguity."""


def load_config(path: str | Path) -> dict[str, Any]:
    """Load one JSON object, with errors that identify the configuration file."""
    config_path = Path(path)
    try:
        text = config_path.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise ConfigError(f"configuration file not found: {config_path}") from error
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise ConfigError(
            f"invalid JSON in {config_path} at line {error.lineno}, column {error.colno}: {error.msg}"
        ) from error
    if not isinstance(payload, dict):
        raise ConfigError(f"configuration root in {config_path} must be a JSON object")
    return payload


def validate_known_keys(
    config: Mapping[str, Any], allowed_keys: Iterable[str], *, context: str = "configuration"
) -> None:
    """Reject unknown keys instead of silently ignoring misspelled parameters."""
    unknown = sorted(set(config) - set(allowed_keys))
    if unknown:
        raise ConfigError(f"unknown {context} key(s): {', '.join(unknown)}")


def _value_matches(value: Any, default: Any) -> bool:
    if isinstance(default, bool):
        return isinstance(value, bool)
    if isinstance(default, int):
        return isinstance(value, int) and not isinstance(value, bool)
    if isinstance(default, float):
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if isinstance(default, str):
        return isinstance(value, str)
    if isinstance(default, Path):
        return isinstance(value, (str, Path))
    if isinstance(default, Mapping):
        if not isinstance(value, Mapping):
            return False
        if set(value) - set(default):
            return False
        return all(_value_matches(item, default[key]) for key, item in value.items())
    if isinstance(default, (list, tuple)):
        if not isinstance(value, (list, tuple)):
            return False
        if not default:
            return True
        exemplar = default[0]
        return all(_value_matches(item, exemplar) for item in value)
    return isinstance(value, type(default))


def _validate_types(
    values: Mapping[str, Any],
    defaults: Mapping[str, Any],
    nullable_types: Mapping[str, type | tuple[type, ...]],
    *,
    context: str,
) -> None:
    for key, value in values.items():
        default = defaults[key]
        expected = nullable_types.get(key)
        if value is None:
            if key in nullable_types:
                continue
            raise ConfigError(f"invalid null value for {context}.{key}")
        if default is None:
            if expected is None or isinstance(value, bool) or not isinstance(value, expected):
                raise ConfigError(f"invalid type for {context}.{key}: got {type(value).__name__}")
            continue
        if not _value_matches(value, default):
            raise ConfigError(
                f"invalid type for {context}.{key}: expected the type of the code default "
                f"({type(default).__name__}), got {type(value).__name__}"
            )


def merge_config(
    defaults: Mapping[str, Any],
    file_values: Mapping[str, Any] | None = None,
    cli_values: Mapping[str, Any] | None = None,
    *,
    nullable_types: Mapping[str, type | tuple[type, ...]] | None = None,
) -> dict[str, Any]:
    """Merge ``defaults < file_values < cli_values`` with strict key/type checks."""
    file_values = file_values or {}
    cli_values = cli_values or {}
    nullable_types = nullable_types or {}
    validate_known_keys(file_values, defaults, context="configuration")
    validate_known_keys(cli_values, defaults, context="CLI override")
    _validate_types(file_values, defaults, nullable_types, context="configuration")
    _validate_types(cli_values, defaults, nullable_types, context="CLI override")
    merged = dict(defaults)
    merged.update(file_values)
    merged.update(cli_values)
    return merged


def workflow_values(
    document: Mapping[str, Any], workflow: str, *, params_set: str | None = None
) -> dict[str, Any]:
    """Return one workflow overlay from a versioned repository configuration."""
    validate_known_keys(document, _DOCUMENT_KEYS, context="top-level configuration")
    version = document.get("schema_version")
    if isinstance(version, bool) or not isinstance(version, int):
        raise ConfigError("configuration schema_version must be an integer")
    if version != CONFIG_SCHEMA_VERSION:
        raise ConfigError(
            f"unsupported configuration schema_version {version}; expected {CONFIG_SCHEMA_VERSION}"
        )
    if not isinstance(document.get("profile"), str) or not document["profile"]:
        raise ConfigError("configuration profile must be a non-empty string")

    dataset = document.get("dataset")
    if not isinstance(dataset, Mapping):
        raise ConfigError("configuration dataset must be an object")
    validate_known_keys(dataset, _DATASET_KEYS, context="dataset")
    missing_dataset = sorted(_DATASET_KEYS - set(dataset))
    if missing_dataset:
        raise ConfigError(f"configuration dataset is missing: {', '.join(missing_dataset)}")
    if not all(isinstance(dataset[key], str) and dataset[key] for key in _DATASET_KEYS):
        raise ConfigError("all dataset configuration values must be non-empty strings")
    if params_set is not None and dataset["params_set"] != params_set:
        raise ConfigError(
            f"configuration expects SDHCAL_PARAMS_SET={dataset['params_set']!r}, "
            f"but the effective value is {params_set!r}"
        )

    workflows = document.get("workflows")
    if not isinstance(workflows, Mapping):
        raise ConfigError("configuration workflows must be an object")
    validate_known_keys(workflows, SUPPORTED_WORKFLOWS, context="workflow")
    if workflow not in SUPPORTED_WORKFLOWS:
        raise ConfigError(f"unsupported workflow {workflow!r}")
    if workflow not in workflows:
        raise ConfigError(f"configuration profile {document['profile']!r} has no {workflow!r} section")
    values = workflows[workflow]
    if not isinstance(values, Mapping):
        raise ConfigError(f"configuration section {workflow!r} must be an object")
    return dict(values)


def explicit_cli_values(parser: argparse.ArgumentParser, argv: Iterable[str] | None = None) -> dict[str, Any]:
    """Parse only options explicitly present on the command line.

    Suppressing action defaults is what prevents an argparse default from
    overwriting a value loaded from JSON.
    """
    arguments = list(sys.argv[1:] if argv is None else argv)
    saved_defaults = [(action, action.default) for action in parser._actions]
    saved_parser_defaults = dict(parser._defaults)
    try:
        for action, _ in saved_defaults:
            action.default = argparse.SUPPRESS
        parser._defaults.clear()
        return vars(parser.parse_args(arguments))
    finally:
        for action, default in saved_defaults:
            action.default = default
        parser._defaults.update(saved_parser_defaults)


def parse_cli(
    parser: argparse.ArgumentParser, argv: Iterable[str] | None = None
) -> tuple[argparse.Namespace, dict[str, Any]]:
    """Return the historical full Namespace and the explicit CLI subset."""
    explicit = explicit_cli_values(parser, argv)
    defaults = {
        action.dest: action.default
        for action in parser._actions
        if action.dest != argparse.SUPPRESS and action.default is not argparse.SUPPRESS
    }
    defaults.update(parser._defaults)
    defaults.update(explicit)
    return argparse.Namespace(**defaults), explicit


def relevant_environment() -> dict[str, dict[str, str]]:
    """Return effective path-selection settings and whether each was overridden."""
    from . import paths

    effective = {
        "SDHCAL_DATA_DIR": str(paths.data_dir()),
        "SDHCAL_MODEL_DIR": str(paths.model_dir()),
        "SDHCAL_OUT_DIR": str(paths.output_dir()),
        "SDHCAL_SHARDS_DIR": str(paths.shards_dir()),
        "SDHCAL_HIT_SHARDS_DIR": str(paths.hit_shards_dir()),
        "SDHCAL_PARAMS_SET": paths.params_set(),
    }
    return {
        name: {
            "value": value,
            "source": "environment" if os.environ.get(name) is not None else "code_default",
        }
        for name, value in effective.items()
    }


def build_resolved_config(
    workflow: str,
    values: Mapping[str, Any],
    *,
    config_path: str | Path | None = None,
    cli_overrides: Iterable[str] = (),
    profile: str | None = None,
    dataset: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the complete, JSON-compatible configuration record for one run."""
    return {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "workflow": workflow,
        "profile": profile,
        "values": to_jsonable(values),
        "sources": {
            "config_file": str(config_path) if config_path is not None else None,
            "cli_overrides": list(cli_overrides),
            "environment": relevant_environment(),
        },
        "dataset": to_jsonable(dataset) if dataset is not None else None,
    }


def update_resolved_values(config: Mapping[str, Any], **overrides: Any) -> dict[str, Any]:
    """Copy a resolved document and replace values that vary per mode or seed."""
    updated = dict(config)
    values = dict(config.get("values", {}))
    values.update(to_jsonable(overrides))
    updated["values"] = values
    return updated


def write_resolved_config(path: str | Path, config: Mapping[str, Any]) -> Path:
    """Write a deterministic JSON representation of an effective configuration."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(to_jsonable(config), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output
