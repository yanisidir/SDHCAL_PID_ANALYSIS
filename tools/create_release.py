#!/usr/bin/env python3
"""Create and validate a lightweight SDHCAL scientific release manifest."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path

from sdhcal.manifest import (
    EVENT_COUNT_FIELDS,
    build_manifest,
    event_counts_from_run_record,
    write_manifest,
)


def _named_paths(values: list[str], parser: argparse.ArgumentParser) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name or not path:
            parser.error(f"expected NAME=PATH, got {value!r}")
        if name in parsed:
            parser.error(f"duplicate reference name {name!r}")
        parsed[name] = Path(path)
    return parsed


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--release-id", required=True)
    result.add_argument("--experiment-type", required=True)
    result.add_argument("--command", required=True, help="the scientific workflow command, recorded verbatim")
    result.add_argument("--results-root", type=Path, default=Path("results"))
    result.add_argument("--manifest", type=Path, help="default: RESULTS_ROOT/RELEASE_ID/manifest.json")
    result.add_argument("--config", type=Path, help="resolved R2 configuration JSON")
    result.add_argument(
        "--run-record",
        type=Path,
        help="existing RunLogger/run.json record from which exact split counts are copied",
    )
    for option in ("dataset", "calibration", "model", "prediction", "output"):
        result.add_argument(f"--{option}", action="append", default=[], metavar="NAME=PATH")
    result.add_argument("--seed", action="append", type=int, default=[])
    for field in EVENT_COUNT_FIELDS:
        result.add_argument(f"--{field.replace('_', '-')}", type=int)
    result.add_argument(
        "--hash-datasets",
        action="store_true",
        help="hash dataset files too; disabled by default for large ROOT inputs",
    )
    result.add_argument(
        "--no-hash-generated",
        action="store_true",
        help="do not hash generated artifacts (hashing is the reproducible default)",
    )
    return result


def main(argv=None) -> int:
    argument_parser = parser()
    args = argument_parser.parse_args(argv)
    destination = args.manifest or args.results_root / args.release_id / "manifest.json"
    counts = {name: getattr(args, name) for name in EVENT_COUNT_FIELDS if getattr(args, name) is not None}
    outputs = _named_paths(args.output, argument_parser)
    if args.run_record is not None:
        try:
            run_record = json.loads(args.run_record.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            argument_parser.error(f"cannot read --run-record {args.run_record}: {error}")
        if not isinstance(run_record, Mapping):
            argument_parser.error("--run-record must contain a JSON object")
        for name, value in event_counts_from_run_record(run_record).items():
            if value is None:
                continue
            if name in counts and counts[name] != value:
                argument_parser.error(
                    f"--{name.replace('_', '-')}={counts[name]} conflicts with the run record value {value}"
                )
            counts[name] = value
        if "run_record" in outputs:
            argument_parser.error("--output run_record=... conflicts with --run-record")
        outputs["run_record"] = args.run_record
    payload = build_manifest(
        destination,
        release_id=args.release_id,
        experiment_type=args.experiment_type,
        command=args.command,
        resolved_configuration=args.config,
        datasets=_named_paths(args.dataset, argument_parser),
        calibrations=_named_paths(args.calibration, argument_parser),
        models=_named_paths(args.model, argument_parser),
        predictions=_named_paths(args.prediction, argument_parser),
        outputs=outputs,
        random_seeds=args.seed,
        event_counts=counts,
        hash_datasets=args.hash_datasets,
        hash_generated=not args.no_hash_generated,
    )
    write_manifest(destination, payload)
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
