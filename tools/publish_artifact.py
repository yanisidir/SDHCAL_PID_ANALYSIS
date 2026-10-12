#!/usr/bin/env python3
"""Publish an explicitly selected run artifact, with an opt-in legacy fallback."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from sdhcal.artifact_selection import publish_artifact, select_latest, select_run_artifact

LOGGER = logging.getLogger("publish_artifact")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--template", help="filename template containing {run_id}")
    parser.add_argument("--run-id")
    parser.add_argument("--pattern", help="glob pattern for the historical latest-file fallback")
    parser.add_argument("--latest", action="store_true", help="explicitly use the historical mtime selector")
    args = parser.parse_args(argv)

    explicit = args.run_id is not None or args.template is not None
    if explicit:
        if args.run_id is None or args.template is None or args.latest:
            parser.error("--run-id and --template must be provided together, without --latest")
        source = select_run_artifact(args.directory, args.template, args.run_id)
    else:
        if not args.latest or not args.pattern:
            parser.error("use --run-id/--template, or explicitly request --latest/--pattern")
        LOGGER.warning("using legacy modification-time artifact selection")
        source = select_latest(str(args.directory / args.pattern))

    destination = publish_artifact(source, args.destination)
    print(f"published {source} -> {destination}")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)8s | %(message)s")
    raise SystemExit(main())
