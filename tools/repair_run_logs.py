#!/usr/bin/env python3
"""Repair the run-parameter CSVs whose columns were shifted by a schema change.

The old logger wrote the header once, when the file was created, then appended
rows built from the config's current keys. When ``val_size`` was added to the
configuration, every later row gained a value the header did not have, so all
columns after it shifted left by one. In ``PID/BDT/run_parameters.csv`` run 9
reads::

    run_id,file_paths,tree_name,test_size,random_state,learning_rate,...
    9,...,paramsTree,0.2,0.2,42,0.01,...
                         ^^^ val_size, recorded under random_state

making it look as though ``random_state=0.2`` and ``learning_rate=42``. The
hyper-parameters of ``lgbm_model_9.joblib``, the classifier used by the coupling
study, were no longer recoverable from the log.

This script detects rows with more fields than the header, inserts the missing
column name at the right position and rewrites the file, keeping a ``.bak``
copy. Rows that already match the header are untouched.

Usage::

    python tools/repair_run_logs.py --dry-run
    python tools/repair_run_logs.py
    python tools/repair_run_logs.py --paths PID/BDT/run_parameters.csv --insert val_size --after test_size
"""

from __future__ import annotations

import argparse
import csv
import logging
import shutil
from pathlib import Path

from sdhcal import paths

logging.basicConfig(level=logging.INFO, format="%(levelname)8s | %(message)s")
logger = logging.getLogger("repair_run_logs")

#: The schema change that caused the shift, as reconstructed from the code:
#: ``_filter_cfg`` returned file_paths, tree_name, test_size, val_size,
#: random_state, ... while the header had been written before ``val_size``
#: existed.
DEFAULT_INSERT = "val_size"
DEFAULT_AFTER = "test_size"


def find_logs() -> list[Path]:
    """Locate the run-parameter CSVs still in the repository."""
    root = paths.repo_root()
    found = []
    for pattern in ("**/run_parameters*.csv", "**/run_comments*.csv", "**/*performances*.csv"):
        found.extend(p for p in root.glob(pattern) if ".git" not in p.parts)
    return sorted(set(found))


def inspect(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        return [], []
    return rows[0], rows[1:]


def repair(
    path: Path, insert: str, after: str, *, dry_run: bool = False
) -> tuple[int, int] | None:
    """Widen the header of one file. Returns (rows_fixed, rows_padded) or None."""
    header, rows = inspect(path)
    if not header:
        return None

    widths = {len(row) for row in rows if row}
    extra = {w for w in widths if w > len(header)}
    if not extra:
        logger.info("%-58s consistent (%d columns)", _rel(path), len(header))
        return None
    if len(extra) > 1:
        logger.error(
            "%s has rows of several widths %s; repair it by hand",
            _rel(path),
            sorted(widths),
        )
        return None

    surplus = extra.pop() - len(header)
    if surplus != 1:
        logger.error("%s: rows carry %d extra fields, not 1; repair by hand", _rel(path), surplus)
        return None
    if after not in header:
        logger.error("%s: cannot place %r, no %r column", _rel(path), insert, after)
        return None

    position = header.index(after) + 1
    new_header = header[:position] + [insert] + header[position:]

    fixed = padded = 0
    out_rows = []
    for row in rows:
        if not row:
            continue
        if len(row) == len(new_header):
            out_rows.append(row)
            fixed += 1
        elif len(row) == len(header):
            out_rows.append(row[:position] + [""] + row[position:])
            padded += 1
        else:
            logger.error("%s: unexpected row width %d; aborting", _rel(path), len(row))
            return None

    logger.info(
        "%-58s inserting %r after %r: %d shifted rows realigned, %d older rows padded",
        _rel(path),
        insert,
        after,
        fixed,
        padded,
    )
    if dry_run:
        return fixed, padded

    shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(new_header)
        writer.writerows(out_rows)
    return fixed, padded


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(paths.repo_root()))
    except ValueError:
        return str(path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--paths", nargs="+", type=Path, default=None, help="files to repair")
    parser.add_argument("--insert", default=DEFAULT_INSERT, help="column name to insert")
    parser.add_argument("--after", default=DEFAULT_AFTER, help="column it follows")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    targets = args.paths if args.paths else find_logs()
    if not targets:
        logger.info("no run log found")
        return 0

    repaired = 0
    for path in targets:
        if not path.is_file():
            logger.warning("%s: not found", path)
            continue
        if repair(path, args.insert, args.after, dry_run=args.dry_run) is not None:
            repaired += 1

    logger.info(
        "%d file(s) %s. New runs are written by sdhcal.runlog, which widens the header instead.",
        repaired,
        "would be repaired" if args.dry_run else "repaired",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
