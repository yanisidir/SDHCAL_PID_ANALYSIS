#!/bin/bash
set -euo pipefail

# Locations come from the environment, with the repository layout as the default,
# instead of the absolute internship paths that used to be written in here.
#
#   BIN  the extract_parallel binary   (make shower-analyzer builds it)
#   IN   directory of digitized .root files
#   OUT  directory for the parameter files
#
# Example:
#   SDHCAL_DATA_DIR=/data IN=/data/digitized/proton OUT=/data/params/proton ./parallel.sh

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DATA_DIR=${SDHCAL_DATA_DIR:-$SCRIPT_DIR/../data}

BIN=${BIN:-$SCRIPT_DIR/extract_parallel}
IN=${IN:-$DATA_DIR/digitized}
OUT=${OUT:-$DATA_DIR/params}
export BIN OUT

if [ ! -x "$BIN" ]; then
  echo "binary not found: $BIN" >&2
  echo "build it with: make shower-analyzer" >&2
  exit 1
fi
if [ ! -d "$IN" ]; then
  echo "input directory not found: $IN (set IN= or SDHCAL_DATA_DIR=)" >&2
  exit 1
fi
mkdir -p "$OUT"

# (Optionnel) pour faire taire le message de citation :
# parallel --bibtex >/dev/null 2>&1 || true

# The single-quoted program is intentionally evaluated by each GNU parallel
# worker; BIN and OUT are exported above for that worker environment.
# shellcheck disable=SC2016
parallel -j 40 --bar '
  f={};                                 # fichier dentrée
  base=$(basename "$f" .root)
  out="$OUT/${base}_params.root"

  if [ -f "$out" ]; then
    echo "[SKIP] $out existe"
    exit 0
  fi

  "$BIN" "$f" "$out" tree && echo "[OK]  $base" || echo "[ERR] $base"
' ::: "$IN"/*.root



## chmod +x parallel.sh
## ./parallel.sh
