#!/bin/bash
set -euo pipefail

# Locations come from the environment, with the repository layout as the default,
# instead of the absolute internship paths that used to be written in here.
#
#   BIN  the computeParams_parallel binary   (make shower-analyzer builds it)
#   IN   directory of digitized .root files
#   OUT  directory for the parameter files
#
# Example:
#   SDHCAL_DATA_DIR=/data IN=/data/digitized/proton OUT=/data/params/proton ./run_parallel.sh

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DATA_DIR=${SDHCAL_DATA_DIR:-$SCRIPT_DIR/../data}

BIN=${BIN:-$SCRIPT_DIR/computeParams_parallel}
IN=${IN:-$DATA_DIR/digitized}
OUT=${OUT:-$DATA_DIR/params}

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

parallel -j 40 --bar '
  f={};                                 # fichier dentrée
  base=$(basename "$f" .root)
  out="'$OUT'/${base}_params.root"

  if [ -f "$out" ]; then
    echo "[SKIP] $out existe"
    exit 0
  fi

  "'$BIN'" "$f" "$out" tree && echo "[OK]  $base" || echo "[ERR] $base"
' ::: "$IN"/*.root



## chmod +x run_parallel.sh
## ./run_parallel.sh
