#!/bin/bash
set -euo pipefail

BIN=/gridgroup/ilc/midir/analyse/ShowerAnalyzer/computeParams_parallel
IN=/gridgroup/ilc/midir/Timing/files/analysisYanis/100k_proton_discret
OUT=/gridgroup/ilc/midir/Timing/files/analysisYanis_params/100k_proton_discret
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
