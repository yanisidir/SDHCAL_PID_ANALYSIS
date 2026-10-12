# shellcheck shell=bash
# Shared environment for the two-class pion/proton analysis.
#
# Sourced by every script in this directory. It exists because three of the
# settings below are not optional and none of them announce themselves when
# wrong: the run simply reads a different sample and reports plausible numbers.
#
# SDHCAL_SHARDS_DIR is the one that already cost a day. The library defaults to
# `new_data_params`, the lot produced *before* the ShowerAnalyzer corrections
# (annex 1 of the report); these recipes override it with `new_data_params_v2`.
# A model trained without this variable set is trained on the uncorrected
# eccentricity and the unbounded Gaisser-Hillas fit, and nothing says so.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

: "${PYTHON:=python3}"
: "${ROOT_EXE:=root}"

# Corrected parameter shards (training = 200k lot, evaluation = discret lot).
: "${SDHCAL_SHARDS_DIR:=$REPO/new_data_params_v2}"
# Raw hit shards, for the hit-level networks.
: "${SDHCAL_HIT_SHARDS_DIR:=$REPO/new_data}"
# Two-class layout: pi/proton only, chi2_coefficients_2class.json.
: "${SDHCAL_PARAMS_SET:=2class}"

export PYTHON ROOT_EXE
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export SDHCAL_SHARDS_DIR SDHCAL_HIT_SHARDS_DIR SDHCAL_PARAMS_SET

if [ ! -d "$SDHCAL_SHARDS_DIR" ]; then
    echo "missing shard directory: $SDHCAL_SHARDS_DIR" >&2
    exit 1
fi

if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "Python interpreter not found: $PYTHON" >&2
    echo "set PYTHON=/path/to/python (an activated environment may use PYTHON=python)" >&2
    exit 1
fi

step() { printf '\n=== %s ===\n' "$*"; }

publish_latest() {
    local directory=$1 pattern=$2 destination=$3
    "$PYTHON" tools/publish_artifact.py \
        --directory "$directory" --pattern "$pattern" --latest --destination "$destination"
}

publish_selected() {
    local directory=$1 template=$2 fallback_pattern=$3 destination=$4 run_id=${5:-}
    if [ -n "$run_id" ]; then
        "$PYTHON" tools/publish_artifact.py \
            --directory "$directory" --template "$template" --run-id "$run_id" \
            --destination "$destination"
        return
    fi
    printf 'warning: no run id supplied for %s; using historical latest-file selection\n' \
        "$destination" >&2
    publish_latest "$directory" "$fallback_pattern" "$destination"
}
