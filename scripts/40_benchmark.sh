#!/usr/bin/env bash
# Apply every model to the evaluation lot, compare them, draw the report figures.
#
# The work itself lives in the Makefile - `predictions-2class`, `benchmark-2class`
# and `report-figures` - because the model lists and the figure names belong in
# one place, and the Makefile is where the rest of the pipeline already reads
# them. This script exists so the four stages can be run in order without
# knowing which targets to name.
#
# The comparison restricts itself to the events *all* models predicted, so one
# table is over one sample: 6,027 events for the PID, where the hit-level
# networks saw the fewest, and 100,744 for the energy.

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

step "predictions, comparison tables, ROOT figures, report/Fig"
make predictions-2class benchmark-2class

step "done"
