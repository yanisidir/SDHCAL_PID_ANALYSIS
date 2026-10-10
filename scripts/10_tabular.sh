#!/usr/bin/env bash
# Shower-variable models: the pion/proton classifier and the energy regressors.
#
# These are the models the Makefile already knows how to build; the reason they
# are repeated here is `--seeds 5`. A single seed gives a number with no width,
# and the comparison in chapters 6 and 8 turns on differences of a few 1e-4 -
# smaller than the seed spread, which is how we know they are not differences.
#
# Produces: models/pid_{bdt,mlp}.joblib, models/energy_bdt_{pi,proton,mixed,ratio}.joblib

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

step "PID BDT, 5 seeds"
$PYTHON analysis/3_pid/bdt/train.py --seeds 5

# The MLP takes --seed, one at a time, rather than --seeds. The seeds match the
# BDT's convention (random_state + i, i = 0..4) so the two spreads are comparable.
step "PID MLP, 5 seeds"
for seed in 42 43 44 45 46; do
    $PYTHON analysis/3_pid/mlp/train.py --seed "$seed" --comment "seed sweep"
done

step "Energy BDT, per species"
$PYTHON analysis/4_energy/ml_bdt/train.py --require-contained

step "Energy BDT on the mixture, 5 seeds"
$PYTHON analysis/4_energy/ml_bdt/train.py --particles mixed --require-contained --seeds 5

# The control for the claim that the gain comes from the information rather than
# the target: the same regressor, the same features, only the target changes.
step "Energy BDT on the ratio target"
$PYTHON analysis/4_energy/ml_bdt/train.py --particles mixed --require-contained --ratio-to-chi2

step "chi2 calibration, fitted on the training lot"
$PYTHON tools/fit_chi2_2class.py

step "done"
