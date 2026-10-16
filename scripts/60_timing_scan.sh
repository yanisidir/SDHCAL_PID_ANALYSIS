#!/usr/bin/env bash
# The two timing points that are missing, framed as the detector question.
#
# The T-SDHCAL proposal asks one thing: does moving from the present
# sub-nanosecond GRPCs to ~100 ps MGRPCs change the reconstruction? Four points
# answer it, and two of them already exist:
#
#   time feature blanked   lower bound                 done (50_energy_seeds era)
#   sigma_t = 1 ns         present GRPCs               this script
#   sigma_t = 100 ps       MGRPC target                done (published run)
#   sigma_t = 0            ideal limit                 this script
#
# Deliberately ONE seed each, not three. How many seeds a point needs depends on
# the seed spread, which 50_energy_seeds.sh is measuring right now; committing to
# three seeds per point before that number exists would be the very mistake the
# seed sweep was launched to avoid. One seed each gives a first look for about
# five hours of machine time, and the spread then says whether the gap between
# points is resolvable and how many more runs it takes.
#
# Everything else matches the published run exactly, so the four points differ
# only in the time resolution.

# shellcheck source=scripts/common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

SEED=${SEED:-42}
COMMON=(--architecture deepsets --cluster-hits --require-contained
        --target ratio-chi2 --baseline-coefficients pi
        --particles mix-pi-proton
        --max-shards-per-particle 40 --max-events-per-shard 2500
        --epochs 30 --batch-size 128 --patience 8 --seed "$SEED")

step "sigma_t = 1 ns (present GRPC resolution)"
$PYTHON analysis/4_energy/gnn/train.py "${COMMON[@]}" \
    --sigma-time 1.0 \
    --output-dir "$REPO/output/energy_reco_gnn_sigma1000ps"

step "sigma_t = 0 (ideal timing)"
$PYTHON analysis/4_energy/gnn/train.py "${COMMON[@]}" \
    --sigma-time 0 \
    --output-dir "$REPO/output/energy_reco_gnn_sigma0ps"

step "done -- four timing points now exist; compare with tools/seed_spread.py and tools/timing_ablation.py"
