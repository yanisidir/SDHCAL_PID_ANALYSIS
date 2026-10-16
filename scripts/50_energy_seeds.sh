#!/usr/bin/env bash
# Seed spread of the cluster network, which every network result still lacks.
#
# The published energy result (a = 45.5 %), the PID comparison and the timing
# ablation all rest on a single seed. The only measured width in the whole study
# is the BDT's, sigma/E = 0.1200 +- 0.0008, and nothing says the network's is the
# same: a neural net usually varies more than a boosted tree.
#
# That matters beyond tidiness. The timing ablation differs by 0.0027 per point
# above 35 GeV; whether that is a result or a fluctuation depends entirely on a
# number nobody has measured. And the planned sigma_t scan would compare eight
# configurations separated by 0.002 to 0.017, which cannot be read without it.
#
# Seed 42 is the published run and is not repeated here; this adds three more at
# the same configuration, giving four in total. Each takes about two hours.

# shellcheck source=scripts/common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

SEEDS=${SEEDS:-"43 44 45"}
COMMON=(--architecture deepsets --cluster-hits --require-contained
        --target ratio-chi2 --baseline-coefficients pi
        --particles mix-pi-proton
        --max-shards-per-particle 40 --max-events-per-shard 2500
        --epochs 30 --batch-size 128 --patience 8)

for seed in $SEEDS; do
    step "cluster network, seed $seed"
    $PYTHON analysis/4_energy/gnn/train.py "${COMMON[@]}" \
        --seed "$seed" \
        --output-dir "$REPO/output/energy_reco_gnn_seed$seed"
done

step "done -- compare with tools/seed_spread.py"
