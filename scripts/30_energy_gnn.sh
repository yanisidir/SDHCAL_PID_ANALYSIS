#!/usr/bin/env bash
# Hit-level energy: the graph network, and the control that interprets it.
#
# Two runs, and both are needed to make the claim in chapter 7.
#
#   ratio   learns log(E_true / E_chi2): the network corrects the semi-digital
#           calibration instead of replacing it. This is the published result,
#           a = 45.5 % against 54 % for every estimator built on descriptors.
#
#   energy  learns log(E) directly, same architecture, same data. It collapses
#           to sigma/E = 0.203 with the response drifting from 1.21 at 5 GeV to
#           0.81 at 100 GeV - regression towards the sample mean. Without this
#           run the ratio result could be read as a trick of the target, and
#           10_tabular.sh runs the mirror control on the BDT, where the same
#           target changes nothing.
#
# DeepSets rather than attention: measured at 18.5 vs 17.6 ms/event/epoch, the
# architecture was not the lever. Clustering was.

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

# Sampling budget. The network sees fewer events than the BDT (79,451 against
# 176,000); the comparison is therefore unfavourable to it, which only
# strengthens the conclusion. Raise these if there is time.
SAMPLING=(--max-shards-per-particle 40 --max-events-per-shard 2500)
COMMON=(--architecture deepsets --cluster-hits --require-contained
        --particles mix-pi-proton --epochs 30 --batch-size 128)
N_TRAIN=79451

step "energy GNN, ratio target (published)"
$PYTHON analysis/4_energy/gnn/train.py "${COMMON[@]}" "${SAMPLING[@]}" \
    --target ratio-chi2 --baseline-coefficients pi \
    --patience 8 --output-dir "$REPO/output/energy_reco_gnn_ratio"

step "energy GNN, plain energy target (control)"
$PYTHON analysis/4_energy/gnn/train.py "${COMMON[@]}" "${SAMPLING[@]}"

step "import both into the benchmark"
$PYTHON tools/import_predictions.py --kind energy \
    --npz "output/energy_reco_gnn_ratio/mix-pi-proton/arrays/validation_predictions_*.npz" \
    --name gnn_ratio --n-train "$N_TRAIN"
$PYTHON tools/import_predictions.py --kind energy \
    --npz "output/energy_reco_gnn/mix-pi-proton/arrays/validation_predictions_*.npz" \
    --name gnn --n-train "$N_TRAIN"

step "done"
