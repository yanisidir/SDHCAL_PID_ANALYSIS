#!/usr/bin/env bash
# Hit-level PID: the CNN and the graph network, both on clusters.
#
# The point of these two runs is to answer an objection to the tabular result,
# not to beat it: if the pion/proton confusion were an artefact of summarising a
# shower into a few dozen descriptors, a model reading the hits would do better.
# It does not (chapter 6).
#
# Clustering is what makes them affordable. Grouping fired pads by
# four-connectivity within a layer - the same definition ShowerAnalyzer uses for
# NClusters - takes ~753 hits to ~191 nodes, about a factor four per epoch.
#
# Training uses the 200k lot, evaluation the discret lot: two independent Geant4
# productions, so the separation is by construction rather than by a split.

# shellcheck source=scripts/common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

HITS="$REPO/data/hits_v2"
# Historical requested/recorded budget retained in the benchmark's n_train
# column. Exact split sizes are read from each trainer's run.json; do not infer
# them from this value. Update this budget if the export below changes.
: "${N_TRAIN:=8400}"

step "export hits, with the I/J/K cell indices the clustering needs"
"$PYTHON" tools/export_hits.py --role train --events-per-shard 150 --out-dir "$HITS/train"
"$PYTHON" tools/export_hits.py --role val   --events-per-shard 150 --out-dir "$HITS/val"

for model in cnn gnn; do
    case $model in
        cnn) epochs=40; patience=8 ;;
        gnn) epochs=30; patience=6 ;;
    esac
    step "PID $model on clusters"
    "$PYTHON" "analysis/3_pid/$model/train.py" \
        --pion-csv "$HITS/train/pi_hits.csv" --proton-csv "$HITS/train/proton_hits.csv" \
        --external-pion-csv "$HITS/val/pi_hits.csv" --external-proton-csv "$HITS/val/proton_hits.csv" \
        --cluster-hits --epochs "$epochs" --patience "$patience" --run-name clustered

    step "import $model into the benchmark"
    "$PYTHON" tools/import_predictions.py --kind pid \
        --npz "output/pid_hit_${model}/clustered/external_predictions.npz" \
        --name "$model" --n-train "$N_TRAIN"
done

step "done"
