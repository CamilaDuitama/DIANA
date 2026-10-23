#!/bin/bash
# v19 HV test (PROTOCOLS.md amendments of 2026-10-15): training-set evaluation and held-out
# read 3 (ledger) of the hypervector baselines in the released database's projection, beside
# the unitig baselines (same stored predictions as read 1 and the training-set evaluation);
# paired hypervector minus unitig per model. Run after the 44_ --on-train and held-out arrays.
set -euo pipefail
cd /pasteur/helix/scratch/cduitama/EDID/decOM-classify
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
O=results/heldout_v19/scores_read3
[ -e "$O/classification.tsv" ] && { echo "$O already holds read 3; refusing to score twice" >&2; exit 1; }
models() {  # unitig folder, hypervector folder
    echo --model "LogReg=$1/probs_LogisticRegression_Bal.tsv" --model "LogRegHV=$2/probs_LogisticRegression_Bal.tsv" \
         --model "RandomForest=$1/probs_RandomForest_Bal.tsv" --model "RandomForestHV=$2/probs_RandomForest_Bal.tsv" \
         --model "kNN=$1/probs_kNN.tsv" --model "kNNHV=$2/probs_kNN.tsv"
}
PAIRS=(--pair LogRegHV:LogReg --pair RandomForestHV:RandomForest --pair kNNHV:kNN)
TU=results/train_fit_v19/baselines/{task}; TH=results/train_fit_v19/baselines_hvdb/{task}
HU=results/heldout_v19/baselines/{task}; HH=results/heldout_v19/baselines_hvdb/{task}
./env/bin/python scripts/analysis/68_regime_tables_v19.py --train-fit $(models $TU $TH) \
    --pred-only "LinearSVM=$TU/pred_LinearSVM_Bal.tsv" --pred-only "LinearSVMHV=$TH/pred_LinearSVM_Bal.tsv" \
    "${PAIRS[@]}" --pair LinearSVMHV:LinearSVM --out results/train_fit_v19/regime_hvdb.tsv
./env/bin/python scripts/analysis/43_classification_v19.py --heldout $(models $HU $HH) \
    --pred "LinearSVM=$HU/pred_LinearSVM_Bal.tsv" --pred "LinearSVMHV=$HH/pred_LinearSVM_Bal.tsv" \
    "${PAIRS[@]}" --pair LinearSVMHV:LinearSVM --out "$O"
for kind in mixed record; do
    ./env/bin/python scripts/analysis/41_flagging_v19.py --heldout --split-dir data/splits_v19 --plantings results/planted_v19/heldout \
        --pattern "planted_test_${kind}_r0.1_s*.tsv" $(models $HU $HH) "${PAIRS[@]}" --out "$O/flag_$( [ $kind = mixed ] && echo single || echo record )"
done
./env/bin/python scripts/analysis/68_regime_tables_v19.py --heldout --plantings results/planted_v19/heldout \
    --pattern "planted_test_mixed_r0.1_s*.tsv" $(models $HU $HH) \
    --pred-only "LinearSVM=$HU/pred_LinearSVM_Bal.tsv" --pred-only "LinearSVMHV=$HH/pred_LinearSVM_Bal.tsv" \
    "${PAIRS[@]}" --pair LinearSVMHV:LinearSVM --out "$O/regime_single.tsv"
