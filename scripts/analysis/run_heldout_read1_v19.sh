#!/bin/bash
# v19 held-out read 1 (ledger, approved 2026-10-12): the standard baselines only.
set -euo pipefail
cd /pasteur/helix/scratch/cduitama/EDID/decOM-classify
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
H=results/heldout_v19/baselines/{task}
O=results/heldout_v19/scores_read1
[ -e "$O/classification.tsv" ] && { echo "$O already holds read 1; refusing to score twice" >&2; exit 1; }
MODELS=(--model "LogReg=$H/probs_LogisticRegression_Bal.tsv" --model "RandomForest=$H/probs_RandomForest_Bal.tsv"
        --model "kNN=$H/probs_kNN.tsv" --model "Majority=$H/probs_MajorityClass.tsv")
./env/bin/python scripts/analysis/43_classification_v19.py --heldout "${MODELS[@]}" \
    --pred "LinearSVM=$H/pred_LinearSVM_Bal.tsv" --pair LogReg:RandomForest --out "$O"
./env/bin/python scripts/analysis/41_flagging_v19.py --heldout --split-dir data/splits_v19 --plantings results/planted_v19/heldout \
    --pattern "planted_test_mixed_r0.1_s*.tsv" "${MODELS[@]}" --pair LogReg:RandomForest --out "$O/flag_single"
./env/bin/python scripts/analysis/41_flagging_v19.py --heldout --split-dir data/splits_v19 --plantings results/planted_v19/heldout \
    --pattern "planted_test_record_r0.1_s*.tsv" "${MODELS[@]}" --pair LogReg:RandomForest --out "$O/flag_record"
./env/bin/python scripts/analysis/68_regime_tables_v19.py --heldout --plantings results/planted_v19/heldout \
    --pattern "planted_test_mixed_r0.1_s*.tsv" "${MODELS[@]}" --pred-only "LinearSVM=$H/pred_LinearSVM_Bal.tsv" \
    --out "$O/regime_single.tsv"
