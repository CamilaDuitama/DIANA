#!/bin/bash
# v19 held-out read 2 (ledger, approved 2026-10-13): the final DIANA beside the read-1 baselines
# (same stored predictions); classification, both flagging benchmarks, by regime, paired DIANA minus
# each baseline over held-out BioProjects; no-call and reliability for DIANA.
set -euo pipefail
cd /pasteur/helix/scratch/cduitama/EDID/decOM-classify
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
H=results/heldout_v19/baselines/{task}
D=results/heldout_v19/diana
O=results/heldout_v19/scores_read2
[ -e "$O/classification.tsv" ] && { echo "$O already holds read 2; refusing to score twice" >&2; exit 1; }
./env/bin/python scripts/analysis/49_diana_heldout_v19.py
MODELS=(--model "DIANA=$D/probs_DIANA_{task}.tsv" --model "LogReg=$H/probs_LogisticRegression_Bal.tsv"
        --model "RandomForest=$H/probs_RandomForest_Bal.tsv" --model "kNN=$H/probs_kNN.tsv"
        --model "Majority=$H/probs_MajorityClass.tsv")
CPAIRS=(--pair DIANA:LogReg --pair DIANA:RandomForest --pair DIANA:LinearSVM --pair DIANA:kNN)
FPAIRS=(--pair DIANA:LogReg --pair DIANA:RandomForest)
./env/bin/python scripts/analysis/43_classification_v19.py --heldout "${MODELS[@]}" \
    --pred "LinearSVM=$H/pred_LinearSVM_Bal.tsv" "${CPAIRS[@]}" --out "$O"
./env/bin/python scripts/analysis/41_flagging_v19.py --heldout --split-dir data/splits_v19 --plantings results/planted_v19/heldout \
    --pattern "planted_test_mixed_r0.1_s*.tsv" "${MODELS[@]}" "${FPAIRS[@]}" --out "$O/flag_single"
./env/bin/python scripts/analysis/41_flagging_v19.py --heldout --split-dir data/splits_v19 --plantings results/planted_v19/heldout \
    --pattern "planted_test_record_r0.1_s*.tsv" "${MODELS[@]}" "${FPAIRS[@]}" --out "$O/flag_record"
./env/bin/python scripts/analysis/68_regime_tables_v19.py --heldout --plantings results/planted_v19/heldout \
    --pattern "planted_test_mixed_r0.1_s*.tsv" "${MODELS[@]}" --pred-only "LinearSVM=$H/pred_LinearSVM_Bal.tsv" \
    "${CPAIRS[@]}" --out "$O/regime_single.tsv"
