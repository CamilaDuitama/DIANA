#!/bin/bash
# v19 HV test (PROTOCOLS.md amendments of 2026-10-14): the standard baselines on hypervectors
# against the same baselines on unitig fractions, dev folds only, paired over BioProjects;
# classification, mislabel detection, regimes. Run after the 40_baselines_v19.py array on
# results/hypervectors_v19/hv_train.npz. No held-out number is produced.
# Optional arguments: hypervector baseline folder and output tag (defaults: the 2026-10-14 run).
set -euo pipefail
cd /pasteur/helix/scratch/cduitama/EDID/decOM-classify
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
B=results/baselines_v19/{task}
H=${1:-results/baselines_hv_v19}/{task}
TAG=${2:-hv}
MODELS=(--model "LogReg=$B/oof_probs_LogisticRegression_Bal.tsv" --model "LogRegHV=$H/oof_probs_LogisticRegression_Bal.tsv"
        --model "RandomForest=$B/oof_probs_RandomForest_Bal.tsv" --model "RandomForestHV=$H/oof_probs_RandomForest_Bal.tsv"
        --model "kNN=$B/oof_probs_kNN.tsv" --model "kNNHV=$H/oof_probs_kNN.tsv")
PREDS=(--pred "LinearSVM=$B/oof_pred_LinearSVM_Bal.tsv" --pred "LinearSVMHV=$H/oof_pred_LinearSVM_Bal.tsv")
PAIRS=(--pair LogRegHV:LogReg --pair RandomForestHV:RandomForest --pair kNNHV:kNN)
./env/bin/python scripts/analysis/43_classification_v19.py "${MODELS[@]}" "${PREDS[@]}" "${PAIRS[@]}" \
    --pair LinearSVMHV:LinearSVM --out results/classification_v19/dev_$TAG
./env/bin/python scripts/analysis/41_flagging_v19.py --split-dir data/splits_v19 --plantings results/planted_v19/dev \
    --pattern "planted_train_mixed_r0.1_s*.tsv" "${MODELS[@]}" "${PAIRS[@]}" --out results/flagging_v19/dev_single_$TAG
./env/bin/python scripts/analysis/68_regime_tables_v19.py "${MODELS[@]}" \
    --pred-only "LinearSVM=$B/oof_pred_LinearSVM_Bal.tsv" --pred-only "LinearSVMHV=$H/oof_pred_LinearSVM_Bal.tsv" \
    "${PAIRS[@]}" --pair LinearSVMHV:LinearSVM --out results/flagging_v19/regime_${TAG}_dev.tsv
