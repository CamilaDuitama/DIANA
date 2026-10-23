#!/bin/bash
# v19 similarity baselines (PROTOCOLS.md amendment of 2026-10-15, A and B) against the unitig
# baselines on the dev folds: classification, mislabel detection, regimes, paired over BioProjects.
set -euo pipefail
cd /pasteur/helix/scratch/cduitama/EDID/decOM-classify
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
B=results/baselines_v19/{task}; V=results/baselines_sim_v19/{task}; P=results/baselines_simprofile_v19/{task}
MODELS=(--model "LogReg=$B/oof_probs_LogisticRegression_Bal.tsv" --model "RandomForest=$B/oof_probs_RandomForest_Bal.tsv"
        --model "kNN=$B/oof_probs_kNN.tsv" --model "SimVote=$V/oof_probs_SimVote.tsv"
        --model "LogRegSP=$P/oof_probs_LogisticRegression_Bal.tsv" --model "RandomForestSP=$P/oof_probs_RandomForest_Bal.tsv"
        --model "kNNSP=$P/oof_probs_kNN.tsv")
PAIRS=(--pair SimVote:kNN --pair SimVote:LogReg --pair LogRegSP:LogReg --pair RandomForestSP:RandomForest --pair kNNSP:kNN)
./env/bin/python scripts/analysis/43_classification_v19.py "${MODELS[@]}" \
    --pred "LinearSVM=$B/oof_pred_LinearSVM_Bal.tsv" --pred "LinearSVMSP=$P/oof_pred_LinearSVM_Bal.tsv" \
    "${PAIRS[@]}" --pair LinearSVMSP:LinearSVM --out results/classification_v19/dev_sim
./env/bin/python scripts/analysis/41_flagging_v19.py --split-dir data/splits_v19 --plantings results/planted_v19/dev \
    --pattern "planted_train_mixed_r0.1_s*.tsv" "${MODELS[@]}" "${PAIRS[@]}" --out results/flagging_v19/dev_single_sim
./env/bin/python scripts/analysis/68_regime_tables_v19.py "${MODELS[@]}" \
    --pred-only "LinearSVM=$B/oof_pred_LinearSVM_Bal.tsv" --pred-only "LinearSVMSP=$P/oof_pred_LinearSVM_Bal.tsv" \
    "${PAIRS[@]}" --pair LinearSVMSP:LinearSVM --out results/flagging_v19/regime_sim_dev.tsv
