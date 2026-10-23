#!/bin/bash
# v19 plain DIANA against the standard baselines on the dev folds (classification,
# mislabel detection, regimes). Run after run_seeds_v19.sbatch.
set -euo pipefail
cd /pasteur/helix/scratch/cduitama/EDID/decOM-classify
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
D=results/devfold_probs_v19
B=results/baselines_v19/{task}
./env/bin/python scripts/analysis/42_oof_probs_v19.py
MODELS=(--model "Separate=$D/probs_Separate_{task}.tsv" --model "Shared=$D/probs_Shared_{task}.tsv"
        --model "Separate1=$D/probs_Separate1_{task}.tsv" --model "Shared1=$D/probs_Shared1_{task}.tsv"
        --model "LogReg=$B/oof_probs_LogisticRegression_Bal.tsv" --model "RandomForest=$B/oof_probs_RandomForest_Bal.tsv"
        --model "kNN=$B/oof_probs_kNN.tsv" --model "Majority=$B/oof_probs_MajorityClass.tsv")
PAIRS=(--pair Separate:Shared --pair Separate:Separate1 --pair Shared:Shared1 --pair Separate:LogReg
       --pair Shared:LogReg --pair Separate:RandomForest --pair Shared:RandomForest)
./env/bin/python scripts/analysis/43_classification_v19.py "${MODELS[@]}" --pred "LinearSVM=$B/oof_pred_LinearSVM_Bal.tsv" \
    "${PAIRS[@]}" --out results/classification_v19/dev
./env/bin/python scripts/analysis/41_flagging_v19.py --split-dir data/splits_v19 --plantings results/planted_v19/dev \
    --pattern "planted_train_mixed_r0.1_s*.tsv" "${MODELS[@]}" "${PAIRS[@]}" --out results/flagging_v19/dev_single
./env/bin/python scripts/analysis/68_regime_tables_v19.py --model "Separate=$D/probs_Separate_{task}.tsv" \
    --model "Shared=$D/probs_Shared_{task}.tsv" --model "LogReg=$B/oof_probs_LogisticRegression_Bal.tsv" \
    --model "RandomForest=$B/oof_probs_RandomForest_Bal.tsv" --model "kNN=$B/oof_probs_kNN.tsv" \
    --pred-only "LinearSVM=$B/oof_pred_LinearSVM_Bal.tsv" --out results/flagging_v19/regime_dev.tsv
