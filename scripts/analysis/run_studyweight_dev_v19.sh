#!/bin/bash
# v19 equal weight per study (PROTOCOLS.md amendment of 2026-10-12) on the dev folds:
# candidate (weighted shared network, DIANA's composition) paired against DIANA, and weighted
# against unweighted logistic regression; classification, mislabel detection, regimes.
# Run after run_studyweight_v19.sbatch and the 40_baselines_v19.py --study-weights array.
set -euo pipefail
cd /pasteur/helix/scratch/cduitama/EDID/decOM-classify
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
D=results/devfold_probs_v19
W=results/studyweight_v19
./env/bin/python scripts/analysis/48_studyweight_probs_v19.py
MODELS=(--model "DIANA=$D/probs_DIANA_{task}.tsv" --model "DIANAW=$W/probs_DIANAW_{task}.tsv"
        --model "LogReg=results/baselines_v19/{task}/oof_probs_LogisticRegression_Bal.tsv"
        --model "LogRegW=$W/baselines/{task}/oof_probs_LogisticRegression_Bal.tsv")
PAIRS=(--pair DIANAW:DIANA --pair LogRegW:LogReg)
./env/bin/python scripts/analysis/43_classification_v19.py "${MODELS[@]}" "${PAIRS[@]}" --out results/classification_v19/dev_studyweight
./env/bin/python scripts/analysis/41_flagging_v19.py --split-dir data/splits_v19 --plantings results/planted_v19/dev \
    --pattern "planted_train_mixed_r0.1_s*.tsv" "${MODELS[@]}" "${PAIRS[@]}" --out results/flagging_v19/dev_single_studyweight
./env/bin/python scripts/analysis/68_regime_tables_v19.py "${MODELS[@]}" --out results/flagging_v19/regime_studyweight_dev.tsv
