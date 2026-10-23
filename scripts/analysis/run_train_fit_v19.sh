#!/bin/bash
# v19 training-set evaluation: DIANA and the standard baselines scored on the training runs
# they were fitted on, by regime (classification and mislabel detection on the dev plantings).
# Run after run_train_fit_v19.sbatch and the 44_baselines_heldout_v19.py --on-train array.
set -euo pipefail
cd /pasteur/helix/scratch/cduitama/EDID/decOM-classify
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
D=results/train_fit_v19
B=$D/baselines/{task}
./env/bin/python scripts/analysis/47_train_fit_probs_v19.py
./env/bin/python scripts/analysis/68_regime_tables_v19.py --train-fit --model "DIANA=$D/probs_DIANA_{task}.tsv" \
    --model "LogReg=$B/probs_LogisticRegression_Bal.tsv" --model "RandomForest=$B/probs_RandomForest_Bal.tsv" \
    --model "kNN=$B/probs_kNN.tsv" --model "Majority=$B/probs_MajorityClass.tsv" \
    --pred-only "LinearSVM=$B/pred_LinearSVM_Bal.tsv" --out "$D/regime_single.tsv"
