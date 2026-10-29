#!/bin/bash
#SBATCH --partition=edid,common
#SBATCH --exclude=maestro-2004,maestro-2008,maestro-2012,maestro-2016,maestro-2018,maestro-2022
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-1
#SBATCH --job-name=cl_dev_v19
#SBATCH --output=logs/cl_dev_v19_%a_%j.out
#SBATCH --error=logs/cl_dev_v19_%a_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
# Confident Learning flag score against 1 - P(stated), dev folds (PROTOCOLS.md v19 amendment 2026-10-26).
# Task 0: single-label benchmark (primary); task 1: whole-record benchmark.
set -euo pipefail
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg
KIND=$([ "$SLURM_ARRAY_TASK_ID" = 0 ] && echo mixed || echo record)
D="results/devfold_probs_v19/probs_DIANA_{task}.tsv"; B="results/baselines_v19/{task}"
./env/bin/python scripts/analysis/41_flagging_v19.py --split-dir data/splits_v19 --plantings results/planted_v19/dev \
    --pattern "planted_train_${KIND}_r0.1_s*.tsv" \
    --model "DIANA=$D" --model "DIANA_CL=$D" \
    --model "LogReg=$B/oof_probs_LogisticRegression_Bal.tsv" --model "LogReg_CL=$B/oof_probs_LogisticRegression_Bal.tsv" \
    --model "RandomForest=$B/oof_probs_RandomForest_Bal.tsv" --model "RandomForest_CL=$B/oof_probs_RandomForest_Bal.tsv" \
    --cl DIANA_CL --cl LogReg_CL --cl RandomForest_CL \
    --pair DIANA_CL:DIANA --pair LogReg_CL:LogReg --pair RandomForest_CL:RandomForest \
    --out "results/flagging_v19/dev_cl_${KIND}"
