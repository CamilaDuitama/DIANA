#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-124
#SBATCH --exclude=maestro-3444,maestro-3445,maestro-3448
#SBATCH --job-name=diana_seeds_v19
#SBATCH --output=logs/seeds_v19/seed_%a_%j.out
#SBATCH --error=logs/seeds_v19/seed_%a_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
################################################################################
# v19 plain DIANA on the dev folds: each arm at its selected configuration and epoch
# count (configs/v19_final/<arm>.json), seeds 42, 1, 2, 3, 4, every fold; softmax
# probabilities of the last epoch saved (fold-absent classes exactly 0).
# Task i: arm = i / 25, seed = SEEDS[(i % 25) / 5], fold = i % 5.
################################################################################
set -euo pipefail
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg
ARMS=(community_type feature sample_host material multitask)
SEEDS=(42 1 2 3 4)
i=$SLURM_ARRAY_TASK_ID
ARM=${ARMS[$((i / 25))]}
SEED=${SEEDS[$(((i % 25) / 5))]}
FOLD=$((i % 5))
CFG=configs/v19_final/${ARM}.json
EPOCHS=$(./env/bin/python -c "import json; e=json.load(open('$CFG'))['epochs']; assert int(e)>0; print(int(e))")
OUT=results/seeds_v19/${ARM}/seed${SEED}/fold${FOLD}
echo "=== v19 seeds  $ARM seed $SEED fold $FOLD, $EPOCHS epochs -> $OUT  $(hostname)  $(date) ==="
./env/bin/python scripts/training/04_epoch_budget_dev_folds.py "$CFG" --fold "$FOLD" --out "$OUT" \
    --dev-folds data/splits_v19/dev_folds.tsv --max-epochs "$EPOCHS" --seed "$SEED" --save-probs
echo "=== COMPLETE $(date) ==="
