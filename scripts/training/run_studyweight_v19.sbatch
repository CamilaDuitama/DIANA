#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-24
#SBATCH --exclude=maestro-3444,maestro-3445,maestro-3448
#SBATCH --job-name=diana_studyw_v19
#SBATCH --output=logs/studyweight_v19/fit_%a_%j.out
#SBATCH --error=logs/studyweight_v19/fit_%a_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
################################################################################
# v19 equal weight per study (PROTOCOLS.md amendment of 2026-10-12) on the dev folds:
# the shared network exactly as in run_seeds_v19.sbatch (configs/v19_final/multitask.json,
# its epoch count, seeds 42, 1, 2, 3, 4, every fold) plus --study-weights, one weight
# column per field. Task i: seed = SEEDS[i / 5], fold = i % 5.
################################################################################
set -euo pipefail
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg
SEEDS=(42 1 2 3 4)
i=$SLURM_ARRAY_TASK_ID
SEED=${SEEDS[$((i / 5))]}
FOLD=$((i % 5))
CFG=configs/v19_final/multitask.json
EPOCHS=$(./env/bin/python -c "import json; e=json.load(open('$CFG'))['epochs']; assert int(e)>0; print(int(e))")
OUT=results/studyweight_v19/multitask/seed${SEED}/fold${FOLD}
echo "=== v19 study weights  multitask seed $SEED fold $FOLD, $EPOCHS epochs -> $OUT  $(hostname)  $(date) ==="
./env/bin/python scripts/training/04_epoch_budget_dev_folds.py "$CFG" --fold "$FOLD" --out "$OUT" \
    --dev-folds data/splits_v19/dev_folds.tsv --max-epochs "$EPOCHS" --seed "$SEED" --save-probs --study-weights
echo "=== COMPLETE $(date) ==="
