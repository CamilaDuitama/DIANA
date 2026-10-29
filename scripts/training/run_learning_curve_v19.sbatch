#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-44
#SBATCH --exclude=maestro-3444,maestro-3445,maestro-3448
#SBATCH --job-name=lc_net_v19
#SBATCH --output=logs/learning_curve_v19/net_%a_%j.out
#SBATCH --error=logs/learning_curve_v19/net_%a_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
# Learning curve (PROTOCOLS.md v19 amendment 2026-10-26): the shared network at its selected settings, seed 42, 173 epochs,
# fitted on a subset of the training-fold BioProjects of one dev fold (results/learning_curve_v19/jobs.tsv, row = array index).
set -euo pipefail
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg
ROW=$(awk -F'\t' -v i=$SLURM_ARRAY_TASK_ID 'NR>1 && $1==i {print $4"\t"$5}' results/learning_curve_v19/jobs.tsv)
FOLD=$(echo "$ROW" | cut -f1); DIR=$(echo "$ROW" | cut -f2)
OUT=${DIR/subsets/network}
echo "=== job $SLURM_ARRAY_TASK_ID fold $FOLD $DIR -> $OUT $(hostname) $(date) ==="
./env/bin/python scripts/training/04_epoch_budget_dev_folds.py "$DIR/config.json" --fold "$FOLD" --out "$OUT" \
    --dev-folds "$DIR/dev_folds.tsv" --max-epochs 173 --seed 42 --save-probs
echo "=== COMPLETE $(date) ==="
