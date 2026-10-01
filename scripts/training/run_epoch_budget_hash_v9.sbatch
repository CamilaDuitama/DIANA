#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=160G
#SBATCH --array=0-99
#SBATCH --job-name=diana_hash_budget
#SBATCH --output=logs/epoch_budget_hash_v9/fit_%a_%j.out
#SBATCH --error=logs/epoch_budget_hash_v9/fit_%a_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
################################################################################
# H3: DIANA on the four k-mer hash matrices (H1), v9 dev folds, X1 epoch-budget recipe
# (cap 600, loss guard in 37_), hyperparameters unchanged from configs/final_fixed_v9/.
#   tasks  0-24 s1000_f272 | 25-49 s1000_f28 | 50-74 s1000_f14 | 75-99 s100_f272
# 160 GB because the loader holds the matrix dense in float32 (2,716 x ~2 M at floor 14
# is ~25 GB). afterok run_hash_matrix_v9.sbatch. Afterwards: run_hash_select_v9.sbatch.
################################################################################
set -o pipefail
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg
ARMS=(multitask community_type feature sample_host material)
TAGS=(s1000_f272 s1000_f28 s1000_f14 s100_f272)
i=$SLURM_ARRAY_TASK_ID
TAG=${TAGS[$((i / 25))]}; j=$((i % 25)); ARM=${ARMS[$((j / 5))]}; FOLD=$((j % 5))
CFG=configs/hash_v9/${ARM}_${TAG}.json
OUT=results/epoch_budget_hash_${TAG}_v9/${ARM}/fold${FOLD}
[ -f "$CFG" ] || { echo "no config at $CFG (H1)" >&2; exit 1; }
[ -f "$OUT/preds_by_epoch.npz" ] && { echo "refusing to overwrite $OUT" >&2; exit 1; }
echo "=== H3  $CFG fold $FOLD -> $OUT  job $SLURM_JOB_ID  $(hostname)  $(date) ==="
./env/bin/python scripts/training/04_epoch_budget_dev_folds.py "$CFG" --fold "$FOLD" --out "$OUT" \
    --max-epochs 600 --dev-folds data/splits_v9/dev_folds.tsv
rc=$?
echo "=== exit $rc — $(date) ==="
exit $rc
