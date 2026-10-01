#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --array=0-49
#SBATCH --job-name=diana_v13_budget
#SBATCH --output=logs/epoch_budget_v13/fit_%a_%j.out
#SBATCH --error=logs/epoch_budget_v13/fit_%a_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
################################################################################
# K3: DIANA on the v13 dev folds (class set: >= 2 training BioProjects, data/splits_v13/).
# X1 epoch-budget recipe (04_epoch_budget_dev_folds.py, cap 600, loss guard in 37_),
# hyperparameters unchanged from configs/final_fixed_v9/ (configs/final_fixed_v13/ differ
# only in the metadata and train-id paths).
#   tasks  0-24: fractions                 -> results/epoch_budget_v13/<arm>/fold<k>
#   tasks 25-49: fractions + S8.8 signature columns (configs/sig_v13/)
#                                          -> results/epoch_budget_sig_v13/<arm>/fold<k>
# One partition, no --time (CLAUDE.md). Afterwards, afterok: run_budget_select_v13.sbatch.
################################################################################
set -o pipefail
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg
ARMS=(multitask community_type feature sample_host material)
i=$SLURM_ARRAY_TASK_ID
j=$((i % 25)); ARM=${ARMS[$((j / 5))]}; FOLD=$((j % 5))
if [ "$i" -lt 25 ]; then
  CFG=configs/final_fixed_v13/${ARM}.json; OUT=results/epoch_budget_v13/${ARM}/fold${FOLD}
else
  CFG=configs/sig_v13/${ARM}_fold${FOLD}.json; OUT=results/epoch_budget_sig_v13/${ARM}/fold${FOLD}
fi
[ -f "$CFG" ] || { echo "no config at $CFG (K1)" >&2; exit 1; }
[ -f "$OUT/preds_by_epoch.npz" ] && { echo "refusing to overwrite $OUT" >&2; exit 1; }
echo "=== K3  $CFG fold $FOLD -> $OUT  job $SLURM_JOB_ID  $(hostname)  $(date) ==="
./env/bin/python scripts/training/04_epoch_budget_dev_folds.py "$CFG" --fold "$FOLD" --out "$OUT" \
    --max-epochs 600 --dev-folds data/splits_v13/dev_folds.tsv
rc=$?
echo "=== exit $rc — $(date) ==="
exit $rc
