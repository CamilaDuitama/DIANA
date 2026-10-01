#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --array=0-4
#SBATCH --job-name=diana_v13_fits
#SBATCH --output=logs/final_budget_v13/fit_%a_%j.out
#SBATCH --error=logs/final_budget_v13/fit_%a_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
################################################################################
# K4: the five fraction arms refitted on all 2,716 training runs (v13 labels) for the
# epoch budget each arm was assigned on the v13 dev folds (results/epoch_budget_v13/
# epoch_budget.tsv), no inner validation split. Hyperparameters, features, train ids and
# seed unchanged from configs/final_fixed_v9/. Writes results/final_budget_v13/.
# Task 0 is the multi-task net; 1-4 the single-task nets. afterok run_budget_select_v13.
################################################################################
set -e
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg
ARMS=(multitask community_type feature sample_host material)
ARM=${ARMS[$SLURM_ARRAY_TASK_ID]}
CFG=configs/final_budget_v13/${ARM}.json
[ -f "$CFG" ] || { echo "ERROR: no config at $CFG (run_budget_select_v13 did not write it)" >&2; exit 1; }
[ -f "results/final_budget_v13/$ARM/final_model/best_model.pth" ] && \
    { echo "refusing to overwrite results/final_budget_v13/$ARM" >&2; exit 1; }
echo "=== K4 fit  $CFG  $(hostname)  $(date) ==="
./env/bin/python scripts/training/02_train_final_model.py "$CFG"
echo "=== COMPLETE — $(date) ==="
