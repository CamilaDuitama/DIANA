#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --array=0-4
#SBATCH --job-name=diana_budget_fits
#SBATCH --output=logs/final_budget_v9/fit_%a_%j.out
#SBATCH --error=logs/final_budget_v9/fit_%a_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
################################################################################
# X1: the five v9 arms refitted on all 2,716 training runs for the epoch budget
# each arm was assigned on the grouped dev folds (results/epoch_budget_v9/
# epoch_budget.tsv), with no inner validation split.
#
# Everything else is held fixed: same hyperparameters, features, metadata, train
# ids and seed as results/search_v9_final* (via configs/final_fixed_v9/). The
# configs in configs/final_budget_v9/ add only `epoch_budget`.
#
# Writes to results/final_budget_v9/. No earlier results directory is touched.
# --partition=gpu, one partition, no --time: see CLAUDE.md.
#
# Task 0 is the multi-task net; 1-4 are the single-task nets.
################################################################################
set -e
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg

ARMS=(multitask community_type feature sample_host material)
ARM=${ARMS[$SLURM_ARRAY_TASK_ID]}
CFG=configs/final_budget_v9/${ARM}.json
[ -f "$CFG" ] || { echo "ERROR: no config at $CFG — run 38_make_budget_configs.py" >&2; exit 1; }
[ -f "results/final_budget_v9/$ARM/final_model/best_model.pth" ] && \
    { echo "refusing to overwrite results/final_budget_v9/$ARM" >&2; exit 1; }

echo "=== budget fit  task $SLURM_ARRAY_TASK_ID  $CFG  $(hostname)  $(date) ==="
./env/bin/python scripts/training/02_train_final_model.py "$CFG"
echo "=== COMPLETE — $(date) ==="
