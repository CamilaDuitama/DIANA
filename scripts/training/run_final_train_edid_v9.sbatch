#!/bin/bash
#SBATCH --partition=edid_rtx6000,gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --job-name=diana_final_v9
#SBATCH --output=logs/final_train_v9/final_%j.out
#SBATCH --error=logs/final_train_v9/final_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
################################################################################
# DIANA v9 — final fit on all 2,716 training runs.
#
# GPU: 02_train_final_model.py takes no --use_gpu flag; it auto-detects
# `cuda if torch.cuda.is_available()`. So the GPU request above is what puts it on
# one -- submitting this to a CPU partition would still run, silently and slowly.
#
# Hyperparameters come from the single search over all of train
# (run_final_search_v9.sbatch), not from the 25-fold search: per-outer-fold
# hyperparameters would be selected on whichever fold was easiest.
#
# Held-out (922) is NOT evaluated here. That is a separate, single evaluation and
# it must use the same f1_macro_eligible definition as 09_baselines_v9.py, or the
# comparison against the baseline table is not like-for-like.
#
# USAGE:
#   TRAIN_CONFIG=results/search_v9_final/final_training_config.json \
#     sbatch scripts/training/run_final_train_edid_v9.sbatch
################################################################################
set -e
TRAIN_CONFIG=${TRAIN_CONFIG:-results/search_v9_final/final_training_config.json}
[ -f "$TRAIN_CONFIG" ] || { echo "ERROR: no config at $TRAIN_CONFIG" >&2; exit 1; }

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg
mkdir -p logs/final_train_v9

echo "=== DIANA v9 final fit ==="
echo "Config: $TRAIN_CONFIG"
echo "Node:   $(hostname)   Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

./env/bin/python scripts/training/02_train_final_model.py "$TRAIN_CONFIG"
echo "=== COMPLETE — $(date) ==="
