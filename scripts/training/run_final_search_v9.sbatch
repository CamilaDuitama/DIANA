#!/bin/bash
#SBATCH --partition=edid_rtx6000,gpu
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --job-name=diana_final_search
#SBATCH --output=logs/final_search_v9/search_%j.out
#SBATCH --error=logs/final_search_v9/search_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
################################################################################
# One Optuna search over ALL of train -- the nested-CV endpoint.
#
# The 25-fold search gave the generalisation estimate and showed task sharing
# helps. It cannot supply the final hyperparameters: each outer fold produced its
# own set, and picking by fold score selects on fold difficulty rather than on
# hyperparameter quality.
#
# The audited dev folds are the inner CV here, so each candidate is scored on all
# 5 BioProject-disjoint folds. 100 trials x 5 folds ~ 8 h at the measured 58 s per
# trial-fold. Held-out (922) is untouched.
################################################################################
set -e
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg
mkdir -p logs/final_search_v9
echo "=== final search  $(hostname)  $(date) ==="
./env/bin/python scripts/training/01_train_multitask_single_fold.py \
    --run-config configs/train_config_v9_final_search.json \
    --fold_id 0 --use_gpu
echo "=== COMPLETE — $(date) ==="
