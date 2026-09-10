#!/bin/bash
#SBATCH --partition=edid_rtx6000,gpu
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --array=0-3
#SBATCH --job-name=diana_single_search
#SBATCH --output=logs/final_search_v9/single_%a_%j.out
#SBATCH --error=logs/final_search_v9/single_%a_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
################################################################################
# One Optuna search per single-task control, over ALL of train.
#
# The 25-fold arms already showed task sharing helps on dev folds (Table 8), but
# that comparison cannot be put next to the baselines: it is dev-fold f1_macro over
# all classes, while Table 2 is held-out f1_macro_eligible. To show the progression
# logistic regression -> single-task MLP -> multi-task MLP on the SAME 922 held-out
# runs and the same metric, each control needs its own final hyperparameters.
#
# Identical budget and space to the multi-task search, so neither arm is favoured.
# Held-out is not touched here.
################################################################################
set -e
TASKS=(community_type feature sample_host material)
T=${TASKS[$SLURM_ARRAY_TASK_ID]}
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg
mkdir -p logs/final_search_v9
echo "=== single-task search: $T  $(hostname)  $(date) ==="
./env/bin/python scripts/training/01_train_multitask_single_fold.py \
    --run-config "configs/train_config_v9_final_search_single_${T}.json" \
    --fold_id 0 --use_gpu
echo "=== COMPLETE $T — $(date) ==="
