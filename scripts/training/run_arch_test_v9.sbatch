#!/bin/bash
#SBATCH --partition=edid_rtx6000,gpu
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-124
#SBATCH --job-name=diana_arch_test
#SBATCH --output=logs/arch_test_v9/t_%a_%j.out
#SBATCH --error=logs/arch_test_v9/t_%a_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
################################################################################
# Multi-task vs single-task, decided properly. (R1.3, R3.12)
#
# Hyperparameters are FIXED per arm to whatever the all-train searches chose, so
# architecture is the only difference. Five seeds per fold separate the
# architecture effect from run-to-run noise. Each (fold, seed) gives a PAIRED
# comparison because both arms see identical data.
#
# 125 tasks = 5 configs (1 multi-task + 4 single-task) x 5 folds x 5 seeds.
# ~1 min per fit measured, so this is ~2 GPU-hours of work spread over the array.
#
# Prerequisite: run_final_search_v9.sbatch and run_final_search_single_v9.sbatch
# must have finished, and prepare_arch_test_configs.py must have written the
# fixed-hyperparameter configs.
#
# Held-out is NOT touched. This is a dev-fold decision.
################################################################################
set -e
CONFIGS=(
    "configs/arch_test_multitask.json"
    "configs/arch_test_single_community_type.json"
    "configs/arch_test_single_feature.json"
    "configs/arch_test_single_sample_host.json"
    "configs/arch_test_single_material.json"
)
SEEDS=(1 2 3 4 5)

I=$SLURM_ARRAY_TASK_ID
CFG_I=$(( I / 25 ))
REST=$(( I % 25 ))
FOLD=$(( REST / 5 ))
SEED_I=$(( REST % 5 ))
RUN_CONFIG="${CONFIGS[$CFG_I]}"
SEED="${SEEDS[$SEED_I]}"

[ -f "$RUN_CONFIG" ] || { echo "ERROR: $RUN_CONFIG missing — run prepare_arch_test_configs.py" >&2; exit 1; }

echo "=== $RUN_CONFIG  fold $FOLD  seed $SEED  $(hostname)  $(date) ==="
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg
mkdir -p logs/arch_test_v9
./env/bin/python scripts/training/01_train_multitask_single_fold.py \
    --run-config "$RUN_CONFIG" --fold_id "$FOLD" --seed "$SEED" --use_gpu
echo "=== COMPLETE — $(date) ==="
