#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --array=0-4
#SBATCH --job-name=diana_f1_fits
#SBATCH --output=logs/final_groupedf1_v9/fit_%a_%j.out
#SBATCH --error=logs/final_groupedf1_v9/fit_%a_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
################################################################################
# Refit of the five v9 final arms after the early-stopping validation split was
# grouped by archive_project.
#
# Everything except the split is held fixed: same hyperparameters, features,
# metadata, train ids and seed as results/search_v9_final*. So the difference
# against those models isolates the split fix rather than confounding it.
#
# Writes to results/final_groupedf1_v9/. The original results/search_v9_final*
# models are NOT touched, because referee responses cite numbers measured on them.
#
# --partition=gpu, not edid_rtx6000: that node is sm_120 and this env's torch
# 2.5.1 is built for sm_50-sm_90. A GPU job must also name ONE partition; a comma
# list is rejected with BadConstraints. Both recorded in CLAUDE.md.
# No --time: the partition default applies.
#
# Task 0 is the multi-task net; 1-4 are the single-task heads that carry the
# held-out table.
################################################################################
set -e
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg

CONFIGS=(
    "configs/final_groupedf1_v9/multitask.json"
    "configs/final_groupedf1_v9/community_type.json"
    "configs/final_groupedf1_v9/feature.json"
    "configs/final_groupedf1_v9/sample_host.json"
    "configs/final_groupedf1_v9/material.json"
)
CFG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}
[ -f "$CFG" ] || { echo "ERROR: no config at $CFG" >&2; exit 1; }

echo "=== grouped refit  task $SLURM_ARRAY_TASK_ID  $CFG  $(hostname)  $(date) ==="
./env/bin/python scripts/training/02_train_final_model.py "$CFG"
echo "=== COMPLETE — $(date) ==="
