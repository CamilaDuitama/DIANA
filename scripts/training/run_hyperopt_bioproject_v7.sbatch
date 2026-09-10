#!/bin/bash
#SBATCH --partition=edid_rtx6000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --array=0-4
#SBATCH --job-name=diana_hp_v7
#SBATCH --output=logs/hyperopt_v7/fold_%a_%j.out
#SBATCH --error=logs/hyperopt_v7/fold_%a_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cduitama@pasteur.fr
#SBATCH --chdir=/pasteur/appa/scratch/cduitama/EDID/decOM-classify

################################################################################
# DIANA v7 Hyperparameter Optimisation
#
# Config-driven: all settings read from configs/train_config_bioproject_v7.json
#   features_path, metadata_path, output_dir, task_names, task_types,
#   n_folds, n_trials, max_epochs, patience, n_inner_splits, random_seed, ...
#
# Tasks:
#   Classification: community_type, sample_host, material
#   Regression:     sample_age, latitude, longitude
#     - Regression targets normalised to [0,1]; NaN masked out in loss
#     - Optuna objective: mean of per-task scores
#         classification → (balanced_acc + macro_f1) / 2
#         regression     → 1 - mean_abs_error (both in [0,1])
#
# USAGE:
#   sbatch scripts/training/run_hyperopt_bioproject_v7.sbatch
#
# After all 5 folds complete:
#   1. Aggregate CV results:
#        conda run -p ./env python scripts/training/aggregate_cv_results.py \
#            --cv_dir results/training_bioproject_v7/cv_results \
#            --n_folds 5
#   2. Run final training:
#        sbatch scripts/training/run_final_train_edid_v7.sbatch
################################################################################

set -e

RUN_CONFIG="configs/train_config_bioproject_v7.json"

echo "=== DIANA HPO v7 ==="
echo "Job ID:      $SLURM_JOB_ID"
echo "Array task:  $SLURM_ARRAY_TASK_ID"
echo "Node:        $(hostname)"
echo "Start:       $(date)"
echo "Run config:  $RUN_CONFIG"
echo ""

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

eval "$(/pasteur/appa/homes/cduitama/miniconda3/bin/conda shell.bash hook)"
conda activate ./env
echo "Python: $(which python)"

OUTPUT_DIR=$(python -c "import json; print(json.load(open('$RUN_CONFIG'))['output_dir'])")
mkdir -p "$OUTPUT_DIR/cv_results" logs/hyperopt_v7

echo "Output dir:  $OUTPUT_DIR"
echo ""

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

echo "Starting HPO for fold $SLURM_ARRAY_TASK_ID ..."
python scripts/training/01_train_multitask_single_fold.py \
    --run-config "$RUN_CONFIG" \
    --fold_id    "$SLURM_ARRAY_TASK_ID" \
    --use_gpu

echo ""
echo "=== FOLD $SLURM_ARRAY_TASK_ID COMPLETE — $(date) ==="
