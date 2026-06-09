#!/bin/bash
#SBATCH --partition=edid_rtx6000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --array=0-4
#SBATCH --job-name=diana_hp_v5
#SBATCH --output=logs/hyperopt_v5/fold_%a_%j.out
#SBATCH --error=logs/hyperopt_v5/fold_%a_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cduitama@pasteur.fr
#SBATCH --chdir=/pasteur/appa/scratch/cduitama/EDID/decOM-classify

################################################################################
# DIANA Hyperparameter Optimisation v5 — no label smoothing
#
# Same BioProject-disjoint split as v3, but label smoothing is removed from
# the Optuna search space (fixed at 0 for all tasks via --no_label_smoothing).
# This gives properly tuned hyperparameters for a model without ε.
#
# v3 HPO had ε in [0.0, 0.15]; v4 reused v3 hyperparameters without ε, which
# is inconsistent. v5 re-tunes everything from scratch with ε=0 hard-coded.
#
# USAGE:
#   sbatch --array=0-4 scripts/training/run_hyperopt_bioproject_v5.sbatch
#
# After all 5 folds complete:
#   1. Aggregate CV results:
#      mamba run -p ./env python scripts/training/aggregate_cv_results.py \
#          results/training_bioproject_v5/cv_results
#   2. Inspect best params and build final_training_config.json:
#      (see run_final_train_edid_v5.sbatch header for instructions)
#   3. Run final training:
#      sbatch scripts/training/run_final_train_edid_v5.sbatch
################################################################################

set -e

echo "=== DIANA HYPEROPT v5 (BioProject, no label smoothing) ==="
echo "Job ID:      $SLURM_JOB_ID"
echo "Array task:  $SLURM_ARRAY_TASK_ID"
echo "Node:        $(hostname)"
echo "Start:       $(date)"
echo ""

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

eval "$(/pasteur/appa/homes/cduitama/miniconda3/bin/conda shell.bash hook)"
conda activate ./env
echo "Python: $(which python)"

RUN_CONFIG="configs/train_config_bioproject_v5.yaml.json"
FEATURES=$(python -c "import json; print(json.load(open('$RUN_CONFIG'))['features_path'])")
METADATA=$(python -c "import json; print(json.load(open('$RUN_CONFIG'))['metadata_path'])")
OUTPUT_DIR=$(python -c "import json; print(json.load(open('$RUN_CONFIG'))['output_dir'])")
N_TRIALS=$(python -c "import json; print(json.load(open('$RUN_CONFIG'))['n_trials'])")
MAX_EPOCHS=$(python -c "import json; print(json.load(open('$RUN_CONFIG'))['max_epochs'])")
N_INNER_SPLITS=$(python -c "import json; print(json.load(open('$RUN_CONFIG'))['n_inner_splits'])")

FOLD_ID=$SLURM_ARRAY_TASK_ID
FOLD_OUTPUT="${OUTPUT_DIR}/cv_results"

mkdir -p "$FOLD_OUTPUT" logs/hyperopt_v5

echo "Config:      $RUN_CONFIG"
echo "Fold:        $FOLD_ID / 5"
echo "Trials:      $N_TRIALS"
echo "Output:      $FOLD_OUTPUT"
echo ""

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

echo "Starting optimisation for fold $FOLD_ID (label smoothing disabled)..."
python scripts/training/01_train_multitask_single_fold.py \
    --fold_id        $FOLD_ID \
    --total_folds    5 \
    --features       "$FEATURES" \
    --metadata       "$METADATA" \
    --output         "$FOLD_OUTPUT" \
    --n_trials       $N_TRIALS \
    --max_epochs     $MAX_EPOCHS \
    --n_inner_splits $N_INNER_SPLITS \
    --use_gpu \
    --random_seed    42 \
    --no_label_smoothing

echo ""
echo "=== FOLD $FOLD_ID COMPLETE — $(date) ==="
