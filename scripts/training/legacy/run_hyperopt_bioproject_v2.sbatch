#!/bin/bash
#SBATCH --partition=edid_rtx6000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --array=0-4
#SBATCH --job-name=diana_hp_v2
#SBATCH --output=logs/hyperopt_v2/fold_%a_%j.out
#SBATCH --error=logs/hyperopt_v2/fold_%a_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cduitama@pasteur.fr

################################################################################
# DIANA Hyperparameter Optimisation v2 — BioProject split + label_smoothing
################################################################################
#
# Identical to the original bioproject hyperopt, with label_smoothing added
# to the Optuna search space (range [0.0, 0.15]).
#
# USAGE:
#   sbatch --array=0-4 scripts/training/run_hyperopt_bioproject_v2.sbatch
#
# After all 5 folds complete, run final training:
#   RUN_CONFIG=configs/train_config_bioproject_v2.yaml.json \
#     sbatch scripts/training/run_final_train_edid.sbatch
################################################################################

set -e

echo "=== DIANA HYPEROPT v2 (BioProject + label_smoothing) ==="
echo "Job ID:      $SLURM_JOB_ID"
echo "Array task:  $SLURM_ARRAY_TASK_ID"
echo "Node:        $(hostname)"
echo "Start:       $(date)"
echo ""

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

eval "$(/pasteur/appa/homes/cduitama/miniconda3/bin/conda shell.bash hook)"
conda activate ./env
echo "Python: $(which python)"

RUN_CONFIG="configs/train_config_bioproject_v2.yaml.json"
FEATURES=$(python -c "import json; print(json.load(open('$RUN_CONFIG'))['features_path'])")
METADATA=$(python -c "import json; print(json.load(open('$RUN_CONFIG'))['metadata_path'])")
OUTPUT_DIR=$(python -c "import json; print(json.load(open('$RUN_CONFIG'))['output_dir'])")
N_TRIALS=$(python -c "import json; print(json.load(open('$RUN_CONFIG'))['n_trials'])")
MAX_EPOCHS=$(python -c "import json; print(json.load(open('$RUN_CONFIG'))['max_epochs'])")
N_INNER_SPLITS=$(python -c "import json; print(json.load(open('$RUN_CONFIG'))['n_inner_splits'])")
TRAIN_IDS=$(python -c "import json; print(json.load(open('$RUN_CONFIG'))['train_ids_path'])")

FOLD_ID=$SLURM_ARRAY_TASK_ID
FOLD_OUTPUT="${OUTPUT_DIR}/cv_results"

mkdir -p "$FOLD_OUTPUT" logs/hyperopt_v2

echo "Config:      $RUN_CONFIG"
echo "Fold:        $FOLD_ID / 5"
echo "Trials:      $N_TRIALS"
echo "Output:      $FOLD_OUTPUT"
echo ""

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

echo "Starting optimisation for fold $FOLD_ID..."
python scripts/training/01_train_multitask_single_fold.py \
    --fold_id     $FOLD_ID \
    --total_folds 5 \
    --features    "$FEATURES" \
    --metadata    "$METADATA" \
    --output      "$FOLD_OUTPUT" \
    --n_trials    $N_TRIALS \
    --max_epochs  $MAX_EPOCHS \
    --n_inner_splits $N_INNER_SPLITS \
    --use_gpu \
    --random_seed 42

echo ""
echo "=== FOLD $FOLD_ID COMPLETE — $(date) ==="
