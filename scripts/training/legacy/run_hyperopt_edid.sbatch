#!/bin/bash
#SBATCH --partition=edid_rtx6000
#SBATCH --array=0-4
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --job-name=diana_hyperopt_bp
#SBATCH --output=logs/multitask/hyperopt_bp/hyperopt_%A_%a.out
#SBATCH --error=logs/multitask/hyperopt_bp/hyperopt_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cduitama@pasteur.fr

################################################################################
# DIANA Hyperparameter Optimisation — BioProject-Disjoint Split (edid_rtx6000)
################################################################################
#
# Runs 5-fold cross-validation hyperparameter search on the BioProject-disjoint
# training split (data/splits_bioproject/train_metadata.tsv, 2515 samples).
#
# Partition: edid_rtx6000 — 2x RTX 6000 (96 GB each), 32 cores, 340 GB RAM.
# No time limit on this partition; add -t if you want one.
#
# USAGE:
#   sbatch --array=0-4 scripts/training/run_hyperopt_edid.sbatch
#
# Or submit via diana-train CLI (which calls this script internally):
#   mamba run -p /pasteur/appa/scratch/cduitama/EDID/DIANA/env diana-train multitask \
#     --config configs/train_config_bioproject.yaml \
#     --output results/training_bioproject \
#     --mode optimize
#
# NEXT STEP (after all 5 folds complete):
#   sbatch scripts/training/run_final_train_edid.sbatch
################################################################################

set -e

echo "=== DIANA HYPERPARAMETER OPTIMISATION (BioProject-disjoint split) ==="
echo "Job ID:        $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node:          $(hostname)"
echo "Date:          $(date)"
echo ""

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Activate environment
eval "$(/pasteur/appa/homes/cduitama/miniconda3/bin/conda shell.bash hook)"
conda activate ./env
echo "Python: $(which python)"

# Configuration (override via environment variables if needed)
if [ -n "$RUN_CONFIG" ] && [ -f "$RUN_CONFIG" ]; then
    echo "Reading configuration from: $RUN_CONFIG"
    FEATURES=$(python -c "import json; print(json.load(open('$RUN_CONFIG'))['features_path'])")
    METADATA=$(python -c "import json; print(json.load(open('$RUN_CONFIG'))['metadata_path'])")
    OUTPUT_DIR=$(python -c "import json; print(json.load(open('$RUN_CONFIG'))['output_dir'])")
    TOTAL_FOLDS=$(python -c "import json; print(json.load(open('$RUN_CONFIG'))['n_folds'])")
    N_TRIALS=$(python -c "import json; print(json.load(open('$RUN_CONFIG'))['n_trials'])")
    MAX_EPOCHS=$(python -c "import json; print(json.load(open('$RUN_CONFIG'))['max_epochs'])")
    N_INNER_SPLITS=$(python -c "import json; print(json.load(open('$RUN_CONFIG'))['n_inner_splits'])")
    USE_CLI=true
else
    FEATURES=${FEATURES:-"data/matrices/large_matrix_3070_with_frac/unitigs.frac.mat"}
    METADATA=${METADATA:-"data/splits_bioproject/train_metadata.tsv"}
    OUTPUT_DIR=${OUTPUT_DIR:-"results/training_bioproject"}
    TOTAL_FOLDS=${TOTAL_FOLDS:-5}
    N_TRIALS=${N_TRIALS:-50}
    MAX_EPOCHS=${MAX_EPOCHS:-200}
    N_INNER_SPLITS=${N_INNER_SPLITS:-3}
    USE_CLI=false
fi

FOLD_ID=$SLURM_ARRAY_TASK_ID
LOG_DIR="logs/multitask/hyperopt_bp"

if [ "$USE_CLI" = "false" ]; then
    OUTPUT_DIR="${OUTPUT_DIR}/cv_results"
fi

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

echo "Configuration:"
echo "  Fold:          $FOLD_ID / $TOTAL_FOLDS"
echo "  Optuna trials: $N_TRIALS"
echo "  Max epochs:    $MAX_EPOCHS"
echo "  Features:      $FEATURES"
echo "  Metadata:      $METADATA"
echo "  Output:        $OUTPUT_DIR"
echo ""

echo "GPU:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

echo "Starting optimisation for fold $FOLD_ID..."
python scripts/training/01_train_multitask_single_fold.py \
    --fold_id $FOLD_ID \
    --total_folds $TOTAL_FOLDS \
    --features "$FEATURES" \
    --metadata "$METADATA" \
    --output "$OUTPUT_DIR" \
    --n_trials $N_TRIALS \
    --max_epochs $MAX_EPOCHS \
    --n_inner_splits $N_INNER_SPLITS \
    --use_gpu \
    --random_seed 42

echo ""
echo "=== FOLD $FOLD_ID COMPLETE — $(date) ==="
