#!/bin/bash
#SBATCH --partition=edid_rtx6000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --job-name=diana_final_bp
#SBATCH --output=logs/final_training/diana_final_bp_%j.out
#SBATCH --error=logs/final_training/diana_final_bp_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cduitama@pasteur.fr

################################################################################
# DIANA Final Model Training — BioProject-Disjoint Split (edid_rtx6000)
################################################################################
#
# Trains the final production model on the full BioProject-disjoint training
# set (2,515 samples) using the best hyperparameters from cross-validation.
#
# Run AFTER run_hyperopt_edid.sbatch (all 5 folds complete).
#
# USAGE:
#   sbatch scripts/training/run_final_train_edid.sbatch
#
# Or via diana-train CLI:
#   mamba run -p /pasteur/appa/scratch/cduitama/EDID/DIANA/env diana-train multitask \
#     --config configs/train_config_bioproject.yaml \
#     --output results/training_bioproject \
#     --mode train
#
# EVALUATION AFTER TRAINING:
#   mamba run -p /pasteur/appa/scratch/cduitama/EDID/DIANA/env diana-test \
#     --model results/training_bioproject/best_model.pth \
#     --config results/training_bioproject/final_training_config.json \
#     --matrix data/matrices/large_matrix_3070_with_frac/unitigs.frac.mat \
#     --metadata data/splits_bioproject/test_metadata.tsv \
#     --test-ids data/splits_bioproject/test_ids.txt \
#     --output results/test_evaluation_bioproject
################################################################################

set -e

echo "=== DIANA FINAL TRAINING (BioProject-disjoint split) ==="
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $(hostname)"
echo "Start:   $(date)"
echo ""

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Activate environment
eval "$(/pasteur/appa/homes/cduitama/miniconda3/bin/conda shell.bash hook)"
conda activate ./env
echo "Python: $(which python)"

# Load training config (set via environment variable or default)
TRAIN_CONFIG=${TRAIN_CONFIG:-"results/training_bioproject/final_training_config.json"}

if [ ! -f "$TRAIN_CONFIG" ]; then
    echo "ERROR: Training config not found: $TRAIN_CONFIG"
    echo "Run hyperopt first: sbatch --array=0-4 scripts/training/run_hyperopt_edid.sbatch"
    exit 1
fi

echo "Config: $TRAIN_CONFIG"
FEATURES=$(python -c "import json; print(json.load(open('$TRAIN_CONFIG'))['features_path'])")
METADATA=$(python -c "import json; print(json.load(open('$TRAIN_CONFIG'))['metadata_path'])")
OUTPUT_DIR=$(python -c "import json; print(json.load(open('$TRAIN_CONFIG'))['output_dir'])")

echo "Features:  $FEATURES"
echo "Metadata:  $METADATA"
echo "Output:    $OUTPUT_DIR"
echo ""

mkdir -p "$OUTPUT_DIR" logs/final_training

echo "GPU:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

echo "Starting final training..."
python scripts/training/02_train_final_model.py "$TRAIN_CONFIG"

echo ""
echo "=== FINAL TRAINING COMPLETE — $(date) ==="
echo ""
echo "Next: evaluate on test set:"
echo "  mamba run -p /pasteur/appa/scratch/cduitama/EDID/DIANA/env diana-test \\"
echo "    --model $OUTPUT_DIR/best_model.pth \\"
echo "    --config $OUTPUT_DIR/final_training_config.json \\"
echo "    --matrix data/matrices/large_matrix_3070_with_frac/unitigs.frac.mat \\"
echo "    --metadata data/splits_bioproject/test_metadata.tsv \\"
echo "    --test-ids data/splits_bioproject/test_ids.txt \\"
echo "    --output results/test_evaluation_bioproject"
