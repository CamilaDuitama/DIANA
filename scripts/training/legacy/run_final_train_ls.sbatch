#!/bin/bash
#SBATCH --partition=edid_rtx6000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --job-name=diana_bp_ls
#SBATCH --output=logs/final_training/diana_bp_ls_%j.out
#SBATCH --error=logs/final_training/diana_bp_ls_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cduitama@pasteur.fr

################################################################################
# DIANA Final Model Training — BioProject-Disjoint Split + Label Smoothing
################################################################################
#
# Retrains the production model with label_smoothing to improve
# probability calibration (reduces ECE, especially for community_type
# and material tasks which showed severe overconfidence).
#
# Identical architecture and hyperparameters to results/training_bioproject/
# except label_smoothing in CrossEntropyLoss.
#
# USAGE:
#   # ε=0.05 (default, recommended):
#   sbatch scripts/training/run_final_train_ls.sbatch
#
#   # ε=0.1 (too aggressive for rare-class tasks):
#   TRAIN_CONFIG=results/training_bioproject_ls/final_training_config.json \
#     sbatch scripts/training/run_final_train_ls.sbatch
#
# OUTPUT:
#   results/training_bioproject_ls05/best_model.pth  (ε=0.05)
#   results/training_bioproject_ls/best_model.pth    (ε=0.1)
################################################################################

set -e

echo "=== DIANA FINAL TRAINING (BioProject + Label Smoothing ε=0.1) ==="
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $(hostname)"
echo "Start:   $(date)"
echo ""

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Activate environment
eval "$(/pasteur/appa/homes/cduitama/miniconda3/bin/conda shell.bash hook)"
conda activate ./env
echo "Python: $(which python)"

TRAIN_CONFIG=${TRAIN_CONFIG:-"results/training_bioproject_ls05/final_training_config.json"}

if [ ! -f "$TRAIN_CONFIG" ]; then
    echo "ERROR: Training config not found: $TRAIN_CONFIG"
    exit 1
fi

echo "Config: $TRAIN_CONFIG"
OUTPUT_DIR=$(python -c "import json; print(json.load(open('$TRAIN_CONFIG'))['output_dir'])")
echo "Output: $OUTPUT_DIR"
echo ""

mkdir -p "$OUTPUT_DIR" logs/final_training

echo "GPU:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

echo "Starting training..."
python scripts/training/02_train_final_model.py "$TRAIN_CONFIG"

echo ""
echo "=== TRAINING COMPLETE — $(date) ==="
echo ""
echo "Next: evaluate on test set:"
echo "  ./env/bin/diana-test \\"
echo "    --model $OUTPUT_DIR/best_model.pth \\"
echo "    --config $OUTPUT_DIR/final_training_config.json \\"
echo "    --matrix data/matrices/large_matrix_3070_with_frac/unitigs.frac.mat \\"
echo "    --metadata data/splits_bioproject/test_metadata.tsv \\"
echo "    --test-ids data/splits_bioproject/test_ids.txt \\"
echo "    --output results/test_evaluation_bioproject_ls \\"
echo "    --device cpu"
