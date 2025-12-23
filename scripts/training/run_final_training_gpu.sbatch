#!/bin/bash
#SBATCH --job-name=diana_final_train
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=logs/final_training/diana_final_%j.out
#SBATCH --error=logs/final_training/diana_final_%j.err

# Final model training script for multi-task classifier
# This script trains the final production model on all training data
# using the best hyperparameters from cross-validation

set -e  # Exit on error

echo "=========================================="
echo "DIANA Final Training Job"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo ""

# Load training configuration
if [ -z "$TRAIN_CONFIG" ]; then
    # Use default config if not provided
    TRAIN_CONFIG="results/full_training/final_training_config.json"
    echo "Using default config: $TRAIN_CONFIG"
fi

if [ ! -f "$TRAIN_CONFIG" ]; then
    echo "ERROR: Training config file not found: $TRAIN_CONFIG"
    exit 1
fi

echo "Loading configuration from: $TRAIN_CONFIG"

# Extract parameters from JSON config
FEATURES=$(python -c "import json; print(json.load(open('$TRAIN_CONFIG'))['features_path'])")
METADATA=$(python -c "import json; print(json.load(open('$TRAIN_CONFIG'))['metadata_path'])")
OUTPUT_DIR=$(python -c "import json; print(json.load(open('$TRAIN_CONFIG'))['output_dir'])")
HYPERPARAMS_JSON=$(python -c "import json; import sys; json.dump(json.load(open('$TRAIN_CONFIG'))['hyperparameters'], sys.stdout)")

echo ""
echo "Configuration:"
echo "  Features: $FEATURES"
echo "  Metadata: $METADATA"
echo "  Output: $OUTPUT_DIR"
echo "  Hyperparameters: $HYPERPARAMS_JSON"
echo ""

# Create output directories
mkdir -p "$OUTPUT_DIR"
mkdir -p logs/final_training

# Check GPU availability
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# Run final training
echo "=========================================="
echo "Starting final model training..."
echo "=========================================="
echo ""

mamba run -p ./env python scripts/training/train_final_model.py "$TRAIN_CONFIG"

echo ""
echo "=========================================="
echo "Job completed successfully!"
echo "End time: $(date)"
echo "=========================================="
