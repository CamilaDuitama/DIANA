#!/bin/bash
#SBATCH --partition=edid_rtx6000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --job-name=diana_final_bp_v5
#SBATCH --output=logs/final_training/diana_final_bp_v5_%j.out
#SBATCH --error=logs/final_training/diana_final_bp_v5_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cduitama@pasteur.fr
#SBATCH --chdir=/pasteur/appa/scratch/cduitama/EDID/decOM-classify

################################################################################
# DIANA v5 Final Model Training — BioProject-Disjoint Split
#
# v5 = fresh HPO with label smoothing removed from the search space (ε=0 fixed).
#      Properly tuned hyperparameters for a model without label smoothing.
#
# PREREQUISITES — run in order after all 5 HPO folds complete:
#
#   1. Aggregate CV results and build final_training_config.json:
#        mamba run -p ./env python scripts/training/aggregate_cv_results.py \
#            --cv_dir results/training_bioproject_v5/cv_results \
#            --n_folds 5
#
#      This writes:
#        results/training_bioproject_v5/cv_results/aggregated_results.json
#        results/training_bioproject_v5/final_training_config.json
#
#   2. Then submit this script:
#        sbatch scripts/training/run_final_train_edid_v5.sbatch
#
# NEXT STEPS after training completes:
#   1. Test evaluation:
#        sbatch scripts/evaluation/run_test_eval_v5.sbatch
#   2. Validation inference:
#        sbatch --array=1-611 scripts/validation/run_inference_bioproject_v5.sbatch
#   3. Training-set metrics:
#        mamba run -p ./env python scripts/evaluation/06_compute_training_metrics.py
#        (update config.py to point to v5 first)
#   4. Patch baseline metrics:
#        mamba run -p ./env python scripts/evaluation/patch_diana_in_metrics.py
#   5. Regenerate all paper materials:
#        bash scripts/paper/generate_all_paper_materials.sh
################################################################################

set -e

echo "=== DIANA v5 FINAL TRAINING (BioProject-disjoint, no label smoothing, re-tuned HPO) ==="
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $(hostname)"
echo "Start:   $(date)"
echo ""

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

eval "$(/pasteur/appa/homes/cduitama/miniconda3/bin/conda shell.bash hook)"
conda activate ./env
echo "Python: $(which python)"

TRAIN_CONFIG="results/training_bioproject_v5/final_training_config.json"

if [ ! -f "$TRAIN_CONFIG" ]; then
    echo "ERROR: Training config not found: $TRAIN_CONFIG"
    echo "Run aggregate_cv_results.py first (see script header for instructions)."
    exit 1
fi

echo "Config: $TRAIN_CONFIG"
OUTPUT_DIR=$(python -c "import json; print(json.load(open('$TRAIN_CONFIG'))['output_dir'])")
echo "Output:  $OUTPUT_DIR"
echo ""

mkdir -p "$OUTPUT_DIR" logs/final_training

echo "GPU:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

echo "Starting final training..."
python scripts/training/02_train_final_model.py "$TRAIN_CONFIG"

echo ""
echo "=== TRAINING COMPLETE — $(date) ==="
