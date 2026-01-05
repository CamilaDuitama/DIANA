#!/bin/bash
#SBATCH --job-name=diana-predict-test
#SBATCH --output=logs/inference/diana_predict_test_%j.out
#SBATCH --error=logs/inference/diana_predict_test_%j.err
#SBATCH --cpus-per-task=10
#SBATCH --mem=100G
#SBATCH --partition=seqbio

set -e

# Create log directory
mkdir -p logs/inference

echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: 100G"
echo ""

# Run diana-predict with mamba environment
mamba run -p ./env diana-predict \
  --sample data/validation/raw/ERR12206679/ERR12206679.fastq.gz \
  --model results/full_training/best_model.pth \
  --muset-matrix data/matrices/large_matrix_3070_with_frac \
  --output results/inference_test \
  --threads $SLURM_CPUS_PER_TASK \
  --verbose

echo ""
echo "Job completed at $(date)"
