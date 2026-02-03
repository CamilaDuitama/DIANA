#!/bin/bash
#SBATCH --job-name=training_metrics
#SBATCH --output=logs/training_metrics_%j.out
#SBATCH --error=logs/training_metrics_%j.err
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=20
#SBATCH --mem=40G
#SBATCH --partition=seqbio


echo "Starting training set evaluation..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"

# Force unbuffered output for real-time logging
export PYTHONUNBUFFERED=1

mamba run -p ./env python scripts/evaluation/06_compute_training_metrics.py

echo "Done!"
