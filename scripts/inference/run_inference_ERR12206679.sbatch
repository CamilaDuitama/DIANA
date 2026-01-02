#!/bin/bash
#SBATCH --job-name=diana_infer_ERR12206679
#SBATCH --output=logs/inference/infer_ERR12206679_%j.out
#SBATCH --error=logs/inference/infer_ERR12206679_%j.err
#SBATCH --partition=seqbio
#SBATCH --nodelist=maestro-2010
#SBATCH --cpus-per-task=10
#SBATCH --mem=100G

set -e

# Print job information
echo "============================================"
echo "   SLURM Job Information"
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo "============================================"
echo ""

# Activate the environment using conda
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /pasteur/appa/scratch/cduitama/EDID/decOM-classify/env

bash scripts/inference/inference_pipeline.sh configs/sample_config_ERR12206679.sh

echo ""
echo "============================================"
echo "Job finished: $(date)"
echo "============================================"