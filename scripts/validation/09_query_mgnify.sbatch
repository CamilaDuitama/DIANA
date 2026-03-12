#!/bin/bash
#SBATCH --job-name=mgnify_query
#SBATCH --output=logs/validation/mgnify_query_%j.log
#SBATCH --error=logs/validation/mgnify_query_%j.err
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --partition=seqbio
cd /pasteur/appa/scratch/cduitama/EDID/decOM-classify

echo "Starting MGnify query at $(date)"
echo "Running from: $(hostname)"

./env/bin/python scripts/validation/09_query_mgnify_modern_samples.py

echo "Finished at $(date)"
