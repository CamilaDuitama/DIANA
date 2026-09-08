#!/bin/bash
#SBATCH --job-name=diana_base_v9
#SBATCH --partition=edid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=logs/baseline/baselines_v9_%j.out
#SBATCH --error=logs/baseline/baselines_v9_%j.err

# Baselines on the v9 BioProject-disjoint partition (PROJECT.md G7).
# CPU work, so edid (seqbio as fallback if edid is full).
set -euo pipefail
cd /pasteur/helix/scratch/cduitama/EDID/decOM-classify
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export KMP_INIT_AT_FORK=FALSE
export KMP_AFFINITY=disabled
export MPLBACKEND=Agg

echo "Node: $(hostname)  Start: $(date)"
./env/bin/python scripts/evaluation/09_baselines_v9.py "$@"
echo "End: $(date)"
