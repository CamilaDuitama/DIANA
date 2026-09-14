#!/bin/bash
#SBATCH --job-name=unitig_redund
#SBATCH --partition=edid
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=logs/redundancy/red_%j.out
#SBATCH --error=logs/redundancy/red_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
set -euo pipefail
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg
echo "=== unitig sequence redundancy, full corpus === $(date)"
./env/bin/python scripts/analysis/28_unitig_redundancy.py --k 15
echo "=== done $(date)"
