#!/bin/bash
#SBATCH --partition=edid
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --job-name=diana_heldout_matrix
#SBATCH --output=logs/heldout_matrix_v9/build_%j.out
#SBATCH --error=logs/heldout_matrix_v9/build_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
# CPU work: reads 922 per-sample vectors and writes one matrix. No GPU.
set -e
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg
echo "=== build held-out matrix  $(hostname)  $(date) ==="
./env/bin/python scripts/data_prep/10_build_heldout_matrix_v9.py
echo "=== COMPLETE — $(date) ==="
