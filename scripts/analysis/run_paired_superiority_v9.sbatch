#!/bin/bash
#SBATCH --partition=edid
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --job-name=diana_paired
#SBATCH --output=logs/paired_v9/paired_%j.out
#SBATCH --error=logs/paired_v9/paired_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
# A4: the paired bootstrap that decides superiority. CPU only.
set -e
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg
./env/bin/python scripts/analysis/18_paired_superiority.py --n-boot 2000
echo "=== COMPLETE — $(date) ==="
