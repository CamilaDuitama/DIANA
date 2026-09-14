#!/bin/bash
#SBATCH --job-name=binarise_mat
#SBATCH --partition=edid
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/binarise/bin_%j.out
#SBATCH --error=logs/binarise/bin_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
set -euo pipefail
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled
echo "=== binarising the v9 fraction matrix === $(date)"
./env/bin/python scripts/data_prep/12_binarise_fraction_matrix.py
echo "=== done $(date)"
