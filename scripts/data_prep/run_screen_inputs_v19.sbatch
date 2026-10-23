#!/bin/bash
#SBATCH --partition=edid,common
#SBATCH --exclude=maestro-2004,maestro-2008,maestro-2012,maestro-2016,maestro-2018,maestro-2022
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --job-name=screen_inputs_v19
#SBATCH --output=logs/screens_v19/inputs_%j.out
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
# v19 screens (PROTOCOLS.md 2026-10-21): unitig clusters of the v17 unitigs (15_, containment >= 0.30
# on canonical 15-mers, as on v9), then every representation input (41_).
set -euo pipefail
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg
O=results/screens_v19/inputs; mkdir -p "$O"
./env/bin/python scripts/data_prep/15_unitig_clusters.py --fasta data/matrices/matrix_v17_train/unitigs.fa --out "$O"
./env/bin/python scripts/data_prep/41_screen_inputs_v19.py --clusters "$O/cluster_labels_k15_t0.30.npy"
echo "=== COMPLETE $(date) ==="
