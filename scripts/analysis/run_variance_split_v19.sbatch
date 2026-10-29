#!/bin/bash
#SBATCH --partition=edid,common
#SBATCH --exclude=maestro-2004,maestro-2008,maestro-2012,maestro-2016,maestro-2018,maestro-2022
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --job-name=varsplit_v19
#SBATCH --output=logs/varsplit_v19/varsplit_%j.out
#SBATCH --error=logs/varsplit_v19/varsplit_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
# VARSPLIT (PROTOCOLS.md v19 amendment 2026-10-27): class against BioProject within class, training runs only.
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg
./env/bin/python scripts/analysis/88_variance_split_v19.py
