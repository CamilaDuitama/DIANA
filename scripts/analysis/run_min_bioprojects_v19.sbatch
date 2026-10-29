#!/bin/bash
#SBATCH --partition=edid,common
#SBATCH --exclude=maestro-2004,maestro-2008,maestro-2012,maestro-2016,maestro-2018,maestro-2022
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --array=0-3
#SBATCH --job-name=minbp_v19
#SBATCH --output=logs/minbp_v19_%a_%j.out
#SBATCH --error=logs/minbp_v19_%a_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
# Minimum number of held-out BioProjects for a paired comparison (PROTOCOLS.md v19 amendment 2026-10-26).
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg
TASKS=(community_type feature sample_host material)
./env/bin/python scripts/analysis/84_min_bioprojects_v19.py --task "${TASKS[$SLURM_ARRAY_TASK_ID]}"
