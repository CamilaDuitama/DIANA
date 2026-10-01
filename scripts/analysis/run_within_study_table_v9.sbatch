#!/bin/bash
#SBATCH --partition=edid
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --job-name=diana_within_table
#SBATCH --output=logs/within_study_v9/table_%j.out
#SBATCH --error=logs/within_study_v9/table_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
################################################################################
# W4: within-study (new run, seen study) beside across-study (held-out read 9), all
# models, same metric; DIANA vs best baseline paired over BioProjects on the within-study
# side (52_within_vs_across_study.py). afterok the W2 and W3 arrays. Re-scoring only.
################################################################################
set -e
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg
./env/bin/python scripts/analysis/52_within_vs_across_study.py
echo "=== COMPLETE — $(date) ==="
