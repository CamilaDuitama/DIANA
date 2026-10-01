#!/bin/bash
#SBATCH --partition=edid
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --job-name=diana_v13_pool
#SBATCH --output=logs/baselines_v13/pool_%j.out
#SBATCH --error=logs/baselines_v13/pool_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
################################################################################
# K2, second step (afterok run_baselines_v13.sbatch): per model the selected sig config
# against the selected fraction config, paired over BioProjects on the pooled dev-fold
# predictions; and the selections written as best_per_model_<rep>.tsv, which the
# held-out baseline run (09_baselines_v9.py --tuned) reads.
################################################################################
set -e
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg
echo "=== K2 pool  job $SLURM_JOB_ID  $(hostname)  $(date) ==="
./env/bin/python scripts/evaluation/14_tune_baselines_s8a_v9.py --pool --splits data/splits_v13 \
    --out results/baselines_v13 --rep-a sig --rep-b fraction
[ -s results/baselines_v13/best_per_model_fraction.tsv ] || { echo "best_per_model_fraction.tsv not written" >&2; exit 1; }
n=$(tail -n +2 results/baselines_v13/best_per_model_fraction.tsv | wc -l)
[ "$n" -eq 16 ] || { echo "expected 16 selected fraction configs, found $n" >&2; exit 1; }
echo "=== COMPLETE — $(date) ==="
