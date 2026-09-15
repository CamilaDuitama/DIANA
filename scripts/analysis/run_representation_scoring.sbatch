#!/bin/bash
#SBATCH --job-name=rep_scoring
#SBATCH --partition=edid
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/scoring/score_%j.out
#SBATCH --error=logs/scoring/score_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
################################################################################
# Score every representation arm against the fraction, two ways.
#
# No --time: the partition default applies.
#
# Both are bootstrap-heavy and cannot run on the login node: the per-regime test
# alone is 4 tasks x 3 regimes x 4 arms x 2,000 resamples, each computing two
# macro-F1 scores over up to 2,639 rows.
################################################################################
set -euo pipefail
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg
echo "=== per task === $(date)"
./env/bin/python scripts/analysis/26_pooled_representation_test.py
echo
echo "=== per regime === $(date)"
./env/bin/python scripts/analysis/29_regime_representation_test.py
echo "=== done $(date)"
