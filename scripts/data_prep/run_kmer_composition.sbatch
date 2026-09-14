#!/bin/bash
#SBATCH --job-name=s1_kmer_comp
#SBATCH --partition=edid
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=logs/kmer_comp/s1_%j.out
#SBATCH --error=logs/kmer_comp/s1_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
################################################################################
# S1: append canonical 4-mer composition columns to the v9 fraction matrix.
#
# Each unitig is described once by its canonical 4-mer frequencies (136 features,
# counted with sklearn's CountVectorizer, not a hand-rolled loop). Each sample gets
# the fraction-weighted average of the descriptions of the unitigs it contains, and
# those 136 columns are APPENDED to the 110,202 fractions.
#
# Appended, not substituted: replacing the fractions with a projection is a fixed
# linear map, and that family has tied 36 times (16 DIANA, 20 logistic regression).
#
# One matrix per dev fold, because the appended columns are standardised using the
# mean and sd of that fold's four TRAINING folds only.
################################################################################
set -euo pipefail
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg
echo "=== S1 k-mer composition matrices === $(date)"
./env/bin/python scripts/data_prep/14_kmer_composition_features.py --k 4
echo "=== done $(date)"
