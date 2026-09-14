#!/bin/bash
#SBATCH --partition=edid
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --array=0-19
#SBATCH --job-name=diana_pa
#SBATCH --output=logs/pa_test/pa_%a_%j.out
#SBATCH --error=logs/pa_test/pa_%a_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
################################################################################
# Screening test: does unitig ABUNDANCE beat the completeness FRACTION?
#
# MUSET defines them differently. Fraction f(u,S) averages a BINARY present/absent
# indicator over the unitig's k-mers, bounded in [0,1]. Abundance A(u,S) averages the
# k-mer COUNTS, unbounded. DIANA has only ever used fraction, so it discards all
# depth-of-coverage information. R3.7 names this distinction directly.
#
# Same 5 BioProject-grouped dev folds, same hyperparameters and same protocol as the
# calibration refits, which therefore serve as the fraction baseline on exactly these
# folds. Features are log1p(abundance), which lands at mean 0.276 / sd 0.681 against
# fraction's mean ~0.16, so the trainer consumes it unchanged and nothing is fitted
# across folds.
#
# The handicap is deliberate and must be read correctly: the hyperparameters were
# searched for fraction and are not re-searched here. A worse result is therefore
# uninformative. A comparable or better one justifies the full 6-8 h search.
#
# Held-out is not touched.
################################################################################
set -e
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg
TASKS=(community_type feature sample_host material)
T=${TASKS[$(( SLURM_ARRAY_TASK_ID / 5 ))]}
F=$(( SLURM_ARRAY_TASK_ID % 5 ))
CFG="results/pa_test_v9/configs/pa_${T}_fold${F}.json"
[ -f "$CFG" ] || { echo "ERROR: no config at $CFG" >&2; exit 1; }
echo "=== presence/absence screening: $T fold $F  $(date) ==="
./env/bin/python scripts/training/02_train_final_model.py "$CFG"
./env/bin/diana-test \
  --model "results/pa_test_v9/fit_${T}_fold${F}/best_model.pth" \
  --config "$CFG" \
  --matrix data/matrices/matrix_v9_train/unitigs.pa.mat \
  --metadata data/splits_v9/train_metadata.tsv \
  --eligibility data/splits_v9/class_eligibility.tsv \
  --test-ids "results/calibration_v9/ids/cal_eval_fold${F}.txt" \
  --output "results/pa_test_v9/oof_${T}_fold${F}" --device cpu
echo "=== COMPLETE — $(date) ==="
