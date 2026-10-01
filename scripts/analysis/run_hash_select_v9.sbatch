#!/bin/bash
#SBATCH --partition=edid
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --job-name=diana_hash_verdict
#SBATCH --output=logs/epoch_budget_hash_v9/verdict_%j.out
#SBATCH --error=logs/epoch_budget_hash_v9/verdict_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
################################################################################
# H4: verdicts on the v9 dev folds, per hash arm. Baselines: the selected hash config
# against the selected fraction config (14_ --pool). DIANA: budgets and A6 for the hash
# arms (37_), then hash vs fraction at each arm's budget (48_). afterok the H2 and H3 arrays.
################################################################################
set -e
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg
PY=./env/bin/python
for TAG in s1000_f272 s1000_f28 s1000_f14 s100_f272; do
  echo "=== baselines, $TAG: hash vs fraction ==="
  $PY scripts/evaluation/14_tune_baselines_s8a_v9.py --pool --splits data/splits_v9 --out results/baselines_hash_${TAG}_v9 \
      --rep-a hash --rep-b fraction
  echo "=== DIANA, $TAG: budgets and A6 ==="
  $PY scripts/analysis/37_epoch_budget_select.py --base results/epoch_budget_hash_${TAG}_v9 --splits data/splits_v9
  echo "=== DIANA, $TAG: hash vs fraction ==="
  $PY scripts/analysis/48_paired_sig_vs_fraction.py --sig results/epoch_budget_hash_${TAG}_v9 --frac results/epoch_budget_v9 \
      --splits data/splits_v9 --name-a hash_${TAG}
done
echo "=== COMPLETE — $(date) ==="
