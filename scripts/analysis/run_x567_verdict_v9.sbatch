#!/bin/bash
#SBATCH --partition=edid
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --job-name=diana_x567_verdict
#SBATCH --output=logs/x7_v9/verdict_%j.out
#SBATCH --error=logs/x7_v9/verdict_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
################################################################################
# X5 to X7 verdicts on the grouped dev folds, every variant paired against the current
# fixed-budget arms (results/epoch_budget_v9). afterok the four fit arrays.
################################################################################
set -e
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg
PY=./env/bin/python
FRAC=results/epoch_budget_v9
echo "################ X5: stopping rules ################"
for RULE in random grouped; do
  echo "=== X5 $RULE: pooled predictions at the selected epochs ==="
  $PY scripts/analysis/53_pool_selected_epoch.py --base results/x5_${RULE}_v9
  echo "=== X5 $RULE vs fixed budget ==="
  $PY scripts/analysis/48_paired_sig_vs_fraction.py --sig results/x5_${RULE}_v9 --frac $FRAC --name-a x5_${RULE}
done
echo "################ X6: learning rate and schedule under the budget ################"
for d in results/x6_grid_v9/lr*_*/; do
  echo "=== budgets for $d ==="
  $PY scripts/analysis/37_epoch_budget_select.py --base "$d" --arms community_type,feature,sample_host,material
done
$PY scripts/analysis/55_x6_select.py --base results/x6_grid_v9 --out results/x6_selected_v9
echo "=== X6 selected vs current arms (selected-then-tested on the same folds: optimistic) ==="
$PY scripts/analysis/48_paired_sig_vs_fraction.py --sig results/x6_selected_v9 --frac $FRAC --name-a x6 --single-only
echo "################ X7a: weight averaging ################"
$PY scripts/analysis/37_epoch_budget_select.py --base results/x7a_ema_v9
$PY scripts/analysis/48_paired_sig_vs_fraction.py --sig results/x7a_ema_v9 --frac $FRAC --name-a x7a_ema
echo "################ X7b: seed ensemble and seed spread ################"
$PY scripts/analysis/54_seed_ensemble.py --base results/x7b_seeds_v9 --seeds 42,1,2,3,4 --out results/x7b_ensemble_v9
$PY scripts/analysis/48_paired_sig_vs_fraction.py --sig results/x7b_ensemble_v9 --frac $FRAC --name-a x7b_ens --single-only
echo "=== COMPLETE — $(date) ==="
