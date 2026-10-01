#!/bin/bash
#SBATCH --partition=edid
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --job-name=diana_v13_select
#SBATCH --output=logs/epoch_budget_v13/select_%j.out
#SBATCH --error=logs/epoch_budget_v13/select_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
################################################################################
# K3, second step (afterok run_epoch_budget_v13.sbatch): epoch budgets and the A6
# decision on the v13 dev folds for the fraction arms and for the signature arms, the
# paired test DIANA sig vs fraction, and the budget configs for the final fits.
# 38_ exits non-zero if an arm did not peak below the cap, which stops the chain.
################################################################################
set -e
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg
PY=./env/bin/python
echo "=== K3 select  job $SLURM_JOB_ID  $(hostname)  $(date) ==="
echo "--- budgets and A6, fractions"
$PY scripts/analysis/37_epoch_budget_select.py --base results/epoch_budget_v13 --splits data/splits_v13
echo "--- budgets and A6, fractions + signatures"
$PY scripts/analysis/37_epoch_budget_select.py --base results/epoch_budget_sig_v13 --splits data/splits_v13
echo "--- DIANA sig vs fraction, paired over BioProjects"
$PY scripts/analysis/48_paired_sig_vs_fraction.py --sig results/epoch_budget_sig_v13 \
    --frac results/epoch_budget_v13 --splits data/splits_v13
echo "--- budget configs for the final fits (fraction arms)"
$PY scripts/analysis/38_make_budget_configs.py --budget results/epoch_budget_v13/epoch_budget.tsv \
    --src configs/final_fixed_v13 --dst configs/final_budget_v13 --results results/final_budget_v13
echo "=== COMPLETE — $(date) ==="
