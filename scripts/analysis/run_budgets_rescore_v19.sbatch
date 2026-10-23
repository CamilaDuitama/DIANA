#!/bin/bash
#SBATCH --partition=edid,common
#SBATCH --exclude=maestro-2004,maestro-2008,maestro-2012,maestro-2016,maestro-2018,maestro-2022
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-1
#SBATCH --job-name=budgets_v19
#SBATCH --output=logs/screens_v19/budgets_%a_%j.out
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
# Re-score (not a read; PROJECT.md ledger, recorded before scoring): paired planted-mislabel detection of the
# released DIANA at 5, 10 and 20 % budgets, by class group (task 0) and whole field (task 1).
set -euo pipefail
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg
PY=./env/bin/python; D=results/heldout_v19/diana_refit; H=results/heldout_v19/baselines/{task}; M=results/multiplicity_v19
MODELS=(--model "DIANA=$D/probs_DIANA_{task}.tsv" --model "LogReg=$H/probs_LogisticRegression_Bal.tsv"
        --model "RandomForest=$H/probs_RandomForest_Bal.tsv" --model "kNN=$H/probs_kNN.tsv" --model "Majority=$H/probs_MajorityClass.tsv")
ARGS=(--heldout --plantings results/planted_v19/heldout --pattern "planted_test_mixed_r0.1_s*.tsv" "${MODELS[@]}"
      --pair DIANA:LogReg --pair DIANA:RandomForest --pair-budgets 0.05,0.1,0.2)
case $SLURM_ARRAY_TASK_ID in
 0) $PY scripts/analysis/68_regime_tables_v19.py "${ARGS[@]}" --out "$M/refit_regime_single_budgets.tsv" ;;
 1) $PY scripts/analysis/68_regime_tables_v19.py --whole-field "${ARGS[@]}" --out "$M/refit_whole_field_budgets.tsv" ;;
esac
echo "=== COMPLETE task $SLURM_ARRAY_TASK_ID $(date) ==="
