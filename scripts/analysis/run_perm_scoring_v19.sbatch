#!/bin/bash
#SBATCH --partition=edid,common
#SBATCH --exclude=maestro-2004,maestro-2008,maestro-2012,maestro-2016,maestro-2018,maestro-2022
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --array=0-3
#SBATCH --job-name=perm_score_v19
#SBATCH --output=logs/perm_score_v19_%a_%j.out
#SBATCH --error=logs/perm_score_v19_%a_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
# Re-score of the stored held-out predictions (reads 1 and 5) with the paired BioProject permutation test
# (PROTOCOLS.md v19 amendment 2026-10-26). New output files; earlier results are left in place.
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg
PY=./env/bin/python; D=results/heldout_v19/diana_refit; H=results/heldout_v19/baselines/{task}; M=results/multiplicity_v19
MODELS=(--model "DIANA=$D/probs_DIANA_{task}.tsv" --model "LogReg=$H/probs_LogisticRegression_Bal.tsv"
        --model "RandomForest=$H/probs_RandomForest_Bal.tsv" --model "kNN=$H/probs_kNN.tsv" --model "Majority=$H/probs_MajorityClass.tsv")
PAIRS=(--pair DIANA:LogReg --pair DIANA:RandomForest)
HT=(--heldout --plantings results/planted_v19/heldout)
case $SLURM_ARRAY_TASK_ID in
 0) $PY scripts/analysis/75_classification_metrics_v19.py --heldout --diana "$D/probs_DIANA_{task}.tsv" --out "$M/perm_classification_heldout" ;;
 1) $PY scripts/analysis/68_regime_tables_v19.py "${HT[@]}" --pattern "planted_test_mixed_r0.1_s*.tsv" "${MODELS[@]}" "${PAIRS[@]}" \
        --out "$M/perm_regime_single.tsv" ;;
 2) $PY scripts/analysis/68_regime_tables_v19.py --whole-field "${HT[@]}" --pattern "planted_test_mixed_r0.1_s*.tsv" "${MODELS[@]}" "${PAIRS[@]}" \
        --out "$M/perm_whole_field.tsv" ;;
 3) $PY scripts/analysis/82_genus_level_v19.py ;;
esac
