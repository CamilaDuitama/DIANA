#!/bin/bash
#SBATCH --partition=edid,common
#SBATCH --exclude=maestro-2004,maestro-2008,maestro-2012,maestro-2016,maestro-2018,maestro-2022
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-5
#SBATCH --job-name=refit_score_v19
#SBATCH --output=logs/screens_v19/refit_score_%a_%j.out
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
# Held-out read 5 (PROJECT.md ledger, recorded before scoring): the released DIANA (results/final_v19_weights)
# scored exactly as read 2. Task 0 builds DIANA's held-out tables (49_); submit tasks 1-5 after it.
set -euo pipefail
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg
PY=./env/bin/python; D=results/heldout_v19/diana_refit; H=results/heldout_v19/baselines/{task}; M=results/multiplicity_v19
MODELS=(--model "DIANA=$D/probs_DIANA_{task}.tsv" --model "LogReg=$H/probs_LogisticRegression_Bal.tsv"
        --model "RandomForest=$H/probs_RandomForest_Bal.tsv" --model "kNN=$H/probs_kNN.tsv" --model "Majority=$H/probs_MajorityClass.tsv")
PAIRS=(--pair DIANA:LogReg --pair DIANA:RandomForest)
HT=(--heldout --plantings results/planted_v19/heldout)
case $SLURM_ARRAY_TASK_ID in
 0) $PY scripts/analysis/49_diana_heldout_v19.py --fits results/final_v19_weights/multitask --out "$D" ;;
 1) $PY scripts/analysis/75_classification_metrics_v19.py --heldout --diana "$D/probs_DIANA_{task}.tsv" --out "$M/refit_classification_heldout" ;;
 2) for kind in mixed record; do
        $PY scripts/analysis/41_flagging_v19.py "${HT[@]}" --split-dir data/splits_v19 --pattern "planted_test_${kind}_r0.1_s*.tsv" \
            "${MODELS[@]}" "${PAIRS[@]}" --out "results/heldout_v19/scores_refit/flag_$([ $kind = mixed ] && echo single || echo record)"
    done ;;
 3) $PY scripts/analysis/68_regime_tables_v19.py "${HT[@]}" --pattern "planted_test_mixed_r0.1_s*.tsv" "${MODELS[@]}" "${PAIRS[@]}" \
        --out "$M/refit_regime_single.tsv" ;;
 4) $PY scripts/analysis/68_regime_tables_v19.py --whole-field "${HT[@]}" --pattern "planted_test_mixed_r0.1_s*.tsv" "${MODELS[@]}" "${PAIRS[@]}" \
        --out "$M/refit_whole_field.tsv" ;;
 5) $PY scripts/analysis/73_real_mislabel_replay_v19.py --diana "$D/probs_DIANA_{task}.tsv" --out results/heldout_v19/real_mislabels_refit ;;
esac
echo "=== COMPLETE task $SLURM_ARRAY_TASK_ID $(date) ==="
