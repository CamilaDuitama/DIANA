#!/bin/bash
#SBATCH --partition=edid,common
#SBATCH --exclude=maestro-2004,maestro-2008,maestro-2012,maestro-2016,maestro-2018,maestro-2022
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-13
#SBATCH --job-name=screen_score_v19
#SBATCH --output=logs/screens_v19/score_%a_%j.out
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
# v19 screens (PROTOCOLS.md 2026-10-21): each arm's candidate against DIANA on the dev folds, paired
# over BioProjects: classification (43_), single-label detection (41_), by regime (68_); for the input
# arms also logistic regression on the new input against logistic regression on the fractions.
# Task i = line i + 1 of configs/screens_v19/arms.tsv. Probability tables from 71_ (run before).
set -euo pipefail
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg
ARMS=${ARMS:-configs/screens_v19/arms.tsv}   # arms_adapted.tsv for the five adapted arms (--array=0-4)
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$ARMS")
IFS=$'\t' read -r ARM TMPL _ _ <<< "$LINE"
P=results/screens_v19/probs; O=results/screens_v19/scores/$ARM
[ -e "$O/classification.tsv" ] && { echo "$O already scored; refusing to score twice" >&2; exit 1; }
MODELS=(--model "DIANA=results/devfold_probs_v19/probs_DIANA_{task}.tsv" --model "$ARM=$P/probs_${ARM}_{task}.tsv")
PAIRS=(--pair "$ARM:DIANA")
if [[ "$TMPL" == configs/screens_v19/* ]]; then
    MODELS+=(--model "LogReg=results/baselines_v19/{task}/oof_probs_LogisticRegression_Bal.tsv"
             --model "LogReg_$ARM=results/screens_v19/lr/$ARM/{task}/oof_probs_LogisticRegression_Bal.tsv")
    PAIRS+=(--pair "LogReg_$ARM:LogReg")
fi
mkdir -p "$O"
./env/bin/python scripts/analysis/43_classification_v19.py "${MODELS[@]}" "${PAIRS[@]}" --out "$O"
./env/bin/python scripts/analysis/41_flagging_v19.py --split-dir data/splits_v19 --plantings results/planted_v19/dev \
    --pattern "planted_train_mixed_r0.1_s*.tsv" "${MODELS[@]}" "${PAIRS[@]}" --out "$O/flag_single"
./env/bin/python scripts/analysis/68_regime_tables_v19.py "${MODELS[@]}" "${PAIRS[@]}" --out "$O/regime.tsv"
echo "=== COMPLETE $ARM $(date) ==="
