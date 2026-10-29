#!/bin/bash
#SBATCH --partition=edid,common
#SBATCH --exclude=maestro-2004,maestro-2008,maestro-2012,maestro-2016,maestro-2018,maestro-2022
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --array=0-17
#SBATCH --job-name=net_screens_v19
#SBATCH --output=logs/net_v19/screen_%a_%j.out
#SBATCH --error=logs/net_v19/screen_%a_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
# Dev-fold screens re-scored with DIANA without logistic regression: each arm's network alone against the shared network
# (PROTOCOLS.md v19 amendment 2026-10-26). Arms = the scored arms of results/screens_v19/scores/.
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg
# ADV2 (PROTOCOLS.md 2026-10-27): ARMS=configs/screens_v19/arms_adv.tsv SCORES=results/screens_v19/scores_net_adv --array=0-1
if [ -n "${ARMS:-}" ]; then ARM=$(cut -f1 "$ARMS" | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")
else ARM=$(ls results/screens_v19/scores/ | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p"); fi
[ -n "$ARM" ] || { echo "no arm for task $SLURM_ARRAY_TASK_ID" >&2; exit 1; }
O=${SCORES:-results/screens_v19/scores_net}/$ARM
[ -e "$O/classification.tsv" ] && { echo "$O already scored; refusing to score twice" >&2; exit 1; }
mkdir -p "$O"
MODELS=(--model "DIANA=results/devfold_probs_v19net/probs_DIANA_{task}.tsv" --model "$ARM=results/screens_v19/probs/probs_${ARM}net_{task}.tsv")
./env/bin/python scripts/analysis/43_classification_v19.py "${MODELS[@]}" --pair "$ARM:DIANA" --out "$O"
./env/bin/python scripts/analysis/41_flagging_v19.py --split-dir data/splits_v19 --plantings results/planted_v19/dev \
    --pattern "planted_train_mixed_r0.1_s*.tsv" "${MODELS[@]}" --pair "$ARM:DIANA" --out "$O/flag_single"
echo "=== COMPLETE $ARM $(date) ==="
