#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-4
#SBATCH --exclude=maestro-3444,maestro-3445,maestro-3448
#SBATCH --job-name=diana_final_weights_v19
#SBATCH --output=logs/final_v19/weights_%a_%j.out
#SBATCH --error=logs/final_v19/weights_%a_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
################################################################################
# The final DIANA network refitted with its weights saved (PROJECT.md ZENODO-MODEL, 2026-10-22):
# exactly run_final_fits_v19.sbatch for the shared network (configs/v19_final/multitask.json, its 173
# epochs, seeds 42, 1, 2, 3, 4, all 2,775 training runs, held-out predictions) plus --save-model.
# Written to results/final_v19_weights/ so results/final_v19/ (the read-2 fits) stays as it is.
################################################################################
set -euo pipefail
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg
SEEDS=(42 1 2 3 4)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
CFG=configs/v19_final/multitask.json
EPOCHS=$(./env/bin/python -c "import json; e=json.load(open('$CFG'))['epochs']; assert int(e)>0; print(int(e))")
OUT=results/final_v19_weights/multitask/seed${SEED}
[ -e "$OUT/model.pt" ] && { echo "$OUT/model.pt exists; refusing to overwrite" >&2; exit 1; }
echo "=== v19 final fit with weights, seed $SEED, $EPOCHS epochs -> $OUT  $(hostname)  $(date) ==="
./env/bin/python scripts/training/04_epoch_budget_dev_folds.py "$CFG" --fold 0 --out "$OUT" \
    --dev-folds data/splits_v19/dev_folds.tsv --max-epochs "$EPOCHS" --seed "$SEED" --save-probs --save-model \
    --also-predict-train --heldout results/v19_heldout/heldout_frac.npz \
    --heldout-ids results/v19_heldout/heldout_ids.txt --heldout-metadata results/v19_heldout/heldout_metadata.tsv
echo "=== COMPLETE $(date) ==="
