#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-4
#SBATCH --exclude=maestro-3444,maestro-3445,maestro-3448
#SBATCH --job-name=diana_trainfit_v19
#SBATCH --output=logs/train_fit_v19/fit_%a_%j.out
#SBATCH --error=logs/train_fit_v19/fit_%a_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
################################################################################
# v19 training-set evaluation: the shared (multi-task) arm refitted exactly as in
# run_final_fits_v19.sbatch (all 2,775 training runs, selected epoch count, same five
# seeds), now also saving probabilities on its own training runs (train_probs.npz).
# The final fits did not keep them. GPU training is not bit-for-bit reproducible, so these
# networks are refits of the same recipe, not the final networks; the held-out
# probabilities they also write are not used.
################################################################################
set -euo pipefail
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg
SEEDS=(42 1 2 3 4)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
CFG=configs/v19_final/multitask.json
EPOCHS=$(./env/bin/python -c "import json,sys; e=json.load(open('$CFG'))['epochs']; assert int(e)>0; print(int(e))")
OUT=results/train_fit_v19/multitask/seed${SEED}
echo "=== v19 train-set refit  multitask seed $SEED, $EPOCHS epochs -> $OUT  $(hostname)  $(date) ==="
./env/bin/python scripts/training/04_epoch_budget_dev_folds.py "$CFG" --fold 0 --out "$OUT" \
    --dev-folds data/splits_v19/dev_folds.tsv --max-epochs "$EPOCHS" --seed "$SEED" --save-probs \
    --also-predict-train --heldout results/v19_heldout/heldout_frac.npz \
    --heldout-ids results/v19_heldout/heldout_ids.txt --heldout-metadata results/v19_heldout/heldout_metadata.tsv
echo "=== COMPLETE $(date) ==="
