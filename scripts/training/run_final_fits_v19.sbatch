#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-24
#SBATCH --exclude=maestro-3444,maestro-3445,maestro-3448
#SBATCH --job-name=diana_final_v19
#SBATCH --output=logs/final_v19/fit_%a_%j.out
#SBATCH --error=logs/final_v19/fit_%a_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
################################################################################
# v19 P11 (PROTOCOLS.md v19, C10): each arm refitted on all 2,775 training runs at its
# selected epoch count (configs/v19_final/<arm>.json "epochs"), five seeds; the fitted
# networks predict the held-out runs (probabilities saved, NOT scored here: scoring is
# the held-out read and needs Camila's approval) and their own training runs (train-set
# gate). Task i: arm = i / 5, seed = SEEDS[i % 5].
################################################################################
set -euo pipefail
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg
ARMS=(community_type feature sample_host material multitask)
SEEDS=(42 1 2 3 4)
ARM=${ARMS[$((SLURM_ARRAY_TASK_ID / 5))]}
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID % 5))]}
CFG=configs/v19_final/${ARM}.json
[ -f "$CFG" ] || { echo "no selected config $CFG" >&2; exit 1; }
EPOCHS=$(./env/bin/python -c "import json,sys; e=json.load(open('$CFG'))['epochs']; assert int(e)>0; print(int(e))")
OUT=results/final_v19/${ARM}/seed${SEED}
echo "=== v19 final fit  $ARM seed $SEED, $EPOCHS epochs -> $OUT  $(hostname)  $(date) ==="
./env/bin/python scripts/training/04_epoch_budget_dev_folds.py "$CFG" --fold 0 --out "$OUT" \
    --dev-folds data/splits_v19/dev_folds.tsv --max-epochs "$EPOCHS" --seed "$SEED" --save-probs \
    --also-predict-train --heldout results/v19_heldout/heldout_frac.npz \
    --heldout-ids results/v19_heldout/heldout_ids.txt --heldout-metadata results/v19_heldout/heldout_metadata.tsv
echo "=== COMPLETE $(date) ==="
