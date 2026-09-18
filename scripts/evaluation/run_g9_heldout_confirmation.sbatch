#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --job-name=diana_g9
#SBATCH --output=logs/g9/g9_%j.out
#SBATCH --error=logs/g9/g9_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
################################################################################
# G9 — the pre-registered held-out confirmation. Written 2026-09-14, run 2026-09-18.
#
# ONE arm, ONE task: presence/absence on `feature`, nothing else. It is the only cell in
# 72 arm-task comparisons with an established dev-fold win (+0.086, [+0.008, +0.174]).
# Abundance is NOT included despite sitting at 98 % of bootstraps on community_type,
# because the pre-registration named one arm and one task before any of this was known.
#
# The model: one final fit on all 2,716 training runs with the `feature` single-task
# configuration, hyperparameters UNCHANGED from the fraction search. That handicaps
# presence/absence and is accepted, as recorded in the pre-registration.
#
# Held-out features: the existing fraction vectors binarised at > 0, the identical rule
# applied to training. No refitting, no recounting.
#
# Decision rule, fixed and not to be renegotiated: the paired interval on Δ excluding zero
# in presence/absence's favour is a confirmation. A tie or a negative Δ is reported as the
# dev-fold result FAILING TO REPLICATE, and the fraction stays as DIANA's input.
#
# This is an additional held-out read and is disclosed in "Read first" whatever it returns.
# No --time.
################################################################################
set -e
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg
mkdir -p logs/g9
R=results/g9_pa_feature_v9
CFG=$R/configs/g9_pa_feature.json
TRAIN_MAT=data/matrices/matrix_v9_train/unitigs.pa.mat
HELD_MAT=data/matrices/matrix_v9_heldout/unitigs.pa.mat
for f in "$CFG" "$TRAIN_MAT" "$HELD_MAT"; do
  [ -f "$f" ] || { echo "ERROR: missing $f" >&2; exit 1; }
done
echo "=== G9 final fit: presence/absence, feature  $(date) ==="
./env/bin/python scripts/training/02_train_final_model.py "$CFG"
echo "=== G9 held-out read ==="
./env/bin/diana-test --model "$R/fit_feature/best_model.pth" --config "$CFG" \
  --matrix "$HELD_MAT" --metadata data/splits_v9/test_metadata.tsv \
  --eligibility data/splits_v9/class_eligibility.tsv \
  --test-ids data/splits_v9/test_accessions.txt \
  --output "$R/heldout_feature" --device cpu
echo "=== COMPLETE — $(date) ==="
