#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --array=0-3
#SBATCH --job-name=diana_offfinal
#SBATCH --output=logs/offlist_final/%x_%a_%j.out
#SBATCH --error=logs/offlist_final/%x_%a_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
################################################################################
# S7 final fits and BOTH evaluations: train and held-out.
#
# One model per task on all 2,716 training runs, then scored on train and on the 922
# held-out runs, same matrix vocabulary, ids, metadata, eligibility table and metric as
# the 2026-09-11 pass, so the numbers land directly comparable to Table 2's DIANA column.
#
# THE HELD-OUT PASS IS AN ADDITIONAL READ and is disclosed in PROJECT.md "Read first".
# It is taken on request rather than because the arm cleared its pre-registered gate; the
# dev-fold screen (S7f) is running in parallel and its verdict is recorded beside these
# numbers whatever it says.
#
# The scrambled arm gets the same treatment, so the control exists on held-out too — it
# did not for S5, which left "is it the DNA or the extra columns" unanswerable there.
#
# Both matrices are standardised on all 2,716 training runs. No --time.
################################################################################
set -e
ARM="${ARM:-real}"
case "$ARM" in real) A=offlist; TAG="" ;; shuffled) A=offlistshuf; TAG="shuf" ;;
  *) echo "ERROR: ARM must be real or shuffled" >&2; exit 1 ;; esac
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg
TASKS=(community_type feature sample_host material)
T=${TASKS[$SLURM_ARRAY_TASK_ID]}
R="results/${A}_final_v9"
CFG="$R/configs/off${TAG}_final_${T}.json"
TRAIN_MAT="data/matrices/matrix_v9_train/unitigs.frac.off${TAG}.alltrain.mat"
HELD_MAT="data/matrices/matrix_v9_heldout/unitigs.frac.off${TAG}.mat"
for f in "$CFG" "$TRAIN_MAT" "$HELD_MAT"; do
  [ -f "$f" ] || { echo "ERROR: missing $f" >&2; exit 1; }
done
mkdir -p logs/offlist_final
echo "=== S7 final fit [$ARM]: $T  $(date) ==="
./env/bin/python scripts/training/02_train_final_model.py "$CFG"

echo "=== TRAIN evaluation: $T ==="
./env/bin/diana-test --model "$R/fit_${T}/best_model.pth" --config "$CFG" \
  --matrix "$TRAIN_MAT" --metadata data/splits_v9/train_metadata.tsv \
  --eligibility data/splits_v9/class_eligibility.tsv \
  --test-ids data/splits_v9/train_accessions.txt \
  --output "$R/train_${T}" --device cpu

echo "=== HELD-OUT evaluation (additional read, disclosed): $T ==="
./env/bin/diana-test --model "$R/fit_${T}/best_model.pth" --config "$CFG" \
  --matrix "$HELD_MAT" --metadata data/splits_v9/test_metadata.tsv \
  --eligibility data/splits_v9/class_eligibility.tsv \
  --test-ids data/splits_v9/test_accessions.txt \
  --output "$R/heldout_${T}" --device cpu
echo "=== COMPLETE — $(date) ==="
