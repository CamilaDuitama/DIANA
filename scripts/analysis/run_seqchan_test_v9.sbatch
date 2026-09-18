#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --array=0-19
#SBATCH --job-name=diana_seqchan
#SBATCH --output=logs/seqchan/%x_%a_%j.out
#SBATCH --error=logs/seqchan/%x_%a_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
################################################################################
# S6: a sequence channel ADDED beside DIANA's input layer, which is kept.
#
# S1, S4 and S5 each removed or rewrote what the existing input layer does, and all
# three lost accuracy. The train numbers say why: the published model fits its own
# training data to 0.968 on `community_type` and keeps 0.568 on held-out, while S5
# reached 0.293 on the data it trained on, so it was not overfitting, it could not fit.
#
# This adds instead, and keeps the 110,202 x H0 weights:
#   h = Linear(110202, H0)(fractions) + Linear(heads*value_dim, H0)(pooled DNA)
# The second term starts at exactly zero, so an untrained model reproduces the current
# one bit for bit and the branch has to earn any contribution against validation loss.
#
# A null here is weak evidence, not proof that sequence is uninformative: the model can
# drive an unhelpful channel to zero, so "no change" is what both "nothing there" and
# "nothing reachable this way" look like.
#
# ARM=real     -> results/seqchan_test_v9   (descriptions of the real sequences)
# ARM=shuffled -> results/seqchan_shuf_v9   (same, bases permuted within each unitig)
#
# The control separates two different claims: "the model reads DNA" and "attention beats
# a weighted sum". The shuffled arm keeps the second and removes the first.
#
# `gpu`, not `edid_rtx6000`: that node is sm_120 and the ./env torch is a cu118 build
# capped at sm_90. No --time; the partition default applies.
#
# USAGE:
#   sbatch --export=ALL,ARM=real     scripts/analysis/run_seqchan_test_v9.sbatch
#   sbatch --export=ALL,ARM=shuffled scripts/analysis/run_seqchan_test_v9.sbatch
################################################################################
set -e

ARM="${ARM:-real}"
case "$ARM" in
  real)     OUT_ROOT=results/seqchan_test_v9 ;;
  shuffled) OUT_ROOT=results/seqchan_shuf_v9 ;;
  *) echo "ERROR: ARM must be 'real' or 'shuffled', got '$ARM'" >&2; exit 1 ;;
esac

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg

TASKS=(community_type feature sample_host material)
T=${TASKS[$(( SLURM_ARRAY_TASK_ID / 5 ))]}
F=$(( SLURM_ARRAY_TASK_ID % 5 ))
CFG="$OUT_ROOT/configs/seqchan_${T}_fold${F}.json"
[ -f "$CFG" ] || { echo "ERROR: no config at $CFG (run prepare_seqchan_configs.py)" >&2; exit 1; }

mkdir -p logs/seqchan
echo "=== S5d attention pooling [$ARM]: $T fold $F  $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

./env/bin/python scripts/training/02_train_final_model.py "$CFG"

./env/bin/diana-test \
  --model "$OUT_ROOT/fit_${T}_fold${F}/best_model.pth" \
  --config "$CFG" \
  --matrix data/matrices/matrix_v9_train/unitigs.frac.mat \
  --metadata data/splits_v9/train_metadata.tsv \
  --eligibility data/splits_v9/class_eligibility.tsv \
  --test-ids "results/calibration_v9/ids/cal_eval_fold${F}.txt" \
  --output "$OUT_ROOT/oof_${T}_fold${F}" --device cuda

echo "=== COMPLETE — $(date) ==="
