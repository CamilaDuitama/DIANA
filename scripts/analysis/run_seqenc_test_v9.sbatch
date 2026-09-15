#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --array=0-19
#SBATCH --job-name=diana_seqenc
#SBATCH --output=logs/seqenc_test/%x_%a_%j.out
#SBATCH --error=logs/seqenc_test/%x_%a_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
################################################################################
# S4: does reading the bases beat storing a free vector per unitig?
#
# The input layer is Linear(110202, H0), so column u of its weight is a private vector
# of H0 numbers for unitig u: 14.1 M free numbers at H0=128 (community_type) up to
# 56.4 M at H0=512 (material), all fitted to 2,716 samples. This arm computes that
# vector from the unitig's bases with one small CNN shared by all 110,202 unitigs
# (about 100 k numbers), and gives each unitig two of them, one scaled by the fraction
# and one by presence.
#
# Everything else is held fixed: same fraction matrix, same 5 BioProject-grouped dev
# folds, same training ids, same Optuna-selected hyperparameters, same layers after the
# first. `results/calibration_v9` is therefore the paired fraction baseline on exactly
# these folds, and 26_pooled_representation_test.py is the scorer.
#
# ARM=real     -> results/seqenc_test_v9   (the sequences as they are)
# ARM=shuffled -> results/seqenc_shuf_v9   (bases permuted within each unitig, seed 42)
#
# The shuffled arm is not optional. It keeps every unitig's base composition and length
# and destroys only order, so a gain that appears in both arms is a capacity or
# regularisation effect, not a sequence effect.
#
# No --time: the partition default applies.
#
# `gpu`, not `edid_rtx6000`, and not the comma list. Two separate reasons, both probed on
# 2026-09-15:
#
#   1. edid_rtx6000 carries an RTX PRO 6000 Blackwell, compute capability sm_120, and the
#      torch 2.5.1 in ./env is built for sm_50 to sm_90. Job 14472258 reached "Using
#      device: cuda" on maestro-3447 and could not run a kernel. The `gpu` partition's
#      A100 (sm_80) and A40 (sm_86) nodes are supported.
#   2. ONE partition, not the comma list `edid_rtx6000,gpu` that run_fixed_arms_v9.sbatch uses.
# That list is now rejected outright: probed on 2026-09-15, `--partition=edid_rtx6000,gpu
# --gres=gpu:1` sits at BadConstraints while the identical request to either partition
# alone is accepted (jobs 14471553, 14471554 pending vs 14471699, 14471700 rejected). The
# edid_rtx6000 node advertises `gpu:rtx6000:7,gpu:2g.48gb+gfx:2`, so a generic gpu:1 does
# not resolve across both partitions. If edid_rtx6000 queues, resubmit the remaining
# indices with `sbatch --partition=gpu ...`, which needs no account change.
#
# USAGE:
#   sbatch --export=ALL,ARM=real     scripts/analysis/run_seqenc_test_v9.sbatch
#   sbatch --export=ALL,ARM=shuffled scripts/analysis/run_seqenc_test_v9.sbatch
#   sbatch --export=ALL,ARM=real --array=5 scripts/analysis/run_seqenc_test_v9.sbatch
################################################################################
set -e

ARM="${ARM:-real}"
case "$ARM" in
  real)     OUT_ROOT=results/seqenc_test_v9 ;;
  shuffled) OUT_ROOT=results/seqenc_shuf_v9 ;;
  *) echo "ERROR: ARM must be 'real' or 'shuffled', got '$ARM'" >&2; exit 1 ;;
esac

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg

TASKS=(community_type feature sample_host material)
T=${TASKS[$(( SLURM_ARRAY_TASK_ID / 5 ))]}
F=$(( SLURM_ARRAY_TASK_ID % 5 ))
CFG="$OUT_ROOT/configs/seqenc_${T}_fold${F}.json"
[ -f "$CFG" ] || { echo "ERROR: no config at $CFG (run prepare_seqenc_configs.py)" >&2; exit 1; }

mkdir -p logs/seqenc_test
echo "=== S4 sequence encoder [$ARM]: $T fold $F  $(date) ==="
echo "Config: $CFG"
echo "Node:   $(hostname)"
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
