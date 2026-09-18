#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --array=0-19
#SBATCH --job-name=diana_offtest
#SBATCH --output=logs/offlist_test/%x_%a_%j.out
#SBATCH --error=logs/offlist_test/%x_%a_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
################################################################################
# S7f: the dev-fold screen for the appended off-list arm.
#
# 4 tasks x 5 BioProject-grouped dev folds. Each fit uses this arm's OWN searched
# hyperparameters, which is the difference from S6: there a null meant nothing because
# only the fraction side had been tuned.
#
# ARM=real     -> results/offlist_test_v9
# ARM=shuffled -> results/offlistshuf_test_v9, the control that separates "the DNA content
#                 helps" from "3,072 more columns help". Not optional: the condition fixed
#                 on 2026-09-15 requires a clearing cell NOT to clear here.
#
# `gpu`, not edid_rtx6000 (sm_120 against a cu118 torch). No --time.
################################################################################
set -e
ARM="${ARM:-real}"
case "$ARM" in real) A=offlist; TAG="" ;; shuffled) A=offlistshuf; TAG="shuf" ;;
  *) echo "ERROR: ARM must be real or shuffled" >&2; exit 1 ;; esac
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg
TASKS=(community_type feature sample_host material)
T=${TASKS[$(( SLURM_ARRAY_TASK_ID / 5 ))]}
F=$(( SLURM_ARRAY_TASK_ID % 5 ))
CFG="results/${A}_test_v9/configs/off${TAG}_${T}_fold${F}.json"
[ -f "$CFG" ] || { echo "ERROR: no config at $CFG" >&2; exit 1; }
MATRIX="data/matrices/matrix_v9_train/unitigs.frac.off${TAG}.fold${F}.mat"
[ -f "$MATRIX" ] || { echo "ERROR: no matrix at $MATRIX" >&2; exit 1; }
mkdir -p logs/offlist_test
echo "=== S7f [$ARM]: $T fold $F  $(date) ==="
./env/bin/python scripts/training/02_train_final_model.py "$CFG"
./env/bin/diana-test \
  --model "results/${A}_test_v9/fit_${T}_fold${F}/best_model.pth" \
  --config "$CFG" --matrix "$MATRIX" \
  --metadata data/splits_v9/train_metadata.tsv \
  --eligibility data/splits_v9/class_eligibility.tsv \
  --test-ids "results/calibration_v9/ids/cal_eval_fold${F}.txt" \
  --output "results/${A}_test_v9/oof_${T}_fold${F}" --device cpu
echo "=== COMPLETE — $(date) ==="
