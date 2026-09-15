#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --array=0-4
#SBATCH --job-name=diana_seqlr
#SBATCH --output=logs/seqenc_lrsweep/%x_%a_%j.out
#SBATCH --error=logs/seqenc_lrsweep/%x_%a_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
################################################################################
# Reader learning-rate sweep on one dev fold. TUNING, NOT A RESULT.
#
# Tests one explanation for the S4 collapse: that the cards went uniform because the
# reader was driven at a rate chosen for an input layer 267x larger. `material` swept
# down from its inherited 5.57e-03, `feature` swept up from its inherited 2.41e-05. The
# MLP keeps its searched rate in every arm, so only the reader's rate changes.
#
# `gpu`, not `edid_rtx6000`: that node is sm_120 and the ./env torch is a cu118 build
# capped at sm_90. No --time; the partition default applies.
#
# USAGE:
#   sbatch scripts/analysis/run_seqenc_lrsweep.sbatch
################################################################################
set -e
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg

ORDER=results/seqenc_lrsweep_v9/configs/sweep_order.txt
TAG=$(sed -n "$(( SLURM_ARRAY_TASK_ID + 1 ))p" "$ORDER")
[ -n "$TAG" ] || { echo "ERROR: no entry $SLURM_ARRAY_TASK_ID in $ORDER" >&2; exit 1; }
CFG="results/seqenc_lrsweep_v9/configs/${TAG}.json"
TASK=${TAG%%_fold*}

mkdir -p logs/seqenc_lrsweep
echo "=== reader lr sweep: $TAG  $(date) ==="
echo "Config: $CFG"
nvidia-smi --query-gpu=name --format=csv,noheader || true

./env/bin/python scripts/training/02_train_final_model.py "$CFG"

./env/bin/diana-test \
  --model "results/seqenc_lrsweep_v9/fit_${TAG}/best_model.pth" \
  --config "$CFG" \
  --matrix data/matrices/matrix_v9_train/unitigs.frac.mat \
  --metadata data/splits_v9/train_metadata.tsv \
  --eligibility data/splits_v9/class_eligibility.tsv \
  --test-ids "results/calibration_v9/ids/cal_eval_fold0.txt" \
  --output "results/seqenc_lrsweep_v9/oof_${TAG}" --device cuda

echo "=== COMPLETE — $(date) ==="
