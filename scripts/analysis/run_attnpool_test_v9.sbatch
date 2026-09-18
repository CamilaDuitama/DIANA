#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --array=0-19
#SBATCH --job-name=diana_attnpool
#SBATCH --output=logs/attnpool/%x_%a_%j.out
#SBATCH --error=logs/attnpool/%x_%a_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
################################################################################
# S5d: attention pooling over frozen DNABERT-2 descriptions.
#
# A sample holds about 26,600 of the 110,202 unitigs, most of them shared with every
# other sample. Every arm so far reduced that set with a FIXED weighted sum, which
# cannot say "this unitig matters in this sample but not that one", and both cheap
# sequence arms failed in exactly that way: S1 averaged and washed out, S4's learned
# weights drifted into being alike. Here the weights compete across only the unitigs
# present, and the descriptions are frozen so they cannot drift at all.
#
# ARM=real     -> results/attnpool_test_v9   (DNABERT-2 on the real sequences)
# ARM=shuffled -> results/attnpool_shuf_v9   (same, bases permuted within each unitig)
#
# The control separates two different claims: "the model reads DNA" and "attention beats
# a weighted sum". The shuffled arm keeps the second and removes the first.
#
# `gpu`, not `edid_rtx6000`: that node is sm_120 and the ./env torch is a cu118 build
# capped at sm_90. No --time; the partition default applies.
#
# USAGE:
#   sbatch --export=ALL,ARM=real     scripts/analysis/run_attnpool_test_v9.sbatch
#   sbatch --export=ALL,ARM=shuffled scripts/analysis/run_attnpool_test_v9.sbatch
################################################################################
set -e

ARM="${ARM:-real}"
case "$ARM" in
  real)     OUT_ROOT=results/attnpool_test_v9 ;;
  shuffled) OUT_ROOT=results/attnpool_shuf_v9 ;;
  *) echo "ERROR: ARM must be 'real' or 'shuffled', got '$ARM'" >&2; exit 1 ;;
esac

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg

TASKS=(community_type feature sample_host material)
T=${TASKS[$(( SLURM_ARRAY_TASK_ID / 5 ))]}
F=$(( SLURM_ARRAY_TASK_ID % 5 ))
CFG="$OUT_ROOT/configs/attnpool_${T}_fold${F}.json"
[ -f "$CFG" ] || { echo "ERROR: no config at $CFG (run prepare_attnpool_configs.py)" >&2; exit 1; }

mkdir -p logs/attnpool
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
