#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --job-name=dnabert2_embed
#SBATCH --output=logs/embed/%x_%j.out
#SBATCH --error=logs/embed/%x_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
################################################################################
# S5a / S5b: cache one frozen DNABERT-2 description per unitig.
#
# A GPU is not optional here even for the 10-sequence probe: DNABERT-2's attention is a
# Triton flash-attention kernel that asserts `q.is_cuda` and fails outright on CPU.
#
# `gpu`, not `edid_rtx6000`: that node is sm_120 and the ./env torch is a cu118 build
# capped at sm_90. No --time; the partition default applies.
#
# USAGE:
#   MODE=probe sbatch scripts/data_prep/run_embed_dnabert2.sbatch     # S5a
#   MODE=full  sbatch scripts/data_prep/run_embed_dnabert2.sbatch     # S5b, real sequences
#   MODE=shuf  sbatch scripts/data_prep/run_embed_dnabert2.sbatch     # S5b, scrambled control
################################################################################
set -e
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg
export HF_HUB_DISABLE_PROGRESS_BARS=1

mkdir -p logs/embed
MODE="${MODE:-probe}"
echo "=== DNABERT-2 embedding [$MODE]  $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

case "$MODE" in
  probe) ARGS="--probe 10" ;;
  full)  ARGS="" ;;
  shuf)  ARGS="--shuffled" ;;
  *) echo "ERROR: MODE must be probe, full or shuf" >&2; exit 1 ;;
esac

./env/bin/python scripts/data_prep/18_embed_unitigs_dnabert2.py $ARGS
echo "=== COMPLETE — $(date) ==="
