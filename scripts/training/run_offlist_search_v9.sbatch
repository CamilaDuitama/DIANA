#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --array=0-3
#SBATCH --job-name=diana_offsearch
#SBATCH --output=logs/offlist_search/%x_%a_%j.out
#SBATCH --error=logs/offlist_search/%x_%a_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
################################################################################
# S7e: the hyperparameter search the appended-off-list arm needs.
#
# S6 taught us this is not optional. Its null was uninformative because it ran settings
# tuned for an input 768 columns narrower, and `feature` hit the 200-epoch cap with its
# best epoch AT the cap, so the arm with more parameters to fit was differentially
# truncated. Same trial count and search space as the fraction arm; only the matrix and
# the epoch cap differ.
#
# `gpu`, not `edid_rtx6000`: that node is sm_120 and the ./env torch is a cu118 build
# capped at sm_90. No --time.
#
# USAGE:
#   sbatch --export=ALL,ARM=real     scripts/training/run_offlist_search_v9.sbatch
#   sbatch --export=ALL,ARM=shuffled scripts/training/run_offlist_search_v9.sbatch
################################################################################
set -e
ARM="${ARM:-real}"
case "$ARM" in real) TAG="" ;; shuffled) TAG="shuf" ;;
  *) echo "ERROR: ARM must be real or shuffled" >&2; exit 1 ;; esac
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg
TASKS=(community_type feature sample_host material)
T=${TASKS[$SLURM_ARRAY_TASK_ID]}
CFG="results/offlist${TAG}_search_v9/configs/search_${T}.json"
[ -f "$CFG" ] || { echo "ERROR: no config at $CFG" >&2; exit 1; }
mkdir -p logs/offlist_search
echo "=== S7e search [$ARM]: $T  $(date) ==="
nvidia-smi --query-gpu=name --format=csv,noheader || true
# --fold_id 0 exactly as the fraction arm's search used it: the search runs over ALL of
# train with the dev folds as inner CV, so the argument selects the outer split the
# script was written around rather than restricting the search to one fold. Omitting it
# is a hard error, which is how the first submission died in 20 seconds.
./env/bin/python scripts/training/01_train_multitask_single_fold.py \
    --run-config "$CFG" --fold_id 0 --use_gpu
echo "=== COMPLETE — $(date) ==="
