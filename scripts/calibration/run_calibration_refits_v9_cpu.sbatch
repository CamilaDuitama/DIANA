#!/bin/bash
#SBATCH --partition=edid
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-19
#SBATCH --job-name=diana_calib_refit
#SBATCH --output=logs/calibration_v9/cal_%a_%j.out
#SBATCH --error=logs/calibration_v9/cal_%a_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
################################################################################
# C1, step 1 of 2: out-of-fold refits for temperature scaling (R3.14).
#
# Why not scripts/calibration/01_calibrate_model.py: it fits the temperature on a
# random 10 % of train, stratified by `sample_type`. That task was dropped in v9 so
# it would raise, and more importantly a random split shares BioProjects with the
# fitting data, which makes the model look better calibrated than it is on unseen
# studies. Same leak class as R3.4.
#
# Instead: refit the final hyperparameters on 4 of the 5 BioProject-grouped dev
# folds, 4 tasks x 5 folds = 20 fits. Each held-back fold is BioProject-disjoint
# from its own fit, so pooling the 5 gives out-of-fold probabilities for all 2,716
# training runs, none of them predicted by a model that saw its study.
#
# Step 2 fits one temperature per task on those pooled probabilities.
# Held-out is not touched at any point.
################################################################################
set -e
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg

TASKS=(community_type feature sample_host material)
T=${TASKS[$(( SLURM_ARRAY_TASK_ID / 5 ))]}
F=$(( SLURM_ARRAY_TASK_ID % 5 ))
CFG="results/calibration_v9/configs/cal_${T}_fold${F}.json"
[ -f "$CFG" ] || { echo "ERROR: no config at $CFG" >&2; exit 1; }

echo "=== calibration refit: $T fold $F  $(hostname)  $(date) ==="
./env/bin/python scripts/training/02_train_final_model.py "$CFG"

echo "=== predicting the held-back fold ==="
./env/bin/diana-test \
  --model   "results/calibration_v9/fit_${T}_fold${F}/best_model.pth" \
  --config  "$CFG" \
  --matrix  data/matrices/matrix_v9_train/unitigs.frac.mat \
  --metadata data/splits_v9/train_metadata.tsv \
  --eligibility data/splits_v9/class_eligibility.tsv \
  --test-ids "results/calibration_v9/ids/cal_eval_fold${F}.txt" \
  --output  "results/calibration_v9/oof_${T}_fold${F}" \
  --device cpu
echo "=== COMPLETE — $(date) ==="
