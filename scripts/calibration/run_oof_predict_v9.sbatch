#!/bin/bash
#SBATCH --partition=edid
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-19
#SBATCH --job-name=diana_oof_predict
#SBATCH --output=logs/calibration_v9/oof_%a_%j.out
#SBATCH --error=logs/calibration_v9/oof_%a_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
# C1 step 1b: predict each held-back dev fold with the model fitted on the other
# four. Split out from the refit job because the refits already succeeded and only
# the prediction path was wrong (02_train_final_model.py writes into output_dir
# itself, not into an output_dir/final_model subdirectory).
set -e
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg
TASKS=(community_type feature sample_host material)
T=${TASKS[$(( SLURM_ARRAY_TASK_ID / 5 ))]}
F=$(( SLURM_ARRAY_TASK_ID % 5 ))
M="results/calibration_v9/fit_${T}_fold${F}/best_model.pth"
[ -f "$M" ] || { echo "ERROR: no model at $M" >&2; exit 1; }
echo "=== out-of-fold predict: $T fold $F  $(date) ==="
./env/bin/diana-test \
  --model "$M" \
  --config "results/calibration_v9/configs/cal_${T}_fold${F}.json" \
  --matrix data/matrices/matrix_v9_train/unitigs.frac.mat \
  --metadata data/splits_v9/train_metadata.tsv \
  --eligibility data/splits_v9/class_eligibility.tsv \
  --test-ids "results/calibration_v9/ids/cal_eval_fold${F}.txt" \
  --output "results/calibration_v9/oof_${T}_fold${F}" \
  --device cpu
echo "=== COMPLETE — $(date) ==="
