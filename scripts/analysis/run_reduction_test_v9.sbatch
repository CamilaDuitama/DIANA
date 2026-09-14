#!/bin/bash
#SBATCH --partition=edid
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-79
#SBATCH --job-name=diana_reduction
#SBATCH --output=logs/reduction_test/red_%a_%j.out
#SBATCH --error=logs/reduction_test/red_%a_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
################################################################################
# G5 / G6: does DIANA do better on a reduced feature space?
#
# G1 tried reduction under logistic regression only. Four arms here, all rank ~192
# because that is DIANA's own trunk width, so the question is whether an unsupervised
# projection matches the supervised one the trunk learns:
#
#   frac_pca192     centred PCA-192 of the fraction
#   pa_pca192       centred PCA-192 of presence/absence
#   frac_pca191nd   the same, depth component dropped
#   pa_pca191nd     the same, depth component dropped
#
# The depth arms exist because it was MEASURED that the leading component is a
# library-size axis: |r| with the per-sample non-zero unitig count is 0.956 for the
# fraction and 1.000 for presence/absence, and centring does not remove it because
# depth genuinely IS the dominant variance direction. Dropping that one component
# takes the worst correlation to 0.151 / 0.135. R3.7 and R3.8 are about exactly this.
#
# The basis for each fold is fitted on that fold's four TRAINING folds only.
# Held-out is not touched.
################################################################################
set -e
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg

ARMS=(frac_pca192 pa_pca192 frac_pca191nd pa_pca191nd)
TASKS=(community_type feature sample_host material)
I=$SLURM_ARRAY_TASK_ID
A=${ARMS[$(( I / 20 ))]}
T=${TASKS[$(( (I % 20) / 5 ))]}
F=$(( I % 5 ))
CFG="results/reduction_test_v9/configs/${A}_${T}_fold${F}.json"
[ -f "$CFG" ] || { echo "ERROR: no config at $CFG" >&2; exit 1; }
MAT=$(./env/bin/python -c "import json;print(json.load(open('$CFG'))['features_path'])")

echo "=== $A / $T / fold $F  matrix=$MAT  $(date) ==="
./env/bin/python scripts/training/02_train_final_model.py "$CFG"
./env/bin/diana-test \
  --model "results/reduction_test_v9/fit_${A}_${T}_fold${F}/best_model.pth" \
  --config "$CFG" \
  --matrix "$MAT" \
  --metadata data/splits_v9/train_metadata.tsv \
  --eligibility data/splits_v9/class_eligibility.tsv \
  --test-ids "results/calibration_v9/ids/cal_eval_fold${F}.txt" \
  --output "results/reduction_test_v9/oof_${A}_${T}_fold${F}" --device cpu
echo "=== COMPLETE — $(date) ==="
