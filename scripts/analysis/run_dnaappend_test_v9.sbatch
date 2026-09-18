#!/bin/bash
#SBATCH --partition=edid
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --array=0-19
#SBATCH --job-name=diana_dnaapp
#SBATCH --output=logs/dna_append/%x_%a_%j.out
#SBATCH --error=logs/dna_append/%x_%a_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
################################################################################
# S6: keep the 110,202 fractions and APPEND 768 DNABERT-2 columns.
#
# The one sequence wiring never tested. S1 and S2 appended summaries that are close to
# linear functions of the fractions, which the first layer could already compute. S4 and S5
# REPLACED the 110,202 private per-unitig weight vectors, and that is where their losses
# came from: S5 lost 0.208 to 0.359 on held-out while its scrambled control scored the same
# as the real arm, so nothing there turned on the sequence. Appending keeps every private
# vector, so this arm can barely lose and a gain is attributable to the columns.
#
# The model is unchanged; its input layer widens from 110,202 to 110,970. Same folds, ids
# and hyperparameters as the fraction arm, so the calibration refits are the paired
# baseline on exactly these folds.
#
# SUMMARY=max   the per-dimension maximum over the unitigs present. Non-linear, ~12
#               effective dimensions of between-sample variation, the arm that can carry a
#               result.
# SUMMARY=mean  the fraction-weighted average. The literal form, near-redundant by
#               construction, ~5 effective dimensions. The comparison arm.
#
# CPU: no encoder is trained here, the descriptions are frozen and already pooled. No --time.
#
# USAGE:
#   sbatch --export=ALL,SUMMARY=max,TABLE=real      scripts/analysis/run_dnaappend_test_v9.sbatch
#   sbatch --export=ALL,SUMMARY=max,TABLE=shuffled  scripts/analysis/run_dnaappend_test_v9.sbatch
#   (likewise SUMMARY=mean; the shuffled arms are mandatory, see below)
#
# TABLE=shuffled is the control that makes a clearing cell reportable. max-versus-mean does
# NOT substitute for it: that varies the pooling function while holding the DNA fixed.
################################################################################
set -e
SUMMARY="${SUMMARY:-max}"
case "$SUMMARY" in max|mean) ;; *) echo "ERROR: SUMMARY must be max or mean" >&2; exit 1 ;; esac
TABLE="${TABLE:-real}"
case "$TABLE" in real) SUF="" ;; shuffled) SUF="shuf" ;;
  *) echo "ERROR: TABLE must be real or shuffled" >&2; exit 1 ;; esac
OUT_ROOT="results/dna${SUMMARY}${SUF}_test_v9"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg

TASKS=(community_type feature sample_host material)
T=${TASKS[$(( SLURM_ARRAY_TASK_ID / 5 ))]}
F=$(( SLURM_ARRAY_TASK_ID % 5 ))
CFG="$OUT_ROOT/configs/dna${SUMMARY}${SUF}_${T}_fold${F}.json"
[ -f "$CFG" ] || { echo "ERROR: no config at $CFG" >&2; exit 1; }

mkdir -p logs/dna_append
echo "=== S6 [$SUMMARY / $TABLE]: $T fold $F  $(date) ==="
./env/bin/python scripts/training/02_train_final_model.py "$CFG"
./env/bin/diana-test \
  --model "$OUT_ROOT/fit_${T}_fold${F}/best_model.pth" \
  --config "$CFG" \
  --matrix "data/matrices/matrix_v9_train/unitigs.frac.dna${SUMMARY}${SUF}.fold${F}.mat" \
  --metadata data/splits_v9/train_metadata.tsv \
  --eligibility data/splits_v9/class_eligibility.tsv \
  --test-ids "results/calibration_v9/ids/cal_eval_fold${F}.txt" \
  --output "$OUT_ROOT/oof_${T}_fold${F}" --device cpu
echo "=== COMPLETE — $(date) ==="
