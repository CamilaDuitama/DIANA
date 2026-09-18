#!/bin/bash
#SBATCH --partition=edid
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --array=0-3
#SBATCH --job-name=diana_dnafinal
#SBATCH --output=logs/heldout_eval_v9/dnafinal_%a_%j.out
#SBATCH --error=logs/heldout_eval_v9/dnafinal_%a_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
################################################################################
# S6 final fits and the held-out read, one array task per target.
#
# Fits on all 2,716 training runs with the appended columns, then reads the 922 held-out
# runs. Same matrix vocabulary, ids, metadata, eligibility table and metric as the
# 2026-09-11 pass, so the numbers land directly comparable to Table 2's DIANA column.
#
# AN ADDITIONAL HELD-OUT READ, requested explicitly. It is disclosed in PROJECT.md
# "Read first" alongside the dev-fold verdict for this arm, which is that all eight
# whole-task comparisons are ties.
#
# Training and held-out matrices are both standardised on all 2,716 training runs. The
# first version of this job trained on a fold matrix and tested on the all-train one, which
# straddled two scalings; those results were discarded rather than adjusted.
#
# The appended block on held-out is standardised on TRAIN statistics only. Using held-out's
# own mean and sd would put held-out information into every held-out input and no score
# would reveal it.
#
# No --time.
#
# USAGE: sbatch --export=ALL,SUMMARY=max scripts/evaluation/run_dnaappend_final_and_heldout.sbatch
################################################################################
set -e
SUMMARY="${SUMMARY:-max}"
TABLE="${TABLE:-real}"
case "$TABLE" in real) SUF="" ;; shuffled) SUF="shuf" ;;
  *) echo "ERROR: TABLE must be real or shuffled" >&2; exit 1 ;; esac
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg

TASKS=(community_type feature sample_host material)
T=${TASKS[$SLURM_ARRAY_TASK_ID]}
ROOT_OUT="results/dna${SUMMARY}${SUF}_final_v9"
CFG="$ROOT_OUT/configs/dna${SUMMARY}${SUF}_final_${T}.json"
[ -f "$CFG" ] || { echo "ERROR: no config at $CFG" >&2; exit 1; }

mkdir -p logs/heldout_eval_v9
echo "=== S6 final fit [$SUMMARY / $TABLE]: $T  $(date) ==="
./env/bin/python scripts/training/02_train_final_model.py "$CFG"

echo "=== held-out read: $T ==="
./env/bin/diana-test \
  --model "$ROOT_OUT/fit_${T}/best_model.pth" \
  --config "$CFG" \
  --matrix "data/matrices/matrix_v9_heldout/unitigs.frac.dna${SUMMARY}${SUF}.mat" \
  --metadata data/splits_v9/test_metadata.tsv \
  --eligibility data/splits_v9/class_eligibility.tsv \
  --test-ids data/splits_v9/test_accessions.txt \
  --output "$ROOT_OUT/heldout_${T}" --device cpu
echo "=== COMPLETE — $(date) ==="
