#!/bin/bash
#SBATCH --partition=edid
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --array=0-3
#SBATCH --job-name=diana_offhold
#SBATCH --output=logs/offlist_final/hold_%x_%a_%j.out
#SBATCH --error=logs/offlist_final/hold_%x_%a_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
################################################################################
# Re-evaluate the FROZEN S7 final models on the corrected held-out matrix.
#
# No retraining: the models were fitted on the all-train matrix, which never changed
# because every training sample already had its own unitig assembly. What changed is
# held-out, where 93 of 922 samples previously received a placeholder block equal to the
# training mean. Their Logan assemblies were downloaded on 2026-09-18, the pipeline rerun
# for them, and the matrices now have 922 of 922 distinct blocks.
#
# Still an additional held-out read, disclosed in PROJECT.md "Read first". No --time.
################################################################################
set -e
ARM="${ARM:-real}"
case "$ARM" in real) A=offlist; TAG="" ;; shuffled) A=offlistshuf; TAG="shuf" ;;
  *) echo "ERROR: ARM must be real or shuffled" >&2; exit 1 ;; esac
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg
TASKS=(community_type feature sample_host material)
T=${TASKS[$SLURM_ARRAY_TASK_ID]}
R="results/${A}_final_v9"
CFG="$R/configs/off${TAG}_final_${T}.json"
MAT="data/matrices/matrix_v9_heldout/unitigs.frac.off${TAG}.mat"
for f in "$CFG" "$MAT" "$R/fit_${T}/best_model.pth"; do
  [ -f "$f" ] || { echo "ERROR: missing $f" >&2; exit 1; }
done
echo "=== S7 held-out re-read [$ARM]: $T  $(date) ==="
./env/bin/diana-test --model "$R/fit_${T}/best_model.pth" --config "$CFG" \
  --matrix "$MAT" --metadata data/splits_v9/test_metadata.tsv \
  --eligibility data/splits_v9/class_eligibility.tsv \
  --test-ids data/splits_v9/test_accessions.txt \
  --output "$R/heldout_${T}" --device cpu
echo "=== COMPLETE — $(date) ==="
