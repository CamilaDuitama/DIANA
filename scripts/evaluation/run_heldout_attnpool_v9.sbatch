#!/bin/bash
#SBATCH --partition=edid
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=diana_heldout_attnpool
#SBATCH --output=logs/heldout_eval_v9/attnpool_%j.out
#SBATCH --error=logs/heldout_eval_v9/attnpool_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
################################################################################
# AN ADDITIONAL HELD-OUT READ, requested explicitly on 2026-09-16.
#
# The four attention-pooling final models, frozen before this ran, over the 922
# held-out runs. One pass, four models, reported as ONE read and disclosed in
# PROJECT.md "Read first" whatever it returns.
#
# This was not the pre-registered plan. The dev-fold screen said the arm is worse
# than the fraction on all four tasks and improved none of 11 regime cells, and on
# that evidence the arm had not earned a read. It is being taken because it was
# asked for, and the dev-fold verdict is on record above it.
#
# Same matrix, ids, metadata, eligibility table and metric as the 2026-09-11 read,
# so the numbers are directly comparable to Table 2's DIANA column. No --time.
################################################################################
set -e
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg

MATRIX=data/matrices/matrix_v9_heldout/unitigs.frac.mat
META=data/splits_v9/test_metadata.tsv
ELIG=data/splits_v9/class_eligibility.tsv
IDS=data/splits_v9/test_accessions.txt

mkdir -p logs/heldout_eval_v9
for t in community_type feature sample_host material; do
  echo "===================== $t ====================="
  ./env/bin/diana-test \
    --model   "results/attnpool_final_v9/fit_$t/best_model.pth" \
    --config  "results/attnpool_final_v9/configs/attnpool_final_$t.json" \
    --matrix  "$MATRIX" --metadata "$META" --eligibility "$ELIG" --test-ids "$IDS" \
    --output  "results/attnpool_final_v9/heldout_$t" --device cpu
done
echo "=== COMPLETE — $(date) ==="
