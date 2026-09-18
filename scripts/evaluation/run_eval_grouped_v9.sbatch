#!/bin/bash
#SBATCH --partition=edid
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=diana_grouped_eval
#SBATCH --output=logs/final_eval_grouped_v9/eval_%j.out
#SBATCH --error=logs/final_eval_grouped_v9/eval_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
################################################################################
# Train and held-out evaluation of the five arms refitted with a BioProject-grouped
# early-stopping split (results/final_grouped_v9/).
#
# THIS IS A SECOND READ OF HELD-OUT, and it is declared as one. The first read was
# A3b (run_heldout_eval_v9.sbatch) on the models fitted with an ungrouped
# early-stopping split. Those models were selected by a criterion that let runs from
# one BioProject sit on both sides, so their stopping epoch was chosen on a
# contaminated signal. The refit corrects that defect; the numbers it produces must
# be reported as a second read, never merged with the first.
#
# All five models were frozen by run_final_fits_grouped_v9.sbatch (job 18007446)
# before this job was written, so this is one read of the 922 runs, not five.
# Nothing here is selected on held-out: all five arms are reported.
#
# Writes to results/final_eval_grouped_v9/. results/final_eval_v9/ is NOT touched;
# referee responses cite numbers measured there.
#
# Same matrix, metadata, eligibility table and ids as the first read, so the two are
# comparable. No --time: the partition default applies.
################################################################################
set -e
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg

ELIG=data/splits_v9/class_eligibility.tsv
OUT=results/final_eval_grouped_v9

evaluate () {   # arm  split  matrix  metadata  ids
  echo "===================== $2 / $1 ====================="
  ./env/bin/diana-test \
    --model   "results/final_grouped_v9/$1/final_model/best_model.pth" \
    --config  "results/final_grouped_v9/$1/final_model/final_training_config.json" \
    --matrix  "$3" --metadata "$4" --eligibility "$ELIG" --test-ids "$5" \
    --output  "$OUT/$2_$1" --device cpu
}

# --- train first: not the protected set, and it catches a broken model before the
# --- held-out read is spent on it.
for a in multitask community_type feature sample_host material; do
  evaluate "$a" train \
    data/matrices/matrix_v9_train/unitigs.frac.mat \
    data/splits_v9/train_metadata.tsv \
    data/splits_v9/train_accessions.txt
done

# --- the declared second read
for a in multitask community_type feature sample_host material; do
  evaluate "$a" heldout \
    data/matrices/matrix_v9_heldout/unitigs.frac.mat \
    data/splits_v9/test_metadata.tsv \
    data/splits_v9/test_accessions.txt
done
echo "=== COMPLETE — $(date) ==="
