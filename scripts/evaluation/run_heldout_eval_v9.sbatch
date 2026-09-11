#!/bin/bash
#SBATCH --partition=edid
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --job-name=diana_heldout_eval
#SBATCH --output=logs/heldout_eval_v9/eval_%j.out
#SBATCH --error=logs/heldout_eval_v9/eval_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
################################################################################
# The held-out read (A3b). All five frozen models, one pass, 922 runs.
#
# Both architectures are evaluated so nothing is selected: the multi-task net and
# the four single-task controls are reported beside the tuned baselines. Every
# model was frozen by run_final_fits_v9.sbatch before this ran, so this is one read,
# not five.
#
# Feature alignment was proved before this job: matrix_v9_heldout was built by
# 10_build_heldout_matrix_v9.py, which reads its own output back through
# MatrixLoader and compares 25 runs' rows against their original vector files. The
# fof order equals the column order, and the metadata is a 1:1 match on all 922.
################################################################################
set -e
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg

MATRIX=data/matrices/matrix_v9_heldout/unitigs.frac.mat
META=data/splits_v9/test_metadata.tsv
ELIG=data/splits_v9/class_eligibility.tsv
IDS=data/splits_v9/test_accessions.txt

run () {   # name  search_dir
  echo "===================== $1 ====================="
  ./env/bin/diana-test \
    --model   "$2/final_model/best_model.pth" \
    --config  "$2/final_training_config.json" \
    --matrix  "$MATRIX" --metadata "$META" --eligibility "$ELIG" --test-ids "$IDS" \
    --output  "results/final_eval_v9/heldout_$1" --device cpu
}

run multitask                results/search_v9_final
for t in community_type feature sample_host material; do
  run "single_$t"            "results/search_v9_final_single_$t"
done
echo "=== COMPLETE — $(date) ==="
