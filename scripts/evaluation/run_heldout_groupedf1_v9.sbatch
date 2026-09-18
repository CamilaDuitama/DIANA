#!/bin/bash
#SBATCH --partition=edid
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=diana_f1_heldout
#SBATCH --output=logs/final_eval_groupedf1_v9/heldout_%j.out
#SBATCH --error=logs/final_eval_groupedf1_v9/heldout_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
################################################################################
# THIRD READ of the 922 held-out runs, declared as one. Requested explicitly after
# the train numbers were seen.
#
#   read 1  results/final_eval_v9/heldout_*            ungrouped early stopping
#   read 2  results/final_eval_grouped_v9/heldout_*    grouped split, stop on loss
#   read 3  THIS                                       grouped split, stop on macro-F1
#
# All five models were frozen by job 18013297 before this ran, and all five are
# reported, so nothing is selected on held-out.
#
# READ THESE NUMBERS WITH THE TRAIN NUMBERS BESIDE THEM. Two arms are known to be
# underfit on train before this job ran, so their cells describe broken models and
# not DIANA: single-task `feature` (train f1_macro_eligible 0.1392, only 561 of 2,716
# runs labelled so the validation signal is too thin) and single-task `sample_host`
# (0.1777, validation macro-F1 saturated at 1.0000 by epoch 0 so the criterion could
# not select). The interpretable arm is the multi-task net, which recovers to
# 0.9577 / 0.7789 / 0.6219 / 0.7241 on train against the original 0.9587 / 0.8631 /
# 0.6435 / 0.7923.
#
# Same matrix, metadata, eligibility table and ids as reads 1 and 2.
################################################################################
set -e
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg

for a in multitask community_type feature sample_host material; do
  echo "===================== heldout / $a ====================="
  ./env/bin/diana-test \
    --model   "results/final_groupedf1_v9/$a/final_model/best_model.pth" \
    --config  "results/final_groupedf1_v9/$a/final_model/final_training_config.json" \
    --matrix  data/matrices/matrix_v9_heldout/unitigs.frac.mat \
    --metadata data/splits_v9/test_metadata.tsv \
    --eligibility data/splits_v9/class_eligibility.tsv \
    --test-ids data/splits_v9/test_accessions.txt \
    --output  "results/final_eval_groupedf1_v9/heldout_$a" --device cpu
done
echo "=== COMPLETE — $(date) ==="
