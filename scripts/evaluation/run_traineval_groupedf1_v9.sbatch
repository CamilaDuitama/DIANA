#!/bin/bash
#SBATCH --partition=edid
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=diana_f1_traineval
#SBATCH --output=logs/final_eval_groupedf1_v9/traineval_%j.out
#SBATCH --error=logs/final_eval_groupedf1_v9/traineval_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
################################################################################
# TRAIN ONLY. No held-out read.
#
# Scores the five arms refitted with a BioProject-grouped early-stopping split AND
# early stopping on mean macro-F1 instead of validation loss
# (results/final_groupedf1_v9/, job 18013297).
#
# Deliberately separate from any held-out evaluation, because the previous round put
# both in one job and the train collapse could therefore not gate the read. Train is
# not the protected set; whether a further held-out read is taken is a separate
# decision made after these numbers are seen.
################################################################################
set -e
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg

for a in multitask community_type feature sample_host material; do
  echo "===================== train / $a ====================="
  ./env/bin/diana-test \
    --model   "results/final_groupedf1_v9/$a/final_model/best_model.pth" \
    --config  "results/final_groupedf1_v9/$a/final_model/final_training_config.json" \
    --matrix  data/matrices/matrix_v9_train/unitigs.frac.mat \
    --metadata data/splits_v9/train_metadata.tsv \
    --eligibility data/splits_v9/class_eligibility.tsv \
    --test-ids data/splits_v9/train_accessions.txt \
    --output  "results/final_eval_groupedf1_v9/train_$a" --device cpu
done
echo "=== COMPLETE — $(date) ==="
