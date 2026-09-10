#!/bin/bash
#SBATCH --partition=edid_rtx6000,gpu
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --job-name=diana_pooled_search
#SBATCH --output=logs/pooled_search_v9/pooled_%A_%a.out
#SBATCH --error=logs/pooled_search_v9/pooled_%A_%a.err
#SBATCH --array=0-4
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
################################################################################
# The same five arms as run_final_search_v9.sbatch, under the POOLED objective.
#
# Why both sets exist. The first v9 searches maximise (balanced_accuracy +
# f1_macro) / 2 over all classes with out-of-vocabulary rows scored wrong, while
# the baselines are selected on f1_macro_eligible with those rows dropped
# (12_tune_baselines_v9.py). The two arms of the R1.3 comparison were therefore
# tuned toward different targets. Optuna's TPE picks each trial from the previous
# trials' values, so the objective decides which configurations are ever tried and
# no after-the-fact re-ranking can repair it.
#
# These run ALONGSIDE the originals rather than replacing them: nothing already
# computed is discarded, and the choice between the two sets is made on the dev
# folds by the rule written into PROJECT.md before either finished. Held-out is
# untouched by both.
#
# Task 0 is the multi-task arm; 1-4 are the single-task controls. All five must
# share an objective or the architecture test (A2e) compares arms that differ in
# two things at once.
################################################################################
set -e
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg

CONFIGS=(
  configs/train_config_v9_pooled_search.json
  configs/train_config_v9_pooled_search_single_community_type.json
  configs/train_config_v9_pooled_search_single_feature.json
  configs/train_config_v9_pooled_search_single_sample_host.json
  configs/train_config_v9_pooled_search_single_material.json
)
CFG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}
echo "=== pooled search  task $SLURM_ARRAY_TASK_ID  $CFG  $(hostname)  $(date) ==="
./env/bin/python scripts/training/01_train_multitask_single_fold.py \
    --run-config "$CFG" \
    --fold_id 0 --use_gpu
echo "=== COMPLETE — $(date) ==="
