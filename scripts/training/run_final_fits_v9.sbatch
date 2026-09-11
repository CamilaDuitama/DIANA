#!/bin/bash
#SBATCH --partition=edid_rtx6000,gpu
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-4
#SBATCH --job-name=diana_final_fits
#SBATCH --output=logs/final_train_v9/fit_%a_%j.out
#SBATCH --error=logs/final_train_v9/fit_%a_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
################################################################################
# The final fits: all five arms on all 2,716 training runs, no search.
#
# Both architectures are fitted rather than one being chosen, so the held-out read
# reports multi-task, four single-task heads and the baselines side by side instead
# of a single selected model. Every model is frozen before held-out is touched, so
# this is still one read.
#
# The architecture question is then answered on held-out by the paired bootstrap
# over BioProjects (A4), not on the dev folds. That is the weaker place to test it
# (34 projects, wide intervals), and the reason is recorded in PROJECT.md.
#
# Task 0 is the multi-task net; 1-4 are the single-task controls.
################################################################################
set -e
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg

CONFIGS=(
    "results/search_v9_final/final_training_config.json"
    "results/search_v9_final_single_community_type/final_training_config.json"
    "results/search_v9_final_single_feature/final_training_config.json"
    "results/search_v9_final_single_sample_host/final_training_config.json"
    "results/search_v9_final_single_material/final_training_config.json"
)
CFG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}
[ -f "$CFG" ] || { echo "ERROR: no config at $CFG — run 03_final_config_from_search.py" >&2; exit 1; }

echo "=== final fit  task $SLURM_ARRAY_TASK_ID  $CFG  $(hostname)  $(date) ==="
./env/bin/python scripts/training/02_train_final_model.py "$CFG"
echo "=== COMPLETE — $(date) ==="
