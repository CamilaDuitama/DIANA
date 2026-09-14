#!/bin/bash
#SBATCH --job-name=g8_baselines_rep
#SBATCH --partition=edid
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --array=0-39%10
#SBATCH --output=logs/baselines_rep/g8_%a_%j.out
#SBATCH --error=logs/baselines_rep/g8_%a_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
################################################################################
# G8: the tuned baselines on fraction vs presence/absence, dev folds only.
#
# 2 representations x 4 tasks x 5 folds = 40 array tasks, 5 models each.
#
# Why an array and not one process: the first attempt (job 13794322, cancelled)
# looped all 200 fits in one process and ran at 88 % of ONE core on a 16-core
# allocation. LinearSVC is single-threaded and LogisticRegression's n_jobs does
# nothing under lbfgs, so only RandomForest used the cores. Parallelism has to come
# from SLURM. Throttled to 10 concurrent because each task holds a dense
# 2,716 x 110,202 float32 matrix plus scaler copies.
#
# Convergence is RECORDED per model: in G1, 49 of 120 logistic fits hit the
# iteration limit, which silently weakens any comparison built on them.
#
# Then: --pool to score. Pooled out-of-fold predictions, eligibility applied once,
# paired bootstrap over BioProjects -- never fold means.
################################################################################
set -e
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg

REPS=(fraction presence_absence)
TASKS=(community_type feature sample_host material)
I=$SLURM_ARRAY_TASK_ID
R=${REPS[$(( I / 20 ))]}
T=${TASKS[$(( (I % 20) / 5 ))]}
F=$(( I % 5 ))

echo "=== G8 $R / $T / fold $F === $(date)"
./env/bin/python scripts/analysis/27_baselines_representation_devfold.py \
    --rep "$R" --task "$T" --fold "$F"
echo "=== done $(date)"
