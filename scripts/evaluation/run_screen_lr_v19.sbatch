#!/bin/bash
#SBATCH --partition=edid,common
#SBATCH --exclude=maestro-2004,maestro-2008,maestro-2012,maestro-2016,maestro-2018,maestro-2022
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-75
#SBATCH --job-name=screen_lr_v19
#SBATCH --output=logs/screens_v19/lr_%a_%j.out
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
# v19 screens (PROTOCOLS.md 2026-10-21), logistic regression parts, after run_screen_inputs_v19.sbatch.
# Tasks 0-39: per task and dev fold, the network-on-logistic-regression inputs (19_ linear init,
# 18_ logit offsets) from the v19 fraction matrix at the v19 selected C. Tasks 40-75: logistic
# regression (v19 grid and selection, 40_) on each new input, per task.
set -euo pipefail
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg
TASKS=(community_type feature sample_host material)
REPS=(pa abund frac_pca192 pa_pca192 frac_pca191nd pa_pca191nd frac_kmer4 frac_clust logdepth)
declare -A FILE=([pa]=pa.npz [abund]=abund_log1p.npz [frac_pca192]=frac_pca192_fold{fold}.npz [pa_pca192]=pa_pca192_fold{fold}.npz
                 [frac_pca191nd]=frac_pca191nd_fold{fold}.npz [pa_pca191nd]=pa_pca191nd_fold{fold}.npz
                 [frac_kmer4]=frac_kmer4_fold{fold}.npz [frac_clust]=frac_clust_fold{fold}.npz [logdepth]=frac_logdepth.npz)
I=results/screens_v19/inputs; L=results/screens_v19/lr_inputs
P="results/baselines_v19/{task}/best_LogisticRegression_Bal.json"
i=$SLURM_ARRAY_TASK_ID
if [ "$i" -lt 40 ]; then
    T=${TASKS[$(((i % 20) / 5))]}; F=$((i % 5))
    if [ "$i" -lt 20 ]; then
        ./env/bin/python scripts/evaluation/19_logreg_init_v9.py --task "$T" --fold "$F" --splits data/splits_v19 \
            --npz "$I/frac.npz" --out "$L/linear_init" --params-json "$P" --n-jobs "$SLURM_CPUS_PER_TASK"
    else
        ./env/bin/python scripts/evaluation/18_logreg_offsets_v9.py --task "$T" --fold "$F" --splits data/splits_v19 \
            --npz "$I/frac.npz" --out "$L/offsets" --params-json "$P" --n-jobs "$SLURM_CPUS_PER_TASK"
    fi
else
    j=$((i - 40)); R=${REPS[$((j / 4))]}; T=${TASKS[$((j % 4))]}
    ./env/bin/python scripts/evaluation/40_baselines_v19.py --task "$T" --model LogisticRegression_Bal \
        --split-dir data/splits_v19 --matrix "$I/${FILE[$R]}" --out "results/screens_v19/lr/$R" --n-jobs "$SLURM_CPUS_PER_TASK"
fi
echo "=== COMPLETE task $i $(date) ==="
