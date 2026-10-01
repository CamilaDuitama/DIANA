#!/bin/bash
#SBATCH --partition=edid
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --array=0-7
#SBATCH --job-name=diana_flags
#SBATCH --output=logs/flag_postprocessing_v9/%x_%a_%j.out
#SBATCH --error=logs/flag_postprocessing_v9/%x_%a_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
################################################################################
# X11 (Phase 3): flag post-processing steps on the dev folds, 60_flag_postprocessing.py.
# Array index = benchmark (0: single, 1: record) x task; STEPS from the environment.
#   sbatch --export=ALL,STEPS=3.1:3.4 scripts/analysis/run_flag_postprocessing_v9.sbatch   (colon-separated: a comma would split --export)
#   sbatch --export=ALL,STEPS=3.7,COMBINED=1 --array=0 ...   (whole-record benchmark, all tasks)
#   sbatch --export=ALL,POOL=1 --array=0 ...                  (gather the paired tables)
################################################################################
set -o pipefail
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg
BENCHES=(single record); TASKS=(community_type feature sample_host material)
: "${STEPS:=}"
i=$SLURM_ARRAY_TASK_ID
if [ -n "${POOL:-}" ]; then
    ARGS=(--pool)
elif [ -n "${COMPOSE:-}" ]; then     # 3.8: COMPOSE="reference=3.2+3.5;LogisticRegression_Bal=3.2+3.5"
    ARGS=(--bench "${BENCHES[$((i / 4))]}" --task "${TASKS[$((i % 4))]}")
    IFS=';' read -ra SPECS <<< "$COMPOSE"
    for sp in "${SPECS[@]}"; do sp=${sp//+/,}; sp=${sp//=/:}; ARGS+=(--compose "$sp"); done
elif [ -n "${COMBINED:-}" ]; then
    ARGS=(--bench record --steps "${STEPS//:/,}")
else
    ARGS=(--bench "${BENCHES[$((i / 4))]}" --task "${TASKS[$((i % 4))]}" --steps "${STEPS//:/,}")
fi
echo "=== X11 flags ${ARGS[*]}  job $SLURM_JOB_ID  $(hostname)  $(date) ==="
./env/bin/python scripts/analysis/60_flag_postprocessing.py "${ARGS[@]}"
rc=$?
echo "=== exit $rc — $(date) ==="
exit $rc
