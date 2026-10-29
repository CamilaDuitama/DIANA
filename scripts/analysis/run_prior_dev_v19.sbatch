#!/bin/bash
#SBATCH --partition=edid,common
#SBATCH --exclude=maestro-2004,maestro-2008,maestro-2012,maestro-2016,maestro-2018,maestro-2022
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --job-name=prior_dev_v19
#SBATCH --output=logs/prior_dev_v19_%j.out
#SBATCH --error=logs/prior_dev_v19_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
# Class prior added back on the dev folds (CAL-B2, PROTOCOLS.md v19 amendment 2026-10-26): calibration, then detection
# on both benchmarks, corrected minus original for DIANA and logistic regression.
set -euo pipefail
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg
PY=./env/bin/python
$PY scripts/analysis/83_prior_correction_v19.py
for KIND in mixed record; do
  $PY scripts/analysis/41_flagging_v19.py --split-dir data/splits_v19 --plantings results/planted_v19/dev \
      --pattern "planted_train_${KIND}_r0.1_s*.tsv" \
      --model "DIANA=results/devfold_probs_v19/probs_DIANA_{task}.tsv" --model "DIANAprior=results/prior_v19/probs_DIANAprior_{task}.tsv" \
      --model "LogReg=results/baselines_v19/{task}/oof_probs_LogisticRegression_Bal.tsv" --model "LogRegprior=results/prior_v19/probs_LogRegprior_{task}.tsv" \
      --pair DIANAprior:DIANA --pair LogRegprior:LogReg --out "results/prior_v19/flag_${KIND}"
done
