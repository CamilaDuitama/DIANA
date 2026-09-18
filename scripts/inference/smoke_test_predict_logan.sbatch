#!/bin/bash
#SBATCH --job-name=diana_smoke_predict
#SBATCH --partition=edid
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/g9/smoke_predict_%j.out
#SBATCH --error=logs/g9/smoke_predict_%j.err
set -euo pipefail
cd /pasteur/helix/scratch/cduitama/EDID/decOM-classify
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg
export PATH="$PWD/env/bin:$PATH"
rm -rf results/predict_wiring_test/out2
./env/bin/python -m diana.cli.predict \
  --sample data/unitigs/ERR13896924.unitigs.fa.gz \
  --model results/search_v9_final_single_feature/final_model/best_model.pth \
  --muset-matrix results/predict_wiring_test/matrix \
  --output results/predict_wiring_test/out2 \
  --threads 4
