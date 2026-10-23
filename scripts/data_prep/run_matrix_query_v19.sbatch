#!/bin/bash
#SBATCH --partition=edid,common
#SBATCH --exclude=maestro-2004,maestro-2008,maestro-2012,maestro-2016,maestro-2018,maestro-2022
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=hv_matrix_query
#SBATCH --output=logs/hv_v19/matrix_query_%j.out
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
################################################################################
# Matrix check for the report to Faure et al.: build their `query` (Similarity-Matrix, commit
# below, CMake flags include -march=native, so built and run in one job) and fetch the stored
# neighbours of 200 of our runs from the released filtered matrix.
################################################################################
set -euo pipefail
SHA=6db91876c38eda0ceca5cd41b6affa22c2574e90
SRC=external/faure_sketches/SRA-metagenome-sketches-databases-$SHA/Similarity-Matrix
M=data/external/faure_matrix
W=results/hypervectors_v19/matrix_check
module load gcc/11.3.0 cmake/3.27.7 hdf5/1.14.6 zlib/1.3.1
ls -la "$M" "$M/db"; ls "$M/matrix" | wc -l
rm -rf "$SRC/build_query" && mkdir -p "$SRC/build_query"
( cd "$SRC/build_query" && cmake .. > cmake.log 2>&1 && make -j "$SLURM_CPUS_PER_TASK" query > make.log 2>&1 )
MAT=$M/matrix
DB=$M/db
echo "matrix $MAT  db $DB"
mkdir -p "$W/neighbours"
"$SRC/build_query/query" --matrix "$MAT" --db "$DB" --query_file "$W/query_runs.txt" --show_all \
    --thread "$SLURM_CPUS_PER_TASK" --batch_size 50 --write_to_file "$W/neighbours/n.tsv"
ls "$W/neighbours" | wc -l
echo "=== COMPLETE $(date) ==="
