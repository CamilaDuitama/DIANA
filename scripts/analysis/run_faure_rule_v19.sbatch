#!/bin/bash
#SBATCH --partition=edid,common
#SBATCH --exclude=maestro-2004,maestro-2008,maestro-2012,maestro-2016,maestro-2018,maestro-2022
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --job-name=faure_rule
#SBATCH --output=logs/hv_v19/faure_rule_%j.out
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
# Faure et al. relabeling rule for our 2,618 runs: their `query` (built here, -march=native) fetches
# every stored neighbour from the released filtered matrix, then 70_ applies the rule and compares.
set -euo pipefail
SRC=external/faure_sketches/SRA-metagenome-sketches-databases-6db91876c38eda0ceca5cd41b6affa22c2574e90/Similarity-Matrix
M=data/external/faure_matrix
W=results/hypervectors_v19/matrix_check
module load gcc/11.3.0 cmake/3.27.7 hdf5/1.14.6 zlib/1.3.1
rm -rf "$SRC/build_query" && mkdir -p "$SRC/build_query"
( cd "$SRC/build_query" && cmake .. > cmake.log 2>&1 && make -j "$SLURM_CPUS_PER_TASK" query > make.log 2>&1 )
mkdir -p "$W/neighbours_all"
"$SRC/build_query/query" --matrix "$M/matrix" --db "$M/db" --query_file "$W/all_query_runs.txt" --show_all \
    --thread "$SLURM_CPUS_PER_TASK" --batch_size 100 --write_to_file "$W/neighbours_all/n.tsv" > "$W/neighbours_all.log"
ls "$W/neighbours_all" | wc -l
./env/bin/python scripts/analysis/70_faure_relabel_rule_v19.py "$W/neighbours_all"
