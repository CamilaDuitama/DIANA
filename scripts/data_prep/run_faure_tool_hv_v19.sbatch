#!/bin/bash
#SBATCH --partition=edid,common
#SBATCH --exclude=maestro-2004,maestro-2008,maestro-2012,maestro-2016,maestro-2018,maestro-2022
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --job-name=hv_faure_tool
#SBATCH --output=logs/hv_v19/faure_tool_%j.out
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
################################################################################
# v19 HV test, PROTOCOLS.md amendment of 2026-10-15: hypervectors of all 3,638 runs with the
# authors' released code (commit below, `project_everything build`, d = 2,048), from our
# sketches repackaged one zip per run. Built and run in the same job because their CMake
# flags include -march=native.
################################################################################
set -euo pipefail
SHA=6db91876c38eda0ceca5cd41b6affa22c2574e90
SRC=external/faure_sketches/SRA-metagenome-sketches-databases-$SHA
OUT=results/hypervectors_v19/faure_tool
mkdir -p external/faure_sketches "$OUT"
if [ ! -d "$SRC" ]; then
    curl -sSL "https://codeload.github.com/RolandFaure/SRA-metagenome-sketches-databases/tar.gz/$SHA" -o external/faure_sketches/repo_$SHA.tar.gz
    tar xzf external/faure_sketches/repo_$SHA.tar.gz -C external/faure_sketches
fi
module load gcc/11.3.0 cmake/3.27.7 hdf5/1.14.6 zlib/1.3.1
echo "=== build on $(hostname) $(date) ==="
rm -rf "$SRC/Similarity-Matrix/build" && mkdir -p "$SRC/Similarity-Matrix/build"
( cd "$SRC/Similarity-Matrix/build" && cmake .. > cmake.log && make -j "$SLURM_CPUS_PER_TASK" project_everything > make.log )
BIN=$SRC/Similarity-Matrix/build/project_everything
./env/bin/python scripts/data_prep/37_split_sketch_zip_v19.py "$OUT/sig_zips"
echo "=== project_everything build $(date) ==="
"$BIN" build "$OUT/sig_zips" "$OUT/db" -t "$SLURM_CPUS_PER_TASK" -d 2048 > "$OUT/project_everything.log"
ls -la "$OUT/db"; cat "$OUT/db/dimension.txt" "$OUT/db/dtype.txt"; wc -l "$OUT/db/vector_norms.txt"
echo "=== COMPLETE $(date) ==="
