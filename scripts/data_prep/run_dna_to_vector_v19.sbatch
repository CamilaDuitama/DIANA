#!/bin/bash
#SBATCH --partition=edid,common
#SBATCH --exclude=maestro-2004,maestro-2008,maestro-2012,maestro-2016,maestro-2018,maestro-2022
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --array=0-39
#SBATCH --job-name=hv_dna_to_vector
#SBATCH --output=logs/hv_v19/dna_to_vector_%a_%j.out
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
################################################################################
# End-to-end check with the authors' own tool (Hypervectors/bin/dna_to_vector, commit below,
# assembly mode, k = 31, scaled = 1000, d = 2,048) on the Logan unitig files of the runs not in
# their released database, plus 20 runs that are (control). The tool writes its sketch next to
# its input, so it runs on symlinks in a per-task work folder, never in data/unitigs.
# Task i takes every 40th line of runs.txt starting at line i.
################################################################################
set -euo pipefail
SHA=6db91876c38eda0ceca5cd41b6affa22c2574e90
SRC=external/faure_sketches/SRA-metagenome-sketches-databases-$SHA/Hypervectors
W=results/hypervectors_v19/dna_to_vector
module load gcc/11.3.0 sourmash/4.9.4
BIN=$PWD/$SRC/bin/dna_to_vector
[ -x "$BIN" ] || { echo "missing $BIN (build first)" >&2; exit 1; }
WORK=$W/work_$SLURM_ARRAY_TASK_ID; mkdir -p "$WORK" "$W/vec"
awk -v t="$SLURM_ARRAY_TASK_ID" 'NR % 40 == t' "$W/runs.txt" | while read -r acc; do
    [ -s "$W/vec/$acc.bin" ] && continue
    ln -sf "$PWD/data/unitigs/$acc.unitigs.fa.gz" "$WORK/$acc.unitigs.fa.gz"
    "$BIN" "$WORK/$acc.unitigs.fa.gz" "$W/vec/$acc.bin" -m assembly > "$WORK/$acc.log" 2>&1 || { echo "FAILED $acc"; cat "$WORK/$acc.log"; exit 1; }
    echo "$acc $(grep -o 'Extracted [0-9]* hashes' "$WORK/$acc.log")"
done
echo "=== COMPLETE $(date) ==="
