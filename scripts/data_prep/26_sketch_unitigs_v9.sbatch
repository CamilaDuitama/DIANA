#!/bin/bash
#SBATCH --partition=edid
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --job-name=diana_sketch_v9
#SBATCH --output=logs/sketches_v9/sketch_%j.out
#SBATCH --error=logs/sketches_v9/sketch_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
################################################################################
# S8.0: FracMinHash sketch of every run in the v9 split (2,716 train + 922
# held-out), from the same Logan unitig files the matrix was built from.
#
# sourmash 4.9.4 from the module tree, whose branchwater plugin (Rust) provides
# `manysketch`: one process, all files, 32 cores. Probe on one core:
# 5.7 MB gz/s (22 s for the median 97 MB file), so ~73 core-hours for the
# 1.5 TB corpus and about 2.5 h here.
#
# k=31 to match the matrix; scaled=1000 keeps 1 in 1,000 k-mers by hash value,
# the same slice for every sample; noabund because a unitig carries each k-mer
# once, so abundance would be meaningless on this input.
#
# Output: data/sketches_v9/unitigs_split_v9.zip, one signature per run, named
# by accession. Nothing existing is touched. No --time (edid has no limit).
################################################################################
set -o pipefail
for f in /opt/gensoft/adm/Modules/5.6.1/init/bash /opt/gensoft/adm/Modules/init/bash; do
    [ -f "$f" ] && { source "$f"; break; }
done
module load sourmash/4.9.4 || { echo "ERROR: cannot load sourmash/4.9.4" >&2; exit 1; }

OUT=data/sketches_v9
ZIP=$OUT/unitigs_split_v9.zip
MANIFEST=$OUT/manifest_split_v9.csv
[ -e "$ZIP" ] && { echo "refusing to overwrite $ZIP" >&2; exit 1; }
mkdir -p "$OUT" logs/sketches_v9

echo "=== S8.0 sketch  job $SLURM_JOB_ID  $(hostname)  $(date) ==="
echo "name,genome_filename,protein_filename" > "$MANIFEST"
MISSING=0
while read -r acc; do
    f=data/unitigs/$acc.unitigs.fa.gz
    if [ -s "$f" ]; then echo "$acc,$f," >> "$MANIFEST"; else MISSING=$((MISSING+1)); echo "missing or empty: $f" >&2; fi
done < <(cat data/splits_v9/train_accessions.txt data/splits_v9/test_accessions.txt)
N=$(($(wc -l < "$MANIFEST") - 1))
echo "manifest: $N runs, $MISSING missing"
[ "$MISSING" -eq 0 ] && [ "$N" -eq 3638 ] || { echo "ERROR: expected 3,638 runs with a file" >&2; exit 1; }

sourmash scripts manysketch -p dna,k=31,scaled=1000,noabund -c "$SLURM_CPUS_PER_TASK" -o "$ZIP" "$MANIFEST"
rc=$?
echo "manysketch exit $rc  $(date)"
[ $rc -eq 0 ] || exit $rc
sourmash sig summarize "$ZIP" 2>&1 | tail -8
echo "=== COMPLETE — $(date) ==="
