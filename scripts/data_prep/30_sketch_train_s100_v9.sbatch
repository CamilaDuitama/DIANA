#!/bin/bash
#SBATCH --partition=edid
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --job-name=diana_sketch_s100
#SBATCH --output=logs/sketches_v9/sketch_s100_%j.out
#SBATCH --error=logs/sketches_v9/sketch_s100_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
################################################################################
# S8.6 (Experiment A): the 2,716 TRAINING runs re-sketched ten times finer than
# S8.0 (scaled=100 keeps 1 k-mer in 100), with k=21 and k=31 in one pass so a
# shorter-k probe needs no second sketch. The vocabulary (unitigs.fa) is sketched
# with the same parameters so "off-list" is exact under the same cut.
# Training runs only: this is a description of what the training data can offer
# each class; held-out is never read here.
################################################################################
set -o pipefail
for f in /opt/gensoft/adm/Modules/5.6.1/init/bash /opt/gensoft/adm/Modules/init/bash; do
    [ -f "$f" ] && { source "$f"; break; }
done
module load sourmash/4.9.4 || { echo "ERROR: cannot load sourmash/4.9.4" >&2; exit 1; }
OUT=data/sketches_v9
ZIP=$OUT/unitigs_train_v9_s100.zip
MANIFEST=$OUT/manifest_train_v9.csv
[ -e "$ZIP" ] && [ "$(stat -c %s "$ZIP")" -gt 1000 ] && { echo "refusing to overwrite $ZIP" >&2; exit 1; }
rm -f "$ZIP"
echo "=== S8.6 sketch  job $SLURM_JOB_ID  $(hostname)  $(date) ==="
echo "name,genome_filename,protein_filename" > "$MANIFEST"
while read -r acc; do f=data/unitigs/$acc.unitigs.fa.gz; [ -s "$f" ] && echo "$acc,$f," >> "$MANIFEST"; done < data/splits_v9/train_accessions.txt
N=$(($(wc -l < "$MANIFEST") - 1)); echo "manifest: $N training runs"
[ "$N" -eq 2716 ] || { echo "ERROR: expected 2,716" >&2; exit 1; }
sourmash scripts manysketch -p dna,k=21,k=31,scaled=100,noabund -c "$SLURM_CPUS_PER_TASK" -o "$ZIP" "$MANIFEST" || exit 1
sourmash sketch dna -p k=21,k=31,scaled=100,noabund --name vocab_v9 -o "$OUT/vocab_v9_s100.sig.zip" \
    data/matrices/matrix_v9_train/unitigs.fa || exit 1
sourmash sig summarize "$ZIP" 2>&1 | tail -4
echo "=== COMPLETE — $(date) ==="
