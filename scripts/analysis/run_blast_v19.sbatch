#!/bin/bash
#SBATCH --partition=edid,common
#SBATCH --exclude=maestro-2004,maestro-2008,maestro-2012,maestro-2016,maestro-2018,maestro-2022
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --array=0-39
#SBATCH --job-name=diana_blast_v19
#SBATCH --output=logs/blast_v19/blast_%a_%j.out
#SBATCH --error=logs/blast_v19/blast_%a_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
# BLAST of the whole v17 vocabulary (85,986 unitigs, 40 chunks) against ONE pinned nt release
# (nt of 2026-10-05; results/blast_v19/database_info.txt), so every hit comes from the same
# database version (Camila, 2026-10-09). Same settings as the earlier searches: megablast,
# e-value 1e-5, top 5 hits, taxonomy columns.
set -euo pipefail
module load blast+/2.17.0
DB=/pasteur/helix/services/banques-prod/prod/rel/nt/nt_2026-10-5/blast+/2.10.0/nt
CHUNK=$(printf "%02d" "$SLURM_ARRAY_TASK_ID")
IN=results/blast_v19/chunks/chunk_${CHUNK}.fasta
OUT=results/blast_v19/chunks/chunk_${CHUNK}_blast.tsv
[ -s "$OUT" ] && { echo "$OUT exists; refusing to overwrite" >&2; exit 1; }
blastn -task megablast -db "$DB" -query "$IN" \
  -evalue 1e-5 -max_target_seqs 5 -num_threads "$SLURM_CPUS_PER_TASK" \
  -outfmt "6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore staxids sscinames stitle" \
  > "$OUT.tmp" && mv "$OUT.tmp" "$OUT"
echo "chunk $CHUNK: $(cut -f1 "$OUT" | sort -u | wc -l) of $(grep -c '>' "$IN") queries with at least one hit"
