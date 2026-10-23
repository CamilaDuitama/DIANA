#!/bin/bash
#SBATCH --partition=edid
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --job-name=diana_blast_v9
#SBATCH --output=logs/blast_v9/blast_%j.out
#SBATCH --error=logs/blast_v9/blast_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
# BLAST of the 568 attributed v9 unitigs never searched (results/attribution_v9/unsearched_unitigs_v9.fasta)
# against the cluster's nt (module blast+, cluster bank index). Same spirit as the Feb 2026 search:
# megablast, top hits with taxonomy columns.
set -o pipefail
module load blast+/2.17.0
DB=/local/databases/index/blast+/nt/nt
blastn -task megablast -db "$DB" -query results/blast_v9/unsearched_unitigs_v9_safeid.fasta \
  -evalue 1e-5 -max_target_seqs 5 -num_threads "$SLURM_CPUS_PER_TASK" \
  -outfmt "6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore staxids sscinames stitle" \
  > results/blast_v9/unsearched_unitigs_v9_blast_safeid.tsv
rc=$?
echo "hits: $(cut -f1 results/blast_v9/unsearched_unitigs_v9_blast_safeid.tsv | sort -u | wc -l) of $(grep -c '>' results/attribution_v9/unsearched_unitigs_v9.fasta) queries with at least one hit"
exit $rc
