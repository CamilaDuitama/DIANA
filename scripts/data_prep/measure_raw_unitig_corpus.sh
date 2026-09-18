#!/bin/bash
# How much sequence is in the per-sample unitig files, and what would embedding it cost?
# Measured on a size-stratified sample rather than the whole 1.5 TB corpus.
set -e
cd /pasteur/helix/scratch/cduitama/EDID/decOM-classify
command -v seqkit >/dev/null || module load SeqKit/2.8.2
mapfile -t FILES < <(ls -S data/unitigs/*.unitigs.fa.gz)
N=${#FILES[@]}
echo "corpus: $N files"
printf "%-32s %12s %14s %10s\n" file gzip_MB bases n_seqs
TOTAL_B=0; TOTAL_G=0
for frac in 2 10 25 50 75 90 98; do          # percentiles by size, largest first
  i=$(( N * frac / 100 )); f="${FILES[$i]}"
  mb=$(( $(stat -c%s "$f") / 1000000 ))
  read -r n b <<< "$(seqkit stats -T "$f" 2>/dev/null | awk 'NR==2{print $4, $5}')"
  printf "%-32s %12d %14d %10d\n" "$(basename "$f" .unitigs.fa.gz)" "$mb" "${b:-0}" "${n:-0}"
  # bases per gzipped MB, for extrapolation
  if [ "${mb:-0}" -gt 0 ] && [ "${b:-0}" -gt 0 ]; then
    TOTAL_B=$(( TOTAL_B + b )); TOTAL_G=$(( TOTAL_G + mb ))
  fi
done
echo
echo "bases per gzipped MB (from the sampled files): $(( TOTAL_B / TOTAL_G ))"
echo "corpus is 1502 GB gzipped -> estimated total bases: $(( 1502 * 1000 * TOTAL_B / TOTAL_G ))"
