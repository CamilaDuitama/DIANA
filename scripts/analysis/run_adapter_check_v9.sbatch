#!/bin/bash
#SBATCH --partition=edid
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --job-name=diana_adapter_v9
#SBATCH --output=logs/adapter_v9/%x_%j.out
#SBATCH --error=logs/adapter_v9/%x_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
# Adapter-contamination check on the v9 unitig vocabulary (110,202 unitigs):
# the 39 curated Illumina adapters (data/adapters_v9, from the paper branch)
# aligned with Bowtie 2 in end-to-end and local mode. Replaces the v7-era
# check cited in the manuscript (Methods TODO).
set -eo pipefail
module load bowtie2
cd results/adapter_check_v9
bowtie2 --version | head -1
bowtie2-build --threads 8 -q ../../data/matrices/matrix_v9_train/unitigs.fa unitigs_v9
for mode in end-to-end local; do
  bowtie2 -f -x unitigs_v9 -U ../../data/adapters_v9/adapters.fasta \
      --$mode --no-unal -p 8 -S adapters_${mode}.sam 2> adapters_${mode}.stats
  echo "== $mode =="; cat adapters_${mode}.stats
  grep -v '^@' adapters_${mode}.sam | awk '{print $1, $3, $6}' | sort -u > adapters_${mode}_hits.txt || true
  echo "aligned adapters ($mode): $(cut -d' ' -f1 adapters_${mode}_hits.txt | sort -u | wc -l)"
done
rm -f unitigs_v9*.bt2
echo DONE
