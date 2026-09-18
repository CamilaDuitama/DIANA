#!/bin/bash
#SBATCH --partition=edid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --array=0-3
#SBATCH --job-name=diana_dnaappend
#SBATCH --output=logs/dna_append/%x_%a_%j.out
#SBATCH --error=logs/dna_append/%x_%a_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
################################################################################
# S6: keep the 110,202 fractions and APPEND a DNABERT-2 summary.
#
# The design asked for from the start, and the one no arm has tested. S4 and S5 both
# replaced the input layer and both lost (S5 by 0.208 to 0.359 on held-out), while their
# scrambled controls showed the sequence content contributed nothing either way, so the
# loss came from discarding the 110,202 private per-unitig vectors rather than from the
# sequence. Appending keeps every one of them.
#
# Tasks 0,1: max and mean over the real descriptions.
# Tasks 2,3: the same two over descriptions of sequences whose bases were permuted within
#            each unitig. These are NOT optional. The pre-registered condition for reporting
#            any clearing cell is that the same cell does not clear in the scrambled arm, and
#            max-versus-mean cannot supply that: it varies the pooling while holding the DNA
#            fixed, which is a different question.
#
# CPU only, the DNABERT-2 table is already built. No --time.
################################################################################
set -e
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg
# 0,1 real; 2,3 the mandatory scrambled controls
case "$SLURM_ARRAY_TASK_ID" in
  0) SUMMARY=max;  TABLE=real     ;;
  1) SUMMARY=mean; TABLE=real     ;;
  2) SUMMARY=max;  TABLE=shuffled ;;
  3) SUMMARY=mean; TABLE=shuffled ;;
  *) echo "ERROR: unexpected array index" >&2; exit 1 ;;
esac
mkdir -p logs/dna_append
echo "=== S6 append DNABERT-2 summary [$SUMMARY / $TABLE]  $(date) ==="
./env/bin/python scripts/data_prep/19_append_dnabert_summary.py --summary "$SUMMARY" --table "$TABLE"
echo "=== COMPLETE — $(date) ==="
