#!/bin/bash
#SBATCH --partition=edid
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --job-name=diana_offlist_hashes
#SBATCH --output=logs/sketches_v9/offlist_%j.out
#SBATCH --error=logs/sketches_v9/offlist_%j.err
#SBATCH --chdir=/pasteur/helix/scratch/cduitama/EDID/decOM-classify
################################################################################
# S8.1: sketch the vocabulary with the same parameters as the samples, then
# count, on the 2,716 training sketches only, which off-list hashes recur across
# samples and studies (scripts/analysis/42_offlist_shared_hashes.py).
# Submit with --dependency=afterok:<S8.0 job>. Memory: up to ~1e9 (hash, sample)
# entries as uint64 + int32, sorted once.
################################################################################
set -o pipefail
for f in /opt/gensoft/adm/Modules/5.6.1/init/bash /opt/gensoft/adm/Modules/init/bash; do
    [ -f "$f" ] && { source "$f"; break; }
done
module load sourmash/4.9.4 || { echo "ERROR: cannot load sourmash/4.9.4" >&2; exit 1; }
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_INIT_AT_FORK=FALSE KMP_AFFINITY=disabled MPLBACKEND=Agg

SK=data/sketches_v9
[ -s "$SK/unitigs_split_v9.zip" ] || { echo "ERROR: S8.0 output missing" >&2; exit 1; }
echo "=== S8.1  job $SLURM_JOB_ID  $(hostname)  $(date) ==="
if [ ! -s "$SK/vocab_v9.sig.zip" ]; then
    sourmash sketch dna -p k=31,scaled=1000,noabund --name vocab_v9 \
        -o "$SK/vocab_v9.sig.zip" data/matrices/matrix_v9_train/unitigs.fa || exit 1
fi
./env/bin/python scripts/analysis/42_offlist_shared_hashes.py
rc=$?
echo "=== exit $rc — $(date) ==="
exit $rc
