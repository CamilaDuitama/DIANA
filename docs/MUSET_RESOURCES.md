# MUSET Unitig Matrix Generation - Resource Usage

## Summary

The unitig fraction matrix for DIANA was generated using MUSET v0.6.1 on 3,070 ancient and modern metagenomic samples. The process took approximately **3.2 days (76.7 hours)** on a high-memory compute node.

## Resource Allocation

| Resource | Value |
|----------|-------|
| **Compute** | 32 CPU cores |
| **Memory** | 500 GB RAM |
| **Walltime** | 76 hours 42 minutes (~3.2 days) |
| **Partition** | seqbio (SLURM) |

## Input/Output

| Metric | Value |
|--------|-------|
| **Input Samples** | 3,070 unitig files (.fa.gz) |
| **Output Matrix** | 1.6 GB (unitigs.frac.mat) |
| **Matrix Dimensions** | 3,070 samples × 107,480 features |
| **Data Format** | Sparse matrix (unitig fractions) |

## Processing Statistics

| Stage | Details |
|-------|---------|
| **K-mer Counting** | 436,456,132,058 total k-mers detected |
| **K-mer Filtering** | 18,953,119 k-mers retained (0.0043%) |
| **Unitig Assembly** | 2,755,241 unitigs assembled |
| **Unitig Filtering** | 107,480 unitigs retained (3.9%) |
| **Retention Criteria** | Present in 10-90% of samples |

## Key Parameters

```bash
muset \
  --file diana_samples.fof \
  --threads 32 \
  --kmer-size 31 \
  --min-abundance 2 \
  --min-unitig-length 61 \
  --min-unitig-fraction 0 \
  --fraction-absent 0.1 \
  --fraction-present 0.1 \
  --out-frac \
  --logan \
  --keep-temp
```

## Timeline

| Timestamp | Event | Elapsed Time |
|-----------|-------|--------------|
| Dec 19, 13:13 | Start: K-mer matrix building | 0h |
| Dec 19, 18:35 | Compute minimizer repartition complete | 5.4h |
| Dec 21, 21:17 | K-mer counting complete, start filtering | 56h |
| Dec 22, 17:41 | K-mer filtering complete | 76.5h |
| Dec 22, 17:43 | Unitig assembly complete (GGCAT) | 76.5h |
| Dec 22, 17:55 | **Matrix generation complete** | **76.7h** |

## Notes

- K-mer counting (kmtricks) was the most time-consuming step (~56 hours)
- Unitig assembly with GGCAT took only ~1 minute due to filtered k-mer set
- Memory usage stayed well within the 500 GB allocation
- No samples failed; all 3,070 samples successfully processed
- One unitig (SRR17604786_6749535) skipped due to missing abundance data

## SLURM Job Configuration

```bash
#SBATCH --cpus-per-task=32
#SBATCH --mem=500G
#SBATCH --partition=seqbio
```

**Job Script**: `scripts/create_umat/02_regenerate_matrix_with_frac.sbatch`

**Log File**: `data/matrices/large_matrix_3070_with_frac/muset_20251219_131306.log`

---
**Generated**: January 20, 2026
