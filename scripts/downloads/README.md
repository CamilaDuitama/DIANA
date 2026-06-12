# Download Scripts for DIANA v7 Splits

This directory contains scripts to download missing data files for the v7 training/test/validation splits.

## Overview

Based on the v7 label distribution report:
- **Unitig files needed**: 903 files (2,888 train + 502 test, with 3,112 already available)
- **FASTQ files needed**: 379 files for validation set (504 total, with 125 already available)

## Quick Start

### 1. Download Missing Unitig Files from Logan

```bash
# Step 1: Identify missing files and generate download list
python scripts/downloads/01_download_missing_unitigs.py

# Step 2: Download from Logan S3 (choose one method)

# Option A: Sequential download (slow but simple)
bash scripts/downloads/download_unitigs.sh

# Option B: Parallel download with GNU parallel (faster, requires 'parallel' command)
parallel -j 8 --bar 'aws s3 cp s3://logan-pub/u/{}/{}.unitigs.fa.zst data/unitigs/ --no-sign-request' \
  :::: scripts/downloads/missing_unitigs.txt

# Option C: SLURM array job (recommended for cluster)
N=$(wc -l < scripts/downloads/missing_unitigs.txt)
sbatch --array=1-$N%50 scripts/downloads/02_download_unitigs.sbatch

# Step 3: Convert zst files to gzip format
find data/unitigs -name "*.zst" > scripts/downloads/unitigs_to_convert.txt
N=$(wc -l < scripts/downloads/unitigs_to_convert.txt)
sbatch --array=1-$N%50 scripts/downloads/03_convert_unitigs_to_gzip.sbatch
```

### 2. Download Missing Validation FASTQ Files

```bash
# Step 1: Identify missing files and generate download list
python scripts/downloads/04_download_validation_fastq.py

# Step 2: Download and convert SRA → FASTQ (SLURM array job recommended)
N=$(wc -l < scripts/downloads/missing_validation_accessions.txt)
sbatch --array=1-$N%20 scripts/downloads/05_download_validation_fastq.sbatch
```

## Scripts Description

### Unitig Downloads (Logan S3)

| Script | Purpose | Usage |
|--------|---------|-------|
| `01_download_missing_unitigs.py` | Identify missing unitigs by comparing v7 splits against existing files | `python scripts/downloads/01_download_missing_unitigs.py` |
| `02_download_unitigs.sbatch` | SLURM array job to download .zst files from Logan S3 | `sbatch --array=1-N%50 scripts/downloads/02_download_unitigs.sbatch` |
| `03_convert_unitigs_to_gzip.sbatch` | Convert .zst files to .fa.gz format (removes .zst after successful conversion) | `sbatch --array=1-N%50 scripts/downloads/03_convert_unitigs_to_gzip.sbatch` |

**Source**: [Logan Unitigs](https://github.com/IndexThePlanet/Logan/blob/main/Unitigs.md)  
**Format**: Files are downloaded as `.unitigs.fa.zst` (zstandard compressed), then converted to `.unitigs.fa.gz` (gzip)  
**URL pattern**: `s3://logan-pub/u/{accession}/{accession}.unitigs.fa.zst`

### Validation FASTQ Downloads (NCBI SRA)

| Script | Purpose | Usage |
|--------|---------|-------|
| `04_download_validation_fastq.py` | Identify missing validation FASTQ files | `python scripts/downloads/04_download_validation_fastq.py` |
| `05_download_validation_fastq.sbatch` | Download SRA → convert to FASTQ → compress with pigz | `sbatch --array=1-N%20 scripts/downloads/05_download_validation_fastq.sbatch` |

**Pipeline**: `prefetch` (SRA) → `fasterq-dump` (FASTQ) → `pigz` (compress) → cleanup SRA  
**Output**: `data/validation/raw/{accession}/*.fastq.gz`

## Resource Requirements

### Unitig Downloads
- **Partition**: edid
- **CPU**: 2 cores
- **Memory**: 4 GB
- **Time**: 1 hour per file
- **Concurrency**: 50 parallel jobs recommended

### Unitig Conversion (zst → gzip)
- **Partition**: edid
- **CPU**: 4 cores (for parallel compression)
- **Memory**: 8 GB
- **Time**: 2 hours per file
- **Concurrency**: 50 parallel jobs recommended

### FASTQ Downloads
- **Partition**: edid
- **CPU**: 8 cores (for fasterq-dump and pigz)
- **Memory**: 16 GB
- **Time**: 4 hours per accession
- **Concurrency**: 20 parallel jobs recommended

## Monitoring Progress

```bash
# Check running jobs
squeue -u $USER

# Check job efficiency after completion
reportseff <job_id>

# Count downloaded files
find data/unitigs -name "*.gz" | wc -l
find data/validation/raw -name "*.fastq.gz" | wc -l

# Check logs for errors
grep -l "✗" logs/downloads/*.err
```

## Expected Output Structure

```
data/
├── unitigs/
│   ├── {accession}.unitigs.fa.gz     # Train/test unitigs (gzip format)
│   └── ...
└── validation/
    ├── sra/                           # Temporary SRA files (auto-cleaned)
    └── raw/
        ├── {accession}/
        │   ├── {accession}_1.fastq.gz # Paired-end read 1
        │   ├── {accession}_2.fastq.gz # Paired-end read 2 (if paired)
        │   └── ...
        └── ...
```

## Troubleshooting

### AWS CLI not found
```bash
# Install AWS CLI (no account needed for public buckets)
conda install -c conda-forge awscli
```

### SRA toolkit not found
```bash
# Install SRA toolkit
conda install -c bioconda sra-tools
```

### Parallel compression not available
```bash
# Install pigz (parallel gzip)
conda install -c conda-forge pigz
```

### Download failures
- Check internet connectivity
- Verify accession exists in Logan/SRA
- Check logs in `logs/downloads/` for specific errors
- Re-run the same SLURM command (scripts skip existing files automatically)

## Notes

- **All scripts skip existing files** - safe to re-run after failures
- **Automatic cleanup** - SRA files are removed after successful FASTQ conversion
- **Space requirements**: ~2-3 TB for all unitigs + validation FASTQ files
- **Network**: Logan S3 is public (no AWS account needed)
- **Compression**: zstd → gzip conversion uses parallel compression (4 cores by default)

---
**Last Updated**: June 2026
