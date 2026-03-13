# Validation Data Processing Pipeline

## Overview
Process validation samples from AncientMetagenomeDir v25.09.0 for DIANA model evaluation.

## Data Source
- **AncientMetagenomeDir version**: v25.09.0 (September 2025)
- **Host-associated samples**: 880 samples from `ancientmetagenome-hostassociated_samples_v25.09.0.tsv`
- **Environmental samples**: 803 samples from `ancientmetagenome-environmental_samples_v25.09.0.tsv`

## Pipeline Steps

### 1. Expand Metadata & Fetch Run Accessions
**Script**: `01_expand_metadata.py`

```bash
./env/bin/python scripts/validation/01_expand_metadata.py \
    --input data/validation/validation_metadata_v25.09.0.tsv \
    --output data/validation/validation_metadata_expanded.tsv \
    --cache data/validation/ena_cache.json
```

**What it does**:
- Expands comma-separated archive accessions (SRS/ERS) to individual rows
- Queries ENA API to get run accessions (SRR/ERR)
- Caches ENA responses to avoid redundant API calls
- Adds columns: `run_accession`, `fastq_ftp`, `fastq_bytes`, `fastq_md5`

**Input**: Curated metadata with archive accessions
**Output**: Expanded metadata with run accessions (one row per run)

---

### 2. Remove Training/Test Overlap
**Script**: Manual filtering with `polars` (see below)

```python
import polars as pl

# Load datasets
val_expanded = pl.read_csv('data/validation/validation_metadata_expanded.tsv', separator='\t')
train_ids = pl.read_csv('data/splits/train_metadata.tsv', separator='\t')
test_ids = pl.read_csv('data/splits/test_metadata.tsv', separator='\t')

# Get train/test run accessions
train_runs = set(train_ids['Run_accession'].to_list())
test_runs = set(test_ids['Run_accession'].to_list())
overlap_runs = train_runs | test_runs

# Filter validation
val_filtered = val_expanded.filter(~pl.col('run_accession').is_in(list(overlap_runs)))

# Save
val_filtered.write_csv('data/validation/validation_metadata_expanded.tsv', separator='\t')
```

**Output**: Validation metadata with overlapping samples removed

---

### 3. Prepare Download List
**Script**: `02_prepare_download.py`

```bash
./env/bin/python scripts/validation/02_prepare_download.py \
    --metadata data/validation/validation_metadata_expanded.tsv \
    --output data/validation/accessions.txt
```

**What it does**:
- Extracts unique run accessions from metadata
- Creates list for SLURM array job (one accession per line)

**Output**: `data/validation/accessions.txt` (one run accession per line)

---

### 4. Download SRA Files (Submit Node)
**Script**: `03_prefetch_all.sh`

```bash
bash scripts/validation/03_prefetch_all.sh
```

**What it does**:
- Downloads .sra files from NCBI SRA to `data/validation/sra/`
- Runs on submit node (requires internet access)
- Uses `prefetch` from sra-tools
- Skips already downloaded files
- Logs failed downloads to `failed_prefetch.txt`

**Requirements**: 
- Submit node with internet access
- sra-tools module loaded

**Output**: `data/validation/sra/{accession}/{accession}.sra`

---

### 5. Convert SRA to FASTQ (Compute Nodes)
**Script**: `04_convert_sra_to_fastq.sbatch`

```bash
# Get number of samples
N_SAMPLES=$(wc -l < data/validation/accessions.txt)

# Submit SLURM array job
sbatch --array=1-${N_SAMPLES}%20 scripts/validation/04_convert_sra_to_fastq.sbatch
```

**What it does**:
- Converts .sra files to FASTQ format
- Uses `fasterq-dump` from sra-tools
- Compresses output with gzip
- Runs on seqbio partition (no internet needed, reads local .sra files)
- Parallel processing with SLURM array (20 tasks at a time)

**Output**: `data/validation/raw/{accession}/*.fastq.gz`

---

### 6. Run DIANA Predictions (Compute Nodes)
**Script**: `05_run_predictions.sbatch`

```bash
# Get number of samples
N_SAMPLES=$(wc -l < data/validation/validation_metadata_expanded.tsv)
N_SAMPLES=$((N_SAMPLES - 1))  # Subtract header

# Submit SLURM array job
sbatch --array=1-${N_SAMPLES}%10 scripts/validation/05_run_predictions.sbatch
```

**What it does**:
- Runs `diana-predict` on each validation sample
- Uses best trained model: `results/full_training/best_model.pth`
- Outputs predictions as JSON files
- 64GB RAM per task (some samples are memory-intensive)

**Output**: `results/validation_predictions/{run_accession}_predictions.json`

---

### 7. Handle OOM Samples (Optional)
**Scripts**: 
- `07_run_oom_samples.sbatch` (128GB)
- `08_run_oom_256gb.sbatch` (256GB)

If some samples fail with out-of-memory errors, rerun with higher memory:

```bash
# Identify failed samples (manual check of logs)
# Create retry list in data/validation/retry_high_mem_tasks.txt

# Retry with 128GB
sbatch scripts/validation/07_run_oom_samples.sbatch

# If still failing, retry with 256GB
sbatch scripts/validation/08_run_oom_256gb.sbatch
```

---

### 8. Compare Predictions to True Labels
**Script**: `06_compare_predictions.py`

```bash
./env/bin/python scripts/validation/06_compare_predictions.py \
    --metadata data/validation/validation_metadata_expanded.tsv \
    --predictions-dir results/validation_predictions \
    --label-encoders results/full_training/label_encoders.pkl \
    --output-dir paper/tables/validation \
    --figures-dir paper/figures/validation
```

**What it does**:
- Loads true labels from metadata
- Loads predictions from JSON files
- Computes accuracy, confusion matrices, calibration metrics
- Generates tables (TSV) and figures (PNG/HTML)
- Analyzes confidence distributions
- Identifies wrong predictions for error analysis

**Output**: 
- Tables in `paper/tables/validation/`
- Figures in `paper/figures/validation/`

---

## Scripts Reference

### Active Scripts (Used)
1. ✅ `01_expand_metadata.py` - ENA API expansion
2. ✅ `02_prepare_download.py` - Create accessions list
3. ✅ `03_prefetch_all.sh` - Download SRA files (submit node)
4. ✅ `04_convert_sra_to_fastq.sbatch` - Convert SRA→FASTQ (compute)
5. ✅ `05_run_predictions.sbatch` - Run DIANA predictions
6. ✅ `06_compare_predictions.py` - Analysis and figures
7. ✅ `07_run_oom_samples.sbatch` - Retry OOM samples (128GB)
8. ✅ `08_run_oom_256gb.sbatch` - Retry OOM samples (256GB)

### Unused Scripts
- ❌ `test_download_submit_node.sh` - Testing script (not part of pipeline)

---

## Current Status (Host-Associated Only)

### Summary
- **Input metadata**: 507 host-associated samples from v25.09.0
- **After ENA expansion**: 880 samples (some archive accessions → multiple runs)
- **After overlap removal**: 616 samples (264 overlapping with train/test removed)
- **Downloaded**: 618 samples (2 extra runs)
- **Predictions**: 608 samples (98.7% coverage, 8 OOM failures)

### Breakdown by Sample Source
- Host-associated: 616 (100%)
- Environmental: 0 (0%)

---

## Notes

- **Internet access**: Only step 4 (prefetch) requires internet, must run on submit node
- **Compute jobs**: Steps 5-6 run on seqbio partition (no time limits)
- **Memory**: Most samples need 64GB, some require 128-256GB
- **ENA cache**: Saves API responses to avoid redundant queries on re-runs
- **Overlap check**: Critical to ensure validation samples are independent from train/test
