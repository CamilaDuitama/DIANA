# Modern Sample Collection - Quick Start Guide

## Problem Summary

Your current validation set has a severe distribution mismatch with training:

| Category | Training Modern | Current Validation | Target Validation |
|----------|----------------|-------------------|-------------------|
| **Total modern** | 199 (7.6%) | 41 (4.5%) ⚠️ | ~140 (7%) ✓ |
| **Skin** | 34.7% | 22.0% | 35% ✓ |
| **Oral** | 33.7% | 2.4% ⚠️ | 34% ✓ |
| **soil** | 28.1% | 63.4% ⚠️ | 28% ✓ |
| **gut** | 0% | 12.2% ⚠️ | 0% ✓ |

## Solution: Improved MGnify Query

### Step 1: Query MGnify with Better Filters

```bash
cd /pasteur/appa/scratch/cduitama/EDID/decOM-classify
mamba run -p ./env python scripts/validation/10_improved_mgnify_query.py
```

**What it does:**
- Searches for ~200 modern samples (accounting for 30% QC failure rate)
- Uses specific search terms:
  - Oral: "saliva metagenome", "oral microbiome", "tongue microbiome"
  - Skin: "skin microbiome", "dermal microbiome", "cutaneous microbiome"
  - soil: "soil metagenome", "agricultural soil", "forest soil"
  - sediment: "marine sediment", "lake sediment"
- **Automatically excludes:**
  - Pooled/multiplexed samples (like MGYS00001980)
  - 16S/amplicon sequencing
  - Rhizosphere/phytometer studies
  - Mock communities

**Output:** `data/validation/modern_samples_mgnify_v2.tsv`

### Step 2: Expand Sample → Run Accessions

```bash
mamba run -p ./env python scripts/validation/11_expand_modern_samples_v2.py
```

**What it does:**
- Queries ENA API for each MGnify sample
- Gets run accessions (SRR/ERR)
- Filters for WGS shotgun metagenomics only
- Excludes amplicon/16S sequencing

**Output:** `data/validation/modern_samples_expanded_v2.tsv`

### Step 3: Balance to Match Training Distribution

```bash
mamba run -p ./env python scripts/validation/12_balance_modern_samples_v2.py
```

**What it does:**
- Selects ~140 samples matching training distribution:
  - Skin: 49 samples (35%)
  - Oral: 47 samples (34%)
  - soil: 39 samples (28%)
  - sediment: 5 samples (3%)
- Ensures diversity across studies (max 10 samples per study)

**Output:** `data/validation/modern_accessions_balanced_v2.txt`

### Step 4: Download and Run Predictions

```bash
# Append to validation accessions
cat data/validation/modern_accessions_balanced_v2.txt >> data/validation/accessions.txt

# Download (if needed)
bash scripts/validation/03_prefetch_all.sh

# Convert to FASTQ (if needed)
sbatch --array=1-1100%20 scripts/validation/04_convert_sra_to_fastq.sbatch

# Run predictions
bash scripts/validation/submit_validation_with_retry.sh

# After predictions complete, update metadata and regenerate figures
mamba run -p ./env python scripts/validation/06_compare_predictions.py
```

## Expected Outcome

**Final validation set:**
- Total: ~1000 samples
- Ancient: ~862 (93%) - from AncientMetagenomeDir
- Modern: ~140 (7%) - from MGnify (matches training 7.6%)

**Modern sample distribution will match training:**
- Skin: 35% (training: 34.7%)
- Oral: 34% (training: 33.7%)
- soil: 28% (training: 28.1%)
- sediment: 3% (training: 3.5%)

## Quality Assurance

After each step, the scripts will:
1. Show distribution summaries
2. Flag excluded studies (pooled/amplicon)
3. Report study diversity metrics
4. Compare with training distribution

**Manual review recommended:**
- Check `data/validation/modern_samples_mgnify_v2.tsv` for study names
- Verify no HMP/pooled studies included
- Ensure WGS shotgun metagenomics only

## Troubleshooting

**If you get too few samples:**
- Increase targets in `10_improved_mgnify_query.py`
- Expand search terms (more specific queries)
- Lower QC filters if necessary

**If you get wrong distribution:**
- Adjust targets in `12_balance_modern_samples_v2.py`
- Check category mapping is correct

**If you still find pooled samples:**
- Add study IDs to exclusion list in `10_improved_mgnify_query.py`
- Query ENA metadata manually for suspicious studies
