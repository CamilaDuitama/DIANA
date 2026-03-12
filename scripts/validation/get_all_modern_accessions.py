#!/usr/bin/env python3
"""
Get SRA metadata for ALL modern samples in validation set:
1. Existing modern samples in validation_metadata.tsv
2. New 109 samples from MGnify query

This will help us properly label all modern samples consistently.
"""

import polars as pl
from pathlib import Path

PROJECT_ROOT = Path("/pasteur/appa/scratch/cduitama/EDID/decOM-classify")

# Load current validation metadata
validation_df = pl.read_csv(PROJECT_ROOT / "paper/metadata/validation_metadata.tsv", separator='\t')

# Filter for modern samples
modern_existing = validation_df.filter(pl.col('sample_type') == 'modern_metagenome')

print("="*80)
print("CURRENT VALIDATION SET - MODERN SAMPLES")
print("="*80)
print(f"\nTotal modern samples in validation: {len(modern_existing)}")
print(f"\nMaterial distribution:")
print(modern_existing.group_by('material').agg(pl.count('Run_accession').alias('count')).sort('count', descending=True))
print(f"\nSample host distribution:")
print(modern_existing.group_by('sample_host').agg(pl.count('Run_accession').alias('count')).sort('count', descending=True))
print(f"\nCommunity type distribution:")
print(modern_existing.group_by('community_type').agg(pl.count('Run_accession').alias('count')).sort('count', descending=True))

# Get accessions
existing_accessions = modern_existing['Run_accession'].to_list()

# Load new proposed samples
new_accessions_file = PROJECT_ROOT / "data/validation/modern_accessions_balanced_v2.txt"
with open(new_accessions_file) as f:
    new_accessions = [line.strip() for line in f if line.strip()]

print("\n" + "="*80)
print("NEW SAMPLES FROM MGNIFY QUERY")
print("="*80)
print(f"\nNew samples to add: {len(new_accessions)}")

# Find overlap
existing_set = set(existing_accessions)
new_set = set(new_accessions)
overlap = existing_set & new_set

print(f"\nOverlap check:")
print(f"  Samples already in validation: {len(overlap)}")
if overlap:
    print(f"  Overlapping accessions: {list(overlap)[:5]}...")

# Create combined list
all_modern_accessions = list(existing_set | new_set)
print(f"\n" + "="*80)
print("COMBINED MODERN SAMPLE SET")
print("="*80)
print(f"\nTotal unique modern samples: {len(all_modern_accessions)}")
print(f"  - Currently in validation: {len(existing_accessions)}")
print(f"  - New from MGnify: {len(new_accessions)}")
print(f"  - Overlap: {len(overlap)}")
print(f"  - Net new: {len(new_set - existing_set)}")

# Save combined accessions list for metadata fetching
output_file = PROJECT_ROOT / "data/validation/all_modern_accessions_for_labeling.txt"
with open(output_file, 'w') as f:
    f.write('\n'.join(sorted(all_modern_accessions)))

print(f"\n💾 Saved {len(all_modern_accessions)} accessions to:")
print(f"   {output_file}")

print("\n" + "="*80)
print("NEXT STEP:")
print("="*80)
print(f"""
Run SRA metadata fetch for ALL {len(all_modern_accessions)} modern samples:

  mamba run -p ./env python scripts/validation/fetch_all_modern_metadata.py

This will take ~{len(all_modern_accessions) * 0.4 / 60:.0f} minutes with rate limiting.
""")
