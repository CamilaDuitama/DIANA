#!/usr/bin/env python3
"""
Remove train/test overlap from expanded validation metadata
"""

import polars as pl
from pathlib import Path

PROJECT_ROOT = Path("/pasteur/appa/scratch/cduitama/EDID/decOM-classify")

print("="*80)
print("REMOVING TRAIN/TEST OVERLAP FROM VALIDATION")
print("="*80)

# Load expanded validation (RAW - with potential overlaps)
val_raw = pl.read_csv(
    PROJECT_ROOT / "data" / "validation" / "validation_metadata_expanded_RAW.tsv",
    separator='\t'
)

print(f"\n1. Expanded validation (before overlap removal): {len(val_raw)} rows")
print(f"   - With run accessions: {val_raw.filter(pl.col('run_accession') != '').shape[0]}")
print(f"   - Without run accessions: {val_raw.filter(pl.col('run_accession') == '').shape[0]}")

# Load train/test
train = pl.read_csv(PROJECT_ROOT / "data" / "splits" / "train_metadata.tsv", separator='\t')
test = pl.read_csv(PROJECT_ROOT / "data" / "splits" / "test_metadata.tsv", separator='\t')

train_runs = set(train['Run_accession'].to_list())
test_runs = set(test['Run_accession'].to_list())
all_train_test_runs = train_runs | test_runs

print(f"\n2. Train/test run accessions: {len(all_train_test_runs)}")
print(f"   - Train: {len(train_runs)}")
print(f"   - Test: {len(test_runs)}")

# Remove overlaps
val_filtered = val_raw.filter(~pl.col('run_accession').is_in(list(all_train_test_runs)))

print(f"\n3. After overlap removal: {len(val_filtered)} rows")
print(f"   - Removed: {len(val_raw) - len(val_filtered)} overlapping samples")

# Check sample_source distribution
print(f"\n4. Sample source distribution (after overlap removal):")
source_counts = val_filtered.group_by('sample_source').len().sort('len', descending=True)
for row in source_counts.iter_rows(named=True):
    print(f"   {row['sample_source']}: {row['len']}")

# Check community_type distribution
print(f"\n5. Community type distribution:")
comm_counts = val_filtered.group_by('community_type').len().sort('len', descending=True)
for row in comm_counts.iter_rows(named=True):
    print(f"   {row['community_type']}: {row['len']}")

# Material distribution
print(f"\n6. Material distribution (top 15):")
material_counts = val_filtered.group_by('material').len().sort('len', descending=True)
for row in material_counts.head(15).iter_rows(named=True):
    print(f"   {row['material']}: {row['len']}")

# Save filtered validation
output_file = PROJECT_ROOT / "data" / "validation" / "validation_metadata_expanded.tsv"
val_filtered.write_csv(output_file, separator='\t')
print(f"\n✓ Saved to {output_file}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Total validation samples: {len(val_filtered)}")
print(f"  Host-associated: {val_filtered.filter(pl.col('sample_source') == 'host_associated').shape[0]}")
print(f"  Environmental: {val_filtered.filter(pl.col('sample_source') == 'environmental').shape[0]}")
print(f"\nNEW environmental samples: {val_filtered.filter(pl.col('sample_source') == 'environmental').shape[0]}")
print(f"Previous host-only validation: 616 samples")
print(f"TOTAL increase: {len(val_filtered) - 616} samples")
print("="*80)
