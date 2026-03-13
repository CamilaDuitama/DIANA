#!/usr/bin/env python3
"""
Map modern samples to training labels using SRA metadata + MGnify categories.
"""

import polars as pl
from pathlib import Path
import re

PROJECT_ROOT = Path("/pasteur/appa/scratch/cduitama/EDID/decOM-classify")

# Load training metadata to see available labels
train_df = pl.read_csv(PROJECT_ROOT / "paper/metadata/train_metadata.tsv", separator='\t')

print("="*80)
print("TRAINING SET LABELS (for reference)")
print("="*80)
print("\nMaterial classes:")
print(train_df.group_by('material').agg(pl.count('Run_accession').alias('count')).sort('count', descending=True))
print("\nSample Host classes:")
print(train_df.group_by('sample_host').agg(pl.count('Run_accession').alias('count')).sort('count', descending=True))
print("\nCommunity Type classes:")
print(train_df.group_by('community_type').agg(pl.count('Run_accession').alias('count')).sort('count', descending=True))

# Load modern metadata
sra_df = pl.read_csv(PROJECT_ROOT / "data/validation/modern_sra_metadata_full.tsv", separator='\t', infer_schema_length=10000)
mgnify_df = pl.read_csv(PROJECT_ROOT / "data/validation/modern_samples_balanced_v2.tsv", separator='\t')

# Merge MGnify category with SRA metadata
merged_df = sra_df.join(
    mgnify_df.select(['run_accession', 'category']),
    on='run_accession',
    how='left'
)

print("\n" + "="*80)
print("LABEL MAPPING FOR 109 MODERN SAMPLES")
print("="*80)

# Helper function to extract from attributes
def extract_attribute(attrs_str, key):
    """Extract value from 'key: value' pairs in attributes string."""
    if not attrs_str or attrs_str == 'null':
        return None
    pattern = rf'{key}:\s*([^|]+)'
    match = re.search(pattern, attrs_str, re.IGNORECASE)
    return match.group(1).strip() if match else None

# Add extracted fields
merged_df = merged_df.with_columns([
    pl.col('all_attributes').map_elements(lambda x: extract_attribute(x, 'tissue'), return_dtype=pl.Utf8).alias('tissue'),
    pl.col('all_attributes').map_elements(lambda x: extract_attribute(x, 'isolate'), return_dtype=pl.Utf8).alias('isolate'),
])

print("\n1. MATERIAL MAPPING:")
print("-" * 80)

# Show what data we have for material mapping
print("\nMGnify categories:")
print(merged_df.group_by('category').agg(pl.count('run_accession').alias('count')))

print("\nIsolation sources (top 10):")
iso_sources = merged_df.filter(pl.col('isolation_source').is_not_null()).group_by('isolation_source').agg(pl.count('run_accession').alias('count')).sort('count', descending=True).head(10)
print(iso_sources)

print("\nTissue types (from attributes):")
tissues = merged_df.filter(pl.col('tissue').is_not_null()).group_by('tissue').agg(pl.count('run_accession').alias('count')).sort('count', descending=True)
print(tissues)

print("\nIsolate types (from attributes):")
isolates = merged_df.filter(pl.col('isolate').is_not_null()).group_by('isolate').agg(pl.count('run_accession').alias('count')).sort('count', descending=True)
print(isolates)

print("\nEnv Material (when available):")
env_mat = merged_df.filter(pl.col('env_material').is_not_null()).group_by('env_material').agg(pl.count('run_accession').alias('count')).sort('count', descending=True)
print(env_mat)

print("\n" + "="*80)
print("\n2. SAMPLE HOST MAPPING:")
print("-" * 80)
print("\nOrganisms:")
print(merged_df.group_by('organism').agg(pl.count('run_accession').alias('count')).sort('count', descending=True))

print("\nHost field (when available):")
hosts = merged_df.filter(pl.col('host').is_not_null()).group_by('host').agg(pl.count('run_accession').alias('count')).sort('count', descending=True)
print(hosts)

print("\n" + "="*80)
print("\n3. DETAILED SAMPLE BREAKDOWN BY CATEGORY:")
print("-" * 80)

for cat in ['oral', 'skin', 'soil', 'sediment']:
    subset = merged_df.filter(pl.col('category') == cat)
    print(f"\n{cat.upper()} ({len(subset)} samples):")
    print(f"  Organisms: {subset.group_by('organism').agg(pl.count('run_accession').alias('count')).to_dicts()}")
    print(f"  Isolation sources: {subset.filter(pl.col('isolation_source').is_not_null()).group_by('isolation_source').agg(pl.count('run_accession').alias('count')).to_dicts()}")
    print(f"  Tissues: {subset.filter(pl.col('tissue').is_not_null()).group_by('tissue').agg(pl.count('run_accession').alias('count')).to_dicts()}")
    print(f"  Isolates: {subset.filter(pl.col('isolate').is_not_null()).group_by('isolate').agg(pl.count('run_accession').alias('count')).to_dicts()}")

print("\n" + "="*80)
print("RECOMMENDATION:")
print("="*80)
print("""
Based on the metadata, we should:

1. Create a mapping function that uses:
   - MGnify category as primary indicator
   - isolation_source for validation
   - tissue/isolate attributes for refinement
   
2. Manual review needed for:
   - Samples where category doesn't match isolation_source
   - Ambiguous materials (e.g., "G_DNA_Stool" in oral category)
   
3. Save mapped metadata for review before download
""")

# Save merged metadata for inspection
output_file = PROJECT_ROOT / "data/validation/modern_metadata_for_mapping.tsv"
merged_df.write_csv(output_file, separator='\t')
print(f"\n💾 Saved merged metadata to: {output_file}")
print("   Review this file to verify label assignments before proceeding.")
