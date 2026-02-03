#!/usr/bin/env python3
"""
Analyze what labels we can extract from modern sample metadata.
"""

import polars as pl
from pathlib import Path

PROJECT_ROOT = Path("/pasteur/appa/scratch/cduitama/EDID/decOM-classify")

# Load MGnify metadata with categories
mgnify_df = pl.read_csv(PROJECT_ROOT / "data/validation/modern_samples_balanced_v2.tsv", separator='\t')

# Load SRA sample metadata
sra_df = pl.read_csv(PROJECT_ROOT / "data/validation/modern_sra_metadata_sample.tsv", separator='\t')

print("="*80)
print("MGNIFY CATEGORIES (what we queried for)")
print("="*80)
print(mgnify_df.group_by('category').agg(pl.count('run_accession').alias('count')).sort('category'))

print("\n" + "="*80)
print("SRA ORGANISMS (from sample metadata)")
print("="*80)
print(sra_df.group_by('organism').agg(pl.count('run_accession').alias('count')))

print("\n" + "="*80)
print("SRA ISOLATION SOURCES (first 10 samples)")
print("="*80)
print(sra_df.select(['run_accession', 'organism', 'isolation_source', 'all_attributes']))

print("\n" + "="*80)
print("LABEL MAPPING PROPOSAL")
print("="*80)
print("\nWe have MGnify 'category' for all 109 samples:")
print("  - oral (42 samples)")
print("  - soil (39 samples)")
print("  - skin (25 samples)")
print("  - sediment (3 samples)")

print("\nBut we DON'T have direct columns for:")
print("  ❌ Material (need to map from category + isolation_source + attributes)")
print("  ❌ Community Type (need to infer from material)")
print("  ⚠️  Sample Host (have 'organism' but only for 10 samples checked)")

print("\n" + "="*80)
print("QUESTION FOR USER:")
print("="*80)
print("Should we:")
print("  A) Map MGnify category → training labels using rules")
print("     (e.g., oral→Oral, soil→soil, skin→Skin, sediment→sediment)")
print("  B) Fetch FULL metadata for all 109 samples to see actual SRA fields")
print("  C) Use ENA API to get better metadata (sample_description, etc.)")
print("\nWithout full metadata, we're making assumptions based on the query category.")
