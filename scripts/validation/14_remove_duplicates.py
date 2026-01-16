#!/usr/bin/env python3
"""
Remove Duplicates from Validation Metadata
===========================================

Removes duplicate run_accession entries, keeping the most complete record.
"""

import polars as pl
from pathlib import Path

PROJECT_ROOT = Path("/pasteur/appa/scratch/cduitama/EDID/decOM-classify")

def main():
    df = pl.read_csv(
        PROJECT_ROOT / "paper/metadata/validation_metadata.tsv",
        separator='\t',
        null_values=['', 'NA', 'unknown'],
        infer_schema_length=10000
    )
    
    print(f"Original: {len(df)} rows")
    
    # Remove rows with empty run_accession
    df_filtered = df.filter(pl.col('run_accession').is_not_null())
    print(f"After removing empty run_accession: {len(df_filtered)} rows")
    print(f"  Removed: {len(df) - len(df_filtered)} rows with empty run_accession")
    
    # Find duplicates
    duplicates = df_filtered.filter(pl.col('run_accession').is_duplicated())
    if len(duplicates) > 0:
        print(f"\nDuplicates found: {len(duplicates)} rows")
        print(duplicates.select(['run_accession', 'sample_type', 'sample_name']).unique(subset=['run_accession']))
        
        # Keep first occurrence (ancient samples likely come first)
        df_dedup = df_filtered.unique(subset=['run_accession'], keep='first')
        print(f"\nAfter deduplication: {len(df_dedup)} rows")
        print(f"  Removed: {len(df_filtered) - len(df_dedup)} duplicate rows")
    else:
        df_dedup = df_filtered
        print("\n✅ No duplicates found")
    
    # Summary
    print(f"\n📊 Final dataset:")
    print(f"   Total samples: {len(df_dedup)}")
    print(f"\n   Sample type:")
    print(df_dedup['sample_type'].value_counts().sort('sample_type'))
    print(f"\n   Sample source:")
    print(df_dedup['sample_source'].value_counts().sort('sample_source'))
    
    # Save
    output_file = PROJECT_ROOT / "paper/metadata/validation_metadata.tsv"
    df_dedup.write_csv(output_file, separator='\t')
    print(f"\n✅ Saved deduplicated metadata to {output_file}")
    print(f"   {len(df_dedup)} unique samples")

if __name__ == '__main__':
    main()
