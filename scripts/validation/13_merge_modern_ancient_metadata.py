#!/usr/bin/env python3
"""
Merge Modern and Ancient Validation Metadata
=============================================

Combines ancient metagenome samples with modern metagenome samples
into a unified validation metadata file.

USAGE:
------
python scripts/validation/13_merge_modern_ancient_metadata.py

Creates: paper/metadata/validation_metadata.tsv
"""

import polars as pl
from pathlib import Path

PROJECT_ROOT = Path("/pasteur/appa/scratch/cduitama/EDID/decOM-classify")

def main():
    # Load ancient samples (current validation metadata)
    ancient_df = pl.read_csv(
        PROJECT_ROOT / "paper/metadata/validation_metadata.tsv",
        separator='\t'
    )
    print(f"📚 Loaded {len(ancient_df)} ancient metagenome samples")
    print(f"   Columns: {ancient_df.columns}")
    
    # Load modern samples (newly expanded)
    modern_df = pl.read_csv(
        PROJECT_ROOT / "data/validation/modern_samples_expanded_full.tsv",
        separator='\t'
    )
    print(f"\n📚 Loaded {len(modern_df)} modern metagenome samples")
    print(f"   Columns: {modern_df.columns}")
    
    # Ensure column order matches
    # ancient_df has: archive_accession, sample_name, sample_type, sample_source, sample_host, 
    #                 material, community_type, geo_loc_name, site_name, latitude, longitude, 
    #                 sample_age, sample_age_doi, project_name, publication_year, publication_doi, 
    #                 run_accession, fastq_ftp, fastq_bytes, fastq_md5
    
    # Reorder modern_df to match
    modern_df = modern_df.select(ancient_df.columns)
    
    # Cast data types to match ancient_df schema
    # Convert string lat/lon to float, handling "unknown" values
    modern_df = modern_df.with_columns([
        pl.when(pl.col('latitude').str.strip_chars() == '')
          .then(None)
          .when(pl.col('latitude') == 'unknown')
          .then(None)
          .otherwise(pl.col('latitude').cast(pl.Float64, strict=False))
          .alias('latitude'),
        pl.when(pl.col('longitude').str.strip_chars() == '')
          .then(None)
          .when(pl.col('longitude') == 'unknown')
          .then(None)
          .otherwise(pl.col('longitude').cast(pl.Float64, strict=False))
          .alias('longitude'),
        pl.col('sample_age').cast(pl.Int64, strict=False)
    ])
    
    # Concatenate
    combined_df = pl.concat([ancient_df, modern_df], how='vertical_relaxed')
    
    print(f"\n📊 Combined metadata:")
    print(f"   Total samples: {len(combined_df)}")
    print(f"   Ancient: {len(ancient_df)} ({len(ancient_df)*100//len(combined_df)}%)")
    print(f"   Modern: {len(modern_df)} ({len(modern_df)*100//len(combined_df)}%)")
    
    # Check sample_type distribution
    print(f"\n   Sample type distribution:")
    print(combined_df['sample_type'].value_counts())
    
    # Check sample_source distribution
    print(f"\n   Sample source distribution:")
    print(combined_df['sample_source'].value_counts())
    
    # Check for duplicates
    n_duplicates = combined_df['run_accession'].n_unique() - len(combined_df)
    if n_duplicates != 0:
        print(f"\n⚠️  WARNING: {-n_duplicates} duplicate run accessions found!")
        dupes = combined_df.filter(pl.col('run_accession').is_duplicated())
        print(f"   Duplicates: {dupes['run_accession'].to_list()[:10]}")
    else:
        print(f"\n✅ No duplicate run accessions")
    
    # Save
    output_file = PROJECT_ROOT / "paper/metadata/validation_metadata.tsv"
    combined_df.write_csv(output_file, separator='\t')
    print(f"\n✅ Saved combined metadata to {output_file}")
    print(f"   {len(combined_df)} total samples")

if __name__ == '__main__':
    main()
