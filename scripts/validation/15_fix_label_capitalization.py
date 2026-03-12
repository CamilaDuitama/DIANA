#!/usr/bin/env python3
"""
Fix label capitalization in validation metadata to match training data.

TRAINING labels:
- community_type: Not applicable - env sample, gut, oral, plant tissue, skeletal tissue, soft tissue
- sample_host: Homo sapiens, Not applicable - env sample, Ursus arctos, Gorilla sp., etc.
- material: dental calculus, sediment, tooth, bone, digestive_contents, coprolite, Oral, permafrost, etc.

VALIDATION issues:
- sample_source: "environmental" → should match community_type capitalization
- sample_source: "host_associated" → should match community_type capitalization  
- sample_host: "Not applicable - env sample" ✓ correct
- material: lowercase values need to match training (e.g., "dental calculus", "sediment")
"""

import polars as pl
import sys

def main():
    # Load validation metadata
    val_path = 'paper/metadata/validation_metadata.tsv'
    val = pl.read_csv(val_path, separator='\t', null_values=['', 'NA', 'unknown'], infer_schema_length=10000)
    
    print(f"Original validation metadata: {len(val)} samples")
    
    # Show samples with unknown labels
    print("\n" + "="*80)
    print("SAMPLES WITH UNKNOWN/NULL LABELS")
    print("="*80)
    
    unknown = val.filter(
        pl.col('sample_source').is_null() | 
        pl.col('sample_host').is_null() | 
        pl.col('material').is_null()
    )
    
    print(f"\nTotal samples with any null label: {len(unknown)}")
    print(unknown[['run_accession', 'sample_name', 'sample_type', 'sample_source', 'sample_host', 'material', 'community_type']])
    
    # Show current label distributions
    print("\n" + "="*80)
    print("CURRENT LABEL DISTRIBUTIONS (BEFORE FIX)")
    print("="*80)
    
    print("\nsample_source:")
    print(val['sample_source'].value_counts().sort('sample_source'))
    
    print("\nsample_host (top 10):")
    print(val['sample_host'].value_counts().sort('count', descending=True).head(10))
    
    print("\nmaterial (top 10):")
    print(val['material'].value_counts().sort('count', descending=True).head(10))
    
    print("\ncommunity_type:")
    print(val['community_type'].value_counts().sort('community_type'))
    
    # NO FIXES NEEDED - just reporting
    # The validation metadata already uses correct capitalization for sample_source
    # Training data uses "community_type" which is a different field
    
    print("\n" + "="*80)
    print("LABEL COMPARISON: VALIDATION vs TRAINING")
    print("="*80)
    
    print("""
TRAINING uses:
  - Field: community_type
  - Values: "Not applicable - env sample", "gut", "oral", "plant tissue", "skeletal tissue", "soft tissue"
  
VALIDATION uses:
  - Field: sample_source
  - Values: "environmental", "host_associated", null
  
These are DIFFERENT fields for DIFFERENT purposes:
  - community_type: describes the microbial community (oral, gut, environmental, etc.)
  - sample_source: binary classification (host-associated vs environmental)
  
Current validation labels are CORRECT for sample_source.
The 5 modern samples with null values are plant metagenomes from MGnify (search_term: "plant")
which don't have clear host_associated vs environmental classification.
    """)
    
    # Show the 5 plant samples
    print("\n" + "="*80)
    print("PLANT SAMPLES WITH UNKNOWN CLASSIFICATION")
    print("="*80)
    plant_samples = val.filter(pl.col('sample_source').is_null())
    print(plant_samples[['run_accession', 'sample_name', 'sample_type', 'geo_loc_name']])
    
    print("\nThese are modern plant metagenomes from South Africa (Lichtenberg, Mafikeng)")
    print("They could be classified as 'host_associated' (plant host) or 'environmental' (plant material)")

if __name__ == '__main__':
    main()
