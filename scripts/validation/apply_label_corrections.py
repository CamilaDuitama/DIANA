#!/usr/bin/env python3
"""
Apply label corrections to validation_metadata.tsv based on interactive review
"""

import polars as pl

# Load current validation metadata
val_meta = pl.read_csv('paper/metadata/validation_metadata.tsv', separator='\t')

print(f"Current validation metadata: {len(val_meta)} samples")

# 1. Standardize plaque labels to lowercase "plaque"
print("\n1. Standardizing plaque labels...")
val_meta = val_meta.with_columns([
    pl.when(pl.col('material').str.contains('supragingival plaque|Supragingival plaque', literal=False))
    .then(pl.lit('plaque'))
    .when(pl.col('material').str.contains('subgingival plaque|Subgingival plaque', literal=False))
    .then(pl.lit('plaque'))
    .otherwise(pl.col('material'))
    .alias('material')
])

# 2. Convert all label columns to lowercase
print("2. Converting all labels to lowercase...")
val_meta = val_meta.with_columns([
    pl.col('material').str.to_lowercase().alias('material'),
    pl.col('sample_host').str.to_lowercase().alias('sample_host'),
    pl.col('community_type').str.to_lowercase().alias('community_type')
])

# 3. Fix ERR4605147 sample_host to Homo sapiens (from interactive review)
print("3. Fixing ERR4605147 sample_host...")
val_meta = val_meta.with_columns([
    pl.when(pl.col('Run_accession') == 'ERR4605147')
    .then(pl.lit('homo sapiens'))
    .otherwise(pl.col('sample_host'))
    .alias('sample_host')
])

# 4. Fix all skin metagenome samples (should be Homo sapiens host)
print("4. Fixing skin metagenome samples (setting host to homo sapiens)...")
val_meta = val_meta.with_columns([
    pl.when((pl.col('material') == 'skin') & (pl.col('sample_host') != 'homo sapiens'))
    .then(pl.lit('homo sapiens'))
    .otherwise(pl.col('sample_host'))
    .alias('sample_host')
])

# Save backup
print("\n5. Creating backup...")
val_meta_original = pl.read_csv('paper/metadata/validation_metadata.tsv', separator='\t')
val_meta_original.write_csv('paper/metadata/validation_metadata.tsv.backup', separator='\t')
print("   Backup saved: paper/metadata/validation_metadata.tsv.backup")

# Save updated metadata
print("6. Saving updated metadata...")
val_meta.write_csv('paper/metadata/validation_metadata.tsv', separator='\t')
print("   ✅ Updated: paper/metadata/validation_metadata.tsv")

# Summary of changes
print("\n" + "="*100)
print("SUMMARY OF CHANGES")
print("="*100)

# Check plaque
plaque_count = len(val_meta.filter(pl.col('material') == 'plaque'))
print(f"\n✅ Plaque samples (standardized to lowercase 'plaque'): {plaque_count}")

# Check skin
skin_samples = val_meta.filter(pl.col('material') == 'skin')
print(f"\n✅ Skin samples: {len(skin_samples)}")
skin_homo_sapiens = len(skin_samples.filter(pl.col('sample_host') == 'homo sapiens'))
print(f"   - With host='homo sapiens': {skin_homo_sapiens}/{len(skin_samples)}")

# Check ERR4605147
err_sample = val_meta.filter(pl.col('Run_accession') == 'ERR4605147')
if len(err_sample) > 0:
    print(f"\n✅ ERR4605147:")
    for row in err_sample.select(['Run_accession', 'material', 'sample_host', 'community_type']).iter_rows(named=True):
        print(f"   material: {row['material']}")
        print(f"   sample_host: {row['sample_host']}")
        print(f"   community_type: {row['community_type']}")

# Material distribution
print("\n" + "="*100)
print("MATERIAL DISTRIBUTION (after corrections)")
print("="*100)
mat_dist = val_meta.filter(pl.col('sample_type') == 'modern_metagenome').group_by('material').agg(pl.len().alias('count')).sort('count', descending=True)
for row in mat_dist.iter_rows(named=True):
    print(f"  {row['material']:30s}: {row['count']:3d}")

print("\n✅ All corrections applied!")
