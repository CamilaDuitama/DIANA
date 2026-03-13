#!/usr/bin/env python3
"""
Merge the 149 reviewed modern samples into validation_metadata.tsv
"""

import polars as pl
from pathlib import Path

print("="*100)
print("MERGING REVIEWED MODERN SAMPLES INTO VALIDATION METADATA")
print("="*100)

# 1. Load interactive review log
review_log = pl.read_csv('data/validation/interactive_review_log.tsv', separator='\t')
print(f"\n1. Loaded review log: {len(review_log)} samples reviewed")

# 2. Filter for approved and corrected samples only
approved = review_log.filter(
    (pl.col('decision') == 'APPROVED') | 
    (pl.col('decision') == 'CORRECTED')
)
print(f"   Approved/Corrected: {len(approved)} samples")
print(f"   Excluded: {len(review_log.filter(pl.col('decision') == 'EXCLUDED'))} samples")

# 3. Get list of accessions to keep
approved_accessions = set(approved['run_accession'].to_list())

# 4. Load current validation metadata
val_meta = pl.read_csv('paper/metadata/validation_metadata.tsv', separator='\t')
print(f"\n2. Current validation metadata: {len(val_meta)} samples")

# 5. Remove old modern metagenome samples that were reviewed
old_modern_accessions = set(val_meta.filter(pl.col('sample_type') == 'modern_metagenome')['Run_accession'].to_list())
samples_to_remove = old_modern_accessions & approved_accessions
samples_to_keep_ancient = val_meta.filter(
    (pl.col('sample_type') != 'modern_metagenome') | 
    (~pl.col('Run_accession').is_in(list(approved_accessions)))
)

print(f"\n3. Removing {len(samples_to_remove)} old modern samples that were re-reviewed")
print(f"   Keeping: {len(samples_to_keep_ancient)} samples (ancient + old modern not reviewed)")

# 6. Load SRA metadata for all reviewed samples
existing_sra = pl.read_csv('data/validation/existing_41_sra_metadata.tsv', separator='\t')
new_sra = pl.read_csv('data/validation/modern_sra_metadata_full.tsv', separator='\t', infer_schema_length=10000)

print(f"\n4. Loaded SRA metadata:")
print(f"   Existing: {len(existing_sra)} samples")
print(f"   New: {len(new_sra)} samples")

# 7. Combine SRA metadata
# Standardize column names
existing_sra_std = existing_sra.select([
    'run_accession',
    'organism',
    'isolation_source',
    'env_material',
    'host',
    'library_strategy',
    'library_source',
    'all_attributes'
])

new_sra_std = new_sra.select([
    'run_accession',
    'organism',
    'isolation_source',
    'env_material',
    'host',
    'library_strategy',
    'library_source',
    'all_attributes',
    'geo_loc_name',
    'sample_accession',
    'collection_date'
])

# Add missing columns to existing_sra
existing_sra_std = existing_sra_std.with_columns([
    pl.lit(None, dtype=pl.Utf8).alias('geo_loc_name'),
    pl.lit(None, dtype=pl.Utf8).alias('sample_accession'),
    pl.lit(None, dtype=pl.Utf8).alias('collection_date')
])

# Combine
all_sra = pl.concat([existing_sra_std, new_sra_std])

# 8. Join approved samples with SRA metadata
approved_with_meta = approved.join(all_sra, on='run_accession', how='left')

print(f"\n5. Joined approved samples with SRA metadata: {len(approved_with_meta)} samples")

# 9. Standardize labels (lowercase, plaque variants)
print(f"\n6. Standardizing labels...")

approved_with_meta = approved_with_meta.with_columns([
    # Standardize plaque variants
    pl.when(pl.col('material').str.to_lowercase().str.contains('plaque'))
    .then(pl.lit('plaque'))
    .otherwise(pl.col('material').str.to_lowercase())
    .alias('material'),
    
    # Lowercase all labels
    pl.col('sample_host').str.to_lowercase().alias('sample_host'),
    pl.col('community_type').str.to_lowercase().alias('community_type')
])

# 10. Create new rows matching validation_metadata.tsv structure
# Get a template row to see all columns
template_cols = val_meta.columns

print(f"\n7. Creating new rows with {len(template_cols)} columns...")

# Create new metadata rows (all columns as String except publication_year as Float64)
new_rows = approved_with_meta.select([
    pl.lit(None, dtype=pl.Utf8).alias('archive_accession'),
    pl.col('run_accession').alias('Run_accession'),
    pl.lit(None, dtype=pl.Utf8).alias('True_label'),
    pl.lit('modern_metagenome').alias('sample_type'),
    pl.lit('completed').alias('status'),
    pl.col('sample_accession').alias('sample_name'),
    pl.lit(None, dtype=pl.Utf8).alias('project_name'),
    pl.lit(None, dtype=pl.Float64).alias('publication_year'),
    pl.col('geo_loc_name'),
    pl.col('material'),
    pl.col('sample_host'),
    pl.col('community_type'),
    pl.lit(None, dtype=pl.Utf8).alias('total_runs_for_sample'),
    pl.lit(None, dtype=pl.Utf8).alias('available_runs_for_sample'),
    pl.lit(None, dtype=pl.Utf8).alias('unavailable_runs_for_sample'),
    pl.lit(None, dtype=pl.Utf8).alias('unitig_size_gb'),
    pl.lit('Not applicable', dtype=pl.Utf8).alias('Assay Type'),
    pl.lit(None, dtype=pl.Utf8).alias('BioProject'),
    pl.lit(None, dtype=pl.Utf8).alias('BioSample'),
    pl.lit(None, dtype=pl.Utf8).alias('Center Name'),
    pl.lit(None, dtype=pl.Utf8).alias('Consent'),
    pl.lit(None, dtype=pl.Utf8).alias('DATASTORE filetype'),
    pl.lit(None, dtype=pl.Utf8).alias('DATASTORE provider'),
    pl.lit(None, dtype=pl.Utf8).alias('DATASTORE region'),
    pl.lit(None, dtype=pl.Utf8).alias('Experiment'),
    pl.lit(None, dtype=pl.Utf8).alias('Instrument'),
    pl.lit(None, dtype=pl.Utf8).alias('LibraryLayout'),
    pl.lit(None, dtype=pl.Utf8).alias('LibrarySelection'),
    pl.lit(None, dtype=pl.Utf8).alias('LibrarySource'),
    pl.col('organism').alias('Organism'),
    pl.lit(None, dtype=pl.Utf8).alias('Platform'),
    pl.lit(None, dtype=pl.Utf8).alias('ReleaseDate'),
    pl.lit(None, dtype=pl.Utf8).alias('SRA Study'),
    pl.lit(None, dtype=pl.Utf8).alias('Type'),
    pl.col('isolation_source').alias('Isolation source'),
    pl.lit(None, dtype=pl.Utf8).alias('Avg_read_len'),
    pl.lit(None, dtype=pl.Utf8).alias('Avg_num_reads'),
    pl.lit(None, dtype=pl.Utf8).alias('seqstats_contigs_n50'),
    pl.lit(None, dtype=pl.Utf8).alias('seqstats_contigs_nbseq'),
    pl.lit(None, dtype=pl.Utf8).alias('seqstats_contigs_maxlen'),
    pl.lit(None, dtype=pl.Utf8).alias('seqstats_contigs_sumlen'),
    pl.lit(None, dtype=pl.Utf8).alias('seqstats_unitigs_n50'),
    pl.lit(None, dtype=pl.Utf8).alias('seqstats_unitigs_maxlen'),
    pl.lit(None, dtype=pl.Utf8).alias('seqstats_unitigs_sumlen'),
    pl.lit(None, dtype=pl.Utf8).alias('size_contigs_after_compression'),
    pl.lit(None, dtype=pl.Utf8).alias('size_contigs_before_compression'),
    pl.lit(None, dtype=pl.Utf8).alias('size_unitigs_before_compression'),
    pl.lit(None, dtype=pl.Utf8).alias('size_unitigs_after_compression')
])

# 11. Merge with existing validation metadata
final_metadata = pl.concat([samples_to_keep_ancient, new_rows])

print(f"\n8. Final validation metadata: {len(final_metadata)} samples")
print(f"   Ancient: {len(final_metadata.filter(pl.col('sample_type') != 'modern_metagenome'))}")
print(f"   Modern: {len(final_metadata.filter(pl.col('sample_type') == 'modern_metagenome'))}")

# 12. Create backup
print(f"\n9. Creating backup...")
val_meta.write_csv('paper/metadata/validation_metadata_before_merge.tsv.backup', separator='\t')
print(f"   Backup saved: paper/metadata/validation_metadata_before_merge.tsv.backup")

# 13. Save new metadata
final_metadata.write_csv('paper/metadata/validation_metadata.tsv', separator='\t')
print(f"\n10. ✅ Saved new validation metadata: paper/metadata/validation_metadata.tsv")

# 14. Summary
print("\n" + "="*100)
print("SUMMARY - NEW MODERN METAGENOME COMPOSITION")
print("="*100)

modern_final = final_metadata.filter(pl.col('sample_type') == 'modern_metagenome')

# Material
print("\nMATERIAL:")
mat_dist = modern_final.group_by('material').agg(pl.len().alias('count')).sort('count', descending=True)
for row in mat_dist.iter_rows(named=True):
    print(f"  {row['material']:30s}: {row['count']:3d}")

# Host
print("\nSAMPLE_HOST:")
host_dist = modern_final.group_by('sample_host').agg(pl.len().alias('count')).sort('count', descending=True)
for row in host_dist.iter_rows(named=True):
    print(f"  {row['sample_host']:40s}: {row['count']:3d}")

# Community
print("\nCOMMUNITY_TYPE:")
comm_dist = modern_final.group_by('community_type').agg(pl.len().alias('count')).sort('count', descending=True)
for row in comm_dist.iter_rows(named=True):
    print(f"  {row['community_type']:40s}: {row['count']:3d}")

print("\n✅ Merge complete!")
