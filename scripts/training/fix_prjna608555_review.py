#!/usr/bin/env python3
"""
Fix review log to exclude PRJNA608555 samples (not modern samples!)
"""

import polars as pl
from pathlib import Path

# The 3 samples from PRJNA608555 that need to be excluded
prjna_samples = ['SRR11176630', 'SRR11176634', 'SRR11176636']

review_log_path = Path('data/training/label_review_log.tsv')

if not review_log_path.exists():
    print(f"ERROR: Review log not found: {review_log_path}")
    exit(1)

# Load review log
print(f"Loading review log from {review_log_path}")
review_log = pl.read_csv(review_log_path, separator='\t')
print(f"Total samples in log: {len(review_log)}")

# Check current status of PRJNA608555 samples
current_status = review_log.filter(pl.col('run_accession').is_in(prjna_samples))
print(f"\nCurrent status of PRJNA608555 samples:")
print(current_status.select(['run_accession', 'decision', 'reason']))

# Update the decision for these samples
print(f"\nUpdating {len(prjna_samples)} samples to EXCLUDED...")

# Create updated review log
review_log = review_log.with_columns([
    pl.when(pl.col('run_accession').is_in(prjna_samples))
    .then(pl.lit('EXCLUDED'))
    .otherwise(pl.col('decision'))
    .alias('decision'),
    
    pl.when(pl.col('run_accession').is_in(prjna_samples))
    .then(pl.lit(None))
    .otherwise(pl.col('corrected_material'))
    .alias('corrected_material'),
    
    pl.when(pl.col('run_accession').is_in(prjna_samples))
    .then(pl.lit(None))
    .otherwise(pl.col('corrected_host'))
    .alias('corrected_host'),
    
    pl.when(pl.col('run_accession').is_in(prjna_samples))
    .then(pl.lit(None))
    .otherwise(pl.col('corrected_community'))
    .alias('corrected_community'),
    
    pl.when(pl.col('run_accession').is_in(prjna_samples))
    .then(pl.lit('From PRJNA608555 - not a modern sample (ancient dental calculus)'))
    .otherwise(pl.col('reason'))
    .alias('reason')
])

# Save backup
backup_path = review_log_path.parent / f'{review_log_path.stem}_backup.tsv'
print(f"\nCreating backup: {backup_path}")
original = pl.read_csv(review_log_path, separator='\t')
original.write_csv(backup_path, separator='\t')

# Save updated log
print(f"Saving updated review log to {review_log_path}")
review_log.write_csv(review_log_path, separator='\t')

# Verify changes
updated_status = review_log.filter(pl.col('run_accession').is_in(prjna_samples))
print(f"\nUpdated status of PRJNA608555 samples:")
print(updated_status.select(['run_accession', 'decision', 'reason']))

print("\n✓ Done!")
print(f"\nBackup saved to: {backup_path}")
print(f"Updated review log: {review_log_path}")
print(f"\nNext step: Run apply_label_corrections.py to update training metadata")
