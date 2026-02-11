#!/usr/bin/env python3
"""
Apply label corrections from interactive review to training metadata.

Reads the review log and:
1. Updates corrected labels
2. Removes excluded samples
3. Creates new training metadata with corrections applied
4. Generates before/after comparison report
"""

import polars as pl
from pathlib import Path
import shutil
from datetime import datetime


def main():
    """Apply corrections from review log to training metadata"""
    
    # Paths
    review_log_path = Path('data/training/label_review_log.tsv')
    train_meta_path = Path('paper/metadata/train_metadata.tsv')
    output_dir = Path('data/training/corrected_metadata')
    
    # Check if review log exists
    if not review_log_path.exists():
        print(f"ERROR: Review log not found: {review_log_path}")
        print(f"Please run review_modern_training_labels.py first")
        return
    
    # Load review log
    print("Loading review log...")
    review_log = pl.read_csv(review_log_path, separator='\t')
    print(f"✓ Loaded {len(review_log)} reviewed samples")
    
    # Summary of decisions
    print("\nReview Summary:")
    for decision in ['APPROVED', 'CORRECTED', 'EXCLUDED']:
        count = len(review_log.filter(pl.col('decision') == decision))
        print(f"  {decision}: {count}")
    
    # Load training metadata
    print("\nLoading training metadata...")
    train_meta = pl.read_csv(train_meta_path, separator='\t')
    print(f"✓ Loaded {len(train_meta)} training samples")
    
    # Create backup
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = output_dir / f'train_metadata_backup_{timestamp}.tsv'
    shutil.copy(train_meta_path, backup_path)
    print(f"✓ Created backup: {backup_path}")
    
    # Apply corrections
    print("\nApplying corrections...")
    
    corrected_samples = review_log.filter(pl.col('decision') == 'CORRECTED')
    excluded_samples = review_log.filter(pl.col('decision') == 'EXCLUDED')
    
    # Track changes
    changes = []
    
    # Apply label corrections
    if len(corrected_samples) > 0:
        print(f"\nCorrecting {len(corrected_samples)} samples:")
        
        for row in corrected_samples.iter_rows(named=True):
            acc = row['run_accession']
            old_mat = row['original_material']
            new_mat = row['corrected_material']
            old_host = row['original_host']
            new_host = row['corrected_host']
            old_comm = row['original_community']
            new_comm = row['corrected_community']
            reason = row['reason']
            
            # Update in metadata
            mask = train_meta['Run_accession'] == acc
            
            if new_mat != old_mat:
                train_meta = train_meta.with_columns(
                    pl.when(pl.col('Run_accession') == acc)
                    .then(pl.lit(new_mat))
                    .otherwise(pl.col('material'))
                    .alias('material')
                )
                changes.append(f"  {acc}: material {old_mat} → {new_mat} ({reason})")
            
            if new_host != old_host:
                train_meta = train_meta.with_columns(
                    pl.when(pl.col('Run_accession') == acc)
                    .then(pl.lit(new_host))
                    .otherwise(pl.col('sample_host'))
                    .alias('sample_host')
                )
                changes.append(f"  {acc}: sample_host {old_host} → {new_host} ({reason})")
            
            if new_comm != old_comm:
                train_meta = train_meta.with_columns(
                    pl.when(pl.col('Run_accession') == acc)
                    .then(pl.lit(new_comm))
                    .otherwise(pl.col('community_type'))
                    .alias('community_type')
                )
                changes.append(f"  {acc}: community_type {old_comm} → {new_comm} ({reason})")
        
        for change in changes:
            print(change)
    
    # Remove excluded samples
    if len(excluded_samples) > 0:
        print(f"\nExcluding {len(excluded_samples)} samples:")
        
        excluded_accs = excluded_samples['run_accession'].to_list()
        for row in excluded_samples.iter_rows(named=True):
            print(f"  {row['run_accession']}: {row['reason']}")
        
        train_meta = train_meta.filter(~pl.col('Run_accession').is_in(excluded_accs))
    
    # Save corrected metadata
    corrected_path = output_dir / 'train_metadata_corrected.tsv'
    train_meta.write_csv(corrected_path, separator='\t')
    print(f"\n✓ Saved corrected metadata: {corrected_path}")
    
    original_count = len(pl.read_csv(train_meta_path, separator='\t'))
    print(f"  Original samples: {original_count}")
    print(f"  Corrected samples: {len(train_meta)}")
    print(f"  Removed: {len(excluded_samples)}")
    
    # Generate comparison report
    print("\nGenerating comparison report...")
    
    original_meta = pl.read_csv(train_meta_path, separator='\t')
    modern_orig = original_meta.filter(pl.col('sample_type') == 'modern_metagenome')
    modern_corr = train_meta.filter(pl.col('sample_type') == 'modern_metagenome')
    
    report = []
    report.append("="*80)
    report.append("TRAINING METADATA CORRECTION REPORT")
    report.append("="*80)
    report.append(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"\nReview log: {review_log_path}")
    report.append(f"Original metadata: {train_meta_path}")
    report.append(f"Corrected metadata: {corrected_path}")
    report.append(f"Backup: {backup_path}")
    
    report.append(f"\n\nSAMPLE COUNTS:")
    report.append(f"  Total original: {len(original_meta)}")
    report.append(f"  Total corrected: {len(train_meta)}")
    report.append(f"  Samples removed: {len(excluded_samples)}")
    
    report.append(f"\n\nMODERN SAMPLES:")
    report.append(f"  Original: {len(modern_orig)}")
    report.append(f"  Corrected: {len(modern_corr)}")
    
    report.append(f"\n\nMATERIAL DISTRIBUTION (Modern samples only):")
    report.append(f"\nORIGINAL:")
    mat_orig = modern_orig.group_by('material').agg(pl.len()).sort('len', descending=True)
    for row in mat_orig.iter_rows(named=True):
        report.append(f"  {row['material']}: {row['len']}")
    
    report.append(f"\nCORRECTED:")
    mat_corr = modern_corr.group_by('material').agg(pl.len()).sort('len', descending=True)
    for row in mat_corr.iter_rows(named=True):
        report.append(f"  {row['material']}: {row['len']}")
    
    report.append(f"\n\nCHANGES APPLIED:")
    if changes:
        for change in changes:
            report.append(change)
    else:
        report.append("  No label corrections (only exclusions)")
    
    report.append(f"\n\nEXCLUDED SAMPLES:")
    if len(excluded_samples) > 0:
        for row in excluded_samples.iter_rows(named=True):
            report.append(f"  {row['run_accession']}: {row['reason']}")
    else:
        report.append("  None")
    
    report.append("\n" + "="*80)
    
    # Save report
    report_path = output_dir / f'correction_report_{timestamp}.txt'
    report_path.write_text('\n'.join(report))
    print(f"✓ Saved report: {report_path}")
    
    # Print report to console
    print("\n" + '\n'.join(report))
    
    print(f"\n\n{'='*80}")
    print("NEXT STEPS:")
    print("="*80)
    print(f"\n1. Review the corrected metadata: {corrected_path}")
    print(f"2. Review the report: {report_path}")
    print(f"3. If satisfied, replace the original:")
    print(f"   cp {corrected_path} {train_meta_path}")
    print(f"\n4. Update train_ids.txt if samples were excluded:")
    print(f"   (Remove {len(excluded_samples)} run accessions from data/splits/train_ids.txt)")
    print(f"\n5. Consider retraining the model with corrected labels")
    print(f"\nBackup available at: {backup_path}")
    print("="*80)


if __name__ == '__main__':
    main()
