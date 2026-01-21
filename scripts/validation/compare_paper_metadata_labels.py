#!/usr/bin/env python3
"""
Compare labels across train/test/validation in paper/metadata/ directory
for all 4 classification tasks
"""

import polars as pl
from pathlib import Path

PROJECT_ROOT = Path("/pasteur/appa/scratch/cduitama/EDID/decOM-classify")

# Load paper metadata
train = pl.read_csv(PROJECT_ROOT / "paper" / "metadata" / "train_metadata.tsv", separator='\t')
test = pl.read_csv(PROJECT_ROOT / "paper" / "metadata" / "test_metadata.tsv", separator='\t')
val = pl.read_csv(PROJECT_ROOT / "paper" / "metadata" / "validation_metadata.tsv", separator='\t')

print("="*80)
print("COLUMN COMPARISON")
print("="*80)
print(f"\nTrain columns ({len(train.columns)}): {', '.join(train.columns[:15])}...")
print(f"Test columns ({len(test.columns)}):  {', '.join(test.columns[:15])}...")
print(f"Val columns ({len(val.columns)}):   {', '.join(val.columns)}")

print("\n" + "="*80)
print("CLASSIFICATION TASK COLUMNS")
print("="*80)

tasks = ['sample_type', 'community_type', 'sample_host', 'material']

for task in tasks:
    train_has = task in train.columns
    test_has = task in test.columns
    val_has = task in val.columns
    print(f"{task:20} Train: {'✓' if train_has else '✗'}  Test: {'✓' if test_has else '✗'}  Val: {'✓' if val_has else '✗'}")

print("\n" + "="*80)
print("LABEL DISTRIBUTION BY TASK")
print("="*80)

for task in tasks:
    if task not in val.columns:
        print(f"\n⚠️  {task.upper()} - NOT IN VALIDATION METADATA")
        continue
        
    print(f"\n{'='*80}")
    print(f"TASK: {task.upper()}")
    print(f"{'='*80}")
    
    # Get unique labels from each dataset
    train_labels = set(train[task].unique().to_list())
    test_labels = set(test[task].unique().to_list())
    val_labels = set(val[task].unique().to_list())
    
    # All unique labels across all datasets
    all_labels = train_labels | test_labels | val_labels
    
    print(f"\nTotal unique labels:")
    print(f"  Training:   {len(train_labels)}")
    print(f"  Test:       {len(test_labels)}")
    print(f"  Validation: {len(val_labels)}")
    print(f"  Combined:   {len(all_labels)}")
    
    # Count UNSEEN labels in validation
    unseen_in_val = val_labels - train_labels
    print(f"\n🚨 UNSEEN in validation (not in training): {len(unseen_in_val)}")
    
    if len(unseen_in_val) > 0:
        print("\nUNSEEN LABELS:")
        for label in sorted(unseen_in_val):
            count = val.filter(pl.col(task) == label).shape[0]
            print(f"  - {label} ({count} samples)")
    
    # Create comparison table
    print(f"\n{'Label':<40} {'Train':<10} {'Test':<10} {'Val':<10} {'Status':<20}")
    print("-" * 90)
    
    for label in sorted(all_labels):
        train_count = train.filter(pl.col(task) == label).shape[0] if label in train_labels else 0
        test_count = test.filter(pl.col(task) == label).shape[0] if label in test_labels else 0
        val_count = val.filter(pl.col(task) == label).shape[0] if label in val_labels else 0
        
        # Determine status
        in_train = label in train_labels
        in_test = label in test_labels
        in_val = label in val_labels
        
        if in_train and in_test and in_val:
            status = "✅ All datasets"
        elif in_train and in_test:
            status = "⚠️  Train+Test only"
        elif in_train and in_val:
            status = "⚠️  Train+Val only"
        elif in_val and not in_train:
            status = "🚨 UNSEEN by model"
        elif in_test and not in_train:
            status = "🚨 Test error"
        else:
            status = "?"
        
        train_str = str(train_count) if in_train else "-"
        test_str = str(test_count) if in_test else "-"
        val_str = str(val_count) if in_val else "-"
        
        print(f"{label:<40} {train_str:<10} {test_str:<10} {val_str:<10} {status:<20}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("✅ All datasets      - Label in train, test, and validation → OK")
print("⚠️  Train+Test only  - Label in train/test but not validation → OK")
print("🚨 UNSEEN by model  - Label ONLY in validation → MODEL CANNOT PREDICT!")
print("="*80)
