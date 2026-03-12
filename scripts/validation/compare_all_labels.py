#!/usr/bin/env python3
"""
Create comprehensive label comparison table across train/test/validation
for all 4 tasks
"""

import polars as pl
from pathlib import Path
import sys

PROJECT_ROOT = Path("/pasteur/appa/scratch/cduitama/EDID/decOM-classify")

# Load datasets
train = pl.read_csv(PROJECT_ROOT / "data" / "splits" / "train_metadata.tsv", separator='\t')
test = pl.read_csv(PROJECT_ROOT / "data" / "splits" / "test_metadata.tsv", separator='\t')
val = pl.read_csv(PROJECT_ROOT / "data" / "validation" / "validation_metadata_expanded.tsv", separator='\t')

# Tasks
tasks = {
    'sample_type': 'sample_type',
    'community_type': 'community_type', 
    'sample_host': 'sample_host',
    'material': 'material'
}

print("="*80)
print("LABEL COMPARISON: TRAIN vs TEST vs VALIDATION")
print("="*80)

for task_name, col_name in tasks.items():
    print(f"\n{'='*80}")
    print(f"TASK: {task_name.upper()}")
    print(f"{'='*80}")
    
    # Get unique labels from each dataset
    train_labels = set(train[col_name].unique().to_list())
    test_labels = set(test[col_name].unique().to_list())
    val_labels = set(val[col_name].unique().to_list())
    
    # All unique labels across all datasets
    all_labels = train_labels | test_labels | val_labels
    
    print(f"\nTotal unique labels:")
    print(f"  Training:   {len(train_labels)}")
    print(f"  Test:       {len(test_labels)}")
    print(f"  Validation: {len(val_labels)}")
    print(f"  Combined:   {len(all_labels)}")
    
    # Create comparison table
    print(f"\n{'Label':<40} {'Train':<10} {'Test':<10} {'Validation':<10} {'Status':<20}")
    print("-" * 90)
    
    for label in sorted(all_labels):
        train_count = train.filter(pl.col(col_name) == label).shape[0] if label in train_labels else 0
        test_count = test.filter(pl.col(col_name) == label).shape[0] if label in test_labels else 0
        val_count = val.filter(pl.col(col_name) == label).shape[0] if label in val_labels else 0
        
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
            status = "🚨 Val ONLY (UNSEEN)"
        elif in_test and not in_train:
            status = "🚨 Test ONLY (ERROR)"
        else:
            status = "?"
        
        train_str = str(train_count) if in_train else "-"
        test_str = str(test_count) if in_test else "-"
        val_str = str(val_count) if in_val else "-"
        
        print(f"{label:<40} {train_str:<10} {test_str:<10} {val_str:<10} {status:<20}")

print("\n" + "="*80)
print("LEGEND")
print("="*80)
print("✅ All datasets       - Label present in train, test, and validation")
print("⚠️  Train+Test only   - Label in train/test but NOT in validation")
print("⚠️  Train+Val only    - Label in train/val but NOT in test")
print("🚨 Val ONLY (UNSEEN) - Label ONLY in validation - MODEL CANNOT PREDICT")
print("🚨 Test ONLY (ERROR) - Label in test but not train - SHOULD NOT HAPPEN")
print("="*80)
