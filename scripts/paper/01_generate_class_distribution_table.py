#!/usr/bin/env python3
"""
Generate Class Distribution Table (Supplementary Table 1)

PURPOSE:
    Show sample distribution across all classes for each task and dataset.

INPUTS (from config.py):
    - PATHS['train_metadata']: Training set metadata
    - PATHS['test_metadata']: Test set metadata
    - PATHS['validation_metadata']: Validation metadata with predictions
    - PATHS['predictions_dir']: Validation predictions directory
    - PATHS['label_encoders']: Label encoder mappings

OUTPUTS:
    - paper/tables/final/sup_table_01_class_distribution.tex

PROCESS:
    1. Load all three metadata files
    2. For each task, get all unique classes
    3. Count samples per class in each dataset
    4. Format as LaTeX table with class totals

CONFIGURATION:
    All paths imported from config.py

HARDCODED VALUES:
    - Italicize species names for sample_host task

DEPENDENCIES:
    - pandas, json, pathlib
    - config.py

USAGE:
    python scripts/paper/generate_class_distribution_table.py
    
AUTHOR: Refactored from 06_compare_predictions.py
"""

import sys
import json
from pathlib import Path

import pandas as pd

# Add script directory to path for config import
sys.path.insert(0, str(Path(__file__).parent))
from config import PATHS, TASKS, SAMPLE_TYPE_MAP


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_validation_predictions():
    """Load validation predictions to get true labels."""
    # Load metadata
    metadata = pd.read_csv(PATHS['validation_metadata'], sep='\t')
    
    # Load label encoders
    with open(PATHS['label_encoders']) as f:
        label_encoders = json.load(f)
    
    # Find all prediction files
    predictions_dir = Path(PATHS['predictions_dir'])
    prediction_files = list(predictions_dir.rglob('*_predictions.json'))
    
    # Build records
    records = []
    for pred_file in prediction_files:
        sample_id = pred_file.parent.name
        
        with open(pred_file) as f:
            pred = json.load(f)
        
        sample_meta = metadata[metadata['Run_accession'] == sample_id]
        if len(sample_meta) == 0:
            continue
        
        sample_meta = sample_meta.iloc[0]
        
        for task_name in TASKS:
            true_label = sample_meta[task_name]
            
            # Normalize sample_type labels
            if task_name == 'sample_type' and true_label in SAMPLE_TYPE_MAP:
                true_label = SAMPLE_TYPE_MAP[true_label]
            
            records.append({
                'sample_id': sample_id,
                'task': task_name,
                'true_label': str(true_label) if pd.notna(true_label) else 'Unknown'
            })
    
    return pd.DataFrame(records)


# ============================================================================
# TABLE GENERATION
# ============================================================================

def generate_class_distribution_table(output_dir):
    """Generate supplementary table showing class distribution."""
    print("\n[1/4] Loading metadata...")
    train_meta = pd.read_csv(PATHS['train_metadata'], sep='\t')
    test_meta = pd.read_csv(PATHS['test_metadata'], sep='\t')
    
    # Normalize sample_type
    for df in [train_meta, test_meta]:
        if 'sample_type' in df.columns:
            df['sample_type'] = df['sample_type'].map(SAMPLE_TYPE_MAP).fillna(df['sample_type'])
    
    print(f"  ✓ Train: {len(train_meta)}, Test: {len(test_meta)}")
    
    print("\n[2/4] Loading validation predictions...")
    validation_df = load_validation_predictions()
    val_samples = validation_df['sample_id'].nunique()
    print(f"  ✓ Validation: {val_samples} samples")
    
    print("\n[3/4] Computing class distributions...")
    
    lines = []
    lines.append("\\begin{table*}[!t]")
    lines.append("\\caption{Sample distribution across classes for each dataset\\label{tab:class_distribution}}")
    lines.append("\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}llcccc@{\\extracolsep{\\fill}}}")
    lines.append("\\toprule")
    lines.append("Task & Class & Training & Test & Validation & Total \\\\")
    lines.append("\\midrule")
    
    for task in TASKS:
        task_name = task.replace('_', ' ').title()
        
        # Get all unique classes from all datasets
        train_classes = set(train_meta[task].dropna().unique())
        test_classes = set(test_meta[task].dropna().unique())
        val_task_df = validation_df[validation_df['task'] == task]
        val_classes = set(val_task_df['true_label'].dropna().unique())
        all_classes = sorted(train_classes | test_classes | val_classes)
        
        # Remove 'Unknown' from classes
        all_classes = [c for c in all_classes if str(c) != 'Unknown']
        
        lines.append(f"\\multirow{{{len(all_classes)}}}{{*}}{{{task_name}}} ")
        
        first_row = True
        for cls in all_classes:
            train_count = len(train_meta[train_meta[task] == cls])
            test_count = len(test_meta[test_meta[task] == cls])
            val_count = len(val_task_df[val_task_df['true_label'] == cls])
            total = train_count + test_count + val_count
            
            # Format class name
            class_str = str(cls).replace('_', '\\_').replace('&', '\\&')
            if task == 'sample_host' and cls != 'Not applicable - env sample':
                class_str = f"\\textit{{{class_str}}}"
            
            if first_row:
                lines.append(f"& {class_str} & {train_count} & {test_count} & {val_count} & {total} \\\\")
                first_row = False
            else:
                lines.append(f" & {class_str} & {train_count} & {test_count} & {val_count} & {total} \\\\")
        
        lines.append("\\addlinespace")
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular*}")
    lines.append("\\begin{tablenotes}")
    lines.append("\\item Training and test samples from curated AncientMetagenomeDir dataset.")
    lines.append("\\item Validation samples from AncientMetagenomeDir v25.09.0 and MGnify modern samples, excluding overlaps with train/test.")
    lines.append(f"\\item Validation set: {val_samples} samples with successful predictions.")
    lines.append("\\item Classes with 0 validation samples were present in training but not in the external validation set.")
    lines.append("\\item Classes with 0 training samples are UNSEEN by the model and cannot be correctly predicted.")
    lines.append("\\end{tablenotes}")
    lines.append("\\end{table*}")
    
    print("\n[4/4] Writing table...")
    output_file = output_dir / "sup_table_01_class_distribution.tex"
    with open(output_file, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"  ✓ {output_file.name}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("=" * 80)
    print("GENERATING CLASS DISTRIBUTION TABLE (SUP TABLE 1)")
    print("=" * 80)
    
    # Create output directory
    output_dir = Path(PATHS['tables_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate table
    generate_class_distribution_table(output_dir)
    
    print("\n" + "=" * 80)
    print("✓ COMPLETE - Class distribution table generated")
    print("=" * 80)


if __name__ == '__main__':
    main()
