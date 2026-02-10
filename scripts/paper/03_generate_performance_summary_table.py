#!/usr/bin/env python3
"""
Generate Performance Summary Table (Main Table 1)

PURPOSE:
    Create LaTeX table comparing model performance across training, test, and validation sets.

INPUTS (from config.py):
    - PATHS['test_metrics']: Test set performance metrics (JSON)
    - PATHS['training_metrics']: Training set performance metrics (JSON)
    - PATHS['validation_metadata']: Validation metadata with predictions
    - PATHS['predictions_dir']: Validation predictions directory
    - PATHS['label_encoders']: Label encoder mappings

OUTPUTS:
    - paper/tables/final/main_table_01_performance_summary.tex

PROCESS:
    1. Load test metrics (461 samples, held-out test set)
    2. Load training metrics (2,609 samples, full training set)
    3. Calculate validation metrics (seen labels only)
    4. Format as LaTeX table with accuracy, balanced accuracy, F1 scores

CONFIGURATION:
    All paths imported from config.py

HARDCODED VALUES:
    - Task labels (Sample Type, Community Type, etc.)
    - Test set size: 461 samples
    - Number formatting: 1 decimal place for percentages

DEPENDENCIES:
    - pandas, json, pathlib
    - config.py, sklearn.metrics

USAGE:
    python scripts/paper/generate_performance_summary_table.py
    
AUTHOR: Refactored from 06_compare_predictions.py
"""

import sys
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score

# Add script directory to path for config import
sys.path.insert(0, str(Path(__file__).parent))
from config import PATHS, TASKS, SAMPLE_TYPE_MAP


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def filter_valid_labels(df):
    """Filter out samples with 'Unknown' or missing labels."""
    return df[
        (df['true_label'] != 'Unknown') &
        (df['true_label'].notna()) &
        (df['pred_label'].notna())
    ].copy()


def load_validation_predictions():
    """Load validation predictions and metadata."""
    # Load metadata
    metadata = pd.read_csv(PATHS['validation_metadata'], sep='\t')
    
    # Load label encoders
    with open(PATHS['label_encoders']) as f:
        label_encoders = json.load(f)
    
    # Pre-compute normalized training classes for sample_type
    normalized_sample_type_classes = [
        SAMPLE_TYPE_MAP.get(c, c) 
        for c in label_encoders['sample_type']['classes']
    ]
    
    # Find all prediction files
    predictions_dir = Path(PATHS['predictions_dir'])
    prediction_files = list(predictions_dir.rglob('*_predictions.json'))
    
    # Build records
    records = []
    for pred_file in prediction_files:
        # Extract sample ID from parent directory
        sample_id = pred_file.parent.name
        
        with open(pred_file) as f:
            pred = json.load(f)
        
        # Find matching metadata
        sample_meta = metadata[metadata['Run_accession'] == sample_id]
        if len(sample_meta) == 0:
            continue
        
        sample_meta = sample_meta.iloc[0]
        
        # Process each task
        for task_name in TASKS:
            true_label = sample_meta[task_name]
            
            # Normalize sample_type labels (ancient_metagenome → ancient)
            if task_name == 'sample_type' and true_label in SAMPLE_TYPE_MAP:
                true_label = SAMPLE_TYPE_MAP[true_label]
            
            pred_info = pred['predictions'][task_name]
            
            # Decode prediction
            pred_value = pred_info['predicted_class']
            if isinstance(pred_value, str) and pred_value.isdigit():
                class_idx = int(pred_value)
                pred_label = label_encoders[task_name]['classes'][class_idx]
            else:
                pred_label = pred_value
            
            # Normalize pred_label for sample_type
            if task_name == 'sample_type' and pred_label in SAMPLE_TYPE_MAP:
                pred_label = SAMPLE_TYPE_MAP[pred_label]
            
            # Check if seen in training (use pre-computed normalization for sample_type)
            if task_name == 'sample_type':
                is_seen = true_label in normalized_sample_type_classes
            else:
                is_seen = true_label in label_encoders[task_name]['classes']
            
            records.append({
                'sample_id': sample_id,
                'task': task_name,
                'true_label': str(true_label) if pd.notna(true_label) else 'Unknown',
                'pred_label': pred_label,
                'is_correct': pred_label == str(true_label),
                'is_seen': is_seen
            })
    
    return pd.DataFrame(records)


def calculate_validation_metrics(validation_df):
    """Calculate metrics for validation set (seen labels only)."""
    val_metrics = {}
    
    for task in TASKS:
        task_df = validation_df[(validation_df['task'] == task) & (validation_df['is_seen'])].copy()
        task_df = filter_valid_labels(task_df)
        
        if len(task_df) > 0:
            y_true = task_df['true_label'].values
            y_pred = task_df['pred_label'].values
            all_labels = sorted(set(y_true) | set(y_pred))
            
            val_metrics[task] = {
                'n': len(task_df),
                'accuracy': task_df['is_correct'].mean(),
                'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
                'f1_macro': f1_score(y_true, y_pred, labels=all_labels, average='macro', zero_division=0)
            }
        else:
            val_metrics[task] = {
                'n': 0,
                'accuracy': 0.0,
                'balanced_accuracy': 0.0,
                'f1_macro': 0.0
            }
    
    return val_metrics


# ============================================================================
# TABLE GENERATION
# ============================================================================

def generate_performance_summary_table(output_dir):
    """Generate main performance summary table."""
    print("\n[1/4] Loading test metrics...")
    with open(PATHS['test_metrics']) as f:
        test_metrics = json.load(f)
    print(f"  ✓ Loaded test metrics for {len(test_metrics)} tasks")
    
    print("\n[2/4] Loading training metrics...")
    training_metrics_file = Path(PATHS['training_metrics'])
    if training_metrics_file.exists():
        with open(training_metrics_file) as f:
            train_metrics = json.load(f)
        
        # Add sample count (all training samples)
        train_meta = pd.read_csv(PATHS['train_metadata'], sep='\t')
        for task in TASKS:
            if task not in train_metrics:
                train_metrics[task] = {}
            train_metrics[task]['n'] = len(train_meta)
        
        print(f"  ✓ Loaded actual training metrics (full {len(train_meta)} samples)")
        use_actual_training = True
    else:
        print("  ⚠ Training metrics not found, will skip training row")
        train_metrics = {}
        use_actual_training = False
    
    print("\n[3/4] Calculating validation metrics...")
    validation_df = load_validation_predictions()
    val_metrics = calculate_validation_metrics(validation_df)
    total_val = sum(m['n'] for m in val_metrics.values())
    print(f"  ✓ Processed {total_val} validation predictions (seen labels only)")
    
    print("\n[4/4] Generating LaTeX table...")
    
    task_labels = {
        'sample_type': 'Sample Type (ancient/modern)',
        'community_type': 'Community Type (6 types)',
        'sample_host': 'Sample Host (12 species)',
        'material': 'Material (13 types)'
    }
    
    lines = []
    lines.append("\\begin{table*}[!t]")
    lines.append("\\centering")
    lines.append("\\caption{Final model performance across the Training set, the held-out Test set, and the external Validation set.}")
    lines.append("\\label{tab:performance}")
    lines.append("\\small")
    lines.append("\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}llrrrr@{\\extracolsep{\\fill}}}")
    lines.append("\\toprule")
    lines.append("Task & Dataset & n & Acc (\\%) & Bal Acc (\\%) & F1 Score (\\%) \\\\")
    lines.append("\\midrule")
    
    for task in TASKS:
        task_label = task_labels[task]
        
        # Training (if available)
        if use_actual_training and task in train_metrics:
            train_n = train_metrics[task]['n']
            train_acc = train_metrics[task].get('accuracy', 0) * 100
            train_bal = train_metrics[task].get('balanced_accuracy', 0) * 100
            # Use f1_weighted if f1_macro not available
            train_f1 = train_metrics[task].get('f1_macro', train_metrics[task].get('f1_weighted', 0)) * 100
            lines.append(f"{task_label} & Training & {train_n} & {train_acc:.1f} & {train_bal:.1f} & {train_f1:.1f} \\\\")
        
        # Test
        test_acc = test_metrics[task]['accuracy'] * 100
        test_bal = test_metrics[task]['balanced_accuracy'] * 100
        test_f1 = test_metrics[task]['f1_macro'] * 100
        prefix = "" if use_actual_training else task_label
        lines.append(f"{prefix} & Test & 461 & {test_acc:.1f} & {test_bal:.1f} & {test_f1:.1f} \\\\")
        
        # Validation
        if task in val_metrics:
            val_n = val_metrics[task]['n']
            val_acc = val_metrics[task]['accuracy'] * 100
            val_bal = val_metrics[task]['balanced_accuracy'] * 100
            val_f1 = val_metrics[task]['f1_macro'] * 100
            lines.append(f" & Validation & {val_n} & {val_acc:.1f} & {val_bal:.1f} & {val_f1:.1f} \\\\")
        
        lines.append("\\addlinespace")
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular*}")
    lines.append("\\begin{tablenotes}")
    if use_actual_training:
        lines.append(r"\item Training: Performance on all 2,609 training samples (seen labels only).")
    lines.append(r"\item Test: Performance on the held-out test set (n=461).")
    lines.append("\\item Validation: Performance on the external validation set. Metrics computed only on samples with labels seen during training.")
    lines.append("\\item Acc: Accuracy. Bal Acc: Balanced Accuracy (average per-class recall). F1 Score: Macro-averaged F1-score.")
    lines.append("\\end{tablenotes}")
    lines.append("\\end{table*}")
    
    output_file = output_dir / "main_table_01_performance_summary.tex"
    with open(output_file, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"  ✓ {output_file.name}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("=" * 80)
    print("GENERATING PERFORMANCE SUMMARY TABLE (MAIN TABLE 1)")
    print("=" * 80)
    
    # Create output directory
    output_dir = Path(PATHS['tables_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate table
    generate_performance_summary_table(output_dir)
    
    print("\n" + "=" * 80)
    print("✓ COMPLETE - Performance summary table generated")
    print("=" * 80)


if __name__ == '__main__':
    main()
