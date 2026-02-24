#!/usr/bin/env python3
"""
Generate Per-Class Performance Table (Supplementary Table 3)

PURPOSE:
    Show detailed performance metrics for each class in the validation set (seen labels only).

INPUTS (from load_validation_data):
    - Validation predictions DataFrame with is_seen column

OUTPUTS:
    - paper/tables/final/sup_table_03_perclass_performance.tex

PROCESS:
    1. Load validation predictions using shared loader
    2. Filter to seen labels only (is_seen=True)
    3. For each task:
        - Use sklearn classification_report for precision/recall/F1 per class
        - Calculate per-class accuracy (% correct within that class)
        - Count samples per class
    4. Sort classes by sample count (most common first)
    5. Format as LaTeX table with species names italicized

CONFIGURATION:
    All paths and data loading from shared utilities

HARDCODED VALUES:
    - Filter to seen labels only (unseen can't have meaningful metrics)
    - Italicize species names for sample_host
    - Exception: "Not applicable - env sample" not italicized
    - Metrics: Accuracy, Precision, Recall, F1-Score

DEPENDENCIES:
    - pandas, pathlib, sklearn.metrics
    - config.py, load_validation_data.py

USAGE:
    python scripts/paper/generate_perclass_performance_table.py
    
AUTHOR: Refactored from 06_compare_predictions.py
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report

# Add script directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from config import PATHS, TASKS

# Import shared data loader from validation directory
sys.path.insert(0, str(Path(__file__).parent.parent / 'validation'))
from load_validation_data import load_validation_predictions


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


# ============================================================================
# TABLE GENERATION
# ============================================================================

def generate_perclass_performance_table(output_dir):
    """Generate supplementary table with per-class performance metrics."""
    print("\n[1/3] Loading validation predictions...")
    df = load_validation_predictions(quiet=True)
    
    print("\n[2/3] Calculating per-class metrics...")
    
    lines = []
    lines.append("\\caption{Per-class performance on validation set (seen labels only)\\label{tab:perclass_performance}}")
    lines.append("\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}llrrrrr@{\\extracolsep{\\fill}}}")
    lines.append("\\toprule")
    lines.append("Task & Class Label & n & Accuracy & Precision & Recall & F1-Score \\\\")
    lines.append("\\midrule")
    
    for task in TASKS:
        # Filter to seen labels only
        task_df = df[(df['task'] == task) & (df['is_seen'])].copy()
        task_df = filter_valid_labels(task_df)
        
        if len(task_df) == 0:
            continue
        
        task_name = task.replace('_', ' ').title()
        lines.append(f"\\multicolumn{{7}}{{l}}{{\\textbf{{{task_name}}}}} \\\\")
        lines.append("\\addlinespace[0.5em]")
        
        # Get classification report
        report_dict = classification_report(
            task_df['true_label'],
            task_df['pred_label'],
            output_dict=True,
            zero_division=0
        )
        
        # Extract per-class metrics
        class_stats = []
        for label in sorted(task_df['true_label'].unique()):
            if label in report_dict:
                n = len(task_df[task_df['true_label'] == label])
                acc = task_df[task_df['true_label'] == label]['is_correct'].mean()
                prec = report_dict[label]['precision']
                rec = report_dict[label]['recall']
                f1 = report_dict[label]['f1-score']
                class_stats.append((label, n, acc, prec, rec, f1))
        
        # Sort by sample count descending
        class_stats.sort(key=lambda x: x[1], reverse=True)
        
        for cls, n, acc, prec, rec, f1 in class_stats:
            cls_str = str(cls).replace('_', '\\_').replace('&', '\\&')
            if task == 'sample_host' and cls != 'Not applicable - env sample':
                cls_str = f"\\textit{{{cls_str}}}"
            
            lines.append(
                f" & {cls_str} & {n} & {acc*100:.1f}\\% & {prec*100:.1f}\\% & {rec*100:.1f}\\% & {f1*100:.1f}\\% \\\\"
            )
        
        lines.append("\\addlinespace")
    
    lines.append("\\botrule")
    lines.append("\\end{tabular*}")
    lines.append("\\\\[2mm]")
    
    footnote_parts = [
        "Metrics computed only for seen labels (present in training set).",
        "Accuracy: proportion of samples in each class correctly classified.",
        "Precision: proportion of predictions for a class that were correct.",
        "Recall: proportion of true instances of a class that were correctly predicted.",
        "F1-Score: harmonic mean of precision and recall."
    ]
    lines.append("{\\footnotesize " + " ".join(footnote_parts) + "}")
    
    print("\n[3/3] Writing table...")
    output_file = output_dir / "sup_table_03_perclass_performance.tex"
    with open(output_file, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"  ✓ {output_file.name}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("=" * 80)
    print("GENERATING PER-CLASS PERFORMANCE TABLE (SUP TABLE 3)")
    print("=" * 80)
    
    # Create output directory
    output_dir = Path(PATHS['tables_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate table
    generate_perclass_performance_table(output_dir)
    
    print("\n" + "=" * 80)
    print("✓ COMPLETE - Per-class performance table generated")
    print("=" * 80)


if __name__ == '__main__':
    main()
