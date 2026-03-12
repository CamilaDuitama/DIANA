#!/usr/bin/env python3
"""
Generate merged wrong predictions table (sup_table_05)

Shows common misclassification patterns across all tasks with counts and confidence stats.

Input:
- Validation predictions from shared loader (scripts/validation/load_validation_data.py)

Output:
- paper/tables/final/sup_table_05_wrong_predictions_merged.tex

Process:
1. Filter to seen labels with wrong predictions (is_seen=True, is_correct=False)
2. Group by (task, true_label, pred_label) pairs
3. Calculate count, mean confidence, std confidence
4. Show ALL patterns per task (not limited to top N)
5. Format LaTeX with task name in first row only (multirow effect)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add validation scripts to path for shared loader
sys.path.insert(0, str(Path(__file__).parent.parent / 'validation'))
from load_validation_data import load_validation_predictions

# Add paper config
sys.path.insert(0, str(Path(__file__).parent))
from config import PATHS, TASKS


def generate_wrong_predictions_table(df: pd.DataFrame, output_path: Path) -> None:
    """Generate merged table of ALL wrong predictions (A->B) with counts and confidence stats."""
    
    PLAIN_TEXT_HOSTS = {'Not applicable - env sample', 'Other mammal'}

    def fmt_host(label):
        s = str(label).replace('_', '\\_')
        if label not in PLAIN_TEXT_HOSTS:
            s = f"\\textit{{{s}}}"
        return s

    lines = []
    lines.append("\\small")
    lines.append("\\begin{longtable}{lp{4cm}p{4cm}r}")
    lines.append("\\caption{Common misclassification patterns across all tasks (Validation set)\\label{tab:wrong_merged}}\\\\")
    lines.append("\\toprule")
    lines.append("Task & True Label & Predicted Label & Count \\\\")
    lines.append("\\midrule")
    lines.append("\\endfirsthead")
    lines.append("\\multicolumn{4}{c}{\\textit{Table \\thetable{} continued from previous page}} \\\\")
    lines.append("\\toprule")
    lines.append("Task & True Label & Predicted Label & Count \\\\")
    lines.append("\\midrule")
    lines.append("\\endhead")
    lines.append("\\midrule")
    lines.append("\\multicolumn{4}{r}{\\textit{Continued on next page}} \\\\")
    lines.append("\\endfoot")
    lines.append("\\bottomrule")
    lines.append("\\endlastfoot")
    
    task_labels = {
        'sample_type': 'Sample Type',
        'community_type': 'Community Type',
        'sample_host': 'Sample Host',
        'material': 'Material'
    }
    
    for task in TASKS:
        task_df = df[df['task'] == task]
        wrong_df = task_df[task_df['is_seen'] & ~task_df['is_correct']].copy()
        
        if len(wrong_df) == 0:
            continue
        
        # Group by true->predicted pairs (ALL patterns, not top N)
        confusion_pairs = wrong_df.groupby(['true_label', 'pred_label']).agg({
            'confidence': ['count']
        }).reset_index()
        
        confusion_pairs.columns = ['true_label', 'pred_label', 'n']
        confusion_pairs = confusion_pairs.sort_values('n', ascending=False)  # Sort by frequency
        
        # Add task name to first row only
        for idx, row in confusion_pairs.iterrows():
            true_lbl = str(row['true_label']).replace('_', '\\_')
            pred_lbl = str(row['pred_label']).replace('_', '\\_')
            
            if task == 'sample_host':
                true_lbl = fmt_host(row['true_label'])
                pred_lbl = fmt_host(row['pred_label'])
            
            if idx == confusion_pairs.index[0]:
                task_str = task_labels[task]
            else:
                task_str = ""
            
            lines.append(f"{task_str} & {true_lbl} & {pred_lbl} & {int(row['n'])} \\\\")
        
        lines.append("\\addlinespace")
    
    # Remove last addlinespace
    if lines[-1] == "\\addlinespace":
        lines = lines[:-1]
    
    lines.append("\\end{longtable}")
    lines.append("\\addcontentsline{toc}{subsection}{Supplementary Table 5: Common misclassification patterns}")
    lines.append("\\\\[2mm]")
    lines.append("{\\footnotesize Only includes misclassifications of seen classes (present in training data). "
                 "Count: Number of samples misclassified. "
                 "Sorted by frequency (most common patterns first).}")
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))


def main():
    print("=" * 80)
    print("GENERATING WRONG PREDICTIONS TABLE (SUP TABLE 5)")
    print("=" * 80)
    print()
    
    # Step 1: Load validation predictions
    print("[1/3] Loading validation predictions...")
    df = load_validation_predictions()
    print()
    
    # Step 2: Generate table
    print("[2/3] Generating LaTeX table...")
    output_path = Path(PATHS['tables_dir']) / "sup_table_05_wrong_predictions_merged.tex"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    generate_wrong_predictions_table(df, output_path)
    print()
    
    # Step 3: Report
    print("[3/3] Summary:")
    print(f"  ✓ {output_path.name}")
    print()
    print("=" * 80)
    print("✓ COMPLETE - Wrong predictions table generated")
    print("=" * 80)


if __name__ == "__main__":
    main()
