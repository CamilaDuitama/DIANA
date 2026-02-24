#!/usr/bin/env python3
"""
Generate Supplementary Table 7: Seen vs Unseen Validation Performance

PURPOSE:
    Generate table showing validation set performance breakdown:
    - Seen labels (correct predictions)
    - Seen labels (top 10 most frequent misclassifications)
    - Unseen labels (top 10 most frequent predictions)
    
    Shows how the model handles labels it was trained on vs novel labels.

OUTPUTS:
    - paper/tables/final/sup_table_07_seen_unseen_validation.tex

USAGE:
    python scripts/paper/17_generate_seen_unseen_validation_table.py

AUTHOR: Paper generation pipeline
"""

from pathlib import Path
import pandas as pd
import sys

# Add validation scripts to path for shared loader
sys.path.insert(0, str(Path(__file__).parent.parent / 'validation'))
from load_validation_data import load_validation_predictions

# Add paper config
sys.path.insert(0, str(Path(__file__).parent))
from config import PATHS, TASKS

def load_validation_data():
    """Load validation predictions with metadata."""
    print("Loading validation data...")
    df = load_validation_predictions(quiet=True)
    print(f"  ✓ Loaded {len(df)} predictions for {len(df['sample_id'].unique())} samples")
    return df


def generate_seen_unseen_validation_table(
    df: pd.DataFrame,
    output_path: Path
) -> None:
    """Generate table showing SEEN vs UNSEEN predictions in validation set with error patterns.
    
    Shows distribution of seen/unseen samples and common misclassification patterns
    (True Label A -> Predicted Label B) with percentages relative to total validation samples.
    
    Args:
        df: Validation predictions dataframe with is_seen, is_correct columns
        output_path: Path to output LaTeX file
    """
    # Total validation samples (unique sample_ids)
    total_samples = len(df['sample_id'].unique())
    
    # Build table
    lines = []
    lines.append(r"\small")
    lines.append(r"\begin{longtable}{llp{3.5cm}p{3.5cm}rrr}")
    lines.append(r"\caption{Validation set performance: Seen vs unseen labels with top 10 most frequent misclassification patterns\label{tab:seen_unseen_validation}} \\")
    lines.append(r"\toprule")
    lines.append(r"Task & Category & True Label & Predicted Label & Count & \% Task & \% Total \\")
    lines.append(r"\midrule")
    lines.append(r"\endfirsthead")
    lines.append("")
    lines.append(r"\multicolumn{7}{c}{{\tablename\ \thetable{} -- continued from previous page}} \\")
    lines.append(r"\toprule")
    lines.append(r"Task & Category & True Label & Predicted Label & Count & \% Task & \% Total \\")
    lines.append(r"\midrule")
    lines.append(r"\endhead")
    lines.append("")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{7}{r}{{Continued on next page}} \\")
    lines.append(r"\endfoot")
    lines.append("")
    lines.append(r"\endlastfoot")
    
    task_labels = {
        'sample_type': 'Sample Type',
        'community_type': 'Community Type',
        'sample_host': 'Sample Host',
        'material': 'Material'
    }
    
    for task in TASKS:
        task_df = df[df['task'] == task].copy()
        task_total = len(task_df)
        
        if task_total == 0:
            continue
        
        task_name = task_labels[task]
        
        # Seen correct
        seen_correct = task_df[task_df['is_seen'] & task_df['is_correct']]
        seen_correct_count = len(seen_correct)
        pct_task = (seen_correct_count / task_total * 100) if task_total > 0 else 0
        pct_total = (seen_correct_count / total_samples * 100) if total_samples > 0 else 0
        
        lines.append(f"{task_name} & Seen - Correct & — & — & {seen_correct_count} & {pct_task:.1f}\\% & {pct_total:.1f}\\% \\\\")
        
        # Seen incorrect (show top 10 confusion patterns)
        seen_wrong = task_df[task_df['is_seen'] & ~task_df['is_correct']]
        if len(seen_wrong) > 0:
            confusion = seen_wrong.groupby(['true_label', 'pred_label']).size().reset_index(name='count')
            confusion = confusion.sort_values('count', ascending=False).head(10)
            
            for idx, row in confusion.iterrows():
                true_lbl = str(row['true_label']).replace('_', r'\_')
                pred_lbl = str(row['pred_label']).replace('_', r'\_')
                count = int(row['count'])
                pct_task = (count / task_total * 100) if task_total > 0 else 0
                pct_total = (count / total_samples * 100) if total_samples > 0 else 0
                
                # Format italics for species names
                if task == 'sample_host':
                    if row['true_label'] != 'Not applicable - env sample':
                        true_lbl = f"\\textit{{{true_lbl}}}"
                    if row['pred_label'] != 'Not applicable - env sample':
                        pred_lbl = f"\\textit{{{pred_lbl}}}"
                
                if idx == confusion.index[0]:
                    lines.append(f" & Seen - Wrong & {true_lbl} & {pred_lbl} & {count} & {pct_task:.1f}\\% & {pct_total:.1f}\\% \\\\")
                else:
                    lines.append(f" &  & {true_lbl} & {pred_lbl} & {count} & {pct_task:.1f}\\% & {pct_total:.1f}\\% \\\\")
        
        # Unseen (show top 10 confusion patterns)
        unseen = task_df[~task_df['is_seen']]
        if len(unseen) > 0:
            confusion_unseen = unseen.groupby(['true_label', 'pred_label']).size().reset_index(name='count')
            confusion_unseen = confusion_unseen.sort_values('count', ascending=False).head(10)
            
            for idx, row in confusion_unseen.iterrows():
                true_lbl = str(row['true_label']).replace('_', r'\_')
                pred_lbl = str(row['pred_label']).replace('_', r'\_')
                count = int(row['count'])
                pct_task = (count / task_total * 100) if task_total > 0 else 0
                pct_total = (count / total_samples * 100) if total_samples > 0 else 0
                
                # Format italics for species names
                if task == 'sample_host':
                    if row['true_label'] != 'Not applicable - env sample':
                        true_lbl = f"\\textit{{{true_lbl}}}"
                    if row['pred_label'] != 'Not applicable - env sample':
                        pred_lbl = f"\\textit{{{pred_lbl}}}"
                
                if idx == confusion_unseen.index[0]:
                    lines.append(f" & Unseen & {true_lbl} & {pred_lbl} & {count} & {pct_task:.1f}\\% & {pct_total:.1f}\\% \\\\")
                else:
                    lines.append(f" &  & {true_lbl} & {pred_lbl} & {count} & {pct_task:.1f}\\% & {pct_total:.1f}\\% \\\\")
        
        lines.append(r"\addlinespace")
    
    lines.append(r"\botrule")
    lines.append(r"\multicolumn{7}{p{0.95\linewidth}}{\footnotesize")
    lines.append(r"\textbf{Category:} Seen labels were present in training; Unseen labels were not in training. ")
    lines.append(r"\textbf{Seen - Correct:} Samples correctly classified with labels seen during training. ")
    lines.append(r"\textbf{Seen - Wrong:} Top 10 most frequent misclassification patterns for seen labels (True Label $\to$ Predicted Label). ")
    lines.append(r"\textbf{Unseen:} Top 10 most frequent predictions for unseen labels (model maps to semantically similar seen classes). ")
    lines.append(r"\textbf{\% Task:} Percentage relative to total samples for that task in validation set. ")
    lines.append(r"\textbf{\% Total:} Percentage relative to total validation samples across all tasks. ")
    lines.append(f"Total validation samples: {total_samples} (unique sample IDs across {len(TASKS)} tasks = {len(df)} predictions).")
    lines.append(r"} \\")
    lines.append(r"\end{longtable}")
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"  ✓ Table saved: {output_path}")


def main():
    """Main execution."""
    print("=" * 80)
    print("GENERATING SUPPLEMENTARY TABLE 7: SEEN/UNSEEN VALIDATION PERFORMANCE")
    print("=" * 80)
    print()
    
    # Load validation data
    df = load_validation_data()
    
    # Generate table
    output_path = Path(PATHS['tables_dir']) / "sup_table_07_seen_unseen_validation.tex"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("\nGenerating seen/unseen validation table...")
    generate_seen_unseen_validation_table(df, output_path)
    
    print()
    print("=" * 80)
    print("✓ SUPPLEMENTARY TABLE 7 GENERATED SUCCESSFULLY")
    print("=" * 80)
    print()
    print(f"Output: {output_path}")
    print()


if __name__ == "__main__":
    main()
