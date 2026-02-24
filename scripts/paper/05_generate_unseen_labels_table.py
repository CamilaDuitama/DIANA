#!/usr/bin/env python3
"""
Generate Unseen Labels Table (Supplementary Table 2)

PURPOSE:
    Show how the model handles labels not seen during training.

INPUTS (from config.py via load_validation_data):
    - Validation predictions DataFrame with is_seen column

OUTPUTS:
    - paper/tables/final/sup_table_02_unseen_labels.tex

PROCESS:
    1. Load validation predictions using shared loader
    2. Dynamically identify tasks with unseen labels (is_seen=False)
    3. For each task with unseen labels:
        - Group by (true_label, pred_label) and count
        - Show confusion: which unseen labels map to which seen labels
    4. Format as LaTeX table with italics for species names

CONFIGURATION:
    All paths and data loading from shared utilities

HARDCODED VALUES:
    - Italicize species names for sample_host
    - Exception: "Not applicable - env sample" not italicized

DEPENDENCIES:
    - pandas, pathlib
    - config.py, load_validation_data.py

USAGE:
    python scripts/paper/generate_unseen_labels_table.py
    
AUTHOR: Refactored from 06_compare_predictions.py
"""

import sys
from pathlib import Path

import pandas as pd

# Add script directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from config import PATHS

# Import shared data loader from validation directory
sys.path.insert(0, str(Path(__file__).parent.parent / 'validation'))
from load_validation_data import load_validation_predictions, get_unseen_tasks


# ============================================================================
# TABLE GENERATION
# ============================================================================

def generate_unseen_labels_table(output_dir):
    """Generate supplementary table showing unseen label predictions."""
    print("\n[1/3] Loading validation predictions...")
    df = load_validation_predictions(quiet=True)
    
    print("\n[2/3] Identifying tasks with unseen labels...")
    unseen_tasks = get_unseen_tasks(df)
    print(f"  ✓ Found {len(unseen_tasks)} tasks with unseen labels: {unseen_tasks}")
    
    # Filter to unseen predictions only
    unseen_df = df[~df['is_seen']].copy()
    
    # Remove 'Unknown' labels (these are missing data, not truly unseen)
    unseen_df = unseen_df[unseen_df['true_label'] != 'Unknown']
    
    if len(unseen_df) == 0:
        print("  ⚠ No unseen labels found (excluding Unknown)")
        return
    
    print("\n[3/3] Generating LaTeX table...")
    
    lines = []
    lines.append("\\centering")
    lines.append("\\caption{Model predictions for unseen labels\\label{tab:unseen_labels}}")
    lines.append("\\begin{tabular*}{\\columnwidth}{@{\\extracolsep{\\fill}}llrr@{\\extracolsep{\\fill}}}")
    lines.append("\\toprule")
    lines.append("Task & True Label (Unseen) & Predicted Label & Count \\\\")
    lines.append("\\midrule")
    
    for task in unseen_tasks:
        task_unseen = unseen_df[unseen_df['task'] == task]
        
        # Skip if only 'Unknown' labels
        if len(task_unseen) == 0:
            continue
        
        task_name = task.replace('_', ' ').title()
        lines.append(f"\\multicolumn{{4}}{{l}}{{\\textbf{{{task_name}}}}} \\\\")
        lines.append("\\addlinespace[0.5em]")
        
        # Count predictions per true label
        confusion = task_unseen.groupby(['true_label', 'pred_label']).size().reset_index(name='count')
        confusion = confusion.sort_values('count', ascending=False)  # Sort by count descending
        
        for _, row in confusion.iterrows():
            true_lbl = str(row['true_label']).replace('_', '\\_').replace('&', '\\&')
            pred_lbl = str(row['pred_label']).replace('_', '\\_').replace('&', '\\&')
            count = int(row['count'])
            
            # Italicize species names for sample_host
            if task == 'sample_host':
                if row['true_label'] != 'Not applicable - env sample':
                    true_lbl = f"\\textit{{{true_lbl}}}"
                if row['pred_label'] != 'Not applicable - env sample':
                    pred_lbl = f"\\textit{{{pred_lbl}}}"
            
            lines.append(f" & {true_lbl} & {pred_lbl} & {count} \\\\")
        
        lines.append("\\addlinespace")
    
    lines.append("\\botrule")
    lines.append("\\end{tabular*}")
    lines.append("\\\\[2mm]")
    
   # Build footnote
    footnote_parts = [
        "Unseen labels are categories not present in the training set.",
        "Sample Host: Novel species/subspecies correctly mapped to genus/species-level training classes.",
        "Material: Novel material types mapped to semantically similar training classes."
    ]
    
    # Add task counts
    for task in unseen_tasks:
        task_unseen_count = len(unseen_df[unseen_df['task'] == task])
        if task_unseen_count > 0:
            task_display = task.replace('_', ' ').title()
            footnote_parts.append(f"{task_display}: {task_unseen_count} unseen predictions.")
    
    lines.append("{\\footnotesize " + " ".join(footnote_parts) + "}")
    
    output_file = output_dir / "sup_table_02_unseen_labels.tex"
    with open(output_file, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"  ✓ {output_file.name}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("=" * 80)
    print("GENERATING UNSEEN LABELS TABLE (SUP TABLE 2)")
    print("=" * 80)
    
    # Create output directory
    output_dir = Path(PATHS['tables_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate table
    generate_unseen_labels_table(output_dir)
    
    print("\n" + "=" * 80)
    print("✓ COMPLETE - Unseen labels table generated")
    print("=" * 80)


if __name__ == '__main__':
    main()
