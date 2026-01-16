#!/usr/bin/env python3
"""
Generate complete class distribution table for all tasks showing ALL labels
across train/test/validation datasets.
"""

import polars as pl
from pathlib import Path

PROJECT_ROOT = Path("/pasteur/appa/scratch/cduitama/EDID/decOM-classify")

# Load datasets from paper/metadata
train = pl.read_csv(PROJECT_ROOT / "paper" / "metadata" / "train_metadata.tsv", separator='\t')
test = pl.read_csv(PROJECT_ROOT / "paper" / "metadata" / "test_metadata.tsv", separator='\t')
val = pl.read_csv(PROJECT_ROOT / "paper" / "metadata" / "validation_metadata.tsv", separator='\t')

tasks = {
    'sample_type': 'Sample Type',
    'community_type': 'Community Type',
    'sample_host': 'Sample Host',
    'material': 'Material'
}

# Create LaTeX table
latex = []
latex.append(r"\begin{table*}[!t]")
latex.append(r"\caption{Sample distribution across classes for each dataset.\label{tab:class_distribution}}")
latex.append(r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llcccc@{\extracolsep{\fill}}}")
latex.append(r"\toprule")
latex.append(r"Task & Class & Training & Test & Validation & Total \\")
latex.append(r"\midrule")

for task_col, task_name in tasks.items():
    # Get all unique labels across all datasets (filter out None/NaN)
    train_labels = set(l for l in train[task_col].unique().to_list() if l is not None)
    test_labels = set(l for l in test[task_col].unique().to_list() if l is not None)
    val_labels = set(l for l in val[task_col].unique().to_list() if l is not None)
    all_labels = sorted(train_labels | test_labels | val_labels)
    
    # Count samples for each label
    counts = []
    for label in all_labels:
        train_count = train.filter(pl.col(task_col) == label).shape[0]
        test_count = test.filter(pl.col(task_col) == label).shape[0]
        val_count = val.filter(pl.col(task_col) == label).shape[0]
        total = train_count + test_count + val_count
        counts.append((label, train_count, test_count, val_count, total))
    
    # Sort by total count descending
    counts.sort(key=lambda x: x[4], reverse=True)
    
    # Add to table
    for i, (label, train_c, test_c, val_c, total) in enumerate(counts):
        # Escape underscores for LaTeX
        label_escaped = label.replace('_', r'\_')
        
        if i == 0:
            # First row includes task name with multirow
            latex.append(f"\\multirow{{{len(counts)}}}{{*}}{{{task_name}}}")
            latex.append(f"  & {label_escaped} & {train_c} & {test_c} & {val_c} & {total} \\\\")
        else:
            latex.append(f"  & {label_escaped} & {train_c} & {test_c} & {val_c} & {total} \\\\")
    
    latex.append(r"\addlinespace")

# Remove last \addlinespace and add closing
latex.pop()  # Remove last addlinespace
latex.append(r"\bottomrule")
latex.append(r"\end{tabular*}")
latex.append(r"\begin{tablenotes}")
latex.append(r"\item Training and test samples from curated AncientMetagenomeDir dataset (Logan et al.).")
latex.append(r"\item Validation samples from AncientMetagenomeDir v25.09.0 and MGnify modern samples, excluding overlaps with train/test.")
latex.append(r"\item Validation set: 1082 samples (937 ancient + 145 modern; 772 host-associated, 305 environmental; 28.2\% environmental).")
latex.append(r"\item Classes with 0 validation samples were present in training but not in external validation set.")
latex.append(r"\item Classes with 0 training samples are UNSEEN by the model and cannot be correctly predicted.")
latex.append(r"\end{tablenotes}")
latex.append(r"\end{table*}")

# Write LaTeX to file
output_file = PROJECT_ROOT / "paper" / "tables" / "class_distribution.tex"
with open(output_file, 'w') as f:
    f.write('\n'.join(latex))

print(f"✓ LaTeX table written to {output_file}")

# Also save as TSV for easy viewing
tsv_lines = [f"Task\tClass\tTraining (n={len(train)})\tTest (n={len(test)})\tValidation (n={len(val)})\tTotal"]
for task_col, task_name in tasks.items():
    train_labels = set(l for l in train[task_col].unique().to_list() if l is not None)
    test_labels = set(l for l in test[task_col].unique().to_list() if l is not None)
    val_labels = set(l for l in val[task_col].unique().to_list() if l is not None)
    all_labels = sorted(train_labels | test_labels | val_labels)
    
    counts = []
    for label in all_labels:
        train_count = train.filter(pl.col(task_col) == label).shape[0]
        test_count = test.filter(pl.col(task_col) == label).shape[0]
        val_count = val.filter(pl.col(task_col) == label).shape[0]
        total = train_count + test_count + val_count
        counts.append((label, train_count, test_count, val_count, total))
    
    counts.sort(key=lambda x: x[4], reverse=True)
    
    for label, train_c, test_c, val_c, total in counts:
        tsv_lines.append(f"{task_name}\t{label}\t{train_c}\t{test_c}\t{val_c}\t{total}")

tsv_file = PROJECT_ROOT / "paper" / "tables" / "class_distribution.tsv"
with open(tsv_file, 'w') as f:
    f.write('\n'.join(tsv_lines))

print(f"✓ TSV table written to {tsv_file}")
print(f"\nTotal unique classes per task:")
for task_col, task_name in tasks.items():
    train_labels = set(train[task_col].unique().to_list())
    test_labels = set(test[task_col].unique().to_list())
    val_labels = set(val[task_col].unique().to_list())
    all_labels = train_labels | test_labels | val_labels
    print(f"  {task_name}: {len(all_labels)} classes")
