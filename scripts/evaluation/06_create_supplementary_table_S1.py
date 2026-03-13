#!/usr/bin/env python3
"""
Create Supplementary Table S1 with complete metrics for all datasets.
"""

import json
import pandas as pd
from pathlib import Path

# Load training metrics
with open("results/full_training/training_set_metrics.json") as f:
    train_data = json.load(f)

# Load test metrics
test_df = pd.read_csv("paper/tables/model_evaluation/test_set_performance_summary.csv")
test_data = {}
for _, row in test_df.iterrows():
    task = row['Task'].lower().replace(' ', '_').replace('-', '_')
    test_data[task] = {
        'accuracy': row['Accuracy'],
        'balanced_accuracy': row['Balanced Accuracy'],
        'f1_weighted': row['F1-Weighted'],
        'precision_macro': row['Precision-Macro'],
        'recall_macro': row['Recall-Macro']
    }

# Load validation metrics (seen only)
with open("results/validation_predictions/validation_metrics_seen_only.json") as f:
    val_seen = json.load(f)

# Load validation summary for ALL samples
with open("paper/tables/validation/validation_comparison.json") as f:
    val_all = json.load(f)

# Task names
tasks = {
    'sample_type': 'Sample Type',
    'community_type': 'Community Type',
    'sample_host': 'Sample Host',
    'material': 'Material'
}

rows = []

for task_key, task_name in tasks.items():
    # Training
    if task_key in train_data:
        rows.append({
            'Task': task_name,
            'Dataset': 'Training',
            'N': 2609,
            'Accuracy': f"{train_data[task_key]['accuracy'] * 100:.1f}",
            'Balanced Accuracy': f"{train_data[task_key]['balanced_accuracy'] * 100:.1f}",
            'F1-weighted': f"{train_data[task_key]['f1_weighted'] * 100:.1f}",
            'Precision (macro)': f"{train_data[task_key]['precision_macro'] * 100:.1f}",
            'Recall (macro)': f"{train_data[task_key]['recall_macro'] * 100:.1f}"
        })
    
    # Test
    if task_key in test_data:
        rows.append({
            'Task': task_name,
            'Dataset': 'Test',
            'N': 461,
            'Accuracy': f"{test_data[task_key]['accuracy']:.1f}",
            'Balanced Accuracy': f"{test_data[task_key]['balanced_accuracy']:.1f}",
            'F1-weighted': f"{test_data[task_key]['f1_weighted']:.1f}",
            'Precision (macro)': f"{test_data[task_key]['precision_macro']:.1f}",
            'Recall (macro)': f"{test_data[task_key]['recall_macro']:.1f}"
        })
    
    # Validation (seen labels only)
    if task_key in val_seen and val_seen[task_key]['total_samples'] > 0:
        n_seen = val_seen[task_key]['total_samples']
        rows.append({
            'Task': task_name,
            'Dataset': 'Validation (seen labels)',
            'N': n_seen,
            'Accuracy': f"{val_seen[task_key]['accuracy'] * 100:.1f}",
            'Balanced Accuracy': f"{val_seen[task_key]['balanced_accuracy'] * 100:.1f}",
            'F1-weighted': f"{val_seen[task_key]['f1_macro'] * 100:.1f}",
            'Precision (macro)': f"{val_seen[task_key]['precision_macro'] * 100:.1f}",
            'Recall (macro)': f"{val_seen[task_key]['recall_macro'] * 100:.1f}"
        })
    
    # Validation (all samples including unseen)
    if task_key in val_all:
        all_data = val_all[task_key]['all']
        seen_data = val_all[task_key]['seen']
        n_all = all_data['total']
        
        # For "all", use simple accuracy
        rows.append({
            'Task': task_name,
            'Dataset': 'Validation (all samples)',
            'N': n_all,
            'Accuracy': f"{all_data['accuracy'] * 100:.1f}",
            'Balanced Accuracy': '—',  # Not meaningful with unseen classes
            'F1-weighted': '—',
            'Precision (macro)': '—',
            'Recall (macro)': '—'
        })

# Create DataFrame
df = pd.DataFrame(rows)

# Save as TSV
output_file = Path("paper/tables/supplementary_table_S1_complete_metrics.tsv")
df.to_csv(output_file, sep='\t', index=False)

print(f"✅ Saved to {output_file}")
print(f"\nSummary:")
print(df.to_string(index=False))
