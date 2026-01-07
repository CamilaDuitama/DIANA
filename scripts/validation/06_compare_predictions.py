#!/usr/bin/env python3
"""
Compare validation predictions to true labels from metadata.
Generate comprehensive figures and tables for paper.

This script uses a DataFrame-centric workflow for flexible analysis and visualization.
"""

import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import roc_curve, auc, precision_recall_curve


# ============================================================================
# CONFIGURATION AND UTILITIES
# ============================================================================

SAMPLE_TYPE_MAP = {
    'ancient_metagenome': 'ancient',
    'modern_metagenome': 'modern'
}

TASKS = ['sample_type', 'community_type', 'sample_host', 'material']


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare validation predictions to true labels and generate analysis figures/tables"
    )
    parser.add_argument(
        '--metadata',
        type=str,
        default='data/validation/validation_metadata_expanded.tsv',
        help='Path to validation metadata TSV file'
    )
    parser.add_argument(
        '--predictions-dir',
        type=str,
        default='results/validation_predictions',
        help='Directory containing prediction JSON files'
    )
    parser.add_argument(
        '--label-encoders',
        type=str,
        default='results/full_training/label_encoders.json',
        help='Path to label encoders JSON file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='paper',
        help='Base output directory for figures and tables'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose console output (only generate files)'
    )
    return parser.parse_args()


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(metadata_file, predictions_dir, label_encoders_file, quiet=False):
    """
    Load all data and create master DataFrame with one row per sample per task.
    
    Returns:
        df: Master DataFrame with columns [sample_id, task, true_label, pred_label, 
            confidence, is_correct, is_seen, probabilities]
        label_encoders: Dictionary of label encoders per task
        metadata: Original metadata DataFrame
    """
    if not quiet:
        print("="*80)
        print("LOADING DATA")
        print("="*80)
    
    # Load label encoders
    with open(label_encoders_file) as f:
        label_encoders = json.load(f)
    if not quiet:
        print(f"✓ Loaded label encoders")
    
    # Load metadata
    metadata = pd.read_csv(metadata_file, sep='\t')
    if not quiet:
        print(f"✓ Loaded metadata: {len(metadata)} samples")
    
    # Find prediction files
    predictions_dir = Path(predictions_dir)
    prediction_files = list(predictions_dir.glob("*/*_predictions.json"))
    if not quiet:
        print(f"✓ Found {len(prediction_files)} prediction files")
    
    # Build master DataFrame
    records = []
    
    for pred_file in prediction_files:
        sample_id = pred_file.parent.name
        
        # Load prediction
        with open(pred_file) as f:
            pred = json.load(f)
        
        # Get metadata for this sample
        sample_meta = metadata[metadata['run_accession'] == sample_id]
        if len(sample_meta) == 0:
            print(f"  WARNING: No metadata for {sample_id}")
            continue
        
        sample_meta = sample_meta.iloc[0]
        
        # Process each task
        for task_name in TASKS:
            true_label = sample_meta[task_name]
            
            # Normalize sample_type labels
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
            
            confidence = pred_info['confidence']
            probs = pred_info.get('probabilities', {})
            
            # Check if seen in training (normalize classes for sample_type comparison)
            if task_name == 'sample_type':
                # Normalize training classes for comparison
                training_classes = [SAMPLE_TYPE_MAP.get(c, c) for c in label_encoders[task_name]['classes']]
                is_seen = true_label in training_classes
            else:
                is_seen = true_label in label_encoders[task_name]['classes']
            
            is_correct = true_label == pred_label
            
            records.append({
                'sample_id': sample_id,
                'task': task_name,
                'true_label': true_label,
                'pred_label': pred_label,
                'confidence': confidence,
                'is_correct': is_correct,
                'is_seen': is_seen,
                'probabilities': probs  # Store as dict directly
            })
    
    df = pd.DataFrame(records)
    if not quiet:
        print(f"✓ Created master DataFrame: {len(df)} records ({len(df['sample_id'].unique())} samples × {len(TASKS)} tasks)")
        print()
    
    return df, label_encoders, metadata


# ============================================================================
# METRICS CALCULATION
# ============================================================================

def calculate_summary_metrics(df, quiet=False):
    """Calculate accuracy and confusion statistics from master DataFrame."""
    if not quiet:
        print("="*80)
        print("CALCULATING SUMMARY METRICS")
        print("="*80)
    
    summary_data = []
    
    for task in TASKS:
        task_df = df[df['task'] == task]
        task_title = task.replace('_', ' ').title()
        
        if not quiet:
            print(f"\n{task_title}")
            print("-" * 40)
        
        # Metrics for SEEN only
        seen_df = task_df[task_df['is_seen']]
        if len(seen_df) > 0:
            seen_acc = seen_df['is_correct'].mean() * 100
            if not quiet:
                print(f"  SEEN ONLY:   {seen_df['is_correct'].sum():3d}/{len(seen_df):3d} ({seen_acc:.1f}%)")
            summary_data.append({
                'Task': task_title,
                'Subset': 'SEEN ONLY',
                'Correct': seen_df['is_correct'].sum(),
                'Total': len(seen_df),
                'Accuracy (%)': f"{seen_acc:.1f}"
            })
        
        # Metrics for ALL
        all_acc = task_df['is_correct'].mean() * 100
        if not quiet:
            print(f"  ALL SAMPLES: {task_df['is_correct'].sum():3d}/{len(task_df):3d} ({all_acc:.1f}%)")
        summary_data.append({
            'Task': task_title,
            'Subset': 'ALL SAMPLES',
            'Correct': task_df['is_correct'].sum(),
            'Total': len(task_df),
            'Accuracy (%)': f"{all_acc:.1f}"
        })
        
        # Top confusion pairs
        if not quiet:
            print("  Top confusion pairs:")
            confusion = task_df.groupby(['true_label', 'pred_label']).size().sort_values(ascending=False).head(10)
            for (true_l, pred_l), count in confusion.items():
                match = "✓" if true_l == pred_l else "✗"
                print(f"    {match} True: {true_l:30s} | Pred: {pred_l:30s} | {count:4d}")
    
    if not quiet:
        print()
    return pd.DataFrame(summary_data)


def analyze_unseen_labels(df, quiet=False):
    """Analyze how model handles unseen labels."""
    if not quiet:
        print("="*80)
        print("UNSEEN LABEL ANALYSIS")
        print("="*80)
    
    for task in TASKS:
        task_df = df[df['task'] == task]
        unseen_df = task_df[~task_df['is_seen']]
        
        if len(unseen_df) == 0:
            continue
        
        task_title = task.replace('_', ' ').title()
        if not quiet:
            print(f"\n{task_title} - {len(unseen_df)} unseen samples:")
        
        for true_label in unseen_df['true_label'].unique():
            label_df = unseen_df[unseen_df['true_label'] == true_label]
            pred_counts = label_df['pred_label'].value_counts()
            
            if not quiet:
                print(f"\n  True: {true_label} ({len(label_df)} samples)")
                for pred_label, count in pred_counts.items():
                    pct = count / len(label_df) * 100
                    avg_conf = label_df[label_df['pred_label'] == pred_label]['confidence'].mean()
                    print(f"    → {pred_label:30s} ({count:3d}, {pct:5.1f}%, conf: {avg_conf:.3f})")
    
    if not quiet:
        print()


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_confusion_matrix(df, task, output_dir, quiet=False):
    """Generate confusion matrix heatmap for a specific task."""
    task_df = df[df['task'] == task]
    task_title = task.replace('_', ' ').title()
    
    # Get confusion matrix
    confusion = task_df.groupby(['true_label', 'pred_label']).size().reset_index(name='count')
    all_labels = sorted(set(task_df['true_label'].unique()) | set(task_df['pred_label'].unique()))
    
    # Build matrix
    cm = np.zeros((len(all_labels), len(all_labels)))
    for _, row in confusion.iterrows():
        i = all_labels.index(row['true_label'])
        j = all_labels.index(row['pred_label'])
        cm[i, j] = row['count']
    
    # Normalize by row
    cm_norm = cm / (cm.sum(axis=1, keepdims=True) + 1e-10)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=cm_norm,
        x=all_labels,
        y=all_labels,
        colorscale='Blues',
        text=cm.astype(int),
        texttemplate='%{text}',
        textfont={"size": 8},
        colorbar=dict(title="Proportion")
    ))
    
    fig.update_layout(
        title=f"{task_title} - Confusion Matrix (n={len(task_df)})",
        xaxis_title="Predicted",
        yaxis_title="True",
        width=max(600, len(all_labels) * 50),
        height=max(600, len(all_labels) * 50),
        font=dict(size=10)
    )
    
    # Save
    html_file = output_dir / f"confusion_matrix_{task}.html"
    png_file = output_dir / f"confusion_matrix_{task}.png"
    fig.write_html(html_file)
    fig.write_image(png_file, width=max(800, len(all_labels) * 60), height=max(800, len(all_labels) * 60))
    if not quiet:
        print(f"  ✓ {png_file.name}")


def plot_per_class_performance(df, task, output_dir, quiet=False):
    """Generate per-class performance metrics (Precision, Recall, F1) as grouped bar chart."""
    task_df = df[df['task'] == task]
    task_title = task.replace('_', ' ').title()
    
    # Calculate per-class metrics
    all_labels = sorted(task_df['true_label'].unique())
    class_metrics = []
    
    for label in all_labels:
        tp = len(task_df[(task_df['true_label'] == label) & (task_df['pred_label'] == label)])
        total_true = len(task_df[task_df['true_label'] == label])
        total_pred = len(task_df[task_df['pred_label'] == label])
        
        precision = tp / total_pred if total_pred > 0 else 0
        recall = tp / total_true if total_true > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        class_metrics.append({
            'Class': label,
            'Samples': total_true,
            'Precision': precision,
            'Recall': recall,
            'F1': f1
        })
    
    df_metrics = pd.DataFrame(class_metrics)
    
    # Reshape for grouped bar chart
    df_melted = df_metrics.melt(
        id_vars=['Class'],
        value_vars=['Precision', 'Recall', 'F1'],
        var_name='Metric',
        value_name='Score'
    )
    
    # Create grouped bar chart using plotly express
    fig = px.bar(
        df_melted,
        x='Class',
        y='Score',
        color='Metric',
        barmode='group',
        title=f"{task_title} - Per-Class Performance",
        color_discrete_map={
            'Precision': '#2E86AB',
            'Recall': '#A23B72',
            'F1': '#6A994E'
        },
        text='Score'
    )
    
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig.update_layout(
        yaxis_title="Score",
        yaxis_range=[0, 1.1],
        height=500,
        width=max(800, len(all_labels) * 120),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    html_file = output_dir / f"per_class_performance_{task}.html"
    png_file = output_dir / f"per_class_performance_{task}.png"
    fig.write_html(html_file)
    fig.write_image(png_file, width=max(800, len(all_labels) * 120), height=500)
    if not quiet:
        print(f"  ✓ {png_file.name}")


def plot_accuracy_comparison(df, task, output_dir, quiet=False):
    """Compare accuracy on SEEN vs ALL labels."""
    task_df = df[df['task'] == task]
    task_title = task.replace('_', ' ').title()
    
    acc_data = []
    
    # SEEN only
    seen_df = task_df[task_df['is_seen']]
    if len(seen_df) > 0:
        acc_data.append({
            'Subset': 'SEEN',
            'Accuracy': seen_df['is_correct'].mean() * 100,
            'Samples': len(seen_df)
        })
    
    # ALL
    acc_data.append({
        'Subset': 'ALL',
        'Accuracy': task_df['is_correct'].mean() * 100,
        'Samples': len(task_df)
    })
    
    df_acc = pd.DataFrame(acc_data)
    
    fig = go.Figure(data=[
        go.Bar(x=df_acc['Subset'], y=df_acc['Accuracy'],
               text=[f"{acc:.1f}%<br>(n={n})" for acc, n in zip(df_acc['Accuracy'], df_acc['Samples'])],
               textposition='outside',
               marker_color=['#2E86AB', '#A23B72'])
    ])
    
    fig.update_layout(
        title=f"{task_title} - Accuracy: Seen vs All Labels",
        yaxis_title="Accuracy (%)",
        yaxis_range=[0, 110],
        showlegend=False,
        height=400,
        width=500
    )
    
    html_file = output_dir / f"accuracy_comparison_{task}.html"
    png_file = output_dir / f"accuracy_comparison_{task}.png"
    fig.write_html(html_file)
    fig.write_image(png_file, width=500, height=400)
    if not quiet:
        print(f"  ✓ {png_file.name}")


def plot_confidence_distribution(df, task, output_dir, quiet=False):
    """
    Plot confidence distribution split by:
    - Correct/Incorrect
    - Seen/Unseen
    
    Creates box plot with jittered points showing model calibration.
    """
    task_df = df[df['task'] == task]
    task_title = task.replace('_', ' ').title()
    
    # Create category combinations
    task_df = task_df.copy()
    task_df['category'] = task_df.apply(
        lambda row: f"{'Correct' if row['is_correct'] else 'Incorrect'} - {'Seen' if row['is_seen'] else 'Unseen'}",
        axis=1
    )
    
    # Filter to only categories that exist
    if len(task_df) == 0:
        return
    
    # Create box plot with points overlay using plotly express
    fig = px.box(
        task_df,
        x='category',
        y='confidence',
        color='category',
        points='all',  # Show all points with jitter
        title=f"{task_title} - Confidence Distribution by Correctness and Label Type",
        color_discrete_map={
            'Correct - Seen': '#2E86AB',
            'Incorrect - Seen': '#A6192E',
            'Correct - Unseen': '#6A994E',
            'Incorrect - Unseen': '#F77F00'
        },
        category_orders={
            'category': ['Correct - Seen', 'Incorrect - Seen', 'Correct - Unseen', 'Incorrect - Unseen']
        }
    )
    
    fig.update_layout(
        yaxis_title="Confidence",
        xaxis_title="Category",
        yaxis_range=[0, 1.05],
        showlegend=False,
        height=500,
        width=900
    )
    
    html_file = output_dir / f"confidence_distribution_{task}.html"
    png_file = output_dir / f"confidence_distribution_{task}.png"
    fig.write_html(html_file)
    fig.write_image(png_file, width=900, height=500)
    if not quiet:
        print(f"  ✓ {png_file.name}")


def plot_roc_pr_curves(df, task, label_encoders, output_dir, quiet=False):
    """Generate ROC and Precision-Recall curves using macro-average for multi-class tasks."""
    from sklearn.preprocessing import label_binarize
    
    task_df = df[df['task'] == task]
    task_title = task.replace('_', ' ').title()
    classes = label_encoders[task]['classes']
    n_classes = len(classes)
    
    # Build arrays of true labels and probability scores
    y_true_labels = []
    y_scores_matrix = []
    
    for _, row in task_df.iterrows():
        probs = row['probabilities']
        
        # Get probability for each class
        class_probs = []
        for i, cls in enumerate(classes):
            prob = float(probs.get(str(i), 0.0))
            class_probs.append(prob)
        
        y_true_labels.append(row['true_label'])
        y_scores_matrix.append(class_probs)
    
    # Convert to numpy arrays
    y_scores = np.array(y_scores_matrix)
    
    # Check if we have multiple classes in the validation data
    unique_labels = set(y_true_labels)
    if len(unique_labels) < 2:
        if not quiet:
            print(f"  ⚠ Skipping ROC/PR for {task}: only {len(unique_labels)} class in validation data")
        return
    
    # Binarize the labels for multi-class
    y_true_bin = label_binarize([classes.index(lbl) if lbl in classes else -1 for lbl in y_true_labels], 
                                 classes=list(range(n_classes)))
    
    # Convert to dense array if sparse (label_binarize may return sparse matrix)
    if hasattr(y_true_bin, 'toarray'):
        y_true_bin = y_true_bin.toarray()
    
    # Check if we have valid data
    if y_true_bin.shape[0] == 0 or np.sum(y_true_bin) == 0:
        if not quiet:
            print(f"  ⚠ Skipping ROC/PR for {task}: insufficient data")
        return
    
    # Compute macro-average ROC curve and AUC
    fpr_dict = {}
    tpr_dict = {}
    roc_auc_dict = {}
    
    for i in range(n_classes):
        if np.sum(y_true_bin[:, i]) > 0:  # Only if class exists in data
            fpr_dict[i], tpr_dict[i], _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
            roc_auc_dict[i] = auc(fpr_dict[i], tpr_dict[i])
    
    # Compute macro-average (interpolate all ROC curves)
    all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in fpr_dict.keys()]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in fpr_dict.keys():
        mean_tpr += np.interp(all_fpr, fpr_dict[i], tpr_dict[i])
    mean_tpr /= len(fpr_dict)
    
    fpr_macro = all_fpr
    tpr_macro = mean_tpr
    roc_auc_macro = auc(fpr_macro, tpr_macro)
    
    # Compute macro-average PR curve
    precision_dict = {}
    recall_dict = {}
    pr_auc_dict = {}
    
    for i in range(n_classes):
        if np.sum(y_true_bin[:, i]) > 0:
            precision_dict[i], recall_dict[i], _ = precision_recall_curve(y_true_bin[:, i], y_scores[:, i])
            pr_auc_dict[i] = auc(recall_dict[i], precision_dict[i])
    
    # Compute macro-average PR (interpolate all curves)
    all_recall = np.unique(np.concatenate([recall_dict[i] for i in recall_dict.keys()]))
    all_recall = np.sort(all_recall)
    mean_precision = np.zeros_like(all_recall)
    for i in precision_dict.keys():
        mean_precision += np.interp(all_recall, recall_dict[i][::-1], precision_dict[i][::-1])
    mean_precision /= len(precision_dict)
    
    recall_macro = all_recall
    precision_macro = mean_precision
    pr_auc_macro = auc(recall_macro, precision_macro)
    
    # Create subplot with macro-average curves
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            f'ROC Curve (Macro AUC = {roc_auc_macro:.3f})',
            f'Precision-Recall (Macro AUC = {pr_auc_macro:.3f})'
        )
    )
    
    # ROC curve - macro average
    fig.add_trace(
        go.Scatter(x=fpr_macro, y=tpr_macro, mode='lines',
                   name=f'Macro-avg (AUC={roc_auc_macro:.2f})',
                   line=dict(color='#2E86AB', width=3)),
        row=1, col=1
    )
    
    # Add diagonal reference line
    fig.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                   line=dict(color='gray', dash='dash', width=1),
                   showlegend=False),
        row=1, col=1
    )
    
    # PR curve - macro average  
    fig.add_trace(
        go.Scatter(x=recall_macro, y=precision_macro, mode='lines',
                   name=f'Macro-avg (AUC={pr_auc_macro:.2f})',
                   line=dict(color='#A23B72', width=3)),
        row=1, col=2
    )
    
    # Update axes
    fig.update_xaxes(title_text="False Positive Rate", row=1, col=1, range=[0, 1])
    fig.update_yaxes(title_text="True Positive Rate", row=1, col=1, range=[0, 1])
    fig.update_xaxes(title_text="Recall", row=1, col=2, range=[0, 1])
    fig.update_yaxes(title_text="Precision", row=1, col=2, range=[0, 1])
    
    fig.update_layout(
        title=f"{task_title} - Performance Curves (n={len(task_df)}, {n_classes} classes)",
        height=500,
        width=1200,
        showlegend=False
    )
    
    html_file = output_dir / f"roc_pr_curves_{task}.html"
    png_file = output_dir / f"roc_pr_curves_{task}.png"
    fig.write_html(html_file)
    fig.write_image(png_file, width=1200, height=500)
    if not quiet:
        print(f"  ✓ {png_file.name}")


# ============================================================================
# TABLE GENERATION
# ============================================================================

def save_tables(df, summary_df, output_dir, quiet=False):
    """Save summary tables to TSV and HTML."""
    if not quiet:
        print("\nGenerating tables...")
    
    # Table 1: Summary accuracy
    summary_df.to_csv(output_dir / "validation_accuracy_summary.tsv", sep='\t', index=False)
    summary_df.to_html(output_dir / "validation_accuracy_summary.html", index=False)
    if not quiet:
        print(f"  ✓ validation_accuracy_summary.tsv")
    
    # Table 2: Unseen predictions per task
    for task in TASKS:
        task_df = df[df['task'] == task]
        unseen_df = task_df[~task_df['is_seen']]
        
        if len(unseen_df) > 0:
            unseen_export = unseen_df[['sample_id', 'true_label', 'pred_label', 'confidence']].copy()
            filename = f"unseen_predictions_{task}.tsv"
            unseen_export.to_csv(output_dir / filename, sep='\t', index=False)
            if not quiet:
                print(f"  ✓ {filename}")
    
    # Table 3: Detailed results JSON (save in same directory as tables)
    output_file = output_dir / "validation_comparison.json"
    
    output_json = {}
    for task in TASKS:
        task_df = df[df['task'] == task]
        output_json[task] = {}
        
        for subset_name, subset_df in [('all', task_df), ('seen', task_df[task_df['is_seen']])]:
            if len(subset_df) > 0:
                output_json[task][subset_name] = {
                    'accuracy': subset_df['is_correct'].mean(),
                    'correct': int(subset_df['is_correct'].sum()),
                    'total': len(subset_df)
                }
    
    with open(output_file, 'w') as f:
        json.dump(output_json, f, indent=2)
    
    if not quiet:
        print(f"  ✓ validation_comparison.json")


# ============================================================================
# JOB PERFORMANCE ANALYSIS
# ============================================================================

def load_job_performance_data(metadata, quiet=False):
    """
    Load job performance metrics from SLURM accounting using sacct and reportseff.
    
    Returns DataFrame with columns:
    - sample_id
    - input_size_mb
    - mem_allocated_gb
    - threads
    - elapsed_time
    - cpu_efficiency
    - mem_efficiency
    - state
    """
    import subprocess
    
    if not quiet:
        print("="*80)
        print("LOADING JOB PERFORMANCE DATA")
        print("="*80)
        print()
    
    # Job IDs from validation runs
    job_ids = ['64031197', '64138622']  # 64GB and 128GB jobs
    
    # Get reportseff data for both jobs
    all_job_data = []
    for job_id in job_ids:
        try:
            result = subprocess.run(['reportseff', job_id], capture_output=True, text=True, check=True)
            lines = result.stdout.strip().split('\n')[1:]  # Skip header
            
            for line in lines:
                parts = line.split()
                if len(parts) >= 6 and '_' in parts[0]:
                    task_num = int(parts[0].split('_')[1])
                    all_job_data.append({
                        'job_id': job_id,
                        'task': task_num,
                        'state': parts[1],
                        'elapsed': parts[2],
                        'cpu_eff': float(parts[4].replace('%', '')) if parts[4] != 'N/A' else 0.0,
                        'mem_eff': float(parts[5].replace('%', '')) if parts[5] != 'N/A' else 0.0
                    })
        except Exception as e:
            if not quiet:
                print(f"  Warning: Could not get reportseff data for job {job_id}: {e}")
    
    # Get memory allocation and CPU info from sacct
    try:
        job_ids_str = ','.join(job_ids)
        result = subprocess.run(
            ['sacct', '-j', job_ids_str, '--format=JobID,AllocCPUs,ReqMem', '-X'],
            capture_output=True, text=True, check=True
        )
        
        mem_info = {}
        for line in result.stdout.strip().split('\n')[2:]:  # Skip headers
            parts = line.split()
            if len(parts) >= 3 and '_' in parts[0]:
                job_task = parts[0]
                cpus = int(parts[1])
                mem_str = parts[2]  # e.g., "64G" or "128G"
                mem_gb = int(mem_str.replace('G', '').replace('M', '')) 
                if 'M' in mem_str:
                    mem_gb = mem_gb / 1024
                
                mem_info[job_task] = {'cpus': cpus, 'mem_gb': mem_gb}
    except Exception as e:
        if not quiet:
            print(f"  Warning: Could not get sacct data: {e}")
        mem_info = {}
    
    # Map task numbers to sample IDs
    # Job 64031197 tasks 1-629 map to metadata rows 0-628
    # Job 64138622 tasks 1-45 map to failed tasks from first job
    
    # Read failed task list for 128GB job
    try:
        with open('data/validation/oom_failed_tasks.txt') as f:
            oom_task_ids = [int(line.strip()) for line in f]
    except:
        oom_task_ids = []
        if not quiet:
            print("  Warning: Could not load oom_failed_tasks.txt")
    
    # Get file sizes from data/validation/raw
    from pathlib import Path
    
    # Build performance DataFrame
    perf_records = []
    
    for job_data in all_job_data:
        job_id = job_data['job_id']
        task = job_data['task']
        job_task_id = f"{job_id}_{task}"
        
        # Map to metadata index
        if job_id == '64031197':
            meta_idx = task - 1  # Tasks 1-629 → rows 0-628
        elif job_id == '64138622':
            if task <= len(oom_task_ids):
                meta_idx = oom_task_ids[task - 1]
            else:
                continue
        else:
            continue
        
        if meta_idx >= len(metadata):
            continue
        
        sample_id = metadata.iloc[meta_idx]['run_accession']
        
        # Get file size
        sample_dir = Path(f'data/validation/raw/{sample_id}')
        total_size_mb = 0
        if sample_dir.exists():
            for f in sample_dir.rglob('*'):
                if f.is_file():
                    total_size_mb += f.stat().st_size / 1024 / 1024
        
        # Get memory and CPU info
        if job_task_id in mem_info:
            threads = mem_info[job_task_id]['cpus']
            mem_allocated_gb = mem_info[job_task_id]['mem_gb']
        else:
            threads = 6  # default
            mem_allocated_gb = 64 if job_id == '64031197' else 128
        
        perf_records.append({
            'sample_id': sample_id,
            'input_size_mb': total_size_mb,
            'mem_allocated_gb': mem_allocated_gb,
            'threads': threads,
            'elapsed_time': job_data['elapsed'],
            'cpu_efficiency': job_data['cpu_eff'],
            'mem_efficiency': job_data['mem_eff'],
            'state': job_data['state']
        })
    
    df_perf = pd.DataFrame(perf_records)
    
    # Remove duplicates (keep latest run, which is 128GB for retried samples)
    df_perf = df_perf.sort_values('mem_allocated_gb', ascending=False)
    df_perf = df_perf.drop_duplicates(subset=['sample_id'], keep='first')
    
    if not quiet:
        print(f"✓ Loaded performance data for {len(df_perf)} samples")
        print(f"  Successful: {len(df_perf[df_perf['state'] == 'COMPLETED'])}")
        print(f"  Failed: {len(df_perf[df_perf['state'] != 'COMPLETED'])}")
        print()
    
    return df_perf


def plot_performance_metrics(df_perf, output_dir, quiet=False):
    """Generate plots for job performance analysis."""
    if not quiet:
        print("Generating performance plots...")
    
    # Plot 1: Memory efficiency vs file size
    fig = px.scatter(
        df_perf,
        x='input_size_mb',
        y='mem_efficiency',
        color='state',
        size='mem_allocated_gb',
        hover_data=['sample_id', 'elapsed_time', 'threads'],
        title='Memory Efficiency vs Input File Size',
        labels={
            'input_size_mb': 'Input Size (MB)',
            'mem_efficiency': 'Memory Efficiency (%)',
            'state': 'Job State',
            'mem_allocated_gb': 'RAM (GB)'
        },
        color_discrete_map={
            'COMPLETED': '#2E86AB',
            'OUT_OF_MEMORY': '#A6192E',
            'FAILED': '#F77F00'
        }
    )
    
    fig.update_layout(height=600, width=1000)
    fig.write_html(output_dir / "performance_memory_vs_size.html")
    fig.write_image(output_dir / "performance_memory_vs_size.png", width=1000, height=600)
    if not quiet:
        print(f"  ✓ performance_memory_vs_size.png")
    
    # Plot 2: Runtime distribution by success/failure
    fig = px.box(
        df_perf[df_perf['state'].isin(['COMPLETED', 'OUT_OF_MEMORY'])],
        x='state',
        y='elapsed_time',
        color='state',
        points='all',
        title='Job Runtime Distribution',
        labels={
            'elapsed_time': 'Elapsed Time (HH:MM:SS)',
            'state': 'Job State'
        },
        color_discrete_map={
            'COMPLETED': '#2E86AB',
            'OUT_OF_MEMORY': '#A6192E'
        }
    )
    
    fig.update_layout(height=600, width=700, showlegend=False)
    fig.write_html(output_dir / "performance_runtime_distribution.html")
    fig.write_image(output_dir / "performance_runtime_distribution.png", width=700, height=600)
    if not quiet:
        print(f"  ✓ performance_runtime_distribution.png")
    
    # Plot 3: CPU efficiency histogram
    fig = px.histogram(
        df_perf[df_perf['state'] == 'COMPLETED'],
        x='cpu_efficiency',
        nbins=30,
        title='CPU Efficiency Distribution (Successful Jobs)',
        labels={'cpu_efficiency': 'CPU Efficiency (%)'},
        color_discrete_sequence=['#6A994E']
    )
    
    fig.update_layout(height=500, width=800, showlegend=False)
    fig.write_html(output_dir / "performance_cpu_efficiency.html")
    fig.write_image(output_dir / "performance_cpu_efficiency.png", width=800, height=500)
    if not quiet:
        print(f"  ✓ performance_cpu_efficiency.png")
    
    # Plot 4: Memory allocated vs actual usage (successful jobs only)
    df_success = df_perf[df_perf['state'] == 'COMPLETED'].copy()
    df_success['mem_used_gb'] = df_success['mem_allocated_gb'] * df_success['mem_efficiency'] / 100
    
    fig = px.scatter(
        df_success,
        x='mem_allocated_gb',
        y='mem_used_gb',
        hover_data=['sample_id', 'input_size_mb', 'mem_efficiency'],
        title='Allocated vs Actual Memory Usage (Successful Jobs)',
        labels={
            'mem_allocated_gb': 'Allocated RAM (GB)',
            'mem_used_gb': 'Actual Usage (GB)'
        },
        color_discrete_sequence=['#2E86AB']
    )
    
    # Add diagonal line (perfect usage)
    max_mem = df_success['mem_allocated_gb'].max()
    fig.add_trace(
        go.Scatter(
            x=[0, max_mem],
            y=[0, max_mem],
            mode='lines',
            line=dict(color='gray', dash='dash'),
            name='100% efficiency',
            showlegend=True
        )
    )
    
    fig.update_layout(height=600, width=800)
    fig.write_html(output_dir / "performance_memory_usage.html")
    fig.write_image(output_dir / "performance_memory_usage.png", width=800, height=600)
    if not quiet:
        print(f"  ✓ performance_memory_usage.png")


def save_performance_table(df_perf, output_dir, quiet=False):
    """Save job performance table."""
    # Reorder and format columns
    df_export = df_perf[[
        'sample_id', 'input_size_mb', 'mem_allocated_gb', 'threads',
        'elapsed_time', 'cpu_efficiency', 'mem_efficiency', 'state'
    ]].copy()
    
    df_export = df_export.rename(columns={
        'sample_id': 'Sample ID',
        'input_size_mb': 'Input Size (MB)',
        'mem_allocated_gb': 'RAM Allocated (GB)',
        'threads': 'Threads',
        'elapsed_time': 'Runtime',
        'cpu_efficiency': 'CPU Efficiency (%)',
        'mem_efficiency': 'Memory Efficiency (%)',
        'state': 'State'
    })
    
    # Sort by input size
    df_export = df_export.sort_values('Input Size (MB)', ascending=False)
    
    # Save
    tsv_file = output_dir / "validation_job_performance.tsv"
    html_file = output_dir / "validation_job_performance.html"
    
    df_export.to_csv(tsv_file, sep='\t', index=False)
    df_export.to_html(html_file, index=False)
    
    if not quiet:
        print(f"  ✓ validation_job_performance.tsv")
        print(f"  ✓ validation_job_performance.html")


# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def main():
    """Main execution workflow."""
    # Parse arguments
    args = parse_arguments()
    
    # Setup output directories
    figures_dir = Path(args.output_dir) / "figures" / "validation"
    tables_dir = Path(args.output_dir) / "tables" / "validation"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all data into master DataFrame
    df, label_encoders, metadata = load_data(
        args.metadata,
        args.predictions_dir,
        args.label_encoders,
        quiet=args.quiet
    )
    
    # Calculate and print summary metrics
    summary_df = calculate_summary_metrics(df, quiet=args.quiet)
    
    # Analyze unseen labels
    analyze_unseen_labels(df, quiet=args.quiet)
    
    # Generate all figures
    if not args.quiet:
        print("="*80)
        print("GENERATING FIGURES")
        print("="*80)
        print()
    
    for task in TASKS:
        if not args.quiet:
            task_title = task.replace('_', ' ').title()
            print(f"{task_title}:")
        plot_confusion_matrix(df, task, figures_dir, quiet=args.quiet)
        plot_per_class_performance(df, task, figures_dir, quiet=args.quiet)
        plot_accuracy_comparison(df, task, figures_dir, quiet=args.quiet)
        plot_confidence_distribution(df, task, figures_dir, quiet=args.quiet)
        plot_roc_pr_curves(df, task, label_encoders, figures_dir, quiet=args.quiet)
        if not args.quiet:
            print()
    
    # Save tables
    if not args.quiet:
        print("="*80)
    save_tables(df, summary_df, tables_dir, quiet=args.quiet)
    
    # Job performance analysis
    if not args.quiet:
        print()
        print("="*80)
    
    df_perf = load_job_performance_data(metadata, quiet=args.quiet)
    plot_performance_metrics(df_perf, figures_dir, quiet=args.quiet)
    save_performance_table(df_perf, tables_dir, quiet=args.quiet)
    
    # Summary
    if not args.quiet:
        print()
        print("="*80)
        print("SUMMARY")
        print("="*80)
        print(f"Figures saved to: {figures_dir}")
        print(f"Tables saved to:  {tables_dir}")
        print(f"Total samples analyzed: {len(df['sample_id'].unique())}")
        print(f"Total predictions: {len(df)}")
        print(f"Job performance records: {len(df_perf)}")
        print("="*80)


if __name__ == "__main__":
    main()
