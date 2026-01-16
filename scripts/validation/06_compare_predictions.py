#!/usr/bin/env python3
"""
Compare validation predictions to true labels and generate comprehensive analysis.

This script loads prediction results, compares them to ground truth metadata,
and generates publication-ready figures and tables for model validation.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)
from sklearn.preprocessing import label_binarize

# ============================================================================
# CONFIGURATION
# ============================================================================

SAMPLE_TYPE_MAP = {
    'ancient_metagenome': 'ancient',
    'modern_metagenome': 'modern'
}

TASKS = ['sample_type', 'community_type', 'sample_host', 'material']

# Positive class for binary classification (for ROC/PR curves)
BINARY_POSITIVE_CLASS = {'sample_type': 'modern'}

# Centralized plotting configuration
PLOT_CONFIG = {
    'colors': {
        'correct_seen': '#2E86AB',
        'incorrect_seen': '#A6192E',
        'correct_unseen': '#6A994E',
        'incorrect_unseen': '#F77F00',
        'palette': px.colors.qualitative.Set2
    },
    'template': 'plotly_white',
    'font_size': 12,
    'confusion_matrix': {
        'colorscale': 'Blues',
        'text_size': 8
    },
    'sizes': {
        'default_width': 800,
        'default_height': 600,
        'confusion_min': 600
    }
}


# ============================================================================
# ARGUMENT PARSING
# ============================================================================

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare validation predictions to ground truth"
    )
    parser.add_argument(
        '--metadata', type=str,
        default='data/validation/validation_metadata_expanded.tsv',
        help='Path to validation metadata TSV'
    )
    parser.add_argument(
        '--predictions-dir', type=str,
        default='results/validation_predictions',
        help='Directory containing prediction JSON files'
    )
    parser.add_argument(
        '--label-encoders', type=str,
        default='results/full_training/label_encoders.json',
        help='Path to label encoders JSON'
    )
    parser.add_argument(
        '--output-dir', type=str, default='paper',
        help='Base output directory'
    )
    parser.add_argument(
        '--quiet', action='store_true',
        help='Suppress verbose output'
    )
    return parser.parse_args()


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(
    metadata_file: str,
    predictions_dir: str,
    label_encoders_file: str,
    quiet: bool = False
) -> Tuple[pd.DataFrame, Dict, pd.DataFrame]:
    """
    Load predictions and metadata into a master DataFrame.
    
    Returns:
        - df: Master DataFrame with one row per (sample, task) pair
        - label_encoders: Dictionary of label encoders per task
        - metadata: Original metadata DataFrame
    """
    # Load label encoders
    with open(label_encoders_file) as f:
        label_encoders = json.load(f)
    
    # Load metadata
    metadata = pd.read_csv(metadata_file, sep='\t')
    
    # Find prediction files
    predictions_dir = Path(predictions_dir)
    prediction_files = list(predictions_dir.glob("*/*_predictions.json"))
    
    if not quiet:
        print(f"Loading {len(prediction_files)} predictions...")
    
    # Build records
    records = []
    for pred_file in prediction_files:
        sample_id = pred_file.parent.name
        
        with open(pred_file) as f:
            pred = json.load(f)
        
        sample_meta = metadata[metadata['run_accession'] == sample_id]
        if len(sample_meta) == 0:
            continue
        
        sample_meta = sample_meta.iloc[0]
        
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
            
            # Normalize pred_label for sample_type
            if task_name == 'sample_type' and pred_label in SAMPLE_TYPE_MAP:
                pred_label = SAMPLE_TYPE_MAP[pred_label]
            
            # Check if seen in training
            if task_name == 'sample_type':
                normalized_training = [
                    SAMPLE_TYPE_MAP.get(c, c) 
                    for c in label_encoders[task_name]['classes']
                ]
                is_seen = true_label in normalized_training
            else:
                is_seen = true_label in label_encoders[task_name]['classes']
            
            records.append({
                'sample_id': sample_id,
                'task': task_name,
                'true_label': true_label,
                'pred_label': pred_label,
                'confidence': pred_info['confidence'],
                'is_correct': true_label == pred_label,
                'is_seen': is_seen,
                'probabilities': pred_info.get('probabilities', {})
            })
    
    df = pd.DataFrame(records)
    
    if not quiet:
        print(f"Loaded {len(df)} records from {len(df['sample_id'].unique())} samples\n")
    
    return df, label_encoders, metadata


# ============================================================================
# METRICS CALCULATION
# ============================================================================

def calculate_summary_metrics(df: pd.DataFrame, quiet: bool = False) -> pd.DataFrame:
    """Calculate accuracy metrics per task."""
    summary_data = []
    
    for task in TASKS:
        task_df = df[df['task'] == task].dropna(subset=['true_label', 'pred_label'])
        
        if len(task_df) == 0:
            continue
        
        task_title = task.replace('_', ' ').title()
        
        # Seen-only metrics
        seen_df = task_df[task_df['is_seen']]
        if len(seen_df) > 0:
            summary_data.append({
                'Task': task_title,
                'Subset': 'SEEN ONLY',
                'Correct': seen_df['is_correct'].sum(),
                'Total': len(seen_df),
                'Accuracy (%)': f"{seen_df['is_correct'].mean() * 100:.1f}"
            })
        
        # All samples
        summary_data.append({
            'Task': task_title,
            'Subset': 'ALL SAMPLES',
            'Correct': task_df['is_correct'].sum(),
            'Total': len(task_df),
            'Accuracy (%)': f"{task_df['is_correct'].mean() * 100:.1f}"
        })
    
    return pd.DataFrame(summary_data)


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_confusion_matrix(df: pd.DataFrame, task: str, output_dir: Path, quiet: bool = False) -> None:
    """Generate confusion matrix heatmap using scikit-learn."""
    task_df = df[df['task'] == task].dropna(subset=['true_label', 'pred_label'])
    
    if len(task_df) == 0:
        return
    
    # Use sklearn confusion_matrix
    labels = sorted(task_df['true_label'].unique())
    cm = confusion_matrix(task_df['true_label'], task_df['pred_label'], labels=labels)
    cm_norm = cm / (cm.sum(axis=1, keepdims=True) + 1e-10)
    
    fig = go.Figure(data=go.Heatmap(
        z=cm_norm,
        x=labels,
        y=labels,
        colorscale=PLOT_CONFIG['confusion_matrix']['colorscale'],
        text=cm.astype(int),
        texttemplate='%{text}',
        textfont={"size": PLOT_CONFIG['confusion_matrix']['text_size']},
        colorbar=dict(title="Proportion")
    ))
    
    size = max(PLOT_CONFIG['sizes']['confusion_min'], len(labels) * 50)
    fig.update_layout(
        title=f"{task.replace('_', ' ').title()} - Confusion Matrix (n={len(task_df)})",
        xaxis_title="Predicted",
        yaxis_title="True",
        width=size,
        height=size,
        template=PLOT_CONFIG['template'],
        font=dict(size=PLOT_CONFIG['font_size'])
    )
    
    png_file = output_dir / f"confusion_matrix_{task}.png"
    fig.write_html(str(png_file.with_suffix('.html')))
    fig.write_image(str(png_file), width=size, height=size)
    
    if not quiet:
        print(f"  ✓ {png_file.name}")


def plot_per_class_performance(df: pd.DataFrame, task: str, output_dir: Path, quiet: bool = False) -> None:
    """Generate per-class performance metrics using classification_report."""
    task_df = df[df['task'] == task].dropna(subset=['true_label', 'pred_label'])
    
    if len(task_df) == 0:
        return
    
    # Use sklearn classification_report
    report_dict = classification_report(
        task_df['true_label'],
        task_df['pred_label'],
        output_dict=True,
        zero_division=0
    )
    
    # Convert to DataFrame (exclude accuracy, macro avg, weighted avg)
    class_metrics = []
    for label, metrics in report_dict.items():
        if label not in ['accuracy', 'macro avg', 'weighted avg']:
            class_metrics.append({
                'Class': label,
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1': metrics['f1-score'],
                'Support': metrics['support']
            })
    
    df_metrics = pd.DataFrame(class_metrics)
    df_melted = df_metrics.melt(
        id_vars=['Class'],
        value_vars=['Precision', 'Recall', 'F1'],
        var_name='Metric',
        value_name='Score'
    )
    
    fig = px.bar(
        df_melted,
        x='Class',
        y='Score',
        color='Metric',
        barmode='group',
        title=f"{task.replace('_', ' ').title()} - Per-Class Performance",
        color_discrete_map={
            'Precision': PLOT_CONFIG['colors']['correct_seen'],
            'Recall': '#A23B72',
            'F1': PLOT_CONFIG['colors']['correct_unseen']
        },
        text='Score'
    )
    
    width = max(800, len(df_metrics) * 120)
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig.update_layout(
        yaxis_title="Score",
        yaxis_range=[0, 1.1],
        height=500,
        width=width,
        template=PLOT_CONFIG['template'],
        font=dict(size=PLOT_CONFIG['font_size'])
    )
    
    png_file = output_dir / f"per_class_performance_{task}.png"
    fig.write_html(str(png_file.with_suffix('.html')))
    fig.write_image(str(png_file), width=width, height=500)
    
    if not quiet:
        print(f"  ✓ {png_file.name}")


def plot_roc_pr_curves(
    df: pd.DataFrame,
    task: str,
    label_encoders: Dict,
    output_dir: Path,
    quiet: bool = False
) -> None:
    """Generate ROC and PR curves with explicit positive class for binary tasks."""
    task_df = df[df['task'] == task].dropna(subset=['true_label', 'pred_label'])
    
    if len(task_df) == 0:
        return
    
    # Get classes and normalize for sample_type
    classes = label_encoders[task]['classes']
    if task == 'sample_type':
        classes = [SAMPLE_TYPE_MAP.get(c, c) for c in classes]
    
    n_classes = len(classes)
    
    # Check if we have multiple classes in validation data
    unique_labels = task_df['true_label'].unique()
    if len(unique_labels) < 2:
        return
    
    # Build probability matrix
    y_true_labels = []
    y_scores_matrix = []
    
    for _, row in task_df.iterrows():
        probs = row['probabilities']
        class_probs = []
        for i, cls in enumerate(classes):
            prob = float(probs.get(str(i), probs.get(cls, 0.0)))
            class_probs.append(prob)
        y_true_labels.append(row['true_label'])
        y_scores_matrix.append(class_probs)
    
    y_scores = np.array(y_scores_matrix)
    
    # Binarize labels
    y_true_bin = label_binarize(
        [classes.index(lbl) if lbl in classes else -1 for lbl in y_true_labels],
        classes=list(range(n_classes))
    )
    
    if hasattr(y_true_bin, 'toarray'):
        y_true_bin = y_true_bin.toarray()
    
    # Handle binary classification (sklearn returns shape (n, 1))
    if n_classes == 2 and y_true_bin.shape[1] == 1:
        y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])
    
    if y_true_bin.shape[0] == 0 or np.sum(y_true_bin) == 0:
        return
    
    # For binary tasks, explicitly define positive class
    if task in BINARY_POSITIVE_CLASS:
        positive_class = BINARY_POSITIVE_CLASS[task]
        if positive_class in classes:
            # Reorder so positive class is always last (index 1)
            pos_idx = classes.index(positive_class)
            if pos_idx == 0:
                # Swap columns
                y_true_bin = y_true_bin[:, [1, 0]]
                y_scores = y_scores[:, [1, 0]]
                classes = [classes[1], classes[0]]
    
    # ROC Curves
    fig_roc = go.Figure()
    
    for i in range(n_classes):
        if np.sum(y_true_bin[:, i]) > 0:
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
            roc_auc = auc(fpr, tpr)
            
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f'{classes[i]} (AUC={roc_auc:.3f})',
                line=dict(width=2, color=PLOT_CONFIG['colors']['palette'][i % len(PLOT_CONFIG['colors']['palette'])])
            ))
    
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random',
        line=dict(dash='dash', color='gray', width=1)
    ))
    
    fig_roc.update_layout(
        title=f'ROC Curves - {task.replace("_", " ").title()} (Validation Set)',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        template=PLOT_CONFIG['template'],
        font=dict(size=PLOT_CONFIG['font_size']),
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        legend=dict(x=0.6, y=0.1),
        height=PLOT_CONFIG['sizes']['default_height'],
        width=PLOT_CONFIG['sizes']['default_width']
    )
    
    roc_png = output_dir / f"roc_curves_{task}.png"
    fig_roc.write_html(str(roc_png.with_suffix('.html')))
    fig_roc.write_image(str(roc_png), width=800, height=600)
    
    if not quiet:
        print(f"  ✓ {roc_png.name}")
    
    # PR Curves
    fig_pr = go.Figure()
    
    for i in range(n_classes):
        if np.sum(y_true_bin[:, i]) > 0:
            precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_scores[:, i])
            avg_precision = average_precision_score(y_true_bin[:, i], y_scores[:, i])
            
            fig_pr.add_trace(go.Scatter(
                x=recall, y=precision,
                mode='lines',
                name=f'{classes[i]} (AP={avg_precision:.3f})',
                line=dict(width=2, color=PLOT_CONFIG['colors']['palette'][i % len(PLOT_CONFIG['colors']['palette'])])
            ))
    
    fig_pr.update_layout(
        title=f'Precision-Recall Curves - {task.replace("_", " ").title()} (Validation Set)',
        xaxis_title='Recall',
        yaxis_title='Precision',
        template=PLOT_CONFIG['template'],
        font=dict(size=PLOT_CONFIG['font_size']),
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1.05]),
        legend=dict(x=0.05, y=0.2),
        height=PLOT_CONFIG['sizes']['default_height'],
        width=PLOT_CONFIG['sizes']['default_width']
    )
    
    pr_png = output_dir / f"pr_curves_{task}.png"
    fig_pr.write_html(str(pr_png.with_suffix('.html')))
    fig_pr.write_image(str(pr_png), width=800, height=600)
    
    if not quiet:
        print(f"  ✓ {pr_png.name}")


def plot_confidence_distribution(df: pd.DataFrame, task: str, output_dir: Path, quiet: bool = False) -> None:
    """Plot confidence distribution by correctness and label type."""
    task_df = df[df['task'] == task].dropna(subset=['true_label', 'pred_label']).copy()
    
    if len(task_df) == 0:
        return
    
    task_df['category'] = task_df.apply(
        lambda row: f"{'Correct' if row['is_correct'] else 'Incorrect'} - {'Seen' if row['is_seen'] else 'Unseen'}",
        axis=1
    )
    
    fig = px.box(
        task_df,
        x='category',
        y='confidence',
        color='category',
        points='all',
        title=f"{task.replace('_', ' ').title()} - Confidence Distribution",
        color_discrete_map={
            'Correct - Seen': PLOT_CONFIG['colors']['correct_seen'],
            'Incorrect - Seen': PLOT_CONFIG['colors']['incorrect_seen'],
            'Correct - Unseen': PLOT_CONFIG['colors']['correct_unseen'],
            'Incorrect - Unseen': PLOT_CONFIG['colors']['incorrect_unseen']
        }
    )
    
    fig.update_layout(
        yaxis_title="Confidence",
        xaxis_title="Category",
        yaxis_range=[0, 1.05],
        showlegend=False,
        height=500,
        width=900,
        template=PLOT_CONFIG['template'],
        font=dict(size=PLOT_CONFIG['font_size'])
    )
    
    png_file = output_dir / f"confidence_distribution_{task}.png"
    fig.write_html(str(png_file.with_suffix('.html')))
    fig.write_image(str(png_file), width=900, height=500)
    
    if not quiet:
        print(f"  ✓ {png_file.name}")


def plot_accuracy_comparison(df: pd.DataFrame, task: str, output_dir: Path, quiet: bool = False) -> None:
    """Compare accuracy on SEEN vs ALL labels."""
    task_df = df[df['task'] == task].dropna(subset=['true_label', 'pred_label'])
    
    if len(task_df) == 0:
        return
    
    acc_data = []
    seen_df = task_df[task_df['is_seen']]
    
    if len(seen_df) > 0:
        acc_data.append({
            'Subset': 'SEEN',
            'Accuracy': seen_df['is_correct'].mean() * 100,
            'Samples': len(seen_df)
        })
    
    acc_data.append({
        'Subset': 'ALL',
        'Accuracy': task_df['is_correct'].mean() * 100,
        'Samples': len(task_df)
    })
    
    df_acc = pd.DataFrame(acc_data)
    
    fig = go.Figure(data=[
        go.Bar(
            x=df_acc['Subset'],
            y=df_acc['Accuracy'],
            text=[f"{acc:.1f}%<br>(n={n})" for acc, n in zip(df_acc['Accuracy'], df_acc['Samples'])],
            textposition='outside',
            marker_color=[PLOT_CONFIG['colors']['correct_seen'], '#A23B72']
        )
    ])
    
    fig.update_layout(
        title=f"{task.replace('_', ' ').title()} - Accuracy: Seen vs All",
        yaxis_title="Accuracy (%)",
        yaxis_range=[0, 110],
        showlegend=False,
        height=400,
        width=500,
        template=PLOT_CONFIG['template'],
        font=dict(size=PLOT_CONFIG['font_size'])
    )
    
    png_file = output_dir / f"accuracy_comparison_{task}.png"
    fig.write_html(str(png_file.with_suffix('.html')))
    fig.write_image(str(png_file), width=500, height=400)
    
    if not quiet:
        print(f"  ✓ {png_file.name}")


# ============================================================================
# TABLE GENERATION
# ============================================================================

def save_tables(
    df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_dir: Path,
    quiet: bool = False
) -> None:
    """Save summary tables."""
    # Summary accuracy
    summary_df.to_csv(output_dir / "validation_accuracy_summary.tsv", sep='\t', index=False)
    
    # Unseen predictions
    for task in TASKS:
        task_df = df[df['task'] == task]
        unseen_df = task_df[~task_df['is_seen']]
        
        if len(unseen_df) > 0:
            unseen_export = unseen_df[['sample_id', 'true_label', 'pred_label', 'confidence']]
            unseen_export.to_csv(output_dir / f"unseen_predictions_{task}.tsv", sep='\t', index=False)
    
    # Wrong predictions
    for task in TASKS:
        task_df = df[df['task'] == task]
        wrong_df = task_df[task_df['is_seen'] & ~task_df['is_correct']]
        
        if len(wrong_df) > 0:
            wrong_export = wrong_df[['sample_id', 'true_label', 'pred_label', 'confidence']]
            wrong_export.to_csv(output_dir / f"wrong_predictions_seen_{task}.tsv", sep='\t', index=False)
    
    # Validation comparison JSON
    output_json = {}
    for task in TASKS:
        task_df = df[df['task'] == task]
        output_json[task] = {}
        
        for subset_name, subset_df in [('all', task_df), ('seen', task_df[task_df['is_seen']])]:
            if len(subset_df) > 0:
                cm = {}
                for _, row in subset_df.iterrows():
                    key = f"{row['true_label']}_to_{row['pred_label']}"
                    cm[key] = cm.get(key, 0) + 1
                
                output_json[task][subset_name] = {
                    'accuracy': float(subset_df['is_correct'].mean()),
                    'correct': int(subset_df['is_correct'].sum()),
                    'total': len(subset_df),
                    'confusion_matrix': cm
                }
    
    with open(output_dir / "validation_comparison.json", 'w') as f:
        json.dump(output_json, f, indent=2)
    
    if not quiet:
        print(f"\n  ✓ Saved all tables")


# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def main() -> None:
    """Main execution workflow."""
    args = parse_arguments()
    
    # Setup output directories
    figures_dir = Path(args.output_dir) / "figures" / "validation"
    tables_dir = Path(args.output_dir) / "tables" / "validation"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df, label_encoders, metadata = load_data(
        args.metadata,
        args.predictions_dir,
        args.label_encoders,
        quiet=args.quiet
    )
    
    # Calculate summary metrics
    summary_df = calculate_summary_metrics(df, quiet=args.quiet)
    
    if not args.quiet:
        print(summary_df.to_string(index=False))
        print()
    
    # Generate figures
    if not args.quiet:
        print("Generating figures...")
    
    for task in TASKS:
        if not args.quiet:
            print(f"\n{task.replace('_', ' ').title()}:")
        
        plot_confusion_matrix(df, task, figures_dir, quiet=args.quiet)
        plot_per_class_performance(df, task, figures_dir, quiet=args.quiet)
        plot_accuracy_comparison(df, task, figures_dir, quiet=args.quiet)
        plot_confidence_distribution(df, task, figures_dir, quiet=args.quiet)
        plot_roc_pr_curves(df, task, label_encoders, figures_dir, quiet=args.quiet)
    
    # Save tables
    save_tables(df, summary_df, tables_dir, quiet=args.quiet)
    
    if not args.quiet:
        print(f"\n{'='*80}")
        print(f"Complete! Analyzed {len(df['sample_id'].unique())} samples")
        print(f"Figures: {figures_dir}")
        print(f"Tables: {tables_dir}")
        print(f"{'='*80}")


if __name__ == "__main__":
    main()
