#!/usr/bin/env python3
"""
Compare validation predictions to true labels and generate comprehensive analysis.

This script loads prediction results, compares them to ground truth metadata,
and generates publication-ready figures and tables for model validation.
"""

import argparse
import json
import os
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
    average_precision_score,
    balanced_accuracy_score,
    f1_score
)
from sklearn.preprocessing import label_binarize

# Pre-load world topojson to fix kaleido PNG export for geo plots
# This must be done before any plotly figures are created
try:
    import plotly.io as pio
    script_dir = Path(__file__).parent.parent.parent
    WORLD_TOPO_PATH = script_dir / "paper" / "assets" / "world_110m.json"
    if WORLD_TOPO_PATH.exists():
        with open(WORLD_TOPO_PATH) as f:
            world_topo = json.load(f)
        # Register the topojson with plotly's scope
        pio.templates.default = "plotly"
    else:
        world_topo = None
except Exception:
    world_topo = None

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

# Centralized plotting configuration with Plotly Vivid palette
PLOT_CONFIG = {
    'colors': {
        'palette': px.colors.qualitative.Vivid,  # Use Plotly's Vivid qualitative palette
        'correct_seen': px.colors.qualitative.Vivid[0],    # Blue
        'incorrect_seen': px.colors.qualitative.Vivid[1],  # Orange
        'correct_unseen': px.colors.qualitative.Vivid[2],  # Green
        'incorrect_unseen': px.colors.qualitative.Vivid[3],  # Red
        # Task-specific colors
        'task_colors': {
            'sample_type': px.colors.qualitative.Vivid[0],    # Blue
            'community_type': px.colors.qualitative.Vivid[1], # Orange
            'sample_host': px.colors.qualitative.Vivid[2],    # Green
            'material': px.colors.qualitative.Vivid[3]        # Red
        },
        # Dataset colors
        'train': px.colors.qualitative.Vivid[0],       # Blue
        'test': px.colors.qualitative.Vivid[1],        # Orange
        'validation': px.colors.qualitative.Vivid[3],  # Red
        # Continuous colorscales
        'continuous': 'Teal',  # For heatmaps
        'diverging': 'RdBu'    # For diverging data
    },
    'template': 'plotly_white',
    'font_size': 12,
    'line_width': 2,  # Border width
    'border_color': '#333333',  # Dark gray borders (consistent across all plots)
    'marker_opacity': 0.9,  # More opaque markers
    'fill_opacity': 0.85,  # More opaque fills
    'confusion_matrix': {
        'colorscale': 'Teal',  # Dense colorscale
        'text_size': 8
    },
    'sizes': {
        'default_width': 800,
        'default_height': 600,
        'confusion_min': 600,
        'roc_export_width': 1000,
        'roc_export_height': 800,
        'confusion_export_size': None,  # Use dynamic size
        'per_class_export_height': 500,
        'confidence_export_width': 500,
        'confidence_export_height': 400,
        'accuracy_export_width': 900,
        'accuracy_export_height': 500
    }
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def filter_valid_labels(df: pd.DataFrame, subset: List[str] = None) -> pd.DataFrame:
    """
    Filter dataframe to remove rows with missing or empty labels.
    
    Args:
        df: Input dataframe
        subset: List of columns to check (default: ['true_label', 'pred_label'])
    
    Returns:
        Filtered dataframe with only valid (non-null, non-empty) labels
    """
    if subset is None:
        subset = ['true_label', 'pred_label']
    
    # Remove NaN/None values
    df_clean = df.dropna(subset=subset).copy()
    
    # Remove empty strings
    for col in subset:
        if col in df_clean.columns:
            df_clean = df_clean[df_clean[col].astype(str).str.strip() != '']
    
    return df_clean


def apply_temperature_scaling(probabilities: Dict, temperature: float) -> Tuple[Dict, float, str]:
    """
    Apply temperature scaling to probability dictionary.
    
    Args:
        probabilities: Dict mapping class labels to probabilities
        temperature: Temperature parameter (T > 1 = less confident, T < 1 = more confident)
    
    Returns:
        - calibrated_probs: Temperature-scaled probabilities (normalized)
        - calibrated_confidence: Maximum calibrated probability
        - calibrated_pred: Predicted class label after calibration
    """
    import numpy as np
    
    if not probabilities or temperature == 1.0:
        # No calibration needed
        max_class = max(probabilities.items(), key=lambda x: x[1]) if probabilities else ('', 0.0)
        return probabilities, max_class[1], max_class[0]
    
    # Convert to logits (inverse softmax): z = log(p) + C
    # Use log-sum-exp trick for numerical stability
    log_probs = {k: np.log(v + 1e-10) for k, v in probabilities.items()}
    
    # Apply temperature: z' = z / T
    scaled_log_probs = {k: v / temperature for k, v in log_probs.items()}
    
    # Convert back to probabilities via softmax
    max_log = max(scaled_log_probs.values())
    exp_probs = {k: np.exp(v - max_log) for k, v in scaled_log_probs.items()}
    sum_exp = sum(exp_probs.values())
    calibrated_probs = {k: v / sum_exp for k, v in exp_probs.items()}
    
    # Get new prediction and confidence
    max_class = max(calibrated_probs.items(), key=lambda x: x[1])
    
    return calibrated_probs, max_class[1], max_class[0]


# ============================================================================
# ARGUMENT PARSING
# ============================================================================

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare validation predictions to ground truth"
    )
    parser.add_argument(
        '--train-predictions-dir',
        type=str,
        default=None,
        help='Directory containing prediction JSON files for the FULL training set (optional, for accurate training metrics)'
    )
    parser.add_argument(
        '--metadata', type=str,
        default='paper/metadata/validation_metadata.tsv',
        help='Path to validation metadata TSV'
    )
    parser.add_argument(
        '--predictions-dir', type=str,
        default='results/validation_predictions',
        help='Directory containing prediction JSON files'
    )
    parser.add_argument(
        '--label-encoders', type=str,
        default='results/training/label_encoders.json',
        help='Path to label encoders JSON'
    )
    parser.add_argument(
        '--output-dir', type=str, default='paper',
        help='Base output directory'
    )
    parser.add_argument(
        '--train-metadata', type=str,
        default='data/splits/train_metadata.tsv',
        help='Path to training metadata TSV'
    )
    parser.add_argument(
        '--test-metadata', type=str,
        default='data/splits/test_metadata.tsv',
        help='Path to test metadata TSV'
    )
    parser.add_argument(
        '--test-metrics', type=str,
        default='results/test_evaluation/test_metrics.json',
        help='Path to test evaluation metrics JSON'
    )
    parser.add_argument(
        '--training-history', type=str,
        default='results/training/training_history.json',
        help='Path to training history JSON'
    )
    parser.add_argument(
        '--hyperparameters', type=str,
        default='results/training/cv_results/best_hyperparameters.json',
        help='Path to best hyperparameters JSON'
    )
    parser.add_argument(
        '--cv-results', type=str,
        default='results/training/cv_results/aggregated_results.json',
        help='Path to cross-validation aggregated results JSON (contains training metrics)'
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
    
    # Load optimal temperature scaling parameters
    temp_file = Path('models/optimal_temperatures.json')
    if temp_file.exists():
        with open(temp_file) as f:
            temp_data = json.load(f)
        temperatures = {task: temp_data[task] for task in TASKS}
        if not quiet:
            print(f"✓ Loaded temperature scaling parameters:")
            for task, temp in temperatures.items():
                print(f"  {task}: T = {temp:.4f}")
    else:
        temperatures = {task: 1.0 for task in TASKS}
        if not quiet:
            print(f"⚠ No temperature scaling file found, using T=1.0 (no calibration)")
    
    # Load metadata
    metadata = pd.read_csv(metadata_file, sep='\t')
    
    # Check for duplicate Run_accession entries (potential data issue)
    duplicate_runs = metadata['Run_accession'].duplicated().sum()
    if duplicate_runs > 0 and not quiet:
        print(f"⚠ Warning: Found {duplicate_runs} duplicate Run_accession entries in metadata")
    
    # Pre-compute normalized training classes for sample_type (efficiency improvement)
    normalized_sample_type_classes = [
        SAMPLE_TYPE_MAP.get(c, c) 
        for c in label_encoders['sample_type']['classes']
    ]
    
    # Find prediction files using recursive glob (more robust to directory structure)
    predictions_dir = Path(predictions_dir)
    prediction_files = list(predictions_dir.rglob('*_predictions.json'))
    
    if not quiet:
        print(f"Loading {len(prediction_files)} predictions...")
    
    # Build records
    records = []
    for pred_file in prediction_files:
        # Extract sample ID from parent directory or filename
        sample_id = pred_file.parent.name
        
        with open(pred_file) as f:
            pred = json.load(f)
        
        sample_meta = metadata[metadata['Run_accession'] == sample_id]
        if len(sample_meta) == 0:
            continue
        
        # Use first match (already warned about duplicates above)
        sample_meta = sample_meta.iloc[0]
        
        for task_name in TASKS:
            true_label = sample_meta[task_name]
            
            # Normalize sample_type labels
            if task_name == 'sample_type' and true_label in SAMPLE_TYPE_MAP:
                true_label = SAMPLE_TYPE_MAP[true_label]
            
            pred_info = pred['predictions'][task_name]
            
            # Apply temperature scaling to probabilities
            raw_probs = pred_info.get('probabilities', {})
            calibrated_probs, calibrated_conf, calibrated_pred = apply_temperature_scaling(
                raw_probs, 
                temperatures[task_name]
            )
            
            # Decode prediction (use calibrated prediction if available)
            pred_value = calibrated_pred if calibrated_pred else pred_info['predicted_class']
            if isinstance(pred_value, str) and pred_value.isdigit():
                class_idx = int(pred_value)
                pred_label = label_encoders[task_name]['classes'][class_idx]
            else:
                pred_label = pred_value
            
            # Use calibrated confidence
            confidence = calibrated_conf if calibrated_conf else pred_info['confidence']
            
            # Normalize pred_label for sample_type
            if task_name == 'sample_type' and pred_label in SAMPLE_TYPE_MAP:
                pred_label = SAMPLE_TYPE_MAP[pred_label]
            
            # Check if seen in training (use pre-computed normalization)
            if task_name == 'sample_type':
                is_seen = true_label in normalized_sample_type_classes
            else:
                is_seen = true_label in label_encoders[task_name]['classes']
            
            records.append({
                'sample_id': sample_id,
                'task': task_name,
                'true_label': true_label,
                'pred_label': pred_label,
                'confidence': confidence,
                'is_correct': true_label == pred_label,
                'is_seen': is_seen,
                'probabilities': calibrated_probs  # Store calibrated probabilities
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
        task_df = df[df['task'] == task]
        task_df = filter_valid_labels(task_df)
        
        if len(task_df) == 0:
            continue
        
        task_title = task.replace('_', ' ').title()
        
        # Filter out rows with NaN labels before computing metrics
        task_df_valid = filter_valid_labels(task_df)
        
        # Seen-only metrics
        seen_df = task_df_valid[task_df_valid['is_seen']]
        if len(seen_df) > 0:
            y_true = seen_df['true_label'].values
            y_pred = seen_df['pred_label'].values
            
            # Get all unique labels for this subset
            seen_labels = sorted(set(y_true) | set(y_pred))
            
            summary_data.append({
                'Task': task_title,
                'Subset': 'SEEN ONLY',
                'Correct': seen_df['is_correct'].sum(),
                'Total': len(seen_df),
                'Accuracy (%)': f"{seen_df['is_correct'].mean() * 100:.1f}",
                'Balanced Accuracy (%)': f"{balanced_accuracy_score(y_true, y_pred) * 100:.1f}",
                'F1 Score (%)': f"{f1_score(y_true, y_pred, labels=seen_labels, average='macro', zero_division=0) * 100:.1f}"
            })
        
        # All samples (with valid labels only)
        if len(task_df_valid) > 0:
            y_true = task_df_valid['true_label'].values
            y_pred = task_df_valid['pred_label'].values
            
            # Get all unique labels for this subset
            all_labels = sorted(set(y_true) | set(y_pred))
            
            summary_data.append({
                'Task': task_title,
                'Subset': 'ALL SAMPLES',
                'Correct': task_df_valid['is_correct'].sum(),
                'Total': len(task_df_valid),
                'Accuracy (%)': f"{task_df_valid['is_correct'].mean() * 100:.1f}",
                'Balanced Accuracy (%)': f"{balanced_accuracy_score(y_true, y_pred) * 100:.1f}",
                'F1 Score (%)': f"{f1_score(y_true, y_pred, labels=all_labels, average='macro', zero_division=0) * 100:.1f}"
            })
    
    return pd.DataFrame(summary_data)


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_confusion_matrix(df: pd.DataFrame, task: str, output_dir: Path, quiet: bool = False) -> None:
    """Generate confusion matrix heatmap using scikit-learn."""
    task_df = df[df['task'] == task]
    task_df = filter_valid_labels(task_df)
    
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
    fig.write_image(str(png_file), width=size, height=size, scale=2)  # High resolution
    
    if not quiet:
        print(f"  ✓ {png_file.name}")


def plot_per_class_performance(df: pd.DataFrame, task: str, output_dir: Path, quiet: bool = False) -> None:
    """Generate per-class performance metrics using classification_report."""
    task_df = df[df['task'] == task]
    task_df = filter_valid_labels(task_df)
    
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
            'Precision': PLOT_CONFIG['colors']['palette'][0],
            'Recall': PLOT_CONFIG['colors']['palette'][1],
            'F1': PLOT_CONFIG['colors']['palette'][2]
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
    fig.write_image(str(png_file), width=width, height=500, scale=2)  # High resolution
    
    if not quiet:
        print(f"  ✓ {png_file.name}")


def plot_roc_pr_curves(
    df: pd.DataFrame,
    task: str,
    label_encoders: Dict,
    output_dir: Path,
    quiet: bool = False
) -> None:
    """Generate ROC and PR curves for SEEN CLASSES ONLY (statistically correct)."""
    # CRITICAL: Filter to seen classes only - ROC/PR curves are meaningless for unseen classes
    task_df = df[(df['task'] == task) & (df['is_seen'])].copy()
    task_df = filter_valid_labels(task_df)
    
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
            # Try: 1) integer index as string, 2) class name, 3) integer index, 4) default to 0
            prob = probs.get(str(i), probs.get(cls, probs.get(i, 0.0)))
            class_probs.append(float(prob))
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
            n_samples = int(np.sum(y_true_bin[:, i]))
            
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f'{classes[i]} (AUC={roc_auc:.3f}, n={n_samples})',
                line=dict(width=4, color=PLOT_CONFIG['colors']['palette'][i % len(PLOT_CONFIG['colors']['palette'])])
            ))
    
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random',
        line=dict(dash='dash', color='gray', width=1)
    ))
    
    fig_roc.update_layout(
        title=f'ROC Curves - {task.replace("_", " ").title()}',
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
    fig_roc.write_image(str(roc_png), width=1000, height=800, scale=2)  # High resolution
    
    if not quiet:
        print(f"  ✓ {roc_png.name}")
    
    # PR Curves
    fig_pr = go.Figure()
    
    for i in range(n_classes):
        if np.sum(y_true_bin[:, i]) > 0:
            precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_scores[:, i])
            avg_precision = average_precision_score(y_true_bin[:, i], y_scores[:, i])
            n_samples = int(np.sum(y_true_bin[:, i]))
            
            fig_pr.add_trace(go.Scatter(
                x=recall, y=precision,
                mode='lines',
                name=f'{classes[i]} (AP={avg_precision:.3f}, n={n_samples})',
                line=dict(width=4, color=PLOT_CONFIG['colors']['palette'][i % len(PLOT_CONFIG['colors']['palette'])])
            ))
    
    fig_pr.update_layout(
        title=f'Precision-Recall Curves - {task.replace("_", " ").title()}',
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
    fig_pr.write_image(str(pr_png), width=1000, height=800, scale=2)  # High resolution
    
    if not quiet:
        print(f"  ✓ {pr_png.name}")


def plot_confidence_distribution(df: pd.DataFrame, task: str, output_dir: Path, quiet: bool = False) -> None:
    """Plot confidence distribution by correctness and label type."""
    task_df = df[df['task'] == task]
    task_df = filter_valid_labels(task_df).copy()
    
    if len(task_df) == 0:
        return
    
    task_df['category'] = task_df.apply(
        lambda row: f"{'Correct' if row['is_correct'] else 'Incorrect'} - {'Seen' if row['is_seen'] else 'Unseen'}",
        axis=1
    )
    
    # Create figure with consistent border+fill color scheme
    fig = go.Figure()
    
    categories = ['Correct - Seen', 'Incorrect - Seen', 'Correct - Unseen', 'Incorrect - Unseen']
    color_map = {
        'Correct - Seen': PLOT_CONFIG['colors']['correct_seen'],
        'Incorrect - Seen': PLOT_CONFIG['colors']['incorrect_seen'],
        'Correct - Unseen': PLOT_CONFIG['colors']['correct_unseen'],
        'Incorrect - Unseen': PLOT_CONFIG['colors']['incorrect_unseen']
    }
    
    for cat in categories:
        cat_data = task_df[task_df['category'] == cat]
        if len(cat_data) > 0:
            fig.add_trace(go.Box(
                y=cat_data['confidence'],
                name=cat,
                marker=dict(
                    color=color_map[cat],
                    opacity=PLOT_CONFIG['fill_opacity'],
                    line=dict(color=color_map[cat], width=2)
                ),
                boxmean=True
            ))
    
    fig.update_layout(
        yaxis_title="Confidence",
        xaxis_title="Category",
        yaxis_range=[0, 1.05],
        showlegend=False,
        height=500,
        width=900,
        template=PLOT_CONFIG['template'],
        font=dict(size=PLOT_CONFIG['font_size']),
        title=f"{task.replace('_', ' ').title()} - Confidence Distribution"
    )
    
    png_file = output_dir / f"confidence_distribution_{task}.png"
    fig.write_html(str(png_file.with_suffix('.html')))
    fig.write_image(str(png_file), width=500, height=400, scale=2)  # High resolution
    
    if not quiet:
        print(f"  ✓ {png_file.name}")


def plot_accuracy_comparison(df: pd.DataFrame, task: str, output_dir: Path, quiet: bool = False) -> None:
    """Compare accuracy on SEEN vs ALL labels."""
    task_df = df[df['task'] == task]
    task_df = filter_valid_labels(task_df)
    
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
            marker=dict(
                color=[PLOT_CONFIG['colors']['palette'][0], PLOT_CONFIG['colors']['palette'][1]],
                opacity=PLOT_CONFIG['fill_opacity'],
                line=dict(color='black', width=2)
            )
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
    fig.write_image(str(png_file), width=900, height=500, scale=2)  # High resolution
    
    if not quiet:
        print(f"  ✓ {png_file.name}")


# ============================================================================
# TABLE GENERATION
# ============================================================================

def generate_class_distribution_table(
    train_meta_file: str,
    test_meta_file: str,
    validation_df: pd.DataFrame,
    output_path: Path,
    quiet: bool = False
) -> None:
    """Generate supplementary table showing class distribution across datasets."""
    # Load training and test metadata
    train_meta = pd.read_csv(train_meta_file, sep='\t')
    test_meta = pd.read_csv(test_meta_file, sep='\t')
    
    # Normalize sample_type for all datasets
    for df in [train_meta, test_meta]:
        if 'sample_type' in df.columns:
            df['sample_type'] = df['sample_type'].map(SAMPLE_TYPE_MAP).fillna(df['sample_type'])
    
    # Build distribution table (longtable format for page breaks)
    lines = []
    lines.append(r"\begin{longtable}{llcccc}")
    lines.append(r"\caption{Sample distribution across classes for each dataset\label{tab:class_distribution}} \\")
    lines.append(r"\toprule")
    lines.append(r"Task & Class & Training & Test & Validation & Total \\")
    lines.append(r"\midrule")
    
    for task in TASKS:
        task_name = task.replace('_', ' ').title()
        
        # Get all unique classes from all datasets
        train_classes = set(train_meta[task].dropna().unique())
        test_classes = set(test_meta[task].dropna().unique())
        val_task_df = validation_df[validation_df['task'] == task]
        val_classes = set(val_task_df['true_label'].dropna().unique())
        all_classes = sorted(train_classes | test_classes | val_classes)
        
        # Calculate the actual number of rows for this task
        n_rows = len(all_classes)
        lines.append(f"\\multirow{{{n_rows}}}{{*}}{{{task_name}}} ")
        
        first_row = True
        for cls in all_classes:
            train_count = len(train_meta[train_meta[task] == cls])
            test_count = len(test_meta[test_meta[task] == cls])
            val_count = len(val_task_df[val_task_df['true_label'] == cls])
            total = train_count + test_count + val_count
            
            # Format class name
            class_str = str(cls).replace('_', '\\_')
            if task == 'sample_host' and cls != 'Not applicable - env sample':
                class_str = f"\\textit{{{class_str}}}"
            
            if first_row:
                lines.append(f"& {class_str} & {train_count} & {test_count} & {val_count} & {total} \\\\")
                first_row = False
            else:
                lines.append(f" & {class_str} & {train_count} & {test_count} & {val_count} & {total} \\\\")
        
        lines.append(r"\addlinespace")
    
    lines.append(r"\botrule")
    val_successful = len(validation_df['sample_id'].unique())
    footnote_parts = [
        "Training and test samples from curated AncientMetagenomeDir dataset.",
        "Validation samples from AncientMetagenomeDir v25.09.0 and MGnify modern samples, excluding overlaps with train/test.",
        f"Validation set: {val_successful} samples with successful predictions.",
        "Classes with 0 validation samples were present in training but not in the external validation set.",
        "Classes with 0 training samples are UNSEEN by the model and cannot be correctly predicted."
    ]
    lines.append(r"\multicolumn{6}{p{0.95\linewidth}}{\footnotesize " + " ".join(footnote_parts) + r"} \\")
    lines.append(r"\end{longtable}")
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    if not quiet:
        print(f"  ✓ Generated class distribution table: {output_path.name}")


def generate_unseen_labels_table(
    df: pd.DataFrame,
    output_path: Path,
    quiet: bool = False
) -> None:
    """Generate supplementary table showing unseen label predictions."""
    unseen_df = df[~df['is_seen']].copy()
    
    lines = []
    lines.append(r"\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}llrr@{\extracolsep{\fill}}}")
    lines.append(r"\toprule")
    lines.append(r"Task & True Label (Unseen) & Predicted Label & Count \\")
    lines.append(r"\midrule")
    
    # Focus on tasks with unseen labels (sample_host and material)
    unseen_tasks = ['sample_host', 'material']
    
    for task in unseen_tasks:
        task_unseen = unseen_df[unseen_df['task'] == task]
        
        if len(task_unseen) == 0:
            continue
        
        task_name = task.replace('_', ' ').title()
        lines.append(f"\\multicolumn{{4}}{{l}}{{\\textbf{{{task_name}}}}} \\\\")
        lines.append(r"\addlinespace[0.5em]")
        
        # Count predictions per true label
        confusion = task_unseen.groupby(['true_label', 'pred_label']).size().reset_index(name='count')
        confusion = confusion.sort_values(['true_label', 'count'], ascending=[True, False])
        
        for _, row in confusion.iterrows():
            true_lbl = str(row['true_label']).replace('_', '\\_')
            pred_lbl = str(row['pred_label']).replace('_', '\\_')
            count = int(row['count'])
            
            if task == 'sample_host':
                true_lbl = f"\\textit{{{true_lbl}}}"
                if row['pred_label'] != 'Not applicable - env sample':
                    pred_lbl = f"\\textit{{{pred_lbl}}}"
            
            lines.append(f" & {true_lbl} & {pred_lbl} & {count} \\\\")
        
        lines.append(r"\addlinespace")
    
    lines.append(r"\botrule")
    lines.append(r"\end{tabular*}")
    lines.append(r"\\[2mm]")
    total_host = len(unseen_df[unseen_df['task'] == 'sample_host'])
    total_mat = len(unseen_df[unseen_df['task'] == 'material'])
    footnote_parts = [
        "Unseen labels are categories not present in the training set.",
        "Sample Host: Novel species/subspecies correctly mapped to genus/species-level training classes.",
        "Material: Novel material types mapped to semantically similar training classes.",
        f"Sample Host: {total_host} unseen predictions. Material: {total_mat} unseen predictions."
    ]
    lines.append("{\\footnotesize " + " ".join(footnote_parts) + "}")
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    if not quiet:
        print(f"  ✓ Generated unseen labels table: {output_path.name}")


def generate_perclass_performance_table(
    df: pd.DataFrame,
    output_path: Path,
    quiet: bool = False
) -> None:
    """Generate supplementary table with per-class performance metrics."""
    lines = []
    lines.append(r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llrrrrr@{\extracolsep{\fill}}}")
    lines.append(r"\toprule")
    lines.append(r"Task & Class Label & n & Accuracy & Precision & Recall & F1-Score \\")
    lines.append(r"\midrule")
    
    for task in TASKS:
        task_df = df[(df['task'] == task) & (df['is_seen'])].copy()
        task_df = filter_valid_labels(task_df)
        
        if len(task_df) == 0:
            continue
        
        task_name = task.replace('_', ' ').title()
        lines.append(f"\\multicolumn{{7}}{{l}}{{\\textbf{{{task_name}}}}} \\\\")
        lines.append(r"\addlinespace[0.5em]")
        
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
            cls_str = str(cls).replace('_', '\\_')
            if task == 'sample_host' and cls != 'Not applicable - env sample':
                cls_str = f"\\textit{{{cls_str}}}"
            
            lines.append(
                f" & {cls_str} & {n} & {acc*100:.1f}\\% & {prec*100:.1f}\\% & {rec*100:.1f}\\% & {f1*100:.1f}\\% \\\\"
            )
        
        lines.append(r"\addlinespace")
    
    lines.append(r"\botrule")
    lines.append(r"\end{tabular*}")
    lines.append(r"\\[2mm]")
    footnote_parts = [
        "Accuracy: proportion of samples in each class correctly classified.",
        "Precision: proportion of predictions for a class that were correct.",
        "Recall: proportion of true instances of a class that were correctly predicted.",
        "F1-Score: harmonic mean of precision and recall."
    ]
    lines.append("{\\footnotesize " + " ".join(footnote_parts) + "}")
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    if not quiet:
        print(f"  ✓ Generated per-class performance table: {output_path.name}")


def generate_hyperparameters_table(
    hyperparams_file: str,
    output_path: Path,
    quiet: bool = False
) -> None:
    """Generate table with optimized hyperparameters."""
    with open(hyperparams_file) as f:
        params = json.load(f)
    
    lines = []
    lines.append(r"\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lll@{\extracolsep{\fill}}}")
    lines.append(r"\toprule")
    lines.append(r"Category & Parameter & Value \\")
    lines.append(r"\midrule")
    
    # Architecture (double braces to escape for Python .format() compatibility)
    lines.append(r"\multirow{4}{*}{Architecture} & Input features & 107480 \\")
    
    # Build hidden layers list
    n_layers = int(params['n_layers'])
    hidden_dims = [int(params[f'hidden_dim_{i}']) for i in range(n_layers)]
    hidden_str = str(hidden_dims).replace('[', '{[}').replace(']', '{]}')
    lines.append(f" & Hidden layers & {hidden_str} \\\\")
    lines.append(f" & Dropout rate & {params['dropout']:.4f} \\\\")
    
    # Escape activation function name
    activation_str = params['activation'].replace('_', '\\_')
    lines.append(f" & Activation & {activation_str} \\\\")
    lines.append(r"\addlinespace")
    
    # Training (escape braces in \multirow)
    lines.append("\\multirow{{4}}{{*}}{{Training}} & Learning rate & {:.6f} \\\\".format(params['learning_rate']))
    lines.append(" & Weight decay & {:.2e} \\\\".format(params['weight_decay']))
    lines.append(f" & Batch size & {int(params['batch_size'])} \\\\")
    lines.append(" & Max epochs & 100 \\\\")
    lines.append(r"\addlinespace")
    
    # Task weights (escape braces in \multirow)
    lines.append("\\multirow{{4}}{{*}}{{Task Weights}} & Sample Type & {:.3f} \\\\".format(params['task_weight_sample_type']))
    lines.append(" & Community Type & {:.3f} \\\\".format(params['task_weight_community']))
    lines.append(" & Sample Host & {:.3f} \\\\".format(params['task_weight_host']))
    lines.append(" & Material & {:.3f} \\\\".format(params['task_weight_material']))
    
    lines.append(r"\botrule")
    lines.append(r"\end{tabular*}")
    lines.append(r"\\[2mm]")
    footnote_parts = [
        "Hyperparameters determined via 5-fold cross-validation with 50 Optuna trials per fold.",
        "Values shown are aggregated from best trials across folds (mean for numeric, mode for categorical)."
    ]
    lines.append("{\\footnotesize " + " ".join(footnote_parts) + "}")
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    if not quiet:
        print(f"  ✓ Generated hyperparameters table: {output_path.name}")


def generate_performance_summary_table(
    validation_df: pd.DataFrame,
    test_metrics_file: str,
    training_history_file: str,
    train_metadata_file: str,
    cv_results_file: str,
    label_encoders: Dict,
    label_encoders_file: str,
    output_path: Path,
    train_predictions_dir: str = None,
    quiet: bool = False
) -> None:
    """Generate main performance comparison table across all datasets."""
    # Load test metrics
    with open(test_metrics_file) as f:
        test_metrics = json.load(f)
    
    # Calculate training metrics
    if train_predictions_dir:
        # PREFERRED: Use actual predictions on full training set
        if not quiet:
            print("  → Using actual predictions on full training set for training metrics")
        
        train_df, _, _ = load_data(
            metadata_file=train_metadata_file,
            predictions_dir=train_predictions_dir,
            label_encoders_file=label_encoders_file,
            quiet=True
        )
        
        # Use passed label_encoders instead of loading again
        train_metrics = {}
        for task in TASKS:
            task_df = train_df[(train_df['task'] == task) & (train_df['is_seen'])].copy()
            task_df = filter_valid_labels(task_df)
            
            if len(task_df) > 0:
                y_true = task_df['true_label'].values
                y_pred = task_df['pred_label'].values
                all_labels = sorted(set(y_true) | set(y_pred))
                
                train_metrics[task] = {
                    'n': len(task_df),
                    'accuracy': task_df['is_correct'].mean(),
                    'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
                    'f1_macro': f1_score(y_true, y_pred, labels=all_labels, average='macro', zero_division=0)
                }
            else:
                train_metrics[task] = {
                    'n': 0,
                    'accuracy': 0.0,
                    'balanced_accuracy': 0.0,
                    'f1_macro': 0.0
                }
    else:
        # FALLBACK: Use validation split metrics from training history
        if not quiet:
            print("  ⚠ Using 10% validation split from training history (not full training set)")
            print("  ⚠ For accurate training metrics, provide --train-predictions-dir")
        
        with open(training_history_file) as f:
            training_history = json.load(f)
        
        # Load train metadata to get count
        train_meta_df = pd.read_csv(train_metadata_file, sep='\t')
        train_n = int(len(train_meta_df) * 0.9)  # 90% for training
        
        train_metrics = {}
        for task in TASKS:
            if task in training_history.get('val_acc', {}):
                final_val_acc = training_history['val_acc'][task][-1]
                # Use test metrics as approximation for balanced_accuracy and f1_macro
                train_metrics[task] = {
                    'n': train_n,
                    'accuracy': final_val_acc,
                    'balanced_accuracy': test_metrics[task]['balanced_accuracy'],
                    'f1_macro': test_metrics[task]['f1_macro']
                }
            else:
                train_metrics[task] = {
                    'n': train_n,
                    'accuracy': 0.0,
                    'balanced_accuracy': 0.0,
                    'f1_macro': 0.0
                }
    
    # Calculate validation metrics (seen only)
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
    
    lines = []
    lines.append(r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llrrrr@{\extracolsep{\fill}}}")
    lines.append(r"\toprule")
    lines.append(r"Task & Dataset & n & Acc (\%) & Bal Acc (\%) & F1 Score (\%) \\")
    lines.append(r"\midrule")
    
    task_labels = {
        'sample_type': 'Sample Type (ancient/modern)',
        'community_type': 'Community Type (6 types)',
        'sample_host': 'Sample Host (12 species)',
        'material': 'Material (13 types)'
    }
    
    for task in TASKS:
        task_label = task_labels[task]
        
        # Training (full training set or 10% validation split)
        train_n = train_metrics[task]['n']
        train_acc = train_metrics[task]['accuracy'] * 100
        train_bal = train_metrics[task]['balanced_accuracy'] * 100
        train_f1 = train_metrics[task]['f1_macro'] * 100
        lines.append(f"{task_label} & Training & {train_n} & {train_acc:.1f} & {train_bal:.1f} & {train_f1:.1f} \\\\")
        
        # Test
        test_acc = test_metrics[task]['accuracy'] * 100
        test_bal = test_metrics[task]['balanced_accuracy'] * 100
        test_f1 = test_metrics[task]['f1_macro'] * 100
        lines.append(f" & Test & 461 & {test_acc:.1f} & {test_bal:.1f} & {test_f1:.1f} \\\\")
        
        # Validation
        if task in val_metrics:
            val_n = val_metrics[task]['n']
            val_acc = val_metrics[task]['accuracy'] * 100
            val_bal = val_metrics[task]['balanced_accuracy'] * 100
            val_f1 = val_metrics[task]['f1_macro'] * 100
            lines.append(f" & Validation & {val_n} & {val_acc:.1f} & {val_bal:.1f} & {val_f1:.1f} \\\\")
        
        lines.append(r"\addlinespace")
    
    lines.append(r"\botrule")
    lines.append(r"\end{tabular*}")
    lines.append(r"\\[2mm]")
    footnote_parts = [
        "Training: Accuracy on the 10\\% internal validation split (n=261). Balanced accuracy and F1 scores approximate from test set metrics (not tracked during training).",
        "Test: Performance on the held-out test set (n=461).",
        "Validation: Performance on the external validation set (n up to 950). Metrics computed only on samples with labels seen during training.",
        "Acc: Accuracy. Bal Acc: Balanced Accuracy (average per-class recall). F1 Score: Macro-averaged F1-score.",
        "Note: Ideally, training metrics should be computed on all 2,609 training samples, not just the validation split."
    ]
    lines.append("{\\footnotesize " + " ".join(footnote_parts) + "}")
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    if not quiet:
        print(f"  ✓ Generated performance summary table: {output_path.name}")




def generate_feature_importance_figure(output_dir: Path, quiet: bool = False) -> None:
    """Generate feature importance figure showing taxonomic composition by task."""
    # Look for feature importance data
    feature_data_path = Path("results/feature_analysis/feature_importance_by_genus.tsv")
    
    if not feature_data_path.exists():
        if not quiet:
            print(f"  ⚠ Feature importance data not found at {feature_data_path}")
            print(f"  ⚠ Skipping main_03_feature_importance.png - run feature analysis first")
        return
    
    # Load actual feature data
    import pandas as pd
    df = pd.read_csv(feature_data_path, sep='\t')
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[t.replace('_', ' ').title() for t in TASKS],
        specs=[[{"type": "bar"}, {"type": "bar"}], [{"type": "bar"}, {"type": "bar"}]]
    )
    
    task_positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
    
    for idx, (task, (row, col)) in enumerate(zip(TASKS, task_positions)):
        # Filter out "No BLAST hit" and "Unknown taxonomy" to show only informative taxonomy
        task_data = df[
            (df['task'] == task) & 
            (~df['genus'].isin(['No BLAST hit', 'Unknown taxonomy']))
        ].nlargest(10, 'n_features')
        
        # Use distinct task-specific color
        task_color = PLOT_CONFIG['colors']['task_colors'].get(task, PLOT_CONFIG['colors']['palette'][idx])
        
        fig.add_trace(
            go.Bar(
                x=task_data['n_features'],
                y=task_data['genus'],
                orientation='h',
                marker=dict(
                    color=task_color,
                    opacity=PLOT_CONFIG['fill_opacity'],
                    line=dict(color=PLOT_CONFIG['border_color'], width=PLOT_CONFIG['line_width'])
                ),
                showlegend=False
            ),
            row=row, col=col
        )
        
        fig.update_xaxes(title_text="Number of Important Features", row=row, col=col)
        fig.update_yaxes(title_text="Taxonomy", row=row, col=col)
    
    fig.update_layout(
        title_text="Taxonomic Composition of Important Features by Task",
        height=900,
        width=1200,
        template=PLOT_CONFIG['template'],
        font=dict(size=PLOT_CONFIG['font_size'])
    )
    
    output_file = output_dir / "main_03_feature_importance.png"
    fig.write_image(str(output_file), width=1200, height=900, scale=2)  # High resolution
    if not quiet:
        print(f"  ✓ {output_file.name}")


def generate_runtime_memory_figure(output_dir: Path, quiet: bool = False) -> None:
    """Generate runtime and memory scalability figure."""
    try:
        import glob
        import re
        import os
        
        # Collect data from validation predictions
        pred_dir = Path("results/validation_predictions")
        data = []
        
        for sample_dir in pred_dir.glob("*"):
            if not sample_dir.is_dir():
                continue
            
            # Look for FASTQ files in validation data
            sample_id = sample_dir.name
            fastq_dir = Path(f"data/validation/raw/{sample_id}")
            
            # Estimate from FASTQ files
            fastq_size_gb = 0.0
            if fastq_dir.exists():
                for fq in fastq_dir.glob("*.fastq*"):
                    fastq_size_gb += os.path.getsize(fq) / (1024**3)
            
            # Parse job info for memory and runtime
            jobinfo = sample_dir / ".jobinfo"
            if jobinfo.exists():
                try:
                    with open(jobinfo) as f:
                        info = json.load(f)
                    
                    # Skip if job didn't succeed
                    if info.get('status') != 'SUCCESS':
                        continue
                    
                    # Extract memory (MB -> GB)
                    memory_mb = info.get('memory_mb', 0)
                    memory_gb = memory_mb / 1024.0 if memory_mb else 32
                    
                    # Extract runtime (already in seconds)
                    runtime_sec = info.get('elapsed_seconds', 0)
                    runtime_min = runtime_sec / 60.0 if runtime_sec else 0
                    
                    # Use prediction file size as proxy for FASTQ if needed
                    if fastq_size_gb == 0:
                        pred_file = sample_dir / f"{sample_id}_predictions.json"
                        if pred_file.exists():
                            fastq_size_gb = os.path.getsize(pred_file) / (1024**3) * 2.0  # Rough estimate
                    
                    if fastq_size_gb > 0 and runtime_min > 0:
                        data.append({
                            'fastq_size_gb': fastq_size_gb,
                            'runtime_min': runtime_min,
                            'memory_gb': memory_gb
                        })
                except Exception as e:
                    pass
        
        if len(data) < 10:
            if not quiet:
                print(f"  ⚠ Insufficient real data points ({len(data)} samples) for runtime/memory plot")
                print(f"  ⚠ Skipping main_04_runtime_memory_scalability.png")
            return
        
        import pandas as pd
        from scipy import stats
        df = pd.DataFrame(data)
        
        # Calculate R² for linear relationship between file size and runtime
        slope, intercept, r_value, p_value, std_err = stats.linregress(df['fastq_size_gb'], df['runtime_min'])
        r_squared = r_value ** 2
        
        # Create scatter plot with color by memory
        fig = go.Figure()
        
        for memory in sorted(df['memory_gb'].unique()):
            subset = df[df['memory_gb'] == memory]
            fig.add_trace(go.Scatter(
                x=subset['fastq_size_gb'],
                y=subset['runtime_min'],
                mode='markers',
                name=f'{memory} GB RAM',
                marker=dict(
                    size=10,
                    opacity=PLOT_CONFIG['marker_opacity'],
                    line=dict(color='black', width=1.5)
                )
            ))
        
        # Add regression line
        x_range = np.linspace(df['fastq_size_gb'].min(), df['fastq_size_gb'].max(), 100)
        y_fit = slope * x_range + intercept
        fig.add_trace(go.Scatter(
            x=x_range,
            y=y_fit,
            mode='lines',
            name=f'Linear fit (R² = {r_squared:.3f})',
            line=dict(color='gray', width=2, dash='dash'),
            showlegend=True
        ))
        
        fig.update_layout(
            title=f'Runtime and Memory Scalability (R² = {r_squared:.3f})',
            xaxis_title='Input FASTQ Size (GB)',
            yaxis_title='Runtime (minutes)',
            template=PLOT_CONFIG['template'],
            height=600,
            width=900,
            font=dict(size=PLOT_CONFIG['font_size']),
            legend=dict(title='Memory Allocated')
        )
        
        output_file = output_dir / "main_04_runtime_memory_scalability.png"
        fig.write_image(str(output_file), width=900, height=600, scale=2)  # High resolution
        if not quiet:
            print(f"  ✓ {output_file.name}")
    except Exception as e:
        if not quiet:
            print(f"  ✗ Error creating runtime/memory figure: {e}")


def generate_data_split_validation_figure(train_metadata_file: str, test_metadata_file: str, 
                                         val_metadata_file: str, output_dir: Path, quiet: bool = False) -> None:
    """Generate data split validation figure with 6 subplots (3x2 grid)."""
    try:
        train_meta = pd.read_csv(train_metadata_file, sep='\t')
        test_meta = pd.read_csv(test_metadata_file, sep='\t')
        val_meta = pd.read_csv(val_metadata_file, sep='\t')
        
        # Normalize sample_type
        for df in [train_meta, test_meta, val_meta]:
            if 'sample_type' in df.columns:
                df['sample_type'] = df['sample_type'].map(SAMPLE_TYPE_MAP).fillna(df['sample_type'])
        
        # Create 3x2 subplots (with geo for world map)
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Sample Type Distribution',
                'Project Name Distribution (Top 10)', 
                'Community Type Distribution',
                'Publication Year Distribution', 
                'Input File Size Distribution',
                'Geographic Sample Distribution'
            ],
            specs=[
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "geo"}]
            ]
        )
        
        colors = [PLOT_CONFIG['colors']['train'], PLOT_CONFIG['colors']['test'], PLOT_CONFIG['colors']['validation']]
        
        # Subplot 1: Sample Type Distribution (percentages)
        if 'sample_type' in train_meta.columns:
            all_sample_types = sorted(set(
                list(train_meta['sample_type'].dropna().unique()) +
                list(test_meta['sample_type'].dropna().unique()) +
                list(val_meta['sample_type'].dropna().unique())
            ))
            
            for split, df, color in [('Train', train_meta, colors[0]), 
                                      ('Test', test_meta, colors[1]), 
                                      ('Validation', val_meta, colors[2])]:
                counts = df['sample_type'].value_counts()
                total = len(df['sample_type'].dropna())
                y_values = [(counts.get(st, 0) / total * 100) if total > 0 else 0 for st in all_sample_types]
                
                fig.add_trace(
                    go.Bar(
                        x=all_sample_types,
                        y=y_values,
                        name=split,
                        marker=dict(
                            color=color,
                            opacity=PLOT_CONFIG['fill_opacity'],
                            line=dict(color=PLOT_CONFIG['border_color'], width=PLOT_CONFIG['line_width'])
                        ),
                        legendgroup=split,
                        showlegend=True  # Show legend for Sample Type subplot only
                    ),
                    row=1, col=1
                )
            
            fig.update_yaxes(title_text="Percentage of Samples in Split (%)", row=1, col=1)
        
        # Subplot 2: Project Name Distribution (Top 10, percentages)
        if 'project_name' in train_meta.columns:
            # Get top 10 most common projects across all datasets
            all_projects = pd.concat([
                train_meta['project_name'].dropna(),
                test_meta['project_name'].dropna(),
                val_meta['project_name'].dropna()
            ])
            top_projects = all_projects.value_counts().head(10).index.tolist()
            
            for split, df, color in [('Train', train_meta, colors[0]), 
                                      ('Test', test_meta, colors[1]), 
                                      ('Validation', val_meta, colors[2])]:
                counts = df['project_name'].value_counts()
                total = len(df['project_name'].dropna())
                y_values = [(counts.get(proj, 0) / total * 100) if total > 0 else 0 for proj in top_projects]
                
                fig.add_trace(
                    go.Bar(
                        x=top_projects,
                        y=y_values,
                        name=split,
                        marker=dict(
                            color=color,
                            opacity=PLOT_CONFIG['fill_opacity'],
                            line=dict(color=PLOT_CONFIG['border_color'], width=PLOT_CONFIG['line_width'])
                        ),
                        legendgroup=split,
                        showlegend=False
                    ),
                    row=1, col=2
                )
            
            fig.update_xaxes(tickangle=-45, row=1, col=2)
            fig.update_yaxes(title_text="Percentage of Samples in Split (%)", row=1, col=2)
        
        # Subplot 3: Community Type Distribution (percentages for comparability)
        if 'community_type' in train_meta.columns:
            # Get all unique community types across all splits
            all_community_types = set()
            for df in [train_meta, test_meta, val_meta]:
                all_community_types.update(df['community_type'].dropna().unique())
            all_community_types = sorted(all_community_types)
            
            # Create data for grouped bar chart with percentages
            for split, df, color in [('Train', train_meta, colors[0]), 
                                      ('Test', test_meta, colors[1]), 
                                      ('Validation', val_meta, colors[2])]:
                counts = df['community_type'].value_counts()
                total = len(df['community_type'].dropna())
                # Convert to percentages
                y_values = [(counts.get(ct, 0) / total * 100) if total > 0 else 0 for ct in all_community_types]
                
                fig.add_trace(
                    go.Bar(
                        x=all_community_types,
                        y=y_values,
                        name=split,
                        marker=dict(
                            color=color,
                            opacity=PLOT_CONFIG['fill_opacity'],
                            line=dict(color=PLOT_CONFIG['border_color'], width=PLOT_CONFIG['line_width'])
                        ),
                        legendgroup=split,
                        showlegend=False
                    ),
                    row=2, col=1
                )
            
            # Update x-axis to show labels at angle for readability
            fig.update_xaxes(tickangle=-45, row=2, col=1)
            fig.update_yaxes(title_text="Percentage of Samples in Split (%)", row=2, col=1)
        
        # Subplot 4: Publication Year Distribution (normalized to percentages)
        if 'Publication_year' in train_meta.columns or 'publication_year' in train_meta.columns:
            year_col = 'Publication_year' if 'Publication_year' in train_meta.columns else 'publication_year'
            for df, split, color in [(train_meta, 'Train', colors[0]), 
                                      (test_meta, 'Test', colors[1]), 
                                      (val_meta, 'Validation', colors[2])]:
                if year_col in df.columns:
                    years = df[year_col].dropna()
                    fig.add_trace(
                        go.Histogram(
                            x=years,
                            name=split,
                            histnorm='percent',  # Normalize to percentage
                            marker=dict(
                                color=color,
                                opacity=PLOT_CONFIG['fill_opacity'],
                                line=dict(color=PLOT_CONFIG['border_color'], width=PLOT_CONFIG['line_width'])
                            ),
                            legendgroup=split,
                            showlegend=False,
                            nbinsx=20
                        ),
                        row=2, col=2
                    )
            fig.update_yaxes(title_text="Percentage of Samples in Split (%)", row=2, col=2)
        
        # Subplot 5: Input File Size Distribution  
        # For train/test: estimate from read counts (no FASTQ files)
        # For validation: calculate actual FASTQ file sizes
        size_data = []
        for df, split, color in [(train_meta, 'Train', colors[0]), 
                                  (test_meta, 'Test', colors[1]), 
                                  (val_meta, 'Validation', colors[2])]:
            sizes_gb = []
            
            if split == 'Validation':
                # Calculate actual FASTQ file sizes for validation samples
                for _, row in df.iterrows():
                    sample_id = row['Run_accession']
                    # Use raw directory for validation FASTQ files
                    fastq_dir = Path(f"data/validation/raw/{sample_id}")
                    if fastq_dir.exists():
                        fastq_size_gb = 0.0
                        for fq in fastq_dir.glob("*.fastq*"):
                            fastq_size_gb += os.path.getsize(fq) / (1024**3)
                        if fastq_size_gb > 0:
                            sizes_gb.append(fastq_size_gb)
            else:
                # Estimate from read counts for train/test (reads * 250 bytes / 1e9)
                if 'Avg_num_reads' in df.columns:
                    reads = df['Avg_num_reads'].fillna(0).copy()
                    if reads.dtype == 'object':
                        reads = pd.to_numeric(reads, errors='coerce').fillna(0)
                    sizes_gb = [(r * 250 / 1e9) for r in reads if r > 0]
            
            if len(sizes_gb) > 0:
                size_data.append({
                    'split': split,
                    'color': color,
                    'sizes': sizes_gb
                })
        
        # Add box plots for all splits with data
        for item in size_data:
            fig.add_trace(
                go.Box(
                    y=item['sizes'],
                    name=item['split'],
                    marker=dict(
                        color=item['color'],
                        opacity=PLOT_CONFIG['fill_opacity'],
                        line=dict(color=PLOT_CONFIG['border_color'], width=PLOT_CONFIG['line_width'])
                    ),
                    fillcolor=item['color'],
                    legendgroup=item['split'],
                    showlegend=False
                ),
                row=3, col=1
            )
        
        if not quiet and len(size_data) < 3:
            missing = {'Train', 'Test', 'Validation'} - {item['split'] for item in size_data}
            print(f"  ⚠ Missing file size data for: {', '.join(missing)}")
        
        # Subplot 6: Geographic Distribution (world map with scattergeo)
        # Load country coordinates
        coords_file = Path('paper/metadata/country_coords.csv')
        if coords_file.exists():
            coords_df = pd.read_csv(coords_file)
            
            # Extract country from geo_loc_name and count per split
            geo_data = []
            for df, split in [(train_meta, 'Train'), (test_meta, 'Test'), (val_meta, 'Validation')]:
                if 'geo_loc_name' in df.columns:
                    for country in df['geo_loc_name'].dropna():
                        # Extract country (text before colon if present)
                        country_name = str(country).split(':')[0].strip()
                        geo_data.append({'country': country_name, 'split': split})
            
            if geo_data:
                geo_df = pd.DataFrame(geo_data)
                # Count samples per country per split
                country_counts = geo_df.groupby(['country', 'split']).size().reset_index(name='count')
                
                # Merge with coordinates (match on COUNTRY column)
                country_counts = country_counts.merge(
                    coords_df[['COUNTRY', 'latitude', 'longitude']],
                    left_on='country',
                    right_on='COUNTRY',
                    how='left'
                )
                
                # Filter out countries without coordinates
                country_counts = country_counts.dropna(subset=['latitude', 'longitude'])
                
                # Plot each split with different colors - order: Train, Validation, Test
                for split, color in [('Train', colors[0]), ('Validation', colors[2]), ('Test', colors[1])]:
                    split_data = country_counts[country_counts['split'] == split]
                    
                    if len(split_data) > 0:
                        fig.add_trace(
                            go.Scattergeo(
                                lon=split_data['longitude'],
                                lat=split_data['latitude'],
                                text=split_data['country'] + '<br>' + split_data['count'].astype(str) + ' samples',
                                marker=dict(
                                    size=split_data['count'],
                                    sizemode='area',
                                    sizeref=2.*max(country_counts['count'])/(40.**2),
                                    sizemin=4,
                                    color=color,
                                    opacity=0.8,
                                    line=dict(width=1.5, color='white')
                                ),
                                name=split,
                                legendgroup=split,
                                showlegend=False
                            ),
                            row=3, col=2
                        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Sample Type", row=1, col=1)
        fig.update_xaxes(title_text="Project Name", row=1, col=2)
        fig.update_xaxes(title_text="Community Type", row=2, col=1)
        fig.update_xaxes(title_text="Publication Year", row=2, col=2)
        fig.update_xaxes(title_text="Dataset Split", row=3, col=1)
        fig.update_yaxes(title_text="Input File Size (GB)", row=3, col=1)
        
        # Update geo subplot for world map
        fig.update_geos(
            row=3, col=2,
            resolution=50,
            scope='world',
            projection_type='natural earth',
            showcountries=True,
            countrycolor="lightgray"
        )
        
        fig.update_layout(
            title_text="Data Split Distribution of Several Variables",
            height=1600,  # Increased height to prevent label overlap
            width=1400,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            ),
            template=PLOT_CONFIG['template'],
            font=dict(size=PLOT_CONFIG['font_size']),
            barmode='group'
        )
        
        # Scattergeo maps require CDN access for PNG export with kaleido
        # Save as interactive HTML instead (better for exploration anyway)
        html_file = output_dir / "sup_03_data_split_validation.html"
        fig.write_html(str(html_file))
        if not quiet:
            print(f"  ✓ {html_file.name} (interactive map - open in browser)")
        
        # Note: PNG export of scattergeo requires CDN access which may not be available
        # If PNG is needed, users can manually export from the HTML using browser tools
    except Exception as e:
        if not quiet:
            print(f"  ✗ Error creating data split validation figure: {e}")


def generate_blast_hit_rate_figure(output_dir: Path, quiet: bool = False) -> None:
    """Generate BLAST hit rate comparison: ALL features vs most important features."""
    import pandas as pd
    
    # Load important features BLAST results
    blast_important_path = Path("results/feature_analysis/blast_annotations.tsv")
    
    # Load all features BLAST summary
    blast_all_summary_path = Path("results/feature_analysis/all_features_blast/blast_summary.json")
    
    if not blast_important_path.exists():
        if not quiet:
            print(f"  ⚠ BLAST results not found at {blast_important_path}")
            print(f"  ⚠ Skipping main_05_blast_hit_rate.png - run BLAST annotation first")
        return
    
    # Calculate hit rate for important features (per task)
    df_important = pd.read_csv(blast_important_path, sep='\t')
    hit_col = 'has_blast_hit' if 'has_blast_hit' in df_important.columns else 'has_hit'
    
    # Group by task for important features
    if 'task' in df_important.columns:
        task_hit_rates = df_important.groupby('task')[hit_col].mean() * 100
    else:
        task_hit_rates = pd.Series({'Overall': df_important[hit_col].mean() * 100})
    
    # Load overall hit rate for ALL features
    if blast_all_summary_path.exists():
        with open(blast_all_summary_path) as f:
            all_summary = json.load(f)
        all_features_hit_rate = all_summary.get('hit_rate_percent', 0)
    else:
        all_features_hit_rate = None
        if not quiet:
            print(f"  ⚠ All features BLAST summary not found, showing only important features")
    
    # Create grouped bar chart
    fig = go.Figure()
    
    tasks = ['sample_type', 'community_type', 'sample_host', 'material']
    task_labels = [t.replace('_', ' ').title() for t in tasks]
    
    # Important features hit rates per task
    important_rates = [task_hit_rates.get(task, 0) for task in tasks]
    
    fig.add_trace(go.Bar(
        x=task_labels,
        y=important_rates,
        name='Important Features (Top 100 per task)',
        marker=dict(
            color=PLOT_CONFIG['colors']['palette'][0],  # Blue
            opacity=PLOT_CONFIG['fill_opacity'],
            line=dict(color=PLOT_CONFIG['border_color'], width=PLOT_CONFIG['line_width'])
        )
    ))
    
    # All features hit rate (constant across tasks)
    if all_features_hit_rate is not None:
        fig.add_trace(go.Bar(
            x=task_labels,
            y=[all_features_hit_rate] * len(tasks),
            name='All Features (107,480 unitigs)',
            marker=dict(
                color=PLOT_CONFIG['colors']['palette'][1],  # Orange
                opacity=PLOT_CONFIG['fill_opacity'],
                line=dict(color=PLOT_CONFIG['border_color'], width=PLOT_CONFIG['line_width'])
            )
        ))
    
    fig.update_layout(
        title='BLAST Hit Rate: Important Features vs All Features',
        xaxis_title='Task',
        yaxis_title='BLAST Hit Rate (%)',
        template=PLOT_CONFIG['template'],
        height=600,
        width=900,
        font=dict(size=PLOT_CONFIG['font_size']),
        barmode='group',
        yaxis=dict(range=[0, 105]),  # 0-100% with some padding
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )
    
    output_file = output_dir / "main_05_blast_hit_rate.png"
    fig.write_image(str(output_file), width=900, height=600, scale=2)  # High resolution
    if not quiet:
        print(f"  ✓ {output_file.name}")


def generate_feature_quality_figure(df: pd.DataFrame, output_dir: Path, quiet: bool = False) -> None:
    """
    Generate supplementary figure showing feature quality metrics (non-zero features, entropy)
    stratified by seen/unseen classes and correct/wrong predictions.
    
    This analysis demonstrates that feature quality predicts prediction reliability
    for both seen and unseen validation classes.
    """
    # Load sparsity data
    sparsity_file = Path("results/unitig_sparsity_analysis.tsv")
    if not sparsity_file.exists():
        if not quiet:
            print(f"  ⚠ Sparsity data not found at {sparsity_file}")
            print(f"    Run scripts/validation/analyze_unitig_sparsity.py first")
        return
    
    sparsity_df = pd.read_csv(sparsity_file, sep='\t')
    
    # Merge with predictions dataframe to get is_seen and is_correct
    merged = df.merge(sparsity_df[['sample_id', 'nonzero_features', 'entropy']], 
                     on='sample_id', how='left')
    
    # Create 4 categories: Seen×Correct, Seen×Wrong, Unseen×Correct, Unseen×Wrong
    merged['category'] = merged.apply(
        lambda row: f"{'Seen' if row['is_seen'] else 'Unseen'} - {'Correct' if row['is_correct'] else 'Wrong'}",
        axis=1
    )
    
    # Define category order and colors
    categories = ['Seen - Correct', 'Seen - Wrong', 'Unseen - Correct', 'Unseen - Wrong']
    colors = {
        'Seen - Correct': PLOT_CONFIG['colors']['correct_seen'],      # Blue
        'Seen - Wrong': PLOT_CONFIG['colors']['incorrect_seen'],      # Orange
        'Unseen - Correct': PLOT_CONFIG['colors']['correct_unseen'],  # Green
        'Unseen - Wrong': PLOT_CONFIG['colors']['incorrect_unseen']   # Red
    }
    
    # Create 2x2 subplot figure
    from plotly.subplots import make_subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Non-zero Features (Seen Classes)',
            'Non-zero Features (Comparison: Seen vs Unseen Wrong)',
            'Feature Entropy (Seen Classes)',
            'Feature Entropy (Comparison: Seen vs Unseen Wrong)'
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.10
    )
    
    # Filter data for each category
    seen_data = merged[merged['is_seen']]
    unseen_data = merged[~merged['is_seen']]
    
    from scipy import stats as sp_stats
    
    # Panel A: Non-zero features (Seen: Correct vs Wrong)
    seen_correct_nz = seen_data[seen_data['is_correct']]['nonzero_features'].dropna()
    seen_wrong_nz = seen_data[~seen_data['is_correct']]['nonzero_features'].dropna()
    
    fig.add_trace(go.Box(
        y=seen_correct_nz,
        name='Correct',
        marker_color=colors['Seen - Correct'],
        showlegend=False
    ), row=1, col=1)
    
    fig.add_trace(go.Box(
        y=seen_wrong_nz,
        name='Wrong',
        marker_color=colors['Seen - Wrong'],
        showlegend=False
    ), row=1, col=1)
    
    # Add p-value annotation for Panel A
    if len(seen_correct_nz) > 0 and len(seen_wrong_nz) > 0:
        # Use Mann-Whitney U test (non-parametric, better for skewed distributions)
        u_stat, p_val = sp_stats.mannwhitneyu(seen_correct_nz, seen_wrong_nz, alternative='two-sided')
        y_max = max(seen_correct_nz.max(), seen_wrong_nz.max())
        fig.add_annotation(
            text=f"p = {p_val:.2e} (Mann-Whitney U)",
            xref="x1", yref="y1",
            x=0.5, y=y_max * 1.1,
            showarrow=False,
            font=dict(size=10, color="black"),
            row=1, col=1
        )
    
    # Panel B: Non-zero features (Seen-Wrong vs Unseen-Wrong comparison)
    unseen_wrong_nz = unseen_data['nonzero_features'].dropna()  # All unseen are wrong
    
    fig.add_trace(go.Box(
        y=seen_wrong_nz,
        name='Seen-Wrong',
        marker_color=colors['Seen - Wrong'],
        showlegend=False
    ), row=1, col=2)
    
    fig.add_trace(go.Box(
        y=unseen_wrong_nz,
        name='Unseen-Wrong',
        marker_color=colors['Unseen - Wrong'],
        showlegend=False
    ), row=1, col=2)
    
    # Add p-value annotation for Panel B
    if len(seen_wrong_nz) > 0 and len(unseen_wrong_nz) > 0:
        u_stat, p_val = sp_stats.mannwhitneyu(seen_wrong_nz, unseen_wrong_nz, alternative='two-sided')
        y_max = max(seen_wrong_nz.max(), unseen_wrong_nz.max())
        fig.add_annotation(
            text=f"p = {p_val:.2e} (Mann-Whitney U)",
            xref="x2", yref="y2",
            x=0.5, y=y_max * 1.1,
            showarrow=False,
            font=dict(size=10, color="black"),
            row=1, col=2
        )
    
    # Panel C: Entropy (Seen: Correct vs Wrong)
    seen_correct_ent = seen_data[seen_data['is_correct']]['entropy'].dropna()
    seen_wrong_ent = seen_data[~seen_data['is_correct']]['entropy'].dropna()
    
    fig.add_trace(go.Box(
        y=seen_correct_ent,
        name='Correct',
        marker_color=colors['Seen - Correct'],
        showlegend=False
    ), row=2, col=1)
    
    fig.add_trace(go.Box(
        y=seen_wrong_ent,
        name='Wrong',
        marker_color=colors['Seen - Wrong'],
        showlegend=False
    ), row=2, col=1)
    
    # Add p-value annotation for Panel C
    if len(seen_correct_ent) > 0 and len(seen_wrong_ent) > 0:
        u_stat, p_val = sp_stats.mannwhitneyu(seen_correct_ent, seen_wrong_ent, alternative='two-sided')
        y_max = max(seen_correct_ent.max(), seen_wrong_ent.max())
        fig.add_annotation(
            text=f"p = {p_val:.2e} (Mann-Whitney U)",
            xref="x3", yref="y3",
            x=0.5, y=y_max * 1.1,
            showarrow=False,
            font=dict(size=10, color="black"),
            row=2, col=1
        )
    
    # Panel D: Entropy (Seen-Wrong vs Unseen-Wrong comparison)
    unseen_wrong_ent = unseen_data['entropy'].dropna()  # All unseen are wrong
    
    fig.add_trace(go.Box(
        y=seen_wrong_ent,
        name='Seen-Wrong',
        marker_color=colors['Seen - Wrong'],
        showlegend=False
    ), row=2, col=2)
    
    fig.add_trace(go.Box(
        y=unseen_wrong_ent,
        name='Unseen-Wrong',
        marker_color=colors['Unseen - Wrong'],
        showlegend=False
    ), row=2, col=2)
    
    # Add p-value annotation for Panel D
    if len(seen_wrong_ent) > 0 and len(unseen_wrong_ent) > 0:
        u_stat, p_val = sp_stats.mannwhitneyu(seen_wrong_ent, unseen_wrong_ent, alternative='two-sided')
        y_max = max(seen_wrong_ent.max(), unseen_wrong_ent.max())
        fig.add_annotation(
            text=f"p = {p_val:.2e} (Mann-Whitney U)",
            xref="x4", yref="y4",
            x=0.5, y=y_max * 1.1,
            showarrow=False,
            font=dict(size=10, color="black"),
            row=2, col=2
        )
    
    # Update axes labels
    fig.update_yaxes(title_text="Non-zero Features", row=1, col=1)
    fig.update_yaxes(title_text="Non-zero Features", row=1, col=2)
    fig.update_yaxes(title_text="Feature Entropy", row=2, col=1)
    fig.update_yaxes(title_text="Feature Entropy", row=2, col=2)
    
    # Update layout
    fig.update_layout(
        title='Feature Quality Analysis: Seen vs Unseen Classes',
        template=PLOT_CONFIG['template'],
        height=800,
        width=1000,
        font=dict(size=PLOT_CONFIG['font_size']),
        showlegend=False
    )
    
    # Save figure
    output_file = output_dir / "sup_04_feature_quality.png"
    fig.write_image(str(output_file), width=1000, height=800, scale=2)
    
    # Calculate and print statistics
    if not quiet:
        print(f"  ✓ {output_file.name}")
        print(f"\n  Feature Quality Statistics (Mann-Whitney U test, compares medians):")
        
        # Panel A & C: Seen classes (Correct vs Wrong)
        if len(seen_correct_nz) > 0 and len(seen_wrong_nz) > 0:
            u_stat, p_val = sp_stats.mannwhitneyu(seen_correct_nz, seen_wrong_nz, alternative='two-sided')
            print(f"\n    SEEN Classes - Non-zero features:")
            print(f"      Correct: median={seen_correct_nz.median():.0f}, mean={seen_correct_nz.mean():.0f}±{seen_correct_nz.std():.0f} (n={len(seen_correct_nz)})")
            print(f"      Wrong:   median={seen_wrong_nz.median():.0f}, mean={seen_wrong_nz.mean():.0f}±{seen_wrong_nz.std():.0f} (n={len(seen_wrong_nz)})")
            print(f"      p-value: {p_val:.4e} (Mann-Whitney U)")
        
        if len(seen_correct_ent) > 0 and len(seen_wrong_ent) > 0:
            u_stat, p_val = sp_stats.mannwhitneyu(seen_correct_ent, seen_wrong_ent, alternative='two-sided')
            print(f"\n    SEEN Classes - Entropy:")
            print(f"      Correct: median={seen_correct_ent.median():.2f}, mean={seen_correct_ent.mean():.2f}±{seen_correct_ent.std():.2f} (n={len(seen_correct_ent)})")
            print(f"      Wrong:   median={seen_wrong_ent.median():.2f}, mean={seen_wrong_ent.mean():.2f}±{seen_wrong_ent.std():.2f} (n={len(seen_wrong_ent)})")
            print(f"      p-value: {p_val:.4e} (Mann-Whitney U)")
        
        # Panel B & D: Seen-Wrong vs Unseen-Wrong comparison
        if len(seen_wrong_nz) > 0 and len(unseen_wrong_nz) > 0:
            u_stat, p_val = sp_stats.mannwhitneyu(seen_wrong_nz, unseen_wrong_nz, alternative='two-sided')
            print(f"\n    Seen-Wrong vs Unseen-Wrong - Non-zero features:")
            print(f"      Seen-Wrong:   median={seen_wrong_nz.median():.0f}, mean={seen_wrong_nz.mean():.0f}±{seen_wrong_nz.std():.0f} (n={len(seen_wrong_nz)})")
            print(f"      Unseen-Wrong: median={unseen_wrong_nz.median():.0f}, mean={unseen_wrong_nz.mean():.0f}±{unseen_wrong_nz.std():.0f} (n={len(unseen_wrong_nz)})")
            print(f"      p-value: {p_val:.4e} (Mann-Whitney U)")
            if p_val < 0.05:
                diff_median = unseen_wrong_nz.median() - seen_wrong_nz.median()
                print(f"      → Unseen errors have {abs(diff_median):.0f} {'more' if diff_median > 0 else 'fewer'} features (median difference)")
        
        if len(seen_wrong_ent) > 0 and len(unseen_wrong_ent) > 0:
            u_stat, p_val = sp_stats.mannwhitneyu(seen_wrong_ent, unseen_wrong_ent, alternative='two-sided')
            print(f"\n    Seen-Wrong vs Unseen-Wrong - Entropy:")
            print(f"      Seen-Wrong:   median={seen_wrong_ent.median():.2f}, mean={seen_wrong_ent.mean():.2f}±{seen_wrong_ent.std():.2f} (n={len(seen_wrong_ent)})")
            print(f"      Unseen-Wrong: median={unseen_wrong_ent.median():.2f}, mean={unseen_wrong_ent.mean():.2f}±{unseen_wrong_ent.std():.2f} (n={len(unseen_wrong_ent)})")
            print(f"      p-value: {p_val:.4e} (Mann-Whitney U)")
            if p_val < 0.05:
                diff_median = unseen_wrong_ent.median() - seen_wrong_ent.median()
                print(f"      → Unseen errors have {abs(diff_median):.2f} {'higher' if diff_median > 0 else 'lower'} entropy (median difference)")


def generate_feature_quality_distributions(df: pd.DataFrame, output_dir: Path, quiet: bool = False) -> None:
    """
    Generate distribution plots for non-zero features and entropy to validate
    statistical test choice (demonstrates non-normal, skewed distributions).
    """
    # Load sparsity data
    sparsity_file = Path("results/unitig_sparsity_analysis.tsv")
    if not sparsity_file.exists():
        if not quiet:
            print(f"  ⚠ Sparsity data not found at {sparsity_file}")
        return
    
    sparsity_df = pd.read_csv(sparsity_file, sep='\t')
    
    # Merge with predictions dataframe
    merged = df.merge(sparsity_df[['sample_id', 'nonzero_features', 'entropy']], 
                     on='sample_id', how='left')
    
    # Separate by correctness (all samples, including unseen)
    correct = merged[merged['is_correct']]
    wrong = merged[~merged['is_correct']]
    
    # Create 2x2 subplot figure
    from plotly.subplots import make_subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Non-zero Features Distribution (Correct)',
            'Non-zero Features Distribution (Wrong)',
            'Entropy Distribution (Correct)',
            'Entropy Distribution (Wrong)'
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.10
    )
    
    # Panel A: Non-zero features histogram (Correct)
    fig.add_trace(go.Histogram(
        x=correct['nonzero_features'].dropna(),
        nbinsx=50,
        marker_color=PLOT_CONFIG['colors']['correct_seen'],
        opacity=0.7,
        showlegend=False,
        name='Correct'
    ), row=1, col=1)
    
    # Panel B: Non-zero features histogram (Wrong)
    fig.add_trace(go.Histogram(
        x=wrong['nonzero_features'].dropna(),
        nbinsx=50,
        marker_color=PLOT_CONFIG['colors']['incorrect_seen'],
        opacity=0.7,
        showlegend=False,
        name='Wrong'
    ), row=1, col=2)
    
    # Panel C: Entropy histogram (Correct)
    fig.add_trace(go.Histogram(
        x=correct['entropy'].dropna(),
        nbinsx=50,
        marker_color=PLOT_CONFIG['colors']['correct_seen'],
        opacity=0.7,
        showlegend=False,
        name='Correct'
    ), row=2, col=1)
    
    # Panel D: Entropy histogram (Wrong)
    fig.add_trace(go.Histogram(
        x=wrong['entropy'].dropna(),
        nbinsx=50,
        marker_color=PLOT_CONFIG['colors']['incorrect_seen'],
        opacity=0.7,
        showlegend=False,
        name='Wrong'
    ), row=2, col=2)
    
    # Update axes labels
    fig.update_xaxes(title_text="Non-zero Features", row=1, col=1)
    fig.update_xaxes(title_text="Non-zero Features", row=1, col=2)
    fig.update_xaxes(title_text="Feature Entropy", row=2, col=1)
    fig.update_xaxes(title_text="Feature Entropy", row=2, col=2)
    
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=2)
    
    # Update layout
    fig.update_layout(
        title='Feature Quality Distributions (All Validation Samples)',
        template=PLOT_CONFIG['template'],
        height=800,
        width=1000,
        font=dict(size=PLOT_CONFIG['font_size']),
        showlegend=False
    )
    
    # Save figure
    output_file = output_dir / "sup_05_feature_quality_distributions.png"
    fig.write_image(str(output_file), width=1000, height=800, scale=2)
    
    if not quiet:
        print(f"  ✓ {output_file.name}")
        
        # Calculate skewness to show non-normality
        from scipy import stats as sp_stats
        
        correct_nz = correct['nonzero_features'].dropna()
        wrong_nz = wrong['nonzero_features'].dropna()
        correct_ent = correct['entropy'].dropna()
        wrong_ent = wrong['entropy'].dropna()
        
        print(f"\n  Skewness (0 = symmetric, >1 = highly skewed):")
        print(f"    Non-zero features - Correct: {sp_stats.skew(correct_nz):.2f}")
        print(f"    Non-zero features - Wrong:   {sp_stats.skew(wrong_nz):.2f}")
        print(f"    Entropy - Correct: {sp_stats.skew(correct_ent):.2f}")
        print(f"    Entropy - Wrong:   {sp_stats.skew(wrong_ent):.2f}")
        print(f"  → Highly skewed distributions justify non-parametric Mann-Whitney U test")


def generate_computational_resources_table(output_path: Path, quiet: bool = False) -> None:
    """Generate computational resources table stratified by memory tier."""
    import os
    import numpy as np
    
    # Collect data from validation predictions
    pred_dir = Path("results/validation_predictions")
    data = []
    
    for sample_dir in pred_dir.glob("*"):
        if not sample_dir.is_dir():
            continue
        
        sample_id = sample_dir.name
        jobinfo = sample_dir / ".jobinfo"
        
        if jobinfo.exists():
            try:
                with open(jobinfo) as f:
                    info = json.load(f)
                
                # Only include successful jobs
                if info.get('status') != 'SUCCESS':
                    continue
                
                # Extract memory (MB -> GB)
                memory_mb = info.get('memory_mb', 0)
                memory_gb = memory_mb / 1024.0 if memory_mb else 0
                
                # Extract runtime (seconds -> minutes)
                runtime_sec = info.get('elapsed_seconds', 0)
                runtime_min = runtime_sec / 60.0 if runtime_sec else 0
                
                # Extract CPUs
                cpus = info.get('cpus', 6)  # Default to 6 if not found
                
                # Get FASTQ input size
                fastq_dir = Path(f"data/validation/raw/{sample_id}")
                fastq_size_gb = 0.0
                if fastq_dir.exists():
                    for fq in fastq_dir.glob("*.fastq*"):
                        fastq_size_gb += os.path.getsize(fq) / (1024**3)
                
                # Skip if missing data
                if memory_gb == 0 or runtime_min == 0:
                    continue
                
                data.append({
                    'memory_gb': memory_gb,
                    'runtime_min': runtime_min,
                    'input_size_gb': fastq_size_gb,
                    'cpus': cpus
                })
            except Exception:
                continue
    
    if len(data) == 0:
        if not quiet:
            print(f"  ⚠ No computational resource data found")
        return
    
    # Convert to DataFrame
    import pandas as pd
    df = pd.DataFrame(data)
    
    # Define memory tiers (round to nearest standard tier)
    def assign_tier(mem_gb):
        if mem_gb <= 35:
            return 31
        elif mem_gb <= 70:
            return 62
        elif mem_gb <= 140:
            return 125
        elif mem_gb <= 280:
            return 250
        else:
            return 500
    
    df['tier'] = df['memory_gb'].apply(assign_tier)
    
    # Aggregate by tier
    tier_stats = df.groupby('tier').agg({
        'runtime_min': ['count', 'mean', 'std'],
        'input_size_gb': ['mean', 'std'],
        'cpus': 'first'  # CPUs should be constant per tier
    }).reset_index()
    
    tier_stats.columns = ['tier', 'n', 'runtime_mean', 'runtime_std', 'input_mean', 'input_std', 'cpus']
    tier_stats = tier_stats.sort_values('tier')
    
    # Generate LaTeX table
    lines = []
    lines.append(r"\begin{tabular}{lrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Memory (GB) & CPUs & N & Runtime (min) & Input size (GB) \\")
    lines.append(r"\midrule")
    
    for _, row in tier_stats.iterrows():
        runtime_str = f"{row['runtime_mean']:.2f} $\\pm$ {row['runtime_std']:.2f}" if not np.isnan(row['runtime_std']) else f"{row['runtime_mean']:.2f}"
        input_str = f"{row['input_mean']:.2f} $\\pm$ {row['input_std']:.2f}" if not np.isnan(row['input_std']) else f"{row['input_mean']:.2f}"
        lines.append(f"{int(row['tier'])} & {int(row['cpus'])} & {int(row['n'])} & {runtime_str} & {input_str} \\\\")
    
    lines.append(r"\botrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\\[2mm]")
    lines.append("{\\footnotesize Runtime includes the creation of a vector of unitig feature abundances per sample and neural network inference.}")
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    if not quiet:
        print(f"  ✓ Generated computational resources table: {output_path.name}")


def generate_merged_wrong_predictions_table(df: pd.DataFrame, output_path: Path, quiet: bool = False) -> None:
    """Generate merged table of wrong predictions (A->B) with counts and confidence stats."""
    import numpy as np
    
    lines = []
    lines.append(r"\begin{longtable}{lp{4cm}p{4cm}rr}")
    lines.append(r"\caption{Common misclassification patterns across all tasks (Validation set)} \\")
    lines.append(r"\toprule")
    lines.append(r"Task & True Label & Predicted Label & Count & Confidence (\%) \\")
    lines.append(r"\midrule")
    
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
        
        # Group by true->predicted pairs
        confusion_pairs = wrong_df.groupby(['true_label', 'pred_label']).agg({
            'confidence': ['count', 'mean', 'std']
        }).reset_index()
        
        confusion_pairs.columns = ['true_label', 'pred_label', 'n', 'conf_mean', 'conf_std']
        confusion_pairs = confusion_pairs.sort_values('n', ascending=False).head(5)  # Top 5 per task
        
        # Add task name to first row only
        for idx, row in confusion_pairs.iterrows():
            true_lbl = str(row['true_label']).replace('_', '\\_')
            pred_lbl = str(row['pred_label']).replace('_', '\\_')
            conf_mean = row['conf_mean'] * 100
            conf_std = row['conf_std'] * 100 if not np.isnan(row['conf_std']) else 0
            
            if idx == confusion_pairs.index[0]:
                task_str = task_labels[task]
            else:
                task_str = ""
            
            if not np.isnan(conf_std) and conf_std > 0:
                conf_str = f"{conf_mean:.1f} $\\pm$ {conf_std:.1f}"
            else:
                conf_str = f"{conf_mean:.1f}"
            
            lines.append(f"{task_str} & {true_lbl} & {pred_lbl} & {int(row['n'])} & {conf_str} \\\\")
        
        lines.append(r"\addlinespace")
    
    lines.append(r"\botrule")
    lines.append(r"\multicolumn{5}{p{0.95\linewidth}}{\footnotesize Only includes misclassifications of seen classes (present in training data). Count: Number of samples misclassified. Confidence: Mean prediction confidence (± SD) for incorrect predictions. Top 5 most common patterns per task.} \\")
    lines.append(r"\end{longtable}")
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    if not quiet:
        print(f"  ✓ Generated merged wrong predictions table: {output_path.name}")


def generate_seen_unseen_validation_table(
    df: pd.DataFrame,
    output_path: Path,
    quiet: bool = False
) -> None:
    """Generate table showing SEEN vs UNSEEN predictions in validation set with error patterns.
    
    Shows distribution of seen/unseen samples and common misclassification patterns
    (True Label A -> Predicted Label B) with percentages relative to total validation samples.
    
    Args:
        df: Validation predictions dataframe with is_seen, is_correct columns
        output_path: Path to output LaTeX file
        quiet: If True, suppress output messages
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
                true_lbl = str(row['true_label']).replace('_', '\\_')
                pred_lbl = str(row['pred_label']).replace('_', '\\_')
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
                true_lbl = str(row['true_label']).replace('_', '\\_')
                pred_lbl = str(row['pred_label']).replace('_', '\\_')
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
    
    if not quiet:
        print(f"  ✓ Generated seen/unseen validation distribution table: {output_path.name}")


def save_tables(
    df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_dir: Path,
    args: argparse.Namespace,
    label_encoders: Dict,
    quiet: bool = False
) -> None:
    """Save summary tables in TSV, JSON, and LaTeX formats."""
    # Summary accuracy (TSV)
    summary_df.to_csv(output_dir / "validation_accuracy_summary.tsv", sep='\t', index=False)
    
    # Main table 01: Performance summary (Training/Test/Validation comparison)
    generate_performance_summary_table(
        validation_df=df,
        test_metrics_file=args.test_metrics,
        training_history_file=args.training_history,
        train_metadata_file=args.train_metadata,
        cv_results_file=args.cv_results,
        label_encoders=label_encoders,
        label_encoders_file=args.label_encoders,
        output_path=output_dir / "main_table_01_performance_summary.tex",
        train_predictions_dir=args.train_predictions_dir,
        quiet=quiet
    )
    
    # Supplementary table: Class distribution
    generate_class_distribution_table(
        train_meta_file=args.train_metadata,
        test_meta_file=args.test_metadata,
        validation_df=df,
        output_path=output_dir / "sup_table_01_class_distribution.tex",
        quiet=quiet
    )
    
    # Supplementary table: Unseen labels
    generate_unseen_labels_table(
        df=df,
        output_path=output_dir / "sup_table_02_unseen_labels.tex",
        quiet=quiet
    )
    
    # Supplementary table: Per-class performance
    generate_perclass_performance_table(
        df=df,
        output_path=output_dir / "sup_table_03_perclass_performance.tex",
        quiet=quiet
    )
    
    # Supplementary table: Hyperparameters
    generate_hyperparameters_table(
        hyperparams_file=args.hyperparameters,
        output_path=output_dir / "sup_table_04_hyperparameters.tex",
        quiet=quiet
    )
    
    # Merged wrong predictions table (sup_table_05)
    generate_merged_wrong_predictions_table(
        df=df,
        output_path=output_dir / "sup_table_05_wrong_predictions_merged.tex",
        quiet=quiet
    )
    
    # Seen/unseen validation distribution table (sup_table_07)
    generate_seen_unseen_validation_table(
        df=df,
        output_path=output_dir / "sup_table_07_seen_unseen_validation.tex",
        quiet=quiet
    )
    
    # Validation comparison JSON
    output_json = {}
    for task in TASKS:
        task_df = df[df['task'] == task]
        output_json[task] = {}
        
        for subset_name, subset_df in [('all', task_df), ('seen', task_df[task_df['is_seen']])]:
            if len(subset_df) > 0:
                # Filter out rows with NaN true labels (missing metadata)
                valid_subset = subset_df[subset_df['true_label'].notna()]
                
                if len(valid_subset) == 0:
                    continue
                
                y_true = valid_subset['true_label'].values
                y_pred = valid_subset['pred_label'].values
                
                # Get all unique labels (union of true and predicted)
                all_labels = sorted(set(y_true) | set(y_pred))
                
                cm = {}
                for _, row in valid_subset.iterrows():
                    key = f"{row['true_label']}_to_{row['pred_label']}"
                    cm[key] = cm.get(key, 0) + 1
                
                output_json[task][subset_name] = {
                    'accuracy': float(valid_subset['is_correct'].mean()),
                    'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
                    'f1_macro': float(f1_score(y_true, y_pred, labels=all_labels, average='macro', zero_division=0)),
                    'correct': int(valid_subset['is_correct'].sum()),
                    'total': len(valid_subset),
                    'confusion_matrix': cm
                }
    
    with open(output_dir / "validation_comparison.json", 'w') as f:
        json.dump(output_json, f, indent=2)
    
    # Main table: Computational resources
    generate_computational_resources_table(
        output_path=output_dir / "main_table_02_computational_resources.tex",
        quiet=quiet
    )
    
    if not quiet:
        print(f"\n  ✓ Saved all tables (TSV, JSON, and LaTeX formats)")
        print(f"  ✓ Generated 3 main tables (performance, resources) + 5 supplementary tables")


def create_paper_grid_figures(validation_dir: Path, quiet: bool = False) -> None:
    """
    Create publication-ready 2x2 grid figures by combining individual task plots.
    
    Generates:
    - main_01_roc_curves.png: ROC curves for all 4 tasks
    - main_02_pr_curves.png: PR curves for all 4 tasks  
    - main_03_confusion_matrices.png: Confusion matrices for all 4 tasks
    - sup_02_confidence_scores.png: Confidence distributions for all 4 tasks
    
    Args:
        validation_dir: Directory containing individual validation plot PNGs
        quiet: If True, suppress output messages
    """
    try:
        from PIL import Image
    except ImportError:
        if not quiet:
            print("\n⚠ PIL/Pillow not available, skipping paper grid figures")
        return
    
    if not quiet:
        print("\nGenerating paper-ready grid figures...")
    
    paper_dir = Path("paper/figures/final")
    paper_dir.mkdir(parents=True, exist_ok=True)
    
    tasks = ['sample_type', 'community_type', 'sample_host', 'material']
    
    # Define grid figure specifications
    grids = [
        {
            'name': 'main_01_roc_curves',
            'pattern': 'roc_curves_{}.png',
            'description': 'ROC Curves'
        },
        {
            'name': 'main_02_pr_curves',
            'pattern': 'pr_curves_{}.png',
            'description': 'PR Curves'
        },
        {
            'name': 'main_03_confusion_matrices',
            'pattern': 'confusion_matrix_{}.png',
            'description': 'Confusion Matrices'
        },
        {
            'name': 'sup_02_confidence_scores',
            'pattern': 'confidence_distribution_{}.png',
            'description': 'Confidence Distributions'
        }
    ]
    
    for grid_spec in grids:
        try:
            # Load images for all 4 tasks
            images = []
            for task in tasks:
                img_path = validation_dir / grid_spec['pattern'].format(task)
                if not img_path.exists():
                    if not quiet:
                        print(f"  ⚠ Missing {img_path.name}, skipping {grid_spec['name']}")
                    break
                images.append(Image.open(img_path))
            
            if len(images) != 4:
                continue
            
            # For confusion matrices, make sample_type and community_type larger
            if grid_spec['name'] == 'main_03_confusion_matrices':
                # Resize: sample_type and community_type should be 1.5x larger
                scale_factors = [1.5, 1.5, 1.0, 1.0]  # sample_type, community_type, sample_host, material
                resized_images = []
                for img, scale in zip(images, scale_factors):
                    if scale != 1.0:
                        new_width = int(img.width * scale)
                        new_height = int(img.height * scale)
                        resized_images.append(img.resize((new_width, new_height), Image.Resampling.LANCZOS))
                    else:
                        resized_images.append(img)
                
                # Calculate grid dimensions
                # Top row: sample_type (1.5x) + community_type (1.5x)
                # Bottom row: sample_host (1.0x) + material (1.0x)
                top_width = resized_images[0].width + resized_images[1].width
                bottom_width = resized_images[2].width + resized_images[3].width
                grid_width = max(top_width, bottom_width)
                
                top_height = max(resized_images[0].height, resized_images[1].height)
                bottom_height = max(resized_images[2].height, resized_images[3].height)
                grid_height = top_height + bottom_height
                
                grid = Image.new('RGB', (grid_width, grid_height), 'white')
                
                # Paste images centered in their respective positions
                # Top left: sample_type
                grid.paste(resized_images[0], (0, 0))
                # Top right: community_type
                grid.paste(resized_images[1], (resized_images[0].width, 0))
                # Bottom left: sample_host (centered)
                x_offset_left = (grid_width // 2 - resized_images[2].width) // 2
                grid.paste(resized_images[2], (x_offset_left, top_height))
                # Bottom right: material (centered)
                x_offset_right = grid_width // 2 + (grid_width // 2 - resized_images[3].width) // 2
                grid.paste(resized_images[3], (x_offset_right, top_height))
            else:
                # Standard 2x2 grid for other figure types
                widths = [img.width for img in images]
                heights = [img.height for img in images]
                max_width = max(widths)
                max_height = max(heights)
                
                # Resize all to same size (maintaining aspect ratio)
                resized_images = []
                for img in images:
                    if img.width != max_width or img.height != max_height:
                        # Create white canvas
                        canvas = Image.new('RGB', (max_width, max_height), 'white')
                        # Center the image
                        x_offset = (max_width - img.width) // 2
                        y_offset = (max_height - img.height) // 2
                        canvas.paste(img, (x_offset, y_offset))
                        resized_images.append(canvas)
                    else:
                        resized_images.append(img)
                
                # Create 2x2 grid
                grid_width = max_width * 2
                grid_height = max_height * 2
                grid = Image.new('RGB', (grid_width, grid_height), 'white')
                
                # Paste images: top-left, top-right, bottom-left, bottom-right
                grid.paste(resized_images[0], (0, 0))
                grid.paste(resized_images[1], (max_width, 0))
                grid.paste(resized_images[2], (0, max_height))
                grid.paste(resized_images[3], (max_width, max_height))
            
            # Save with high resolution
            output_path = paper_dir / f"{grid_spec['name']}.png"
            grid.save(output_path, dpi=(300, 300), optimize=False)  # High resolution, no compression
            
            if not quiet:
                size_kb = output_path.stat().st_size / 1024
                print(f"  ✓ {grid_spec['description']}: {output_path.name} ({size_kb:.0f} KB)")
        
        except Exception as e:
            if not quiet:
                print(f"  ⚠ Error creating {grid_spec['name']}: {e}")


# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def main() -> None:
    """Main execution workflow."""
    args = parse_arguments()
    
    # Setup output directories
    figures_dir = Path(args.output_dir) / "figures" / "final"
    tables_dir = Path(args.output_dir) / "tables" / "final"
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
    save_tables(df, summary_df, tables_dir, args, label_encoders, quiet=args.quiet)
    
    # Generate paper-ready grid figures
    create_paper_grid_figures(figures_dir, quiet=args.quiet)
    
    # Generate additional manuscript figures
    if not args.quiet:
        print("\nGenerating additional manuscript figures...")
    
    generate_feature_importance_figure(figures_dir, quiet=args.quiet)
    generate_runtime_memory_figure(figures_dir, quiet=args.quiet)
    generate_data_split_validation_figure(
        args.train_metadata, 
        args.test_metadata,
        args.metadata,
        figures_dir, 
        quiet=args.quiet
    )
    generate_blast_hit_rate_figure(figures_dir, quiet=args.quiet)
    generate_feature_quality_figure(df, figures_dir, quiet=args.quiet)
    generate_feature_quality_distributions(df, figures_dir, quiet=args.quiet)
    
    if not args.quiet:
        print(f"\n{'='*80}")
        print(f"Complete! Analyzed {len(df['sample_id'].unique())} samples")
        print(f"Figures: {figures_dir}")
        print(f"Tables: {tables_dir}")
        print(f"{'='*80}")


if __name__ == "__main__":
    main()
