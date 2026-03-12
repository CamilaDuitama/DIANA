#!/usr/bin/env python3
"""
Generate ROC and Precision-Recall Curves for All Tasks

PURPOSE:
    Create ROC (Receiver Operating Characteristic) and Precision-Recall curves
    showing classifier performance across different decision thresholds for each task.
    Curves are generated for SEEN CLASSES ONLY (statistically valid).

INPUTS:
    - paper/metadata/validation_metadata.tsv: Validation sample metadata with true labels
    - results/validation/{sample}/predictions.json: Model predictions with probabilities
    - results/training/label_encoders.json: Label encoders for decoding predictions

OUTPUTS:
    - paper/figures/final/main_02_roc_curves_sample_type.png
    - paper/figures/final/main_02_roc_curves_community_type.png
    - paper/figures/final/main_02_roc_curves_sample_host.png
    - paper/figures/final/main_02_roc_curves_material.png
    - paper/figures/final/main_02_pr_curves_sample_type.png
    - paper/figures/final/main_02_pr_curves_community_type.png
    - paper/figures/final/main_02_pr_curves_sample_host.png
    - paper/figures/final/main_02_pr_curves_material.png
    - Corresponding .html interactive versions

PROCESS:
    1. Load validation metadata (ground truth labels)
    2. Load predictions with class probabilities
    3. Identify which validation samples have SEEN vs UNSEEN labels (training set comparison)
    4. For each task (SEEN classes only):
        a. Extract true labels and predicted probabilities
        b. Binarize labels for one-vs-rest evaluation
        c. Compute ROC curves (FPR vs TPR) with AUC scores
        d. Compute PR curves (Recall vs Precision) with AP scores
        e. Generate plotly line plots with:
           - One curve per class
           - Vivid palette colors
           - AUC/AP scores in legend
           - Sample counts per class
        f. Save as high-resolution PNG and interactive HTML

CONFIGURATION:
    All styling, paths, and constants imported from config.py:
    - PATHS: File locations for inputs/outputs
    - TASKS: List of classification tasks
    - PLOT_CONFIG: Color palette, template, font sizes
    - SAMPLE_TYPE_MAP: Normalization of training labels

DEPENDENCIES:
    - pandas, numpy, plotly, scikit-learn
    - config.py (same directory)

USAGE:
    python scripts/paper/generate_roc_pr_curves.py
    
AUTHOR: Generated via refactoring of 06_compare_predictions.py
"""

import sys
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)
from sklearn.preprocessing import label_binarize

# Add script directory to path for config import
sys.path.insert(0, str(Path(__file__).parent))
from config import PATHS, TASKS, PLOT_CONFIG, SAMPLE_TYPE_MAP


# Binary positive class for sample_type (modern vs ancient)
BINARY_POSITIVE_CLASS = {'sample_type': 'modern'}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def filter_valid_labels(df: pd.DataFrame, subset: list = None) -> pd.DataFrame:
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


def identify_seen_labels(validation_metadata_path: str, train_metadata_path: str) -> Dict[str, set]:
    """
    Identify which labels were present in the training set for each task.
    
    Returns:
        Dictionary mapping task names to sets of seen labels
    """
    train_df = pd.read_csv(train_metadata_path, sep='\t')
    
    seen_labels = {}
    for task in TASKS:
        if task in train_df.columns:
            # Get unique non-null training labels
            task_labels = train_df[task].dropna().unique()
            
            # Normalize sample_type labels
            if task == 'sample_type':
                task_labels = [SAMPLE_TYPE_MAP.get(lbl, lbl) for lbl in task_labels]
            
            seen_labels[task] = set(task_labels)
    
    return seen_labels


# ============================================================================
# DATA LOADING
# ============================================================================

def load_predictions_data() -> pd.DataFrame:
    """
    Load predictions with probabilities and metadata into a master DataFrame.
    
    Returns:
        DataFrame with columns: sample_id, task, true_label, pred_label, probabilities, is_seen
    """
    # Load label encoders
    with open(PATHS['label_encoders']) as f:
        label_encoders = json.load(f)
    
    # Load validation metadata
    metadata = pd.read_csv(PATHS['validation_metadata'], sep='\t')
    
    # Identify seen labels from training set
    seen_labels = identify_seen_labels(PATHS['validation_metadata'], PATHS['train_metadata'])
    
    # Find all prediction files
    predictions_dir = Path(PATHS['predictions_dir'])
    prediction_files = list(predictions_dir.rglob('*_predictions.json'))
    
    print(f"Loading {len(prediction_files)} prediction files...")
    
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
            
            # Get probabilities
            probabilities = pred_info.get('probabilities', {})
            
            # Check if label was seen in training
            is_seen = true_label in seen_labels.get(task_name, set())
            
            records.append({
                'sample_id': sample_id,
                'task': task_name,
                'true_label': true_label,
                'pred_label': pred_label,
                'probabilities': probabilities,
                'is_seen': is_seen
            })
    
    df = pd.DataFrame(records)
    print(f"✓ Loaded {len(df)} predictions for {df['sample_id'].nunique()} samples")
    
    # Print seen/unseen statistics
    for task in TASKS:
        task_df = df[df['task'] == task]
        n_seen = task_df['is_seen'].sum()
        n_unseen = (~task_df['is_seen']).sum()
        print(f"  {task}: {n_seen} seen, {n_unseen} unseen")
    
    return df


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_roc_pr_curves(
    df: pd.DataFrame,
    task: str,
    label_encoders: Dict,
    output_dir: Path
) -> None:
    """
    Generate ROC and PR curves for a single task (SEEN classes only).
    
    Args:
        df: Master predictions dataframe
        task: Task name (e.g., 'sample_type')
        label_encoders: Dictionary of label encoders
        output_dir: Directory to save output files
    """
    # CRITICAL: Filter to seen classes only - ROC/PR curves are meaningless for unseen classes
    task_df = df[(df['task'] == task) & (df['is_seen'])].copy()
    task_df = filter_valid_labels(task_df)
    
    if len(task_df) == 0:
        print(f"  ⚠ No valid SEEN data for {task}")
        return
    
    # Get classes and normalize for sample_type
    classes = label_encoders[task]['classes']
    if task == 'sample_type':
        classes = [SAMPLE_TYPE_MAP.get(c, c) for c in classes]
    
    n_classes = len(classes)
    
    # Check if we have multiple classes in validation data
    unique_labels = task_df['true_label'].unique()
    if len(unique_labels) < 2:
        print(f"  ⚠ Only 1 class in validation data for {task}")
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
        print(f"  ⚠ No positive examples for {task}")
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
    
    # ========================================================================
    # ROC Curves
    # ========================================================================
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
                line=dict(
                    width=4,
                    color=PLOT_CONFIG['colors']['palette'][i % len(PLOT_CONFIG['colors']['palette'])]
                )
            ))
    
    # Add random classifier line
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
    
    # Save with proper figure numbering
    roc_png = output_dir / f"main_02_roc_curves_{task}.png"
    fig_roc.write_html(str(roc_png.with_suffix('.html')))
    fig_roc.write_image(str(roc_png), width=1000, height=800, scale=2)
    
    print(f"  ✓ {roc_png.name}")
    
    # ========================================================================
    # Precision-Recall Curves
    # ========================================================================
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
                line=dict(
                    width=4,
                    color=PLOT_CONFIG['colors']['palette'][i % len(PLOT_CONFIG['colors']['palette'])]
                )
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
    
    # Save with proper figure numbering
    pr_png = output_dir / f"main_02_pr_curves_{task}.png"
    fig_pr.write_html(str(pr_png.with_suffix('.html')))
    fig_pr.write_image(str(pr_png), width=1000, height=800, scale=2)
    
    print(f"  ✓ {pr_png.name}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("=" * 80)
    print("GENERATING ROC AND PRECISION-RECALL CURVES")
    print("=" * 80)
    
    # Load label encoders
    with open(PATHS['label_encoders']) as f:
        label_encoders = json.load(f)
    
    # Create output directory
    output_dir = Path(PATHS['figures_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\n[1/2] Loading predictions and metadata...")
    df = load_predictions_data()
    
    # Generate ROC and PR curves
    print(f"\n[2/2] Generating ROC and PR curves for {len(TASKS)} tasks (SEEN classes only)...")
    for task in TASKS:
        plot_roc_pr_curves(df, task, label_encoders, output_dir)
    
    print("\n" + "=" * 80)
    print("✓ COMPLETE - All ROC and PR curves generated")
    print("=" * 80)


if __name__ == '__main__':
    main()
