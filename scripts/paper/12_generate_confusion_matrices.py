#!/usr/bin/env python3
"""
Generate Confusion Matrix Heatmaps for All Tasks

PURPOSE:
    Create normalized confusion matrices showing prediction accuracy for each classification task.
    Matrices display raw counts and normalized proportions (row-normalized).

INPUTS:
    - paper/metadata/validation_metadata.tsv: Validation sample metadata with true labels
    - results/validation/{sample}/predictions.json: Model predictions for each sample
    - results/training/label_encoders.json: Label encoders for decoding predictions

OUTPUTS:
    - paper/figures/final/main_01_confusion_matrix_sample_type.png
    - paper/figures/final/main_01_confusion_matrix_community_type.png
    - paper/figures/final/main_01_confusion_matrix_sample_host.png
    - paper/figures/final/main_01_confusion_matrix_material.png
    - Corresponding .html interactive versions

PROCESS:
    1. Load validation predictions using shared loader (includes is_seen flag)
    2. Filter to SEEN labels only (present in training set)
    3. For each task:
        a. Filter to valid (non-null) labels
        b. Compute confusion matrix using sklearn
        c. Normalize by row (true class)
        d. Generate plotly heatmap with:
           - Teal colorscale (from config)
           - Raw counts as text annotations
           - Proportions as heatmap colors
        e. Save as high-resolution PNG and interactive HTML

CONFIGURATION:
    All styling, paths, and constants imported from config.py:
    - PATHS: File locations for inputs/outputs
    - TASKS: List of classification tasks
    - PLOT_CONFIG: Colorscale, template, font sizes, borders

DEPENDENCIES:
    - pandas, numpy, plotly, scikit-learn
    - config.py (same directory)
    - load_validation_data.py (shared loader)

USAGE:
    python scripts/paper/generate_confusion_matrices.py
    
AUTHOR: Generated via refactoring of 06_compare_predictions.py
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix

# Add script directory to path for config import
sys.path.insert(0, str(Path(__file__).parent))
from config import PATHS, TASKS, PLOT_CONFIG

# Add validation scripts to path for shared loader
sys.path.insert(0, str(Path(__file__).parent.parent / 'validation'))
from load_validation_data import load_validation_predictions


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


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_confusion_matrix(df: pd.DataFrame, task: str, output_dir: Path) -> None:
    """
    Generate confusion matrix heatmap for a single task (SEEN LABELS ONLY).
    
    Args:
        df: Master predictions dataframe (must have is_seen column)
        task: Task name (e.g., 'sample_type')
        output_dir: Directory to save output files
    """
    # Filter to task, SEEN labels only, and remove invalid labels
    task_df = df[(df['task'] == task) & (df['is_seen'])].copy()
    task_df = filter_valid_labels(task_df)
    
    if len(task_df) == 0:
        print(f"  ⚠ No valid seen labels for {task}")
        return
    
    # Compute confusion matrix
    labels = sorted(task_df['true_label'].unique())
    cm = confusion_matrix(task_df['true_label'], task_df['pred_label'], labels=labels)
    cm_norm = cm / (cm.sum(axis=1, keepdims=True) + 1e-10)
    
    # Create heatmap
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
    
    # Dynamic sizing based on number of classes
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
    
    # Save outputs with proper figure numbering
    png_file = output_dir / f"main_01_confusion_matrix_{task}.png"
    fig.write_html(str(png_file.with_suffix('.html')))
    fig.write_image(str(png_file), width=size, height=size, scale=2)  # High resolution
    
    print(f"  ✓ {png_file.name}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("=" * 80)
    print("GENERATING CONFUSION MATRICES (SEEN LABELS ONLY)")
    print("=" * 80)
    
    # Create output directory
    output_dir = Path(PATHS['figures_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data using shared loader (includes is_seen flag)
    print("\n[1/2] Loading validation predictions...")
    df = load_validation_predictions()
    
    # Generate confusion matrices
    print(f"\n[2/2] Generating confusion matrices for {len(TASKS)} tasks (seen labels only)...")
    for task in TASKS:
        plot_confusion_matrix(df, task, output_dir)
    
    print("\n" + "=" * 80)
    print("✓ COMPLETE - All confusion matrices generated (seen labels only)")
    print("=" * 80)


if __name__ == '__main__':
    main()
