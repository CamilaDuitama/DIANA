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
    1. Load validation metadata (ground truth labels)
    2. Load predictions from all validation samples
    3. Merge predictions with true labels
    4. For each task:
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
    - SAMPLE_TYPE_MAP: Normalization of training labels

DEPENDENCIES:
    - pandas, numpy, plotly, scikit-learn
    - config.py (same directory)

USAGE:
    python scripts/paper/generate_confusion_matrices.py
    
AUTHOR: Generated via refactoring of 06_compare_predictions.py
"""

import sys
import json
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix

# Add script directory to path for config import
sys.path.insert(0, str(Path(__file__).parent))
from config import PATHS, TASKS, PLOT_CONFIG, SAMPLE_TYPE_MAP


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
# DATA LOADING
# ============================================================================

def load_predictions_data() -> pd.DataFrame:
    """
    Load predictions and metadata into a master DataFrame.
    
    Returns:
        DataFrame with columns: sample_id, task, true_label, pred_label, confidence
    """
    # Load label encoders
    with open(PATHS['label_encoders']) as f:
        label_encoders = json.load(f)
    
    # Load validation metadata
    metadata = pd.read_csv(PATHS['validation_metadata'], sep='\t')
    
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
            
            confidence = pred_info['confidence']
            
            records.append({
                'sample_id': sample_id,
                'task': task_name,
                'true_label': true_label,
                'pred_label': pred_label,
                'confidence': confidence
            })
    
    df = pd.DataFrame(records)
    print(f"✓ Loaded {len(df)} predictions for {df['sample_id'].nunique()} samples")
    
    return df


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_confusion_matrix(df: pd.DataFrame, task: str, output_dir: Path) -> None:
    """
    Generate confusion matrix heatmap for a single task.
    
    Args:
        df: Master predictions dataframe
        task: Task name (e.g., 'sample_type')
        output_dir: Directory to save output files
    """
    # Filter to task and remove invalid labels
    task_df = df[df['task'] == task]
    task_df = filter_valid_labels(task_df)
    
    if len(task_df) == 0:
        print(f"  ⚠ No valid data for {task}")
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
    print("GENERATING CONFUSION MATRICES")
    print("=" * 80)
    
    # Create output directory
    output_dir = Path(PATHS['figures_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\n[1/2] Loading predictions and metadata...")
    df = load_predictions_data()
    
    # Generate confusion matrices
    print(f"\n[2/2] Generating confusion matrices for {len(TASKS)} tasks...")
    for task in TASKS:
        plot_confusion_matrix(df, task, output_dir)
    
    print("\n" + "=" * 80)
    print("✓ COMPLETE - All confusion matrices generated")
    print("=" * 80)


if __name__ == '__main__':
    main()
