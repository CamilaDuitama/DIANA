#!/usr/bin/env python3
"""
Configuration file for paper figure and table generation.

This module contains all file paths, constants, and plot configurations
used across all paper generation scripts.
"""

import plotly.express as px

# ============================================================================
# FILE PATHS
# ============================================================================

PATHS = {
    # Metadata files
    'validation_metadata': 'paper/metadata/validation_metadata.tsv',
    'train_metadata': 'paper/metadata/train_metadata.tsv',
    'test_metadata': 'paper/metadata/test_metadata.tsv',
    
    # Prediction results
    'predictions_dir': 'results/validation_predictions',
    
    # Model files
    'label_encoders': 'results/training/label_encoders.json',
    'best_model': 'results/training/best_model.pth',
    'model_config': 'results/training/final_training_config.json',
    
    # Performance metrics
    'test_metrics': 'results/test_evaluation/test_metrics.json',
    'training_metrics': 'results/training/training_set_metrics.json',  # Full training set (2,609 samples)
    'training_history': 'results/training/training_history.json',
    
    # Cross-validation and hyperparameters
    'hyperparameters': 'results/training/cv_results/best_hyperparameters.json',
    'cv_results': 'results/training/cv_results/aggregated_results.json',
    
    # Feature analysis
    'feature_importance_dir': 'results/feature_analysis',
    'blast_results': 'results/feature_analysis/blast_annotations.tsv',
    
    # Output directories
    'output_base': 'paper',
    'figures_dir': 'paper/figures/final',
    'tables_dir': 'paper/tables/final',
}

# Shared validation data loader
# Location: scripts/validation/load_validation_data.py
# Import this in paper scripts to avoid code duplication:
#   sys.path.insert(0, str(Path(__file__).parent.parent / 'validation'))
#   from load_validation_data import load_validation_predictions, get_unseen_tasks
# This ensures all scripts use the same data loading logic

# ============================================================================
# CONSTANTS
# ============================================================================

# Task names (in order)
TASKS = ['sample_type', 'community_type', 'sample_host', 'material']

# Sample type mapping for display
SAMPLE_TYPE_MAP = {
    'ancient_metagenome': 'ancient',
    'modern_metagenome': 'modern'
}

# Positive class for binary classification (for ROC/PR curves)
BINARY_POSITIVE_CLASS = {
    'sample_type': 'modern'
}

# ============================================================================
# PLOT CONFIGURATION
# ============================================================================

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
