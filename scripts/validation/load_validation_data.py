#!/usr/bin/env python3
"""
Shared Validation Data Loader

PURPOSE:
    Centralized function to load validation predictions with all metadata.
    Used by all table and figure generation scripts to ensure consistency.

INPUTS (from config.py):
    - PATHS['validation_metadata']: Validation sample metadata with true labels
    - PATHS['predictions_dir']: Model predictions for each sample
    - PATHS['label_encoders']: Label encoders for decoding predictions

OUTPUTS:
    - DataFrame with columns: sample_id, task, true_label, pred_label, confidence, is_correct, is_seen

PROCESS:
    1. Load label encoders to get training classes
    2. Pre-compute normalized sample_type classes (ancient_metagenome → ancient)
    3. Load all prediction JSON files
    4. For each prediction:
        - Decode predicted class using label encoder
        - Get true label from metadata
        - Normalize sample_type labels
        - Determine if label was seen during training
        - Calculate is_correct

USAGE:
    # From scripts/paper directory:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent / 'validation'))
    from load_validation_data import load_validation_predictions
    
    df = load_validation_predictions()
    # df has: sample_id, task, true_label, pred_label, confidence, is_correct, is_seen
    
AUTHOR: Refactored from 06_compare_predictions.py load_data function
"""

import sys
import json
from pathlib import Path
from typing import Tuple

import pandas as pd

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent / 'paper'))
from config import PATHS, TASKS, SAMPLE_TYPE_MAP


def load_validation_predictions(quiet: bool = False) -> pd.DataFrame:
    """
    Load validation predictions with metadata.
    
    Args:
        quiet: If True, suppress progress messages
    
    Returns:
        DataFrame with columns:
            - sample_id: Run accession ID
            - task: Classification task name
            - true_label: Ground truth label
            - pred_label: Predicted label
            - confidence: Prediction confidence score
            - is_correct: Boolean, true if prediction matches ground truth
            - is_seen: Boolean, true if label was seen during training
    """
    # Load metadata
    metadata = pd.read_csv(PATHS['validation_metadata'], sep='\t')
    
    # Load label encoders
    with open(PATHS['label_encoders']) as f:
        label_encoders = json.load(f)
    
    # Pre-compute normalized training classes for sample_type
    # This is needed because training used "ancient_metagenome"/"modern_metagenome"
    # but predictions/validation use "ancient"/"modern"
    normalized_sample_type_classes = [
        SAMPLE_TYPE_MAP.get(c, c) 
        for c in label_encoders['sample_type']['classes']
    ]
    
    # Find all prediction files
    predictions_dir = Path(PATHS['predictions_dir'])
    prediction_files = list(predictions_dir.rglob('*_predictions.json'))
    
    if not quiet:
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
            
            # Normalize pred_label for sample_type
            if task_name == 'sample_type' and pred_label in SAMPLE_TYPE_MAP:
                pred_label = SAMPLE_TYPE_MAP[pred_label]
            
            # Check if seen in training (use pre-computed normalization for sample_type)
            if task_name == 'sample_type':
                is_seen = true_label in normalized_sample_type_classes
            else:
                is_seen = true_label in label_encoders[task_name]['classes']
            
            # Get confidence
            confidence = pred_info.get('confidence', 0.0)
            
            records.append({
                'sample_id': sample_id,
                'task': task_name,
                'true_label': str(true_label) if pd.notna(true_label) else 'Unknown',
                'pred_label': pred_label,
                'confidence': confidence,
                'is_correct': pred_label == str(true_label),
                'is_seen': is_seen
            })
    
    df = pd.DataFrame(records)
    
    if not quiet:
        print(f"✓ Loaded {len(df)} predictions for {df['sample_id'].nunique()} samples")
        # Show seen/unseen breakdown
        for task in TASKS:
            task_df = df[df['task'] == task]
            n_seen = task_df['is_seen'].sum()
            n_unseen = (~task_df['is_seen']).sum()
            print(f"  {task}: {n_seen} seen, {n_unseen} unseen")
    
    return df


def get_unseen_tasks(df: pd.DataFrame) -> list:
    """
    Determine which tasks have unseen labels in the validation set.
    
    Args:
        df: Validation predictions DataFrame
    
    Returns:
        List of task names that have at least one unseen label
    """
    unseen_tasks = []
    for task in TASKS:
        task_df = df[df['task'] == task]
        if (~task_df['is_seen']).sum() > 0:
            unseen_tasks.append(task)
    
    return unseen_tasks


if __name__ == '__main__':
    """Test the data loader."""
    print("=" * 80)
    print("TESTING VALIDATION DATA LOADER")
    print("=" * 80)
    print()
    
    df = load_validation_predictions()
    
    print()
    print("DataFrame shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print()
    
    print("Tasks with unseen labels:")
    unseen_tasks = get_unseen_tasks(df)
    for task in unseen_tasks:
        task_df = df[(df['task'] == task) & (~df['is_seen'])]
        unseen_labels = task_df['true_label'].unique()
        print(f"  {task}: {len(unseen_labels)} unseen labels")
        print(f"    Examples: {sorted(unseen_labels)[:3]}")
    
    print()
    print("=" * 80)
