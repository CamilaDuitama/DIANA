#!/usr/bin/env python
"""
DIANA Test CLI - Evaluate trained multi-task classifier on test data
====================================================================

Command-line interface for evaluating the final trained model on held-out
test data and generating comprehensive evaluation metrics and visualizations.

Usage:
    diana-test --model results/full_training/best_model.pth \\
               --config results/full_training/final_training_config.json \\
               --test-ids data/splits/test_ids.txt \\
               --output results/test_evaluation

Features:
    - Load trained model and test data
    - Generate predictions for all tasks
    - Compute comprehensive metrics (accuracy, F1, confusion matrices)
    - Save results as JSON and visualizations
    - Support for batch processing of large test sets
"""

import argparse
import json
import torch
import numpy as np
import polars as pl
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from diana.data.loader import MatrixLoader
from diana.models.multitask_mlp import MultiTaskMLP


def load_model(model_path: Path, config: dict, device: str = 'cuda'):
    """
    Load trained model from checkpoint.
    
    Args:
        model_path: Path to saved model
        config: Configuration dictionary with model params
        device: Device to load model on
        
    Returns:
        Loaded model in eval mode
    """
    logger.info(f"Loading model from {model_path}")
    
    # Get model architecture from config
    hyperparams = config['hyperparameters']
    task_names = config['task_names']
    
    # Need to know number of classes for each task - will infer from label encoders
    label_encoders_path = model_path.parent / 'label_encoders.json'
    with open(label_encoders_path, 'r') as f:
        encoders_data = json.load(f)
    
    task_info = {task: len(data['classes']) for task, data in encoders_data.items()}
    
    # Create model (don't know input_dim yet, will get from data)
    return task_info, encoders_data


def load_test_data(matrix_path: Path, metadata_path: Path, test_ids_path: Path):
    """
    Load test data and filter to test samples.
    
    Args:
        matrix_path: Path to full matrix
        metadata_path: Path to metadata
        test_ids_path: Path to test IDs file
        
    Returns:
        X_test, metadata_test (filtered to test samples)
    """
    logger.info(f"Loading test data from {matrix_path}")
    
    # Load test IDs
    with open(test_ids_path, 'r') as f:
        test_ids = set(line.strip() for line in f if line.strip())
    
    logger.info(f"Test set contains {len(test_ids)} samples")
    
    # Load full matrix with metadata
    loader = MatrixLoader(matrix_path)
    X_full, metadata_pl = loader.load_with_metadata(
        metadata_path=metadata_path,
        align_to_matrix=True
    )
    
    # Convert to pandas for easier filtering
    metadata = metadata_pl.to_pandas()
    
    # Filter to test samples
    test_mask = metadata['Run_accession'].isin(test_ids)
    X_test = X_full[test_mask]
    metadata_test = metadata[test_mask].reset_index(drop=True)
    
    logger.info(f"Loaded {len(X_test)} test samples with {X_test.shape[1]} features")
    
    return X_test, metadata_test


def encode_labels(metadata: 'pd.DataFrame', task_names: list, encoders_data: dict):
    """
    Encode test labels using saved encoders.
    
    Args:
        metadata: Test metadata
        task_names: List of task names
        encoders_data: Saved encoder information
        
    Returns:
        Dictionary of encoded labels
    """
    y_test = {}
    
    for task in task_names:
        # Create encoder from saved classes
        encoder = LabelEncoder()
        encoder.classes_ = np.array(encoders_data[task]['classes'])
        
        # Transform test labels
        y_test[task] = encoder.transform(metadata[task].values)
        logger.info(f"Encoded {task}: {len(y_test[task])} samples")
    
    return y_test


def evaluate_model(model, X_test, y_test, task_names, device, batch_size=96):
    """
    Evaluate model on test data.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels (dict)
        task_names: List of task names
        device: Device to run on
        batch_size: Batch size for inference
        
    Returns:
        Dictionary of predictions and metrics
    """
    logger.info("Running inference on test set...")
    
    model.eval()
    predictions = {task: [] for task in task_names}
    probabilities = {task: [] for task in task_names}
    
    # Run inference in batches
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            batch_X = torch.FloatTensor(X_test[i:i+batch_size]).to(device)
            outputs = model(batch_X)
            
            for task in task_names:
                # Get predicted classes
                preds = torch.argmax(outputs[task], dim=1).cpu().numpy()
                predictions[task].extend(preds)
                
                # Get probabilities (apply softmax to logits)
                probs = torch.softmax(outputs[task], dim=1).cpu().numpy()
                probabilities[task].extend(probs)
    
    # Convert to arrays
    predictions = {task: np.array(preds) for task, preds in predictions.items()}
    probabilities = {task: np.array(probs) for task, probs in probabilities.items()}
    
    # Compute metrics for each task
    results = {}
    for task in task_names:
        y_true = y_test[task]
        y_pred = predictions[task]
        
        task_results = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
            'f1_macro': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
            'f1_weighted': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
            'precision_macro': float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
            'recall_macro': float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
            'classification_report': classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        }
        
        results[task] = task_results
        
        logger.info(f"{task}: Accuracy={task_results['accuracy']:.4f}, "
                   f"F1={task_results['f1_weighted']:.4f}")
    
    return predictions, probabilities, results


def save_results(results: dict, predictions: dict, probabilities: dict, y_test: dict, 
                metadata: 'pd.DataFrame', output_dir: Path, encoders_data: dict):
    """
    Save evaluation results.
    
    Args:
        results: Metrics dictionary
        predictions: Predictions dictionary
        probabilities: Probability distributions dictionary
        y_test: True labels
        metadata: Test metadata
        output_dir: Output directory
        encoders_data: Label encoder data
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics
    metrics_path = output_dir / 'test_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved metrics to {metrics_path}")
    
    # Save predictions with metadata
    predictions_df = metadata[['Run_accession']].copy()
    
    for task in predictions.keys():
        # Add predicted class index
        predictions_df[f'{task}_pred_idx'] = predictions[task]
        predictions_df[f'{task}_true_idx'] = y_test[task]
        
        # Add predicted class name
        encoder_classes = encoders_data[task]['classes']
        predictions_df[f'{task}_pred'] = [encoder_classes[idx] for idx in predictions[task]]
        predictions_df[f'{task}_true'] = [encoder_classes[idx] for idx in y_test[task]]
        
        # Add probabilities for each class
        task_probs = probabilities[task]
        for i, class_name in enumerate(encoder_classes):
            predictions_df[f'{task}_prob_{i}'] = task_probs[:, i]
    
    predictions_path = output_dir / 'test_predictions.tsv'
    predictions_df.to_csv(predictions_path, sep='\t', index=False)
    logger.info(f"Saved predictions to {predictions_path}")
    
    # Print summary
    logger.info("\n" + "="*70)
    logger.info("TEST SET EVALUATION SUMMARY")
    logger.info("="*70)
    logger.info(f"\nTest samples: {len(metadata)}")
    logger.info("\nPer-Task Performance:\n")
    
    for task, metrics in results.items():
        logger.info(f"{task.upper()}:")
        logger.info(f"  Accuracy:          {metrics['accuracy']:.4f}")
        logger.info(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
        logger.info(f"  F1 (weighted):     {metrics['f1_weighted']:.4f}")
        logger.info(f"  F1 (macro):        {metrics['f1_macro']:.4f}")
        logger.info("")
    
    logger.info("="*70)


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate trained DIANA model on test data',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--model', type=Path, required=True,
                       help='Path to trained model (.pth file)')
    parser.add_argument('--config', type=Path, required=True,
                       help='Path to training config JSON')
    parser.add_argument('--matrix', type=Path, 
                       default=Path('data/matrices/large_matrix_3070_with_frac/unitigs.frac.mat'),
                       help='Path to feature matrix')
    parser.add_argument('--metadata', type=Path,
                       default=Path('data/metadata/DIANA_metadata.tsv'),
                       help='Path to metadata file')
    parser.add_argument('--test-ids', type=Path,
                       default=Path('data/splits/test_ids.txt'),
                       help='Path to test IDs file')
    parser.add_argument('--output', type=Path, required=True,
                       help='Output directory for results')
    parser.add_argument('--batch-size', type=int, default=96,
                       help='Batch size for inference')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to run on')
    
    args = parser.parse_args()
    
    # Check GPU availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        args.device = 'cpu'
    
    logger.info("="*70)
    logger.info("DIANA Test Evaluation")
    logger.info("="*70)
    logger.info(f"Model: {args.model}")
    logger.info(f"Config: {args.config}")
    logger.info(f"Test IDs: {args.test_ids}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Device: {args.device}")
    logger.info("")
    
    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    task_names = config['task_names']
    
    # Load label encoders and get task info
    task_info, encoders_data = load_model(args.model, config, args.device)
    
    # Load test data
    X_test, metadata_test = load_test_data(args.matrix, args.metadata, args.test_ids)
    
    # Encode labels
    y_test = encode_labels(metadata_test, task_names, encoders_data)
    
    # Initialize model with correct architecture
    model = MultiTaskMLP(
        input_dim=X_test.shape[1],
        num_classes=task_info,
        **config['hyperparameters']['model_params']
    )
    
    # Load weights
    checkpoint = torch.load(args.model, map_location=args.device, weights_only=False)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(args.device)
    model.eval()
    
    logger.info("Model loaded successfully")
    
    # Run evaluation
    predictions, probabilities, results = evaluate_model(
        model, X_test, y_test, task_names, args.device, args.batch_size
    )
    
    # Save results
    save_results(results, predictions, probabilities, y_test, metadata_test, args.output, encoders_data)
    
    logger.info("\nEvaluation complete!")


if __name__ == '__main__':
    main()
