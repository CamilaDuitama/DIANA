#!/usr/bin/env python3
"""
Evaluate the already-trained final model on the full training set.
This computes the 5 metrics we need without retraining.
"""

import sys
import json
import torch
import polars as pl
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    f1_score, precision_score, recall_score
)
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from diana.data.loader import MatrixLoader
from diana.models.multitask_mlp import MultiTaskMLP


def main():
    logger.info("="*60)
    logger.info("Evaluating Final Model on FULL Training Set")
    logger.info("="*60)
    
    # Load training config to get model architecture
    config_path = Path("results/training/final_training_config.json")
    logger.info(f"\nLoading config from {config_path}")
    with open(config_path) as f:
        config = json.load(f)
    
    hyperparams = config['hyperparameters']
    
    # Load model
    model_path = Path("results/training/best_model.pth")
    logger.info(f"Loading model from {model_path}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    logger.info(f"Using device: {device}")
    
    # Load training data - use train_metadata directly for speed
    logger.info("\nLoading training data...")
    matrix_path = Path(config['features_path'])
    
    loader = MatrixLoader(matrix_path)
    X_train, metadata_pl = loader.load_with_metadata(
        metadata_path=Path("paper/metadata/train_metadata.tsv"),
        align_to_matrix=True,
        filter_matrix_to_metadata=True  # Only load training samples
    )
    
    metadata_train = metadata_pl.to_pandas()
    
    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Feature dim: {X_train.shape[1]}")
    
    # Load label encoders
    with open("results/training/label_encoders.json") as f:
        encoders_data = json.load(f)
    
    task_names = list(encoders_data.keys())
    
    # Reconstruct task info
    task_info = {}
    for task in task_names:
        task_info[task] = len(encoders_data[task]['classes'])
    
    logger.info(f"Tasks: {task_names}")
    logger.info(f"Num classes: {task_info}")
    
    # Reconstruct model
    model = MultiTaskMLP(
        input_dim=X_train.shape[1],
        num_classes=task_info,
        **hyperparams['model_params']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Encode labels
    from sklearn.preprocessing import LabelEncoder
    import numpy as np
    y_true = {}
    for task in task_names:
        encoder = LabelEncoder()
        encoder.classes_ = np.array(encoders_data[task]['classes'])
        y_true[task] = encoder.transform(metadata_train[task].values)
    
    # Make predictions - single forward pass
    logger.info("\nMaking predictions...")
    
    X_tensor = torch.FloatTensor(X_train).to(device)
    
    predictions = {}
    with torch.no_grad():
        outputs = model(X_tensor)
        for task in task_names:
            predictions[task] = outputs[task].argmax(dim=1).cpu().numpy()
    
    # Calculate metrics
    logger.info("\n" + "="*60)
    logger.info("TRAINING SET PERFORMANCE")
    logger.info("="*60)
    
    training_metrics = {}
    for task in task_names:
        y_pred = predictions[task]
        y_t = y_true[task]
        
        training_metrics[task] = {
            'accuracy': float(accuracy_score(y_t, y_pred)),
            'balanced_accuracy': float(balanced_accuracy_score(y_t, y_pred)),
            'f1_weighted': float(f1_score(y_t, y_pred, average='weighted', zero_division=0)),
            'precision_macro': float(precision_score(y_t, y_pred, average='macro', zero_division=0)),
            'recall_macro': float(recall_score(y_t, y_pred, average='macro', zero_division=0))
        }
        
        logger.info(f"\n{task}:")
        for metric, value in training_metrics[task].items():
            logger.info(f"  {metric:20s}: {value:.4f} ({value*100:.2f}%)")
    
    # Save metrics
    output_path = Path("results/training/training_set_metrics.json")
    with open(output_path, 'w') as f:
        json.dump(training_metrics, f, indent=2)
    
    logger.info(f"\n✅ Metrics saved to {output_path}")
    logger.info("="*60)


if __name__ == "__main__":
    main()
