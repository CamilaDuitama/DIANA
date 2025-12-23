#!/usr/bin/env python
"""
Train final model using configuration from JSON file.
Called by run_final_training_gpu.sbatch.
"""
import argparse
import torch
import polars as pl
import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from diana.data.loader import MatrixLoader
from diana.models.multitask_mlp import MultiTaskMLP
from diana.training.trainer import MultiTaskTrainer

def main():
    # P1: Use argparse for robust CLI argument parsing
    parser = argparse.ArgumentParser(description='Train final multi-task model')
    parser.add_argument('config_file', type=Path, help='Path to training configuration JSON')
    args = parser.parse_args()
    
    config_file = args.config_file
    
    # Load configuration
    with open(config_file) as f:
        config = json.load(f)

    # P1: Hyperparameters now come pre-formatted in nested structure
    hyperparams = config['hyperparameters']
    
    logger.info('='*50)
    logger.info('Training final model with best hyperparameters')
    logger.info('='*50)
    logger.info(f'Model params: {json.dumps(hyperparams["model_params"], indent=2)}')
    logger.info(f'Trainer params: {json.dumps(hyperparams["trainer_params"], indent=2)}')
    logger.info(f'Batch size: {hyperparams["batch_size"]}')

    # Load data
    logger.info(f"Loading training data from {config['features_path']}")
    loader = MatrixLoader(Path(config['features_path']))
    X_full, metadata_pl = loader.load_with_metadata(
        metadata_path=Path(config['metadata_path']),
        align_to_matrix=True
    )
    
    # Convert metadata to pandas for compatibility
    metadata = metadata_pl.to_pandas()

    logger.info(f'Full training data shape: {X_full.shape}')
    logger.info(f'Total samples: {len(X_full)}')

    # Get task info and labels
    task_names = config['task_names']
    task_info = {}
    y_full = {}
    label_encoders = {}
    
    for task_name in task_names:
        # Encode string labels to integers
        encoder = LabelEncoder()
        y_full[task_name] = encoder.fit_transform(metadata[task_name].values)
        label_encoders[task_name] = encoder
        n_classes = len(encoder.classes_)
        task_info[task_name] = n_classes
        logger.info(f"Task '{task_name}': {n_classes} classes")

    # Split into sub-train and validation for early stopping
    validation_split = config['validation_split']
    logger.info(f'Creating validation split: {validation_split * 100:.0f}% for validation')

    indices = np.arange(len(X_full))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=validation_split,
        random_state=42,
        stratify=y_full[task_names[0]]
    )

    X_train, X_val = X_full[train_idx], X_full[val_idx]
    y_train = {task: y_full[task][train_idx] for task in task_names}
    y_val = {task: y_full[task][val_idx] for task in task_names}

    logger.info(f'Sub-train samples: {len(X_train)}')
    logger.info(f'Validation samples: {len(X_val)}')

    # P1: Initialize model with clean nested params (use unpacking)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f'Using device: {device}')
    
    model = MultiTaskMLP(
        input_dim=X_full.shape[1],
        num_classes=task_info,  # Changed from task_info to num_classes
        **hyperparams['model_params']  # Unpacks: hidden_dims, dropout, activation, use_batch_norm
    )

    # P1: Initialize trainer with clean nested params
    trainer = MultiTaskTrainer(
        model=model,
        task_names=task_names,
        device=device,
        learning_rate=hyperparams['trainer_params']['learning_rate'],
        weight_decay=hyperparams['trainer_params']['weight_decay'],
        task_weights=hyperparams['trainer_params']['task_weights']
    )

    # Train with early stopping
    logger.info('Starting training with validation-based early stopping...')
    history = trainer.fit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        max_epochs=config['max_epochs'],
        batch_size=hyperparams['batch_size'],
        patience=config['early_stopping_patience'],
        checkpoint_dir=Path(config['output_dir']),
        verbose=True
    )

    # Save final model
    model_path = Path(config['output_dir']) / 'best_model.pth'
    logger.info(f'Final model saved to: {model_path}')

    # P2: Save training history with proper type conversion
    history_path = Path(config['output_dir']) / 'training_history.json'
    
    # Convert NumPy types to Python types for JSON serialization
    def convert_to_python_types(obj):
        if isinstance(obj, dict):
            return {k: convert_to_python_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_python_types(v) for v in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    history_serializable = convert_to_python_types(history)
    
    with open(history_path, 'w') as f:
        json.dump(history_serializable, f, indent=2)
    logger.info(f'Training history saved to: {history_path}')
    
    # Save label encoders for later use in inference
    encoders_path = Path(config['output_dir']) / 'label_encoders.json'
    encoders_data = {
        task: {
            'classes': encoder.classes_.tolist()
        }
        for task, encoder in label_encoders.items()
    }
    with open(encoders_path, 'w') as f:
        json.dump(encoders_data, f, indent=2)
    logger.info(f'Label encoders saved to: {encoders_path}')

    logger.info('='*50)
    logger.info('Final training complete!')
    logger.info('='*50)

if __name__ == '__main__':
    main()
