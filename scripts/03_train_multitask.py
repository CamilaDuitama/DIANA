#!/usr/bin/env python3
"""Train multi-task MLP model."""

import sys
import argparse
import logging
import torch
from pathlib import Path
from torch.utils.data import DataLoader

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from diana.data.dataset import DianaDataset
from diana.models.multitask_mlp import MultiTaskMLP
from diana.training.trainer import MultiTaskTrainer
from diana.utils.config import load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Train multi-task model')
    parser.add_argument('--config', default='configs/model_multitask.yaml',
                       help='Path to model configuration file')
    parser.add_argument('--data-config', default='configs/data_config.yaml',
                       help='Path to data configuration file')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    args = parser.parse_args()

    # Load configurations
    model_config = load_config(Path(args.config))
    data_config = load_config(Path(args.data_config))
    
    splits_dir = Path(data_config["splits_dir"])
    
    # Check if splits exist
    if not (splits_dir / "train_matrix.pa.mat").exists():
        logger.error("Train matrix not found. Run 01_create_splits.py --extract-matrices first.")
        sys.exit(1)
        
    # Initialize Datasets
    logger.info("Initializing datasets...")
    
    # We use the extracted matrices
    train_dataset = DianaDataset(
        matrix_path=splits_dir / "train_matrix.pa.mat",
        metadata_path=splits_dir / "train_metadata.tsv", # Created by splitter? No, splitter creates train_ids.txt
        # Wait, MatrixExtractor creates matrices but not metadata files?
        # The reference script created metadata files. My splitter creates train_ids.txt.
        # I should use the main metadata file and filter by IDs, or update splitter to save metadata files.
        # DianaDataset supports filtering by IDs.
        metadata_path=data_config["metadata_path"],
        sample_ids_path=splits_dir / "train_ids.txt",
        targets=data_config["targets"]
    )
    
    val_dataset = DianaDataset(
        matrix_path=splits_dir / "val_matrix.pa.mat",
        metadata_path=data_config["metadata_path"],
        sample_ids_path=splits_dir / "val_ids.txt",
        targets=data_config["targets"]
    )
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Initialize Model
    logger.info("Initializing model...")
    input_dim = train_dataset.features.shape[1]
    
    # Get num_classes from dataset
    num_classes = {
        target: len(encoder) 
        for target, encoder in train_dataset.label_encoders.items()
    }
    
    model = MultiTaskMLP(
        input_dim=input_dim,
        hidden_dims=model_config.get("hidden_dims", [512, 256, 128]),
        num_classes=num_classes,
        dropout=model_config.get("dropout", 0.5)
    )
    
    # Initialize Trainer
    trainer = MultiTaskTrainer(
        model=model,
        learning_rate=args.lr
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        save_dir=Path("models/multitask")
    )
    
    logger.info("Training complete!")

if __name__ == "__main__":
    main()