#!/usr/bin/env python3
"""
Multi-Task MLP Hyperparameter Optimization - Single Fold for SLURM Array Jobs
==============================================================================

Performs nested cross-validation with Optuna-based Bayesian hyperparameter optimization
for multi-task classification of ancient DNA samples.

CLASSIFICATION TARGETS (Multi-Task Learning):
----------------------------------------------
1. sample_type (Binary): ancient_metagenome vs modern_metagenome
2. community_type (6 classes): oral, skeletal tissue, gut, plant tissue, soft tissue, env sample
3. sample_host (12 classes): Homo sapiens, Ursus arctos, environmental, etc.
4. material (13 classes): dental calculus, tooth, bone, sediment, etc.

DEPENDENCIES:
-------------
Python packages:
  - numpy, pandas, polars (data manipulation)
  - torch (PyTorch for neural networks)
  - scikit-learn (metrics, cross-validation, label encoding)
  - optuna (Bayesian hyperparameter optimization)

Internal modules:
  - diana.models.multitask_mlp: MultiTaskMLP, MultiTaskLoss
  - diana.data.loader: MatrixLoader (polars-based fast loading)

Input files:
  - K-mer matrix (.pa.mat): Space-separated file with sample IDs in column 0
  - Metadata (.tsv): TSV with Run_accession and classification targets

INPUT DATA:
-----------
- Features: data/splits/train_matrix.pa.mat (2609 samples × 104565 k-mer features)
- Metadata: data/splits/train_metadata.tsv (sample IDs + target labels)

OUTPUT STRUCTURE:
-----------------
results_multitask_gpu/fold_{fold_id}/
├── multitask_fold_{fold_id}_results_{timestamp}.json    # Metrics + hyperparameters
├── best_multitask_model_fold_{fold_id}_{timestamp}.pth  # Trained model weights
└── fold_{fold_id}_training_log_{timestamp}.txt          # Detailed training log

USAGE:
------
# Local testing (CPU):
python scripts/training/01_train_multitask_single_fold.py \\
    --fold_id 0 --total_folds 2 \\
    --features data/test_data/splits/train_matrix_100feat.pa.mat \\
    --metadata data/test_data/splits/train_metadata.tsv \\
    --output results/test --n_trials 3 --max_epochs 20

# SLURM GPU (5 folds parallel):
sbatch --array=0-4 scripts/training/run_multitask_gpu.sbatch

Each array task trains one outer CV fold independently with Optuna optimization.

WORKFLOW:
---------
1. Load matrix (polars) and metadata
2. Split into outer CV fold (stratified by combined sample_type and community_type)
3. Optuna hyperparameter search with inner CV:
   - Search space: hidden layers, activation, dropout, batch norm, learning rate, etc.
   - Objective: Average of (balanced accuracy + macro F1) / 2 across all 4 tasks
   - No pruning (each trial evaluated on all inner folds)
4. Train final model on full training fold with best hyperparameters
5. Save model, hyperparameters, and metrics
"""

import sys
import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score,
    precision_score, recall_score, classification_report
)

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from diana.models.multitask_mlp import MultiTaskMLP, MultiTaskLoss
from diana.data.loader import MatrixLoader
from diana.config import ConfigManager
from diana.utils.config import setup_logging
from diana.utils.checkpointing import CheckpointManager

# Logger will be initialized in main() after args are parsed
logger = None


def load_matrix_data(matrix_path: str, metadata_path: str) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Load k-mer matrix and metadata using MatrixLoader class.
    
    Args:
        matrix_path: Path to .pa.mat file
        metadata_path: Path to metadata TSV
        
    Returns:
        Tuple of (features_matrix, metadata_df as pandas for sklearn compatibility)
    """
    # Use MatrixLoader class for loading
    loader = MatrixLoader(Path(matrix_path))
    features, metadata_pl = loader.load_with_metadata(
        metadata_path=Path(metadata_path),
        align_to_matrix=True
    )
    
    # Convert metadata to pandas for sklearn compatibility
    metadata = metadata_pl.to_pandas()
    
    logger.info(f"Loaded {features.shape[0]} samples with {features.shape[1]} features")
    
    return features, metadata


def prepare_labels(
    metadata: pd.DataFrame,
    targets: List[str] = ["sample_type", "community_type", "sample_host", "material"]
) -> Tuple[Dict[str, np.ndarray], Dict[str, LabelEncoder], Dict[str, int]]:
    """
    Encode labels for all classification targets.
    Dynamically determines number of classes from actual data.
    
    Args:
        metadata: Metadata DataFrame
        targets: List of target column names
        
    Returns:
        Tuple of (labels_dict, encoders_dict, num_classes_dict)
    """
    labels = {}
    encoders = {}
    num_classes = {}
    
    for target in targets:
        encoder = LabelEncoder()
        labels[target] = encoder.fit_transform(metadata[target].values)
        encoders[target] = encoder
        num_classes[target] = len(encoder.classes_)
        
        logger.info(f"{target}: {num_classes[target]} classes - {list(encoder.classes_[:5])}...")
    
    return labels, encoders, num_classes


def compute_class_weights(labels_dict: Dict[str, np.ndarray], num_classes: Dict[str, int], device: torch.device) -> Dict[str, torch.Tensor]:
    """
    Compute class weights for imbalanced datasets.
    
    Args:
        labels_dict: Dictionary of labels for each task
        num_classes: Dictionary of total number of classes per task
        device: Torch device
        
    Returns:
        Dictionary of class weights tensors (with correct size for all classes)
    """
    class_weights = {}
    
    for task_name, labels in labels_dict.items():
        n_total_classes = num_classes[task_name]
        unique, counts = np.unique(labels, return_counts=True)
        total = len(labels)
        
        # Initialize weights for all classes with 1.0 (neutral weight for missing classes)
        weights = np.ones(n_total_classes, dtype=np.float32)
        
        # Compute weights for classes present in this split
        for cls_idx, count in zip(unique, counts):
            weights[cls_idx] = total / (len(unique) * count)
        
        class_weights[task_name] = torch.FloatTensor(weights).to(device)
        
        logger.info(f"{task_name} class weights (present classes): {dict(zip(unique, weights[unique]))}")
    
    return class_weights



def train_outer_fold(
    fold_id: int,
    total_folds: int,
    features: np.ndarray,
    metadata: pd.DataFrame,
    config: Dict[str, Any],
    output_dir: Path
) -> Dict[str, Any]:
    """
    Train one outer CV fold with hyperparameter optimization.
    
    Args:
        fold_id: Fold identifier (0-indexed)
        total_folds: Total number of folds
        features: Feature matrix
        metadata: Metadata DataFrame
        config: Configuration dictionary
        output_dir: Output directory
        
    Returns:
        Results dictionary
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    fold_dir = output_dir / f"fold_{fold_id}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"=== FOLD {fold_id}/{total_folds-1} ===")
    logger.info(f"Output directory: {fold_dir}")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() and config.get('use_gpu', True) else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Prepare labels
    targets = ["sample_type", "community_type", "sample_host", "material"]
    labels_dict, encoders, num_classes = prepare_labels(metadata, targets)
    
    # Outer CV split (stratify by combined sample_type + community_type for better balance)
    # Create stratification key combining sample_type and community_type
    stratify_key = [
        f"{labels_dict['sample_type'][i]}_{labels_dict['community_type'][i]}"
        for i in range(len(labels_dict['sample_type']))
    ]
    
    skf_outer = StratifiedKFold(n_splits=total_folds, shuffle=True, random_state=42)
    splits = list(skf_outer.split(features, stratify_key))
    train_idx, test_idx = splits[fold_id]
    
    logger.info(f"Train: {len(train_idx)}, Test: {len(test_idx)}")
    
    # Inner CV for hyperparameter optimization
    n_inner_splits = config.get("n_inner_splits", 3)
    skf_inner = StratifiedKFold(n_splits=n_inner_splits, shuffle=True, random_state=42)
    
    # Precompute inner CV splits (stratify by combined sample_type + community_type)
    inner_stratify_key = [
        f"{labels_dict['sample_type'][train_idx[i]]}_{labels_dict['community_type'][train_idx[i]]}"
        for i in range(len(train_idx))
    ]
    inner_cv_splits = list(skf_inner.split(
        features[train_idx],
        inner_stratify_key
    ))
    
    logger.info(f"Starting Optuna optimization with {config.get('n_trials', 50)} trials...")
    logger.info(f"Each trial will be evaluated on {n_inner_splits} inner CV folds")
    
    def objective_with_cv(trial: optuna.Trial) -> float:
        """
        Objective function that evaluates each trial on ALL inner CV folds.
        Returns the average performance across folds for robust hyperparameter selection.
        """
        # Hyperparameters to optimize
        n_layers = trial.suggest_int("n_layers", 2, 4)
        hidden_dims = []
        
        for i in range(n_layers):
            hidden_dim = trial.suggest_int(f"hidden_dim_{i}", 64, 512, step=64)
            hidden_dims.append(hidden_dim)
        
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
        use_batch_norm = trial.suggest_categorical("use_batch_norm", [True, False])
        activation = trial.suggest_categorical("activation", ["relu", "gelu", "leaky_relu"])
        
        # Task loss weights
        task_weight_sample_type = trial.suggest_float("task_weight_sample_type", 0.5, 2.0)
        task_weight_community = trial.suggest_float("task_weight_community", 0.5, 2.0)
        task_weight_host = trial.suggest_float("task_weight_host", 0.5, 2.0)
        task_weight_material = trial.suggest_float("task_weight_material", 0.5, 2.0)
        
        # Per-task label smoothing (fixed at 0 when --no_label_smoothing is set)
        if config.get('no_label_smoothing', False):
            ls_sample_type = ls_community_type = ls_sample_host = ls_material = 0.0
        else:
            ls_sample_type = trial.suggest_float("ls_sample_type", 0.0, 0.15)
            ls_community_type = trial.suggest_float("ls_community_type", 0.0, 0.15)
            ls_sample_host = trial.suggest_float("ls_sample_host", 0.0, 0.15)
            ls_material = trial.suggest_float("ls_material", 0.0, 0.15)
        
        task_weights = {
            "sample_type": task_weight_sample_type,
            "community_type": task_weight_community,
            "sample_host": task_weight_host,
            "material": task_weight_material
        }
        
        label_smoothing_per_task = {
            "sample_type": ls_sample_type,
            "community_type": ls_community_type,
            "sample_host": ls_sample_host,
            "material": ls_material,
        }
        
        # Evaluate on ALL inner CV folds
        fold_scores = []
        
        for inner_fold_idx, (inner_train, inner_val) in enumerate(inner_cv_splits):
            # Get actual indices
            fold_train_idx = train_idx[inner_train]
            fold_val_idx = train_idx[inner_val]
            
            # Prepare data as tensors
            X_train = torch.FloatTensor(features[fold_train_idx])
            X_val = torch.FloatTensor(features[fold_val_idx])
            
            y_train = {task: torch.LongTensor(labels_dict[task][fold_train_idx]) 
                      for task in num_classes.keys()}
            y_val = {task: torch.LongTensor(labels_dict[task][fold_val_idx]) 
                    for task in num_classes.keys()}
            
            # Create DataLoaders with proper batch_size
            train_dataset = TensorDataset(
                X_train,
                *[y_train[task] for task in num_classes.keys()]
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=False
            )
            
            val_dataset = TensorDataset(
                X_val,
                *[y_val[task] for task in num_classes.keys()]
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False
            )
            
            # Create model
            model = MultiTaskMLP(
                input_dim=features.shape[1],
                hidden_dims=hidden_dims,
                num_classes=num_classes,
                dropout=dropout,
                use_batch_norm=use_batch_norm,
                activation=activation
            ).to(device)
            
            # Compute class weights for this fold
            class_weights = compute_class_weights(
                {task: labels_dict[task][fold_train_idx] for task in num_classes.keys()},
                num_classes,
                device
            )
            
            # Loss and optimizer
            criterion = MultiTaskLoss(
                task_names=list(num_classes.keys()),
                task_weights=task_weights,
                class_weights=class_weights,
                label_smoothing=label_smoothing_per_task,
            ).to(device)
            
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
            
            # Training
            max_epochs = config.get("max_epochs", 100)
            patience = config.get("patience", 15)
            best_val_score = 0
            patience_counter = 0
            
            for epoch in range(max_epochs):
                model.train()
                
                # Training loop with mini-batches
                for batch_data in train_loader:
                    X_batch = batch_data[0].to(device)
                    y_batch = {task: batch_data[i+1].to(device) for i, task in enumerate(num_classes.keys())}
                    
                    # Forward pass
                    outputs = model(X_batch)
                    total_loss, task_losses = criterion(outputs, y_batch)
                    
                    # Backward pass
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()
                
                # Validation every 5 epochs
                if epoch % 5 == 0:
                    model.eval()
                    val_f1_scores = []
                    
                    with torch.no_grad():
                        all_preds = {task: [] for task in num_classes.keys()}
                        all_true = {task: [] for task in num_classes.keys()}
                        
                        for batch_data in val_loader:
                            X_batch = batch_data[0].to(device)
                            y_batch = {task: batch_data[i+1].to(device) for i, task in enumerate(num_classes.keys())}
                            
                            val_outputs = model(X_batch)
                            
                            for task in num_classes.keys():
                                preds = torch.argmax(val_outputs[task], dim=1).cpu().numpy()
                                true = y_batch[task].cpu().numpy()
                                all_preds[task].extend(preds)
                                all_true[task].extend(true)
                        
                        # Compute balanced accuracy and macro F1 for all tasks
                        for task in num_classes.keys():
                            bal_acc = balanced_accuracy_score(all_true[task], all_preds[task])
                            macro_f1 = f1_score(all_true[task], all_preds[task], average='macro', zero_division=0)
                            # Average of balanced accuracy and macro F1
                            task_score = (bal_acc + macro_f1) / 2.0
                            val_f1_scores.append(task_score)
                        
                        avg_score = np.mean(val_f1_scores)
                    
                    # Early stopping
                    if avg_score > best_val_score:
                        best_val_score = avg_score
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    
                    if patience_counter >= patience:
                        break
            
            # Store this fold's best score
            fold_scores.append(best_val_score)
        
        # Return average score across all inner folds
        avg_score = np.mean(fold_scores)
        return avg_score
    
    # Create study and optimize
    # Note: Using NopPruner because each trial is evaluated on multiple folds
    # and pruning mid-trial would discard incomplete fold evaluations
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42),
        pruner=optuna.pruners.NopPruner()  # No pruning for multi-fold CV
    )
    
    study.optimize(
        objective_with_cv,
        n_trials=config.get('n_trials', 50),
        show_progress_bar=False
    )
    
    best_params = study.best_params
    logger.info(f"Best hyperparameters (averaged over {n_inner_splits} folds): {best_params}")
    logger.info(f"Best CV score: {study.best_value:.4f}")
    
    # Train final model with best hyperparameters
    # Split train_idx into sub-train and sub-val to avoid test set leakage during early stopping
    logger.info("Training final model with proper train/val split...")
    
    # Compute class weights on full train_idx before splitting
    class_weights_full = compute_class_weights(
        {task: labels_dict[task][train_idx] for task in num_classes.keys()},
        num_classes,
        device
    )
    
    sub_train_idx, sub_val_idx = train_test_split(
        train_idx,
        test_size=0.1,  # Use 10% of training fold for validation
        random_state=42,
        stratify=[
            f"{labels_dict['sample_type'][i]}_{labels_dict['community_type'][i]}"
            for i in train_idx
        ]
    )
    
    logger.info(f"Final training split: {len(sub_train_idx)} train, {len(sub_val_idx)} validation, {len(test_idx)} test")
    
    # Build model with best params
    hidden_dims = [best_params[f"hidden_dim_{i}"] for i in range(best_params["n_layers"])]
    
    model = MultiTaskMLP(
        input_dim=features.shape[1],
        hidden_dims=hidden_dims,
        num_classes=num_classes,
        dropout=best_params["dropout"],
        use_batch_norm=best_params["use_batch_norm"],
        activation=best_params["activation"]
    ).to(device)
    
    # Prepare final training data (train/val for early stopping, test for final eval only)
    X_train = torch.FloatTensor(features[sub_train_idx])
    X_val = torch.FloatTensor(features[sub_val_idx])
    X_test = torch.FloatTensor(features[test_idx])
    
    y_train = {task: torch.LongTensor(labels[sub_train_idx]) for task, labels in labels_dict.items()}
    y_val = {task: torch.LongTensor(labels[sub_val_idx]) for task, labels in labels_dict.items()}
    y_test = {task: torch.LongTensor(labels[test_idx]) for task, labels in labels_dict.items()}
    
    # Create DataLoaders with best batch_size
    train_dataset = TensorDataset(
        X_train,
        *[y_train[task] for task in num_classes.keys()]
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(round(best_params["batch_size"])),
        shuffle=True,
        drop_last=False
    )
    
    val_dataset = TensorDataset(
        X_val,
        *[y_val[task] for task in num_classes.keys()]
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(round(best_params["batch_size"])),
        shuffle=False
    )
    
    test_dataset = TensorDataset(
        X_test,
        *[y_test[task] for task in num_classes.keys()]
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=int(round(best_params["batch_size"])),
        shuffle=False
    )
    
    # Task weights from best params
    task_weights = {
        "sample_type": best_params["task_weight_sample_type"],
        "community_type": best_params["task_weight_community"],
        "sample_host": best_params["task_weight_host"],
        "material": best_params["task_weight_material"]
    }
    
    label_smoothing_per_task = {
        "sample_type":    best_params.get("ls_sample_type",    0.0),
        "community_type": best_params.get("ls_community_type", 0.0),
        "sample_host":    best_params.get("ls_sample_host",    0.0),
        "material":       best_params.get("ls_material",       0.0),
    }
    criterion = MultiTaskLoss(
        task_names=list(num_classes.keys()),
        task_weights=task_weights,
        class_weights=class_weights_full,
        label_smoothing=label_smoothing_per_task,
    ).to(device)
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=best_params["learning_rate"],
        weight_decay=best_params["weight_decay"]
    )
    
    # Initialize checkpoint manager
    checkpoint_mgr = CheckpointManager(
        output_dir=fold_dir,
        save_best=True,
        save_frequency=config.get("checkpoint_freq", 10),
        save_final=True,
        keep_last_n=5
    )
    
    # Resume from checkpoint if requested
    start_epoch = 0
    if config.get("resume_from"):
        try:
            start_epoch = checkpoint_mgr.resume_from_checkpoint(
                model, optimizer, config["resume_from"]
            )
            logger.info(f"Resumed from checkpoint at epoch {start_epoch}")
        except Exception as e:
            logger.warning(f"Could not resume from checkpoint: {e}")
    
    # Train
    max_epochs = config.get("max_epochs", 200)
    patience = config.get("patience", 20)
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(start_epoch, max_epochs):
        model.train()
        
        # Training loop with mini-batches
        train_loss_sum = 0.0
        for batch_data in train_loader:
            X_batch = batch_data[0].to(device)
            y_batch = {task: batch_data[i+1].to(device) for i, task in enumerate(num_classes.keys())}
            
            outputs = model(X_batch)
            total_loss, task_losses = criterion(outputs, y_batch)
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            train_loss_sum += total_loss.item()
        
        avg_train_loss = train_loss_sum / len(train_loader)
        
        # Validation (use proper validation set, NOT test set)
        model.eval()
        val_loss_sum = 0.0
        
        with torch.no_grad():
            for batch_data in val_loader:
                X_batch = batch_data[0].to(device)
                y_batch = {task: batch_data[i+1].to(device) for i, task in enumerate(num_classes.keys())}
                
                val_outputs = model(X_batch)
                val_loss, _ = criterion(val_outputs, y_batch)
                
                val_loss_sum += val_loss.item()
        
        avg_val_loss = val_loss_sum / len(val_loader)
        
        # Save checkpoint
        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        checkpoint_mgr.save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch + 1,
            metrics={"train_loss": avg_train_loss, "val_loss": avg_val_loss},
            hyperparams=best_params,
            is_best=is_best
        )
        
        if (epoch + 1) % 20 == 0:
            logger.info(f"Epoch {epoch + 1}/{max_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Early stopping
        if patience_counter >= patience:
            logger.info(f"Early stopping triggered after {epoch + 1} epochs (patience={patience})")
            break
    
    # Load best model for final evaluation
    best_checkpoint = checkpoint_mgr.load_best_model()
    if best_checkpoint:
        model.load_state_dict(best_checkpoint['model_state_dict'])
        logger.info(f"Loaded best model from epoch {best_checkpoint.get('epoch', '?')} (val_loss: {best_checkpoint['metrics']['val_loss']:.4f})")
    
    # Evaluate on test set ONCE (never used for model selection)
    logger.info("Evaluating on held-out test set...")
    model.eval()
    
    with torch.no_grad():
        all_preds = {task: [] for task in num_classes.keys()}
        all_true = {task: [] for task in num_classes.keys()}
        
        for batch_data in test_loader:
            X_batch = batch_data[0].to(device)
            y_batch = {task: batch_data[i+1].to(device) for i, task in enumerate(num_classes.keys())}
            
            test_outputs = model(X_batch)
            
            for task in num_classes.keys():
                preds = torch.argmax(test_outputs[task], dim=1).cpu().numpy()
                true = y_batch[task].cpu().numpy()
                all_preds[task].extend(preds)
                all_true[task].extend(true)
        
        test_metrics = {}
        for task in num_classes.keys():
            test_metrics[task] = {
                "accuracy": float(accuracy_score(all_true[task], all_preds[task])),
                "f1_weighted": float(f1_score(all_true[task], all_preds[task], average='weighted', zero_division=0)),
                "f1_macro": float(f1_score(all_true[task], all_preds[task], average='macro', zero_division=0)),
                "balanced_accuracy": float(balanced_accuracy_score(all_true[task], all_preds[task]))
            }
            
            logger.info(f"{task} test metrics: {test_metrics[task]}")
    
    # Save final model with checkpoint manager
    final_model_path = checkpoint_mgr.save_final_model(
        model=model,
        metrics=test_metrics,
        hyperparams=best_params
    )
    logger.info(f"Final model saved to {final_model_path}")
    
    # Save results
    results = {
        "fold_id": fold_id,
        "total_folds": total_folds,
        "n_outer_train": len(train_idx),
        "n_sub_train": len(sub_train_idx),
        "n_sub_val": len(sub_val_idx),
        "n_test": len(test_idx),
        "num_classes": num_classes,
        "best_params": best_params,
        "test_metrics": test_metrics,
        "best_model_path": str(fold_dir / "best_model.pth"),
        "final_model_path": str(final_model_path),
        "timestamp": timestamp,
        "note": "Test set ONLY used for final evaluation, never for model selection"
    }
    
    results_path = fold_dir / f"multitask_fold_{fold_id}_results_{timestamp}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {results_path}")
    logger.info("=" * 80)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Multi-task MLP hyperparameter optimization')
    
    # Configuration
    parser.add_argument('--config', type=Path, help='YAML configuration file (overrides other args)')
    
    # Fold parameters
    parser.add_argument('--fold_id', type=int, required=True, help='Fold ID (0-indexed)')
    parser.add_argument('--total_folds', type=int, default=5, help='Total number of folds')
    
    # Data paths
    parser.add_argument('--features', type=str, help='Feature matrix path (.pa.mat file)')
    parser.add_argument('--metadata', type=str, help='Metadata path (.tsv file)')
    parser.add_argument('--output', type=str, default='results/experiments/multitask', help='Output directory')
    
    # Training parameters
    parser.add_argument('--n_trials', type=int, default=50, help='Number of Optuna trials')
    parser.add_argument('--max_epochs', type=int, default=200, help='Maximum training epochs')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--n_inner_splits', type=int, default=3, help='Number of inner CV folds')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed')
    
    # Hardware
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU if available')

    # Label smoothing control
    parser.add_argument('--no_label_smoothing', action='store_true',
                        help='Fix all label smoothing at 0 (remove from search space). Use for v5+.')

    # Checkpointing
    parser.add_argument('--resume_from', type=Path, help='Resume from checkpoint')
    parser.add_argument('--checkpoint_freq', type=int, default=10, help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    
    # Load configuration if provided
    config = None
    if args.config:
        try:
            config = ConfigManager.from_yaml(args.config)
            setup_logging(
                log_file=Path(args.output) / f"fold_{args.fold_id}_training.log",
                level=config.get("logging.level", "INFO"),
                log_to_console=config.get("logging.log_to_console", True),
                log_to_file=config.get("logging.log_to_file", True)
            )
        except Exception as e:
            print(f"Error loading config: {e}")
            sys.exit(1)
    else:
        setup_logging(
            log_file=Path(args.output) / f"fold_{args.fold_id}_training.log",
            level="INFO",
            log_to_console=True,
            log_to_file=True
        )
    
    # Set global logger
    global logger
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info("DIANA Multi-Task Hyperparameter Optimization")
    logger.info("=" * 80)
    
    # Get parameters (config overrides command-line args)
    features_path = config.get("data.train_matrix") if config else args.features
    metadata_path = config.get("data.train_metadata") if config else args.metadata
    output_dir = Path(config.get("output.base_dir") if config else args.output)
    n_trials = config.get("training.n_trials") if config else args.n_trials
    max_epochs = config.get("training.max_epochs") if config else args.max_epochs
    patience = config.get("training.early_stopping_patience") if config else args.patience
    n_inner_splits = config.get("training.n_inner_splits") if config else args.n_inner_splits
    random_seed = config.get("training.random_seed") if config else args.random_seed
    use_gpu = config.get("training.use_gpu") if config else args.use_gpu
    checkpoint_freq = config.get("output.checkpoint_frequency") if config else args.checkpoint_freq
    
    # Validate required parameters
    if not features_path or not metadata_path:
        logger.error("Must provide --features and --metadata, or --config with data paths")
        sys.exit(1)
    
    # Save configuration for reproducibility
    fold_dir = output_dir / f"fold_{args.fold_id}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    
    if config:
        config.save(fold_dir / "config_used.yaml")
        logger.info(f"Configuration saved to {fold_dir / 'config_used.yaml'}")
    
    logger.info(f"Fold: {args.fold_id}/{args.total_folds}")
    logger.info(f"Features: {features_path}")
    logger.info(f"Metadata: {metadata_path}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Trials: {n_trials}, Max Epochs: {max_epochs}, Patience: {patience}")
    logger.info(f"GPU: {use_gpu}, Random Seed: {random_seed}")
    
    # Set random seeds
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)  # For multi-GPU
    
    # Load data
    try:
        logger.info(f"Loading data from {features_path}")
        features, metadata = load_matrix_data(features_path, metadata_path)
        logger.info(f"Loaded {features.shape[0]} samples × {features.shape[1]} features")
    except Exception as e:
        logger.error(f"Failed to load data: {e}", exc_info=True)
        sys.exit(1)
    
    # Configuration for training
    train_config = {
        'n_trials': n_trials,
        'max_epochs': max_epochs,
        'patience': patience,
        'use_gpu': use_gpu,
        'n_inner_splits': n_inner_splits,
        'checkpoint_freq': checkpoint_freq,
        'resume_from': args.resume_from,
        'no_label_smoothing': args.no_label_smoothing,
    }
    
    # Train fold
    try:
        results = train_outer_fold(
            fold_id=args.fold_id,
            total_folds=args.total_folds,
            features=features,
            metadata=metadata,
            config=train_config,
            output_dir=output_dir
        )
        
        logger.info("=" * 80)
        logger.info("=== FOLD COMPLETE ===")
        logger.info("=" * 80)
        logger.info(f"Results: {results['test_metrics']}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
