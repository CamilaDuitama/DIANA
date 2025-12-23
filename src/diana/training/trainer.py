"""Training loops for multi-task and single-task models."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Optional, List, Union
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm


class MultiTaskTrainer:
    """Trainer for multi-task classification."""
    
    def __init__(self,
                 model: nn.Module,
                 task_names: List[str],
                 device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
                 learning_rate: float = 1e-3,
                 weight_decay: float = 0.0,
                 task_weights: Optional[Dict[str, float]] = None):
        """
        Initialize trainer.
        
        Args:
            model: Multi-task model
            task_names: List of task names
            device: Device to train on
            learning_rate: Learning rate
            weight_decay: L2 regularization weight decay
            task_weights: Weights for each task loss
        """
        if isinstance(device, str):
            device = torch.device(device)
        
        self.model = model.to(device)
        self.device = device
        self.task_names = task_names
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Task-specific loss functions
        self.criteria = {
            target: nn.CrossEntropyLoss()
            for target in task_names
        }
        
        # Task weights (default: equal weighting)
        if task_weights is None:
            task_weights = {target: 1.0 for target in task_names}
        self.task_weights = task_weights
        
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": {target: [] for target in task_names},
            "val_acc": {target: [] for target in task_names}
        }
        
    def fit(self, 
            X_train: Union[np.ndarray, DataLoader],
            y_train: Union[Dict[str, np.ndarray], None] = None,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[Dict[str, np.ndarray]] = None,
            train_loader: Optional[DataLoader] = None,
            val_loader: Optional[DataLoader] = None,
            max_epochs: int = 200,
            batch_size: int = 32,
            patience: int = 20,
            checkpoint_dir: Optional[Path] = None,
            verbose: bool = True) -> Dict:
        """
        Train model with validation-based early stopping.
        
        This is the main training method that implements proper early stopping
        based on validation loss. It can accept either numpy arrays or DataLoaders.
        
        Args:
            X_train: Training features (numpy array) or DataLoader
            y_train: Training labels dict (required if X_train is array)
            X_val: Validation features (numpy array, optional)
            y_val: Validation labels dict (required if X_val provided)
            train_loader: Training DataLoader (alternative to X_train/y_train)
            val_loader: Validation DataLoader (alternative to X_val/y_val)
            max_epochs: Maximum number of training epochs
            batch_size: Batch size for training
            patience: Early stopping patience (epochs without improvement)
            checkpoint_dir: Directory to save best model checkpoint
            verbose: Whether to print progress
            
        Returns:
            Dictionary with training history (train_loss, val_loss, etc.)
        """
        # Create DataLoaders if numpy arrays provided
        if train_loader is None:
            if isinstance(X_train, np.ndarray):
                X_tensor = torch.FloatTensor(X_train)
                y_tensors = {task: torch.LongTensor(y_train[task]) for task in self.task_names}
                
                # Create TensorDataset
                dataset = TensorDataset(
                    X_tensor,
                    *[y_tensors[task] for task in self.task_names]
                )
                train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            else:
                train_loader = X_train  # Assume it's already a DataLoader
        
        if val_loader is None and X_val is not None:
            X_tensor = torch.FloatTensor(X_val)
            y_tensors = {task: torch.LongTensor(y_val[task]) for task in self.task_names}
            
            dataset = TensorDataset(
                X_tensor,
                *[y_tensors[task] for task in self.task_names]
            )
            val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # Training loop with validation-based early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': {task: [] for task in self.task_names},
            'val_acc': {task: [] for task in self.task_names}
        }
        
        if checkpoint_dir:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            best_model_path = checkpoint_dir / "best_model.pth"
        
        for epoch in range(max_epochs):
            if verbose and (epoch + 1) % 10 == 0:
                print(f"\nEpoch {epoch+1}/{max_epochs}")
            
            # Training phase
            train_metrics = self._train_epoch_from_loader(train_loader)
            history['train_loss'].append(train_metrics['loss'])
            for task in self.task_names:
                history['train_acc'][task].append(train_metrics['accuracy'][task])
            
            # Validation phase
            if val_loader is not None:
                val_metrics = self._validate_from_loader(val_loader)
                history['val_loss'].append(val_metrics['loss'])
                for task in self.task_names:
                    history['val_acc'][task].append(val_metrics['accuracy'][task])
                
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Train Loss: {train_metrics['loss']:.4f}, Val Loss: {val_metrics['loss']:.4f}")
                    for task in self.task_names:
                        print(f"  {task}: Train Acc={train_metrics['accuracy'][task]:.4f}, "
                              f"Val Acc={val_metrics['accuracy'][task]:.4f}")
                
                # Early stopping based on validation loss
                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    patience_counter = 0
                    
                    # Save best model
                    if checkpoint_dir:
                        checkpoint = {
                            'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'val_loss': best_val_loss,
                            'history': history
                        }
                        torch.save(checkpoint, best_model_path)
                        if verbose and (epoch + 1) % 10 == 0:
                            print(f"  → Saved best model (val_loss: {best_val_loss:.4f})")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        if verbose:
                            print(f"\nEarly stopping at epoch {epoch+1}")
                            print(f"Best validation loss: {best_val_loss:.4f}")
                        break
            else:
                # No validation set - just log training metrics
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Train Loss: {train_metrics['loss']:.4f}")
                    for task in self.task_names:
                        print(f"  {task}: Train Acc={train_metrics['accuracy'][task]:.4f}")
        
        # Load best model if checkpoint exists
        if checkpoint_dir and best_model_path.exists():
            checkpoint = torch.load(best_model_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            if verbose:
                print(f"\nLoaded best model from epoch {checkpoint['epoch']+1}")
        
        return history
    
    def _train_epoch_from_loader(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch using a DataLoader."""
        self.model.train()
        
        total_loss = 0
        task_correct = {target: 0 for target in self.task_names}
        task_total = {target: 0 for target in self.task_names}
        
        for batch in train_loader:
            # Unpack batch (X, task1_y, task2_y, ...)
            batch_x = batch[0].to(self.device)
            batch_y = {
                task: batch[i+1].to(self.device)
                for i, task in enumerate(self.task_names)
            }
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(batch_x)
            
            # Compute losses
            losses = {}
            for target in self.task_names:
                loss = self.criteria[target](outputs[target], batch_y[target])
                losses[target] = loss
                
                # Accuracy
                _, predicted = torch.max(outputs[target], 1)
                task_correct[target] += (predicted == batch_y[target]).sum().item()
                task_total[target] += batch_y[target].size(0)
            
            # Combined loss
            total = sum(self.task_weights[t] * losses[t] for t in self.task_names)
            total_loss += total.item()
            
            # Backward pass
            total.backward()
            self.optimizer.step()
        
        # Compute metrics
        metrics = {
            "loss": total_loss / len(train_loader),
            "accuracy": {
                target: task_correct[target] / task_total[target] if task_total[target] > 0 else 0
                for target in self.task_names
            }
        }
        
        return metrics
    
    def _validate_from_loader(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate model using a DataLoader."""
        self.model.eval()
        
        total_loss = 0
        task_correct = {target: 0 for target in self.task_names}
        task_total = {target: 0 for target in self.task_names}
        
        with torch.no_grad():
            for batch in val_loader:
                # Unpack batch
                batch_x = batch[0].to(self.device)
                batch_y = {
                    task: batch[i+1].to(self.device)
                    for i, task in enumerate(self.task_names)
                }
                
                outputs = self.model(batch_x)
                
                # Losses and accuracy
                losses = []
                for target in self.task_names:
                    loss = self.criteria[target](outputs[target], batch_y[target])
                    losses.append(self.task_weights[target] * loss.item())
                    
                    _, predicted = torch.max(outputs[target], 1)
                    task_correct[target] += (predicted == batch_y[target]).sum().item()
                    task_total[target] += batch_y[target].size(0)
                
                total_loss += sum(losses)
        
        metrics = {
            "loss": total_loss / len(val_loader) if len(val_loader) > 0 else 0,
            "accuracy": {
                target: task_correct[target] / task_total[target] if task_total[target] > 0 else 0
                for target in self.task_names
            }
        }
        
        return metrics
    
    def train_epoch(self, X, y, batch_size=32):
        """
        Legacy method for backward compatibility.
        Trains for one epoch using numpy arrays directly.
        
        For new code, use fit() method instead.
        """
        # Convert to DataLoader
        X_tensor = torch.FloatTensor(X)
        y_tensors = {task: torch.LongTensor(y[task]) for task in self.task_names}
        
        dataset = TensorDataset(
            X_tensor,
            *[y_tensors[task] for task in self.task_names]
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        return self._train_epoch_from_loader(loader)


class SingleTaskTrainer:
    """Trainer for single-task classification."""
    
    def __init__(self,
                 model: nn.Module,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 learning_rate: float = 1e-3):
        """Initialize single-task trainer."""
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_x, batch_y in tqdm(train_loader, desc="Training"):
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(batch_x)
            loss = self.criterion(outputs, batch_y)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)
        
        return {
            "loss": total_loss / len(train_loader),
            "accuracy": correct / total
        }
        
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()
        
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == batch_y).sum().item()
                total += batch_y.size(0)
        
        return {
            "loss": total_loss / len(val_loader),
            "accuracy": correct / total
        }
