"""Training callbacks for early stopping and checkpointing."""

import torch
from pathlib import Path
from typing import Optional


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss: float) -> bool:
        """
        Check if should stop training.
        
        Args:
            val_loss: Current validation loss
            
        Returns:
            True if should stop training
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
            
        return self.early_stop


class ModelCheckpoint:
    """Save model checkpoints during training."""
    
    def __init__(self, 
                 filepath: Path,
                 monitor: str = "val_loss",
                 mode: str = "min",
                 save_best_only: bool = True):
        """
        Initialize checkpoint callback.
        
        Args:
            filepath: Path to save checkpoints
            monitor: Metric to monitor
            mode: "min" or "max"
            save_best_only: Only save best model
        """
        self.filepath = Path(filepath)
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.best_value = float('inf') if mode == "min" else float('-inf')
        
    def __call__(self, model: torch.nn.Module, metrics: dict) -> bool:
        """
        Check if should save checkpoint.
        
        Returns:
            True if checkpoint was saved
        """
        current = metrics.get(self.monitor)
        if current is None:
            return False
            
        is_better = (
            (self.mode == "min" and current < self.best_value) or
            (self.mode == "max" and current > self.best_value)
        )
        
        if not self.save_best_only or is_better:
            self.best_value = current
            torch.save(model.state_dict(), self.filepath)
            return True
            
        return False
