"""Training loops for multi-task and single-task models."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, List
from pathlib import Path
import json
from tqdm import tqdm


class MultiTaskTrainer:
    """Trainer for multi-task classification."""
    
    def __init__(self,
                 model: nn.Module,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 learning_rate: float = 1e-3,
                 task_weights: Optional[Dict[str, float]] = None):
        """
        Initialize trainer.
        
        Args:
            model: Multi-task model
            device: Device to train on
            learning_rate: Learning rate
            task_weights: Weights for each task loss
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Task-specific loss functions
        self.criteria = {
            target: nn.CrossEntropyLoss()
            for target in model.targets
        }
        
        # Task weights (default: equal weighting)
        if task_weights is None:
            task_weights = {target: 1.0 for target in model.targets}
        self.task_weights = task_weights
        
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": {target: [] for target in model.targets},
            "val_acc": {target: [] for target in model.targets}
        }
        
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0
        task_losses = {target: 0 for target in self.model.targets}
        task_correct = {target: 0 for target in self.model.targets}
        task_total = {target: 0 for target in self.model.targets}
        
        for batch_x, batch_y in tqdm(train_loader, desc="Training"):
            batch_x = batch_x.to(self.device)
            batch_y = {k: v.to(self.device) for k, v in batch_y.items()}
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(batch_x)
            
            # Compute losses
            losses = {}
            for target in self.model.targets:
                loss = self.criteria[target](outputs[target], batch_y[target])
                losses[target] = loss
                task_losses[target] += loss.item()
                
                # Accuracy
                _, predicted = torch.max(outputs[target], 1)
                task_correct[target] += (predicted == batch_y[target]).sum().item()
                task_total[target] += batch_y[target].size(0)
            
            # Combined loss
            total = sum(self.task_weights[t] * losses[t] for t in self.model.targets)
            total_loss += total.item()
            
            # Backward pass
            total.backward()
            self.optimizer.step()
            
        # Compute metrics
        metrics = {
            "loss": total_loss / len(train_loader),
            "accuracy": {
                target: task_correct[target] / task_total[target]
                for target in self.model.targets
            }
        }
        
        return metrics
        
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()
        
        total_loss = 0
        task_correct = {target: 0 for target in self.model.targets}
        task_total = {target: 0 for target in self.model.targets}
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = {k: v.to(self.device) for k, v in batch_y.items()}
                
                outputs = self.model(batch_x)
                
                # Losses and accuracy
                losses = []
                for target in self.model.targets:
                    loss = self.criteria[target](outputs[target], batch_y[target])
                    losses.append(self.task_weights[target] * loss.item())
                    
                    _, predicted = torch.max(outputs[target], 1)
                    task_correct[target] += (predicted == batch_y[target]).sum().item()
                    task_total[target] += batch_y[target].size(0)
                
                total_loss += sum(losses)
        
        metrics = {
            "loss": total_loss / len(val_loader),
            "accuracy": {
                target: task_correct[target] / task_total[target]
                for target in self.model.targets
            }
        }
        
        return metrics
        
    def fit(self, 
            train_loader: DataLoader,
            val_loader: DataLoader,
            epochs: int,
            save_path: Optional[Path] = None):
        """
        Train model for multiple epochs.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            save_path: Path to save best model
        """
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)
            
            print(f"Train Loss: {train_metrics['loss']:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}")
            
            for target in self.model.targets:
                print(f"  {target}: Train Acc={train_metrics['accuracy'][target]:.4f}, "
                      f"Val Acc={val_metrics['accuracy'][target]:.4f}")
            
            # Save best model
            if save_path and val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                torch.save(self.model.state_dict(), save_path)
                print(f"Saved best model to {save_path}")


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
