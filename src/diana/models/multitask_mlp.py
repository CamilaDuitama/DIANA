"""Multi-task MLP with shared backbone and task-specific heads."""

import torch
import torch.nn as nn
from typing import Dict, List, Optional


class MultiTaskMLP(nn.Module):
    """
    Multi-task MLP for simultaneous prediction of:
    - sample_type (2 classes) - ancient vs modern
    - community_type (6 classes) - oral, skeletal, gut, etc.
    - sample_host (12 classes) - Homo sapiens, environmental, etc.
    - material (13 classes) - dental calculus, tooth, bone, etc.
    
    Architecture:
        Input → Shared Encoder → Task-specific Heads → Outputs
    
    Used by: scripts/training/01_train_multitask_single_fold.py
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int] = [512, 256, 128],
                 num_classes: Dict[str, int] = None,
                 dropout: float = 0.5,
                 use_batch_norm: bool = True,
                 activation: str = "relu"):
        """
        Initialize multi-task MLP.
        
        Args:
            input_dim: Number of input features (k-mers)
            hidden_dims: List of hidden layer dimensions
            num_classes: Dictionary mapping target names to number of classes
            dropout: Dropout probability
            use_batch_norm: Whether to use batch normalization
            activation: Activation function ('relu', 'gelu', 'leaky_relu')
        """
        super().__init__()
        
        if num_classes is None:
            num_classes = {
                "sample_type": 2,
                "community_type": 6,
                "sample_host": 12,
                "material": 13
            }
        
        self.num_classes = num_classes
        self.targets = list(num_classes.keys())
        
        # Select activation
        if activation == "relu":
            act_fn = nn.ReLU
        elif activation == "gelu":
            act_fn = nn.GELU
        elif activation == "leaky_relu":
            act_fn = lambda: nn.LeakyReLU(0.1)
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Shared backbone
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            layers.append(act_fn())
            layers.append(nn.Dropout(dropout))
            
            prev_dim = hidden_dim
            
        self.backbone = nn.Sequential(*layers)
        
        # Task-specific heads (with additional hidden layer for better task separation)
        self.heads = nn.ModuleDict()
        for target, n_classes in num_classes.items():
            self.heads[target] = nn.Sequential(
                nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
                act_fn(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dims[-1] // 2, n_classes)
            )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Dictionary mapping target names to logits
        """
        # Shared features
        features = self.backbone(x)
        
        # Task-specific predictions
        outputs = {
            target: self.heads[target](features)
            for target in self.targets
        }
        
        return outputs
        
    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Make predictions (apply softmax).
        
        Args:
            x: Input tensor
            
        Returns:
            Dictionary mapping target names to class probabilities
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return {
                target: torch.softmax(logits[target], dim=1)
                for target in self.targets
            }
    
    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MultiTaskLoss(nn.Module):
    """
    Weighted multi-task loss combining losses from all tasks.
    
    Uses cross-entropy loss for each task, optionally with class weights
    for handling class imbalance (especially for sample_type: ancient vs modern).
    """
    
    def __init__(
        self,
        task_names: List[str],
        task_weights: Optional[Dict[str, float]] = None,
        class_weights: Optional[Dict[str, torch.Tensor]] = None
    ):
        """
        Initialize multi-task loss.
        
        Args:
            task_names: List of task names
            task_weights: Dictionary mapping task names to loss weights (default: equal)
            class_weights: Dictionary mapping task names to class weight tensors
        """
        super().__init__()
        
        self.task_names = task_names
        
        # Default to equal task weights
        if task_weights is None:
            task_weights = {name: 1.0 for name in task_names}
        self.task_weights = task_weights
        
        # Create loss functions for each task
        self.criterions = nn.ModuleDict()
        for task_name in task_names:
            weight = class_weights.get(task_name) if class_weights else None
            self.criterions[task_name] = nn.CrossEntropyLoss(weight=weight)
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> tuple:
        """
        Compute weighted multi-task loss.
        
        Args:
            predictions: Dictionary of task predictions (logits)
            targets: Dictionary of task targets (class indices)
        
        Returns:
            Tuple of (total_loss, task_losses_dict)
        """
        task_losses = {}
        total_loss = 0.0
        
        for task_name in self.task_names:
            loss = self.criterions[task_name](predictions[task_name], targets[task_name])
            task_losses[task_name] = loss
            total_loss += self.task_weights[task_name] * loss
        
        return total_loss, task_losses
