"""Multi-task MLP with shared backbone and task-specific heads."""

import torch
import torch.nn as nn
from typing import Dict, List


class MultiTaskMLP(nn.Module):
    """
    Multi-task MLP for simultaneous prediction of:
    - sample_type (2 classes)
    - community_type (6 classes)
    - sample_host (12 classes)
    - material (13 classes)
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int] = [512, 256, 128],
                 num_classes: Dict[str, int] = None,
                 dropout: float = 0.5):
        """
        Initialize multi-task MLP.
        
        Args:
            input_dim: Number of input features (unitigs)
            hidden_dims: List of hidden layer dimensions
            num_classes: Dictionary mapping target names to number of classes
            dropout: Dropout probability
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
        
        # Shared backbone
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
            
        self.backbone = nn.Sequential(*layers)
        
        # Task-specific heads
        self.heads = nn.ModuleDict({
            target: nn.Linear(hidden_dims[-1], n_classes)
            for target, n_classes in num_classes.items()
        })
        
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
