"""Single-task MLP for individual target prediction."""

import torch
import torch.nn as nn
from typing import List


class SingleTaskMLP(nn.Module):
    """Standard MLP for single classification task."""
    
    def __init__(self,
                 input_dim: int,
                 num_classes: int,
                 hidden_dims: List[int] = [512, 256, 128],
                 dropout: float = 0.5):
        """
        Initialize single-task MLP.
        
        Args:
            input_dim: Number of input features
            num_classes: Number of output classes
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
        """
        super().__init__()
        
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
            
        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x)
        
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Make predictions (apply softmax)."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=1)
