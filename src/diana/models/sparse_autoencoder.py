"""Sparse autoencoder for unsupervised feature learning."""

import torch
import torch.nn as nn
from typing import List


class SparseAutoencoder(nn.Module):
    """Autoencoder for learning compressed unitig representations."""
    
    def __init__(self,
                 input_dim: int,
                 encoding_dim: int = 128,
                 hidden_dims: List[int] = [512, 256],
                 dropout: float = 0.3,
                 sparsity_penalty: float = 0.01):
        """
        Initialize sparse autoencoder.
        
        Args:
            input_dim: Number of input features
            encoding_dim: Dimension of encoded representation
            hidden_dims: Hidden layer dimensions
            dropout: Dropout probability
            sparsity_penalty: L1 regularization strength
        """
        super().__init__()
        
        self.encoding_dim = encoding_dim
        self.sparsity_penalty = sparsity_penalty
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
            
        encoder_layers.append(nn.Linear(prev_dim, encoding_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder (mirror of encoder)
        decoder_layers = []
        prev_dim = encoding_dim
        
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
            
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation."""
        return self.encoder(x)
        
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to reconstruction."""
        return self.decoder(z)
        
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Returns:
            Tuple of (reconstruction, encoding)
        """
        encoding = self.encode(x)
        reconstruction = self.decode(encoding)
        return reconstruction, encoding
