"""Tests for model architectures."""

import pytest
import torch
from diana.models import MultiTaskMLP, SingleTaskMLP, SparseAutoencoder


def test_multitask_mlp_forward():
    """Test multi-task MLP forward pass."""
    batch_size = 4
    input_dim = 100
    
    model = MultiTaskMLP(
        input_dim=input_dim,
        hidden_dims=[64, 32],
        num_classes={
            "sample_type": 2,
            "community_type": 6,
            "sample_host": 12,
            "material": 13
        }
    )
    
    x = torch.randn(batch_size, input_dim)
    outputs = model(x)
    
    assert "sample_type" in outputs
    assert outputs["sample_type"].shape == (batch_size, 2)
    assert outputs["community_type"].shape == (batch_size, 6)


def test_single_task_mlp_forward():
    """Test single-task MLP forward pass."""
    batch_size = 4
    input_dim = 100
    num_classes = 5
    
    model = SingleTaskMLP(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dims=[64, 32]
    )
    
    x = torch.randn(batch_size, input_dim)
    output = model(x)
    
    assert output.shape == (batch_size, num_classes)


def test_sparse_autoencoder():
    """Test sparse autoencoder."""
    batch_size = 4
    input_dim = 100
    encoding_dim = 20
    
    model = SparseAutoencoder(
        input_dim=input_dim,
        encoding_dim=encoding_dim,
        hidden_dims=[64, 32]
    )
    
    x = torch.randn(batch_size, input_dim)
    reconstruction, encoding = model(x)
    
    assert reconstruction.shape == (batch_size, input_dim)
    assert encoding.shape == (batch_size, encoding_dim)
