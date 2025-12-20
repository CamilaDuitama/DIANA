"""
Unit tests for neural network models.

Tests model initialization and forward passes for models actually used in training.
Only tests MultiTaskMLP and SingleTaskMLP - the models used in the pipeline.
"""

import pytest
import torch
from diana.models.multitask_mlp import MultiTaskMLP
from diana.models.single_task_mlp import SingleTaskMLP


class TestMultiTaskMLP:
    """Test the MultiTaskMLP model for multi-task classification."""
    
    def test_forward_pass(self):
        """
        Test multi-task model forward pass.
        
        Validates that the model produces correct output shapes for all 4 tasks
        used in DIANA: sample_type, community_type, sample_host, material.
        """
        num_classes = {
            'sample_type': 3,
            'community_type': 5,
            'sample_host': 4,
            'material': 7
        }
        
        model = MultiTaskMLP(
            input_dim=100,
            hidden_dims=[64, 32],
            num_classes=num_classes,
            dropout=0.2
        )
        
        # Forward pass
        x = torch.randn(16, 100)  # batch_size=16, features=100
        outputs = model(x)
        
        # Verify all tasks present with correct shapes
        assert 'sample_type' in outputs
        assert outputs['sample_type'].shape == (16, 3)
        assert outputs['community_type'].shape == (16, 5)
        assert outputs['sample_host'].shape == (16, 4)
        assert outputs['material'].shape == (16, 7)


class TestSingleTaskMLP:
    """Test the SingleTaskMLP model for single-task classification."""
    
    def test_forward_pass(self):
        """
        Test single-task model forward pass.
        
        Used for training individual classifiers for each task separately
        (for comparison with multi-task approach).
        """
        model = SingleTaskMLP(
            input_dim=100,
            hidden_dims=[64, 32],
            num_classes=5,
            dropout=0.2
        )
        
        x = torch.randn(16, 100)
        output = model(x)
        
        assert output.shape == (16, 5)
