"""
Tests for inference pipeline components.

Validates that the prediction pipeline works correctly end-to-end,
including model loading, feature extraction detection, and output formatting.
"""

import pytest
import json
import torch
import numpy as np
from pathlib import Path

from diana.inference.predictor import Predictor
from diana.models.multitask_mlp import MultiTaskMLP


class TestPredictor:
    """Test the Predictor class for loading models and making predictions."""
    
    def test_predictor_loads_checkpoint(self, temp_dir):
        """Verify Predictor correctly loads saved model checkpoints."""
        # Create a minimal checkpoint
        input_dim = 100
        num_classes = {
            "sample_type": 2,
            "community_type": 6,
            "sample_host": 12,
            "material": 13
        }
        
        model = MultiTaskMLP(
            input_dim=input_dim,
            hidden_dims=[64, 32],
            num_classes=num_classes,
            dropout=0.3
        )
        
        # Save checkpoint
        checkpoint_path = temp_dir / "test_model.pth"
        torch.save({
            'epoch': 10,
            'model_state_dict': model.state_dict(),
            'model_type': 'multitask',
            'config': {'input_dim': input_dim},
            'val_loss': 0.5,
            'history': {'loss': [1.0, 0.8, 0.6]}
        }, checkpoint_path)
        
        # Load with Predictor
        predictor = Predictor(checkpoint_path, device='cpu')
        
        assert predictor.model_type == 'multitask'
        assert predictor.model is not None
        assert predictor.device == 'cpu'
    
    def test_predictor_inference_shapes(self, temp_dir):
        """Verify predictions have correct shape and structure."""
        # Create and save model
        input_dim = 50
        num_classes = {
            "sample_type": 2,
            "community_type": 6,
            "sample_host": 12,
            "material": 13
        }
        
        model = MultiTaskMLP(
            input_dim=input_dim,
            hidden_dims=[32, 16],
            num_classes=num_classes
        )
        
        checkpoint_path = temp_dir / "model.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_type': 'multitask',
            'config': {'input_dim': input_dim}
        }, checkpoint_path)
        
        # Load and predict
        predictor = Predictor(checkpoint_path, device='cpu')
        
        # Create dummy features
        features = np.random.rand(input_dim).astype(np.float32)
        predictions = predictor.predict(features, return_probabilities=True)
        
        # Check structure
        assert isinstance(predictions, dict)
        assert set(predictions.keys()) == {'sample_type', 'community_type', 'sample_host', 'material'}
        
        # Check each task has predictions and probabilities
        for task, pred in predictions.items():
            assert 'class' in pred
            assert 'probabilities' in pred
            
            # Probabilities should sum to ~1.0
            prob_sum = sum(pred['probabilities'])
            assert 0.99 <= prob_sum <= 1.01, f"{task} probabilities sum to {prob_sum}"
    
    def test_prediction_output_format(self, temp_dir):
        """Verify prediction JSON has all required fields and valid values."""
        # Setup model
        input_dim = 20
        num_classes = {"sample_type": 2, "community_type": 3}
        
        model = MultiTaskMLP(
            input_dim=input_dim,
            hidden_dims=[16],
            num_classes=num_classes
        )
        
        checkpoint_path = temp_dir / "model.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_type': 'multitask',
            'config': {'input_dim': input_dim}
        }, checkpoint_path)
        
        predictor = Predictor(checkpoint_path, device='cpu')
        features = np.random.rand(input_dim).astype(np.float32)
        predictions = predictor.predict(features, return_probabilities=True)
        
        # Validate predicted class is valid integer
        for task, pred in predictions.items():
            assert isinstance(pred['class'], int)
            assert pred['class'] >= 0
            
            # Probabilities should be non-negative and sum to ~1
            assert all(p >= 0.0 for p in pred['probabilities']), \
                f"{task} has negative probabilities"
            prob_sum = sum(pred['probabilities'])
            assert 0.99 <= prob_sum <= 1.01


class TestPairedEndDetection:
    """Test detection of paired-end FASTQ files."""
    
    def test_paired_end_patterns(self, temp_dir):
        """Test detection of various R1/R2 naming patterns."""
        from diana.cli.predict import detect_paired_end
        
        # Test case 1: _1 / _2 pattern
        r1_file = temp_dir / "sample_1.fastq.gz"
        r2_file = temp_dir / "sample_2.fastq.gz"
        r1_file.touch()
        r2_file.touch()
        
        detected = detect_paired_end(r1_file)
        assert len(detected) == 2
        assert r1_file in detected
        assert r2_file in detected
        
        # Test case 2: _R1 / _R2 pattern
        r1_file = temp_dir / "ERR123_R1.fastq.gz"
        r2_file = temp_dir / "ERR123_R2.fastq.gz"
        r1_file.touch()
        r2_file.touch()
        
        detected = detect_paired_end(r1_file)
        assert len(detected) == 2
        assert r1_file in detected
        assert r2_file in detected
    
    def test_single_end_detection(self, temp_dir):
        """Single-end files should return list of length 1."""
        from diana.cli.predict import detect_paired_end
        
        single_file = temp_dir / "sample.fastq.gz"
        single_file.touch()
        
        detected = detect_paired_end(single_file)
        assert len(detected) == 1
        assert detected[0] == single_file
    
    def test_missing_pair_returns_single(self, temp_dir):
        """If R2 doesn't exist, should return only R1."""
        from diana.cli.predict import detect_paired_end
        
        r1_file = temp_dir / "sample_R1.fastq.gz"
        r1_file.touch()
        # R2 does not exist
        
        detected = detect_paired_end(r1_file)
        assert len(detected) == 1
        assert detected[0] == r1_file


class TestModelArchitectureReconstruction:
    """Test that model architecture is correctly inferred from checkpoints."""
    
    def test_infer_batch_norm_from_checkpoint(self, temp_dir):
        """Verify batch norm detection from checkpoint keys."""
        # Model WITH batch norm
        model_bn = MultiTaskMLP(
            input_dim=50,
            hidden_dims=[32],
            num_classes={"task1": 2},
            use_batch_norm=True
        )
        
        checkpoint_path = temp_dir / "model_bn.pth"
        torch.save({
            'model_state_dict': model_bn.state_dict(),
            'model_type': 'multitask'
        }, checkpoint_path)
        
        predictor = Predictor(checkpoint_path, device='cpu')
        
        # Should successfully load (batch norm detected)
        assert predictor.model is not None
        
        # Model WITHOUT batch norm
        model_no_bn = MultiTaskMLP(
            input_dim=50,
            hidden_dims=[32],
            num_classes={"task1": 2},
            use_batch_norm=False
        )
        
        checkpoint_path2 = temp_dir / "model_no_bn.pth"
        torch.save({
            'model_state_dict': model_no_bn.state_dict(),
            'model_type': 'multitask'
        }, checkpoint_path2)
        
        predictor2 = Predictor(checkpoint_path2, device='cpu')
        assert predictor2.model is not None
    
    def test_infer_hidden_dims_from_checkpoint(self, temp_dir):
        """Verify hidden dimensions are correctly inferred."""
        hidden_dims = [128, 64, 32]
        
        model = MultiTaskMLP(
            input_dim=100,
            hidden_dims=hidden_dims,
            num_classes={"task1": 3}
        )
        
        checkpoint_path = temp_dir / "model.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_type': 'multitask'
        }, checkpoint_path)
        
        predictor = Predictor(checkpoint_path, device='cpu')
        
        # Model should load successfully
        assert predictor.model is not None
        
        # Test inference works
        features = np.random.rand(100).astype(np.float32)
        predictions = predictor.predict(features)
        assert 'task1' in predictions
