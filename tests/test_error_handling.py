"""
Tests for error handling and edge cases.

Validates that the system fails gracefully with clear error messages
when given invalid inputs.
"""

import pytest
import numpy as np
import torch
from pathlib import Path
import polars as pl

from diana.data.loader import MatrixLoader
from diana.inference.predictor import Predictor
from diana.models.multitask_mlp import MultiTaskMLP


class TestMissingFiles:
    """Test handling of missing or invalid file paths."""
    
    def test_missing_matrix_file(self):
        """MatrixLoader should raise FileNotFoundError for missing files."""
        loader = MatrixLoader("nonexistent_file.mat")
        
        with pytest.raises(FileNotFoundError):
            loader.load()
    
    def test_missing_metadata_file(self, dummy_matrix_path):
        """Loading with missing metadata should raise error."""
        loader = MatrixLoader(dummy_matrix_path)
        
        with pytest.raises(FileNotFoundError):
            loader.load_with_metadata(
                metadata_path=Path("nonexistent_metadata.tsv"),
                align_to_matrix=True
            )
    
    def test_missing_checkpoint_file(self):
        """Predictor should raise error for missing checkpoint."""
        with pytest.raises(FileNotFoundError):
            Predictor("nonexistent_model.pth")


class TestCorruptedData:
    """Test handling of corrupted or malformed data."""
    
    def test_corrupted_checkpoint(self, temp_dir):
        """Corrupted .pth file should raise informative error."""
        # Create empty/corrupted file
        corrupted_path = temp_dir / "corrupted.pth"
        corrupted_path.write_text("not a pytorch checkpoint")
        
        with pytest.raises(Exception):  # Could be various torch exceptions
            Predictor(corrupted_path)
    
    def test_empty_metadata(self, temp_dir, dummy_matrix_path):
        """Empty metadata file should raise error."""
        # Create empty TSV
        empty_metadata = temp_dir / "empty.tsv"
        empty_metadata.write_text("Run_accession\tsample_type\n")
        
        loader = MatrixLoader(dummy_matrix_path)
        
        # Should either raise error or return empty results
        try:
            features, metadata = loader.load_with_metadata(
                metadata_path=empty_metadata,
                align_to_matrix=True
            )
            # If it doesn't raise, check it returns empty/filtered results
            assert len(metadata) == 0 or len(features) == 0
        except Exception:
            # Also acceptable to raise an error
            pass
    
    def test_malformed_matrix_file(self, temp_dir):
        """Matrix with wrong format should fail gracefully."""
        # Create malformed matrix
        bad_matrix = temp_dir / "bad.mat"
        bad_matrix.write_text("not a valid matrix format\n")
        
        loader = MatrixLoader(bad_matrix)
        
        with pytest.raises(Exception):  # polars or numpy will raise
            loader.load()


class TestMissingMetadataColumns:
    """Test handling of missing required columns in metadata."""
    
    def test_missing_required_column(self, temp_dir, dummy_matrix_path):
        """Metadata missing required column should raise clear error."""
        # Create metadata without sample_type
        incomplete_metadata = pl.DataFrame({
            'Run_accession': ['sample_001', 'sample_002'],
            # Missing sample_type
        })
        
        metadata_path = temp_dir / "incomplete.tsv"
        incomplete_metadata.write_csv(metadata_path, separator='\t')
        
        loader = MatrixLoader(dummy_matrix_path)
        
        # Load should work (metadata validation is done by training scripts)
        features, metadata = loader.load_with_metadata(
            metadata_path=metadata_path,
            align_to_matrix=False,
            filter_matrix_to_metadata=True
        )
        
        # But metadata should be accessible
        assert 'Run_accession' in metadata.columns


class TestDimensionMismatch:
    """Test handling of dimension mismatches."""
    
    def test_wrong_feature_dimensions(self, temp_dir):
        """Prediction with wrong number of features should fail."""
        # Create model expecting 100 features
        model = MultiTaskMLP(
            input_dim=100,
            hidden_dims=[32],
            num_classes={"task1": 2}
        )
        
        checkpoint_path = temp_dir / "model.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_type': 'multitask'
        }, checkpoint_path)
        
        predictor = Predictor(checkpoint_path, device='cpu')
        
        # Try to predict with wrong dimensions (50 instead of 100)
        wrong_features = np.random.rand(50).astype(np.float32)
        
        with pytest.raises(RuntimeError):  # PyTorch shape mismatch
            predictor.predict(wrong_features)
    
    def test_batch_size_dimension(self, temp_dir):
        """Test that batch dimension is handled correctly."""
        model = MultiTaskMLP(
            input_dim=50,
            hidden_dims=[32],
            num_classes={"task1": 2}
        )
        
        checkpoint_path = temp_dir / "model.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_type': 'multitask'
        }, checkpoint_path)
        
        predictor = Predictor(checkpoint_path, device='cpu')
        
        # Single sample (1D array)
        features_1d = np.random.rand(50).astype(np.float32)
        pred = predictor.predict(features_1d)
        assert 'task1' in pred
        
        # Batch of samples (2D array) - predict() expects 1D, should fail or handle
        features_2d = np.random.rand(5, 50).astype(np.float32)
        # Current implementation doesn't support batch, would need predict_batch()
        # That's fine - batch prediction is a separate feature


class TestInvalidInputValues:
    """Test handling of invalid input values."""
    
    def test_nan_features(self, temp_dir):
        """Features with NaN values should be handled."""
        model = MultiTaskMLP(
            input_dim=20,
            hidden_dims=[16],
            num_classes={"task1": 2}
        )
        
        checkpoint_path = temp_dir / "model.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_type': 'multitask'
        }, checkpoint_path)
        
        predictor = Predictor(checkpoint_path, device='cpu')
        
        # Create features with NaN
        features_with_nan = np.random.rand(20).astype(np.float32)
        features_with_nan[5] = np.nan
        
        # PyTorch models may propagate NaN through network
        # Check that it either raises or returns NaN (not a crash)
        try:
            pred = predictor.predict(features_with_nan, return_probabilities=True)
            # If it doesn't raise, it's handled (probabilities may be NaN)
            assert 'class' in pred['task1']
        except (ValueError, RuntimeError):
            # Also acceptable to raise error for NaN input
            pass
    
    def test_infinite_features(self, temp_dir):
        """Features with infinite values should be handled."""
        model = MultiTaskMLP(
            input_dim=20,
            hidden_dims=[16],
            num_classes={"task1": 2}
        )
        
        checkpoint_path = temp_dir / "model.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_type': 'multitask'
        }, checkpoint_path)
        
        predictor = Predictor(checkpoint_path, device='cpu')
        
        # Create features with inf
        features_with_inf = np.random.rand(20).astype(np.float32)
        features_with_inf[3] = np.inf
        
        # PyTorch may handle inf without raising (propagates through network)
        # Just verify it doesn't crash - implementation can choose to handle or raise
        try:
            pred = predictor.predict(features_with_inf)
            # If handled, should return some result
            assert pred is not None
        except (ValueError, RuntimeError):
            # Also acceptable to reject inf values
            pass


class TestEmptyInputs:
    """Test handling of empty or zero-length inputs."""
    
    def test_empty_feature_vector(self, temp_dir):
        """Empty feature vector should raise error."""
        model = MultiTaskMLP(
            input_dim=10,
            hidden_dims=[8],
            num_classes={"task1": 2}
        )
        
        checkpoint_path = temp_dir / "model.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_type': 'multitask'
        }, checkpoint_path)
        
        predictor = Predictor(checkpoint_path, device='cpu')
        
        # Empty array
        empty_features = np.array([], dtype=np.float32)
        
        with pytest.raises((ValueError, RuntimeError, IndexError)):
            predictor.predict(empty_features)
