"""
Unit tests for data loading components.

Tests MatrixLoader and metadata processing to ensure data is loaded correctly
and samples/features are properly aligned.
"""

import pytest
import numpy as np
import pandas as pd
from diana.data.loader import MatrixLoader


class TestMatrixLoader:
    """Test the MatrixLoader class for loading k-mer presence/absence matrices."""
    
    def test_load_matrix_basic(self, dummy_matrix_path):
        """Load matrix file and verify basic properties."""
        loader = MatrixLoader(str(dummy_matrix_path))
        features, sample_ids, _ = loader.load()
        
        # Matrix file has 20 features (rows) × 50 samples (cols)
        # MatrixLoader transposes to (50 samples, 20 features)
        assert features.shape == (50, 20), f"Expected (50 samples, 20 features), got {features.shape}"
        assert len(sample_ids) == 50
        assert features.dtype == np.float32
        assert not np.any(np.isnan(features))
        assert not np.any(np.isinf(features))
    
    def test_matrix_values_binary(self, dummy_matrix_path):
        """Verify matrix contains only binary values (presence/absence)."""
        loader = MatrixLoader(str(dummy_matrix_path))
        features, _, _ = loader.load()
        
        unique_vals = np.unique(features)
        assert all(v in [0, 1] for v in unique_vals), f"Expected binary, got {unique_vals}"
    
    def test_missing_file_error(self):
        """Verify appropriate error when file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            loader = MatrixLoader("nonexistent.mat")
            loader.load()


class TestMetadata:
    """Test metadata loading and validation."""
    
    def test_load_metadata(self, dummy_metadata_path):
        """Load metadata and verify structure."""
        metadata = pd.read_csv(dummy_metadata_path, sep='\t')
        
        assert len(metadata) == 50
        required = ['Run_accession', 'sample_type', 'community_type', 'sample_host', 'material']
        for col in required:
            assert col in metadata.columns
            assert not metadata[col].isna().any()
    
    def test_metadata_sample_alignment(self, dummy_matrix_path, dummy_metadata_path):
        """Verify metadata Run_accessions match matrix sample IDs."""
        loader = MatrixLoader(str(dummy_matrix_path))
        _, sample_ids, _ = loader.load()
        
        metadata = pd.read_csv(dummy_metadata_path, sep='\t')
        metadata_samples = set(metadata['Run_accession'].values)
        
        for sample in sample_ids:
            assert sample in metadata_samples
    
    def test_balanced_classes(self, dummy_metadata_path):
        """Verify test data has multiple classes for each task."""
        metadata = pd.read_csv(dummy_metadata_path, sep='\t')
        
        for task in ['sample_type', 'community_type', 'sample_host', 'material']:
            n_classes = metadata[task].nunique()
            assert n_classes >= 2, f"{task} has only {n_classes} class(es)"


class TestLabelEncoding:
    """Test label encoding for classification tasks."""
    
    def test_label_encoder_invertible(self, dummy_metadata_path):
        """Verify label encoding is reversible."""
        from sklearn.preprocessing import LabelEncoder
        
        metadata = pd.read_csv(dummy_metadata_path, sep='\t')
        encoder = LabelEncoder()
        
        original = metadata['sample_type'].values
        encoded = encoder.fit_transform(original)
        decoded = encoder.inverse_transform(encoded)
        
        assert all(decoded == original)
        assert encoded.min() == 0
        assert encoded.max() == len(encoder.classes_) - 1
    
    def test_class_weights(self, dummy_metadata_path):
        """Verify class weights are computed correctly for imbalanced data."""
        from sklearn.utils.class_weight import compute_class_weight
        
        metadata = pd.read_csv(dummy_metadata_path, sep='\t')
        classes = np.unique(metadata['sample_type'])
        
        weights = compute_class_weight(
            class_weight='balanced',
            classes=classes,
            y=metadata['sample_type']
        )
        
        assert all(w > 0 for w in weights)
        assert len(weights) == len(classes)
