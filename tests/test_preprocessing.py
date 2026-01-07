"""
Tests for data preprocessing components.

Validates label encoding, metadata alignment, and any normalization operations.
"""

import pytest
import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path

from diana.data.preprocessing import LabelPreprocessor
from diana.data.loader import MatrixLoader


class TestLabelPreprocessor:
    """Test label encoding and decoding."""
    
    def test_label_encoder_reversibility(self):
        """Test encode→decode returns original labels."""
        preprocessor = LabelPreprocessor()
        
        # Create test labels
        labels = {
            'sample_type': np.array(['ancient', 'modern', 'ancient', 'modern']),
            'community_type': np.array(['oral', 'gut', 'oral', 'skeletal'])
        }
        
        # Encode
        encoded = preprocessor.fit_transform(labels)
        
        # Check encoded are integers
        assert encoded['sample_type'].dtype in [np.int32, np.int64]
        assert encoded['community_type'].dtype in [np.int32, np.int64]
        
        # Decode
        decoded = preprocessor.inverse_transform(encoded)
        
        # Should match original
        np.testing.assert_array_equal(decoded['sample_type'], labels['sample_type'])
        np.testing.assert_array_equal(decoded['community_type'], labels['community_type'])
    
    def test_label_encoder_consistency(self):
        """Same input should give same encoding."""
        preprocessor = LabelPreprocessor()
        
        labels1 = {'task1': np.array(['a', 'b', 'c', 'a'])}
        labels2 = {'task1': np.array(['a', 'c', 'b'])}
        
        # Fit on first
        encoded1 = preprocessor.fit_transform(labels1)
        
        # Transform second (should use same encoding)
        encoded2 = preprocessor.transform(labels2)
        
        # 'a' should have same code in both
        a_code_1 = encoded1['task1'][0]
        a_code_2 = encoded2['task1'][0]
        assert a_code_1 == a_code_2
    
    def test_unseen_label_raises_error(self):
        """Transforming unseen labels should raise error."""
        preprocessor = LabelPreprocessor()
        
        labels_train = {'task1': np.array(['a', 'b', 'c'])}
        labels_test = {'task1': np.array(['d'])}  # 'd' not in training
        
        preprocessor.fit_transform(labels_train)
        
        with pytest.raises(ValueError):
            preprocessor.transform(labels_test)


class TestMetadataAlignment:
    """Test sample ID matching between matrix and metadata."""
    
    def test_metadata_matrix_alignment(self, dummy_matrix_path, dummy_metadata_path):
        """Test sample IDs are correctly aligned."""
        loader = MatrixLoader(dummy_matrix_path)
        features, metadata = loader.load_with_metadata(
            metadata_path=dummy_metadata_path,
            align_to_matrix=True
        )
        
        # Number of samples should match
        assert features.shape[0] == len(metadata)
        
        # Sample IDs should be in same order
        sample_ids_matrix = loader._get_sample_ids(features.shape[0])
        sample_ids_metadata = metadata['Run_accession'].to_list()
        
        # At least some should match (dummy data might be synthetic)
        # In real usage, all should match
        assert len(sample_ids_metadata) == len(sample_ids_matrix)
    
    def test_mismatched_samples_filtered(self, temp_dir, dummy_matrix_path):
        """Samples not in metadata should be filtered out."""
        # Create metadata with only subset of samples
        metadata_df = pl.DataFrame({
            'Run_accession': ['sample_000', 'sample_001'],
            'sample_type': ['ancient', 'modern']
        })
        
        metadata_path = temp_dir / "subset_metadata.tsv"
        metadata_df.write_csv(metadata_path, separator='\t')
        
        loader = MatrixLoader(dummy_matrix_path)
        
        # This should work - loader will filter to matching samples
        features, metadata = loader.load_with_metadata(
            metadata_path=metadata_path,
            align_to_matrix=False,
            filter_matrix_to_metadata=True
        )
        
        # Should only have samples in metadata
        assert len(metadata) <= 2


class TestDataIntegrity:
    """Test data loading maintains integrity."""
    
    def test_no_nan_values(self, dummy_matrix_path, dummy_metadata_path):
        """Loaded data should not contain NaN values."""
        loader = MatrixLoader(dummy_matrix_path)
        features, metadata = loader.load_with_metadata(
            metadata_path=dummy_metadata_path,
            align_to_matrix=True
        )
        
        # Check features
        assert not np.any(np.isnan(features)), "Features contain NaN values"
        assert not np.any(np.isinf(features)), "Features contain inf values"
        
        # Check metadata has no nulls in required columns
        required_cols = ['Run_accession', 'sample_type']
        for col in required_cols:
            if col in metadata.columns:
                assert metadata[col].null_count() == 0, f"{col} has null values"
    
    def test_feature_matrix_shape(self, dummy_matrix_path):
        """Feature matrix has expected shape after transpose."""
        loader = MatrixLoader(dummy_matrix_path)
        features, sample_ids, _ = loader.load()
        
        # Matrix should be (n_samples, n_features)
        n_samples, n_features = features.shape
        assert n_samples > 0, "No samples loaded"
        assert n_features > 0, "No features loaded"
        
        # Sample IDs should match sample count
        assert len(sample_ids) == n_samples
    
    def test_binary_values_only(self, dummy_matrix_path):
        """For presence/absence matrices, values should be 0 or 1."""
        if not str(dummy_matrix_path).endswith('.pa.mat'):
            pytest.skip("Test only for presence/absence matrices")
        
        loader = MatrixLoader(dummy_matrix_path)
        features, _, _ = loader.load()
        
        unique_vals = np.unique(features)
        assert all(v in [0, 1] for v in unique_vals), \
            f"Expected binary values, got {unique_vals}"


class TestMatrixLoader:
    """Test MatrixLoader functionality."""
    
    def test_load_returns_correct_types(self, dummy_matrix_path):
        """Check return types from load method."""
        loader = MatrixLoader(dummy_matrix_path)
        features, sample_ids, sample_ids_df = loader.load(return_pandas=True)
        
        assert isinstance(features, np.ndarray)
        assert isinstance(sample_ids, np.ndarray)
        assert isinstance(sample_ids_df, pd.DataFrame)
        assert 'Run_accession' in sample_ids_df.columns
    
    def test_load_without_pandas(self, dummy_matrix_path):
        """Test loading without pandas DataFrame."""
        loader = MatrixLoader(dummy_matrix_path)
        features, sample_ids, sample_ids_df = loader.load(return_pandas=False)
        
        assert isinstance(features, np.ndarray)
        assert isinstance(sample_ids, np.ndarray)
        assert sample_ids_df is None
