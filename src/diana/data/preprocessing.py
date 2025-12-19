"""Feature preprocessing and normalization."""

import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Dict, List, Optional


class FeaturePreprocessor:
    """Preprocess sparse unitig matrices."""
    
    def __init__(self, normalize: bool = True, log_transform: bool = False):
        """
        Initialize preprocessor.
        
        Args:
            normalize: Whether to normalize features
            log_transform: Whether to apply log transformation
        """
        self.normalize = normalize
        self.log_transform = log_transform
        self.scaler = None
        
    def fit_transform(self, X: sp.csr_matrix) -> sp.csr_matrix:
        """Fit and transform features."""
        if self.log_transform:
            X = X.copy()
            X.data = np.log1p(X.data)
            
        if self.normalize:
            # TODO: Implement sparse-aware normalization
            pass
            
        return X
        
    def transform(self, X: sp.csr_matrix) -> sp.csr_matrix:
        """Transform features using fitted parameters."""
        if self.log_transform:
            X = X.copy()
            X.data = np.log1p(X.data)
            
        if self.normalize and self.scaler is not None:
            # TODO: Apply fitted normalization
            pass
            
        return X


class MatrixNormalizer:
    """Normalize sparse matrices."""
    
    @staticmethod
    def l2_normalize(X: sp.csr_matrix) -> sp.csr_matrix:
        """L2 normalize rows of sparse matrix."""
        row_norms = sp.linalg.norm(X, axis=1)
        row_norms[row_norms == 0] = 1  # Avoid division by zero
        return X.multiply(1 / row_norms[:, np.newaxis])
        
    @staticmethod
    def standard_scale(X: sp.csr_matrix, 
                      mean: Optional[np.ndarray] = None,
                      std: Optional[np.ndarray] = None) -> sp.csr_matrix:
        """Standard scale sparse matrix."""
        # TODO: Implement sparse standard scaling
        raise NotImplementedError


class LabelPreprocessor:
    """Encode labels for classification."""
    
    def __init__(self):
        self.encoders: Dict[str, LabelEncoder] = {}
        
    def fit_transform(self, labels: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Fit and transform labels for all targets.
        
        Args:
            labels: Dictionary mapping target names to label arrays
            
        Returns:
            Dictionary of encoded labels
        """
        encoded = {}
        for target, y in labels.items():
            encoder = LabelEncoder()
            encoded[target] = encoder.fit_transform(y)
            self.encoders[target] = encoder
            
        return encoded
        
    def transform(self, labels: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Transform labels using fitted encoders."""
        return {
            target: self.encoders[target].transform(y)
            for target, y in labels.items()
        }
        
    def inverse_transform(self, encoded: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Convert encoded labels back to original."""
        return {
            target: self.encoders[target].inverse_transform(y)
            for target, y in encoded.items()
        }
