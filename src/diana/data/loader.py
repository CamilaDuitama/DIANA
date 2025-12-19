"""Data loaders for unitig matrices and metadata."""

import polars as pl
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Dict
import scipy.sparse as sp


class MatrixLoader:
    """
    Load sparse unitig matrices from kmtricks/muset output.
    
    Currently not implemented - use scripts/data_prep/05_extract_and_split_matrices.py
    for matrix extraction instead.
    """
    
    def __init__(self, matrix_path: Path):
        """
        Initialize matrix loader.
        
        Args:
            matrix_path: Path to matrix directory containing kmtricks.fof
        """
        self.matrix_path = Path(matrix_path)
        self.fof_file = self.matrix_path / "kmer_matrix" / "kmtricks.fof"
        
    def load_samples(self, sample_ids: List[str]) -> Tuple[sp.csr_matrix, List[str]]:
        """
        Load unitig matrix for specified samples.
        
        Args:
            sample_ids: List of sample accession IDs
            
        Returns:
            Tuple of (sparse matrix, loaded sample IDs)
        """
        raise NotImplementedError("Use scripts/data_prep/05_extract_and_split_matrices.py instead")
        
    def get_available_samples(self) -> List[str]:
        """Get list of all samples available in the matrix."""
        with open(self.fof_file, 'r') as f:
            return [line.split()[0].strip() for line in f]


class MetadataLoader:
    """
    Load and process metadata for classification.
    
    Handles loading of TSV-formatted metadata files and basic preprocessing.
    Used by all data analysis and training scripts.
    """
    
    def __init__(self, metadata_path: Path):
        """
        Initialize metadata loader.
        
        Args:
            metadata_path: Path to metadata TSV file
        """
        self.metadata_path = Path(metadata_path)
        self.df = None
        
    def load(self) -> pl.DataFrame:
        """Load metadata from TSV."""
        self.df = pl.read_csv(
            self.metadata_path,
            separator="\t",
            infer_schema_length=0
        )
        return self.df
        
    def get_labels(self, sample_ids: List[str], 
                   target: str) -> np.ndarray:
        """
        Get labels for specified samples and target.
        
        Args:
            sample_ids: List of sample IDs
            target: Target column name (sample_type, community_type, etc.)
            
        Returns:
            Array of labels
        """
        if self.df is None:
            self.load()
            
        filtered = self.df.filter(
            pl.col("Run_accession").is_in(sample_ids)
        )
        return filtered[target].to_numpy()
        
    def get_all_labels(self, sample_ids: List[str]) -> Dict[str, np.ndarray]:
        """
        Get all target labels for specified samples.
        
        Args:
            sample_ids: List of sample IDs
            
        Returns:
            Dictionary mapping target names to label arrays
        """
        targets = ["sample_type", "community_type", "sample_host", "material"]
        return {
            target: self.get_labels(sample_ids, target)
            for target in targets
        }
