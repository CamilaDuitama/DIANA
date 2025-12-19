"""PyTorch Dataset for DIANA."""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)

class DianaDataset(Dataset):
    """Dataset for multi-task classification."""
    
    def __init__(self, 
                 matrix_path: Path, 
                 metadata_path: Path,
                 sample_ids_path: Optional[Path] = None,
                 targets: List[str] = ["sample_type", "community_type", "sample_host", "material"],
                 matrix_type: str = "PA"):
        """
        Initialize dataset.
        
        Args:
            matrix_path: Path to the matrix file (.mat)
            metadata_path: Path to the metadata file (.tsv)
            sample_ids_path: Path to a file containing sample IDs to include (optional)
            targets: List of target columns to predict
            matrix_type: "PA" or "Abundance"
        """
        self.matrix_path = Path(matrix_path)
        self.metadata_path = Path(metadata_path)
        self.targets = targets
        self.matrix_type = matrix_type
        
        # Load metadata
        self.metadata = pd.read_csv(metadata_path, sep='\t')
        
        # Filter by sample IDs if provided
        if sample_ids_path:
            with open(sample_ids_path, 'r') as f:
                valid_ids = set(line.strip() for line in f)
            self.metadata = self.metadata[self.metadata['Run_accession'].isin(valid_ids)]
            
        # Create label encoders
        self.label_encoders = {}
        for target in targets:
            # Simple integer encoding
            # Note: In a real scenario, we should fit encoders on training set only and save them
            # For now, we assume the metadata contains all classes or we handle unseen classes
            unique_classes = sorted(self.metadata[target].unique())
            self.label_encoders[target] = {cls: i for i, cls in enumerate(unique_classes)}
            
        # Load matrix
        self._load_matrix()
        
    def _load_matrix(self):
        """Load matrix data."""
        logger.info(f"Loading {self.matrix_type} matrix from {self.matrix_path}...")
        
        # The matrix format from MatrixExtractor is:
        # sample_id unitig1 unitig2 ...
        # sample1 0 1 ...
        
        # We can use pandas to read it, but it might be slow for huge files.
        # Assuming it fits in memory for now.
        
        try:
            # Read header first to get unitig IDs (optional, we might not need them for MLP)
            with open(self.matrix_path, 'r') as f:
                header = f.readline().strip().split()
                # self.unitig_ids = header[1:]
                
            # Read data
            # Using pandas read_csv with space separator
            df = pd.read_csv(self.matrix_path, sep=' ')
            
            # Filter to keep only samples in metadata
            # The matrix file has 'sample_id' as first column
            valid_ids = set(self.metadata['Run_accession'])
            df = df[df['sample_id'].isin(valid_ids)]
            
            # Sort metadata to match matrix order
            self.metadata = self.metadata.set_index('Run_accession')
            self.metadata = self.metadata.reindex(df['sample_id'])
            self.metadata = self.metadata.reset_index()
            
            # Extract features
            self.features = df.iloc[:, 1:].values.astype(np.float32)
            self.sample_ids = df['sample_id'].values
            
            logger.info(f"Loaded matrix: {self.features.shape} (samples x features)")
            
        except Exception as e:
            logger.error(f"Error loading matrix: {e}")
            raise

    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        # Features
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        
        # Labels
        y = {}
        for target in self.targets:
            label_str = self.metadata.iloc[idx][target]
            label_idx = self.label_encoders[target][label_str]
            y[target] = torch.tensor(label_idx, dtype=torch.long)
            
        return x, y
