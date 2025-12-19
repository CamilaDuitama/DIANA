"""Train/validation/test splitting with stratification."""

import numpy as np
import polars as pl
from pathlib import Path
from typing import Tuple, List, Optional
from sklearn.model_selection import train_test_split
import json


class StratifiedSplitter:
    """Create stratified train/val/test splits."""
    
    def __init__(self, 
                 train_size: float = 0.7,
                 val_size: float = 0.15,
                 test_size: float = 0.15,
                 random_state: int = 42):
        """
        Initialize splitter.
        
        Args:
            train_size: Proportion for training
            val_size: Proportion for validation
            test_size: Proportion for testing
            random_state: Random seed for reproducibility
        """
        assert abs(train_size + val_size + test_size - 1.0) < 1e-6
        
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.random_state = random_state
        
    def split(self, 
             metadata: pl.DataFrame,
             stratify_by: str = "sample_type") -> Tuple[List[str], List[str], List[str]]:
        """
        Create stratified splits.
        
        Args:
            metadata: Metadata DataFrame
            stratify_by: Column to use for stratification
            
        Returns:
            Tuple of (train_ids, val_ids, test_ids)
        """
        sample_ids = metadata["Run_accession"].to_numpy()
        stratify_labels = metadata[stratify_by].to_numpy()
        
        # First split: train vs (val + test)
        train_ids, temp_ids, _, temp_labels = train_test_split(
            sample_ids,
            stratify_labels,
            train_size=self.train_size,
            random_state=self.random_state,
            stratify=stratify_labels
        )
        
        # Second split: val vs test
        val_ratio = self.val_size / (self.val_size + self.test_size)
        val_ids, test_ids = train_test_split(
            temp_ids,
            train_size=val_ratio,
            random_state=self.random_state,
            stratify=temp_labels
        )
        
        return train_ids.tolist(), val_ids.tolist(), test_ids.tolist()
        
    def save_splits(self, 
                   train_ids: List[str],
                   val_ids: List[str],
                   test_ids: List[str],
                   output_dir: Path):
        """Save splits to files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save sample IDs
        with open(output_dir / "train_ids.txt", 'w') as f:
            f.write('\n'.join(train_ids))
            
        with open(output_dir / "val_ids.txt", 'w') as f:
            f.write('\n'.join(val_ids))
            
        with open(output_dir / "test_ids.txt", 'w') as f:
            f.write('\n'.join(test_ids))
            
        # Save configuration
        config = {
            "train_size": self.train_size,
            "val_size": self.val_size,
            "test_size": self.test_size,
            "random_state": self.random_state,
            "n_train": len(train_ids),
            "n_val": len(val_ids),
            "n_test": len(test_ids),
        }
        
        with open(output_dir / "split_config.json", 'w') as f:
            json.dump(config, f, indent=2)
            
    @staticmethod
    def load_splits(split_dir: Path) -> Tuple[List[str], List[str], List[str]]:
        """Load splits from files."""
        split_dir = Path(split_dir)
        
        with open(split_dir / "train_ids.txt") as f:
            train_ids = [line.strip() for line in f]
            
        with open(split_dir / "val_ids.txt") as f:
            val_ids = [line.strip() for line in f]
            
        with open(split_dir / "test_ids.txt") as f:
            test_ids = [line.strip() for line in f]
            
        return train_ids, val_ids, test_ids
