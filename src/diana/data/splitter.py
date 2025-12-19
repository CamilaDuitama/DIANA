"""Train/validation/test splitting with stratification."""

import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path
from typing import Tuple, List, Optional, Union
from sklearn.model_selection import train_test_split
import json
import logging

logger = logging.getLogger(__name__)

class StratifiedSplitter:
    """Create stratified train/val/test splits handling class imbalance."""
    
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
        
    def _robust_split(self, 
                     df: pd.DataFrame, 
                     test_size: float, 
                     label_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create stratified train-test split handling severe class imbalance.
        Adapted from Logan's script.
        """
        class_counts = df[label_col].value_counts()
        
        # Separate classes by sample count
        singleton_classes = class_counts[class_counts == 1].index
        small_classes = class_counts[(class_counts > 1) & (class_counts <= 5)].index
        medium_classes = class_counts[(class_counts > 5) & (class_counts <= 20)].index
        large_classes = class_counts[class_counts > 20].index
        
        train_indices = []
        test_indices = []
        
        # Handle singleton classes - all go to training (the larger set)
        for cls in singleton_classes:
            cls_samples = df[df[label_col] == cls].index.tolist()
            train_indices.extend(cls_samples)
        
        # Handle small classes (2-5 samples)
        for cls in small_classes:
            cls_samples = df[df[label_col] == cls].index.tolist()
            np.random.seed(self.random_state)
            np.random.shuffle(cls_samples)
            
            n_samples = len(cls_samples)
            n_test = max(0, min(1, int(n_samples * test_size)))  # At most 1 to test
            n_train = n_samples - n_test
            
            train_indices.extend(cls_samples[:n_train])
            test_indices.extend(cls_samples[n_train:])
            
        # Handle medium classes
        for cls in medium_classes:
            cls_samples = df[df[label_col] == cls].index.tolist()
            cls_labels = [cls] * len(cls_samples)
            
            if len(cls_samples) >= 4:
                train_idx, test_idx = train_test_split(
                    cls_samples, test_size=test_size, random_state=self.random_state,
                    stratify=cls_labels
                )
            else:
                np.random.seed(self.random_state)
                np.random.shuffle(cls_samples)
                n_test = max(1, int(len(cls_samples) * test_size))
                test_idx = cls_samples[:n_test]
                train_idx = cls_samples[n_test:]
            
            train_indices.extend(train_idx)
            test_indices.extend(test_idx)
            
        # Handle large classes
        if len(large_classes) > 0:
            large_df = df[df[label_col].isin(large_classes)]
            train_large, test_large = train_test_split(
                large_df, test_size=test_size, random_state=self.random_state,
                stratify=large_df[label_col]
            )
            train_indices.extend(train_large.index.tolist())
            test_indices.extend(test_large.index.tolist())
            
        return df.loc[train_indices], df.loc[test_indices]

    def split(self, 
             metadata: Union[pl.DataFrame, pd.DataFrame],
             stratify_by: str = "sample_type",
             id_col: str = "Run_accession") -> Tuple[List[str], List[str], List[str]]:
        """
        Create stratified splits.
        
        Args:
            metadata: Metadata DataFrame
            stratify_by: Column to use for stratification
            id_col: Column containing sample IDs
            
        Returns:
            Tuple of (train_ids, val_ids, test_ids)
        """
        if isinstance(metadata, pl.DataFrame):
            df = metadata.to_pandas()
        else:
            df = metadata.copy()
            
        # 1. Split Test set from (Train + Val)
        # The test_size passed to _robust_split should be self.test_size
        train_val_df, test_df = self._robust_split(
            df, 
            test_size=self.test_size, 
            label_col=stratify_by
        )
        
        # 2. Split Val set from Train
        if self.val_size > 0:
            # The validation size relative to the remaining data (Train + Val)
            val_ratio = self.val_size / (self.train_size + self.val_size)
            
            train_df, val_df = self._robust_split(
                train_val_df,
                test_size=val_ratio,
                label_col=stratify_by
            )
        else:
            train_df = train_val_df
            val_df = pd.DataFrame(columns=df.columns)
        
        train_ids = train_df[id_col].tolist()
        val_ids = val_df[id_col].tolist()
        test_ids = test_df[id_col].tolist()
        
        return train_ids, val_ids, test_ids
        
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
