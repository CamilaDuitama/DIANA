"""Tests for data loading."""

import pytest
import polars as pl
from pathlib import Path
from diana.data.splitter import StratifiedSplitter


def test_stratified_splitter():
    """Test stratified splitting."""
    # Create mock metadata
    data = {
        "Run_accession": [f"SRR{i}" for i in range(100)],
        "sample_type": ["ancient"] * 80 + ["modern"] * 20
    }
    df = pl.DataFrame(data)
    
    splitter = StratifiedSplitter(
        train_size=0.7,
        val_size=0.15,
        test_size=0.15,
        random_state=42
    )
    
    train_ids, val_ids, test_ids = splitter.split(df, stratify_by="sample_type")
    
    # Check sizes
    assert len(train_ids) == 70
    assert len(val_ids) == 15
    assert len(test_ids) == 15
    
    # Check no overlap
    assert len(set(train_ids) & set(val_ids)) == 0
    assert len(set(train_ids) & set(test_ids)) == 0
    assert len(set(val_ids) & set(test_ids)) == 0
