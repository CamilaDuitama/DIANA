#!/usr/bin/env python3
"""Create train/validation/test splits with stratification."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from diana.data.loader import MetadataLoader
from diana.data.splitter import StratifiedSplitter
from diana.utils.config import load_config


def main():
    # Load configuration
    config = load_config(Path("configs/data_config.yaml"))
    
    # Load metadata
    print("Loading metadata...")
    metadata_loader = MetadataLoader(config["metadata_path"])
    metadata = metadata_loader.load()
    
    print(f"Total samples: {metadata.height}")
    
    # Create splits
    print("\nCreating stratified splits...")
    splitter = StratifiedSplitter(
        train_size=0.7,
        val_size=0.15,
        test_size=0.15,
        random_state=42
    )
    
    train_ids, val_ids, test_ids = splitter.split(
        metadata,
        stratify_by="sample_type"  # Stratify by ancient/modern
    )
    
    print(f"Train samples: {len(train_ids)}")
    print(f"Val samples: {len(val_ids)}")
    print(f"Test samples: {len(test_ids)}")
    
    # Save splits
    output_dir = Path(config["splits_dir"])
    splitter.save_splits(train_ids, val_ids, test_ids, output_dir)
    
    print(f"\n✓ Splits saved to {output_dir}")
    
    # Print distribution per target
    for target in config["targets"]:
        print(f"\n{target} distribution:")
        train_df = metadata.filter(metadata["Run_accession"].is_in(train_ids))
        val_df = metadata.filter(metadata["Run_accession"].is_in(val_ids))
        test_df = metadata.filter(metadata["Run_accession"].is_in(test_ids))
        
        print(f"  Train: {train_df[target].value_counts()}")
        print(f"  Val: {val_df[target].value_counts()}")
        print(f"  Test: {test_df[target].value_counts()}")


if __name__ == "__main__":
    main()
