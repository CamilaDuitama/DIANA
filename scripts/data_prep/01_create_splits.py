#!/usr/bin/env python3
"""
Create stratified train/test splits for the DIANA dataset.

Generates 85% train / 15% test split stratified by community type to ensure
representative distributions across splits. No validation set is created as
nested cross-validation will be used during training.

Output: train_ids.txt, test_ids.txt, split_config.json in data/splits/
"""

import sys
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from diana.data.loader import MetadataLoader
from diana.data.splitter import StratifiedSplitter
from diana.utils.config import load_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Create stratified train/test splits')
    parser.add_argument('--config', default='configs/data_config.yaml', help='Path to config file')
    args = parser.parse_args()

    config = load_config(Path(args.config))
    
    logger.info("Loading metadata...")
    metadata = MetadataLoader(config["metadata_path"]).load()
    logger.info(f"Total samples: {metadata.height}")
    
    logger.info("Creating stratified splits...")
    splitter = StratifiedSplitter(train_size=0.85, val_size=0.0, test_size=0.15, random_state=42)
    
    stratify_col = "community_type" if "community_type" in metadata.columns else "sample_type"
    logger.info(f"Stratifying by: {stratify_col}")

    train_ids, val_ids, test_ids = splitter.split(metadata, stratify_by=stratify_col, id_col="Run_accession")
    
    logger.info(f"Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")
    
    output_dir = Path(config["splits_dir"])
    splitter.save_splits(train_ids, val_ids, test_ids, output_dir, metadata=metadata, id_col="Run_accession")
    logger.info(f"✓ Splits (IDs and metadata) saved to {output_dir}")

if __name__ == "__main__":
    main()
