#!/usr/bin/env python3
"""Create train/validation/test splits with stratification."""

import sys
import argparse
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from diana.data.loader import MetadataLoader
from diana.data.splitter import StratifiedSplitter
from diana.data.matrix import MatrixExtractor
from diana.utils.config import load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Create stratified splits and extract matrices')
    parser.add_argument('--extract-matrices', action='store_true', 
                       help='Extract matrix subsets for splits')
    parser.add_argument('--config', default='configs/data_config.yaml',
                       help='Path to configuration file')
    args = parser.parse_args()

    # Load configuration
    config = load_config(Path(args.config))
    
    # Load metadata
    logger.info("Loading metadata...")
    metadata_loader = MetadataLoader(config["metadata_path"])
    metadata = metadata_loader.load()
    
    logger.info(f"Total samples: {metadata.height}")
    
    # Create splits
    logger.info("Creating stratified splits...")
    splitter = StratifiedSplitter(
        train_size=0.85,
        val_size=0.0,
        test_size=0.15,
        random_state=42
    )
    
    # Stratify by sample_type (ancient/modern) as primary, 
    # but the robust splitter handles class imbalance if we passed a different column.
    # The reference script used 'True_label' which seems to correspond to 'community_type' or similar in Logan's data.
    # Here we use 'sample_type' as per original script, but maybe we should use a more granular one?
    # The PROJECT_STRUCTURE.md says "Multi-task classification...".
    # Let's stick to 'sample_type' for now or check if we should use something else.
    # Actually, for multi-task, stratifying by the most imbalanced or important target is usually best.
    # 'sample_type' is binary (ancient/modern). 'community_type' has 6 classes.
    # Let's use 'community_type' if available, as it's more granular.
    
    stratify_col = "community_type" if "community_type" in metadata.columns else "sample_type"
    logger.info(f"Stratifying by: {stratify_col}")

    train_ids, val_ids, test_ids = splitter.split(
        metadata,
        stratify_by=stratify_col,
        id_col="Run_accession"
    )
    
    logger.info(f"Train samples: {len(train_ids)}")
    logger.info(f"Val samples: {len(val_ids)}")
    logger.info(f"Test samples: {len(test_ids)}")
    
    # Save splits
    output_dir = Path(config["splits_dir"])
    splitter.save_splits(train_ids, val_ids, test_ids, output_dir)
    
    logger.info(f"Splits saved to {output_dir}")
    
    # Extract matrices if requested
    if args.extract_matrices:
        logger.info("Extracting matrix subsets...")
        extractor = MatrixExtractor(config["matrix_path"])
        success = extractor.extract(train_ids, val_ids, test_ids, output_dir)
        
        if success:
            logger.info("Matrix extraction successful!")
        else:
            logger.error("Matrix extraction failed!")
            sys.exit(1)

if __name__ == "__main__":
    main()
