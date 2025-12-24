#!/usr/bin/env python3
"""
Extract DIANA samples from full unitig matrix and split into train/test.

Reads the large unitig matrix (104K features × 3K samples), extracts only the
DIANA dataset samples (3070 samples), and splits them according to the
stratified train/test split (85%/15%).

Usage:
    python scripts/data_prep/05_extract_and_split_matrices.py \\
        --source-matrix /path/to/large_matrix_3116 \\
        --output data/extracted_matrices
"""

import argparse
import logging
from pathlib import Path
import polars as pl
import numpy as np
import shutil

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def load_sample_order(fof_path: Path) -> list[str]:
    """Load sample IDs from kmtricks.fof (defines matrix column order)."""
    with open(fof_path, 'r') as f:
        sample_ids = [line.strip().split()[0] for line in f]
    return sample_ids

def load_split_ids(split_file: Path) -> list[str]:
    """Load sample IDs from split file."""
    with open(split_file, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def load_matrix(matrix_path: Path) -> np.ndarray:
    """Load matrix from .mat format (first column = feature IDs, rest = data)."""
    logger.info(f"Loading matrix from {matrix_path}")
    df = pl.read_csv(matrix_path, separator=' ', has_header=False, rechunk=True)
    logger.info(f"Matrix loaded: {df.shape[0]:,} features × {df.shape[1]-1:,} samples")
    return df[:, 1:].to_numpy().astype(np.float32)

def save_matrix(matrix: np.ndarray, sample_ids: list[str], output_path: Path):
    """Save matrix in .mat format with sample IDs as first column."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame(matrix.T).insert_column(0, pl.Series("Run_accession", sample_ids))
    df.write_csv(output_path, separator=' ', include_header=False)
    logger.info(f"Saved {len(sample_ids)} samples × {matrix.shape[0]} features to {output_path.name}")

def main():
    parser = argparse.ArgumentParser(
        description='Extract DIANA samples from full matrix and split into train/test',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--source-matrix', type=Path, required=True,
                       help='Path to source matrix directory (e.g., large_matrix_3116)')
    parser.add_argument('--metadata', type=Path,
                       default=PROJECT_ROOT / "data" / "metadata" / "DIANA_metadata.tsv",
                       help='Path to DIANA metadata TSV')
    parser.add_argument('--splits-dir', type=Path,
                       default=PROJECT_ROOT / "data" / "splits",
                       help='Path to directory with train/test split IDs')
    parser.add_argument('--output', type=Path,
                       default=PROJECT_ROOT / "data" / "test_data",
                       help='Output directory for extracted matrices')
    args = parser.parse_args()
    
    logger.info("Starting matrix extraction and splitting...")
    logger.info(f"Source matrix: {args.source_matrix}")
    logger.info(f"Output directory: {args.output}")
    
    # Load sample metadata and splits
    all_sample_ids = load_sample_order(args.source_matrix / "kmer_matrix" / "kmtricks.fof")
    diana_samples = set(pl.read_csv(args.metadata, separator='\t')['Run_accession'].to_list())
    train_ids = load_split_ids(args.splits_dir / "train_ids.txt")
    test_ids = load_split_ids(args.splits_dir / "test_ids.txt")
    
    logger.info(f"Train: {len(train_ids)}, Test: {len(test_ids)}")
    
    # Build sample index mappings
    train_set, test_set = set(train_ids), set(test_ids)
    diana_indices, diana_sample_ids = [], []
    train_indices, train_sample_ids = [], []
    test_indices, test_sample_ids = [], []
    
    for idx, sample_id in enumerate(all_sample_ids):
        if sample_id in diana_samples:
            diana_indices.append(idx)
            diana_sample_ids.append(sample_id)
            
            diana_idx = len(diana_sample_ids) - 1
            if sample_id in train_set:
                train_indices.append(diana_idx)
                train_sample_ids.append(sample_id)
            elif sample_id in test_set:
                test_indices.append(diana_idx)
                test_sample_ids.append(sample_id)
    
    logger.info(f"Found {len(diana_indices)} DIANA samples (Train: {len(train_sample_ids)}, Test: {len(test_sample_ids)})")
    
    # Process both PA and abundance matrices
    for matrix_type in ['pa', 'abundance', 'frac']:
        logger.info(f"\nProcessing {matrix_type} matrix...")
        
        matrix_file = args.source_matrix / f"unitigs.{matrix_type}.mat"
        if not matrix_file.exists():
            logger.warning(f"Matrix not found: {matrix_file}")
            continue
        
        # Load and extract DIANA samples
        matrix = load_matrix(matrix_file)
        diana_matrix = matrix[:, diana_indices]
        
        # Save matrices
        output_dir = args.output / "matrices"
        save_matrix(diana_matrix, diana_sample_ids, output_dir / f"unitigs.{matrix_type}.mat")
        save_matrix(diana_matrix[:, train_indices], train_sample_ids, 
                   args.output / "splits" / f"train_matrix.{matrix_type}.mat")
        save_matrix(diana_matrix[:, test_indices], test_sample_ids,
                   args.output / "splits" / f"test_matrix.{matrix_type}.mat")
    
    # Copy split configuration files
    logger.info("\nCopying split configuration...")
    splits_output = args.output / "splits"
    splits_output.mkdir(parents=True, exist_ok=True)
    
    for split_file in ['train_ids.txt', 'test_ids.txt', 'val_ids.txt', 'split_config.json']:
        src = args.splits_dir / split_file
        if src.exists():
            shutil.copy(src, splits_output / split_file)
    
    shutil.copy(args.metadata, args.output / "metadata.tsv")
    
    logger.info(f"\n✅ Matrix extraction complete! Output: {args.output}")

if __name__ == "__main__":
    main()
