#!/usr/bin/env python3
"""Create a small test dataset from the large dataset."""

import sys
import argparse
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from diana.utils.config import load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def extract_columns(matrix_file, output_file, col_indices, matrix_type="PA"):
    """Extract specific columns from a matrix file."""
    logger.info(f"Processing {matrix_type} matrix from {matrix_file}...")
    
    matrix_data = []
    unitig_ids = []
    
    try:
        with open(matrix_file, 'r') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split()
                    unitig_ids.append(parts[0])
                    
                    # Extract only the requested columns
                    # parts[1:] are the values. col_indices are 0-based indices into this array
                    row_values = []
                    for col_idx in col_indices:
                        val = parts[1 + col_idx] # +1 because parts[0] is unitig_id
                        if matrix_type == "PA":
                            row_values.append(str(int(val) if val.isdigit() else 0))
                        else:
                            # Clean float string
                            clean_val = val.replace('.', '').replace('-', '')
                            if clean_val.isdigit():
                                row_values.append(f"{float(val):.6f}" if float(val) != 0 else "0")
                            else:
                                row_values.append("0")
                    
                    matrix_data.append(row_values)
        
        # Write output
        # Format: unitig_id sample1 sample2 ...
        # Wait, the original format seems to be: unitig_id val1 val2 ...
        # But my MatrixExtractor writes: sample_id unitig1 unitig2 ... (transposed)
        # The user said: "Implement matrix loading from kmtricks format"
        # The kmtricks format is usually unitigs as rows.
        # My MatrixExtractor transposed it.
        # If I want to create a "mini-full" dataset, I should keep the original format (unitigs as rows).
        
        logger.info(f"Writing extracted matrix to {output_file}...")
        with open(output_file, 'w') as f:
            for i, unitig_id in enumerate(unitig_ids):
                row = [unitig_id] + matrix_data[i]
                f.write(' '.join(row) + '\n')
                
        return True
        
    except Exception as e:
        logger.error(f"Error extracting matrix: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Create test dataset')
    parser.add_argument('--n-samples', type=int, default=100, help='Number of samples to extract')
    parser.add_argument('--output-dir', default='data/test_data', help='Output directory')
    parser.add_argument('--config', default='configs/data_config.yaml', help='Config file')
    args = parser.parse_args()

    config = load_config(Path(args.config))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load and sample metadata
    logger.info("Loading metadata...")
    metadata_path = Path(config["metadata_path"])
    df = pd.read_csv(metadata_path, sep='\t')
    
    # Filter for available unitigs if possible
    if 'status' in df.columns:
        df = df[df['status'] == 'available_unitigs']
    
    if len(df) > args.n_samples:
        # Stratified sampling if possible
        if 'community_type' in df.columns:
            try:
                sample_df = df.groupby('community_type', group_keys=False).apply(
                    lambda x: x.sample(min(len(x), int(args.n_samples / df['community_type'].nunique() + 1)))
                )
                # If we have too many, sample again
                if len(sample_df) > args.n_samples:
                    sample_df = sample_df.sample(args.n_samples, random_state=42)
            except:
                sample_df = df.sample(args.n_samples, random_state=42)
        else:
            sample_df = df.sample(args.n_samples, random_state=42)
    else:
        sample_df = df
        
    logger.info(f"Selected {len(sample_df)} samples")
    
    # Save test metadata
    metadata_out = output_dir / "metadata.tsv"
    sample_df.to_csv(metadata_out, sep='\t', index=False)
    logger.info(f"Saved test metadata to {metadata_out}")
    
    # 2. Get column indices from kmtricks.fof
    matrix_dir = Path(config["matrix_path"])
    fof_path = matrix_dir / "kmer_matrix" / "kmtricks.fof"
    
    logger.info(f"Reading sample order from {fof_path}...")
    sample_order = []
    with open(fof_path, 'r') as f:
        for line in f:
            if ':' in line:
                sample_order.append(line.split(':')[0].strip())
                
    # Map selected samples to indices
    selected_ids = set(sample_df['Run_accession'])
    col_indices = [i for i, sid in enumerate(sample_order) if sid in selected_ids]
    selected_sample_order = [sample_order[i] for i in col_indices]
    
    logger.info(f"Found {len(col_indices)} matching samples in matrix")
    
    # 3. Extract matrices
    matrix_out_dir = output_dir / "matrices"
    matrix_out_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dummy fof
    kmer_matrix_dir = matrix_out_dir / "kmer_matrix"
    kmer_matrix_dir.mkdir(parents=True, exist_ok=True)
    
    with open(kmer_matrix_dir / "kmtricks.fof", 'w') as f:
        for sid in selected_sample_order:
            f.write(f"{sid} : /dummy/path/{sid}.fa\n")
            
    # Extract PA and Abundance
    pa_in = matrix_dir / "unitigs.pa.mat"
    ab_in = matrix_dir / "unitigs.abundance.mat"
    
    pa_out = matrix_out_dir / "unitigs.pa.mat"
    ab_out = matrix_out_dir / "unitigs.abundance.mat"
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(extract_columns, pa_in, pa_out, col_indices, "PA"),
            executor.submit(extract_columns, ab_in, ab_out, col_indices, "Abundance")
        ]
        
        for future in futures:
            future.result()
            
    logger.info("Done!")

if __name__ == "__main__":
    main()