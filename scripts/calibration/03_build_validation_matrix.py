#!/usr/bin/env python3
"""
Build Validation Feature Matrix from Unitig Abundances
=======================================================
Combines individual, two-column (unitig_id, abundance) TSV files
from validation predictions into a single, sparse feature matrix.
"""
import argparse
import logging
from pathlib import Path
import numpy as np
import polars as pl
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Build validation matrix from unitig abundance files")
    parser.add_argument("--predictions-dir", type=Path, default=Path("results/validation_predictions"), help="Directory with prediction subdirectories")
    parser.add_argument("--unitigs-fa", type=Path, default=Path("data/matrices/large_matrix_3070_with_frac/unitigs.fa"), help="Path to unitigs.fa reference file")
    parser.add_argument("--output", type=Path, default=Path("data/validation/validation_matrix.tsv"), help="Output path for the validation matrix")
    args = parser.parse_args()

    logger.info("="*80)
    logger.info("BUILDING VALIDATION FEATURE MATRIX")
    logger.info("="*80)

    # 1. Find all unitig abundance files from SUCCESSFUL predictions only
    abundance_files = []
    for sample_dir in sorted(args.predictions_dir.glob("*")):
        if not sample_dir.is_dir():
            continue
        
        jobinfo = sample_dir / ".jobinfo"
        sample_id = sample_dir.name
        abundance_file = sample_dir / f"{sample_id}_unitig_abundance.txt"
        
        # Skip if no abundance file
        if not abundance_file.exists():
            continue
        
        # Check job status if .jobinfo exists
        if jobinfo.exists():
            try:
                import json
                with open(jobinfo) as f:
                    job_data = json.load(f)
                if job_data.get("status") != "SUCCESS":
                    logger.debug(f"Skipping {sample_id}: status={job_data.get('status')}")
                    continue
            except Exception as e:
                logger.warning(f"Failed to read {jobinfo}: {e}")
                # If we can't read jobinfo, still include the file
        
        abundance_files.append(abundance_file)
    
    if not abundance_files:
        raise FileNotFoundError(f"No successful *_unitig_abundance.txt files found in {args.predictions_dir}")
    logger.info(f"Found {len(abundance_files)} unitig abundance files from successful predictions.")

    # 2. Read all files and stack into matrix
    # Files are single-column (just abundances), pre-aligned to training matrix feature order
    logger.info("Reading abundance vectors...")
    sample_ids = []
    feature_vectors = []
    n_features = None
    skipped_samples = []
    
    for f in tqdm(abundance_files, desc="Loading samples"):
        sample_id = f.stem.replace('_unitig_abundance', '')
        
        # Read single-column abundance values
        # Some files may have blank lines (meaning 0.0 abundance for all features)
        try:
            with open(f) as file:
                lines = file.readlines()
            
            # Convert to float, treating empty/blank lines as 0.0
            abundances = np.array([
                float(line.strip()) if line.strip() else 0.0 
                for line in lines
            ], dtype=np.float32)
            
            if abundances.size == 0:
                logger.warning(f"Empty file (0 lines): {sample_id}")
                skipped_samples.append(sample_id)
                continue
            
            # Set expected number of features from first valid file
            if n_features is None:
                n_features = len(abundances)
            elif len(abundances) != n_features:
                logger.warning(f"Feature count mismatch in {sample_id}: expected {n_features}, got {len(abundances)}")
                skipped_samples.append(sample_id)
                continue
            
            sample_ids.append(sample_id)
            feature_vectors.append(abundances.tolist())
        except Exception as e:
            logger.warning(f"Failed to read {sample_id}: {e}")
            skipped_samples.append(sample_id)
            continue
    
    if skipped_samples:
        logger.warning(f"Skipped {len(skipped_samples)} samples with errors: {skipped_samples}")
    
    # 3. Load actual k-mer IDs from unitigs.fa (CRITICAL for matrix alignment)
    logger.info(f"Loading feature IDs from {args.unitigs_fa}...")
    feature_ids = []
    with open(args.unitigs_fa) as f:
        for line in f:
            if line.startswith('>'):
                # Extract k-mer ID (e.g., >496659 LN:i:61 -> 496659)
                kmer_id = line[1:].split()[0]
                feature_ids.append(kmer_id)
    
    if len(feature_ids) == 0:
        raise ValueError(f"No feature IDs found in {args.unitigs_fa}")
    logger.info(f"Loaded {len(feature_ids):,} feature IDs from unitigs.fa")
    
    # 4. Create matrix (samples as columns to match training matrix format)
    # Training matrix format: features as rows, samples as columns, space-separated, NO HEADER
    logger.info("Building feature matrix...")
    n_samples = len(sample_ids)
    n_features = len(feature_vectors[0])
    logger.info(f"Matrix dimensions: {n_features:,} features × {n_samples} samples")
    
    # Verify feature count matches
    if len(feature_ids) != n_features:
        raise ValueError(f"Feature count mismatch: unitigs.fa has {len(feature_ids)} but abundance files have {n_features}")
    
    # Build matrix as list of rows (features as rows)
    matrix_rows = []
    for i in range(n_features):
        row = [feature_ids[i]] + [str(feature_vectors[j][i]) for j in range(n_samples)]
        matrix_rows.append(' '.join(row))
    
    # 5. Save matrix (space-separated, no header, features as rows)
    logger.info(f"Saving matrix to {args.output}...")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        for row in matrix_rows:
            f.write(row + '\n')

    logger.info(f"✓ Validation matrix saved to {args.output}")
    logger.info(f"  Shape: {n_features:,} features × {n_samples} samples")
    logger.info(f"  Size: {args.output.stat().st_size / (1024**2):.1f} MB")
    
    # Save sample IDs for reference
    sample_id_file = args.output.parent / "validation_sample_ids.txt"
    with open(sample_id_file, 'w') as f:
        for sid in sample_ids:
            f.write(f"{sid}\n")
    logger.info(f"  Sample IDs saved to {sample_id_file}")
    
    logger.info("\n" + "="*80)
    logger.info("DONE!")
    logger.info("="*80)

if __name__ == "__main__":
    main()