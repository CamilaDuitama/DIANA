#!/usr/bin/env python3
"""
DIANA Project: Project new samples onto reference PCA space.

This command:
1. Processes FASTQ files to get unitig fractions (using diana-predict pipeline)
2. Projects the sample onto reference PCA coordinates (training samples)
3. Generates interactive visualizations showing where the sample falls
4. Shows nearest neighbors in PCA space

Usage:
    diana-project --sample sample.fastq.gz --model results/training/best_model.pth \\
                  --muset-matrix data/matrices/large_matrix_3070_with_frac/
"""

import argparse
import logging
import sys
import shutil
from pathlib import Path
import json
import subprocess
import tempfile
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
import pickle

# Import from predict module for sample processing
from diana.cli.predict import (
    detect_paired_end, 
    validate_fastq_file,
    setup_logging,
    run_command_streaming
)

logger = logging.getLogger(__name__)


def process_sample_for_pca(
    sample_paths: list,
    model_path: Path,
    muset_matrix_dir: Path,
    output_dir: Path,
    kmer_size: int = 31,
    min_abundance: int = 2,
    threads: int = 10
) -> tuple[Path, str]:
    """
    Process a sample to get unitig fractions (without running full inference).
    
    Returns:
        (unitig_fraction_file, sample_id)
    """
    # Same logic as predict_single_sample but stops after Step 2
    import re
    import time
    
    sample_name = sample_paths[0].name
    sample_name = re.sub(r'\.(fastq|fq)(\.gz)?$', '', sample_name)
    sample_id = re.sub(r'(_R?[12]|\.R?[12])(_.*)?$', '', sample_name)
    
    sample_output_dir = output_dir / sample_id
    sample_output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Processing sample: {sample_id}")
    if len(sample_paths) > 1:
        logger.info(f"  Paired-end: {len(sample_paths)} files")
    logger.info(f"  Output directory: {sample_output_dir}")
    
    # Define file paths
    reference_kmers = muset_matrix_dir / "reference_kmers.fasta"
    unitigs_fa = muset_matrix_dir / "unitigs.fa"
    kmer_counts = sample_output_dir / f"{sample_id}_kmer_counts.txt"
    unitig_abundance = sample_output_dir / f"{sample_id}_unitig_abundance.txt"
    unitig_fraction = sample_output_dir / f"{sample_id}_unitig_fraction.txt"
    
    start_time = time.time()
    logger.info("-" * 60)
    
    try:
        # Step 0: Verify reference k-mers exist
        if not reference_kmers.exists():
            logger.warning(f"Reference k-mers not found: {reference_kmers}")
            logger.info("Generating reference k-mers (one-time operation)...")
            
            run_command_streaming([
                "00_extract_reference_kmers.sh",
                str(muset_matrix_dir),
                str(reference_kmers)
            ], "Extracting reference k-mers")
        else:
            logger.debug(f"Using shared reference k-mers: {reference_kmers}")
        
        # Step 1: Count k-mers in sample
        if len(sample_paths) > 1:
            fastq_filelist = sample_output_dir / f"{sample_id}_fastq_filelist.txt"
            with open(fastq_filelist, 'w') as f:
                for fq in sample_paths:
                    f.write(f"{fq}\n")
            kmer_input = str(fastq_filelist)
        else:
            kmer_input = str(sample_paths[0])
        
        run_command_streaming([
            "01_count_kmers.sh",
            str(reference_kmers),
            kmer_input,
            str(kmer_counts),
            str(threads),
            str(min_abundance)
        ], "Step 1: Counting k-mers in sample")
        
        # Step 2: Aggregate k-mer counts to unitigs
        run_command_streaming([
            "02_aggregate_to_unitigs.sh",
            str(kmer_counts),
            str(unitigs_fa),
            str(kmer_size),
            str(unitig_abundance),
            str(unitig_fraction)
        ], "Step 2: Aggregating k-mers to unitigs")
        
        logger.info("-" * 60)
        elapsed_time = time.time() - start_time
        logger.info(f"Sample processing completed in {elapsed_time:.1f}s")
        
        return unitig_fraction, sample_id
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Sample processing failed: {e}")
        logger.error(f"Command: {' '.join(e.cmd)}")
        raise


def load_pca_reference(models_dir: Path = Path("models")):
    """Load saved PCA model and reference data."""
    pca_path = models_dir / "pca_reference.pkl"
    
    if not pca_path.exists():
        logger.error(f"PCA reference not found: {pca_path}")
        logger.error("")
        logger.error("Please generate PCA reference first:")
        logger.error("  python scripts/paper/06_generate_pca_analysis.py")
        raise FileNotFoundError(f"PCA reference not found: {pca_path}")
    
    logger.info(f"Loading PCA reference from {pca_path}...")
    
    with open(pca_path, 'rb') as f:
        reference_data = pickle.load(f)
    
    logger.info(f"  ✓ PCA with {reference_data['pca_model'].n_components_} components")
    logger.info(f"  ✓ Reference: {len(reference_data['sample_ids'])} samples")
    logger.info(f"  ✓ Features: {reference_data['n_features']} unitigs")
    
    return reference_data


def project_sample(unitig_fraction_file: Path, reference_data: dict, sample_id: str):
    """Project sample onto reference PCA space."""
    logger.info(f"\nProjecting {sample_id} onto reference PCA space...")
    
    # Load unitig fraction vector
    unitig_vector = np.loadtxt(unitig_fraction_file)
    logger.info(f"  Sample vector: {unitig_vector.shape[0]} features")
    
    # Validate dimensions match
    if unitig_vector.shape[0] != reference_data['n_features']:
        raise ValueError(
            f"Feature dimension mismatch: sample has {unitig_vector.shape[0]} features, "
            f"reference expects {reference_data['n_features']}"
        )
    
    # Reshape to (1, n_features)
    sample_matrix = unitig_vector.reshape(1, -1)
    
    # Apply standardization if reference was standardized
    if reference_data.get('standardized', False) and 'scaler' in reference_data:
        logger.info("  Applying standardization (matching reference)...")
        sample_matrix = reference_data['scaler'].transform(sample_matrix)
    
    # Project onto PCA space
    sample_pca = reference_data['pca_model'].transform(sample_matrix)
    logger.info(f"  ✓ Projected to {sample_pca.shape[1]}D PCA space")
    
    return sample_pca


def compute_neighbors(sample_pca, reference_pca, reference_sample_ids, k=5):
    """Find k nearest neighbors in PCA space."""
    from scipy.spatial.distance import cdist
    
    distances = cdist(sample_pca, reference_pca, metric='euclidean')[0]
    nearest_indices = np.argsort(distances)[:k]
    nearest_ids = [reference_sample_ids[i] for i in nearest_indices]
    nearest_distances = distances[nearest_indices]
    
    return nearest_ids, nearest_distances


def plot_pca_projection(
    sample_pca, 
    sample_id, 
    reference_pca, 
    reference_sample_ids,
    reference_metadata,
    output_dir: Path,
    nearest_ids=None
):
    """Create interactive PCA plot showing new sample and reference."""
    logger.info("\nGenerating PCA projection plots...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create plots for each task
    tasks = ['sample_type', 'material', 'sample_host', 'community_type']
    
    for task in tasks:
        logger.info(f"  Creating plot for {task}...")
        
        fig = go.Figure()
        
        # Plot reference samples colored by task
        ref_labels = reference_metadata[task].values
        unique_labels = sorted(list(set(ref_labels)))
        
        for label in unique_labels:
            mask = ref_labels == label
            indices = np.where(mask)[0]
            
            fig.add_trace(go.Scatter(
                x=reference_pca[indices, 0],
                y=reference_pca[indices, 1],
                mode='markers',
                name=label,
                marker=dict(size=5, opacity=0.6),
                text=[reference_sample_ids[i] for i in indices],
                hovertemplate='<b>%{text}</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>'
            ))
        
        # Highlight nearest neighbors if provided
        if nearest_ids:
            neighbor_indices = [i for i, sid in enumerate(reference_sample_ids) if sid in nearest_ids]
            fig.add_trace(go.Scatter(
                x=reference_pca[neighbor_indices, 0],
                y=reference_pca[neighbor_indices, 1],
                mode='markers',
                name='Nearest Neighbors',
                marker=dict(
                    size=12,
                    color='yellow',
                    symbol='diamond',
                    line=dict(width=2, color='black')
                ),
                text=[reference_sample_ids[i] for i in neighbor_indices],
                hovertemplate='<b>%{text}</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>'
            ))
        
        # Plot new sample
        fig.add_trace(go.Scatter(
            x=sample_pca[:, 0],
            y=sample_pca[:, 1],
            mode='markers',
            name=f'New Sample: {sample_id}',
            marker=dict(
                size=15,
                color='red',
                symbol='star',
                line=dict(width=2, color='black')
            ),
            hovertemplate=f'<b>{sample_id}</b><br>PC1: %{{x:.2f}}<br>PC2: %{{y:.2f}}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f'PCA Projection - {task.replace("_", " ").title()}',
            xaxis_title='PC1',
            yaxis_title='PC2',
            template='plotly_white',
            hovermode='closest',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255,255,255,0.8)"
            ),
            width=1000,
            height=800
        )
        
        # Save
        html_path = output_dir / f'pca_projection_{task}.html'
        fig.write_html(str(html_path))
        logger.info(f"    ✓ Saved: {html_path.name}")


def main():
    parser = argparse.ArgumentParser(
        description="DIANA Project: Project new samples onto reference PCA space",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  diana-project --sample data/new_sample.fastq.gz \\
                --model results/training/best_model.pth \\
                --muset-matrix data/matrices/large_matrix_3070_with_frac/

This will:
  1. Process the FASTQ file to generate unitig fractions
  2. Project onto reference PCA space (training samples)
  3. Generate interactive plots showing sample position
  4. Identify and highlight nearest neighbors
        """
    )
    
    # Input
    parser.add_argument(
        '--sample', '-s',
        type=str,
        nargs='+',
        required=True,
        help='Path(s) to FASTQ file(s) (*.fastq.gz, supports paired-end)'
    )
    
    # Model and matrix
    parser.add_argument(
        '--model', '-m',
        type=Path,
        required=True,
        help='Path to trained model (used for labeling)'
    )
    parser.add_argument(
        '--muset-matrix',
        type=Path,
        required=True,
        help='Path to MUSET matrix directory'
    )
    
    # PCA reference
    parser.add_argument(
        '--pca-reference',
        type=Path,
        default=Path("models/pca_reference.pkl"),
        help='Path to PCA reference file (default: models/pca_reference.pkl)'
    )
    
    # Output
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=Path("results/pca_projection"),
        help='Output directory (default: results/pca_projection)'
    )
    
    # Processing parameters
    parser.add_argument(
        '--threads', '-t',
        type=int,
        default=10,
        help='Number of threads (default: 10)'
    )
    parser.add_argument(
        '--k-neighbors',
        type=int,
        default=5,
        help='Number of nearest neighbors to highlight (default: 5)'
    )
    
    # Logging
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    
    logger.info("=" * 70)
    logger.info("DIANA PROJECT: Project sample onto reference PCA space")
    logger.info("=" * 70)
    
    # Validate inputs
    if not args.model.exists():
        logger.error(f"Model not found: {args.model}")
        sys.exit(1)
    
    if not args.muset_matrix.exists():
        logger.error(f"MUSET matrix not found: {args.muset_matrix}")
        sys.exit(1)
    
    # Collect samples
    from glob import glob
    samples = []
    for pattern in args.sample:
        matches = glob(pattern)
        if not matches:
            logger.warning(f"No files match pattern: {pattern}")
        samples.extend([Path(p) for p in matches])
    
    if not samples:
        logger.error("No samples to process")
        sys.exit(1)
    
    # Detect paired-end
    sample_files = detect_paired_end(samples[0])
    
    # Validate files
    for f in sample_files:
        is_valid, error_msg = validate_fastq_file(f)
        if not is_valid:
            logger.error(f"Invalid sample file: {f.name}: {error_msg}")
            sys.exit(1)
    
    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Process sample to get unitig fractions
    logger.info("\n" + "=" * 70)
    logger.info("STEP 1: Processing sample to generate unitig fractions")
    logger.info("=" * 70)
    
    unitig_fraction_file, sample_id = process_sample_for_pca(
        sample_paths=sample_files,
        model_path=args.model,
        muset_matrix_dir=args.muset_matrix,
        output_dir=args.output,
        threads=args.threads
    )
    
    # Step 2: Load PCA reference
    logger.info("\n" + "=" * 70)
    logger.info("STEP 2: Loading reference PCA space")
    logger.info("=" * 70)
    
    reference_data = load_pca_reference(args.pca_reference.parent)
    
    # Step 3: Project sample
    logger.info("\n" + "=" * 70)
    logger.info("STEP 3: Projecting sample onto PCA space")
    logger.info("=" * 70)
    
    sample_pca = project_sample(unitig_fraction_file, reference_data, sample_id)
    
    # Step 4: Find nearest neighbors
    logger.info("\n" + "=" * 70)
    logger.info("STEP 4: Finding nearest neighbors")
    logger.info("=" * 70)
    
    reference_pca = reference_data['pca_coordinates']
    reference_sample_ids = reference_data['sample_ids']
    
    nearest_ids, nearest_distances = compute_neighbors(
        sample_pca, 
        reference_pca, 
        reference_sample_ids,
        k=args.k_neighbors
    )
    
    logger.info(f"\n  Top {args.k_neighbors} nearest neighbors:")
    for i, (neighbor_id, distance) in enumerate(zip(nearest_ids, nearest_distances), 1):
        logger.info(f"    {i}. {neighbor_id} (distance: {distance:.2f})")
    
    # Step 5: Generate plots
    logger.info("\n" + "=" * 70)
    logger.info("STEP 5: Generating PCA projection plots")
    logger.info("=" * 70)
    
    # Load reference metadata
    metadata_path = Path("paper/metadata/train_metadata.tsv")
    if not metadata_path.exists():
        logger.error(f"Metadata not found: {metadata_path}")
        sys.exit(1)
    
    import polars as pl
    metadata = pl.read_csv(metadata_path, separator='\t')
    
    # Ensure sample order matches reference_sample_ids
    metadata = metadata.filter(pl.col('Run_accession').is_in(reference_sample_ids))
    metadata = metadata.sort(by='Run_accession')
    
    plot_pca_projection(
        sample_pca=sample_pca,
        sample_id=sample_id,
        reference_pca=reference_pca,
        reference_sample_ids=reference_sample_ids,
        reference_metadata=metadata,
        output_dir=args.output / sample_id,
        nearest_ids=nearest_ids
    )
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("PROJECTION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Sample: {sample_id}")
    logger.info(f"Output directory: {args.output / sample_id}")
    logger.info(f"\nPlots:")
    for task in ['sample_type', 'material', 'sample_host', 'community_type']:
        plot_file = args.output / sample_id / f'pca_projection_{task}.html'
        if plot_file.exists():
            logger.info(f"  - {plot_file}")
    logger.info("\n" + "=" * 70)


if __name__ == "__main__":
    main()
