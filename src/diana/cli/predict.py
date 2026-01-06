#!/usr/bin/env python3
"""
DIANA Predict: Command-line interface for running inference on new samples.
"""

import argparse
import logging
import sys
from pathlib import Path
import time
import json

logger = logging.getLogger(__name__)


def detect_paired_end(sample_path: Path) -> list:
    """
    Detect if a sample is paired-end and return all FASTQ files.
    
    Checks for common paired-end naming patterns:
    - sample_1.fastq.gz / sample_2.fastq.gz
    - sample_R1.fastq.gz / sample_R2.fastq.gz
    - sample.1.fastq.gz / sample.2.fastq.gz
    
    Args:
        sample_path: Path to first FASTQ file
    
    Returns:
        List of Path objects (1 for single-end, 2+ for paired-end)
    """
    # Patterns to check for paired-end
    patterns = [
        ('_1.', '_2.'),
        ('_R1.', '_R2.'),
        ('.1.', '.2.'),
        ('_1_', '_2_'),
    ]
    
    sample_str = str(sample_path)
    
    for p1, p2 in patterns:
        if p1 in sample_str:
            # Try to find the paired file
            pair_path = Path(sample_str.replace(p1, p2))
            if pair_path.exists():
                logger.debug(f"Detected paired-end: {sample_path.name} + {pair_path.name}")
                return sorted([sample_path, pair_path])  # Sort to ensure consistent order
    
    # Not paired-end or pair not found
    return [sample_path]


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def predict_single_sample(
    sample_paths: list,  # Changed to list to support paired-end
    model_path: Path,
    muset_matrix_dir: Path,
    output_dir: Path,
    kmer_size: int = 31,
    min_abundance: int = 2,
    threads: int = 10,
    generate_plots: bool = True
) -> dict:
    """
    Run inference on a single sample (single-end or paired-end).
    
    Orchestrates the pipeline by calling individual scripts directly:
      1. Extract reference k-mers (if needed)
      2. Count k-mers in sample
      3. Aggregate k-mer counts to unitigs
      4. Run model inference
      5. Generate plots (optional)
    
    Args:
        sample_paths: List of FASTQ file paths (1 for single-end, 2 for paired-end)
        model_path: Path to trained model
        muset_matrix_dir: Path to MUSET matrix directory
        output_dir: Output directory for results
        kmer_size: K-mer size
        min_abundance: Minimum k-mer abundance
        threads: Number of threads
        generate_plots: Whether to generate plots
    
    Returns:
        Dictionary with timing and results information
    """
    import subprocess
    
    # Extract sample_id from first file (remove _1, _2, _R1, _R2 suffixes)
    sample_id = sample_paths[0].stem
    sample_id = sample_id.replace('.fastq', '').replace('.fq', '')
    sample_id = sample_id.replace('_1', '').replace('_2', '').replace('_R1', '').replace('_R2', '')
    
    sample_output_dir = output_dir / sample_id
    sample_output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Processing sample: {sample_id}")
    if len(sample_paths) > 1:
        logger.info(f"  Paired-end: {len(sample_paths)} files")
    logger.info(f"  Output directory: {sample_output_dir}")
    
    # Get the project root and script paths
    project_root = Path(__file__).parent.parent.parent.parent
    scripts_dir = project_root / "scripts" / "inference"
    
    # Define file paths
    reference_kmers = muset_matrix_dir / "reference_kmers.fasta"
    unitigs_fa = muset_matrix_dir / "unitigs.fa"
    kmer_counts = sample_output_dir / f"{sample_id}_kmer_counts.txt"
    unitig_abundance = sample_output_dir / f"{sample_id}_unitig_abundance.txt"
    unitig_fraction = sample_output_dir / f"{sample_id}_unitig_fraction.txt"
    predictions_json = sample_output_dir / f"{sample_id}_predictions.json"
    plots_dir = sample_output_dir / "plots"
    
    start_time = time.time()
    logger.info("-" * 60)
    
    try:
        # ====================================================================
        # Step 0: Verify reference k-mers exist (should be in shared location)
        # ====================================================================
        if not reference_kmers.exists():
            logger.warning(f"Reference k-mers not found: {reference_kmers}")
            logger.info("Generating reference k-mers (this is a one-time operation)...")
            logger.info("TIP: Run install.sh to download pre-generated file from Zenodo")
            step_start = time.time()
            
            result = subprocess.run([
                "bash",
                str(scripts_dir / "00_extract_reference_kmers.sh"),
                str(muset_matrix_dir),
                str(reference_kmers)
            ], check=True, capture_output=True, text=True)
            
            logger.info(f"  ✓ Reference k-mers generated ({time.time() - step_start:.1f}s)")
        else:
            logger.debug(f"Using shared reference k-mers: {reference_kmers}")
        
        # ====================================================================
        # Step 1: Count k-mers in sample
        # ====================================================================
        logger.info("Step 1: Counting k-mers in sample...")
        step_start = time.time()
        
        # Create file list for paired-end samples
        if len(sample_paths) > 1:
            fastq_filelist = sample_output_dir / f"{sample_id}_fastq_filelist.txt"
            with open(fastq_filelist, 'w') as f:
                for fq in sample_paths:
                    f.write(f"{fq}\n")
            kmer_input = str(fastq_filelist)
        else:
            kmer_input = str(sample_paths[0])
        
        result = subprocess.run([
            "bash",
            str(scripts_dir / "01_count_kmers.sh"),
            str(reference_kmers),
            kmer_input,
            str(kmer_counts),
            str(threads),
            str(min_abundance)
        ], check=True, capture_output=True, text=True)
        
        logger.info(f"  ✓ K-mer counting complete ({time.time() - step_start:.1f}s)")
        
        # ====================================================================
        # Step 2: Aggregate k-mer counts to unitigs
        # ====================================================================
        logger.info("Step 2: Aggregating k-mers to unitigs...")
        step_start = time.time()
        
        result = subprocess.run([
            "bash",
            str(scripts_dir / "02_aggregate_to_unitigs.sh"),
            str(kmer_counts),
            str(unitigs_fa),
            str(kmer_size),
            str(unitig_abundance),
            str(unitig_fraction)
        ], check=True, capture_output=True, text=True)
        
        logger.info(f"  ✓ Unitig aggregation complete ({time.time() - step_start:.1f}s)")
        
        # ====================================================================
        # Step 3: Run model inference
        # ====================================================================
        logger.info("Step 3: Running model inference...")
        step_start = time.time()
        
        result = subprocess.run([
            "python",
            str(scripts_dir / "03_run_inference.py"),
            "--model", str(model_path),
            "--input", str(unitig_fraction),
            "--output", str(predictions_json),
            "--sample-id", sample_id
        ], check=True, capture_output=True, text=True)
        
        logger.info(f"  ✓ Model inference complete ({time.time() - step_start:.1f}s)")
        
        # ====================================================================
        # Step 4: Generate plots (optional)
        # ====================================================================
        if generate_plots:
            logger.info("Step 4: Generating plots...")
            step_start = time.time()
            
            # Get label encoders path from model directory
            label_encoders_dir = model_path.parent
            
            result = subprocess.run([
                "python",
                str(scripts_dir / "04_plot_results.py"),
                "--predictions", str(predictions_json),
                "--output_dir", str(plots_dir),
                "--sample_id", sample_id,
                "--label_encoders", str(label_encoders_dir)
            ], check=True, capture_output=True, text=True)
            
            logger.info(f"  ✓ Plots generated ({time.time() - step_start:.1f}s)")
        
        logger.info("-" * 60)
        elapsed_time = time.time() - start_time
        logger.info(f"Pipeline completed in {elapsed_time:.1f}s")
        
        # Load predictions
        if predictions_json.exists():
            with open(predictions_json) as f:
                predictions = json.load(f)
        else:
            predictions = None
            logger.warning(f"Predictions file not found: {predictions_json}")
        
        return {
            "sample_id": sample_id,
            "status": "success",
            "elapsed_time": elapsed_time,
            "predictions": predictions,
            "output_dir": str(sample_output_dir)
        }
        
    except subprocess.CalledProcessError as e:
        logger.info("-" * 60)
        elapsed_time = time.time() - start_time
        logger.error(f"Pipeline failed after {elapsed_time:.1f}s")
        logger.error(f"Command: {' '.join(e.cmd)}")
        logger.error(f"Exit code: {e.returncode}")
        
        # Log stderr if available
        if e.stderr:
            logger.error(f"Error output:\\n{e.stderr}")
        
        return {
            "sample_id": sample_id,
            "status": "failed",
            "elapsed_time": elapsed_time,
            "error": f"Pipeline failed: {e.stderr if e.stderr else 'Unknown error'}",
            "exit_code": e.returncode,
            "output_dir": str(sample_output_dir)
        }
    
    except Exception as e:
        logger.info("-" * 60)
        elapsed_time = time.time() - start_time
        logger.error(f"Unexpected error after {elapsed_time:.1f}s")
        logger.error(f"Error: {str(e)}")
        
        return {
            "sample_id": sample_id,
            "status": "failed",
            "elapsed_time": elapsed_time,
            "error": str(e),
            "output_dir": str(sample_output_dir)
        }


def main():
    parser = argparse.ArgumentParser(
        description="DIANA: Deep learning-based Identification and ANnotation of Ancient DNA samples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single sample
  diana-predict --sample data/sample.fastq.gz --model results/training/best_model.pth \\
                --muset-matrix data/matrices/matrix/ --output results/inference/

  # Multiple samples
  diana-predict --sample data/*.fastq.gz --model results/training/best_model.pth \\
                --muset-matrix data/matrices/matrix/ --output results/inference/

  # Batch mode with sample list
  diana-predict --batch samples.txt --model results/training/best_model.pth \\
                --muset-matrix data/matrices/matrix/ --output results/inference/
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--sample', '-s',
        type=str,
        nargs='+',
        help='Path(s) to non-empty gzipped FASTQ file(s) (*.fastq.gz or *.fq.gz, supports wildcards)'
    )
    input_group.add_argument(
        '--batch', '-b',
        type=Path,
        help='Text file with one sample path per line'
    )
    
    # Model and data
    parser.add_argument(
        '--model', '-m',
        type=Path,
        required=True,
        help='Path to trained model checkpoint'
    )
    parser.add_argument(
        '--muset-matrix',
        type=Path,
        required=True,
        help='Path to MUSET matrix directory'
    )
    
    # Output
    parser.add_argument(
        '--output', '-o',
        type=Path,
        required=True,
        help='Output directory for results'
    )
    
    # Pipeline parameters
    parser.add_argument(
        '--kmer-size', '-k',
        type=int,
        default=31,
        help='K-mer size (default: 31)'
    )
    parser.add_argument(
        '--min-abundance',
        type=int,
        default=2,
        help='Minimum k-mer abundance threshold (default: 2)'
    )
    parser.add_argument(
        '--threads', '-t',
        type=int,
        default=10,
        help='Number of threads (default: 10)'
    )
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip plot generation'
    )
    
    # Logging
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    
    # Validate paths
    if not args.model.exists():
        logger.error(f"Model not found: {args.model}")
        sys.exit(1)
    
    if not args.muset_matrix.exists():
        logger.error(f"MUSET matrix directory not found: {args.muset_matrix}")
        sys.exit(1)
    
    # Collect samples
    samples = []
    if args.sample:
        from glob import glob
        for pattern in args.sample:
            matches = glob(pattern)
            if not matches:
                logger.warning(f"No files match pattern: {pattern}")
            samples.extend([Path(p) for p in matches])
    else:
        with open(args.batch) as f:
            samples = [Path(line.strip()) for line in f if line.strip()]
    
    if not samples:
        logger.error("No samples to process")
        sys.exit(1)
    
    logger.info(f"Found {len(samples)} sample(s) to process")
    
    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    
    # Detect paired-end and group samples
    processed_samples = set()
    sample_groups = []
    
    for sample_path in samples:
        if sample_path in processed_samples:
            continue
        
        # Detect if paired-end
        sample_files = detect_paired_end(sample_path)
        sample_groups.append(sample_files)
        
        # Mark all files in this group as processed
        for f in sample_files:
            processed_samples.add(f)
    
    logger.info(f"Detected {len(sample_groups)} unique sample(s) to process")
    paired_count = sum(1 for g in sample_groups if len(g) > 1)
    if paired_count > 0:
        logger.info(f"  {paired_count} paired-end samples")
        logger.info(f"  {len(sample_groups) - paired_count} single-end samples")
    
    # Process samples
    results = []
    for i, sample_files in enumerate(sample_groups, 1):
        # Check all files exist
        missing = [f for f in sample_files if not f.exists()]
        if missing:
            logger.error(f"Sample file(s) not found: {', '.join(str(f) for f in missing)}")
            results.append({
                "sample_id": sample_files[0].stem,
                "status": "not_found",
                "error": f"File(s) not found: {missing}"
            })
            continue
        
        # Validate files are non-empty gzipped FASTQ
        invalid_files = []
        for f in sample_files:
            # Check file extension
            if not (f.suffix == '.gz' and f.stem.endswith(('.fastq', '.fq'))):
                invalid_files.append(f"{f}: not a gzipped FASTQ file (*.fastq.gz or *.fq.gz)")
            # Check file is not empty
            elif f.stat().st_size == 0:
                invalid_files.append(f"{f}: empty file (0 bytes)")
        
        if invalid_files:
            logger.error(f"Invalid sample file(s): {'; '.join(invalid_files)}")
            results.append({
                "sample_id": sample_files[0].stem,
                "status": "invalid",
                "error": f"Invalid file(s): {'; '.join(invalid_files)}"
            })
            continue
        
        sample_name = sample_files[0].name
        if len(sample_files) > 1:
            sample_name += f" + {len(sample_files)-1} more"
        logger.info(f"[{i}/{len(sample_groups)}] Processing {sample_name}")
        
        result = predict_single_sample(
            sample_paths=sample_files,
            model_path=args.model,
            muset_matrix_dir=args.muset_matrix,
            output_dir=args.output,
            kmer_size=args.kmer_size,
            min_abundance=args.min_abundance,
            threads=args.threads,
            generate_plots=not args.no_plots
        )
        
        results.append(result)
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)
    
    success_count = sum(1 for r in results if r["status"] == "success")
    failed_count = len(results) - success_count
    total_time = sum(r.get("elapsed_time", 0) for r in results)
    
    logger.info(f"Total samples: {len(results)}")
    logger.info(f"Successful: {success_count}")
    logger.info(f"Failed: {failed_count}")
    logger.info(f"Total time: {total_time:.1f}s ({total_time/60:.1f}min)")
    if success_count > 0:
        logger.info(f"Average time per sample: {total_time/success_count:.1f}s")
    
    # Save summary
    summary_file = args.output / "diana_predict_summary.json"
    with open(summary_file, 'w') as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_samples": len(results),
            "successful": success_count,
            "failed": failed_count,
            "total_time_seconds": total_time,
            "results": results
        }, f, indent=2)
    
    logger.info(f"\nSummary saved to: {summary_file}")
    logger.info("="*60)
    
    # Exit with error if any samples failed
    sys.exit(0 if failed_count == 0 else 1)


if __name__ == "__main__":
    main()
