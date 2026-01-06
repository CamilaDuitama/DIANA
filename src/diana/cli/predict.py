#!/usr/bin/env python3
"""
DIANA Predict: Command-line interface for running inference on new samples.
"""

import argparse
import logging
import sys
import shutil
import re
from pathlib import Path
import time
import json
import subprocess

logger = logging.getLogger(__name__)


def run_command_streaming(cmd: list, step_name: str) -> None:
    """
    Execute a command and stream output in real-time.
    
    Args:
        cmd: Command and arguments as list
        step_name: Name of the step for logging
    
    Raises:
        subprocess.CalledProcessError: If command fails
    """
    logger.info(f"{step_name}...")
    step_start = time.time()
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    # Stream output in real-time
    output_lines = []
    if process.stdout:
        for line in process.stdout:
            line = line.rstrip()
            if line:  # Only log non-empty lines
                logger.debug(f"  {line}")
                output_lines.append(line)
    
    # Wait for process to complete
    returncode = process.wait()
    
    if returncode != 0:
        error_output = '\n'.join(output_lines[-20:])  # Last 20 lines
        raise subprocess.CalledProcessError(
            returncode, 
            cmd, 
            stderr=error_output
        )
    
    logger.info(f"  ✓ {step_name} complete ({time.time() - step_start:.1f}s)")


def detect_paired_end(sample_path: Path) -> list:
    """
    Detect if a sample is paired-end and return all FASTQ files.
    
    Uses regex to match paired-end patterns at the end of the filename stem:
    - sample_1.fastq.gz / sample_2.fastq.gz
    - sample_R1.fastq.gz / sample_R2.fastq.gz
    - sample.1.fastq.gz / sample.2.fastq.gz
    
    Args:
        sample_path: Path to first FASTQ file
    
    Returns:
        List of Path objects (1 for single-end, 2+ for paired-end)
    """
    # Remove .fastq.gz or .fq.gz extension
    name = sample_path.name
    name = re.sub(r'\.(fastq|fq)(\.gz)?$', '', name)
    
    # Patterns to check for paired-end (anchored to end of filename)
    patterns = [
        (r'_1$', '_2'),
        (r'_R1$', '_R2'),
        (r'\.1$', '.2'),
        (r'_1_$', '_2_'),
    ]
    
    for pattern, replacement in patterns:
        match = re.search(pattern, name)
        if match:
            # Build the paired filename
            pair_name = re.sub(pattern, replacement, name)
            # Reconstruct full path with original extension
            original_ext = sample_path.name[len(name):]
            pair_path = sample_path.parent / (pair_name + original_ext)
            
            if pair_path.exists():
                logger.debug(f"Detected paired-end: {sample_path.name} + {pair_path.name}")
                return sorted([sample_path, pair_path])
    
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
    sample_paths: list,
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
    
    Orchestrates the pipeline by calling individual scripts:
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
    # Extract sample_id using regex to remove paired-end suffixes
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
    predictions_json = sample_output_dir / f"{sample_id}_predictions.json"
    plots_dir = sample_output_dir / "plots"
    
    start_time = time.time()
    logger.info("-" * 60)
    
    try:
        # ====================================================================
        # Step 0: Verify reference k-mers exist
        # ====================================================================
        if not reference_kmers.exists():
            logger.warning(f"Reference k-mers not found: {reference_kmers}")
            logger.info("Generating reference k-mers (one-time operation)...")
            logger.info("TIP: Run install.sh to download pre-generated file from Zenodo")
            
            run_command_streaming([
                "00_extract_reference_kmers.sh",
                str(muset_matrix_dir),
                str(reference_kmers)
            ], "Extracting reference k-mers")
        else:
            logger.debug(f"Using shared reference k-mers: {reference_kmers}")
        
        # ====================================================================
        # Step 1: Count k-mers in sample
        # ====================================================================
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
        
        # ====================================================================
        # Step 2: Aggregate k-mer counts to unitigs
        # ====================================================================
        run_command_streaming([
            "02_aggregate_to_unitigs.sh",
            str(kmer_counts),
            str(unitigs_fa),
            str(kmer_size),
            str(unitig_abundance),
            str(unitig_fraction)
        ], "Step 2: Aggregating k-mers to unitigs")
        
        # ====================================================================
        # Step 3: Run model inference
        # ====================================================================
        run_command_streaming([
            "03_run_inference.py",
            "--model", str(model_path),
            "--input", str(unitig_fraction),
            "--output", str(predictions_json),
            "--sample-id", sample_id
        ], "Step 3: Running model inference")
        
        # ====================================================================
        # Step 4: Generate plots (optional)
        # ====================================================================
        if generate_plots:
            label_encoders_dir = model_path.parent
            run_command_streaming([
                "04_plot_results.py",
                "--predictions", str(predictions_json),
                "--output_dir", str(plots_dir),
                "--sample_id", sample_id,
                "--label_encoders", str(label_encoders_dir)
            ], "Step 4: Generating plots")
        
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
        error_msg = e.stderr if e.stderr else 'Unknown error'
        if e.stderr:
            logger.error(f"Error output:\n{e.stderr}")
            
            # Detect OOM symptoms and provide helpful guidance
            oom_indicators = [
                "Is the file empty?",
                "Failed to read the first two bytes",
                "Broken pipe",
                "std::bad_alloc",
                "Cannot allocate memory"
            ]
            if any(indicator in e.stderr for indicator in oom_indicators):
                logger.error("")
                logger.error("=" * 60)
                logger.error("DEBUGGING HINT: Possible Out-Of-Memory (OOM) Issue")
                logger.error("=" * 60)
                logger.error("The error message suggests insufficient memory allocation.")
                logger.error("")
                logger.error("Solutions:")
                logger.error("  1. Check scheduler logs for OOM kill messages:")
                logger.error("     - SLURM: Check .err file for 'oom_kill' or 'OOM'")
                logger.error("     - Look for: 'Detected X oom_kill event'")
                logger.error("")
                logger.error("  2. Increase memory allocation in your job:")
                logger.error("     - SLURM: Use --mem=24G or higher")
                logger.error("     - SGE: Use -l mem_free=24G")
                logger.error("")
                logger.error("  3. For very large samples (>500MB), consider:")
                logger.error("     - Using --mem=32G or --mem=48G")
                logger.error("     - Processing on a high-memory node")
                logger.error("=" * 60)
        
        return {
            "sample_id": sample_id,
            "status": "failed",
            "elapsed_time": elapsed_time,
            "error": f"Pipeline failed: {error_msg}",
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

Resource Requirements:
  Memory: Processing large FASTQ files (>100MB) requires substantial RAM.
          Allocate at least 24GB for large samples. For cluster jobs, use:
          --mem=24G (SLURM) or -l mem_free=24G (SGE)
  
  Threads: Use 6-10 threads for optimal performance (--threads flag)
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
    
    # ========================================================================
    # Verify all required pipeline scripts are in PATH
    # ========================================================================
    required_scripts = [
        "00_extract_reference_kmers.sh",
        "01_count_kmers.sh",
        "02_aggregate_to_unitigs.sh",
        "03_run_inference.py",
        "04_plot_results.py"
    ]
    
    missing_scripts = []
    for script in required_scripts:
        if not shutil.which(script):
            missing_scripts.append(script)
    
    if missing_scripts:
        logger.error("Required pipeline scripts not found in PATH:")
        for script in missing_scripts:
            logger.error(f"  - {script}")
        logger.error("")
        logger.error("Please ensure DIANA is properly installed:")
        logger.error("  1. Run: pip install -e .")
        logger.error("  2. Or add scripts/inference/ to your PATH")
        sys.exit(1)
    
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
