#!/usr/bin/env python3
"""
Download Validation FASTQ Files from ENA
=========================================

Downloads sequencing data for validation samples using direct FTP download from ENA.
Uses EXPANDED metadata with individual run accessions (SRR/ERR) and FTP paths.

DEPENDENCIES:
-------------
- polars (for metadata parsing)
- wget or curl (for FTP download)

PREREQUISITE:
-------------
First run 09_expand_validation_metadata.py to create expanded metadata
with run accessions and FTP paths from ENA API.

OUTPUT STRUCTURE:
-----------------
data/validation/raw/
├── <sample_name>/
│   ├── <run_accession>_1.fastq.gz
│   ├── <run_accession>_2.fastq.gz
│   └── download_manifest.txt
├── download_success.txt (successfully downloaded runs)
├── download_failed.txt (failed downloads)
└── download_summary.json (overall statistics)

USAGE:
------
python scripts/data_prep/10_download_validation_files.py \\
    --metadata data/validation/validation_metadata_expanded.tsv \\
    --output data/validation/raw \\
    --log-dir logs/validation_download

For SLURM array job (recommended for large downloads):
sbatch scripts/data_prep/download_validation.sbatch
"""

import argparse
import logging
import subprocess
from pathlib import Path
from typing import List, Dict
import polars as pl
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)



def parse_accessions(metadata_path: Path) -> List[dict]:
    """
    Parse expanded validation metadata with run accessions.
    
    Returns:
        List of dicts with sample_name, run_accession, and metadata
    """
    logger.info(f"Reading expanded metadata from {metadata_path}")
    df = pl.read_csv(metadata_path, separator='\t')
    
    # Filter out rows without run accessions
    df_with_runs = df.filter(pl.col('run_accession') != '')
    
    if len(df_with_runs) == 0:
        raise ValueError("No run accessions found in metadata! Did you run 09_expand_validation_metadata.py first?")
    
    logger.info(f"Found {len(df_with_runs)} runs across {df_with_runs['sample_name'].n_unique()} samples")
    
    samples = []
    for row in df_with_runs.iter_rows(named=True):
        samples.append({
            'sample_name': row['sample_name'],
            'archive_accession': row['archive_accession'],
            'run_accession': row['run_accession'],
            'fastq_ftp': row.get('fastq_ftp', ''),
            'sample_type': row.get('sample_type', 'unknown'),
            'sample_host': row.get('sample_host', 'unknown'),
            'material': row.get('material', 'unknown'),
            'project_name': row.get('project_name', 'unknown')
        })
    
    logger.info(f"Total runs to download: {len(samples)}")
    return samples


def download_run(
    sample_info: dict,
    output_dir: Path
) -> dict:
    """
    Download a single run from ENA FTP using wget.
    
    Args:
        sample_info: Dictionary with sample metadata, run_accession, and fastq_ftp
        output_dir: Base directory to save downloaded files
        
    Returns:
        Dict with download status and details
    """
    sample_name = sample_info['sample_name']
    run_accession = sample_info['run_accession']
    fastq_ftp = sample_info.get('fastq_ftp', '')
    
    result = {
        'sample_name': sample_name,
        'run_accession': run_accession,
        'success': False,
        'files_downloaded': [],
        'error': None
    }
    
    if not fastq_ftp:
        result['error'] = 'No FTP path in metadata'
        logger.error(f"  ✗ {run_accession}: No FTP path available")
        return result
    
    # Create sample-specific subdirectory
    sample_dir = output_dir / sample_name
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Downloading {sample_name} - {run_accession}")
    
    # FTP paths are semicolon-separated for paired-end reads
    ftp_urls = [url.strip() for url in fastq_ftp.split(';')]
    
    downloaded_files = []
    all_success = True
    
    for ftp_url in ftp_urls:
        if not ftp_url:
            continue
            
        # Add ftp:// prefix if not present
        if not ftp_url.startswith('ftp://'):
            ftp_url = f'ftp://{ftp_url}'
        
        # Extract filename
        filename = Path(ftp_url).name
        output_file = sample_dir / filename
        
        # Skip if already exists
        if output_file.exists():
            logger.info(f"  ⊙ {filename} already exists, skipping")
            downloaded_files.append(str(output_file))
            continue
        
        try:
            # Download using wget
            cmd = [
                'wget',
                '--quiet',
                '--timeout=600',
                '--tries=3',
                '-O', str(output_file),
                ftp_url
            ]
            
            logger.info(f"  Downloading {filename}...")
            wget_result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=7200  # 2 hour timeout per file
            )
            
            if wget_result.returncode == 0 and output_file.exists():
                logger.info(f"  ✓ {filename} downloaded successfully")
                downloaded_files.append(str(output_file))
            else:
                logger.error(f"  ✗ {filename} failed: {wget_result.stderr}")
                all_success = False
                result['error'] = wget_result.stderr
                # Clean up partial download
                if output_file.exists():
                    output_file.unlink()
                    
        except subprocess.TimeoutExpired:
            logger.error(f"  ✗ {filename} timed out after 2 hours")
            all_success = False
            result['error'] = 'Download timeout'
            if output_file.exists():
                output_file.unlink()
        except Exception as e:
            logger.error(f"  ✗ {filename} error: {e}")
            all_success = False
            result['error'] = str(e)
            if output_file.exists():
                output_file.unlink()
    
    result['success'] = all_success and len(downloaded_files) > 0
    result['files_downloaded'] = downloaded_files
    
    # Create manifest file for this run
    if downloaded_files:
        manifest_file = sample_dir / f'{run_accession}_manifest.txt'
        with open(manifest_file, 'w') as f:
            f.write(f"run_accession: {run_accession}\n")
            f.write(f"sample_name: {sample_name}\n")
            f.write(f"files:\n")
            for fpath in downloaded_files:
                f.write(f"  - {fpath}\n")
    
    return result


def download_all_samples(
    samples: List[dict],
    output_dir: Path,
    start_idx: int = 0,
    end_idx: int = None
) -> dict:
    """
    Download multiple runs and track results.
    
    Args:
        samples: List of sample info dicts (one per run)
        output_dir: Output directory
        start_idx: Start index (for array jobs)
        end_idx: End index (for array jobs)
        
    Returns:
        Dictionary with download statistics and lists of successes/failures
    """
    if end_idx is None:
        end_idx = len(samples)
    
    subset = samples[start_idx:end_idx]
    logger.info(f"Processing runs {start_idx} to {end_idx} ({len(subset)} runs)")
    
    stats = {
        'total': len(subset),
        'successful': 0,
        'failed': 0,
        'successful_runs': [],
        'failed_runs': []
    }
    
    for i, sample_info in enumerate(subset, start=start_idx):
        logger.info(f"[{i+1}/{end_idx}] Processing {sample_info['sample_name']} - {sample_info['run_accession']}")
        
        result = download_run(sample_info, output_dir)
        
        if result['success']:
            stats['successful'] += 1
            stats['successful_runs'].append({
                'sample_name': result['sample_name'],
                'run_accession': result['run_accession'],
                'files': result['files_downloaded']
            })
        else:
            stats['failed'] += 1
            stats['failed_runs'].append({
                'sample_name': result['sample_name'],
                'run_accession': result['run_accession'],
                'error': result.get('error', 'Unknown error')
            })
    
    # Save success and failure lists
    success_file = output_dir / f'download_success_{start_idx}_{end_idx}.txt'
    with open(success_file, 'w') as f:
        for item in stats['successful_runs']:
            f.write(f"{item['sample_name']}\t{item['run_accession']}\n")
    
    failed_file = output_dir / f'download_failed_{start_idx}_{end_idx}.txt'
    with open(failed_file, 'w') as f:
        for item in stats['failed_runs']:
            f.write(f"{item['sample_name']}\t{item['run_accession']}\t{item['error']}\n")
    
    logger.info(f"Saved success list to {success_file}")
    logger.info(f"Saved failure list to {failed_file}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Download validation samples using seqdd",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--metadata',
        type=Path,
        required=True,
        help='Path to validation metadata TSV file'
    )
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output directory for downloaded files'
    )
    parser.add_argument(
        '--start-idx',
        type=int,
        default=0,
        help='Start index for array jobs (default: 0)'
    )
    parser.add_argument(
        '--end-idx',
        type=int,
        default=None,
        help='End index for array jobs (default: all)'
    )
    parser.add_argument(
        '--log-dir',
        type=Path,
        help='Directory for log files'
    )
    
    args = parser.parse_args()
    
    # Setup file logging if requested
    if args.log_dir:
        args.log_dir.mkdir(parents=True, exist_ok=True)
        log_file = args.log_dir / f'download_{args.start_idx}_{args.end_idx or "end"}.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        logger.addHandler(file_handler)
        logger.info(f"Logging to {log_file}")
    
    # Check wget availability
    try:
        subprocess.run(['wget', '--version'], capture_output=True, check=True)
        logger.info("wget found ✓")
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("wget not found! Please install wget.")
        sys.exit(1)
    
    # Parse metadata
    samples = parse_accessions(args.metadata)
    
    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    
    # Download samples
    logger.info("Starting downloads...")
    stats = download_all_samples(
        samples,
        args.output,
        start_idx=args.start_idx,
        end_idx=args.end_idx
    )
    
    # Print summary
    logger.info("=" * 60)
    logger.info("Download Summary")
    logger.info("=" * 60)
    logger.info(f"Total runs: {stats['total']}")
    logger.info(f"Successful: {stats['successful']}")
    logger.info(f"Failed: {stats['failed']}")
    logger.info(f"Success rate: {stats['successful']/stats['total']*100:.1f}%")
    logger.info(f"")
    logger.info(f"Results saved to:")
    logger.info(f"  Success: {args.output}/download_success_*.txt")
    logger.info(f"  Failed: {args.output}/download_failed_*.txt")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
