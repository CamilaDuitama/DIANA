#!/usr/bin/env python3
"""
Download Validation Files Using seqdd Register Workflow
========================================================

Uses seqdd's register-based workflow for reproducible data download.
Creates a .reg file that can be shared and reused.

INSTALLATION:
-------------
git clone https://github.com/yoann-dufresne/seqdd.git external/seqdd
cd external/seqdd
pip install .

WORKFLOW:
---------
1. Create register from expanded metadata
2. Download data using seqdd (parallel)
3. Export .reg file for reproducibility

USAGE:
------
# Create register from metadata
python scripts/data_prep/10_download_validation_seqdd.py create-register \\
    --metadata data/validation/validation_metadata_expanded.tsv \\
    --register-dir data/validation/.register

# Download all data (uses seqdd internally)
python scripts/data_prep/10_download_validation_seqdd.py download \\
    --register-dir data/validation/.register \\
    --output data/validation/raw \\
    --max-processes 4

# Export register file for sharing
python scripts/data_prep/10_download_validation_seqdd.py export \\
    --register-dir data/validation/.register \\
    --output data/validation/validation.reg
"""

import argparse
import logging
import subprocess
from pathlib import Path
import polars as pl
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_seqdd_installed() -> bool:
    """Check if seqdd is installed."""
    try:
        result = subprocess.run(['seqdd', '--help'], capture_output=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False


def create_register(metadata_path: Path, register_dir: Path):
    """
    Create seqdd register from expanded metadata.
    
    Args:
        metadata_path: Path to expanded metadata with run accessions
        register_dir: Directory for seqdd register
    """
    logger.info(f"Reading metadata from {metadata_path}")
    df = pl.read_csv(metadata_path, separator='\t')
    
    # Filter rows with run accessions
    df_with_runs = df.filter(pl.col('run_accession') != '')
    
    if len(df_with_runs) == 0:
        raise ValueError("No run accessions found! Run 09_expand_validation_metadata.py first")
    
    logger.info(f"Found {len(df_with_runs)} runs")
    
    # Get unique run accessions
    run_accessions = df_with_runs['run_accession'].unique().to_list()
    logger.info(f"Unique run accessions: {len(run_accessions)}")
    
    # Initialize seqdd register
    logger.info(f"Initializing seqdd register in {register_dir}")
    cmd = ['seqdd', '--register-location', str(register_dir), 'init', '--force']
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        logger.error(f"Failed to initialize register: {result.stderr}")
        sys.exit(1)
    
    logger.info("✓ Register initialized")
    
    # Add accessions to register in batches (seqdd can be slow with many accessions)
    batch_size = 100
    total_batches = (len(run_accessions) + batch_size - 1) // batch_size
    
    for i in range(0, len(run_accessions), batch_size):
        batch = run_accessions[i:i+batch_size]
        batch_num = i // batch_size + 1
        
        logger.info(f"Adding batch {batch_num}/{total_batches} ({len(batch)} accessions)")
        
        cmd = [
            'seqdd',
            '--register-location', str(register_dir),
            'add',
            '--source', 'sra',
            '--accessions'
        ] + batch
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.warning(f"Batch {batch_num} had issues: {result.stderr}")
        else:
            logger.info(f"✓ Batch {batch_num} added successfully")
    
    logger.info(f"✓ Register created with {len(run_accessions)} accessions")


def download_data(register_dir: Path, output_dir: Path, max_processes: int = 4):
    """
    Download data using seqdd.
    
    Args:
        register_dir: Directory with seqdd register
        output_dir: Output directory for downloaded files
        max_processes: Number of parallel downloads
    """
    logger.info(f"Downloading data to {output_dir}")
    logger.info(f"Using {max_processes} parallel processes")
    
    cmd = [
        'seqdd',
        '--register-location', str(register_dir),
        'download',
        '--download-directory', str(output_dir),
        '--max-processes', str(max_processes)
    ]
    
    logger.info(f"Running: {' '.join(cmd)}")
    
    # Run with live output
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    for line in process.stdout:
        print(line, end='')
    
    process.wait()
    
    if process.returncode == 0:
        logger.info("✓ Download completed successfully")
    else:
        logger.error(f"Download failed with exit code {process.returncode}")
        sys.exit(1)


def export_register(register_dir: Path, output_file: Path):
    """
    Export register to .reg file for reproducibility.
    
    Args:
        register_dir: Directory with seqdd register
        output_file: Output .reg file
    """
    logger.info(f"Exporting register to {output_file}")
    
    cmd = [
        'seqdd',
        '--register-location', str(register_dir),
        'export',
        '--output-register', str(output_file)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        logger.error(f"Failed to export register: {result.stderr}")
        sys.exit(1)
    
    logger.info(f"✓ Register exported to {output_file}")
    logger.info("This file can be shared for reproducible downloads")


def main():
    parser = argparse.ArgumentParser(
        description="Download validation data using seqdd",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # create-register subcommand
    create_parser = subparsers.add_parser('create-register', help='Create seqdd register')
    create_parser.add_argument('--metadata', type=Path, required=True,
                               help='Expanded metadata with run accessions')
    create_parser.add_argument('--register-dir', type=Path, required=True,
                               help='Directory for seqdd register')
    
    # download subcommand
    download_parser = subparsers.add_parser('download', help='Download data')
    download_parser.add_argument('--register-dir', type=Path, required=True,
                                 help='Directory with seqdd register')
    download_parser.add_argument('--output', type=Path, required=True,
                                 help='Output directory for downloads')
    download_parser.add_argument('--max-processes', type=int, default=4,
                                 help='Number of parallel downloads (default: 4)')
    
    # export subcommand
    export_parser = subparsers.add_parser('export', help='Export register file')
    export_parser.add_argument('--register-dir', type=Path, required=True,
                               help='Directory with seqdd register')
    export_parser.add_argument('--output', type=Path, required=True,
                               help='Output .reg file')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Check seqdd installation
    if not check_seqdd_installed():
        logger.error("seqdd not found!")
        logger.error("Install from: https://github.com/yoann-dufresne/seqdd")
        logger.error("  cd external/")
        logger.error("  git clone https://github.com/yoann-dufresne/seqdd.git")
        logger.error("  cd seqdd")
        logger.error("  pip install .")
        sys.exit(1)
    
    logger.info("seqdd found ✓")
    
    # Execute command
    if args.command == 'create-register':
        create_register(args.metadata, args.register_dir)
    elif args.command == 'download':
        download_data(args.register_dir, args.output, args.max_processes)
    elif args.command == 'export':
        export_register(args.register_dir, args.output)


if __name__ == '__main__':
    main()
