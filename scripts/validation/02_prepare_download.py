#!/usr/bin/env python3
"""
Create Accessions File for SLURM Array Download
================================================

Extracts run accessions from metadata and creates a file for SLURM array job.
The actual download will be done on seqbio partition nodes with internet access.

USAGE:
------
# Create accessions file
python scripts/validation/03_prepare_download.py \\
    --metadata data/validation/validation_metadata_expanded.tsv \\
    --output data/validation/accessions.txt

# Then submit SLURM job:
# sbatch scripts/validation/04_download_validation.sbatch
"""

import argparse
import logging
import subprocess
from pathlib import Path
import polars as pl
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare accessions file for SLURM array download",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('--metadata', type=Path, required=True,
                       help='Expanded metadata with run accessions')
    parser.add_argument('--output', type=Path, required=True,
                       help='Output file for accessions list')
    
    args = parser.parse_args()
    
    # Load metadata and get accessions
    logger.info(f"Reading metadata from {args.metadata}")
    df = pl.read_csv(args.metadata, separator='\t')
    
    df_with_runs = df.filter(pl.col('run_accession') != '')
    run_accessions = df_with_runs['run_accession'].unique().to_list()
    
    logger.info(f"Found {len(run_accessions)} unique run accessions")
    
    # Write accessions to file (one per line)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        for acc in run_accessions:
            f.write(f"{acc}\n")
    
    logger.info(f"✓ Wrote {len(run_accessions)} accessions to {args.output}")
    logger.info(f"\nNext step:")
    logger.info(f"  sbatch --array=1-{len(run_accessions)}%20 scripts/validation/04_download_validation.sbatch")
    logger.info(f"\nThis will download and convert {len(run_accessions)} samples on seqbio partition")


if __name__ == '__main__':
    main()
