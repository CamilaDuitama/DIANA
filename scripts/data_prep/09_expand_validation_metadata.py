#!/usr/bin/env python3
"""
Expand Validation Metadata and Fetch Run Accessions
====================================================

Expands comma-separated archive accessions (SRS/ERS) to individual rows
and fetches corresponding run accessions (SRR/ERR) from ENA API.

DEPENDENCIES:
-------------
- polars
- requests

USAGE:
------
python scripts/data_prep/09_expand_validation_metadata.py \\
    --input data/validation/validation_metadata_v25.09.0.tsv \\
    --output data/validation/validation_metadata_expanded.tsv \\
    --cache data/validation/ena_cache.json

Output columns:
- All original metadata columns
- archive_accession (single, not comma-separated)
- run_accession (SRR/ERR from ENA)
- fastq_ftp (FTP path for download)
- fastq_bytes (file size)
"""

import argparse
import logging
import json
import time
from pathlib import Path
from typing import Dict, List, Optional
import polars as pl
import requests

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ENAClient:
    """Client for querying ENA API to get run accessions."""
    
    BASE_URL = "https://www.ebi.ac.uk/ena/portal/api/filereport"
    
    def __init__(self, cache_file: Optional[Path] = None):
        self.cache_file = cache_file
        self.cache = self._load_cache()
    
    def _load_cache(self) -> Dict:
        """Load cached ENA responses."""
        if self.cache_file and self.cache_file.exists():
            with open(self.cache_file) as f:
                return json.load(f)
        return {}
    
    def _save_cache(self):
        """Save cache to disk."""
        if self.cache_file:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f, indent=2)
    
    def get_run_accessions(self, sample_accession: str) -> List[Dict]:
        """
        Fetch run accessions for a sample/experiment accession.
        
        Args:
            sample_accession: SRS/ERS accession
            
        Returns:
            List of dicts with run_accession, fastq_ftp, fastq_bytes
        """
        # Check cache first
        if sample_accession in self.cache:
            logger.debug(f"Using cached data for {sample_accession}")
            return self.cache[sample_accession]
        
        # Query ENA API
        logger.info(f"Querying ENA for {sample_accession}")
        params = {
            'accession': sample_accession,
            'result': 'read_run',
            'fields': 'run_accession,fastq_ftp,fastq_bytes,fastq_md5',
            'format': 'json',
            'download': 'true'
        }
        
        try:
            response = requests.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Parse results
            runs = []
            for entry in data:
                run_acc = entry.get('run_accession')
                if run_acc:
                    runs.append({
                        'run_accession': run_acc,
                        'fastq_ftp': entry.get('fastq_ftp', ''),
                        'fastq_bytes': entry.get('fastq_bytes', ''),
                        'fastq_md5': entry.get('fastq_md5', '')
                    })
            
            # Cache results
            self.cache[sample_accession] = runs
            self._save_cache()
            
            logger.info(f"  Found {len(runs)} runs for {sample_accession}")
            
            # Rate limiting
            time.sleep(0.2)
            
            return runs
            
        except requests.RequestException as e:
            logger.error(f"Error querying ENA for {sample_accession}: {e}")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing ENA response for {sample_accession}: {e}")
            return []


def expand_metadata(
    input_path: Path,
    output_path: Path,
    cache_file: Optional[Path] = None
) -> pl.DataFrame:
    """
    Expand validation metadata with run accessions.
    
    Args:
        input_path: Path to original metadata TSV
        output_path: Path for expanded metadata TSV
        cache_file: Path to cache file for ENA responses
        
    Returns:
        Expanded DataFrame
    """
    logger.info(f"Reading metadata from {input_path}")
    df = pl.read_csv(input_path, separator='\t')
    
    logger.info(f"Original metadata: {len(df)} samples")
    
    # Initialize ENA client
    ena_client = ENAClient(cache_file)
    
    # Expand rows
    expanded_rows = []
    
    for i, row in enumerate(df.iter_rows(named=True), 1):
        sample_name = row['sample_name']
        accessions_str = row['archive_accession']
        
        # Split comma-separated accessions
        accessions = [acc.strip() for acc in accessions_str.split(',')]
        
        logger.info(f"[{i}/{len(df)}] Processing {sample_name} ({len(accessions)} accessions)")
        
        for archive_acc in accessions:
            # Get run accessions from ENA
            runs = ena_client.get_run_accessions(archive_acc)
            
            if not runs:
                logger.warning(f"  No runs found for {archive_acc}, keeping archive accession only")
                # Still add row but without run info
                expanded_row = dict(row)
                expanded_row['archive_accession'] = archive_acc
                expanded_row['run_accession'] = ''
                expanded_row['fastq_ftp'] = ''
                expanded_row['fastq_bytes'] = ''
                expanded_row['fastq_md5'] = ''
                expanded_rows.append(expanded_row)
            else:
                # Add one row per run
                for run_info in runs:
                    expanded_row = dict(row)
                    expanded_row['archive_accession'] = archive_acc
                    expanded_row['run_accession'] = run_info['run_accession']
                    expanded_row['fastq_ftp'] = run_info['fastq_ftp']
                    expanded_row['fastq_bytes'] = run_info['fastq_bytes']
                    expanded_row['fastq_md5'] = run_info['fastq_md5']
                    expanded_rows.append(expanded_row)
    
    # Create expanded DataFrame
    expanded_df = pl.DataFrame(expanded_rows)
    
    logger.info(f"Expanded metadata: {len(expanded_df)} rows")
    logger.info(f"  {expanded_df.filter(pl.col('run_accession') != '').shape[0]} rows with run accessions")
    logger.info(f"  {expanded_df.filter(pl.col('run_accession') == '').shape[0]} rows without run accessions")
    
    # Save expanded metadata
    expanded_df.write_csv(output_path, separator='\t')
    logger.info(f"Saved expanded metadata to {output_path}")
    
    return expanded_df


def main():
    parser = argparse.ArgumentParser(
        description="Expand validation metadata and fetch run accessions from ENA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--input',
        type=Path,
        required=True,
        help='Input validation metadata TSV (with comma-separated accessions)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output expanded metadata TSV'
    )
    parser.add_argument(
        '--cache',
        type=Path,
        help='Cache file for ENA API responses (speeds up re-runs)'
    )
    
    args = parser.parse_args()
    
    # Expand metadata
    expanded_df = expand_metadata(
        args.input,
        args.output,
        cache_file=args.cache
    )
    
    # Print summary
    logger.info("=" * 60)
    logger.info("Summary")
    logger.info("=" * 60)
    logger.info(f"Total rows: {len(expanded_df)}")
    logger.info(f"Unique samples: {expanded_df['sample_name'].n_unique()}")
    logger.info(f"Unique archive accessions: {expanded_df['archive_accession'].n_unique()}")
    logger.info(f"Unique run accessions: {expanded_df.filter(pl.col('run_accession') != '')['run_accession'].n_unique()}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
