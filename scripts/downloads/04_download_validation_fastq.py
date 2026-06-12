#!/usr/bin/env python3
"""
Download missing FASTQ files for validation set using SRA toolkit.

Identifies missing FASTQ files needed for v7 validation set and generates
download scripts using prefetch + fasterq-dump.

Usage:
    python scripts/downloads/04_download_validation_fastq.py
    
    # Then submit SLURM array job
    sbatch --array=1-N%20 scripts/downloads/05_download_validation_fastq.sbatch
"""

import argparse
from pathlib import Path


def load_validation_accessions(split_dir: Path) -> set:
    """Load validation accessions from v7 splits."""
    accessions = set()
    
    val_file = split_dir / 'val_accessions.txt'
    if val_file.exists():
        with open(val_file) as f:
            accessions.update(line.strip() for line in f if line.strip())
    
    return accessions


def find_existing_fastq(fastq_dir: Path) -> set:
    """Find existing FASTQ files (check both .fastq and .fastq.gz)."""
    existing = set()
    
    if fastq_dir.exists():
        # Each accession should have a directory with FASTQ files
        for acc_dir in fastq_dir.iterdir():
            if acc_dir.is_dir():
                # Check if directory has any FASTQ files
                has_fastq = any(
                    f.suffix in ['.fastq', '.gz'] 
                    for f in acc_dir.glob('*.fastq*')
                )
                if has_fastq:
                    existing.add(acc_dir.name)
    
    return existing


def main():
    parser = argparse.ArgumentParser(description='Generate download list for missing validation FASTQ files')
    parser.add_argument('--splits', type=Path, default=Path('data/splits_v7'),
                        help='Directory containing v7 split files')
    parser.add_argument('--fastq', type=Path, default=Path('data/validation/raw'),
                        help='Directory containing existing FASTQ files')
    parser.add_argument('--output', type=Path, default=Path('scripts/downloads/missing_validation_accessions.txt'),
                        help='Output list of missing accessions')
    args = parser.parse_args()
    
    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    # Load validation accessions
    print("Loading accessions from v7 validation split...")
    required = load_validation_accessions(args.splits)
    print(f"  Required: {len(required)} accessions")
    
    # Find existing FASTQ files
    print("\nChecking existing FASTQ files...")
    existing = find_existing_fastq(args.fastq)
    print(f"  Existing: {len(existing)} FASTQ directories")
    
    # Identify missing
    missing = sorted(required - existing)
    print(f"  Missing:  {len(missing)} FASTQ files")
    
    if not missing:
        print("\n✓ All validation FASTQ files are available!")
        return
    
    # Write missing accessions list
    with open(args.output, 'w') as f:
        f.write('\n'.join(missing) + '\n')
    print(f"\n✓ Wrote missing accessions to {args.output}")
    
    print(f"\nTo download using SLURM array job:")
    print(f"  sbatch --array=1-{len(missing)}%20 scripts/downloads/05_download_validation_fastq.sbatch")
    
    print(f"\nNote: Each job will:")
    print(f"  1. prefetch the SRA file to data/validation/sra/")
    print(f"  2. Convert to FASTQ using fasterq-dump")
    print(f"  3. Compress with pigz (parallel gzip)")
    print(f"  4. Save to data/validation/raw/{{accession}}/")


if __name__ == '__main__':
    main()
