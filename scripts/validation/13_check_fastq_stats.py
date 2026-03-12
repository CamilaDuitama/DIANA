#!/usr/bin/env python3
"""
Check FASTQ file availability and get seqkit stats for validation samples.
Updates validation_metadata.tsv with sequence statistics.

Usage: python 13_check_fastq_stats.py [--threads N]
"""

import polars as pl
import subprocess
from pathlib import Path
import sys
import argparse

PROJECT_ROOT = Path("/pasteur/appa/scratch/cduitama/EDID/decOM-classify")

def check_fastq_exists(run_accession: str) -> tuple:
    """Check if FASTQ files exist for a run accession."""
    sample_dir = PROJECT_ROOT / "data/validation/raw" / run_accession
    
    if not sample_dir.exists():
        return False, None, "Directory not found"
    
    # Look for FASTQ files
    fastq_files = list(sample_dir.glob("*.fastq.gz")) + list(sample_dir.glob("*.fq.gz"))
    
    if not fastq_files:
        return False, None, "No FASTQ files found"
    
    # Check if files are empty
    empty_files = [f for f in fastq_files if f.stat().st_size == 0]
    if empty_files:
        return False, fastq_files, f"Empty files: {len(empty_files)}"
    
    return True, fastq_files, "OK"

def get_seqkit_stats(fastq_files: list, threads: int = 4) -> dict:
    """Run seqkit stats on FASTQ files."""
    try:
        # Run seqkit stats with parallel threads (timeout: 1 hour for large files)
        cmd = ["seqkit", "stats", "-T", "-j", str(threads)] + [str(f) for f in fastq_files]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        
        if result.returncode != 0:
            return None
        
        # Parse output (skip header)
        lines = result.stdout.strip().split('\n')[1:]
        
        total_seqs = 0
        total_len = 0
        min_lens = []
        max_lens = []
        
        for line in lines:
            parts = line.split('\t')
            if len(parts) >= 6:
                total_seqs += int(parts[3].replace(',', ''))
                total_len += int(parts[4].replace(',', ''))
                min_lens.append(int(parts[5].replace(',', '')))
                max_lens.append(int(parts[7].replace(',', '')))
        
        return {
            'num_seqs': total_seqs,
            'sum_len': total_len,
            'min_len': min(min_lens) if min_lens else 0,
            'avg_len': round(total_len / total_seqs) if total_seqs > 0 else 0,
            'max_len': max(max_lens) if max_lens else 0
        }
    except subprocess.TimeoutExpired:
        print(f"  ⚠️  Timeout (>1h) processing files")
        return None
    except Exception as e:
        print(f"  Error running seqkit: {e}")
        return None

def process_sample(row: dict, threads: int = 4, verbose: bool = False) -> dict:
    """Process a single sample."""
    run_acc = row['run_accession']
    
    if verbose:
        # Get file sizes for progress info
        sample_dir = PROJECT_ROOT / "data/validation/raw" / run_acc
        if sample_dir.exists():
            fastq_files_check = list(sample_dir.glob("*.fastq.gz")) + list(sample_dir.glob("*.fq.gz"))
            if fastq_files_check:
                total_size_gb = sum(f.stat().st_size for f in fastq_files_check) / 1e9
                print(f"  Processing {run_acc} ({total_size_gb:.1f} GB)...")
    
    # Check FASTQ existence
    exists, fastq_files, status = check_fastq_exists(run_acc)
    
    if not exists:
        return {
            'run_accession': run_acc,
            'fastq_available': False,
            'num_seqs': None,
            'sum_len': None,
            'min_len': None,
            'avg_len': None,
            'max_len': None,
            'status': status
        }
    
    # Get seqkit stats
    stats = get_seqkit_stats(fastq_files, threads=threads)
    
    if stats is None:
        return {
            'run_accession': run_acc,
            'fastq_available': True,
            'num_seqs': None,
            'sum_len': None,
            'min_len': None,
            'avg_len': None,
            'max_len': None,
            'status': 'seqkit failed'
        }
    else:
        return {
            'run_accession': run_acc,
            'fastq_available': True,
            **stats,
            'status': 'OK'
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--threads', type=int, default=1, help='Number of parallel threads')
    args = parser.parse_args()
    
    print(f"Using {args.threads} threads")
    
    # Load validation metadata
    metadata_file = PROJECT_ROOT / "paper/metadata/validation_metadata.tsv"
    print(f"Loading metadata from: {metadata_file}")
    
    df = pl.read_csv(metadata_file, separator='\t')
    print(f"Total samples in metadata: {len(df)}\n")
    
    # Process samples sequentially (seqkit handles parallelization internally)
    results = []
    
    for i, row in enumerate(df.iter_rows(named=True), 1):
        verbose = (i % 50 == 0 or i == len(df))
        if verbose:
            print(f"\n  Progress: {i}/{len(df)} ({i*100//len(df)}%)")
        
        result = process_sample(row, threads=args.threads, verbose=verbose)
        results.append(result)
    
    # Collect missing and errors
    missing = [(r['run_accession'], r['status']) 
               for r in results if not r['fastq_available']]
    errors = [r['run_accession'] 
              for r in results if r['fastq_available'] and r['status'] == 'seqkit failed']
    
    # Create results dataframe
    results_df = pl.DataFrame(results)
    
    # Join with original metadata
    updated_df = df.join(
        results_df.select(['run_accession', 'fastq_available', 'num_seqs', 'sum_len', 
                           'min_len', 'avg_len', 'max_len']),
        on='run_accession',
        how='left'
    )
    
    # Save updated metadata
    output_file = PROJECT_ROOT / "paper/metadata/validation_metadata.tsv"
    updated_df.write_csv(output_file, separator='\t')
    print(f"\n✅ Updated metadata saved to: {output_file}")
    
    # Save detailed results
    results_file = PROJECT_ROOT / "data/validation/fastq_check_results.tsv"
    results_df.write_csv(results_file, separator='\t')
    print(f"✅ Detailed results saved to: {results_file}")
    
    # Summary
    available = results_df.filter(pl.col('fastq_available') == True)
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total samples: {len(df)}")
    print(f"FASTQ available: {len(available)} ({len(available)*100//len(df)}%)")
    print(f"Missing/Error: {len(missing) + len(errors)}")
    
    if missing:
        print(f"\n⚠️  Missing FASTQ files ({len(missing)}):")
        for acc, reason in missing[:10]:
            print(f"  - {acc}: {reason}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")
    
    if errors:
        print(f"\n⚠️  Seqkit errors ({len(errors)}):")
        for acc in errors[:10]:
            print(f"  - {acc}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")
    
    # Basic stats
    if len(available) > 0:
        print(f"\n📊 Sequence Statistics (available samples):")
        print(f"  Total sequences: {available['num_seqs'].sum():,}")
        print(f"  Total bases: {available['sum_len'].sum():,}")
        print(f"  Avg sequences/sample: {available['num_seqs'].mean():.0f}")
        print(f"  Avg length/read: {available['avg_len'].mean():.0f} bp")

if __name__ == "__main__":
    main()
