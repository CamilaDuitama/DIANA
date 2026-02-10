#!/usr/bin/env python3
"""
Expand MGnify sample accessions to ENA run accessions (v2).
Similar to 01_expand_metadata.py but for modern samples.
Uses the improved query output with categories.
"""

import polars as pl
import requests
from pathlib import Path
import time
import json

PROJECT_ROOT = Path("/pasteur/appa/scratch/cduitama/EDID/decOM-classify")

# ENA API
ENA_API = "https://www.ebi.ac.uk/ena/portal/api/filereport"

# Load ENA cache if exists
cache_file = PROJECT_ROOT / "data/validation/ena_cache_modern_v2.json"
if cache_file.exists():
    with open(cache_file, 'r') as f:
        ena_cache = json.load(f)
    print(f"Loaded {len(ena_cache)} cached ENA queries")
else:
    ena_cache = {}

def query_ena_for_runs(archive_accession: str) -> list:
    """Query ENA API to get run accessions for a sample/study accession."""
    
    # Check cache first
    if archive_accession in ena_cache:
        return ena_cache[archive_accession]
    
    params = {
        'accession': archive_accession,
        'result': 'read_run',
        'fields': 'run_accession,sample_accession,library_strategy,library_source,fastq_ftp,submitted_ftp',
        'format': 'json',
        'limit': 0
    }
    
    try:
        response = requests.get(ENA_API, params=params, timeout=30)
        if response.status_code == 200:
            data = response.json()
            ena_cache[archive_accession] = data
            return data
        else:
            print(f"  ⚠️  ENA error {response.status_code} for {archive_accession}")
            ena_cache[archive_accession] = []
            return []
    except Exception as e:
        print(f"  ⚠️  Exception for {archive_accession}: {e}")
        ena_cache[archive_accession] = []
        return []

def main():
    # Load MGnify samples
    input_file = PROJECT_ROOT / "data/validation/modern_samples_mgnify_v2.tsv"
    df = pl.read_csv(input_file, separator='\t')
    
    print(f"📚 Loaded {len(df)} MGnify sample accessions")
    print(f"\nCategory distribution:")
    print(df.group_by('category').agg(pl.count('sample_accession').alias('count')))
    
    # Get unique sample accessions
    unique_accessions = df['sample_accession'].drop_nulls().unique().to_list()
    print(f"\n🔍 Querying ENA for {len(unique_accessions)} unique sample accessions...")
    
    expanded_rows = []
    
    for i, sample_acc in enumerate(unique_accessions, 1):
        if i % 10 == 0:
            print(f"  Progress: {i}/{len(unique_accessions)} ({i/len(unique_accessions)*100:.1f}%)")
        
        # Get original metadata
        original = df.filter(pl.col('sample_accession') == sample_acc).to_dicts()[0]
        
        # Query ENA
        runs = query_ena_for_runs(sample_acc)
        
        if runs:
            for run in runs:
                # Filter for shotgun metagenomics only (WGS + METAGENOMIC)
                # Excludes: AMPLICON (16S/ITS), RNA-Seq (metatranscriptomics), other strategies
                lib_strategy = run.get('library_strategy', '').upper()
                lib_source = run.get('library_source', '').upper()
                
                # Only accept WGS shotgun metagenomics
                # ENA uses library_strategy='WGS' + library_source='METAGENOMIC' for metagenomics
                if lib_strategy == 'WGS' and lib_source == 'METAGENOMIC':
                    expanded_rows.append({
                        **original,
                        'run_accession': run.get('run_accession'),
                        'library_strategy': lib_strategy,
                        'library_source': lib_source,
                        'fastq_ftp': run.get('fastq_ftp', ''),
                        'submitted_ftp': run.get('submitted_ftp', '')
                    })
        
        # Save cache periodically
        if i % 50 == 0:
            with open(cache_file, 'w') as f:
                json.dump(ena_cache, f)
        
        time.sleep(0.2)  # Rate limiting
    
    # Save final cache
    with open(cache_file, 'w') as f:
        json.dump(ena_cache, f)
    print(f"\n💾 Saved ENA cache: {len(ena_cache)} queries")
    
    # Create DataFrame
    expanded_df = pl.DataFrame(expanded_rows)
    
    print(f"\n📊 Expansion results:")
    print(f"  Input samples: {len(unique_accessions)}")
    print(f"  Output runs: {len(expanded_df)}")
    print(f"  Runs per sample: {len(expanded_df) / len(unique_accessions):.2f}")
    
    print(f"\nRuns by category:")
    print(expanded_df.group_by('category').agg(pl.count('run_accession').alias('count')))
    
    print(f"\nLibrary strategies:")
    print(expanded_df.group_by('library_strategy').agg(pl.count('run_accession').alias('count')))
    
    # Save
    output_file = PROJECT_ROOT / "data/validation/modern_samples_expanded_v2.tsv"
    expanded_df.write_csv(output_file, separator='\t')
    print(f"\n💾 Saved to: {output_file}")
    
    # Also save just run accessions
    run_acc_file = PROJECT_ROOT / "data/validation/modern_run_accessions_v2.txt"
    with open(run_acc_file, 'w') as f:
        f.write('\n'.join(expanded_df['run_accession'].to_list()))
    print(f"💾 Saved {len(expanded_df)} run accessions to: {run_acc_file}")
    
    print("\n" + "="*80)
    print("📝 NEXT STEP:")
    print("="*80)
    print("Balance samples to match training distribution:")
    print("  mamba run -p ./env python scripts/validation/12_balance_modern_samples_v2.py")

if __name__ == "__main__":
    main()
