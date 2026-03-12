#!/usr/bin/env python3
"""
Expand MGnify sample accessions to ENA run accessions.
Similar to 01_expand_metadata.py but for modern samples.
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
cache_file = PROJECT_ROOT / "data/validation/ena_cache.json"
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
        'fields': 'run_accession,sample_accession,fastq_ftp,submitted_ftp',
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
        print(f"  ⚠️  ENA timeout for {archive_accession}: {e}")
        ena_cache[archive_accession] = []
        return []

def main():
    # Load MGnify results
    mgnify_df = pl.read_csv(
        PROJECT_ROOT / "data/validation/modern_samples_mgnify.tsv",
        separator='\t'
    )
    print(f"📚 Loaded {len(mgnify_df)} MGnify samples")
    
    # Get unique sample accessions
    sample_accessions = mgnify_df['sample_accession'].drop_nulls().unique().to_list()
    print(f"🔍 {len(sample_accessions)} unique sample accessions to expand\n")
    
    # Expand to run accessions
    expanded_rows = []
    
    for i, sample_acc in enumerate(sample_accessions, 1):
        if i % 20 == 0:
            print(f"  Progress: {i}/{len(sample_accessions)} ({i*100//len(sample_accessions)}%)")
        
        runs = query_ena_for_runs(sample_acc)
        
        for run in runs:
            run_acc = run.get('run_accession')
            if run_acc:
                expanded_rows.append({
                    'sample_accession': sample_acc,
                    'run_accession': run_acc,
                    'fastq_ftp': run.get('fastq_ftp', ''),
                    'submitted_ftp': run.get('submitted_ftp', '')
                })
        
        time.sleep(0.2)  # Rate limiting
    
    # Save cache
    with open(cache_file, 'w') as f:
        json.dump(ena_cache, f)
    print(f"\n💾 Saved ENA cache ({len(ena_cache)} entries)")
    
    # Create dataframe
    expanded_df = pl.DataFrame(expanded_rows)
    print(f"\n📊 Expanded to {len(expanded_df)} run accessions")
    
    # Join with MGnify metadata
    final_df = expanded_df.join(
        mgnify_df,
        on='sample_accession',
        how='left'
    )
    
    # Remove duplicates
    final_df = final_df.unique(subset=['run_accession'])
    print(f"📊 After deduplication: {len(final_df)} unique run accessions")
    
    # Filter out existing train/test/validation
    train = pl.read_csv(PROJECT_ROOT / "paper/metadata/train_metadata.tsv", separator='\t')
    test = pl.read_csv(PROJECT_ROOT / "paper/metadata/test_metadata.tsv", separator='\t')
    val = pl.read_csv(PROJECT_ROOT / "paper/metadata/validation_metadata.tsv", separator='\t')
    
    existing = set(
        train['Run_accession'].to_list() + 
        test['Run_accession'].to_list() + 
        val['run_accession'].to_list()
    )
    print(f"🔍 Filtering against {len(existing)} existing accessions...")
    
    final_df = final_df.filter(~pl.col('run_accession').is_in(existing))
    print(f"✅ {len(final_df)} NEW modern samples (not in train/test/validation)")
    
    # Save results
    if len(final_df) > 0:
        output_file = PROJECT_ROOT / "data/validation/modern_samples_expanded.tsv"
        final_df.write_csv(output_file, separator='\t')
        print(f"\n💾 Saved to: {output_file}")
        
        # Save accessions for download
        acc_file = PROJECT_ROOT / "data/validation/modern_accessions_to_download.txt"
        with open(acc_file, 'w') as f:
            f.write('\n'.join(final_df['run_accession'].to_list()))
        print(f"💾 Saved {len(final_df)} accessions to: {acc_file}")
        
        # Show distribution
        print(f"\n📊 Distribution by search term:")
        dist = final_df.group_by('search_term').agg(pl.count('run_accession').alias('count'))
        print(dist.sort('count', descending=True))
        
        print(f"\n📝 NEXT STEP: Download with prefetch")
        print(f"   Update accessions.txt or use modern_accessions_to_download.txt")
    else:
        print("\n⚠️  All samples already exist in train/test/validation!")

if __name__ == "__main__":
    main()
