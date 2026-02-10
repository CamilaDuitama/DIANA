#!/usr/bin/env python3
"""
Expand Modern Sample Metadata with SRA Information
===================================================

Takes modern_samples_balanced.tsv and enriches it with SRA metadata
to match the column structure of validation_metadata.tsv for merging.

USAGE:
------
python scripts/validation/12_expand_modern_metadata.py

This creates data/validation/modern_samples_expanded_full.tsv with columns:
- archive_accession (sample_accession)
- sample_name (from MGnify or SRA)
- sample_type: 'modern_metagenome'
- sample_source: 'unknown' (will be filled from search_term)
- sample_host: from search_term
- material: 'unknown'
- community_type: 'unknown'
- geo_loc_name: from SRA
- site_name: 'unknown'
- latitude: 'unknown'
- longitude: 'unknown'
- sample_age: 0
- sample_age_doi: 'Not applicable - modern sample'
- project_name: from study_name
- publication_year: 'unknown'
- publication_doi: 'unknown'
- run_accession
- fastq_ftp
- fastq_bytes
- fastq_md5
"""

import polars as pl
import requests
import json
import time
from pathlib import Path

PROJECT_ROOT = Path("/pasteur/appa/scratch/cduitama/EDID/decOM-classify")

# ENA API
ENA_API = "https://www.ebi.ac.uk/ena/portal/api/filereport"

# Load cache
cache_file = PROJECT_ROOT / "data/validation/ena_cache.json"
if cache_file.exists():
    with open(cache_file, 'r') as f:
        ena_cache = json.load(f)
    print(f"📚 Loaded {len(ena_cache)} cached ENA queries")
else:
    ena_cache = {}

def query_ena_sample_metadata(run_accession: str) -> dict:
    """Query ENA API to get full sample metadata for a run."""
    
    # Check cache first
    cache_key = f"run_{run_accession}"
    if cache_key in ena_cache:
        return ena_cache[cache_key]
    
    params = {
        'accession': run_accession,
        'result': 'read_run',
        'fields': 'run_accession,sample_accession,fastq_ftp,fastq_bytes,fastq_md5,country,location,collection_date,lat,lon',
        'format': 'json',
        'limit': 0
    }
    
    try:
        response = requests.get(ENA_API, params=params, timeout=30)
        if response.status_code == 200:
            data = response.json()
            if data:
                result = data[0]
                ena_cache[cache_key] = result
                return result
        print(f"  ⚠️  ENA error {response.status_code} for {run_accession}")
        ena_cache[cache_key] = {}
        return {}
    except Exception as e:
        print(f"  ⚠️  ENA error for {run_accession}: {e}")
        ena_cache[cache_key] = {}
        return {}

def infer_sample_source(search_term: str) -> str:
    """Infer sample_source from search_term."""
    if not search_term or search_term == "":
        return "unknown"
    
    term_lower = search_term.lower()
    
    # Host-associated terms
    if any(x in term_lower for x in ['human', 'oral', 'gut', 'skin', 'vaginal', 'fecal']):
        return "host_associated"
    
    # Environmental terms
    if any(x in term_lower for x in ['soil', 'water', 'marine', 'freshwater', 'sediment', 'environmental']):
        return "environmental"
    
    return "unknown"

def infer_sample_host(search_term: str) -> str:
    """Infer sample_host from search_term."""
    if not search_term or search_term == "":
        return "unknown"
    
    term_lower = search_term.lower()
    
    if 'human' in term_lower:
        return "Homo sapiens"
    
    # Environmental
    if any(x in term_lower for x in ['soil', 'water', 'marine', 'freshwater', 'sediment', 'environmental']):
        return "Not applicable - env sample"
    
    return "unknown"

def infer_material(search_term: str, sample_host: str) -> str:
    """Infer material from search_term and sample_host."""
    if not search_term or search_term == "":
        return "unknown"
    
    term_lower = search_term.lower()
    
    if 'oral' in term_lower:
        return "oral"
    if 'gut' in term_lower or 'fecal' in term_lower:
        return "gut"
    if 'skin' in term_lower:
        return "skin"
    if 'vaginal' in term_lower:
        return "vaginal"
    if 'soil' in term_lower:
        return "soil"
    if 'water' in term_lower:
        return "water"
    if 'sediment' in term_lower:
        return "sediment"
    
    return "unknown"

def main():
    # Load modern samples
    modern_df = pl.read_csv(
        PROJECT_ROOT / "data/validation/modern_samples_balanced.tsv",
        separator='\t'
    )
    print(f"📚 Loaded {len(modern_df)} modern samples")
    
    # Expand each run with SRA metadata
    expanded_rows = []
    
    for i, row in enumerate(modern_df.iter_rows(named=True), 1):
        if i % 20 == 0:
            print(f"  Progress: {i}/{len(modern_df)} ({i*100//len(modern_df)}%)")
        
        run_acc = row['run_accession']
        
        # Query ENA for full metadata
        ena_data = query_ena_sample_metadata(run_acc)
        
        # Infer metadata from search term
        search_term = row.get('search_term', '')
        sample_source = infer_sample_source(search_term)
        sample_host = infer_sample_host(search_term)
        material = infer_material(search_term, sample_host)
        
        # Build expanded row
        expanded_row = {
            'archive_accession': row.get('sample_accession', ''),
            'sample_name': row.get('sample_name', ''),
            'sample_type': 'modern_metagenome',
            'sample_source': sample_source,
            'sample_host': sample_host,
            'material': material,
            'community_type': search_term if search_term else 'unknown',
            'geo_loc_name': ena_data.get('country', 'unknown'),
            'site_name': ena_data.get('location', 'unknown'),
            'latitude': ena_data.get('lat', 'unknown'),
            'longitude': ena_data.get('lon', 'unknown'),
            'sample_age': 0,
            'sample_age_doi': 'Not applicable - modern sample',
            'project_name': row.get('study_id', ''),
            'publication_year': 'unknown',
            'publication_doi': 'unknown',
            'run_accession': run_acc,
            'fastq_ftp': ena_data.get('fastq_ftp', ''),
            'fastq_bytes': ena_data.get('fastq_bytes', ''),
            'fastq_md5': ena_data.get('fastq_md5', '')
        }
        
        expanded_rows.append(expanded_row)
        time.sleep(0.2)  # Rate limiting
    
    # Save cache
    with open(cache_file, 'w') as f:
        json.dump(ena_cache, f)
    print(f"\n💾 Saved ENA cache ({len(ena_cache)} entries)")
    
    # Create dataframe
    expanded_df = pl.DataFrame(expanded_rows)
    
    # Save
    output_file = PROJECT_ROOT / "data/validation/modern_samples_expanded_full.tsv"
    expanded_df.write_csv(output_file, separator='\t')
    print(f"\n✅ Saved expanded metadata to {output_file}")
    print(f"   Total samples: {len(expanded_df)}")
    
    # Print summary
    print("\n📊 Summary:")
    print(f"   Sample sources: {expanded_df['sample_source'].value_counts().to_dict()}")
    print(f"   Sample hosts: {expanded_df['sample_host'].value_counts().to_dict()}")
    print(f"   Materials: {expanded_df['material'].value_counts().to_dict()}")

if __name__ == '__main__':
    main()
