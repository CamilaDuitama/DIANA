#!/usr/bin/env python3
"""
Simple MGnify query to get modern sample accessions.
We'll expand to run accessions with ENA later (like we did for AMD samples).
"""

import requests
import polars as pl
from pathlib import Path
import time

PROJECT_ROOT = Path("/pasteur/appa/scratch/cduitama/EDID/decOM-classify")
MGNIFY_API = "https://www.ebi.ac.uk/metagenomics/api/v1"

def query_studies(search_term: str, max_studies: int = 20):
    """Get studies matching search term."""
    url = f"{MGNIFY_API}/studies"
    params = {'search': search_term, 'page_size': max_studies}
    
    print(f"\n🔍 Searching: {search_term}")
    try:
        response = requests.get(url, params=params, timeout=30)
        if response.status_code == 200:
            studies = response.json().get('data', [])
            print(f"   Found {len(studies)} studies")
            return studies
        else:
            print(f"   Error: {response.status_code}")
            return []
    except Exception as e:
        print(f"   Timeout/Error: {e}")
        return []

def get_study_samples(study_id: str, max_samples: int = 50):
    """Get sample ACCESSIONS from a study (not run accessions yet)."""
    url = f"{MGNIFY_API}/studies/{study_id}/samples"
    params = {'page_size': max_samples}
    
    try:
        response = requests.get(url, params=params, timeout=30)
        if response.status_code == 200:
            samples_data = response.json().get('data', [])
            samples = []
            for s in samples_data:
                attrs = s.get('attributes', {})
                samples.append({
                    'mgnify_sample_id': s.get('id'),
                    'sample_accession': attrs.get('accession'),  # This is ENA/SRA accession
                    'sample_name': attrs.get('sample-name', ''),
                    'study_id': study_id
                })
            return samples
        return []
    except:
        return []

def main():
    # Search terms and target counts
    # Note: MGnify doesn't have "modern" label - we filter ancient later
    searches = {
        'human oral': 50,
        'human gut': 40,
        'soil': 30,
        'human skin': 20,
        'plant': 10
    }
    
    # Load existing accessions to avoid duplicates
    train_meta = pl.read_csv(PROJECT_ROOT / "paper/metadata/train_metadata.tsv", separator='\t')
    test_meta = pl.read_csv(PROJECT_ROOT / "paper/metadata/test_metadata.tsv", separator='\t')
    val_meta = pl.read_csv(PROJECT_ROOT / "paper/metadata/validation_metadata.tsv", separator='\t')
    
    existing = set(train_meta['Run_accession'].to_list() + 
                   test_meta['Run_accession'].to_list() + 
                   val_meta['run_accession'].to_list())
    print(f"Loaded {len(existing)} existing accessions to avoid")
    
    all_samples = []
    
    for search_term, target in searches.items():
        studies = query_studies(search_term, max_studies=10)
        
        study_count = 0
        sample_count = 0
        
        for study in studies:
            if sample_count >= target:
                break
                
            study_id = study.get('id')
            study_name = study.get('attributes', {}).get('study-name', '')
            
            samples = get_study_samples(study_id, max_samples=10)
            print(f"   📚 {study_id}: {len(samples)} samples")
            
            for sample in samples[:5]:  # Max 5 per study for diversity
                sample['search_term'] = search_term
                sample['study_name'] = study_name
                all_samples.append(sample)
                sample_count += 1
                
                if sample_count >= target:
                    break
            
            study_count += 1
            time.sleep(0.5)  # Be nice to API
        
        print(f"   ✅ Collected {sample_count} samples from {study_count} studies\n")
    
    # Save results
    df = pl.DataFrame(all_samples)
    print(f"\n📊 Total: {len(df)} sample accessions from MGnify")
    print(f"\nSample distribution:")
    print(df.group_by('search_term').agg(pl.count('sample_accession').alias('count')))
    
    output_file = PROJECT_ROOT / "data/validation/modern_samples_mgnify.tsv"
    df.write_csv(output_file, separator='\t')
    print(f"\n💾 Saved to: {output_file}")
    
    # Save just accessions for ENA expansion (next step)
    accessions = df['sample_accession'].drop_nulls().unique().to_list()
    acc_file = PROJECT_ROOT / "data/validation/modern_sample_accessions.txt"
    with open(acc_file, 'w') as f:
        f.write('\n'.join(accessions))
    print(f"💾 Saved {len(accessions)} accessions to: {acc_file}")
    print("\n📝 NEXT STEP: Expand sample accessions → run accessions using ENA API")
    print("   (similar to 01_expand_metadata.py)")

if __name__ == "__main__":
    main()
