#!/usr/bin/env python3
"""
Query MGnify API to find modern metagenome samples for DIANA validation.

This script searches for modern metagenome samples with specific biomes/materials
to balance the validation set (currently 100% ancient → add ~74-150 modern).

Target distribution (matching training):
- 50 modern oral (dental/saliva)  
- 40 modern gut
- 30 modern soil
- 20 modern skin
- 10 modern plant
Total: ~150 modern samples (15.4% of total validation = 974 + 150 = 1124)
"""

import requests
import polars as pl
from pathlib import Path
import time
from typing import List, Dict
import json

PROJECT_ROOT = Path("/pasteur/appa/scratch/cduitama/EDID/decOM-classify")

# MGnify API base URL
MGNIFY_API = "https://www.ebi.ac.uk/metagenomics/api/v1"

def query_mgnify_biome(biome_name: str, max_samples: int = 100) -> List[Dict]:
    """
    Query MGnify API for samples from a specific biome.
    Prioritizes diversity by selecting samples from different projects.
    
    Args:
        biome_name: Biome to search (e.g., "human-oral", "human-gut", "soil")
        max_samples: Maximum number of samples to retrieve
        
    Returns:
        List of sample metadata dictionaries
    """
    samples = []
    project_samples = {}  # Track samples per project for diversity
    url = f"{MGNIFY_API}/biomes/{biome_name}/samples"
    
    params = {
        'page_size': 100,  # Max per page
        'ordering': '-last_update'  # Most recent first
    }
    
    print(f"\nQuerying MGnify for biome: {biome_name}")
    
    while url and len(samples) < max_samples * 3:  # Get more for filtering
        print(f"  Fetching page... ({len(samples)} samples so far)")
        response = requests.get(url, params=params)
        
        if response.status_code != 200:
            print(f"  Error: {response.status_code}")
            break
            
        data = response.json()
        
        for item in data.get('data', []):
            attrs = item.get('attributes', {})
            
            # Get study/project info
            studies_url = item.get('relationships', {}).get('studies', {}).get('links', {}).get('related')
            project_id = None
            
            if studies_url:
                studies_resp = requests.get(studies_url)
                if studies_resp.status_code == 200:
                    studies_data = studies_resp.json()
                    if studies_data.get('data'):
                        project_id = studies_data['data'][0].get('id')
            
            # Get ENA run accessions
            runs_url = item.get('relationships', {}).get('runs', {}).get('links', {}).get('related')
            run_accessions = []
            
            if runs_url:
                runs_resp = requests.get(runs_url)
                if runs_resp.status_code == 200:
                    runs_data = runs_resp.json()
                    for run in runs_data.get('data', []):
                        run_acc = run.get('attributes', {}).get('accession')
                        if run_acc:
                            run_accessions.append(run_acc)
            
            if run_accessions:  # Only keep samples with run accessions
                sample_data = {
                    'mgnify_id': item.get('id'),
                    'biome': biome_name,
                    'project_id': project_id or 'unknown',
                    'sample_accession': attrs.get('accession'),
                    'sample_name': attrs.get('sample-name'),
                    'sample_desc': attrs.get('sample-desc'),
                    'environment_feature': attrs.get('environment-feature'),
                    'environment_material': attrs.get('environment-material'),
                    'geo_loc_name': attrs.get('geo-loc-name'),
                    'latitude': attrs.get('latitude'),
                    'longitude': attrs.get('longitude'),
                    'run_accessions': ','.join(run_accessions)
                }
                
                # Group by project for diversity
                if project_id not in project_samples:
                    project_samples[project_id] = []
                project_samples[project_id].append(sample_data)
                samples.append(sample_data)
                
                if len(samples) >= max_samples * 3:
                    break
        
        # Get next page
        url = data.get('links', {}).get('next')
        time.sleep(0.5)  # Rate limiting
    
    print(f"  Retrieved {len(samples)} samples from {len(project_samples)} different projects")
    
    # Select samples prioritizing project diversity (max 5 per project)
    diverse_samples = []
    max_per_project = max(5, max_samples // max(len(project_samples), 1))
    
    for project_id, proj_samples in project_samples.items():
        diverse_samples.extend(proj_samples[:max_per_project])
    
    print(f"  Selected {len(diverse_samples)} diverse samples (max {max_per_project} per project)")
    return diverse_samples[:max_samples * 2]  # Return buffer for overlap filtering


def get_existing_accessions() -> set:
    """Load all existing run accessions from train/test/validation."""
    existing = set()
    
    # Training
    train = pl.read_csv(PROJECT_ROOT / "data" / "splits" / "train_metadata.tsv", separator='\t')
    existing.update(train['Run_accession'].to_list())
    
    # Test
    test = pl.read_csv(PROJECT_ROOT / "data" / "splits" / "test_metadata.tsv", separator='\t')
    existing.update(test['Run_accession'].to_list())
    
    # Validation
    val = pl.read_csv(PROJECT_ROOT / "paper" / "metadata" / "validation_metadata.tsv", separator='\t')
    existing.update(val['run_accession'].to_list())
    
    print(f"\nLoaded {len(existing)} existing run accessions from train/test/validation")
    return existing


def query_mgnify_studies(search_term: str, max_samples: int = 100) -> List[Dict]:
    """Query MGnify studies by search term and get samples."""
    samples = []
    project_samples = {}
    
    # Search studies
    url = f"{MGNIFY_API}/studies"
    params = {
        'search': search_term,
        'page_size': 20
    }
    
    print(f"\nSearching MGnify studies for: {search_term}")
    response = requests.get(url, params=params, timeout=30)
    
    if response.status_code != 200:
        print(f"  Error searching studies: {response.status_code}")
        return []
    
    studies = response.json().get('data', [])
    print(f"  Found {len(studies)} studies")
    
    for study in studies[:10]:  # Limit to 10 studies for diversity
        study_id = study.get('id')
        study_name = study.get('attributes', {}).get('study-name', 'Unknown')
        
        # Get samples for this study
        samples_url = f"{MGNIFY_API}/studies/{study_id}/samples"
        samples_resp = requests.get(samples_url, params={'page_size': 50}, timeout=30)
        
        if samples_resp.status_code == 200:
            study_samples = samples_resp.json().get('data', [])
            print(f"  Study {study_id}: {len(study_samples)} samples")
            
            for sample in study_samples[:max_per_project]:
                attrs = sample.get('attributes', {})
                
                # Get run accessions
                runs_url = sample.get('relationships', {}).get('runs', {}).get('links', {}).get('related')
                run_accessions = []
                
                if runs_url:
                    runs_resp = requests.get(runs_url, timeout=30)
                    if runs_resp.status_code == 200:
                        for run in runs_resp.json().get('data', []):
                            run_acc = run.get('attributes', {}).get('accession')
                            if run_acc:
                                run_accessions.append(run_acc)
                
                if run_accessions:
                    samples.append({
                        'mgnify_id': sample.get('id'),
                        'study_id': study_id,
                        'study_name': study_name,
                        'sample_accession': attrs.get('accession'),
                        'sample_name': attrs.get('sample-name'),
                        'environment_feature': attrs.get('environment-feature'),
                        'environment_material': attrs.get('environment-material'),
                        'geo_loc_name': attrs.get('geo-loc-name'),
                        'run_accessions': ','.join(run_accessions)
                    })
                
                if len(samples) >= max_samples:
                    break
        
        time.sleep(0.3)
        if len(samples) >= max_samples:
            break
    
    print(f"  Total: {len(samples)} samples from {len(set(s['study_id'] for s in samples))} studies")
    return samples


max_per_project = 5  # Limit samples per project for diversity

def main():
    # Define search terms and target counts
    search_targets = {
        'human oral microbiome': 50,
        'human gut microbiome': 40,
        'soil microbiome': 30,
        'human skin microbiome': 20,
        'plant microbiome': 10
    }
    
    # Get existing accessions to avoid duplicates
    existing_accessions = get_existing_accessions()
    
    all_samples = []
    
    for search_term, target_count in search_targets.items():
        print(f"\n{'='*80}")
        print(f"Searching: {search_term} (target: {target_count} samples)")
        print(f"{'='*80}")
        
        # Query studies
        samples = query_mgnify_studies(search_term, max_samples=target_count * 2)
        
        # Expand run accessions and filter existing
        new_samples = []
        for sample in samples:
            for run_acc in sample['run_accessions'].split(','):
                if run_acc not in existing_accessions:
                    sample_copy = sample.copy()
                    sample_copy['run_accession'] = run_acc
                    new_samples.append(sample_copy)
                    existing_accessions.add(run_acc)  # Avoid duplicates within this run
                    
                    if len(new_samples) >= target_count:
                        break
            
            if len(new_samples) >= target_count:
                break
        
        print(f"  Found {len(new_samples)} NEW modern samples (not in train/test/val)")
        all_samples.extend(new_samples[:target_count])
    
    # Save results
    if all_samples:
        df = pl.DataFrame(all_samples)
        output_file = PROJECT_ROOT / "data" / "validation" / "modern_samples_mgnify.tsv"
        df.write_csv(output_file, separator='\t')
        print(f"\n{'='*80}")
        print(f"SUCCESS!")
        print(f"{'='*80}")
        print(f"Total modern samples found: {len(all_samples)}")
        print(f"Saved to: {output_file}")
        
        # Show distribution
        print("\nSample distribution by study:")
        study_counts = df.group_by('study_name').agg(pl.count('run_accession').alias('count'))
        print(study_counts.head(20))
        
        # Save just run accessions for download
        run_acc_file = PROJECT_ROOT / "data" / "validation" / "modern_samples_accessions.txt"
        with open(run_acc_file, 'w') as f:
            for acc in df['run_accession'].to_list():
                f.write(acc + '\n')
        print(f"\nRun accessions saved to: {run_acc_file}")
        print(f"Ready to download with 03_prefetch_all.sh")
    else:
        print("\n⚠️  No new modern samples found!")


if __name__ == "__main__":
    main()
