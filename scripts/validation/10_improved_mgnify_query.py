#!/usr/bin/env python3
"""
Improved MGnify query to get modern sample accessions.
Focus on matching training distribution and excluding problematic samples.

Training modern distribution:
  - Skin: 34.7% (69/199)
  - Oral: 33.7% (67/199)
  - soil: 28.1% (56/199)
  - sediment: 3.5% (7/199)

Target ~200 modern samples (accounting for ~30% QC failure rate)
"""

import requests
import polars as pl
from pathlib import Path
import time

PROJECT_ROOT = Path("/pasteur/appa/scratch/cduitama/EDID/decOM-classify")
MGNIFY_API = "https://www.ebi.ac.uk/metagenomics/api/v1"

# Excluded keywords indicating problematic studies
EXCLUDED_KEYWORDS = [
    'pooled', 'multiplexed', 'multiplex', 'mock', 'synthetic',
    '16s', 'amplicon', 'rrna', 'its', 'marker gene',
    'rhizosphere', 'phytometer', 'root-associated'
]

def is_excluded_study(study_name: str, study_desc: str = "") -> bool:
    """Check if study should be excluded based on keywords."""
    combined = (study_name + " " + study_desc).lower()
    return any(keyword in combined for keyword in EXCLUDED_KEYWORDS)

def query_studies(search_term: str, max_studies: int = 30):
    """Get studies matching search term."""
    url = f"{MGNIFY_API}/studies"
    params = {'search': search_term, 'page_size': max_studies}
    
    print(f"\n🔍 Searching: '{search_term}'")
    try:
        response = requests.get(url, params=params, timeout=30)
        if response.status_code == 200:
            all_studies = response.json().get('data', [])
            
            # Filter out excluded studies
            studies = []
            for study in all_studies:
                attrs = study.get('attributes', {})
                study_name = attrs.get('study-name', '')
                study_desc = attrs.get('study-abstract', '')
                
                if not is_excluded_study(study_name, study_desc):
                    studies.append(study)
                else:
                    print(f"   ⊗ Excluded: {study.get('id')} ({study_name[:50]}...)")
            
            print(f"   Found {len(studies)} suitable studies (filtered from {len(all_studies)})")
            return studies
        else:
            print(f"   Error: {response.status_code}")
            return []
    except Exception as e:
        print(f"   Timeout/Error: {e}")
        return []

def get_study_samples(study_id: str, max_samples: int = 100):
    """Get sample ACCESSIONS from a study."""
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
                    'sample_accession': attrs.get('accession'),
                    'sample_name': attrs.get('sample-name', ''),
                    'study_id': study_id
                })
            return samples
        return []
    except:
        return []

def main():
    # Improved search terms matching training distribution
    # Target ~200 samples (accounting for ~30% QC failure → ~140 final)
    # Distribution: Skin 35%, Oral 34%, soil 28%, sediment 3%
    
    searches = {
        # ORAL samples (target 70, ~34%)
        'oral': [
            ('saliva metagenome', 20),
            ('oral microbiome', 20),
            ('tongue microbiome', 15),
            ('buccal microbiome', 10),
            ('oral cavity', 5)
        ],
        
        # SKIN samples (target 70, ~35%)
        'skin': [
            ('skin microbiome', 25),
            ('dermal microbiome', 20),
            ('cutaneous microbiome', 15),
            ('skin metagenome', 10)
        ],
        
        # SOIL samples (target 55, ~28%)
        'soil': [
            ('soil metagenome', 20),
            ('agricultural soil', 15),
            ('forest soil', 10),
            ('grassland soil', 10)
        ],
        
        # SEDIMENT samples (target 5, ~3%)
        'sediment': [
            ('marine sediment', 3),
            ('lake sediment', 2)
        ]
    }
    
    # Load existing accessions to avoid duplicates
    train_meta = pl.read_csv(PROJECT_ROOT / "paper/metadata/train_metadata.tsv", separator='\t')
    test_meta = pl.read_csv(PROJECT_ROOT / "paper/metadata/test_metadata.tsv", separator='\t')
    
    # Check if validation metadata exists
    val_file = PROJECT_ROOT / "paper/metadata/validation_metadata.tsv"
    if val_file.exists():
        val_meta = pl.read_csv(val_file, separator='\t')
        existing = set(train_meta['Run_accession'].to_list() + 
                      test_meta['Run_accession'].to_list() + 
                      val_meta['Run_accession'].to_list())
    else:
        existing = set(train_meta['Run_accession'].to_list() + 
                      test_meta['Run_accession'].to_list())
    
    print(f"Loaded {len(existing)} existing accessions to avoid")
    
    all_samples = []
    
    for category, search_configs in searches.items():
        print(f"\n{'='*80}")
        print(f"CATEGORY: {category.upper()}")
        print(f"{'='*80}")
        
        category_samples = []
        
        for search_term, target in search_configs:
            studies = query_studies(search_term, max_studies=20)
            
            sample_count = 0
            
            for study in studies:
                if sample_count >= target:
                    break
                
                study_id = study.get('id')
                study_attrs = study.get('attributes', {})
                study_name = study_attrs.get('study-name', '')
                
                samples = get_study_samples(study_id, max_samples=15)
                
                if samples:
                    print(f"   📚 {study_id}: {len(samples)} samples from '{study_name[:60]}'")
                
                # Take max 10 samples per study for diversity
                for sample in samples[:10]:
                    if sample_count >= target:
                        break
                    
                    sample['search_term'] = search_term
                    sample['category'] = category
                    sample['study_name'] = study_name
                    category_samples.append(sample)
                    sample_count += 1
                
                time.sleep(0.3)  # Be nice to API
            
            print(f"   ✅ {search_term}: {sample_count}/{target} samples collected\n")
        
        all_samples.extend(category_samples)
        print(f"   Total {category}: {len(category_samples)} samples")
    
    # Save results
    df = pl.DataFrame(all_samples)
    print(f"\n{'='*80}")
    print(f"📊 TOTAL: {len(df)} sample accessions from MGnify")
    print(f"{'='*80}")
    
    print(f"\nSample distribution by category:")
    category_dist = df.group_by('category').agg(pl.count('sample_accession').alias('count'))
    for row in category_dist.iter_rows(named=True):
        pct = row['count'] / len(df) * 100
        print(f"  {row['category']:10s}: {row['count']:3d} ({pct:5.1f}%)")
    
    output_file = PROJECT_ROOT / "data/validation/modern_samples_mgnify_v2.tsv"
    df.write_csv(output_file, separator='\t')
    print(f"\n💾 Saved to: {output_file}")
    
    # Save just accessions for ENA expansion (next step)
    accessions = df['sample_accession'].drop_nulls().unique().to_list()
    acc_file = PROJECT_ROOT / "data/validation/modern_sample_accessions_v2.txt"
    with open(acc_file, 'w') as f:
        f.write('\n'.join(accessions))
    print(f"💾 Saved {len(accessions)} unique accessions to: {acc_file}")
    
    print("\n" + "="*80)
    print("📝 NEXT STEPS:")
    print("="*80)
    print("1. Expand sample accessions → run accessions using ENA API")
    print("   mamba run -p ./env python scripts/validation/11_expand_modern_samples.py")
    print("\n2. Balance to match training distribution")
    print("   mamba run -p ./env python scripts/validation/12_balance_modern_samples_v2.py")
    print("\n3. Manually review ENA metadata for selected samples")
    print("   Check for pooled/multiplexed indicators in sample descriptions")

if __name__ == "__main__":
    main()
