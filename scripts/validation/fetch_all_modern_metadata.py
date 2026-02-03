#!/usr/bin/env python3
"""
Fetch SRA metadata for ALL 150 modern samples (existing 41 + new 109).
This will allow comprehensive and consistent labeling.
"""

import requests
import polars as pl
from pathlib import Path
import time
from xml.etree import ElementTree as ET

PROJECT_ROOT = Path("/pasteur/appa/scratch/cduitama/EDID/decOM-classify")

def fetch_sra_metadata(run_accession: str):
    """Fetch SRA metadata using NCBI E-utilities."""
    esearch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    esearch_params = {
        'db': 'sra',
        'term': run_accession,
        'retmode': 'json'
    }
    
    try:
        response = requests.get(esearch_url, params=esearch_params, timeout=10)
        if response.status_code != 200:
            return None
        
        search_data = response.json()
        id_list = search_data.get('esearchresult', {}).get('idlist', [])
        
        if not id_list:
            return None
        
        sra_id = id_list[0]
        
        efetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        efetch_params = {
            'db': 'sra',
            'id': sra_id,
            'retmode': 'xml'
        }
        
        response = requests.get(efetch_url, params=efetch_params, timeout=10)
        if response.status_code != 200:
            return None
        
        root = ET.fromstring(response.content)
        
        metadata = {
            'run_accession': run_accession,
            'sample_accession': None,
            'sample_title': None,
            'organism': None,
            'isolation_source': None,
            'tissue': None,
            'isolate': None,
            'body_site': None,
            'host': None,
            'env_biome': None,
            'env_material': None,
            'library_strategy': None,
            'library_source': None,
            'all_attributes': []
        }
        
        for sample in root.findall('.//SAMPLE'):
            metadata['sample_accession'] = sample.get('accession')
            title = sample.find('.//TITLE')
            if title is not None:
                metadata['sample_title'] = title.text
            
            organism = sample.find('.//SCIENTIFIC_NAME')
            if organism is not None:
                metadata['organism'] = organism.text
            
            for attr in sample.findall('.//SAMPLE_ATTRIBUTE'):
                tag = attr.find('TAG')
                value = attr.find('VALUE')
                if tag is not None and value is not None:
                    tag_text = tag.text
                    value_text = value.text
                    metadata['all_attributes'].append(f"{tag_text}: {value_text}")
                    
                    tag_lower = tag_text.lower()
                    if tag_lower == 'isolation_source':
                        metadata['isolation_source'] = value_text
                    elif tag_lower in ['tissue', 'body_product']:
                        metadata['tissue'] = value_text
                    elif tag_lower == 'isolate':
                        metadata['isolate'] = value_text
                    elif tag_lower in ['body_site', 'body-site', 'body site']:
                        metadata['body_site'] = value_text
                    elif tag_lower in ['host', 'host_subject_id']:
                        metadata['host'] = value_text
                    elif tag_lower in ['env_biome', 'environmental biome']:
                        metadata['env_biome'] = value_text
                    elif tag_lower in ['env_material', 'environmental material']:
                        metadata['env_material'] = value_text
        
        for lib_desc in root.findall('.//LIBRARY_DESCRIPTOR'):
            strategy = lib_desc.find('LIBRARY_STRATEGY')
            if strategy is not None:
                metadata['library_strategy'] = strategy.text
            
            source = lib_desc.find('LIBRARY_SOURCE')
            if source is not None:
                metadata['library_source'] = source.text
        
        metadata['all_attributes'] = ' | '.join(metadata['all_attributes'])
        return metadata
        
    except Exception as e:
        print(f"Error fetching {run_accession}: {e}")
        return None

def main():
    # Load all modern accessions
    accessions_file = PROJECT_ROOT / "data/validation/all_modern_accessions_for_labeling.txt"
    with open(accessions_file) as f:
        accessions = [line.strip() for line in f if line.strip()]
    
    print(f"🔍 Fetching SRA metadata for ALL {len(accessions)} modern samples...")
    print(f"   This will take ~{len(accessions) * 0.4 / 60:.0f} minutes with rate limiting...\n")
    
    metadata_list = []
    failed = []
    
    for i, acc in enumerate(accessions, 1):
        if i % 10 == 0 or i == 1:
            print(f"  Progress: {i}/{len(accessions)} ({i/len(accessions)*100:.1f}%)")
        
        metadata = fetch_sra_metadata(acc)
        
        if metadata:
            metadata_list.append(metadata)
        else:
            failed.append(acc)
        
        time.sleep(0.4)  # Rate limiting
    
    print(f"\n📊 Retrieved metadata for {len(metadata_list)}/{len(accessions)} samples")
    if failed:
        print(f"⚠️  Failed to retrieve: {len(failed)} samples")
        if len(failed) <= 10:
            print(f"   Failed: {', '.join(failed)}")
        else:
            print(f"   First 10 failures: {', '.join(failed[:10])}")
    else:
        print("✓ All samples retrieved successfully!")
    
    # Save to file
    df = pl.DataFrame(metadata_list)
    output_file = PROJECT_ROOT / "data/validation/all_modern_sra_metadata.tsv"
    df.write_csv(output_file, separator='\t')
    
    print(f"\n💾 Saved metadata to: {output_file}")
    
    # Show summary
    print("\n" + "="*80)
    print("METADATA SUMMARY")
    print("="*80)
    print(f"\nOrganisms:")
    print(df.group_by('organism').agg(pl.count('run_accession').alias('count')).sort('count', descending=True))
    
    print(f"\nLibrary strategies:")
    print(df.group_by('library_strategy').agg(pl.count('run_accession').alias('count')))
    
    print(f"\nLibrary sources:")
    print(df.group_by('library_source').agg(pl.count('run_accession').alias('count')))
    
    print("\n" + "="*80)
    print("NEXT STEP:")
    print("="*80)
    print("""
Create manual labeling spreadsheet:
  mamba run -p ./env python scripts/validation/create_labeling_workflow.py
    """)

if __name__ == "__main__":
    main()
