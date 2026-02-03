#!/usr/bin/env python3
"""
Check SRA metadata for modern samples to verify we can extract task labels.
Downloads metadata using NCBI E-utilities and analyzes available fields.
"""

import requests
import polars as pl
from pathlib import Path
import time
from xml.etree import ElementTree as ET

PROJECT_ROOT = Path("/pasteur/appa/scratch/cduitama/EDID/decOM-classify")

def fetch_sra_metadata(run_accession: str):
    """Fetch SRA metadata using NCBI E-utilities."""
    # Step 1: Search for the accession
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
        
        # Step 2: Fetch full record
        efetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        efetch_params = {
            'db': 'sra',
            'id': sra_id,
            'retmode': 'xml'
        }
        
        response = requests.get(efetch_url, params=efetch_params, timeout=10)
        if response.status_code != 200:
            return None
        
        # Parse XML
        root = ET.fromstring(response.content)
        
        # Extract relevant fields
        metadata = {
            'run_accession': run_accession,
            'sample_accession': None,
            'sample_title': None,
            'sample_description': None,
            'library_strategy': None,
            'library_source': None,
            'library_selection': None,
            'platform': None,
            'instrument': None,
            'biosample_model': None,
            'organism': None,
            'isolation_source': None,
            'collection_date': None,
            'geo_loc_name': None,
            'env_biome': None,
            'env_feature': None,
            'env_material': None,
            'body_site': None,
            'host': None,
            'attributes': []
        }
        
        # Extract from RUN_SET
        for run in root.findall('.//RUN'):
            metadata['run_accession'] = run.get('accession', run_accession)
        
        # Extract SAMPLE info
        for sample in root.findall('.//SAMPLE'):
            metadata['sample_accession'] = sample.get('accession')
            title = sample.find('.//TITLE')
            if title is not None:
                metadata['sample_title'] = title.text
            
            desc = sample.find('.//DESCRIPTION')
            if desc is not None:
                metadata['sample_description'] = desc.text
            
            organism = sample.find('.//SCIENTIFIC_NAME')
            if organism is not None:
                metadata['organism'] = organism.text
            
            # Get all SAMPLE_ATTRIBUTES
            for attr in sample.findall('.//SAMPLE_ATTRIBUTE'):
                tag = attr.find('TAG')
                value = attr.find('VALUE')
                if tag is not None and value is not None:
                    tag_text = tag.text
                    value_text = value.text
                    metadata['attributes'].append(f"{tag_text}: {value_text}")
                    
                    # Map specific attributes
                    if tag_text.lower() == 'isolation_source':
                        metadata['isolation_source'] = value_text
                    elif tag_text.lower() == 'collection_date':
                        metadata['collection_date'] = value_text
                    elif tag_text.lower() == 'geo_loc_name':
                        metadata['geo_loc_name'] = value_text
                    elif tag_text.lower() in ['env_biome', 'environmental biome']:
                        metadata['env_biome'] = value_text
                    elif tag_text.lower() in ['env_feature', 'environmental feature']:
                        metadata['env_feature'] = value_text
                    elif tag_text.lower() in ['env_material', 'environmental material']:
                        metadata['env_material'] = value_text
                    elif tag_text.lower() in ['body_site', 'body-site', 'body site']:
                        metadata['body_site'] = value_text
                    elif tag_text.lower() in ['host', 'host_subject_id']:
                        metadata['host'] = value_text
        
        # Extract LIBRARY info
        for lib_desc in root.findall('.//LIBRARY_DESCRIPTOR'):
            strategy = lib_desc.find('LIBRARY_STRATEGY')
            if strategy is not None:
                metadata['library_strategy'] = strategy.text
            
            source = lib_desc.find('LIBRARY_SOURCE')
            if source is not None:
                metadata['library_source'] = source.text
            
            selection = lib_desc.find('LIBRARY_SELECTION')
            if selection is not None:
                metadata['library_selection'] = selection.text
        
        # Extract PLATFORM
        for platform in root.findall('.//PLATFORM'):
            for child in platform:
                metadata['platform'] = child.tag
                instrument = child.find('INSTRUMENT_MODEL')
                if instrument is not None:
                    metadata['instrument'] = instrument.text
        
        return metadata
        
    except Exception as e:
        print(f"Error fetching {run_accession}: {e}")
        return None

def main():
    # Load modern accessions
    accessions_file = PROJECT_ROOT / "data/validation/modern_accessions_balanced_v2.txt"
    with open(accessions_file) as f:
        accessions = [line.strip() for line in f if line.strip()]
    
    print(f"🔍 Fetching metadata for ALL {len(accessions)} modern samples...")
    print(f"   This will take ~5-10 minutes with rate limiting...\n")
    
    # Process all accessions
    sample_accessions = accessions
    
    metadata_list = []
    failed = []
    for i, acc in enumerate(sample_accessions, 1):
        if i % 10 == 0 or i == 1:
            print(f"  Progress: {i}/{len(sample_accessions)} ({i/len(sample_accessions)*100:.1f}%)")
        
        metadata = fetch_sra_metadata(acc)
        
        if metadata:
            metadata_list.append(metadata)
        else:
            failed.append(acc)
        
        time.sleep(0.4)  # Rate limiting
    
    if not metadata_list:
        print("\n❌ No metadata could be retrieved!")
        return
    
    print(f"\n📊 Retrieved metadata for {len(metadata_list)}/{len(sample_accessions)} samples")
    if failed:
        print(f"⚠️  Failed to retrieve: {len(failed)} samples")
        print(f"   First 5 failures: {', '.join(failed[:5])}\n")
    else:
        print("✓ All samples retrieved successfully!\n")
    
    print(f"\n📊 Retrieved metadata for {len(metadata_list)} samples\n")
    
    # Analyze field availability
    print("="*80)
    print("FIELD AVAILABILITY ANALYSIS")
    print("="*80)
    
    fields_to_check = [
        'sample_title', 'sample_description', 'library_strategy', 'library_source',
        'organism', 'isolation_source', 'env_biome', 'env_feature', 'env_material',
        'body_site', 'host', 'geo_loc_name'
    ]
    
    for field in fields_to_check:
        count = sum(1 for m in metadata_list if m.get(field))
        pct = count / len(metadata_list) * 100
        status = "✓" if pct >= 50 else "⚠️" if pct > 0 else "✗"
        print(f"{status} {field:20s}: {count:2d}/{len(metadata_list)} ({pct:5.1f}%)")
    
    print("\n" + "="*80)
    print("SAMPLE EXAMPLES")
    print("="*80)
    
    for i, m in enumerate(metadata_list[:3], 1):
        print(f"\nSample {i}: {m['run_accession']}")
        print(f"  Sample Accession: {m['sample_accession']}")
        print(f"  Title: {m['sample_title']}")
        print(f"  Description: {m['sample_description'][:100] if m['sample_description'] else 'N/A'}...")
        print(f"  Organism: {m['organism']}")
        print(f"  Library: {m['library_strategy']} / {m['library_source']}")
        print(f"  Isolation Source: {m['isolation_source']}")
        print(f"  Body Site: {m['body_site']}")
        print(f"  Host: {m['host']}")
        print(f"  Env Biome: {m['env_biome']}")
        print(f"  Env Material: {m['env_material']}")
        print(f"\n  All attributes:")
        for attr in m['attributes'][:10]:
            print(f"    - {attr}")
    
    # Save full metadata
    output_file = PROJECT_ROOT / "data/validation/modern_sra_metadata_full.tsv"
    
    # Convert to dataframe (exclude attributes list)
    df_data = []
    for m in metadata_list:
        row = {k: v for k, v in m.items() if k != 'attributes'}
        row['all_attributes'] = ' | '.join(m['attributes'])
        df_data.append(row)
    
    df = pl.DataFrame(df_data)
    df.write_csv(output_file, separator='\t')
    print(f"\n💾 Saved sample metadata to: {output_file}")
    
    print("\n" + "="*80)
    print("TASK LABEL ASSESSMENT")
    print("="*80)
    print("\n1. MATERIAL:")
    print("   Can use: isolation_source, env_material, body_site, sample_description")
    print("   ⚠️  Requires manual mapping from free text")
    
    print("\n2. SAMPLE_TYPE:")
    print("   ✓ All are modern_metagenome (by definition)")
    
    print("\n3. COMMUNITY_TYPE:")
    print("   Can use: env_biome, isolation_source, body_site")
    print("   ⚠️  Requires manual mapping")
    
    print("\n4. SAMPLE_HOST:")
    print("   Can use: organism, host")
    print("   ✓ Should be available for most samples")

if __name__ == "__main__":
    main()
