#!/usr/bin/env python3
"""Debug script to inspect the metadata structure."""

import time
import requests
from pathlib import Path
import xml.etree.ElementTree as ET
import json

PROJECT_ROOT = Path("/pasteur/appa/scratch/cduitama/EDID/decOM-classify")

def fetch_sra_metadata(run_accession):
    """Fetch detailed metadata for a single SRA run."""
    # First, get the SRA ID
    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    search_params = {
        "db": "sra",
        "term": run_accession,
        "retmode": "json"
    }
    
    try:
        response = requests.get(search_url, params=search_params)
        response.raise_for_status()
        data = response.json()
        
        if not data.get("esearchresult", {}).get("idlist"):
            print(f"No SRA record found for {run_accession}")
            return None
        
        sra_id = data["esearchresult"]["idlist"][0]
        
        # Fetch full metadata
        fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        fetch_params = {
            "db": "sra",
            "id": sra_id,
            "retmode": "xml"
        }
        
        response = requests.get(fetch_url, params=fetch_params)
        response.raise_for_status()
        
        # Parse XML
        root = ET.fromstring(response.content)
        
        metadata = {
            "run_accession": run_accession,
            "organism": None,
            "isolation_source": None,
            "tissue": None,
            "isolate": None,
            "body_site": None,
            "host": None,
            "env_biome": None,
            "env_material": None,
            "library_strategy": None,
            "all_attributes": ""
        }
        
        # Extract organism
        org_elem = root.find(".//SCIENTIFIC_NAME")
        if org_elem is not None:
            metadata["organism"] = org_elem.text
        
        # Extract library strategy
        lib_elem = root.find(".//LIBRARY_STRATEGY")
        if lib_elem is not None:
            metadata["library_strategy"] = lib_elem.text
        
        # Extract all sample attributes
        attributes = []
        for attr in root.findall(".//SAMPLE_ATTRIBUTE"):
            tag = attr.find("TAG")
            value = attr.find("VALUE")
            if tag is not None and value is not None:
                tag_text = tag.text.lower()
                value_text = value.text
                
                attributes.append(f"{tag.text}={value_text}")
                
                # Map to known fields
                if tag_text in ['isolation_source', 'isolation source']:
                    metadata["isolation_source"] = value_text
                elif tag_text in ['tissue', 'tissue_type']:
                    metadata["tissue"] = value_text
                elif tag_text in ['isolate']:
                    metadata["isolate"] = value_text
                elif tag_text in ['body_site', 'body site', 'body_product']:
                    metadata["body_site"] = value_text
                elif tag_text in ['host']:
                    metadata["host"] = value_text
                elif tag_text in ['env_biome', 'environment (biome)']:
                    metadata["env_biome"] = value_text
                elif tag_text in ['env_material', 'environment (material)']:
                    metadata["env_material"] = value_text
        
        metadata["all_attributes"] = " | ".join(attributes)
        
        return metadata
        
    except Exception as e:
        print(f"Error fetching {run_accession}: {e}")
        return None


# Fetch just the first 10 samples to inspect
accession_file = PROJECT_ROOT / "data/validation/all_modern_accessions_for_labeling.txt"
with open(accession_file) as f:
    accessions = [line.strip() for line in f if line.strip()]

print(f"Fetching first 10 samples to inspect structure...")
metadata_list = []

for i, acc in enumerate(accessions[:10], 1):
    print(f"  {i}/10: {acc}")
    metadata = fetch_sra_metadata(acc)
    if metadata:
        metadata_list.append(metadata)
    time.sleep(0.4)

# Save as JSON to inspect
json_file = PROJECT_ROOT / "data/validation/debug_metadata_sample.json"
with open(json_file, 'w') as f:
    json.dump(metadata_list, f, indent=2)

print(f"\n✓ Saved sample metadata to: {json_file}")

# Print field types
print("\n📊 Field value types:")
for field in metadata_list[0].keys():
    values = [row[field] for row in metadata_list if row[field] is not None]
    if values:
        print(f"  {field}: {type(values[0]).__name__} (e.g., '{values[0][:50] if isinstance(values[0], str) else values[0]}')")
    else:
        print(f"  {field}: None in all samples")
