#!/usr/bin/env python3
"""
Interactive label review for modern TRAINING samples.
Downloads SRA metadata and allows verification/correction of material labels.

Particularly useful for:
- Verifying the 67 "Oral" labeled samples (non-standard label)
- Checking if material labels match original SRA metadata
- Identifying potential mislabeling issues
"""

import polars as pl
import requests
from pathlib import Path
import time
import sys

def fetch_sra_metadata(run_accessions, batch_size=1):
    """
    Fetch ALL metadata from ENA API for given run accessions.
    Fetches everything without filtering to show complete raw SRA data.
    
    Args:
        run_accessions: List of SRA run accessions
        batch_size: Number of accessions per API request (default 1 for stability)
        
    Returns:
        Polars DataFrame with ALL SRA metadata fields
    """
    base_url = "https://www.ebi.ac.uk/ena/portal/api/filereport"
    
    all_metadata = []
    
    for i in range(0, len(run_accessions), batch_size):
        batch = run_accessions[i:i+batch_size]
        
        # Request ALL available fields - no filtering
        params = {
            'accession': ','.join(batch),
            'result': 'read_run',
            'fields': 'all'  # Get everything!
        }
        
        print(f"Fetching {i+1}/{len(run_accessions)}... ", end='', flush=True)
        
        try:
            response = requests.get(base_url, params=params, timeout=30)
            response.raise_for_status()
            
            # Parse TSV response - keep everything as-is
            lines = response.text.strip().split('\n')
            if len(lines) > 1:  # Has data beyond header
                header = lines[0].split('\t')
                for line in lines[1:]:
                    values = line.split('\t')
                    # Pad values if needed (in case of missing trailing columns)
                    values += [''] * (len(header) - len(values))
                    all_metadata.append(dict(zip(header, values)))
                print(f"✓ ({len(header)} fields)")
            else:
                print("✗ No data")
                
            time.sleep(0.3)  # Rate limiting
            
        except Exception as e:
            print(f"✗ {str(e)[:60]}")
            # Continue even on errors - we'll get what we can
            continue
    
    if not all_metadata:
        print("WARNING: No metadata fetched!")
        return None
    
    return pl.DataFrame(all_metadata)


def main():
    """Main interactive review workflow"""
    
    # Load training metadata
    print("Loading training metadata...")
    train_meta = pl.read_csv('paper/metadata/train_metadata.tsv', separator='\t')
    
    # Get modern samples only
    modern_train = train_meta.filter(pl.col('sample_type') == 'modern_metagenome')
    
    print(f"\nFound {len(modern_train)} modern training samples")
    print(f"\nMaterial distribution:")
    mat_dist = modern_train.group_by('material').agg(pl.len()).sort('len', descending=True)
    print(mat_dist)
    
    # Check if SRA metadata already downloaded
    sra_meta_path = Path('data/training/modern_sra_metadata.tsv')
    
    if sra_meta_path.exists():
        print(f"\n✓ Found existing SRA metadata: {sra_meta_path}")
        sra_meta = pl.read_csv(sra_meta_path, separator='\t')
        print(f"  {len(sra_meta)} records loaded")
    else:
        print(f"\n✗ No existing SRA metadata found")
        print(f"  Fetching from ENA API for {len(modern_train)} samples...")
        
        # Create output directory
        sra_meta_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Fetch metadata
        run_accs = modern_train.select('Run_accession').to_series().to_list()
        sra_meta = fetch_sra_metadata(run_accs)
        
        if sra_meta is None:
            print("ERROR: Failed to fetch SRA metadata")
            sys.exit(1)
        
        # Save fetched metadata
        sra_meta.write_csv(sra_meta_path, separator='\t')
        print(f"\n✓ Saved SRA metadata to {sra_meta_path}")
    
    # Join with training metadata
    print("\nJoining with training labels...")
    review_df = modern_train.join(
        sra_meta,
        left_on='Run_accession',
        right_on='run_accession',
        how='left'
    )
    
    # Save combined review file with ALL metadata columns
    review_file = Path('data/training/modern_samples_for_review.tsv')
    review_df.write_csv(review_file, separator='\t')
    print(f"✓ Saved combined review file: {review_file}")
    print(f"  Columns: {len(review_df.columns)}")
    
    # Load existing review log if it exists
    review_log = []
    reviewed_accessions = set()
    
    review_log_path = Path('data/training/label_review_log.tsv')
    
    if review_log_path.exists():
        existing_review = pl.read_csv(review_log_path, separator='\t')
        review_log = existing_review.to_dicts()
        reviewed_accessions = set(existing_review['run_accession'].to_list())
        print(f"\n📋 Found existing review log with {len(reviewed_accessions)} samples already reviewed")
        print(f"   Resuming from sample {len(reviewed_accessions) + 1}/{len(review_df)}\n")
    else:
        print(f"\n📋 Starting fresh review (no existing log found)\n")
    
    # Interactive review loop
    print(f"\n{'='*100}")
    print(f"INTERACTIVE LABEL REVIEW - {len(review_df)} MODERN TRAINING SAMPLES")
    print(f"{'='*100}")
    print(f"\nFor each sample, I will show:")
    print(f"  1. Run accession")
    print(f"  2. CURRENT TRAINING LABELS: Material | Sample Host | Community Type")
    print(f"  3. SRA METADATA: organism, sample_title, isolation_source, body_site, etc.")
    print(f"\nYou respond with:")
    print(f"  YES - Approve current labels")
    print(f"  NO  - Provide corrected labels or EXCLUDE sample")
    print(f"  SKIP - Skip this sample for now")
    print(f"  QUIT - Exit and save progress")
    print(f"\n{'='*100}\n")
    
    try:
        for i, row in enumerate(review_df.iter_rows(named=True), 1):
            acc = row['Run_accession']
            
            # Skip if already reviewed
            if acc in reviewed_accessions:
                continue
            
            # Current training labels
            material = row.get('material', 'NULL')
            sample_host = row.get('sample_host', 'NULL')
            community_type = row.get('community_type', 'NULL')
            
            print(f"\n{'='*100}")
            print(f"[{i}/{len(review_df)}] {acc}")
            print(f"{'='*100}")
            
            print(f"\n🏷️  CURRENT TRAINING LABELS:")
            print(f"  Material: {material}")
            print(f"  Sample Host: {sample_host}")
            print(f"  Community Type: {community_type}")
            print(f"  Sample Type: modern_metagenome")
            
            print(f"\n📋 SRA METADATA (ALL AVAILABLE FIELDS):")
            
            # Display ALL metadata fields (excluding Run_accession and training labels)
            exclude_cols = {'Run_accession', 'material', 'sample_host', 'community_type', 'sample_type'}
            for key in row.keys():
                if key not in exclude_cols:
                    value = row[key]
                    # Only display if value exists and is not empty
                    if value and value != '' and value != 'NULL':
                        print(f"  {key}: {value}")
            
            # Get user input
            response = input(f"\n✅ Decision? (YES/NO/SKIP/QUIT): ").strip().upper()
            
            if response == 'QUIT':
                print("\n💾 Saving progress and exiting...")
                break
                
            elif response == 'SKIP':
                print("   ⏭️  SKIPPED\n")
                continue
                
            elif response == 'YES':
                reason = input(f"   Optional note: ").strip()
                review_log.append({
                    'run_accession': acc,
                    'decision': 'APPROVED',
                    'original_material': material,
                    'original_host': sample_host,
                    'original_community': community_type,
                    'corrected_material': material,
                    'corrected_host': sample_host,
                    'corrected_community': community_type,
                    'reason': reason if reason else 'Approved'
                })
                print(f"   ✅ APPROVED\n")
                
            elif response == 'NO':
                print(f"\n   Please provide:")
                corrected_material = input(f"     Corrected Material (or EXCLUDE): ").strip()
                
                if corrected_material.upper() == 'EXCLUDE':
                    reason = input(f"     Reason for exclusion: ").strip()
                    review_log.append({
                        'run_accession': acc,
                        'decision': 'EXCLUDED',
                        'original_material': material,
                        'original_host': sample_host,
                        'original_community': community_type,
                        'corrected_material': None,
                        'corrected_host': None,
                        'corrected_community': None,
                        'reason': reason
                    })
                    print(f"   ❌ EXCLUDED\n")
                else:
                    corrected_host = input(f"     Corrected Sample Host: ").strip()
                    corrected_community = input(f"     Corrected Community Type: ").strip()
                    reason = input(f"     Reason for correction: ").strip()
                    
                    review_log.append({
                        'run_accession': acc,
                        'decision': 'CORRECTED',
                        'original_material': material,
                        'original_host': sample_host,
                        'original_community': community_type,
                        'corrected_material': corrected_material,
                        'corrected_host': corrected_host,
                        'corrected_community': corrected_community,
                        'reason': reason
                    })
                    print(f"   ✏️  CORRECTED\n")
            else:
                print(f"   Invalid response. Please enter YES/NO/SKIP/QUIT.")
                continue
            
            # Save progress after each review
            if review_log:
                review_log_df = pl.DataFrame(review_log)
                review_log_df.write_csv(review_log_path, separator='\t')
                
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user. Saving progress...")
    
    # Final summary
    print(f"\n{'='*100}")
    print(f"REVIEW SESSION COMPLETE")
    print(f"{'='*100}")
    
    if review_log:
        print(f"\n✅ Review log saved to: {review_log_path}")
        
        review_log_df = pl.DataFrame(review_log)
        approved = len(review_log_df.filter(pl.col('decision') == 'APPROVED'))
        corrected = len(review_log_df.filter(pl.col('decision') == 'CORRECTED'))
        excluded = len(review_log_df.filter(pl.col('decision') == 'EXCLUDED'))
        
        print(f"\nSUMMARY:")
        print(f"  Approved: {approved}")
        print(f"  Corrected: {corrected}")
        print(f"  Excluded: {excluded}")
        print(f"  TOTAL REVIEWED: {len(review_log_df)}")
        print(f"  REMAINING: {len(review_df) - len(review_log_df)}")
    else:
        print("\nNo samples reviewed in this session.")


if __name__ == '__main__':
    main()
