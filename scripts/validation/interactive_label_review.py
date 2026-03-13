#!/usr/bin/env python3
"""
Interactive label review for ALL 150 modern samples.
For each sample, show metadata and proposed labels, ask for YES/NO approval + reason.
"""

import polars as pl
import json
from pathlib import Path

def main():
    # Load existing 41 modern samples with current labels
    existing_meta = pl.read_csv('data/validation/existing_41_for_review.tsv', separator='\t')
    existing_sra = pl.read_csv('data/validation/existing_41_sra_metadata.tsv', separator='\t')
    
    # Join to get full metadata
    existing_full = existing_meta.join(
        existing_sra.select(['run_accession', 'organism', 'isolation_source', 'env_material', 'host', 'library_strategy', 'library_source', 'all_attributes']),
        left_on='Run_accession',
        right_on='run_accession',
        how='left'
    )
    
    # EXCLUDE RNA-based sequencing samples (keep DNA-based including AMPLICON)
    print(f"Loaded {len(existing_full)} existing modern samples")
    print(f"Filtering out RNA-based sequencing samples...")
    
    existing_full = existing_full.filter(
        (pl.col('library_source') != 'TRANSCRIPTOMIC') &
        (pl.col('library_strategy') != 'FL-cDNA') &
        (pl.col('library_strategy') != 'RNA-Seq') &
        (pl.col('library_strategy') != 'ssRNA-seq') &
        (pl.col('library_strategy') != 'miRNA-Seq') &
        (pl.col('library_strategy') != 'EST')
    )
    
    print(f"After filtering: {len(existing_full)} samples remain\n")
    
    # Add source flag
    existing_full = existing_full.with_columns(pl.lit('EXISTING').alias('source'))
    
    # Load new 109 samples with proposed labels
    new_review = pl.read_csv('data/validation/NEW_109_detailed_review.tsv', separator='\t')
    
    # Load SRA metadata for new samples
    new_sra = pl.read_csv('data/validation/modern_sra_metadata_full.tsv', separator='\t', infer_schema_length=10000)
    
    # Join new samples - include ALL metadata columns
    new_full = new_review.join(
        new_sra.select(['run_accession', 'sample_accession', 'sample_title', 'geo_loc_name', 
                       'library_strategy', 'library_source', 'platform', 'instrument',
                       'collection_date', 'host', 'all_attributes']),
        on='run_accession',
        how='left'
    )
    
    # EXCLUDE RNA-based sequencing samples (keep DNA-based including AMPLICON)
    print(f"Loaded {len(new_full)} new modern samples")
    print(f"Filtering out RNA-based sequencing samples...")
    
    new_full = new_full.filter(
        (pl.col('library_source') != 'TRANSCRIPTOMIC') &
        (pl.col('library_strategy') != 'FL-cDNA') &
        (pl.col('library_strategy') != 'RNA-Seq') &
        (pl.col('library_strategy') != 'ssRNA-seq') &
        (pl.col('library_strategy') != 'miRNA-Seq') &
        (pl.col('library_strategy') != 'EST')
    )
    
    print(f"After filtering: {len(new_full)} samples remain\n")
    
    # Rename columns to match
    new_full = new_full.rename({
        'proposed_material': 'material',
        'proposed_host': 'sample_host',
        'proposed_community': 'community_type'
    })
    
    # Add source flag
    new_full = new_full.with_columns(pl.lit('NEW').alias('source'))
    
    # Standardize columns
    existing_cols = existing_full.select([
        pl.col('Run_accession').alias('run_accession'),
        'source',
        'organism',
        'isolation_source',
        'env_material',
        pl.col('host').alias('sra_host'),
        'library_strategy',
        'all_attributes',
        'material',
        'sample_host',
        'community_type',
        pl.lit(None, dtype=pl.Utf8).alias('geo_loc_name'),
        pl.lit(None, dtype=pl.Utf8).alias('sample_title'),
        pl.lit(None, dtype=pl.Utf8).alias('collection_date')
    ])
    
    new_cols = new_full.select([
        'run_accession',
        'source',
        'organism',
        'isolation_source',
        'env_material',
        pl.col('host').alias('sra_host'),
        'library_strategy',
        'all_attributes',
        'material',
        'sample_host',
        'community_type',
        'geo_loc_name',
        'sample_title',
        'collection_date'
    ])
    
    # Combine ALL 150 samples
    all_samples = pl.concat([existing_cols, new_cols])
    
    print(f"\n{'='*100}")
    print(f"INTERACTIVE LABEL REVIEW - ALL {len(all_samples)} MODERN SAMPLES")
    print(f"{'='*100}")
    print(f"\nEXISTING samples (already downloaded): {len(existing_cols)}")
    print(f"NEW samples (not yet downloaded): {len(new_cols)}")
    print(f"\nFor each sample, I will show:")
    print(f"  1. Run accession + source (EXISTING or NEW)")
    print(f"  2. SRA metadata: organism, isolation_source, env_material, library_strategy")
    print(f"  3. all_attributes (first 300 chars)")
    print(f"  4. PROPOSED LABELS: Material | Sample Host | Community Type")
    print(f"  5. Sample Type (always 'modern_metagenome')")
    print(f"\nYou respond with: YES or NO")
    print(f"  - If YES: I record 'APPROVED' + optional reason")
    print(f"  - If NO: You provide corrected labels OR 'EXCLUDE' to remove sample")
    print(f"\n{'='*100}\n")
    
    # Save for review
    all_samples.write_csv('data/validation/ALL_150_samples_for_review.tsv', separator='\t')
    print(f"✅ Saved combined list: data/validation/ALL_150_samples_for_review.tsv\n")
    
    # Load existing review log if it exists
    review_log = []
    reviewed_accessions = set()
    
    from pathlib import Path
    review_log_path = Path('data/validation/interactive_review_log.tsv')
    
    if review_log_path.exists():
        existing_review = pl.read_csv(review_log_path, separator='\t')
        review_log = existing_review.to_dicts()
        reviewed_accessions = set(existing_review['run_accession'].to_list())
        print(f"📋 Found existing review log with {len(reviewed_accessions)} samples already reviewed")
        print(f"   Resuming from sample {len(reviewed_accessions) + 1}/{len(all_samples)}\n")
    else:
        print(f"📋 Starting fresh review (no existing log found)\n")
    
    # Interactive loop
    for i, row in enumerate(all_samples.iter_rows(named=True), 1):
        acc = row['run_accession']
        
        # Skip if already reviewed
        if acc in reviewed_accessions:
            continue
        acc = row['run_accession']
        source = row['source']
        org = row['organism'] if row['organism'] else 'NULL'
        iso = row['isolation_source'] if row['isolation_source'] else 'NULL'
        env_mat = row['env_material'] if row['env_material'] else 'NULL'
        sra_host = row['sra_host'] if row['sra_host'] else 'NULL'
        lib_strat = row['library_strategy'] if row['library_strategy'] else 'NULL'
        geo_loc = row['geo_loc_name'] if row['geo_loc_name'] else 'NULL'
        sample_title = row['sample_title'] if row['sample_title'] else 'NULL'
        collection = row['collection_date'] if row['collection_date'] else 'NULL'
        all_attr = row['all_attributes'] if row['all_attributes'] else 'NULL'
        
        material = row['material'] if row['material'] else 'NULL'
        sample_host = row['sample_host'] if row['sample_host'] else 'NULL'
        community = row['community_type'] if row['community_type'] else 'NULL'
        sample_type = 'modern_metagenome'
        
        print(f"\n{'='*100}")
        print(f"[{i}/{len(all_samples)}] {acc} ({source})")
        print(f"{'='*100}")
        print(f"\n📋 SRA METADATA:")
        print(f"  organism: {org}")
        print(f"  isolation_source: {iso}")
        print(f"  env_material: {env_mat}")
        print(f"  host (SRA): {sra_host}")
        print(f"  geo_loc_name: {geo_loc}")
        print(f"  sample_title: {sample_title}")
        print(f"  collection_date: {collection}")
        print(f"  library_strategy: {lib_strat}")
        print(f"  all_attributes: {all_attr}")
        
        print(f"\n🏷️  PROPOSED LABELS:")
        print(f"  Material: {material}")
        print(f"  Sample Host: {sample_host}")
        print(f"  Community Type: {community}")
        print(f"  Sample Type: {sample_type}")
        
        # Get user input
        response = input(f"\n✅ APPROVE these labels? (YES/NO): ").strip().upper()
        
        if response == 'YES':
            reason = input(f"   Optional reason/note: ").strip()
            review_log.append({
                'run_accession': acc,
                'source': source,
                'decision': 'APPROVED',
                'material': material,
                'sample_host': sample_host,
                'community_type': community,
                'sample_type': sample_type,
                'reason': reason if reason else 'APPROVED'
            })
            print(f"   ✅ APPROVED\n")
            
        elif response == 'NO':
            print(f"\n   Please provide:")
            corrected_material = input(f"     Corrected Material (or EXCLUDE): ").strip()
            
            if corrected_material.upper() == 'EXCLUDE':
                reason = input(f"     Reason for exclusion: ").strip()
                review_log.append({
                    'run_accession': acc,
                    'source': source,
                    'decision': 'EXCLUDED',
                    'material': None,
                    'sample_host': None,
                    'community_type': None,
                    'sample_type': None,
                    'reason': reason
                })
                print(f"   ❌ EXCLUDED\n")
            else:
                corrected_host = input(f"     Corrected Sample Host: ").strip()
                corrected_community = input(f"     Corrected Community Type: ").strip()
                reason = input(f"     Reason for correction: ").strip()
                
                review_log.append({
                    'run_accession': acc,
                    'source': source,
                    'decision': 'CORRECTED',
                    'material': corrected_material,
                    'sample_host': corrected_host,
                    'community_type': corrected_community,
                    'sample_type': sample_type,
                    'reason': reason
                })
                print(f"   ✏️  CORRECTED\n")
        else:
            print(f"   Invalid response. Please enter YES or NO.")
            continue
        
        # Save progress after each sample
        review_df = pl.DataFrame(review_log)
        review_df.write_csv('data/validation/interactive_review_log.tsv', separator='\t')
    
    print(f"\n{'='*100}")
    print(f"REVIEW COMPLETE!")
    print(f"{'='*100}")
    print(f"\n✅ Review log saved to: data/validation/interactive_review_log.tsv")
    
    # Summary
    review_df = pl.DataFrame(review_log)
    approved = len(review_df.filter(pl.col('decision') == 'APPROVED'))
    corrected = len(review_df.filter(pl.col('decision') == 'CORRECTED'))
    excluded = len(review_df.filter(pl.col('decision') == 'EXCLUDED'))
    
    print(f"\nSUMMARY:")
    print(f"  Approved: {approved}")
    print(f"  Corrected: {corrected}")
    print(f"  Excluded: {excluded}")
    print(f"  TOTAL: {len(review_df)}")
    
if __name__ == '__main__':
    main()
