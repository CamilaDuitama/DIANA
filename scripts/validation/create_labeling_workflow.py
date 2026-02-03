#!/usr/bin/env python3
"""
Create a comprehensive labeling workflow for all 150 modern samples.
Combines SRA metadata with existing labels and proposes new labels for review.
"""

import polars as pl
from pathlib import Path
import re

PROJECT_ROOT = Path("/pasteur/appa/scratch/cduitama/EDID/decOM-classify")

# Load all metadata
sra_df = pl.read_csv(PROJECT_ROOT / "data/validation/all_modern_sra_metadata.tsv", separator='\t', infer_schema_length=10000)
validation_df = pl.read_csv(PROJECT_ROOT / "paper/metadata/validation_metadata.tsv", separator='\t')

# Get existing modern sample labels
modern_existing = validation_df.filter(pl.col('sample_type') == 'modern_metagenome')
existing_labels = modern_existing.select(['Run_accession', 'material', 'sample_host', 'community_type'])

# Merge
labeled_df = sra_df.join(existing_labels, left_on='run_accession', right_on='Run_accession', how='left')

# Add proposed labels based on SRA metadata
def propose_material(row):
    """Propose material label based on SRA metadata."""
    iso = (row.get('isolation_source') or '').lower()
    tissue = (row.get('tissue') or '').lower()
    organism = (row.get('organism') or '').lower()
    attrs = (row.get('all_attributes') or '').lower()
    
    # Oral materials
    if any(x in iso + tissue + attrs for x in ['oral', 'saliva', 'plaque', 'tongue', 'buccal', 'gingival']):
        return 'Oral'
    
    # Gut/fecal materials
    if any(x in iso + tissue + attrs for x in ['stool', 'feces', 'fecal', 'gut', 'intestin']):
        return 'gut'
    
    # Skin materials
    if any(x in iso + tissue + attrs for x in ['skin', 'dermal', 'cutaneous', 'nares', 'retroauricular']):
        return 'Skin'
    
    # Soil
    if 'soil' in organism or 'soil' in iso:
        return 'soil'
    
    # Sediment
    if 'sediment' in organism or 'sediment' in iso:
        return 'sediment'
    
    return 'NEEDS_REVIEW'

def propose_host(row):
    """Propose sample_host based on organism."""
    organism = (row.get('organism') or '').lower()
    host = (row.get('host') or '').lower()
    
    if 'homo sapiens' in organism or 'homo sapiens' in host:
        return 'Homo sapiens'
    
    if 'human' in organism:
        return 'Homo sapiens'
    
    if any(x in organism for x in ['soil', 'sediment', 'marine', 'environmental']):
        return 'Not applicable - env sample'
    
    return 'NEEDS_REVIEW'

def propose_community(row):
    """Propose community_type based on material."""
    material = row.get('proposed_material')
    
    if material == 'Oral':
        return 'oral'
    elif material in ['Skin', 'gut']:
        return 'soft tissue'
    elif material in ['soil', 'sediment']:
        return 'Not applicable - env sample'
    
    return 'NEEDS_REVIEW'

# Apply proposals
rows = labeled_df.to_dicts()
for row in rows:
    row['proposed_material'] = propose_material(row)
    row['proposed_host'] = propose_host(row)
    row['proposed_community'] = propose_community(row)
    
    # Flag if proposal differs from existing label
    if row.get('material'):
        row['material_changed'] = row['material'] != row['proposed_material']
    else:
        row['material_changed'] = False
    
    row['needs_review'] = (
        row['proposed_material'] == 'NEEDS_REVIEW' or
        row['proposed_host'] == 'NEEDS_REVIEW' or
        row['proposed_community'] == 'NEEDS_REVIEW' or
        row.get('material_changed', False)
    )

result_df = pl.DataFrame(rows)

# Save for review
output_file = PROJECT_ROOT / "data/validation/modern_samples_for_labeling.tsv"
result_df.write_csv(output_file, separator='\t')

print("="*80)
print("LABELING WORKFLOW CREATED")
print("="*80)

print(f"\nTotal samples: {len(result_df)}")
print(f"  - Already labeled (in validation): {result_df.filter(pl.col('material').is_not_null()).height}")
print(f"  - New samples (need labels): {result_df.filter(pl.col('material').is_null()).height}")

print(f"\n🔍 Samples needing review: {result_df.filter(pl.col('needs_review') == True).height}")

# Show breakdown
review_needed = result_df.filter(pl.col('needs_review') == True)
if len(review_needed) > 0:
    print(f"\nReview reasons:")
    print(f"  - NEEDS_REVIEW material: {review_needed.filter(pl.col('proposed_material') == 'NEEDS_REVIEW').height}")
    print(f"  - NEEDS_REVIEW host: {review_needed.filter(pl.col('proposed_host') == 'NEEDS_REVIEW').height}")
    print(f"  - Label changed from existing: {review_needed.filter(pl.col('material_changed') == True).height}")

print(f"\n📋 Proposed label distribution:")
print(f"\nMaterial:")
print(result_df.group_by('proposed_material').agg(pl.count('run_accession').alias('count')).sort('count', descending=True))

print(f"\nHost:")
print(result_df.group_by('proposed_host').agg(pl.count('run_accession').alias('count')).sort('count', descending=True))

print(f"\nCommunity:")
print(result_df.group_by('proposed_community').agg(pl.count('run_accession').alias('count')).sort('count', descending=True))

print(f"\n💾 Saved labeling workflow to:")
print(f"   {output_file}")

print("\n" + "="*80)
print("NEXT STEPS:")
print("="*80)
print("""
1. Open the TSV file in a spreadsheet editor
2. Review samples where needs_review = true
3. Manually assign final labels
4. Save as modern_samples_labeled_final.tsv
""")

# Create a summary of samples needing review
if len(review_needed) > 0:
    review_file = PROJECT_ROOT / "data/validation/samples_needing_manual_review.tsv"
    review_cols = ['run_accession', 'organism', 'isolation_source', 'tissue', 'proposed_material', 
                   'proposed_host', 'proposed_community', 'material', 'all_attributes']
    review_needed.select(review_cols).write_csv(review_file, separator='\t')
    print(f"\n📝 Samples needing review saved to:")
    print(f"   {review_file}")
