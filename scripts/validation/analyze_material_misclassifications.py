#!/usr/bin/env python3
"""
Analyze Material task misclassifications in validation set.
Extract sample IDs and common characteristics for the top error patterns.
"""

import pandas as pd
import json
from pathlib import Path
from collections import defaultdict

# Error patterns to analyze
ERROR_PATTERNS = [
    ('dental calculus', 'Oral'),
    ('Oral', 'Skin'),
    ('dental calculus', 'tooth'),
    ('tooth', 'dental calculus'),
    ('bone', 'tooth'),
    ('soil', 'Skin')
]

def main():
    # Load validation metadata
    val_meta = pd.read_csv('paper/metadata/validation_metadata.tsv', sep='\t')
    
    # Load validation predictions
    predictions_dir = Path('results/validation_predictions')
    # Get all sample directories (skip JSON summary file)
    sample_dirs = [d for d in predictions_dir.iterdir() if d.is_dir()]
    
    print(f"Loading predictions from {len(sample_dirs)} samples...")
    
    # Build dataframe of Material task predictions
    records = []
    for i, sample_dir in enumerate(sample_dirs):
        if i % 100 == 0:
            print(f"  Processing {i}/{len(sample_dirs)}...")
        
        sample_id = sample_dir.name
        # Find the predictions JSON file
        pred_files = list(sample_dir.glob('*_predictions.json'))
        if not pred_files:
            continue
        
        with open(pred_files[0]) as f:
            pred = json.load(f)
        
        if 'material' in pred['predictions']:
            mat_pred = pred['predictions']['material']
            records.append({
                'sample_id': sample_id,
                'pred_label': mat_pred['predicted_class'],
                'confidence': mat_pred['confidence']
            })
    
    pred_df = pd.DataFrame(records)
    
    # Merge with metadata
    merged = pred_df.merge(
        val_meta[['Run_accession', 'material', 'sample_host', 'community_type', 
                  'geo_loc_name', 'publication_year', 'project_name']], 
        left_on='sample_id', 
        right_on='Run_accession', 
        how='left'
    )
    
    print("\n" + "=" * 80)
    print("MATERIAL TASK MISCLASSIFICATION ANALYSIS")
    print("=" * 80)
    
    # Analyze each error pattern
    all_results = []
    
    for true_label, pred_label in ERROR_PATTERNS:
        subset = merged[(merged['material'] == true_label) & (merged['pred_label'] == pred_label)]
        
        print(f"\n{true_label} → {pred_label}: {len(subset)} samples")
        print(f"  Avg confidence: {subset['confidence'].mean():.2%}")
        
        # Sample hosts
        hosts = subset['sample_host'].value_counts()
        print(f"  Sample hosts ({len(hosts)} unique):")
        for host, count in hosts.head(5).items():
            print(f"    {host}: {count}")
        
        # Community types
        communities = subset['community_type'].value_counts()
        print(f"  Community types:")
        for comm, count in communities.items():
            print(f"    {comm}: {count}")
        
        # Countries
        countries = subset['geo_loc_name'].str.split(':').str[0].value_counts()
        print(f"  Top countries:")
        for country, count in countries.head(3).items():
            print(f"    {country}: {count}")
        
        # Projects
        projects = subset['project_name'].value_counts()
        print(f"  Top projects:")
        for project, count in projects.head(3).items():
            print(f"    {project}: {count}")
        
        # Save sample IDs
        for _, row in subset.iterrows():
            all_results.append({
                'sample_id': row['sample_id'],
                'error_pattern': f"{true_label} → {pred_label}",
                'true_label': true_label,
                'pred_label': pred_label,
                'confidence': row['confidence'],
                'sample_host': row['sample_host'],
                'community_type': row['community_type'],
                'geo_loc_name': row['geo_loc_name'],
                'publication_year': row['publication_year'],
                'project_name': row['project_name']
            })
    
    # Save to file
    output_df = pd.DataFrame(all_results)
    output_file = 'results/material_misclassifications_validation.tsv'
    output_df.to_csv(output_file, sep='\t', index=False)
    
    print(f"\n{'=' * 80}")
    print(f"Saved {len(output_df)} misclassified samples to: {output_file}")
    print(f"{'=' * 80}\n")
    
    # Summary statistics
    print("\nSUMMARY:")
    print(f"  Total misclassified samples: {len(output_df)}")
    if len(output_df) > 0:
        print(f"  Unique samples: {output_df['sample_id'].nunique()}")
        print(f"  Average confidence: {output_df['confidence'].mean():.2%}")
        print(f"\nMost common sample hosts:")
        for host, count in output_df['sample_host'].value_counts().head(5).items():
            print(f"  {host}: {count}")
        print(f"\nMost common projects:")
        for project, count in output_df['project_name'].value_counts().head(5).items():
            print(f"  {project}: {count}")

if __name__ == '__main__':
    main()
