#!/usr/bin/env python3
"""
Analyze Material task misclassifications for SEEN labels in validation set.
Find samples where the model saw both the true label and predicted label during training,
yet still made incorrect predictions.
"""

import pandas as pd
from pathlib import Path

def main():
    # Load data
    val_pred = pd.read_csv('results/validation_predictions/validation_predictions.tsv', sep='\t')
    train_meta = pd.read_csv('paper/metadata/train_metadata.tsv', sep='\t')
    val_meta = pd.read_csv('paper/metadata/validation_metadata.tsv', sep='\t')
    
    # Get SEEN labels in training set
    seen_materials = set(train_meta['material'].dropna().unique())
    print(f"SEEN material labels in training ({len(seen_materials)}):")
    for label in sorted(seen_materials):
        count = (train_meta['material'] == label).sum()
        print(f"  {label}: {count} samples")
    
    # Filter for Material misclassifications
    mat_wrong = val_pred[val_pred['material_pred'] != val_pred['material_true']].copy()
    print(f"\nTotal Material misclassifications: {len(mat_wrong)}")
    
    # Filter for SEEN misclassifications (both true and pred labels seen in training)
    mat_wrong['true_seen'] = mat_wrong['material_true'].isin(seen_materials)
    mat_wrong['pred_seen'] = mat_wrong['material_pred'].isin(seen_materials)
    mat_wrong['both_seen'] = mat_wrong['true_seen'] & mat_wrong['pred_seen']
    
    seen_wrong = mat_wrong[mat_wrong['both_seen']]
    
    print(f"\nMaterial misclassifications with BOTH labels SEEN in training: {len(seen_wrong)}")
    print(f"  (These are concerning - model saw both labels but still confused them)")
    
    # Analyze patterns
    print("\n" + "="*80)
    print("TOP 10 SEEN→SEEN MISCLASSIFICATION PATTERNS")
    print("="*80)
    
    patterns = seen_wrong.groupby(['material_true', 'material_pred']).agg({
        'sample_id': 'count',
        'material_confidence': 'mean'
    }).rename(columns={'sample_id': 'count', 'material_confidence': 'avg_confidence'})
    patterns = patterns.sort_values('count', ascending=False)
    
    for (true_label, pred_label), row in patterns.head(10).iterrows():
        count = int(row['count'])
        conf = row['avg_confidence']
        pct_of_misclass = count / len(mat_wrong) * 100
        pct_of_total = count / len(val_pred) * 100
        
        print(f"\n{true_label} → {pred_label}")
        print(f"  Count: {count} ({pct_of_misclass:.1f}% of all misclassifications, {pct_of_total:.1f}% of total validation)")
        print(f"  Avg confidence: {conf:.1%}")
    
    # Merge with validation metadata for deeper analysis
    seen_wrong_full = seen_wrong.merge(
        val_meta[['Run_accession', 'sample_host', 'community_type', 'geo_loc_name', 
                  'publication_year', 'project_name', 'sample_type']],
        left_on='sample_id',
        right_on='Run_accession',
        how='left'
    )
    
    # Analyze top pattern in detail
    print("\n" + "="*80)
    print("DETAILED ANALYSIS: dental calculus → Oral (top SEEN→SEEN error)")
    print("="*80)
    
    dental_to_oral = seen_wrong_full[
        (seen_wrong_full['material_true'] == 'dental calculus') & 
        (seen_wrong_full['material_pred'] == 'Oral')
    ]
    
    if len(dental_to_oral) > 0:
        print(f"\nSamples: {len(dental_to_oral)}")
        print(f"Avg confidence: {dental_to_oral['material_confidence'].mean():.1%}")
        
        print(f"\nSample type distribution:")
        for stype, count in dental_to_oral['sample_type'].value_counts().items():
            print(f"  {stype}: {count}")
        
        print(f"\nSample host distribution:")
        for host, count in dental_to_oral['sample_host'].value_counts().head(5).items():
            print(f"  {host}: {count}")
        
        print(f"\nCommunity type distribution:")
        for comm, count in dental_to_oral['community_type'].value_counts().items():
            print(f"  {comm}: {count}")
        
        print(f"\nTop countries:")
        countries = dental_to_oral['geo_loc_name'].str.split(':').str[0]
        for country, count in countries.value_counts().head(5).items():
            print(f"  {country}: {count}")
        
        print(f"\nTop projects:")
        for project, count in dental_to_oral['project_name'].value_counts().head(5).items():
            print(f"  {project}: {count}")
        
        # Publication years
        print(f"\nPublication years:")
        for year, count in dental_to_oral['publication_year'].value_counts().sort_index().items():
            print(f"  {year}: {count}")
    
    # Save results
    output_file = Path('results/material_seen_misclassifications_validation.tsv')
    seen_wrong_full.to_csv(output_file, sep='\t', index=False)
    print(f"\n{'='*80}")
    print(f"Saved {len(seen_wrong_full)} SEEN misclassifications to: {output_file}")
    print(f"{'='*80}")
    
    # Summary
    print("\nSUMMARY:")
    print(f"  Total validation samples: {len(val_pred)}")
    print(f"  Total Material misclassifications: {len(mat_wrong)} ({len(mat_wrong)/len(val_pred)*100:.1f}%)")
    print(f"  SEEN→SEEN misclassifications: {len(seen_wrong)} ({len(seen_wrong)/len(mat_wrong)*100:.1f}% of errors)")
    print(f"  Average confidence on SEEN errors: {seen_wrong['material_confidence'].mean():.1%}")

if __name__ == '__main__':
    main()
