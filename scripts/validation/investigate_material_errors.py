#!/usr/bin/env python3
"""
Investigate WHY Material task predictions are poor.
Hypotheses to test:
1. Sparse unitig vectors (insufficient data)
2. Incorrect/ambiguous labels
3. Biologically similar materials (dental calculus vs Oral - both from mouth)
4. Sample quality issues (degradation, contamination)
5. Training data imbalance
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

def load_unitig_fraction(sample_id):
    """Load unitig fraction vector for a sample."""
    pred_dir = Path(f'results/validation_predictions/{sample_id}')
    frac_file = pred_dir / f'{sample_id}_unitig_fraction.txt'
    
    if not frac_file.exists():
        return None
    
    fractions = []
    with open(frac_file) as f:
        for line in f:
            fractions.append(float(line.strip()))
    
    return np.array(fractions)

def analyze_sparsity(misclass_df, correct_df):
    """Compare unitig sparsity between misclassified and correct samples."""
    print("\n" + "="*80)
    print("HYPOTHESIS 1: Sparse/Empty Unitig Vectors")
    print("="*80)
    
    misclass_sparsity = []
    misclass_nonzero = []
    
    print("\nAnalyzing misclassified samples...")
    for i, row in misclass_df.head(50).iterrows():  # Sample first 50
        vec = load_unitig_fraction(row['sample_id'])
        if vec is not None:
            sparsity = (vec == 0).sum() / len(vec) * 100
            nonzero = (vec > 0).sum()
            misclass_sparsity.append(sparsity)
            misclass_nonzero.append(nonzero)
    
    correct_sparsity = []
    correct_nonzero = []
    
    print("Analyzing correctly classified samples...")
    for i, row in correct_df.head(50).iterrows():  # Sample first 50
        vec = load_unitig_fraction(row['sample_id'])
        if vec is not None:
            sparsity = (vec == 0).sum() / len(vec) * 100
            nonzero = (vec > 0).sum()
            correct_sparsity.append(sparsity)
            correct_nonzero.append(nonzero)
    
    if misclass_sparsity and correct_sparsity:
        print(f"\nMisclassified samples (n={len(misclass_sparsity)}):")
        print(f"  Avg sparsity: {np.mean(misclass_sparsity):.1f}%")
        print(f"  Avg non-zero unitigs: {np.mean(misclass_nonzero):.0f}")
        print(f"  Min non-zero unitigs: {np.min(misclass_nonzero):.0f}")
        
        print(f"\nCorrectly classified samples (n={len(correct_sparsity)}):")
        print(f"  Avg sparsity: {np.mean(correct_sparsity):.1f}%")
        print(f"  Avg non-zero unitigs: {np.mean(correct_nonzero):.0f}")
        print(f"  Min non-zero unitigs: {np.min(correct_nonzero):.0f}")
        
        diff_sparsity = np.mean(misclass_sparsity) - np.mean(correct_sparsity)
        diff_nonzero = np.mean(misclass_nonzero) - np.mean(correct_nonzero)
        
        print(f"\nDifference:")
        print(f"  Sparsity: {diff_sparsity:+.1f}% {'(misclassified MORE sparse)' if diff_sparsity > 0 else '(similar sparsity)'}")
        print(f"  Non-zero unitigs: {diff_nonzero:+.0f} {'(misclassified have FEWER unitigs)' if diff_nonzero < 0 else '(similar coverage)'}")

def analyze_biological_similarity(misclass_df):
    """Check if confused labels are biologically related."""
    print("\n" + "="*80)
    print("HYPOTHESIS 2: Biologically Similar Materials")
    print("="*80)
    
    # Define biological groupings
    oral_related = ['dental calculus', 'Oral', 'tooth']
    skeletal = ['bone', 'tooth', 'dental calculus']
    skin_related = ['Skin', 'soft_tissue']
    environmental = ['sediment', 'soil', 'permafrost', 'lake sediment', 'marine sediment']
    
    groupings = {
        'Oral cavity': oral_related,
        'Skeletal tissue': skeletal,
        'Skin/soft tissue': skin_related,
        'Environmental': environmental
    }
    
    print("\nBiological groupings:")
    for group, materials in groupings.items():
        print(f"  {group}: {', '.join(materials)}")
    
    print("\nAnalyzing confusion patterns:")
    total_within_group = 0
    total_across_group = 0
    
    for _, row in misclass_df.iterrows():
        true_label = row['material_true']
        pred_label = row['material_pred']
        
        # Find which group(s) each label belongs to
        true_groups = [g for g, mats in groupings.items() if true_label in mats]
        pred_groups = [g for g, mats in groupings.items() if pred_label in mats]
        
        # Check if within same biological group
        if any(g in pred_groups for g in true_groups):
            total_within_group += 1
        else:
            total_across_group += 1
    
    total = len(misclass_df)
    pct_within = total_within_group / total * 100
    pct_across = total_across_group / total * 100
    
    print(f"\nConfusions within same biological group: {total_within_group}/{total} ({pct_within:.1f}%)")
    print(f"Confusions across different groups: {total_across_group}/{total} ({pct_across:.1f}%)")
    
    if pct_within > 60:
        print("\n⚠ WARNING: Most errors are within biologically similar groups!")
        print("   This suggests the materials are genuinely hard to distinguish,")
        print("   possibly due to similar microbial communities or DNA preservation.")

def analyze_label_quality(misclass_df, val_meta):
    """Check for potential label quality issues."""
    print("\n" + "="*80)
    print("HYPOTHESIS 3: Label Quality Issues")
    print("="*80)
    
    # misclass_df already has metadata merged
    
    print("\nChecking for metadata inconsistencies...")
    
    # Check dental calculus samples - should all be from Homo sapiens
    dental = misclass_df[misclass_df['material_true'] == 'dental calculus']
    if len(dental) > 0:
        print(f"\nDental calculus samples (n={len(dental)}):")
        if 'sample_host' in dental.columns:
            print(f"  Sample hosts: {dental['sample_host'].value_counts().to_dict()}")
            non_human = dental[dental['sample_host'] != 'Homo sapiens']
            if len(non_human) > 0:
                print(f"  ⚠ {len(non_human)} dental calculus samples NOT from Homo sapiens - possible mislabeling!")
    
    # Check Oral/Skin samples - should be from specific hosts
    oral = misclass_df[misclass_df['material_true'] == 'Oral']
    if len(oral) > 0:
        print(f"\nOral samples (n={len(oral)}):")
        if 'sample_host' in oral.columns:
            print(f"  Sample hosts: {oral['sample_host'].value_counts().head(5).to_dict()}")
        if 'community_type' in oral.columns:
            print(f"  Community types: {oral['community_type'].value_counts().to_dict()}")
    
    # Check environmental materials - should be env samples
    env_materials = ['sediment', 'soil', 'permafrost']
    for mat in env_materials:
        mat_samples = misclass_df[misclass_df['material_true'] == mat]
        if len(mat_samples) > 0 and 'sample_host' in mat_samples.columns:
            env_labeled = (mat_samples['sample_host'] == 'Not applicable - env sample').sum()
            pct = env_labeled / len(mat_samples) * 100
            if pct < 80:
                print(f"\n⚠ {mat} samples: only {pct:.0f}% labeled as env samples")

def analyze_confidence_patterns(misclass_df):
    """Analyze confidence score patterns."""
    print("\n" + "="*80)
    print("HYPOTHESIS 4: Low Confidence = Uncertain Predictions")
    print("="*80)
    
    print(f"\nConfidence distribution for misclassified samples:")
    print(f"  Mean: {misclass_df['material_confidence'].mean():.1%}")
    print(f"  Median: {misclass_df['material_confidence'].median():.1%}")
    print(f"  Std: {misclass_df['material_confidence'].std():.1%}")
    
    # Binned analysis
    bins = [0, 0.3, 0.5, 0.7, 0.9, 1.0]
    labels = ['<30%', '30-50%', '50-70%', '70-90%', '>90%']
    misclass_df['conf_bin'] = pd.cut(misclass_df['material_confidence'], bins=bins, labels=labels)
    
    print(f"\nConfidence distribution:")
    for label in labels:
        count = (misclass_df['conf_bin'] == label).sum()
        pct = count / len(misclass_df) * 100
        print(f"  {label}: {count} ({pct:.1f}%)")
    
    high_conf_errors = misclass_df[misclass_df['material_confidence'] > 0.7]
    print(f"\n⚠ High-confidence errors (>70%): {len(high_conf_errors)} ({len(high_conf_errors)/len(misclass_df)*100:.1f}%)")
    print("   These are the most problematic - model is confidently wrong!")
    
    if len(high_conf_errors) > 0:
        print(f"\nTop high-confidence error patterns:")
        patterns = high_conf_errors.groupby(['material_true', 'material_pred']).size().sort_values(ascending=False)
        for (true_label, pred_label), count in patterns.head(5).items():
            print(f"  {true_label} → {pred_label}: {count} samples")

def analyze_project_bias(misclass_df):
    """Check if errors concentrate in specific projects."""
    print("\n" + "="*80)
    print("HYPOTHESIS 5: Project-Specific Issues")
    print("="*80)
    
    print("\nProjects with most misclassifications:")
    project_errors = misclass_df['project_name'].value_counts().head(10)
    for project, count in project_errors.items():
        pct = count / len(misclass_df) * 100
        print(f"  {project}: {count} errors ({pct:.1f}% of all errors)")
    
    # Check if certain projects have systematic biases
    print("\nSystematic biases by project (top 3):")
    for project in project_errors.head(3).index:
        proj_errors = misclass_df[misclass_df['project_name'] == project]
        patterns = proj_errors.groupby(['material_true', 'material_pred']).size().sort_values(ascending=False)
        print(f"\n  {project} (n={len(proj_errors)}):")
        for (true_label, pred_label), count in patterns.head(3).items():
            print(f"    {true_label} → {pred_label}: {count}")

def main():
    # Load data
    print("Loading data...")
    misclass_df = pd.read_csv('results/material_seen_misclassifications_validation.tsv', sep='\t')
    val_pred = pd.read_csv('results/validation_predictions/validation_predictions.tsv', sep='\t')
    val_meta = pd.read_csv('paper/metadata/validation_metadata.tsv', sep='\t')
    
    # Get correctly classified samples for comparison
    correct_df = val_pred[val_pred['material_pred'] == val_pred['material_true']]
    
    print(f"\nDataset sizes:")
    print(f"  Total validation: {len(val_pred)}")
    print(f"  Correct Material predictions: {len(correct_df)} ({len(correct_df)/len(val_pred)*100:.1f}%)")
    print(f"  SEEN misclassifications: {len(misclass_df)} ({len(misclass_df)/len(val_pred)*100:.1f}%)")
    
    # Run analyses
    analyze_sparsity(misclass_df, correct_df)
    analyze_biological_similarity(misclass_df)
    analyze_label_quality(misclass_df, val_meta)
    analyze_confidence_patterns(misclass_df)
    analyze_project_bias(misclass_df)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY & RECOMMENDATIONS")
    print("="*80)
    print("""
Based on the analyses above, the main causes of Material task errors appear to be:

1. If sparsity is higher for misclassified samples:
   → Insufficient unitig coverage - samples may need more sequencing depth
   
2. If confusions are within biological groups:
   → Materials are genuinely similar (e.g., dental calculus contains oral microbiome)
   → Consider merging similar categories or using hierarchical classification
   
3. If metadata inconsistencies found:
   → Label quality issues - review and correct training/validation labels
   
4. If high-confidence errors are common:
   → Model learned wrong patterns - may need feature engineering or retraining
   
5. If errors concentrate in specific projects:
   → Batch effects or systematic differences in sample preparation/sequencing
   → Consider adding project as a covariate or using domain adaptation
    """)

if __name__ == '__main__':
    main()
