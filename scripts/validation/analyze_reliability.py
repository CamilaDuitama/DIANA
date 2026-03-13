#!/usr/bin/env python3
"""
Analyze relationship between sparsity, confidence, and prediction errors.
Propose reliability thresholds for flagging uncertain predictions.
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Load all data
print("Loading data...")
wrong_df = pd.read_csv("results/wrong_predictions_analysis.tsv", sep='\t')
sparsity_df = pd.read_csv("results/unitig_sparsity_analysis.tsv", sep='\t')
coverage_df = pd.read_csv("results/kmer_coverage_analysis.tsv", sep='\t')

# Merge datasets
merged = wrong_df.merge(sparsity_df, on='sample_id', how='left')
merged = merged.merge(coverage_df[['sample_id', 'unique_kmers', 'coverage']], on='sample_id', how='left')

print(f"\nTotal predictions: {len(merged)}")
print(f"Wrong predictions: {len(merged[~merged['is_correct']])}")

# Analyze confidence by correctness
print("\n" + "=" * 80)
print("CONFIDENCE ANALYSIS")
print("=" * 80)

correct = merged[merged['is_correct']]
wrong = merged[~merged['is_correct']]

print(f"\nConfidence scores:")
print(f"  Correct predictions: {correct['confidence'].mean():.2%} ± {correct['confidence'].std():.2%}")
print(f"  Wrong predictions:   {wrong['confidence'].mean():.2%} ± {wrong['confidence'].std():.2%}")

# Check high-confidence wrong predictions
high_conf_threshold = 0.9
high_conf_wrong = wrong[wrong['confidence'] >= high_conf_threshold]
print(f"\nHigh-confidence (≥{high_conf_threshold:.0%}) wrong predictions: {len(high_conf_wrong)} ({len(high_conf_wrong)/len(wrong)*100:.1f}% of errors)")

if len(high_conf_wrong) > 0:
    print(f"  Mean sparsity: {high_conf_wrong['sparsity'].mean():.2%}")
    print(f"  Mean non-zero features: {high_conf_wrong['nonzero_features'].mean():.0f}")
    print(f"  Mean entropy: {high_conf_wrong['entropy'].mean():.2f}")
    
    print(f"\n  Top tasks with high-confidence errors:")
    for task in high_conf_wrong['task'].value_counts().head():
        task_name = high_conf_wrong['task'].value_counts().index[high_conf_wrong['task'].value_counts().tolist().index(task)]
        print(f"    {task_name}: {task}")

# Relationship between sparsity and confidence for WRONG predictions
print("\n" + "=" * 80)
print("SPARSITY vs CONFIDENCE (for wrong predictions)")
print("=" * 80)

if len(wrong) > 0:
    corr_sparsity = stats.pearsonr(wrong['sparsity'].dropna(), wrong['confidence'].dropna())
    corr_features = stats.pearsonr(wrong['nonzero_features'].dropna(), wrong['confidence'].dropna())
    corr_entropy = stats.pearsonr(wrong['entropy'].dropna(), wrong['confidence'].dropna())
    
    print(f"\nCorrelations for wrong predictions:")
    print(f"  Sparsity vs Confidence: r={corr_sparsity[0]:.3f}, p={corr_sparsity[1]:.4e}")
    print(f"  Non-zero features vs Confidence: r={corr_features[0]:.3f}, p={corr_features[1]:.4e}")
    print(f"  Entropy vs Confidence: r={corr_entropy[0]:.3f}, p={corr_entropy[1]:.4e}")

# Propose reliability thresholds
print("\n" + "=" * 80)
print("PROPOSED RELIABILITY WARNING THRESHOLDS")
print("=" * 80)

# Test different thresholds
thresholds = {
    'very_sparse': merged['sparsity'] > 0.995,
    'low_features': merged['nonzero_features'] < 5000,
    'low_entropy': merged['entropy'] < 8,
    'low_kmers': merged['unique_kmers'] < 10000,
    'combined_strict': (merged['sparsity'] > 0.995) | (merged['nonzero_features'] < 1000),
    'combined_moderate': (merged['sparsity'] > 0.99) | (merged['nonzero_features'] < 5000) | (merged['entropy'] < 8),
}

print("\nThreshold performance:")
print(f"{'Threshold':<25} {'Flagged':<10} {'Error Rate':<12} {'% of Errors Caught':<20}")
print("-" * 80)

for name, mask in thresholds.items():
    flagged = merged[mask]
    if len(flagged) > 0:
        error_rate = (~flagged['is_correct']).sum() / len(flagged)
        errors_caught = (~flagged['is_correct']).sum() / (~merged['is_correct']).sum()
        print(f"{name:<25} {len(flagged):<10} {error_rate:<12.1%} {errors_caught:<20.1%}")

# Recommended threshold
recommended_mask = (merged['sparsity'] > 0.99) | (merged['nonzero_features'] < 5000)
print(f"\n✓ RECOMMENDED: Flag samples with sparsity >99% OR <5,000 non-zero features")
print(f"  This flags {recommended_mask.sum()} samples ({recommended_mask.sum()/len(merged)*100:.1f}%)")
print(f"  Error rate in flagged: {(~merged[recommended_mask]['is_correct']).sum()/recommended_mask.sum()*100:.1f}%")
print(f"  Catches {(~merged[recommended_mask]['is_correct']).sum()/(~merged['is_correct']).sum()*100:.1f}% of all errors")

# Check for potential mislabeling (high confidence + high quality features)
print("\n" + "=" * 80)
print("POTENTIAL MISLABELING CANDIDATES")
print("=" * 80)

# These are samples where the model is confident AND has good feature quality
potential_mislabel = wrong[
    (wrong['confidence'] >= 0.85) &  # High confidence
    (wrong['sparsity'] < 0.90) &      # Low sparsity (good features)
    (wrong['nonzero_features'] > 10000) &  # Many features
    (wrong['entropy'] > 10)           # Good diversity
]

print(f"\nHigh-quality wrong predictions (potential mislabeling): {len(potential_mislabel)}")
if len(potential_mislabel) > 0:
    print(f"  Mean confidence: {potential_mislabel['confidence'].mean():.2%}")
    print(f"  Mean sparsity: {potential_mislabel['sparsity'].mean():.2%}")
    print(f"  Mean non-zero features: {potential_mislabel['nonzero_features'].mean():.0f}")
    print(f"\n  Breakdown by task:")
    for task, count in potential_mislabel['task'].value_counts().items():
        print(f"    {task}: {count}")
    
    # Save for manual review
    potential_mislabel.to_csv("results/potential_mislabeling.tsv", sep='\t', index=False)
    print(f"\n  ✓ Saved to: results/potential_mislabeling.tsv")

# Save flagged samples
flagged_samples = merged[recommended_mask].copy()
flagged_samples['warning_reason'] = ''
flagged_samples.loc[flagged_samples['sparsity'] > 0.99, 'warning_reason'] += 'high_sparsity;'
flagged_samples.loc[flagged_samples['nonzero_features'] < 5000, 'warning_reason'] += 'low_features;'
flagged_samples['warning_reason'] = flagged_samples['warning_reason'].str.rstrip(';')

flagged_samples.to_csv("results/reliability_warnings.tsv", sep='\t', index=False)
print(f"\n✓ Saved flagged samples to: results/reliability_warnings.tsv")

print("\n" + "=" * 80)
