#!/usr/bin/env python3
"""
Analyze unitig fraction sparsity for correct vs wrong predictions.
This analyzes the actual feature vectors that enter the model.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import concurrent.futures
from tqdm import tqdm
import os

def process_unitig_fraction(sample_id: str) -> dict | None:
    """
    Analyzes a single unitig fraction file for a given sample_id.
    Returns sparsity statistics for the feature vector.
    """
    unitig_file = Path(f"results/validation_predictions/{sample_id}/{sample_id}_unitig_fraction.txt")
    
    if not unitig_file.exists():
        return None

    try:
        # Read all unitig fractions
        fractions = []
        with open(unitig_file) as f:
            for line in f:
                fractions.append(float(line.strip()))
        
        fractions = np.array(fractions)
        total_features = len(fractions)
        
        # Sparsity statistics
        nonzero_features = np.count_nonzero(fractions)
        sparsity = 1.0 - (nonzero_features / total_features)
        
        # Distribution statistics for non-zero features
        nonzero_fractions = fractions[fractions > 0]
        mean_nonzero = np.mean(nonzero_fractions) if len(nonzero_fractions) > 0 else 0.0
        std_nonzero = np.std(nonzero_fractions) if len(nonzero_fractions) > 0 else 0.0
        max_fraction = np.max(fractions)
        
        # Entropy as measure of feature distribution
        # Higher entropy = more evenly distributed features
        # Lower entropy = few dominant features
        if len(nonzero_fractions) > 0:
            # Normalize non-zero fractions to get probability distribution
            prob = nonzero_fractions / np.sum(nonzero_fractions)
            entropy = -np.sum(prob * np.log2(prob + 1e-10))
        else:
            entropy = 0.0
        
        return {
            'sample_id': sample_id,
            'total_features': total_features,
            'nonzero_features': nonzero_features,
            'sparsity': sparsity,
            'mean_nonzero_fraction': mean_nonzero,
            'std_nonzero_fraction': std_nonzero,
            'max_fraction': max_fraction,
            'entropy': entropy
        }
    except Exception as e:
        print(f"⚠ Error processing {sample_id}: {e}")
        return None

def main():
    """Main function to orchestrate the analysis."""
    # Load wrong predictions
    wrong_df = pd.read_csv("results/wrong_predictions_analysis.tsv", sep='\t')
    wrong_samples = set(wrong_df['sample_id'].unique())

    # Load all validation samples  
    pred_df = pd.read_csv("results/validation_predictions/validation_predictions.tsv", sep='\t')
    all_samples = list(pred_df['sample_id'].unique())

    print("=" * 80)
    print("UNITIG FRACTION SPARSITY ANALYSIS (MODEL INPUT)")
    print("=" * 80)

    sparsity_data = []
    
    # Use parallel processing
    num_workers = min(16, os.cpu_count() or 1)
    print(f"\nAnalyzing unitig fraction files for {len(all_samples)} samples using {num_workers} workers...")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(executor.map(process_unitig_fraction, all_samples), total=len(all_samples)))

    # Filter out None results
    sparsity_data = [r for r in results if r is not None]

    print(f"\n✓ Analyzed {len(sparsity_data)} samples")

    if not sparsity_data:
        print("\n✗ No unitig fraction files found")
        return
        
    sparsity_df = pd.DataFrame(sparsity_data)
    sparsity_df['has_error'] = sparsity_df['sample_id'].isin(wrong_samples)
    
    # Overall statistics
    print(f"\nFeature vector statistics:")
    print(f"  Total features per sample: {sparsity_df['total_features'].iloc[0]:.0f}")
    print(f"  Non-zero features (median): {sparsity_df['nonzero_features'].median():.0f}")
    print(f"  Sparsity (median): {sparsity_df['sparsity'].median():.2%}")
    
    # Compare correct vs wrong predictions
    print(f"\n" + "=" * 80)
    print("COMPARISON: CORRECT vs WRONG PREDICTIONS")
    print("=" * 80)
    
    wrong_data = sparsity_df[sparsity_df['has_error']]
    correct_data = sparsity_df[~sparsity_df['has_error']]
    
    metrics = [
        ('nonzero_features', 'Non-zero features'),
        ('sparsity', 'Sparsity'),
        ('mean_nonzero_fraction', 'Mean non-zero fraction'),
        ('max_fraction', 'Max fraction'),
        ('entropy', 'Feature entropy')
    ]
    
    for metric, label in metrics:
        wrong_vals = wrong_data[metric].dropna()
        correct_vals = correct_data[metric].dropna()
        
        if len(wrong_vals) > 1 and len(correct_vals) > 1:
            print(f"\n{label}:")
            print(f"  Correct predictions: {correct_vals.mean():.4f} ± {correct_vals.std():.4f}")
            print(f"  Wrong predictions:   {wrong_vals.mean():.4f} ± {wrong_vals.std():.4f}")
            
            t_stat, p_value = stats.ttest_ind(wrong_vals, correct_vals, equal_var=False)
            print(f"  t-test p-value: {p_value:.4e}")
            
            if p_value < 0.05:
                diff = wrong_vals.mean() - correct_vals.mean()
                direction = 'higher' if diff > 0 else 'lower'
                print(f"  → Significant! Wrong predictions have {abs(diff):.4f} {direction} {label.lower()}")
    
    # Check extreme cases
    print(f"\n" + "=" * 80)
    print("EXTREME CASES")
    print("=" * 80)
    
    # Very sparse samples
    sparse_threshold = 0.995  # >99.5% sparse
    very_sparse = sparsity_df[sparsity_df['sparsity'] > sparse_threshold]
    if len(very_sparse) > 0:
        wrong_sparse = len(very_sparse[very_sparse['has_error']])
        print(f"\nVery sparse samples (>{sparse_threshold*100:.1f}% sparse): {len(very_sparse)}")
        print(f"  With errors: {wrong_sparse}/{len(very_sparse)} ({wrong_sparse/len(very_sparse)*100:.1f}%)")
    
    # Low non-zero features
    for threshold in [100, 500, 1000, 5000]:
        low_features = sparsity_df[sparsity_df['nonzero_features'] < threshold]
        if len(low_features) > 0:
            wrong_low = len(low_features[low_features['has_error']])
            print(f"  <{threshold} non-zero features: {len(low_features)} samples, {wrong_low} with errors ({wrong_low/len(low_features)*100:.1f}%)")
    
    # Save results
    output_file = "results/unitig_sparsity_analysis.tsv"
    sparsity_df.to_csv(output_file, sep='\t', index=False)
    print(f"\n✓ Saved detailed analysis to: {output_file}")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
