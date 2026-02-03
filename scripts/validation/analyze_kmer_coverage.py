#!/usr/bin/env python3
"""
Analyze k-mer coverage by checking kmer_counts.txt files in parallel.
Compare coverage between correct and wrong predictions.
"""

import pandas as pd
from pathlib import Path
from scipy import stats
import concurrent.futures
from tqdm import tqdm  # For a nice progress bar
import os

# --- Global variable for worker processes ---
# This avoids passing the huge set of reference k-mers to each worker repeatedly.
# It will be initialized once per worker process.
ref_kmers_global = set()

def init_worker(ref_kmers_main_set):
    """Initializer function for each worker process."""
    global ref_kmers_global
    ref_kmers_global = ref_kmers_main_set

def process_kmer_file(sample_id: str) -> dict | None:
    """
    Analyzes a single k-mer count file for a given sample_id.
    This function will be run in parallel by multiple processes.
    """
    kmer_file = Path(f"results/validation_predictions/{sample_id}/{sample_id}_kmer_counts.txt")
    
    if not kmer_file.exists():
        return None

    try:
        sample_kmers = set()
        total_count = 0
        with open(kmer_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    kmer = parts[0]
                    count = int(parts[1])
                    sample_kmers.add(kmer)
                    total_count += count
    except Exception as e:
        print(f"⚠ Error processing {sample_id}: {e}")
        return None
    
    num_unique_kmers = len(sample_kmers)
    
    if ref_kmers_global:
        matched_kmers = len(sample_kmers.intersection(ref_kmers_global))
        coverage = matched_kmers / num_unique_kmers if num_unique_kmers > 0 else 0
    else:
        matched_kmers = None
        coverage = None
        
    return {
        'sample_id': sample_id,
        'total_kmer_count': total_count,
        'unique_kmers': num_unique_kmers,
        'matched_to_matrix': matched_kmers,
        'coverage': coverage
    }

def main():
    """Main function to orchestrate the analysis."""
    # Load wrong predictions
    wrong_df = pd.read_csv("results/wrong_predictions_analysis.tsv", sep='\t')
    wrong_samples = set(wrong_df['sample_id'].unique())

    # Load all validation samples  
    pred_df = pd.read_csv("results/validation_predictions/validation_predictions.tsv", sep='\t')
    all_samples = list(pred_df['sample_id'].unique()) # Use a list for indexing

    # Load reference k-mers (the unitig matrix features)
    print("Loading reference k-mers from matrix...")
    ref_kmers_file = Path("reference_kmers.fasta")
    ref_kmers = set()
    if ref_kmers_file.exists():
        with open(ref_kmers_file) as f:
            for line in f:
                if not line.startswith('>'):
                    ref_kmers.add(line.strip())
        print(f"✓ Loaded {len(ref_kmers)} reference unitig k-mers")
    else:
        print("⚠ reference_kmers.fasta not found")
        print("  Will only count total k-mers per sample")

    print("\n" + "=" * 80)
    print("K-MER COVERAGE ANALYSIS (IN PARALLEL)")
    print("=" * 80)

    coverage_data = []
    
    # Use as many workers as there are CPU cores, but don't overdo it.
    num_workers = min(16, os.cpu_count() or 1)
    print(f"\nAnalyzing k-mer files for {len(all_samples)} samples using {num_workers} workers...")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers, initializer=init_worker, initargs=(ref_kmers,)) as executor:
        # Use executor.map to apply the function to all samples in parallel
        # tqdm gives us a nice progress bar
        results = list(tqdm(executor.map(process_kmer_file, all_samples), total=len(all_samples)))

    # Filter out None results (for files that didn't exist)
    coverage_data = [r for r in results if r is not None]

    print(f"\n✓ Analyzed {len(coverage_data)} samples")

    if not coverage_data:
        print("\n✗ No k-mer count files found")
        return
        
    coverage_df = pd.DataFrame(coverage_data)
    coverage_df['has_error'] = coverage_df['sample_id'].isin(wrong_samples)
    
    # --- The rest of your analysis script is perfect and remains unchanged ---
    
    print(f"\nK-mer statistics:")
    print(f"  Unique k-mers per sample: {coverage_df['unique_kmers'].median():.0f} (median)")
    print(f"  Total k-mer count per sample: {coverage_df['total_kmer_count'].median():.0f} (median)")
    
    if ref_kmers:
        print(f"\nCoverage (k-mers matching unitig matrix):")
        wrong_cov = coverage_df[coverage_df['has_error']]['coverage'].dropna()
        correct_cov = coverage_df[~coverage_df['has_error']]['coverage'].dropna()
        
        if not correct_cov.empty:
             print(f"  Correct predictions: {correct_cov.mean():.2%} ± {correct_cov.std():.2%}")
        if not wrong_cov.empty:
            print(f"  Wrong predictions:   {wrong_cov.mean():.2%} ± {wrong_cov.std():.2%}")
        
        if len(wrong_cov) > 1 and len(correct_cov) > 1:
            t_stat, p_value = stats.ttest_ind(wrong_cov, correct_cov, equal_var=False) # Welch's t-test
            print(f"  t-test p-value: {p_value:.4e}")
            if p_value < 0.05:
                diff = wrong_cov.mean() - correct_cov.mean()
                print(f"  → Significant! Wrong predictions have {abs(diff):.2%} {'higher' if diff > 0 else 'lower'} coverage")
        
        low_cov_threshold = 0.01
        low_cov = coverage_df[coverage_df['coverage'] < low_cov_threshold]
        print(f"\nSamples with <{low_cov_threshold:.0%} k-mer coverage: {len(low_cov)}")
        if len(low_cov) > 0:
            wrong_low = len(low_cov[low_cov['has_error']])
            print(f"  With errors: {wrong_low}/{len(low_cov)} ({wrong_low/len(low_cov):.1%})")
    
    print(f"\nLow k-mer count analysis:")
    for threshold in [1000, 5000, 10000]:
        low_count = coverage_df[coverage_df['unique_kmers'] < threshold]
        if len(low_count) > 0:
            wrong_low = len(low_count[low_count['has_error']])
            print(f"  <{threshold} unique k-mers: {len(low_count)} samples, {wrong_low} with errors ({wrong_low/len(low_count):.1%})")
    
    output_file = "results/kmer_coverage_analysis.tsv"
    coverage_df.to_csv(output_file, sep='\t', index=False)
    print(f"\n✓ Saved detailed analysis to: {output_file}")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()