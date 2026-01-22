#!/usr/bin/env python3
"""
Cross-task consistency checker for DIANA predictions.

Flags biologically implausible prediction combinations across the 4 tasks,
regardless of individual task confidence scores.
"""

import argparse
import polars as pl
from pathlib import Path
from typing import Dict, Tuple, Set


# Define material categories (blacklist approach)
# Note: Oral/Skin removed - they can appear in both environmental and host contexts
ENVIRONMENTAL_MATERIALS = {
    'sediment', 'permafrost', 'soil', 'midden', 'shell', 
    # Validation set additions
    'lake sediment', 'marine sediment', 'shallow marine sediment',
    'palaeofaeces'
}

HOST_MATERIALS = {
    'dental calculus', 'bone', 'tooth', 'soft_tissue', 
    'digestive_contents',
    # Validation set additions
    'dentine', 'brain', 'birch pitch', 'gut', 'intestine'
}

# Materials that specifically require oral context
ORAL_MATERIALS = {'dental calculus', 'tooth'}

# Host-associated communities
HOST_COMMUNITIES = {'oral', 'gut', 'skeletal tissue', 'soft tissue', 'plant tissue', 'Skin'}


def is_env_host(host: str) -> bool:
    """Check if host is environmental (not an organism)."""
    return 'env sample' in host or host == 'environmental'


def is_env_community(community: str) -> bool:
    """Check if community is environmental."""
    return 'env sample' in community or community == 'environmental'


def check_consistency(
    material: str,
    community: str, 
    host: str,
    verbose: bool = False
) -> Tuple[bool, str]:
    """
    Check if combination of predictions is biologically consistent.
    Uses BLACKLIST approach - flags only impossible combinations.
    
    Returns:
        (is_consistent, reason_if_inconsistent)
    """
    reasons = []
    
    # Rule 1: Environmental materials cannot come from animal/human hosts
    if material in ENVIRONMENTAL_MATERIALS and not is_env_host(host):
        reasons.append(f"Environmental material '{material}' cannot come from organism '{host}'")
    
    # Rule 2: Host-associated materials cannot come from environmental samples
    if material in HOST_MATERIALS and is_env_host(host):
        reasons.append(f"Host material '{material}' cannot come from environmental sample")
    
    # Rule 3: Dental calculus specifically requires oral community (unless in skeletal context)
    if material == 'dental calculus' and is_env_community(community):
        reasons.append(f"Dental calculus cannot be in environmental community '{community}'")
    
    # Rule 4: Environmental materials cannot be in host-associated communities
    if material in ENVIRONMENTAL_MATERIALS and community in HOST_COMMUNITIES:
        reasons.append(f"Environmental material '{material}' cannot be in host community '{community}'")
    
    # Rule 5: Environmental hosts must have environmental community
    if is_env_host(host) and not is_env_community(community):
        reasons.append(f"Environmental host requires environmental community, got '{community}'")
    
    if reasons:
        return False, "; ".join(reasons)
    return True, ""


def analyze_test_set(predictions_file: Path, output_dir: Path):
    """Analyze test set predictions for cross-task consistency."""
    
    print(f"\n=== TEST SET CONSISTENCY ANALYSIS ===")
    print(f"Loading: {predictions_file}")
    
    df = pl.read_csv(predictions_file, separator='\t')
    
    # Extract predicted labels
    results = []
    for row in df.iter_rows(named=True):
        sample_id = row['Run_accession']
        
        # Predicted labels
        material_pred = row['material_pred']
        community_pred = row['community_type_pred']
        host_pred = row['sample_host_pred']
        
        # True labels
        material_true = row['material_true']
        community_true = row['community_type_true']
        host_true = row['sample_host_true']
        
        # Check consistency
        is_consistent, reason = check_consistency(material_pred, community_pred, host_pred)
        
        # Check if prediction was correct
        material_correct = material_pred == material_true
        community_correct = community_pred == community_true
        host_correct = host_pred == host_true
        any_wrong = not (material_correct and community_correct and host_correct)
        
        results.append({
            'sample_id': sample_id,
            'material_pred': material_pred,
            'community_pred': community_pred,
            'host_pred': host_pred,
            'material_true': material_true,
            'community_true': community_true,
            'host_true': host_true,
            'is_consistent': is_consistent,
            'inconsistency_reason': reason if not is_consistent else '',
            'material_correct': material_correct,
            'community_correct': community_correct,
            'host_correct': host_correct,
            'any_wrong': any_wrong,
        })
    
    results_df = pl.DataFrame(results)
    
    # Calculate statistics
    total = len(results_df)
    inconsistent = results_df.filter(pl.col('is_consistent') == False)
    n_inconsistent = len(inconsistent)
    
    # Of inconsistent predictions, how many had errors?
    if n_inconsistent > 0:
        inconsistent_with_errors = inconsistent.filter(pl.col('any_wrong'))
        n_inconsistent_with_errors = len(inconsistent_with_errors)
        
        print(f"\nTotal samples: {total}")
        print(f"Inconsistent predictions: {n_inconsistent} ({n_inconsistent/total*100:.1f}%)")
        print(f"  └─ With errors: {n_inconsistent_with_errors} ({n_inconsistent_with_errors/n_inconsistent*100:.1f}%)")
        print(f"  └─ All correct: {n_inconsistent - n_inconsistent_with_errors}")
        
        # Precision: If flagged as inconsistent, what's the chance it's actually wrong?
        precision = n_inconsistent_with_errors / n_inconsistent * 100
        print(f"\nPrecision: {precision:.1f}% (flagged → actually wrong)")
        
        # What errors did inconsistent predictions have?
        print("\nError breakdown for inconsistent predictions:")
        print(f"  Material errors: {inconsistent_with_errors.filter(~pl.col('material_correct')).height}")
        print(f"  Community errors: {inconsistent_with_errors.filter(~pl.col('community_correct')).height}")
        print(f"  Host errors: {inconsistent_with_errors.filter(~pl.col('host_correct')).height}")
    
    # Save results
    output_dir.mkdir(exist_ok=True, parents=True)
    output_file = output_dir / 'test_consistency_check.tsv'
    results_df.write_csv(output_file, separator='\t')
    print(f"\nSaved: {output_file}")
    
    # Save flagged samples
    if n_inconsistent > 0:
        flagged_file = output_dir / 'test_flagged_inconsistent.tsv'
        inconsistent.write_csv(flagged_file, separator='\t')
        print(f"Saved flagged samples: {flagged_file}")
    
    return results_df


def analyze_validation_set(predictions_file: Path, output_dir: Path):
    """Analyze validation set predictions for cross-task consistency."""
    
    print(f"\n=== VALIDATION SET CONSISTENCY ANALYSIS ===")
    print(f"Loading: {predictions_file}")
    
    # Load predictions (wide format: one row per sample with columns like material_pred, material_true)
    df = pl.read_csv(predictions_file, separator='\t')
    
    print(f"Processing {len(df)} samples...")
    
    results = []
    for row in df.iter_rows(named=True):
        sample_id = row.get('sample_id', row.get('Run_accession', 'UNKNOWN'))
        
        # Extract predictions and true labels
        material_pred = row.get('material_pred', 'UNKNOWN')
        community_pred = row.get('community_type_pred', 'UNKNOWN')
        host_pred = row.get('sample_host_pred', 'UNKNOWN')
        
        material_true = row.get('material_true', 'UNKNOWN')
        community_true = row.get('community_type_true', 'UNKNOWN')
        host_true = row.get('sample_host_true', 'UNKNOWN')
        
        # Check consistency
        is_consistent, reason = check_consistency(material_pred, community_pred, host_pred)
        
        # Check if predictions were correct
        material_correct = material_pred == material_true
        community_correct = community_pred == community_true
        host_correct = host_pred == host_true
        any_wrong = not (material_correct and community_correct and host_correct)
        
        results.append({
            'sample_id': sample_id,
            'material_pred': material_pred,
            'community_pred': community_pred,
            'host_pred': host_pred,
            'material_true': material_true,
            'community_true': community_true,
            'host_true': host_true,
            'is_consistent': is_consistent,
            'inconsistency_reason': reason if not is_consistent else '',
            'material_correct': material_correct,
            'community_correct': community_correct,
            'host_correct': host_correct,
            'any_wrong': any_wrong,
        })
    
    results_df = pl.DataFrame(results)
    
    # Calculate statistics
    total = len(results_df)
    inconsistent = results_df.filter(pl.col('is_consistent') == False)
    n_inconsistent = len(inconsistent)
    
    # Of inconsistent predictions, how many had errors?
    if n_inconsistent > 0:
        inconsistent_with_errors = inconsistent.filter(pl.col('any_wrong'))
        n_inconsistent_with_errors = len(inconsistent_with_errors)
        
        print(f"\nTotal samples: {total}")
        print(f"Inconsistent predictions: {n_inconsistent} ({n_inconsistent/total*100:.1f}%)")
        print(f"  └─ With errors: {n_inconsistent_with_errors} ({n_inconsistent_with_errors/n_inconsistent*100:.1f}%)")
        print(f"  └─ All correct: {n_inconsistent - n_inconsistent_with_errors}")
        
        # Precision: If flagged as inconsistent, what's the chance it's actually wrong?
        precision = n_inconsistent_with_errors / n_inconsistent * 100
        print(f"\nPrecision: {precision:.1f}% (flagged → actually wrong)")
        
        # What errors did inconsistent predictions have?
        print("\nError breakdown for inconsistent predictions:")
        print(f"  Material errors: {inconsistent_with_errors.filter(~pl.col('material_correct')).height}")
        print(f"  Community errors: {inconsistent_with_errors.filter(~pl.col('community_correct')).height}")
        print(f"  Host errors: {inconsistent_with_errors.filter(~pl.col('host_correct')).height}")
    else:
        print(f"\nTotal samples: {total}")
        print(f"No inconsistent predictions found! All {total} samples have biologically plausible combinations.")
    
    # Save results
    output_dir.mkdir(exist_ok=True, parents=True)
    output_file = output_dir / 'validation_consistency_check.tsv'
    results_df.write_csv(output_file, separator='\t')
    print(f"\nSaved: {output_file}")
    
    # Save flagged samples
    if n_inconsistent > 0:
        flagged_file = output_dir / 'validation_flagged_inconsistent.tsv'
        inconsistent.write_csv(flagged_file, separator='\t')
        print(f"Saved flagged samples: {flagged_file}")
    
    return results_df


def main():
    parser = argparse.ArgumentParser(description='Cross-task consistency checker')
    parser.add_argument('--test-predictions', type=Path,
                       default=Path('results/test_evaluation/test_predictions.tsv'),
                       help='Test set predictions file')
    parser.add_argument('--validation-predictions', type=Path,
                       default=Path('results/mc_dropout_validation_with_unknown.tsv'),
                       help='Validation set predictions file')
    parser.add_argument('--output-dir', type=Path,
                       default=Path('results/consistency_check'),
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Analyze test set
    test_results = analyze_test_set(args.test_predictions, args.output_dir)
    
    # Analyze validation set
    val_results = analyze_validation_set(args.validation_predictions, args.output_dir)


if __name__ == '__main__':
    main()
