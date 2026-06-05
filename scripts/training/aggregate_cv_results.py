#!/usr/bin/env python3
"""
Aggregate Cross-Validation Results from Multiple Folds
========================================================

Averages hyperparameters and metrics across all CV folds and creates
configuration files for final model training.

USAGE:
    python scripts/training/aggregate_cv_results.py \
        --cv_dir results/training_bioproject_v3/cv_results \
        --n_folds 5

OUTPUT:
    - cv_dir/best_hyperparameters.json: Averaged hyperparameters
    - cv_dir/aggregated_results.json: All fold metrics
    - cv_dir/../final_training_config.json: Config for final training
"""

import argparse
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any
from scipy import stats


def load_fold_results(cv_dir: Path, n_folds: int) -> List[Dict[str, Any]]:
    """Load results JSON from all folds."""
    results = []
    
    for fold_id in range(n_folds):
        fold_dir = cv_dir / f"fold_{fold_id}"
        
        # Find results JSON (has timestamp in filename)
        result_files = list(fold_dir.glob("multitask_fold_*_results_*.json"))
        
        if not result_files:
            raise FileNotFoundError(f"No results file found in {fold_dir}")
        
        # Take the most recent if multiple exist
        result_file = sorted(result_files, key=lambda p: p.stat().st_mtime)[-1]
        
        with open(result_file) as f:
            fold_results = json.load(f)
        
        results.append(fold_results)
        print(f"✓ Loaded fold {fold_id}: {result_file.name}")
    
    return results


def aggregate_hyperparameters(fold_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Average hyperparameters across folds.
    
    - Numeric parameters: mean
    - Categorical parameters: mode (most common value)
    - Boolean parameters: mode
    """
    params_by_key = defaultdict(list)
    
    # Collect all parameter values
    for result in fold_results:
        for key, value in result['best_params'].items():
            params_by_key[key].append(value)
    
    # Aggregate
    aggregated = {}
    
    for key, values in params_by_key.items():
        if isinstance(values[0], (int, float, np.number)):
            # Numeric: take mean
            aggregated[key] = float(np.mean(values))
        elif isinstance(values[0], bool):
            # Boolean: take mode (most common)
            aggregated[key] = float(stats.mode(values, keepdims=False)[0])
        else:
            # Categorical (string): take mode
            aggregated[key] = stats.mode(values, keepdims=False)[0]
    
    return aggregated


def aggregate_metrics(fold_results: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """
    Compute mean and std of test metrics across folds.
    """
    task_names = list(fold_results[0]['test_metrics'].keys())
    metric_names = list(fold_results[0]['test_metrics'][task_names[0]].keys())
    
    aggregated = {}
    
    for task in task_names:
        aggregated[task] = {}
        
        for metric in metric_names:
            values = [result['test_metrics'][task][metric] for result in fold_results]
            aggregated[task][metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values))
            }
    
    return aggregated


def create_final_training_config(
    cv_dir: Path,
    best_params: Dict[str, Any],
    features_path: str = "data/matrices/large_matrix_3070_with_frac/unitigs.frac.mat",
    metadata_path: str = "data/splits_bioproject/train_metadata.tsv",
    train_ids_path: str = "data/splits_bioproject/train_ids.txt"
) -> Dict[str, Any]:
    """
    Create configuration file for final model training.
    
    Format matches what 02_train_final_model.py expects.
    """
    # Extract model architecture params
    n_layers = int(round(best_params['n_layers']))
    hidden_dims = [int(round(best_params[f'hidden_dim_{i}'])) for i in range(n_layers)]
    
    # Extract task weights
    task_weights = {
        'sample_type': best_params['task_weight_sample_type'],
        'community_type': best_params['task_weight_community'],
        'sample_host': best_params['task_weight_host'],
        'material': best_params['task_weight_material']
    }
    
    # Extract label smoothing (if present, v3 has this)
    label_smoothing = {}
    if 'ls_sample_type' in best_params:
        label_smoothing = {
            'sample_type': best_params['ls_sample_type'],
            'community_type': best_params['ls_community_type'],
            'sample_host': best_params['ls_sample_host'],
            'material': best_params['ls_material']
        }
    
    config = {
        'features_path': features_path,
        'metadata_path': metadata_path,
        'train_ids_path': train_ids_path,
        'output_dir': str(cv_dir.parent),
        'hyperparameters': {
            'model_params': {
                'hidden_dims': hidden_dims,
                'dropout': best_params['dropout'],
                'activation': best_params['activation'],
                'use_batch_norm': bool(round(best_params['use_batch_norm']))
            },
            'trainer_params': {
                'learning_rate': best_params['learning_rate'],
                'weight_decay': best_params['weight_decay'],
                'task_weights': task_weights
            },
            'batch_size': int(round(best_params['batch_size']))
        },
        'task_names': ['sample_type', 'community_type', 'sample_host', 'material'],
        'task_weights': task_weights,  # Backward compatibility
        'validation_split': 0.1,
        'max_epochs': 200,
        'early_stopping_patience': 20,
        'random_seed': 42
    }
    
    # Add label smoothing if present (v3 only)
    if label_smoothing:
        config['label_smoothing_per_task'] = label_smoothing
    
    return config


def main():
    parser = argparse.ArgumentParser(description='Aggregate CV results from multiple folds')
    parser.add_argument('--cv_dir', type=Path, required=True,
                       help='Path to cv_results directory')
    parser.add_argument('--n_folds', type=int, default=5,
                       help='Number of CV folds')
    parser.add_argument('--features', type=str,
                       default='data/matrices/large_matrix_3070_with_frac/unitigs.frac.mat',
                       help='Features path for final training config')
    parser.add_argument('--metadata', type=str,
                       default='data/splits_bioproject/train_metadata.tsv',
                       help='Metadata path for final training config')
    parser.add_argument('--train_ids', type=str,
                       default='data/splits_bioproject/train_ids.txt',
                       help='Train IDs path for final training config')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Aggregating Cross-Validation Results")
    print("=" * 80)
    print(f"CV directory: {args.cv_dir}")
    print(f"Number of folds: {args.n_folds}")
    print()
    
    # Load results from all folds
    print("Step 1: Loading fold results...")
    fold_results = load_fold_results(args.cv_dir, args.n_folds)
    print(f"✓ Loaded {len(fold_results)} folds\n")
    
    # Aggregate hyperparameters
    print("Step 2: Aggregating hyperparameters...")
    best_params = aggregate_hyperparameters(fold_results)
    print("✓ Averaged hyperparameters:")
    for key, value in sorted(best_params.items()):
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")
    print()
    
    # Aggregate metrics
    print("Step 3: Aggregating test metrics...")
    aggregated_metrics = aggregate_metrics(fold_results)
    print("✓ Cross-validation performance (mean ± std):")
    for task, metrics in aggregated_metrics.items():
        print(f"\n  {task}:")
        for metric_name, values in metrics.items():
            print(f"    {metric_name}: {values['mean']:.4f} ± {values['std']:.4f}")
    print()
    
    # Save best hyperparameters
    print("Step 4: Saving results...")
    best_params_file = args.cv_dir / 'best_hyperparameters.json'
    with open(best_params_file, 'w') as f:
        json.dump(best_params, f, indent=4)
    print(f"✓ Saved: {best_params_file}")
    
    # Save aggregated results
    aggregated_results = {
        'n_folds': args.n_folds,
        'best_hyperparameters': best_params,
        'cv_metrics': aggregated_metrics,
        'fold_details': [
            {
                'fold_id': r['fold_id'],
                'n_train': r.get('n_train', r.get('n_train')),  # Handle both old and new format
                'n_val': r.get('n_val', 0),  # New format has separate val set
                'n_test': r.get('n_test', r.get('n_test')),
                'test_metrics': r['test_metrics']
            }
            for r in fold_results
        ]
    }
    
    aggregated_file = args.cv_dir / 'aggregated_results.json'
    with open(aggregated_file, 'w') as f:
        json.dump(aggregated_results, f, indent=4)
    print(f"✓ Saved: {aggregated_file}")
    
    # Create final training config
    print("\nStep 5: Creating final training configuration...")
    final_config = create_final_training_config(
        args.cv_dir,
        best_params,
        args.features,
        args.metadata,
        args.train_ids
    )
    
    config_file = args.cv_dir.parent / 'final_training_config.json'
    with open(config_file, 'w') as f:
        json.dump(final_config, f, indent=4)
    print(f"✓ Saved: {config_file}")
    
    print("\n" + "=" * 80)
    print("AGGREGATION COMPLETE")
    print("=" * 80)
    print("\nNext steps:")
    print(f"1. Train final model:")
    print(f"   python scripts/training/02_train_final_model.py {config_file}")
    print(f"\n2. Or via SBATCH:")
    print(f"   TRAIN_CONFIG={config_file} sbatch scripts/training/run_final_train_edid.sbatch")
    print()


if __name__ == '__main__':
    main()
