#!/usr/bin/env python3
"""
Out-of-Distribution (OOD) Detection for DIANA predictions.

Uses penultimate layer embeddings (272-dim) to detect samples far from training distribution.
Flags samples that may have unreliable predictions.
"""

import argparse
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from scipy.spatial.distance import cdist
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Import DIANA model
import sys
sys.path.insert(0, 'src')
from diana.models.multitask_mlp import MultiTaskMLP


def load_model(model_path: str, input_dim: int, device: str = 'cpu'):
    """Load trained DIANA model."""
    print(f"Loading model from {model_path}...")
    
    # Load training config
    config_path = Path(model_path).parent / 'final_training_config.json'
    with open(config_path) as f:
        config = json.load(f)
    
    # Load label encoders to get num_classes
    encoders_path = Path(model_path).parent / 'label_encoders.json'
    with open(encoders_path) as f:
        encoders = json.load(f)
    
    num_classes = {task: len(data['classes']) for task, data in encoders.items()}
    
    # Get model params
    model_params = config['hyperparameters']['model_params']
    
    # Create model
    model = MultiTaskMLP(
        input_dim=input_dim,
        hidden_dims=model_params['hidden_dims'],
        num_classes=num_classes,
        dropout=model_params.get('dropout', 0.5),
        use_batch_norm=model_params.get('use_batch_norm', True),
        activation=model_params.get('activation', 'relu')
    )
    
    # Load weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"  Input dim: {input_dim}")
    print(f"  Hidden dims: {model_params['hidden_dims']}")
    print(f"  Num classes: {num_classes}")
    
    return model


class EmbeddingExtractor:
    """Extract embeddings from penultimate layer."""
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.embeddings = []
        
        # Register hook to capture penultimate layer output (before task heads)
        # In MultiTaskMLP: backbone is the shared feature extractor
        self.hook = model.backbone.register_forward_hook(self._hook_fn)
    
    def _hook_fn(self, module, input, output):
        """Hook function to capture layer output."""
        self.embeddings.append(output.detach().cpu().numpy())
    
    def extract(self, data_loader):
        """Extract embeddings for all samples in data_loader."""
        all_embeddings = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Extracting embeddings"):
                inputs = batch[0].to(self.device)
                self.embeddings = []  # Reset
                _ = self.model(inputs)  # Forward pass triggers hook
                all_embeddings.append(self.embeddings[0])
        
        return np.vstack(all_embeddings)
    
    def remove_hook(self):
        """Remove the hook."""
        self.hook.remove()


def load_data(matrix_file: str, batch_size: int = 64):
    """Load data and create DataLoader."""
    from diana.data.loader import MatrixLoader
    from torch.utils.data import DataLoader, TensorDataset
    
    print(f"Loading data from {matrix_file}...")
    
    # Load matrix
    loader = MatrixLoader(matrix_file)
    X, sample_ids, _ = loader.load()
    
    # Convert to dense if sparse
    if hasattr(X, 'toarray'):
        X = X.toarray()
    
    # Convert to tensors
    X_tensor = torch.FloatTensor(X)
    
    # Create DataLoader
    dataset = TensorDataset(X_tensor)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    return data_loader, sample_ids, X_tensor.shape[1]


def compute_ood_threshold(embeddings: np.ndarray, percentile: float = 95.0):
    """
    Compute OOD threshold as percentile of nearest-neighbor distances in training set.
    
    Args:
        embeddings: Training embeddings (n_samples, n_features)
        percentile: Percentile to use as threshold (e.g., 95.0)
    
    Returns:
        threshold: Distance threshold
        nn_distances: All nearest-neighbor distances
    """
    print(f"Computing {percentile}th percentile threshold...")
    
    n_samples = len(embeddings)
    nn_distances = []
    
    # For each training sample, find distance to nearest neighbor
    for i in tqdm(range(n_samples), desc="Computing NN distances"):
        # Get distances to all other samples
        sample = embeddings[i:i+1]
        distances = cdist(sample, embeddings, metric='euclidean')[0]
        
        # Find minimum distance (excluding self)
        distances[i] = np.inf
        min_dist = distances.min()
        nn_distances.append(min_dist)
    
    nn_distances = np.array(nn_distances)
    threshold = np.percentile(nn_distances, percentile)
    
    print(f"  Threshold: {threshold:.4f}")
    print(f"  Min NN distance: {nn_distances.min():.4f}")
    print(f"  Max NN distance: {nn_distances.max():.4f}")
    print(f"  Mean NN distance: {nn_distances.mean():.4f}")
    
    return threshold, nn_distances


def detect_ood_samples(
    val_embeddings: np.ndarray,
    train_embeddings: np.ndarray,
    threshold: float
):
    """
    Detect OOD samples in validation set.
    
    Args:
        val_embeddings: Validation embeddings (n_val, n_features)
        train_embeddings: Training embeddings (n_train, n_features)
        threshold: OOD threshold
    
    Returns:
        min_distances: Minimum distance to training set for each validation sample
        is_ood: Boolean array indicating OOD samples
    """
    print("Computing distances to training set...")
    
    n_val = len(val_embeddings)
    min_distances = []
    
    # For each validation sample, find minimum distance to training set
    for i in tqdm(range(n_val), desc="Computing min distances"):
        sample = val_embeddings[i:i+1]
        distances = cdist(sample, train_embeddings, metric='euclidean')[0]
        min_dist = distances.min()
        min_distances.append(min_dist)
    
    min_distances = np.array(min_distances)
    is_ood = min_distances > threshold
    
    print(f"\nOOD Detection Results:")
    print(f"  Total samples: {n_val}")
    print(f"  OOD samples: {is_ood.sum()} ({is_ood.sum()/n_val*100:.1f}%)")
    print(f"  In-distribution: {(~is_ood).sum()} ({(~is_ood).sum()/n_val*100:.1f}%)")
    
    return min_distances, is_ood


def analyze_ood_vs_accuracy(
    predictions_file: str,
    is_ood: np.ndarray,
    min_distances: np.ndarray,
    val_sample_ids: np.ndarray,
    output_dir: Path
):
    """Analyze relationship between OOD status and prediction accuracy."""
    
    print("\nAnalyzing OOD vs Accuracy...")
    
    # Load predictions
    preds = pd.read_csv(predictions_file, sep='\t')
    
    # Create DataFrame with sample IDs for alignment
    ood_df = pd.DataFrame({
        'sample_id': val_sample_ids,
        'is_ood': is_ood,
        'min_distance': min_distances
    })
    
    # Merge with predictions (inner join to only keep samples with predictions)
    preds = preds.merge(ood_df, on='sample_id', how='left')
    
    # Report any missing OOD info
    n_missing = preds['is_ood'].isna().sum()
    if n_missing > 0:
        print(f"  Warning: {n_missing} samples in predictions but not in matrix")
        preds = preds.dropna(subset=['is_ood'])
    
    # Analyze for each task
    tasks = ['sample_type', 'community_type', 'sample_host', 'material']
    results = []
    
    for task in tasks:
        # Check correctness
        correct = preds[f'{task}_pred'] == preds[f'{task}_true']
        
        # Split by OOD
        in_dist = ~preds['is_ood']
        ood = preds['is_ood']
        
        # Calculate accuracies
        in_dist_acc = correct[in_dist].mean() * 100
        ood_acc = correct[ood].mean() * 100
        diff = in_dist_acc - ood_acc
        
        # Chi-squared test
        contingency = pd.crosstab(preds['is_ood'], correct)
        chi2, p_value, dof, expected = chi2_contingency(contingency)
        
        results.append({
            'task': task,
            'in_dist_samples': in_dist.sum(),
            'in_dist_accuracy': in_dist_acc,
            'ood_samples': ood.sum(),
            'ood_accuracy': ood_acc,
            'accuracy_diff': diff,
            'chi2': chi2,
            'p_value': p_value
        })
        
        print(f"\n{task}:")
        print(f"  In-dist: {in_dist.sum()} samples, {in_dist_acc:.1f}% accuracy")
        print(f"  OOD: {ood.sum()} samples, {ood_acc:.1f}% accuracy")
        print(f"  Difference: {diff:.1f}%")
        print(f"  Chi-squared test: χ²={chi2:.2f}, p={p_value:.4f}")
    
    # Save results table
    results_df = pd.DataFrame(results)
    results_file = output_dir / 'ood_accuracy_by_task.tsv'
    results_df.to_csv(results_file, sep='\t', index=False)
    print(f"\nSaved results: {results_file}")
    
    # Save full predictions with OOD flags
    preds_file = output_dir / 'validation_predictions_with_ood.tsv'
    preds.to_csv(preds_file, sep='\t', index=False)
    print(f"Saved predictions with OOD flags: {preds_file}")
    
    return results_df, preds


def create_visualizations(
    preds_df: pd.DataFrame,
    nn_distances_train: np.ndarray,
    threshold: float,
    output_dir: Path
):
    """Create visualizations for OOD analysis."""
    
    print("\nCreating visualizations...")
    
    sns.set_style('whitegrid')
    
    # Plot 1: Histogram of distances
    fig, ax = plt.subplots(figsize=(10, 6))
    
    in_dist = preds_df[~preds_df['is_ood']]['min_distance']
    ood = preds_df[preds_df['is_ood']]['min_distance']
    
    ax.hist(nn_distances_train, bins=50, alpha=0.5, label='Training NN distances', color='blue')
    ax.hist(in_dist, bins=50, alpha=0.5, label='Validation in-dist', color='green')
    ax.hist(ood, bins=50, alpha=0.5, label='Validation OOD', color='red')
    ax.axvline(threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold ({threshold:.2f})')
    
    ax.set_xlabel('Distance to Nearest Training Sample', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('OOD Detection: Distance Distribution', fontsize=14)
    ax.legend()
    
    plot_file = output_dir / 'ood_distance_histogram.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {plot_file}")
    
    # Plot 2: Distance vs Correctness (scatter)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    tasks = ['sample_type', 'community_type', 'sample_host', 'material']
    
    for idx, (ax, task) in enumerate(zip(axes.flat, tasks)):
        correct = preds_df[f'{task}_pred'] == preds_df[f'{task}_true']
        
        # Scatter plot with jitter
        jitter = np.random.normal(0, 0.02, len(correct))
        ax.scatter(preds_df['min_distance'], correct.astype(int) + jitter, 
                  alpha=0.3, s=20, c=preds_df['is_ood'].map({True: 'red', False: 'blue'}))
        ax.axvline(threshold, color='black', linestyle='--', linewidth=2)
        
        ax.set_xlabel('Distance to Nearest Training Sample', fontsize=11)
        ax.set_ylabel('Correct Prediction', fontsize=11)
        ax.set_title(f'{task}', fontsize=12)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Incorrect', 'Correct'])
    
    plt.tight_layout()
    plot_file = output_dir / 'ood_distance_vs_correctness.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {plot_file}")
    
    # Plot 3: Boxplots by correctness
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    
    for ax, task in zip(axes, tasks):
        correct = preds_df[f'{task}_pred'] == preds_df[f'{task}_true']
        data = [
            preds_df[correct]['min_distance'],
            preds_df[~correct]['min_distance']
        ]
        
        bp = ax.boxplot(data, labels=['Correct', 'Incorrect'], patch_artist=True)
        ax.axhline(threshold, color='red', linestyle='--', linewidth=2, label='OOD threshold')
        
        ax.set_ylabel('Distance to Nearest Training Sample', fontsize=11)
        ax.set_title(f'{task}', fontsize=12)
        ax.legend()
    
    plt.tight_layout()
    plot_file = output_dir / 'ood_distance_boxplots.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {plot_file}")


def test_multiple_thresholds(
    val_embeddings: np.ndarray,
    train_embeddings: np.ndarray,
    nn_distances_train: np.ndarray,
    val_sample_ids: np.ndarray,
    predictions_file: str,
    output_dir: Path,
    percentiles: list = [90, 95, 99]
):
    """Test different threshold percentiles."""
    
    print("\nTesting multiple thresholds...")
    
    preds = pd.read_csv(predictions_file, sep='\t')
    tasks = ['sample_type', 'community_type', 'sample_host', 'material']
    
    # Compute min distances for validation (only once)
    print("Computing validation distances...")
    val_min_distances = []
    for i in tqdm(range(len(val_embeddings))):
        sample = val_embeddings[i:i+1]
        distances = cdist(sample, train_embeddings, metric='euclidean')[0]
        val_min_distances.append(distances.min())
    val_min_distances = np.array(val_min_distances)
    
    results = []
    
    for percentile in percentiles:
        threshold = np.percentile(nn_distances_train, percentile)
        is_ood = val_min_distances > threshold
        
        # Align with predictions using sample IDs
        ood_df = pd.DataFrame({
            'sample_id': val_sample_ids,
            'is_ood': is_ood
        })
        preds_aligned = preds.merge(ood_df, on='sample_id', how='left')
        preds_aligned = preds_aligned.dropna(subset=['is_ood'])
        
        print(f"\n{percentile}th percentile (threshold={threshold:.4f}):")
        print(f"  OOD samples: {is_ood.sum()} ({is_ood.sum()/len(is_ood)*100:.1f}%)")
        
        for task in tasks:
            correct = preds_aligned[f'{task}_pred'] == preds_aligned[f'{task}_true']
            in_dist_acc = correct[~preds_aligned['is_ood']].mean() * 100
            ood_acc = correct[preds_aligned['is_ood']].mean() * 100 if preds_aligned['is_ood'].sum() > 0 else 0
            
            results.append({
                'percentile': percentile,
                'threshold': threshold,
                'task': task,
                'pct_ood': is_ood.sum() / len(is_ood) * 100,
                'in_dist_accuracy': in_dist_acc,
                'ood_accuracy': ood_acc,
                'accuracy_diff': in_dist_acc - ood_acc
            })
    
    results_df = pd.DataFrame(results)
    results_file = output_dir / 'threshold_comparison.tsv'
    results_df.to_csv(results_file, sep='\t', index=False)
    print(f"\nSaved threshold comparison: {results_file}")
    
    return results_df


def main():
    parser = argparse.ArgumentParser(description='OOD Detection for DIANA')
    parser.add_argument('--model-path', default='results/full_training/best_model.pth',
                       help='Path to trained model')
    parser.add_argument('--train-matrix', default='data/matrices/large_matrix_3070_with_frac/unitigs.frac.mat',
                       help='Training unitig matrix file')
    parser.add_argument('--train-metadata', default='paper/metadata/train_metadata.tsv',
                       help='Training metadata file')
    parser.add_argument('--val-matrix', default='data/validation/validation_matrix.pa.mat',
                       help='Validation matrix file (separate from training)')
    parser.add_argument('--predictions', default='results/validation_predictions/validation_predictions.tsv',
                       help='Validation predictions file')
    parser.add_argument('--output-dir', default='results/ood_detection',
                       help='Output directory')
    parser.add_argument('--percentile', type=float, default=95.0,
                       help='Percentile for OOD threshold')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size for inference')
    parser.add_argument('--device', default='cpu',
                       help='Device (cpu or cuda)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load matrices directly (validation is separate)
    from diana.data.loader import MatrixLoader
    from torch.utils.data import DataLoader, TensorDataset
    
    print("Loading training matrix...")
    train_meta = pd.read_csv(args.train_metadata, sep='\t')
    train_loader_obj = MatrixLoader(args.train_matrix)
    X_train, train_sample_ids, _ = train_loader_obj.load()
    
    # Filter to samples in metadata
    train_ids_set = set(train_meta['Run_accession'].values)
    train_mask = np.isin(train_sample_ids, list(train_ids_set))
    X_train = X_train[train_mask]
    
    print(f"  Train matrix: {X_train.shape}")
    
    print("\nLoading validation matrix...")
    val_loader_obj = MatrixLoader(args.val_matrix)
    X_val, val_sample_ids, _ = val_loader_obj.load()
    
    print(f"  Validation matrix: {X_val.shape}")
    
    # Convert to dense if sparse
    if hasattr(X_train, 'toarray'):
        X_train = X_train.toarray()
    if hasattr(X_val, 'toarray'):
        X_val = X_val.toarray()
    
    # Create DataLoaders
    X_train_tensor = torch.FloatTensor(X_train)
    X_val_tensor = torch.FloatTensor(X_val)
    
    train_loader = DataLoader(TensorDataset(X_train_tensor), 
                              batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(TensorDataset(X_val_tensor), 
                           batch_size=args.batch_size, shuffle=False)
    
    # Load model
    input_dim = X_train.shape[1]
    model = load_model(args.model_path, input_dim, args.device)
    
    extractor = EmbeddingExtractor(model, args.device)
    
    print("\nExtracting training embeddings...")
    train_embeddings = extractor.extract(train_loader)
    
    # Save training embeddings
    np.save(output_dir / 'train_embeddings.npy', train_embeddings)
    print(f"Saved training embeddings: {train_embeddings.shape}")
    
    # Step 3: Compute OOD threshold
    threshold, nn_distances_train = compute_ood_threshold(train_embeddings, args.percentile)
    np.save(output_dir / 'nn_distances_train.npy', nn_distances_train)
    
    # Save threshold
    threshold_info = {
        'percentile': args.percentile,
        'threshold': float(threshold),
        'min_nn_dist': float(nn_distances_train.min()),
        'max_nn_dist': float(nn_distances_train.max()),
        'mean_nn_dist': float(nn_distances_train.mean())
    }
    with open(output_dir / 'threshold_info.json', 'w') as f:
        json.dump(threshold_info, f, indent=2)
    
    # Step 4: Extract validation embeddings and compute distances
    print("\nExtracting validation embeddings...")
    val_embeddings = extractor.extract(val_loader)
    extractor.remove_hook()
    
    np.save(output_dir / 'val_embeddings.npy', val_embeddings)
    print(f"Saved validation embeddings: {val_embeddings.shape}")
    
    # Detect OOD samples
    min_distances, is_ood = detect_ood_samples(
        val_embeddings, train_embeddings, threshold
    )
    
    # Step 5-6: Analyze OOD vs accuracy
    results_df, preds_df = analyze_ood_vs_accuracy(
        args.predictions, is_ood, min_distances, val_sample_ids, output_dir
    )
    
    # Step 7: Create visualizations
    create_visualizations(preds_df, nn_distances_train, threshold, output_dir)
    
    # Step 8: Test different thresholds
    threshold_results = test_multiple_thresholds(
        val_embeddings, train_embeddings, nn_distances_train, val_sample_ids,
        args.predictions, output_dir, percentiles=[90, 95, 99]
    )
    
    print("\n=== OOD Detection Complete ===")
    print(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
