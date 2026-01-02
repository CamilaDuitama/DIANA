#!/usr/bin/env python3
"""
Extract Feature Importance from Trained Multi-Task Model
=========================================================

Analyzes the trained model to identify the most discriminant features for each task.
Uses gradient-based feature attribution and weight analysis.

Methods:
--------
1. Weight-based importance: Analyze first layer weights
2. Gradient-based importance: Compute gradients w.r.t. inputs
3. Permutation importance: Measure performance drop when features are shuffled

OUTPUT:
-------
- Top features per task (tables)
- Feature importance distributions (interactive plots)
- Feature overlap analysis between tasks

USAGE:
------
python scripts/feature_analysis/01_extract_feature_importance.py \\
    --config configs/feature_analysis.yaml
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from tqdm import tqdm
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yaml

from diana.data.loader import MatrixLoader
from diana.models.multitask_mlp import MultiTaskMLP


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_model_and_data(config: Dict):
    """Load trained model and test data."""
    print("Loading model configuration...")
    with open(config['model']['config_path'], 'r') as f:
        model_config = json.load(f)
    
    # Load label encoders
    with open(config['model']['label_encoders_path'], 'r') as f:
        encoders_data = json.load(f)
    
    # Get task info
    task_info = {task: len(data['classes']) for task, data in encoders_data.items()}
    task_names = list(task_info.keys())
    
    print("Loading test data...")
    # Load test IDs
    with open(config['data']['test_ids_path'], 'r') as f:
        test_ids = [line.strip() for line in f if line.strip()]
    
    # Load matrix
    loader = MatrixLoader(config['data']['matrix_path'])
    X_full, sample_ids, _ = loader.load()
    
    # Load metadata
    metadata = pl.read_csv(config['data']['metadata_path'], separator='\t')
    
    # Filter to test samples
    test_mask = np.isin(sample_ids, test_ids)
    X_test = X_full[test_mask]
    test_sample_ids = sample_ids[test_mask]
    
    # Filter metadata
    metadata_test = metadata.filter(pl.col('Run_accession').is_in(test_ids))
    
    print(f"Loaded {X_test.shape[0]} test samples with {X_test.shape[1]} features")
    
    # Initialize model
    print("Loading model...")
    device = config['execution']['device']
    model = MultiTaskMLP(
        input_dim=X_test.shape[1],
        num_classes=task_info,
        **model_config['hyperparameters']['model_params']
    )
    
    # Load checkpoint
    checkpoint = torch.load(config['model']['checkpoint_path'], map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    return model, X_test, metadata_test, task_names, task_info, test_sample_ids, encoders_data


def compute_weight_based_importance(model: MultiTaskMLP, task_names: List[str]) -> Dict[str, np.ndarray]:
    """
    Compute feature importance based on weights (optimized).
    
    Uses efficient matrix multiplication to trace input->output contributions.
    """
    print("\nComputing weight-based importance...")
    
    # Extract backbone weights (do this once for all tasks)
    # Keep signed weights - we'll take abs of final contribution
    backbone_weights = [
        layer.weight.data.cpu().numpy()
        for layer in model.backbone
        if isinstance(layer, torch.nn.Linear)
    ]
    
    # Combine backbone weights efficiently: work forward through layers
    # This gives us (final_hidden_dim, input_dim) mapping
    combined_backbone = backbone_weights[0]  # Start with first layer
    for w in backbone_weights[1:]:
        combined_backbone = w @ combined_backbone  # Accumulate transformations
    
    print(f"Backbone maps {combined_backbone.shape[1]} inputs -> {combined_backbone.shape[0]} hidden units")
    
    # Now compute per-task importance (reusing combined_backbone)
    importance_scores = {}
    
    for task in task_names:
        task_head = model.heads[task]
        
        # Extract head weights efficiently
        # Keep signed weights - we'll take abs of final contribution
        head_weights = [
            layer.weight.data.cpu().numpy()
            for layer in task_head
            if isinstance(layer, torch.nn.Linear)
        ]
        
        # Combine head layers: (n_classes, input_dim) via intermediate layers
        combined_head = head_weights[0]  # First head layer
        for w in head_weights[1:]:
            combined_head = w @ combined_head
        
        # Final combination: head @ backbone = (n_classes, input_dim)
        full_contribution = combined_head @ combined_backbone
        
        # Take absolute value of final contribution, then average across output classes
        importance_scores[task] = np.abs(full_contribution).mean(axis=0)
    
    return importance_scores


def compute_gradient_based_importance(
    model: MultiTaskMLP,
    X_test: np.ndarray,
    metadata_test: pl.DataFrame,
    task_names: List[str],
    encoders_data: Dict,
    batch_size: int = 64
) -> Dict[str, np.ndarray]:
    """
    Compute feature importance using gradient-based attribution (optimized).
    
    Uses vectorized operations and DataLoader for parallel data loading.
    """
    print("\nComputing gradient-based importance (optimized)...")
    
    from sklearn.preprocessing import LabelEncoder
    from torch.utils.data import TensorDataset, DataLoader
    
    # Prepare labels for all tasks
    all_labels = {}
    for task in task_names:
        encoder = LabelEncoder()
        encoder.classes_ = np.array(encoders_data[task]['classes'])
        all_labels[task] = encoder.transform(metadata_test[task].to_numpy())
    
    # Create TensorDataset for parallel loading
    X_tensor = torch.FloatTensor(X_test)
    label_tensors = [torch.LongTensor(all_labels[task]) for task in task_names]
    dataset = TensorDataset(X_tensor, *label_tensors)
    
    # DataLoader with multiple workers for parallel data loading
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,  # Parallel data loading
        pin_memory=False  # CPU only
    )
    
    # Initialize gradient accumulators
    grad_accumulator = {task: torch.zeros(X_test.shape[1]) for task in task_names}
    total_samples = len(X_test)
    
    model.train()  # Need gradients
    
    for batch_data in tqdm(dataloader, desc="Computing gradients"):
        batch_X = batch_data[0]
        batch_X.requires_grad = True
        
        # Forward pass
        outputs = model(batch_X)
        
        # Compute gradients for each task (vectorized)
        for task_idx, task in enumerate(task_names):
            batch_labels = batch_data[task_idx + 1]  # Labels for this task
            logits = outputs[task]  # (batch_size, n_classes)
            
            # Vectorized: gather correct class logits for all samples at once
            # Use torch.gather to select logits[i, batch_labels[i]] for all i
            correct_class_logits = torch.gather(
                logits, 
                dim=1, 
                index=batch_labels.unsqueeze(1)
            ).squeeze(1)  # (batch_size,)
            
            # Single backward pass for entire batch
            loss = correct_class_logits.sum()
            
            # Zero gradients if needed
            if batch_X.grad is not None:
                batch_X.grad.zero_()
            
            loss.backward(retain_graph=True)
            
            # Accumulate absolute gradients: sum across batch dimension
            grad_accumulator[task] += batch_X.grad.abs().sum(dim=0).cpu()
    
    # Average gradients across all samples
    gradient_importance = {}
    for task in task_names:
        gradient_importance[task] = (grad_accumulator[task] / total_samples).numpy()
    
    model.eval()
    
    return gradient_importance


def compute_permutation_importance(
    model: MultiTaskMLP,
    X_test: np.ndarray,
    metadata_test: pl.DataFrame,
    task_names: List[str],
    encoders_data: Dict,
    n_features: int = 100,
    batch_size: int = 128
) -> Dict[str, np.ndarray]:
    """
    Compute permutation importance for top features.
    
    For computational efficiency, only test top features from weight-based importance.
    """
    print("\nComputing permutation importance (top features only)...")
    
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import f1_score
    
    # Get baseline performance
    print("Computing baseline performance...")
    baseline_scores = {}
    
    model.eval()
    with torch.no_grad():
        all_preds = {task: [] for task in task_names}
        
        for i in range(0, len(X_test), batch_size):
            batch_X = torch.FloatTensor(X_test[i:i+batch_size])
            outputs = model(batch_X)
            
            for task in task_names:
                preds = torch.argmax(outputs[task], dim=1).cpu().numpy()
                all_preds[task].extend(preds)
        
        for task in task_names:
            encoder = LabelEncoder()
            encoder.classes_ = np.array(encoders_data[task]['classes'])
            true_labels = encoder.transform(metadata_test[task].to_numpy())
            baseline_scores[task] = f1_score(true_labels, all_preds[task], average='weighted')
    
    print(f"Baseline F1 scores: {baseline_scores}")
    
    # Get top features from weight-based importance
    weight_importance = compute_weight_based_importance(model, task_names)
    
    importance_scores = {task: np.zeros(X_test.shape[1]) for task in task_names}
    
    for task in task_names:
        # Get top features for this task
        top_features = np.argsort(weight_importance[task])[-n_features:]
        
        print(f"\nTesting top {n_features} features for {task}...")
        
        for feature_idx in tqdm(top_features, desc=f"Permuting {task}"):
            # Permute feature
            X_permuted = X_test.copy()
            np.random.shuffle(X_permuted[:, feature_idx])
            
            # Compute new performance
            with torch.no_grad():
                preds = []
                for i in range(0, len(X_permuted), batch_size):
                    batch_X = torch.FloatTensor(X_permuted[i:i+batch_size])
                    outputs = model(batch_X)
                    batch_preds = torch.argmax(outputs[task], dim=1).cpu().numpy()
                    preds.extend(batch_preds)
                
                encoder = LabelEncoder()
                encoder.classes_ = np.array(encoders_data[task]['classes'])
                true_labels = encoder.transform(metadata_test[task].to_numpy())
                permuted_score = f1_score(true_labels, preds, average='weighted')
            
            # Importance = drop in performance
            importance_scores[task][feature_idx] = baseline_scores[task] - permuted_score
    
    return importance_scores


def create_feature_importance_plots(
    importance_scores: Dict[str, Dict[str, np.ndarray]],
    task_names: List[str],
    output_dir: Path,
    top_k: int = 50,
    blast_annotations: pl.DataFrame = None
):
    """Create interactive plots of feature importance."""
    print("\nCreating feature importance plots...")
    
    methods = list(importance_scores.keys())
    
    # 1. Top features heatmap for each method
    for method in methods:
        print(f"Creating heatmap for {method}...")
        
        # Get top features for each task
        top_features_data = []
        all_top_features = set()
        
        for task in task_names:
            scores = importance_scores[method][task]
            top_indices = np.argsort(scores)[-top_k:][::-1]
            all_top_features.update(top_indices)
            
            for rank, idx in enumerate(top_indices):
                top_features_data.append({
                    'task': task,
                    'feature_idx': idx,
                    'importance': scores[idx],
                    'rank': rank + 1
                })
        
        # Create heatmap matrix
        feature_list = sorted(all_top_features)
        heatmap_data = np.zeros((len(task_names), len(feature_list)))
        
        # Create hover text with species annotations if available
        hover_text = [['' for _ in range(len(feature_list))] for _ in range(len(task_names))]
        
        for i, task in enumerate(task_names):
            scores = importance_scores[method][task]
            for j, feat_idx in enumerate(feature_list):
                heatmap_data[i, j] = scores[feat_idx]
                hover_parts = [
                    f"Task: {task}",
                    f"Feature: {feat_idx}",
                    f"Importance: {scores[feat_idx]:.6f}"
                ]
                # Add BLAST annotation if available
                if blast_annotations is not None and len(blast_annotations) > 0:
                    try:
                        # Try to match by task and feature_index
                        annot = blast_annotations.filter(
                            (pl.col('task') == task) & (pl.col('feature_index') == feat_idx)
                        )
                        # If not found, try matching by id (unitig id)
                        if len(annot) == 0:
                            # Try to get id from blast_annotations and feature_list
                            if 'id' in blast_annotations.columns:
                                # Get id for this feature_index from any annotation row
                                id_candidates = blast_annotations.filter(pl.col('feature_index') == feat_idx)
                                if len(id_candidates) > 0:
                                    unitig_id = id_candidates['id'][0]
                                    annot = blast_annotations.filter(
                                        (pl.col('task') == task) & (pl.col('id') == unitig_id)
                                    )
                        if len(annot) > 0:
                            species = annot['best_hit_species'][0] if 'best_hit_species' in annot.columns else ''
                            genus = annot['genus'][0] if 'genus' in annot.columns else ''
                            phylum = annot['phylum'][0] if 'phylum' in annot.columns else ''
                            if species and str(species) != '' and str(species) != 'None':
                                hover_parts.append(f"Species: {species}")
                            if genus and str(genus) != '' and str(genus) != 'None':
                                hover_parts.append(f"Genus: {genus}")
                            if phylum and str(phylum) != '' and str(phylum) != 'None':
                                hover_parts.append(f"Phylum: {phylum}")
                    except Exception as e:
                        hover_parts.append(f"[BLAST lookup error: {e}]")
                hover_text[i][j] = '<br>'.join(hover_parts)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data,
            x=[f"F{idx}" for idx in feature_list],
            y=task_names,
            colorscale='Viridis',
            colorbar=dict(title='Importance'),
            hovertext=hover_text,
            hovertemplate='%{hovertext}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f'Top {top_k} Feature Importance Heatmap ({method.replace("_", " ").title()})',
            xaxis_title='Feature Index',
            yaxis_title='Task',
            height=400,
            width=1200
        )
        
        html_path = output_dir / f'feature_importance_heatmap_{method}.html'
        png_path = output_dir / f'feature_importance_heatmap_{method}.png'
        fig.write_html(html_path)
        fig.write_image(png_path, width=1200, height=400, scale=2)
        print(f"Saved {html_path} and {png_path}")
    
    # 2. Comparison plot: top features across methods (side-by-side)
    print("Creating method comparison plot...")
    
    # Use fewer features for clearer comparison (top 20 instead of all top_k)
    comparison_k = min(20, top_k)
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[t.replace('_', ' ').title() for t in task_names],
        vertical_spacing=0.15,
        horizontal_spacing=0.12
    )
    
    # Define colors for each method
    method_colors = {
        'weight_based': '#1f77b4',
        'gradient_based': '#ff7f0e',
        'permutation': '#2ca02c'
    }
    
    for idx, task in enumerate(task_names):
        row = idx // 2 + 1
        col = idx % 2 + 1
        
        # Get top features from first method (to align comparison)
        first_method = methods[0]
        scores_first = importance_scores[first_method][task]
        top_indices = np.argsort(scores_first)[-comparison_k:][::-1]
        
        # Plot each method's scores for these features
        for method in methods:
            scores = importance_scores[method][task]
            
            fig.add_trace(
                go.Bar(
                    x=list(range(1, comparison_k + 1)),
                    y=scores[top_indices],
                    name=method.replace('_', ' ').title(),
                    legendgroup=method,
                    showlegend=(idx == 0),
                    marker_color=method_colors.get(method, '#999999'),
                    opacity=0.8
                ),
                row=row, col=col
            )
        
        fig.update_xaxes(title_text='Feature Rank', row=row, col=col)
        fig.update_yaxes(title_text='Importance Score', row=row, col=col)
    
    fig.update_layout(
        title_text=f'Feature Importance Comparison: Top {comparison_k} Features per Task',
        height=800,
        width=1400,
        barmode='group',  # Side-by-side bars
        bargap=0.15,
        bargroupgap=0.1
    )
    
    html_path = output_dir / 'feature_importance_comparison.html'
    png_path = output_dir / 'feature_importance_comparison.png'
    fig.write_html(html_path)
    fig.write_image(png_path, width=1400, height=800, scale=2)
    print(f"Saved {html_path} and {png_path}")
    
    # 3. Feature overlap Venn diagram (conceptual - using bar chart)
    print("Creating feature overlap analysis...")
    
    for method in methods:
        overlap_data = []
        
        # For each pair of tasks, compute overlap
        for i, task1 in enumerate(task_names):
            scores1 = importance_scores[method][task1]
            top1 = set(np.argsort(scores1)[-top_k:])
            
            for task2 in task_names[i+1:]:
                scores2 = importance_scores[method][task2]
                top2 = set(np.argsort(scores2)[-top_k:])
                
                overlap = len(top1 & top2)
                overlap_data.append({
                    'pair': f'{task1} ∩ {task2}',
                    'overlap': overlap,
                    'percentage': (overlap / top_k) * 100
                })
        
        df_overlap = pl.DataFrame(overlap_data)
        
        fig = go.Figure(data=[
            go.Bar(
                x=df_overlap['pair'].to_list(),
                y=df_overlap['overlap'].to_list(),
                text=df_overlap['percentage'].to_list(),
                texttemplate='%{text:.1f}%',
                textposition='outside'
            )
        ])
        
        fig.update_layout(
            title=f'Feature Overlap Between Tasks (Top {top_k}, {method.replace("_", " ").title()})',
            xaxis_title='Task Pair',
            yaxis_title=f'Number of Shared Features (out of {top_k})',
            height=500,
            width=1000
        )
        
        html_path = output_dir / f'feature_overlap_{method}.html'
        png_path = output_dir / f'feature_overlap_{method}.png'
        fig.write_html(html_path)
        fig.write_image(png_path, width=1000, height=500, scale=2)
        print(f"Saved {html_path} and {png_path}")


def save_feature_tables(
    importance_scores: Dict[str, Dict[str, np.ndarray]],
    task_names: List[str],
    output_dir: Path,
    top_k: int = 100
):
    """Save tables of top features per task."""
    print("\nSaving feature importance tables...")
    
    # Get tables directory from config (parallel to figures_dir)
    # output_dir is paper/figures/feature_analysis
    # Need to go up 2 levels: parent.parent = paper/
    tables_dir = output_dir.parent.parent / 'tables' / 'feature_analysis'
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    for method in importance_scores.keys():
        all_top_features = []
        
        for task in task_names:
            scores = importance_scores[method][task]
            top_indices = np.argsort(scores)[-top_k:][::-1]
            
            for rank, idx in enumerate(top_indices):
                all_top_features.append({
                    'task': task,
                    'rank': rank + 1,
                    'feature_index': int(idx),
                    'importance_score': float(scores[idx]),
                    'method': method
                })
        
        df = pl.DataFrame(all_top_features)
        
        # Save as CSV
        csv_path = tables_dir / f'top_{top_k}_features_{method}.csv'
        df.write_csv(csv_path)
        print(f"Saved {csv_path}")
        
        # Save as Markdown (simple format)
        md_path = tables_dir / f'top_{top_k}_features_{method}.md'
        with open(md_path, 'w') as f:
            f.write(f"# Top {top_k} Features per Task ({method.replace('_', ' ').title()})\n\n")
            
            for task in task_names:
                task_df = df.filter(pl.col('task') == task).head(20)  # Show top 20 in markdown
                f.write(f"\n## {task.replace('_', ' ').title()}\n\n")
                
                # Manual markdown table
                f.write("| Rank | Feature Index | Importance Score |\n")
                f.write("|------|---------------|------------------|\n")
                for row in task_df.select(['rank', 'feature_index', 'importance_score']).iter_rows():
                    f.write(f"| {row[0]} | {row[1]} | {row[2]:.6f} |\n")
                f.write('\n')
        
        print(f"Saved {md_path}")


def main():
    parser = argparse.ArgumentParser(description='Extract feature importance from trained model')
    parser.add_argument('--config', type=str, default='configs/feature_analysis.yaml',
                       help='Path to feature analysis config file')
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from {args.config}...")
    config = load_config(args.config)
    
    # Create output directories
    output_dir = Path(config['output']['figures_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    tables_dir = Path(config['output']['tables_dir'])
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model and data
    model, X_test, metadata_test, task_names, task_info, test_sample_ids, encoders_data = load_model_and_data(config)
    
    # Compute importance scores
    importance_scores = {}
    
    methods = config['feature_importance']['methods']
    top_k = config['feature_importance']['top_k']
    batch_size = config['feature_importance']['batch_size']
    
    if 'weight' in methods:
        importance_scores['weight_based'] = compute_weight_based_importance(model, task_names)
    
    if 'gradient' in methods:
        importance_scores['gradient_based'] = compute_gradient_based_importance(
            model, X_test, metadata_test, task_names, encoders_data, batch_size=batch_size
        )
    
    if 'permutation' in methods:
        importance_scores['permutation'] = compute_permutation_importance(
            model, X_test, metadata_test, task_names, encoders_data, 
            n_features=top_k, batch_size=batch_size
        )
    
    # Create visualizations
    # Try to load BLAST annotations
    blast_annotations = None
    try:
        tables_dir = Path(config['output']['tables_dir'])
        all_annotations = []
        for annotated_file in sorted(tables_dir.glob('top_features_*_annotated.csv')):
            df = pl.read_csv(annotated_file)
            if 'best_hit_species' in df.columns:
                task = annotated_file.stem.replace('top_features_', '').replace('_annotated', '')
                df = df.with_columns(pl.lit(task).alias('task'))
                all_annotations.append(df)
        if all_annotations:
            blast_annotations = pl.concat(all_annotations)
            print(f"Loaded BLAST annotations for {len(blast_annotations)} features")
    except Exception as e:
        print(f"Could not load BLAST annotations: {e}")
    
    create_feature_importance_plots(importance_scores, task_names, output_dir, top_k, blast_annotations)
    
    # Save tables
    save_feature_tables(importance_scores, task_names, output_dir, top_k)
    
    print("\n✅ Feature importance analysis complete!")
    print(f"Figures saved to: {output_dir}")
    print(f"Tables saved to: {tables_dir}")


if __name__ == "__main__":
    main()
