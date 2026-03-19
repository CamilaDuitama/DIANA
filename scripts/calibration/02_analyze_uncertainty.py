#!/usr/bin/env python3
"""
Monte Carlo Dropout Uncertainty Analysis
=========================================

Uses MC Dropout to quantify prediction uncertainty and analyze whether 
incorrect predictions show higher uncertainty than correct ones.

PROBLEM:
--------
Temperature scaling globally reduces confidence but doesn't identify which
specific predictions are uncertain. MC Dropout provides per-sample uncertainty
estimates by running multiple forward passes with dropout enabled.

MC DROPOUT:
-----------
Instead of a single forward pass:
1. Set model.train() to keep dropout active during inference
2. Run N forward passes (e.g., N=50) on the same sample
3. Get N different predictions due to random dropout masks
4. Calculate mean and variance across the N predictions

Key metrics:
- Mean prediction: Average of N probability vectors (more robust than single pass)
- Uncertainty: Variance across N predictions (epistemic uncertainty)
- High variance = model uncertain about this sample

EXPECTED OUTCOME:
----------------
If MC Dropout captures useful uncertainty:
- Incorrect predictions should have HIGHER uncertainty scores
- Correct predictions should have LOWER uncertainty scores
- Clear separation in uncertainty distributions → actionable threshold

WORKFLOW:
---------
1. Load trained model and validation data
2. For each sample:
   - Run N forward passes with dropout enabled
   - Calculate mean probabilities and uncertainty (variance)
   - Record: true label, predicted label, confidence, uncertainty
3. Analyze results:
   - Compare uncertainty distributions (correct vs incorrect)
   - Generate box plots / violin plots per task
   - Calculate separation metrics (effect size, AUC)
4. Output results TSV with all sample-level predictions + uncertainties

USAGE:
------
# Basic usage (uses validation set from data splits):
python scripts/calibration/02_analyze_uncertainty.py

# Custom paths:
python scripts/calibration/02_analyze_uncertainty.py \\
    --model results/full_training/cv_results/fold_0/best_model.pth \\
    --features data/matrices/training_matrix/unitigs.frac.mat \\
    --metadata data/validation/validation_metadata.tsv \\
    --n-samples 50 \\
    --output results/mc_dropout_analysis.tsv

OUTPUT:
-------
results/mc_dropout_uncertainty.tsv:
sample_id,task,true_label,predicted_label,mean_confidence,uncertainty,correct
ERR123,sample_type,ancient,ancient,0.95,0.002,True
ERR456,sample_type,modern,ancient,0.87,0.045,False
...

INTERPRETATION:
---------------
Good uncertainty quantification shows:
- Median uncertainty (incorrect) >> Median uncertainty (correct)
- Cohen's d > 0.8 (large effect size)
- AUROC > 0.7 for using uncertainty to detect errors
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from diana.models.multitask_mlp import MultiTaskMLP
from diana.data.loader import MatrixLoader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data(matrix_path: Path, metadata_path: Path) -> Tuple[np.ndarray, pd.DataFrame]:
    """Load matrix and metadata."""
    loader = MatrixLoader(matrix_path)
    features, metadata_pl = loader.load_with_metadata(
        metadata_path=metadata_path,
        align_to_matrix=True
    )
    metadata = metadata_pl.to_pandas()
    logger.info(f"Loaded {features.shape[0]} samples with {features.shape[1]} features")
    return features, metadata


def load_label_encoders(label_encoders_path: Path) -> Tuple[Dict[str, LabelEncoder], Dict[str, int]]:
    """
    Load pre-trained label encoders from JSON.
    
    Args:
        label_encoders_path: Path to label_encoders.json
        
    Returns:
        Tuple of (encoders_dict, num_classes_dict)
    """
    import json
    with open(label_encoders_path) as f:
        encoder_data = json.load(f)
    
    encoders = {}
    num_classes = {}
    for task, data in encoder_data.items():
        encoder = LabelEncoder()
        encoder.classes_ = np.array(data['classes'])
        encoders[task] = encoder
        num_classes[task] = len(encoder.classes_)
        logger.info(f"{task}: {num_classes[task]} classes (from trained model)")
    
    return encoders, num_classes


def prepare_labels(
    metadata: pd.DataFrame,
    targets: List[str] = ["sample_type", "community_type", "sample_host", "material"],
    label_encoders: Optional[Dict[str, LabelEncoder]] = None
) -> Tuple[Dict[str, np.ndarray], Dict[str, LabelEncoder], Dict[str, int]]:
    """
    Encode labels for all classification targets.
    
    Args:
        metadata: Metadata dataframe
        targets: List of target columns
        label_encoders: Optional pre-trained label encoders. If provided, will use these
                       and map unknown classes to -1.
    
    Returns:
        Tuple of (labels_dict, encoders_dict, num_classes_dict)
    """
    labels = {}
    encoders = {}
    num_classes = {}
    
    for target in targets:
        if label_encoders and target in label_encoders:
            # Use pre-trained encoder
            encoder = label_encoders[target]
            encoders[target] = encoder
            num_classes[target] = len(encoder.classes_)
            
            # Transform labels, handling unknown classes
            values = metadata[target].values
            labels[target] = np.array([
                encoder.transform([val])[0] if val in encoder.classes_ else -1
                for val in values
            ])
            
            # Count unknown classes
            unknown_count = (labels[target] == -1).sum()
            if unknown_count > 0:
                logger.warning(f"{target}: {unknown_count} samples with unknown classes (will be kept for uncertainty analysis)")
        else:
            # Fit new encoder
            encoder = LabelEncoder()
            labels[target] = encoder.fit_transform(metadata[target].values)
            encoders[target] = encoder
            num_classes[target] = len(encoder.classes_)
            logger.info(f"{target}: {num_classes[target]} classes")
    
    return labels, encoders, num_classes


def mc_dropout_predict(
    model: MultiTaskMLP,
    X: np.ndarray,
    device: torch.device,
    n_samples: int = 50,
    batch_size: int = 256
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Perform MC Dropout inference to get mean predictions and uncertainties.
    
    Args:
        model: Trained MultiTaskMLP
        X: Features array (N_samples x N_features)
        device: Torch device
        n_samples: Number of MC forward passes per sample
        batch_size: Batch size for inference
        
    Returns:
        Tuple of (mean_probs_dict, uncertainties_dict)
        - mean_probs_dict: Dict[task] -> (N_samples x N_classes) mean probabilities
        - uncertainties_dict: Dict[task] -> (N_samples,) uncertainty scores (variance)
    """
    # CRITICAL: Set model to train mode to enable dropout
    model.train()
    
    n_samples_data = X.shape[0]
    tasks = model.targets
    
    # Initialize storage for all MC samples
    mc_predictions = {
        task: np.zeros((n_samples, n_samples_data, model.num_classes[task]))
        for task in tasks
    }
    
    logger.info(f"Running {n_samples} MC dropout iterations...")
    
    # Run N forward passes with dropout
    for mc_iter in range(n_samples):
        if (mc_iter + 1) % 10 == 0:
            logger.info(f"  MC iteration {mc_iter + 1}/{n_samples}")
        
        with torch.no_grad():
            for i in range(0, n_samples_data, batch_size):
                batch = torch.FloatTensor(X[i:i + batch_size]).to(device)
                outputs = model(batch)  # Gets logits
                
                # Apply softmax to get probabilities
                for task, logits in outputs.items():
                    probs = F.softmax(logits, dim=1).cpu().numpy()
                    mc_predictions[task][mc_iter, i:i + len(batch)] = probs
    
    # Calculate mean and uncertainty (variance) across MC samples
    mean_probs = {}
    uncertainties = {}
    
    for task in tasks:
        # Mean probability across MC samples (N_samples x N_classes)
        mean_probs[task] = mc_predictions[task].mean(axis=0)
        
        # Uncertainty: variance of predicted class probability
        # Higher variance = more uncertain
        # Use variance of max probability across MC samples
        max_probs_per_mc = mc_predictions[task].max(axis=2)  # (n_samples, N_samples_data)
        uncertainties[task] = max_probs_per_mc.var(axis=0)  # (N_samples_data,)
    
    return mean_probs, uncertainties


def analyze_task_uncertainty(
    true_labels: np.ndarray,
    mean_probs: np.ndarray,
    uncertainties: np.ndarray,
    encoder: LabelEncoder,
    task_name: str
) -> pd.DataFrame:
    """
    Analyze uncertainty for a single task.
    
    Handles unknown classes (true_labels == -1) by treating them separately.
    
    Returns DataFrame with columns:
    - true_label: True class name (or "UNKNOWN_CLASS" for unseen classes)
    - predicted_label: Predicted class name
    - mean_confidence: Mean confidence score
    - uncertainty: Uncertainty score (variance)
    - correct: Boolean indicating if prediction is correct (False for unknown classes)
    """
    predictions = mean_probs.argmax(axis=1)
    confidences = mean_probs.max(axis=1)
    
    # Separate known and unknown classes
    known_mask = true_labels != -1
    n_unknown = (~known_mask).sum()
    n_known = known_mask.sum()
    
    # For known classes, check correctness
    correct = np.zeros(len(true_labels), dtype=bool)
    correct[known_mask] = predictions[known_mask] == true_labels[known_mask]
    
    # Decode labels
    true_label_names = np.array(['UNKNOWN_CLASS'] * len(true_labels), dtype=object)
    if n_known > 0:
        true_label_names[known_mask] = encoder.inverse_transform(true_labels[known_mask])
    
    pred_label_names = encoder.inverse_transform(predictions)
    
    results_df = pd.DataFrame({
        'task': task_name,
        'true_label': true_label_names,
        'predicted_label': pred_label_names,
        'mean_confidence': confidences,
        'uncertainty': uncertainties,
        'correct': correct
    })
    
    # Log statistics (only for known classes)
    if n_known > 0:
        correct_known = correct[known_mask]
        n_correct = correct_known.sum()
        accuracy = n_correct / n_known
        
        uncertainty_correct = uncertainties[known_mask][correct_known]
        uncertainty_incorrect = uncertainties[known_mask][~correct_known]
        
        logger.info(f"\n{task_name} Results:")
        logger.info(f"  Known classes: {n_known} samples")
        logger.info(f"  Accuracy (known): {n_correct}/{n_known} ({accuracy:.1%})")
        logger.info(f"  Correct predictions - Uncertainty: {uncertainty_correct.mean():.4f} ± {uncertainty_correct.std():.4f}")
        if len(uncertainty_incorrect) > 0:
            logger.info(f"  Incorrect predictions - Uncertainty: {uncertainty_incorrect.mean():.4f} ± {uncertainty_incorrect.std():.4f}")
    else:
        logger.info(f"\n{task_name} Results:")
        logger.info(f"  Known classes: 0 samples")
        uncertainty_correct = np.array([])
        uncertainty_incorrect = np.array([])
    
    # Log unknown class statistics
    if n_unknown > 0:
        uncertainty_unknown = uncertainties[~known_mask]
        logger.info(f"  Unknown classes: {n_unknown} samples")
        logger.info(f"  Unknown class predictions - Uncertainty: {uncertainty_unknown.mean():.4f} ± {uncertainty_unknown.std():.4f}")
        
        # Compare unknown to correct/incorrect
        if len(uncertainty_correct) > 0:
            median_unknown = np.median(uncertainty_unknown)
            median_correct = np.median(uncertainty_correct)
            ratio_vs_correct = median_unknown / median_correct if median_correct > 0 else float('inf')
            logger.info(f"  Uncertainty ratio (unknown/correct): {ratio_vs_correct:.2f}x")
    
    if len(uncertainty_incorrect) > 0 and len(uncertainty_correct) > 0:
        # Cohen's d effect size
        pooled_std = np.sqrt(
            ((len(uncertainty_correct) - 1) * uncertainty_correct.var() +
             (len(uncertainty_incorrect) - 1) * uncertainty_incorrect.var()) /
            (len(uncertainty_correct) + len(uncertainty_incorrect) - 2)
        )
        cohens_d = (uncertainty_incorrect.mean() - uncertainty_correct.mean()) / pooled_std
        
        # Mann-Whitney U test
        statistic, p_value = stats.mannwhitneyu(uncertainty_incorrect, uncertainty_correct, alternative='greater')
        
        logger.info(f"  Cohen's d (effect size): {cohens_d:.3f}")
        logger.info(f"  Mann-Whitney U test p-value: {p_value:.4e}")
        
        # AUROC for using uncertainty to detect errors
        try:
            auroc = roc_auc_score(~correct, uncertainties)
            logger.info(f"  AUROC (uncertainty → error detection): {auroc:.3f}")
        except:
            logger.warning("  Could not calculate AUROC")
    
    return results_df


def plot_uncertainty_distributions(
    results_df: pd.DataFrame,
    output_dir: Path
):
    """
    Plot uncertainty distributions comparing correct vs incorrect predictions.
    """
    tasks = results_df['task'].unique()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, task in enumerate(tasks):
        task_data = results_df[results_df['task'] == task]
        
        ax = axes[idx]
        
        # Prepare data for violin plot
        correct_uncertainty = task_data[task_data['correct']]['uncertainty']
        incorrect_uncertainty = task_data[~task_data['correct']]['uncertainty']
        
        # Create violin plot
        parts = ax.violinplot(
            [correct_uncertainty, incorrect_uncertainty],
            positions=[1, 2],
            showmeans=True,
            showmedians=True
        )
        
        # Customize colors
        for pc in parts['bodies']:
            pc.set_facecolor('#8dd3c7')
            pc.set_alpha(0.7)
        
        # Add box plot on top for clarity
        bp = ax.boxplot(
            [correct_uncertainty, incorrect_uncertainty],
            positions=[1, 2],
            widths=0.3,
            patch_artist=True,
            boxprops=dict(facecolor='lightblue', alpha=0.5),
            medianprops=dict(color='red', linewidth=2)
        )
        
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Correct', 'Incorrect'])
        ax.set_ylabel('Uncertainty (Variance)')
        ax.set_title(f'{task.replace("_", " ").title()}\n'
                    f'n_correct={len(correct_uncertainty)}, '
                    f'n_incorrect={len(incorrect_uncertainty)}')
        ax.grid(axis='y', alpha=0.3)
        
        # Add statistical annotation
        if len(incorrect_uncertainty) > 0:
            _, p_value = stats.mannwhitneyu(
                incorrect_uncertainty, 
                correct_uncertainty, 
                alternative='greater'
            )
            
            y_max = max(correct_uncertainty.max(), incorrect_uncertainty.max())
            y_pos = y_max * 1.1
            
            if p_value < 0.001:
                sig_text = '***'
            elif p_value < 0.01:
                sig_text = '**'
            elif p_value < 0.05:
                sig_text = '*'
            else:
                sig_text = 'n.s.'
            
            ax.plot([1, 2], [y_pos, y_pos], 'k-', linewidth=1)
            ax.text(1.5, y_pos * 1.02, sig_text, ha='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    output_path = output_dir / "uncertainty_distributions.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved uncertainty distribution plot to {output_path}")


def plot_confidence_vs_uncertainty(
    results_df: pd.DataFrame,
    output_dir: Path
):
    """
    Scatter plot showing relationship between confidence and uncertainty.
    """
    tasks = results_df['task'].unique()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, task in enumerate(tasks):
        task_data = results_df[results_df['task'] == task]
        
        ax = axes[idx]
        
        # Separate correct and incorrect
        correct_data = task_data[task_data['correct']]
        incorrect_data = task_data[~task_data['correct']]
        
        # Scatter plot
        ax.scatter(
            correct_data['mean_confidence'],
            correct_data['uncertainty'],
            alpha=0.5,
            s=20,
            c='green',
            label=f'Correct (n={len(correct_data)})'
        )
        
        ax.scatter(
            incorrect_data['mean_confidence'],
            incorrect_data['uncertainty'],
            alpha=0.5,
            s=20,
            c='red',
            label=f'Incorrect (n={len(incorrect_data)})'
        )
        
        ax.set_xlabel('Mean Confidence')
        ax.set_ylabel('Uncertainty (Variance)')
        ax.set_title(f'{task.replace("_", " ").title()}')
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / "confidence_vs_uncertainty.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved confidence vs uncertainty plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze prediction uncertainty using MC Dropout"
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("results/full_training/cv_results/fold_0/best_model.pth"),
        help="Path to trained model"
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("data/matrices/training_matrix/unitigs.frac.mat"),
        help="Path to feature matrix for validation"
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/splits/train_metadata.tsv"),
        help="Path to validation metadata"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/mc_dropout_uncertainty.tsv"),
        help="Path to save results TSV"
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=Path("results/mc_dropout_plots"),
        help="Directory to save plots"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=50,
        help="Number of MC dropout forward passes (default: 50)"
    )
    parser.add_argument(
        "--use-val-split",
        action="store_true",
        help="Use 10%% validation split from training data (same as calibration)"
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state for validation split"
    )
    parser.add_argument(
        "--label-encoders",
        type=Path,
        default=None,
        help="Path to label_encoders.json (if not provided, will look in model directory)"
    )
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create output directories
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info("Loading data...")
    features, metadata = load_data(args.features, args.metadata)
    
    # Load pre-trained label encoders
    logger.info("Loading label encoders...")
    if args.label_encoders:
        label_encoders_path = args.label_encoders
    else:
        # Default: look in model directory
        label_encoders_path = args.model.parent / "label_encoders.json"
    
    if label_encoders_path.exists():
        trained_encoders, num_classes = load_label_encoders(label_encoders_path)
        logger.info(f"Loaded label encoders from {label_encoders_path}")
    else:
        logger.warning(f"Label encoders not found at {label_encoders_path}")
        logger.warning("Will create new encoders from data (may cause class mismatch!)")
        trained_encoders = None
        num_classes = None
    
    # Prepare labels
    logger.info("Preparing labels...")
    labels_dict, encoders, num_classes = prepare_labels(metadata, label_encoders=trained_encoders)
    
    # Report unknown class statistics (but don't filter them out!)
    if trained_encoders is not None:
        for task in labels_dict:
            n_unknown = (labels_dict[task] == -1).sum()
            n_known = (labels_dict[task] != -1).sum()
            if n_unknown > 0:
                logger.info(f"{task}: {n_known} known classes, {n_unknown} unknown classes (kept for OOD detection)")
    
    # Optionally use validation split
    if args.use_val_split:
        from sklearn.model_selection import train_test_split
        logger.info("Creating 10% validation split...")
        _, val_idx = train_test_split(
            np.arange(len(features)),
            test_size=0.1,
            random_state=args.random_state,
            stratify=labels_dict["sample_type"]
        )
        features = features[val_idx]
        labels_dict = {task: labels[val_idx] for task, labels in labels_dict.items()}
        metadata = metadata.iloc[val_idx].reset_index(drop=True)
        logger.info(f"Using {len(features)} validation samples")
    
    # Load trained model
    logger.info(f"Loading model from {args.model}...")
    checkpoint = torch.load(args.model, map_location=device)
    
    # Extract model state and hyperparameters
    if "model_state_dict" in checkpoint:
        model_state = checkpoint["model_state_dict"]
        hyperparams = checkpoint.get("hyperparams", {})
    else:
        model_state = checkpoint
        hyperparams = {}
    
    # Load hyperparameters from training config if not in checkpoint
    if not hyperparams:
        config_path = args.model.parent / "final_training_config.json"
        if config_path.exists():
            import json
            with open(config_path) as f:
                config = json.load(f)
            hyperparams = config.get("hyperparameters", {}).get("model_params", {})
            logger.info(f"Loaded hyperparameters from {config_path}")
        else:
            logger.warning(f"Config not found at {config_path}, using defaults")
    
    # Extract model architecture
    if "n_layers" in hyperparams:
        hidden_dims = [
            hyperparams[f"hidden_dim_{i}"]
            for i in range(hyperparams["n_layers"])
        ]
    else:
        hidden_dims = hyperparams.get("hidden_dims", [512, 256, 128])
    
    # Initialize model
    model = MultiTaskMLP(
        input_dim=features.shape[1],
        hidden_dims=hidden_dims,
        num_classes=num_classes,
        dropout=hyperparams.get("dropout", 0.5),
        use_batch_norm=hyperparams.get("use_batch_norm", True),
        activation=hyperparams.get("activation", "relu")
    )
    
    model.load_state_dict(model_state)
    model.to(device)
    
    logger.info(f"Model loaded successfully ({model.get_num_parameters():,} parameters)")
    logger.info(f"Dropout rate: {hyperparams.get('dropout', 0.5):.3f}")
    
    # Run MC Dropout inference
    logger.info(f"\n{'='*80}")
    logger.info(f"MONTE CARLO DROPOUT UNCERTAINTY ANALYSIS")
    logger.info(f"{'='*80}")
    logger.info(f"Samples: {len(features)}")
    logger.info(f"MC iterations: {args.n_samples}")
    
    mean_probs, uncertainties = mc_dropout_predict(
        model,
        features,
        device,
        n_samples=args.n_samples
    )
    
    # Analyze each task
    all_results = []
    
    logger.info(f"\n{'='*80}")
    logger.info("TASK-SPECIFIC UNCERTAINTY ANALYSIS")
    logger.info(f"{'='*80}")
    
    for task in model.targets:
        task_results = analyze_task_uncertainty(
            labels_dict[task],
            mean_probs[task],
            uncertainties[task],
            encoders[task],
            task
        )
        all_results.append(task_results)
    
    # Combine all results
    results_df = pd.concat(all_results, ignore_index=True)
    
    # Add sample IDs if available
    if 'sample_id' in metadata.columns:
        # Repeat sample_ids for each task
        sample_ids = metadata['sample_id'].values
        results_df.insert(0, 'sample_id', np.tile(sample_ids, len(model.targets)))
    
    # Save results
    results_df.to_csv(args.output, sep='\t', index=False)
    logger.info(f"\nSaved results to {args.output}")
    
    # Generate plots
    logger.info("\nGenerating visualization plots...")
    plot_uncertainty_distributions(results_df, args.plots_dir)
    plot_confidence_vs_uncertainty(results_df, args.plots_dir)
    
    # Summary statistics
    logger.info(f"\n{'='*80}")
    logger.info("OVERALL SUMMARY")
    logger.info(f"{'='*80}")
    
    for task in model.targets:
        task_data = results_df[results_df['task'] == task]
        correct_mask = task_data['correct']
        
        unc_correct = task_data[correct_mask]['uncertainty']
        unc_incorrect = task_data[~correct_mask]['uncertainty']
        
        if len(unc_incorrect) > 0:
            separation_ratio = unc_incorrect.median() / unc_correct.median()
            logger.info(f"\n{task}:")
            logger.info(f"  Median uncertainty ratio (incorrect/correct): {separation_ratio:.2f}x")
            
            if separation_ratio > 2.0:
                logger.info(f"  ✓ EXCELLENT separation - MC Dropout highly useful")
            elif separation_ratio > 1.5:
                logger.info(f"  ✓ GOOD separation - MC Dropout useful")
            elif separation_ratio > 1.2:
                logger.info(f"  ~ MODERATE separation - MC Dropout somewhat useful")
            else:
                logger.info(f"  ✗ POOR separation - MC Dropout may not help much")
    
    logger.info(f"\n{'='*80}")
    logger.info("Analysis complete!")
    logger.info(f"{'='*80}")


if __name__ == "__main__":
    main()
