#!/usr/bin/env python3
"""
Temperature Scaling for Multi-Task Model Calibration
====================================================

Learns optimal temperature parameters for post-hoc calibration of confidence scores.
Uses the same 90/10 internal validation split as used during training.

PROBLEM:
--------
Neural networks often produce overconfident predictions - showing high confidence
even for incorrect predictions. Temperature scaling fixes this by learning a single
parameter T per task that recalibrates confidence scores.

TEMPERATURE SCALING:
-------------------
Instead of: P(y) = softmax(z) = exp(z_i) / Σ exp(z_j)
We use:     P(y) = softmax(z/T) = exp(z_i/T) / Σ exp(z_j/T)

where:
- z = logits (pre-softmax outputs from model)
- T = temperature parameter (T > 1 makes predictions less confident, T < 1 more confident)
- T is optimized per task to minimize Expected Calibration Error (ECE)

EXPECTED CALIBRATION ERROR (ECE):
---------------------------------
Measures how well predicted confidence matches actual accuracy.
- Bin predictions by confidence (e.g., [0.0-0.1], [0.1-0.2], ..., [0.9-1.0])
- For each bin: ECE += |bin_accuracy - bin_confidence| * (bin_size / total)
- Perfect calibration: ECE = 0 (confidence = accuracy for all bins)

WORKFLOW:
---------
1. Load training data and recreate 90/10 split used during training
2. Load trained model (best_model.pth)
3. Get raw logits on 10% internal validation set
4. For each task:
   - Optimize temperature T to minimize ECE using scipy.optimize
   - Log ECE before/after calibration
   - Generate reliability diagram
5. Save optimal temperatures to models/optimal_temperatures.json

USAGE:
------
# Basic usage (uses default paths from training):
python scripts/calibration/01_calibrate_model.py

# Custom paths:
python scripts/calibration/01_calibrate_model.py \\
    --model models/checkpoints/best_model.pth \\
    --features data/splits/train_matrix.pa.mat \\
    --metadata data/splits/train_metadata.tsv \\
    --output models/optimal_temperatures.json

OUTPUT:
-------
models/optimal_temperatures.json:
{
    "sample_type": 1.23,
    "community_type": 1.45,
    "sample_host": 1.67,
    "material": 1.89,
    "metadata": {
        "calibration_date": "2026-01-21",
        "model_path": "models/checkpoints/best_model.pth",
        "n_validation_samples": 261
    }
}

REPRODUCIBILITY:
---------------
- Uses same random_state=42 as training script
- Uses same 90/10 split ratio as training validation
- Ensures temperatures are learned on same data model was validated on
"""

import sys
import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from scipy.optimize import minimize
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


def prepare_labels(
    metadata: pd.DataFrame,
    targets: List[str] = ["sample_type", "community_type", "sample_host", "material"]
) -> Tuple[Dict[str, np.ndarray], Dict[str, LabelEncoder], Dict[str, int]]:
    """Encode labels for all classification targets."""
    labels = {}
    encoders = {}
    num_classes = {}
    
    for target in targets:
        encoder = LabelEncoder()
        labels[target] = encoder.fit_transform(metadata[target].values)
        encoders[target] = encoder
        num_classes[target] = len(encoder.classes_)
        logger.info(f"{target}: {num_classes[target]} classes")
    
    return labels, encoders, num_classes


def create_validation_split(
    features: np.ndarray,
    labels_dict: Dict[str, np.ndarray],
    val_size: float = 0.1,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Create 90/10 train/val split using same strategy as training.
    Stratified by sample_type to match training procedure.
    """
    # Stratification must match the main training pipeline (currently sample_type)
    # If you change stratification in scripts/training/01_train_multitask_single_fold.py,
    # you MUST update it here too to ensure consistency
    train_idx, val_idx = train_test_split(
        np.arange(len(features)),
        test_size=val_size,
        random_state=random_state,
        stratify=labels_dict["sample_type"]
    )
    
    X_train = features[train_idx]
    X_val = features[val_idx]
    
    y_train = {task: labels[train_idx] for task, labels in labels_dict.items()}
    y_val = {task: labels[val_idx] for task, labels in labels_dict.items()}
    
    logger.info(f"Split: {len(train_idx)} train, {len(val_idx)} validation samples")
    
    return X_train, X_val, y_train, y_val


def get_logits(
    model: MultiTaskMLP,
    X: np.ndarray,
    device: torch.device,
    batch_size: int = 256
) -> Dict[str, np.ndarray]:
    """
    Get raw logits (pre-softmax outputs) from model.
    
    Args:
        model: Trained MultiTaskMLP
        X: Features array
        device: Torch device
        batch_size: Batch size for inference
        
    Returns:
        Dictionary mapping task names to logits arrays
    """
    model.eval()
    logits_dict = {task: [] for task in model.targets}
    
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.FloatTensor(X[i:i + batch_size]).to(device)
            outputs = model(batch)
            
            for task, logits in outputs.items():
                logits_dict[task].append(logits.cpu().numpy())
    
    # Concatenate batches
    logits_dict = {
        task: np.concatenate(logits_list, axis=0)
        for task, logits_list in logits_dict.items()
    }
    
    return logits_dict


def compute_ece(
    confidences: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15
) -> float:
    """
    Compute Expected Calibration Error (ECE).
    
    Args:
        confidences: Max confidence scores (probabilities)
        predictions: Predicted class indices
        labels: True class indices
        n_bins: Number of confidence bins
        
    Returns:
        ECE value (0 = perfect calibration, higher = worse calibration)
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        # Find samples in this confidence bin
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        
        if in_bin.sum() > 0:
            # Accuracy in this bin
            bin_acc = (predictions[in_bin] == labels[in_bin]).mean()
            # Average confidence in this bin
            bin_conf = confidences[in_bin].mean()
            # Bin weight (proportion of samples)
            bin_weight = in_bin.sum() / len(confidences)
            
            ece += bin_weight * abs(bin_acc - bin_conf)
    
    return ece


def optimize_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15
) -> Tuple[float, float, float]:
    """
    Optimize temperature parameter to minimize ECE.
    
    Args:
        logits: Raw logits from model (N x C)
        labels: True labels (N,)
        n_bins: Number of bins for ECE calculation
        
    Returns:
        Tuple of (optimal_temperature, ece_before, ece_after)
    """
    # ECE before calibration (temperature = 1.0)
    probs_before = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    confidences_before = probs_before.max(axis=1)
    predictions_before = probs_before.argmax(axis=1)
    ece_before = compute_ece(confidences_before, predictions_before, labels, n_bins)
    
    # Optimization objective: minimize ECE
    def objective(T):
        # Apply temperature scaling
        scaled_logits = logits / T[0]
        probs = np.exp(scaled_logits) / np.exp(scaled_logits).sum(axis=1, keepdims=True)
        confidences = probs.max(axis=1)
        predictions = probs.argmax(axis=1)
        return compute_ece(confidences, predictions, labels, n_bins)
    
    # Optimize temperature (search between 0.1 and 10.0)
    result = minimize(
        objective,
        x0=[1.0],  # Start at T=1.0 (no scaling)
        method='L-BFGS-B',
        bounds=[(0.1, 10.0)]
    )
    
    optimal_T = result.x[0]
    ece_after = result.fun
    
    return optimal_T, ece_before, ece_after


def plot_reliability_diagram(
    logits: np.ndarray,
    labels: np.ndarray,
    temperature: float,
    task_name: str,
    output_dir: Path,
    n_bins: int = 15
):
    """
    Plot reliability diagram showing calibration before/after temperature scaling.
    
    Args:
        logits: Raw logits
        labels: True labels
        temperature: Optimal temperature
        task_name: Name of the task
        output_dir: Directory to save plot
        n_bins: Number of confidence bins
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
    
    # Before calibration (T=1.0)
    probs_before = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    confidences_before = probs_before.max(axis=1)
    predictions_before = probs_before.argmax(axis=1)
    
    bin_accs_before = []
    bin_confs_before = []
    bin_counts_before = []
    
    for i in range(n_bins):
        in_bin = (confidences_before > bin_boundaries[i]) & (confidences_before <= bin_boundaries[i + 1])
        if in_bin.sum() > 0:
            bin_accs_before.append((predictions_before[in_bin] == labels[in_bin]).mean())
            bin_confs_before.append(confidences_before[in_bin].mean())
            bin_counts_before.append(in_bin.sum())
        else:
            bin_accs_before.append(0)
            bin_confs_before.append(bin_centers[i])
            bin_counts_before.append(0)
    
    # After calibration
    scaled_logits = logits / temperature
    probs_after = np.exp(scaled_logits) / np.exp(scaled_logits).sum(axis=1, keepdims=True)
    confidences_after = probs_after.max(axis=1)
    predictions_after = probs_after.argmax(axis=1)
    
    bin_accs_after = []
    bin_confs_after = []
    bin_counts_after = []
    
    for i in range(n_bins):
        in_bin = (confidences_after > bin_boundaries[i]) & (confidences_after <= bin_boundaries[i + 1])
        if in_bin.sum() > 0:
            bin_accs_after.append((predictions_after[in_bin] == labels[in_bin]).mean())
            bin_confs_after.append(confidences_after[in_bin].mean())
            bin_counts_after.append(in_bin.sum())
        else:
            bin_accs_after.append(0)
            bin_confs_after.append(bin_centers[i])
            bin_counts_after.append(0)
    
    # Plot before
    ece_before = compute_ece(confidences_before, predictions_before, labels, n_bins)
    ax1.bar(bin_centers, bin_accs_before, width=1/n_bins, alpha=0.7, label='Accuracy', edgecolor='black')
    ax1.plot([0, 1], [0, 1], 'r--', label='Perfect calibration')
    ax1.scatter(bin_confs_before, bin_accs_before, color='blue', s=50, zorder=5)
    ax1.set_xlabel('Confidence')
    ax1.set_ylabel('Accuracy')
    ax1.set_title(f'{task_name} - Before Calibration\nECE = {ece_before:.4f}')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Plot after
    ece_after = compute_ece(confidences_after, predictions_after, labels, n_bins)
    ax2.bar(bin_centers, bin_accs_after, width=1/n_bins, alpha=0.7, label='Accuracy', edgecolor='black')
    ax2.plot([0, 1], [0, 1], 'r--', label='Perfect calibration')
    ax2.scatter(bin_confs_after, bin_accs_after, color='blue', s=50, zorder=5)
    ax2.set_xlabel('Confidence')
    ax2.set_ylabel('Accuracy')
    ax2.set_title(f'{task_name} - After Calibration (T={temperature:.3f})\nECE = {ece_after:.4f}')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / f"reliability_diagram_{task_name}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved reliability diagram to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate multi-task model using temperature scaling"
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
        default=Path("data/matrices/large_matrix_3070_with_frac/unitigs.frac.mat"),
        help="Path to training feature matrix"
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/splits/train_metadata.tsv"),
        help="Path to training metadata"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/optimal_temperatures.json"),
        help="Path to save optimal temperatures JSON"
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=Path("models/calibration_plots"),
        help="Directory to save reliability diagrams"
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.1,
        help="Validation split size (default: 0.1 = 10%%)"
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state for reproducibility"
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=15,
        help="Number of bins for ECE calculation"
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
    
    # Prepare labels
    logger.info("Preparing labels...")
    labels_dict, encoders, num_classes = prepare_labels(metadata)
    
    # Create validation split (same as training)
    logger.info(f"Creating {args.val_size*100:.0f}% validation split...")
    X_train, X_val, y_train, y_val = create_validation_split(
        features,
        labels_dict,
        val_size=args.val_size,
        random_state=args.random_state
    )
    
    # Load trained model
    logger.info(f"Loading model from {args.model}...")
    
    # Load model config
    config_path = Path(args.model).parent / "final_training_config.json"
    if config_path.exists():
        logger.info(f"Loading config from {config_path}...")
        with open(config_path) as f:
            config = json.load(f)
        model_params = config["hyperparameters"]["model_params"]
        hidden_dims = model_params["hidden_dims"]
        dropout = model_params["dropout"]
        use_batch_norm = model_params["use_batch_norm"]
        activation = model_params["activation"]
    else:
        logger.warning(f"Config not found at {config_path}, using defaults")
        hidden_dims = [512, 256, 128]
        dropout = 0.5
        use_batch_norm = True
        activation = "relu"
    
    checkpoint = torch.load(args.model, map_location=device)
    
    # Extract model state
    if "model_state_dict" in checkpoint:
        model_state = checkpoint["model_state_dict"]
    else:
        model_state = checkpoint
    
    # Initialize model with same architecture as training
    model = MultiTaskMLP(
        input_dim=X_val.shape[1],
        hidden_dims=hidden_dims,
        num_classes=num_classes,
        dropout=dropout,
        use_batch_norm=use_batch_norm,
        activation=activation
    )
    
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()
    
    logger.info(f"Model loaded successfully ({model.get_num_parameters():,} parameters)")
    
    # Get logits on validation set
    logger.info("Extracting logits from validation set...")
    logits_dict = get_logits(model, X_val, device)
    
    # Optimize temperature for each task
    temperatures = {}
    results = {}
    
    logger.info("\n" + "="*80)
    logger.info("TEMPERATURE CALIBRATION RESULTS")
    logger.info("="*80)
    
    for task in model.targets:
        logger.info(f"\nCalibrating {task}...")
        
        optimal_T, ece_before, ece_after = optimize_temperature(
            logits_dict[task],
            y_val[task],
            n_bins=args.n_bins
        )
        
        temperatures[task] = optimal_T
        results[task] = {
            "temperature": optimal_T,
            "ece_before": ece_before,
            "ece_after": ece_after,
            "improvement": ece_before - ece_after
        }
        
        logger.info(f"  Optimal temperature: {optimal_T:.4f}")
        logger.info(f"  ECE before: {ece_before:.4f}")
        logger.info(f"  ECE after:  {ece_after:.4f}")
        logger.info(f"  Improvement: {ece_before - ece_after:.4f} ({(1 - ece_after/ece_before)*100:.1f}%)")
        
        # Generate reliability diagram
        plot_reliability_diagram(
            logits_dict[task],
            y_val[task],
            optimal_T,
            task,
            args.plots_dir,
            n_bins=args.n_bins
        )
    
    # Save temperatures to JSON
    output_data = {
        **temperatures,
        "metadata": {
            "calibration_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_path": str(args.model),
            "n_validation_samples": len(X_val),
            "val_size": args.val_size,
            "random_state": args.random_state,
            "n_bins": args.n_bins,
            "results": results
        }
    }
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Optimal temperatures saved to {args.output}")
    logger.info(f"Reliability diagrams saved to {args.plots_dir}")
    logger.info("="*80)


if __name__ == "__main__":
    main()
