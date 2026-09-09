#!/usr/bin/env python3
"""
Run inference on new samples using trained Diana model.

This script takes unitig fraction files (output from shell scripts)
and uses the trained model to predict sample classifications.

Usage:
    python 03_run_inference.py \\
        --model results/training/best_model.pth \\
        --input sample_unitig_fraction.txt \\
        --output predictions.json

Input format (unitig_fraction.txt):
    Each line: <fraction_value>
    One value per unitig, matching training matrix order (107,480 lines)

Output format (predictions.json):
    {
        "sample_id": "sample_name",
        "predictions": {
            "sample_type": {
                "class": 0,  # 0=ancient, 1=modern
                "probabilities": [0.95, 0.05]
            },
            "damage_pattern": {...},
            "contamination_level": {...}
        }
    }
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from diana.inference.predictor import Predictor
from diana.models.multitask_mlp import denormalize_regression

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def load_unitig_fractions(fraction_file: Path) -> np.ndarray:
    """
    Load unitig fractions from text file.
    
    Each line contains one fraction value (0.0-1.0).
    Must have exactly 107,480 lines (matching training matrix).
    
    Args:
        fraction_file: Path to unitig fraction file from shell pipeline.
        
    Returns:
        NumPy array of shape (107480,) with fraction values.
    """
    logger.info(f"Loading unitig fractions from {fraction_file}")
    
    if not fraction_file.exists():
        raise FileNotFoundError(f"Fraction file not found: {fraction_file}")
    
    fractions = []
    with open(fraction_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    frac = float(line)
                    fractions.append(frac)
                except ValueError:
                    raise ValueError(f"Invalid fraction value: {line}")
    
    features = np.array(fractions, dtype=np.float32)
    logger.info(f"Loaded {len(features)} unitig fractions")
    logger.info(f"  Non-zero features: {np.sum(features > 0)} ({100 * np.mean(features > 0):.2f}%)")
    logger.info(f"  Mean fraction: {np.mean(features):.4f}")
    logger.info(f"  Max fraction: {np.max(features):.4f}")
    
    # Validate dimensions
    expected_dim = 107480
    if len(features) != expected_dim:
        raise ValueError(
            f"Expected {expected_dim} unitig fractions, got {len(features)}. "
            f"Make sure you're using the same MUSET output as training."
        )
    
    return features


def load_class_names(label_encoders_path: Path) -> dict:
    """
    Load class names from label_encoders.json.

    Expected format: {"task": {"classes": ["class0", "class1", ...]}}

    Returns:
        Mapping of task -> list of class names.
    """
    label_encoders_path = Path(label_encoders_path)
    if label_encoders_path.is_dir():
        label_encoders_path = label_encoders_path / "label_encoders.json"
    with open(label_encoders_path) as f:
        encoders = json.load(f)
    return {task: info['classes'] for task, info in encoders.items() if 'classes' in info}


def load_regression_bounds(label_encoders_path: Path) -> dict:
    """Load normalisation bounds for regression tasks from label_encoders.json.

    Regression entries carry {'min', 'max', 'log_transform'} instead of 'classes'.

    Returns:
        Mapping of task -> bounds dict, used to return predictions in original units.
    """
    label_encoders_path = Path(label_encoders_path)
    if label_encoders_path.is_dir():
        label_encoders_path = label_encoders_path / "label_encoders.json"
    with open(label_encoders_path) as f:
        encoders = json.load(f)
    return {task: info for task, info in encoders.items() if 'classes' not in info}



def assess_representation(fractions: np.ndarray) -> dict:
    """Judge whether this sample can be represented in the feature space at all.

    A unitig vector of all zeros does not mean "this sample shares nothing with the
    reference". It means there was effectively nothing to compare: the input was
    truncated, or its reads are shorter than k=31 and so yield no k-mers at all.
    Two of the runs we have seen (SRR867035, SRR867040) have modal read length
    24-26 bp; another (ERR1883466) assembled to 563 nucleotides in total.

    A softmax over zeros still returns a confident-looking answer, so this has to
    be surfaced explicitly or the caller will believe it.
    """
    n = int(fractions.size)
    nonzero = int((fractions > 0).sum())
    coverage = nonzero / n if n else 0.0

    if nonzero == 0:
        return {
            "status": "unrepresentable",
            "nonzero_unitigs": 0,
            "total_unitigs": n,
            "coverage": 0.0,
            "message": (
                "No reference k-mers were detected in this sample, so every unitig "
                "feature is zero. DIANA cannot classify it, and any label below is "
                "an artefact of the model's prior rather than evidence from the "
                "data. The usual causes are a truncated or near-empty input, or "
                "reads shorter than k=31 (ancient libraries are often 24-30 bp), "
                "which yield no k-mers by construction. Check the input depth and "
                "read-length distribution before interpreting anything here."
            ),
        }
    if coverage < 0.01:
        return {
            "status": "poorly_represented",
            "nonzero_unitigs": nonzero,
            "total_unitigs": n,
            "coverage": coverage,
            "message": (
                f"Only {nonzero} of {n} unitigs ({100 * coverage:.2f}%) carry any "
                "signal. This is far below what the model was trained on, so the "
                "predictions are unreliable and their confidences should not be "
                "read as calibrated. Treat them as a hint, not a result."
            ),
        }
    return {
        "status": "ok",
        "nonzero_unitigs": nonzero,
        "total_unitigs": n,
        "coverage": coverage,
        "message": "",
    }


def format_predictions(predictions: dict, class_names: dict = None,
                       regression_bounds: dict = None) -> dict:
    """
    Format raw predictions with human-readable labels.
    
    Args:
        predictions: Raw predictions from Predictor.
        class_names: Optional mapping of target -> list of class names.
        
    Returns:
        Formatted predictions dictionary.
    """
    if class_names is None:
        # Fallback defaults (only sample_type is reliable without label_encoders)
        class_names = {
            'sample_type': ['ancient_metagenome', 'modern_metagenome'],
        }
    
    regression_bounds = regression_bounds or {}

    formatted = {}
    for target, pred in predictions.items():
        if isinstance(pred, dict) and 'value' in pred:
            # Regression head: model emits a normalised [0, 1] scalar.
            normalized = float(pred['value'])
            entry = {'normalized_value': normalized}
            if target in regression_bounds:
                entry['predicted_value'] = denormalize_regression(
                    normalized, regression_bounds[target]
                )
            formatted[target] = entry
        elif isinstance(pred, dict):
            # Has probabilities
            class_idx = pred['class']
            probs = pred['probabilities']
            
            formatted[target] = {
                'predicted_class': class_names.get(target, [str(i) for i in range(len(probs))])[class_idx],
                'class_index': class_idx,
                'probabilities': {
                    name: prob
                    for name, prob in zip(
                        class_names.get(target, [str(i) for i in range(len(probs))]),
                        probs
                    )
                },
                'confidence': float(np.max(probs))
            }
        else:
            # Just class index
            formatted[target] = {
                'predicted_class': class_names.get(target, [str(pred)])[pred],
                'class_index': int(pred)
            }
    
    return formatted


def main():
    parser = argparse.ArgumentParser(
        description="Run inference on new sample using trained Diana model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--model',
        type=Path,
        required=True,
        help='Path to trained model checkpoint (.pth file)'
    )
    
    parser.add_argument(
        '--input',
        type=Path,
        required=True,
        help='Path to unitig fraction file (output from 02_aggregate_to_unitigs.sh)'
    )
    
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Path to output JSON file with predictions'
    )
    
    parser.add_argument(
        '--sample-id',
        type=str,
        default=None,
        help='Sample identifier (default: inferred from input filename)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu', 'auto'],
        default='auto',
        help='Device to use for inference (default: auto-detect)'
    )

    parser.add_argument(
        '--label-encoders',
        type=Path,
        default=None,
        help='Path to label_encoders.json for human-readable class names'
    )
    
    args = parser.parse_args()
    
    # Infer sample ID from filename if not provided
    sample_id = args.sample_id or args.input.stem.replace('_unitig_fraction', '')
    
    # Load model
    logger.info(f"Loading model from {args.model}")
    device = None if args.device == 'auto' else args.device
    predictor = Predictor(args.model, device=device)
    
    # Load features
    features = load_unitig_fractions(args.input)
    
    # Load class names from label encoders if provided
    class_names = None
    regression_bounds = None
    if args.label_encoders and args.label_encoders.exists():
        logger.info(f"Loading class names from {args.label_encoders}")
        class_names = load_class_names(args.label_encoders)
        regression_bounds = load_regression_bounds(args.label_encoders)
        if regression_bounds:
            logger.info(f"Regression tasks: {sorted(regression_bounds)}")
    else:
        logger.warning("No --label-encoders provided; non-sample_type tasks will use numeric class indices")

    # Can this sample be represented at all? An all-zero vector still yields a
    # confident-looking softmax, so say so before reporting any label.
    representation = assess_representation(features)
    if representation["status"] != "ok":
        logger.warning("=" * 70)
        logger.warning("SAMPLE %s: %s", sample_id, representation["status"].upper())
        logger.warning("%s", representation["message"])
        logger.warning("=" * 70)

    # Run inference
    logger.info("Running inference...")
    predictions = predictor.predict(features, return_probabilities=True)
    
    # Format output
    formatted_preds = format_predictions(predictions, class_names=class_names,
                                         regression_bounds=regression_bounds)
    
    output = {
        'sample_id': sample_id,
        'input_file': str(args.input),
        'model_path': str(args.model),
        # Machine-readable, so downstream consumers can filter on it rather than
        # having to parse the log.
        'representation': representation,
        'predictions': ({} if representation['status'] == 'unrepresentable'
                        else formatted_preds),
        'predictions_suppressed': representation['status'] == 'unrepresentable',
    }
    
    # Save results
    logger.info(f"Saving predictions to {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    
    # Print summary
    logger.info("\\n" + "="*60)
    logger.info(f"Predictions for sample: {sample_id}")
    logger.info("="*60)
    for target, pred in formatted_preds.items():
        logger.info(f"\\n{target}:")
        if 'normalized_value' in pred:
            if 'predicted_value' in pred:
                logger.info(f"  Predicted: {pred['predicted_value']:.3f} "
                            f"(normalised {pred['normalized_value']:.3f})")
            else:
                logger.info(f"  Predicted (normalised): {pred['normalized_value']:.3f}")
            continue
        logger.info(f"  Predicted: {pred['predicted_class']} (confidence: {pred.get('confidence', 'N/A'):.3f})")
        if 'probabilities' in pred:
            logger.info("  Probabilities:")
            for class_name, prob in pred['probabilities'].items():
                logger.info(f"    {class_name}: {prob:.3f}")
    logger.info("="*60)


if __name__ == '__main__':
    main()
