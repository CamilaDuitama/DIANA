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


def format_predictions(predictions: dict, class_names: dict = None) -> dict:
    """
    Format raw predictions with human-readable labels.
    
    Args:
        predictions: Raw predictions from Predictor.
        class_names: Optional mapping of target -> class_idx -> name.
        
    Returns:
        Formatted predictions dictionary.
    """
    if class_names is None:
        # Default class names (update based on your actual classes)
        class_names = {
            'sample_type': ['ancient', 'modern'],
            'damage_pattern': ['low', 'medium', 'high'],
            'contamination_level': ['low', 'high']
        }
    
    formatted = {}
    for target, pred in predictions.items():
        if isinstance(pred, dict):
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
    
    args = parser.parse_args()
    
    # Infer sample ID from filename if not provided
    sample_id = args.sample_id or args.input.stem.replace('_unitig_fraction', '')
    
    # Load model
    logger.info(f"Loading model from {args.model}")
    device = None if args.device == 'auto' else args.device
    predictor = Predictor(args.model, device=device)
    
    # Load features
    features = load_unitig_fractions(args.input)
    
    # Run inference
    logger.info("Running inference...")
    predictions = predictor.predict(features, return_probabilities=True)
    
    # Format output
    formatted_preds = format_predictions(predictions)
    
    output = {
        'sample_id': sample_id,
        'input_file': str(args.input),
        'model_path': str(args.model),
        'predictions': formatted_preds
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
        logger.info(f"  Predicted: {pred['predicted_class']} (confidence: {pred.get('confidence', 'N/A'):.3f})")
        if 'probabilities' in pred:
            logger.info("  Probabilities:")
            for class_name, prob in pred['probabilities'].items():
                logger.info(f"    {class_name}: {prob:.3f}")
    logger.info("="*60)


if __name__ == '__main__':
    main()
