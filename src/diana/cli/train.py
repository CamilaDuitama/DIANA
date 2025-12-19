#!/usr/bin/env python3
"""
DIANA Multi-Task Training Pipeline - Command Line Interface
============================================================

High-level CLI for training multi-task classifiers on ancient DNA data.
Orchestrates the complete workflow from hyperparameter optimization to
final model training and evaluation.

DEPENDENCIES:
-------------
Python packages:
  - All dependencies from diana.training, diana.models, diana.data
  - argparse, pathlib, subprocess, logging

Input files:
  - K-mer matrix (.pa.mat)
  - Metadata (.tsv)

USAGE:
------
# Full workflow (hyperparameter optimization + final training):
diana-train multitask \\
    --features data/splits/train_matrix.pa.mat \\
    --metadata data/splits/train_metadata.tsv \\
    --output results/experiments/multitask/run_001 \\
    --mode full

# Hyperparameter optimization only:
diana-train multitask \\
    --features data/splits/train_matrix.pa.mat \\
    --metadata data/splits/train_metadata.tsv \\
    --output results/experiments/multitask/run_001 \\
    --mode optimize \\
    --n-folds 5 \\
    --n-trials 50 \\
    --max-epochs 200

# Train final model with specific hyperparameters:
diana-train multitask \\
    --features data/splits/train_matrix.pa.mat \\
    --metadata data/splits/train_metadata.tsv \\
    --output results/experiments/multitask/final_model \\
    --mode train \\
    --hyperparams results/experiments/multitask/run_001/best_hyperparameters.json

# Evaluate on test set:
diana-train multitask \\
    --features data/splits/test_matrix.pa.mat \\
    --metadata data/splits/test_metadata.tsv \\
    --output results/experiments/multitask/final_model \\
    --mode evaluate \\
    --model results/experiments/multitask/final_model/best_model.pth

MODES:
------
optimize  - Run cross-validation with hyperparameter search
train     - Train final model with given hyperparameters
evaluate  - Evaluate trained model on test set
full      - Complete workflow: optimize → train → evaluate

WORKFLOW:
---------
1. Optimize mode:
   - Runs nested cross-validation with Optuna
   - Saves best hyperparameters and per-fold results
   - Can submit SLURM jobs for parallel GPU training

2. Train mode:
   - Trains final model on full training set
   - Uses best hyperparameters from optimization
   - Saves model weights and training history

3. Evaluate mode:
   - Loads trained model
   - Evaluates on test set
   - Generates comprehensive metrics and plots

4. Full mode:
   - Automatically runs optimize → train → evaluate
   - Produces production-ready model

COMPARISON TO SCRIPTS:
----------------------
This CLI is a high-level wrapper that calls:
  - scripts/training/07_train_multitask_single_fold.py (via SLURM or local)
  - scripts/evaluation/08_collect_multitask_results.py
  - Additional training/evaluation logic

Benefits:
  - Single command for complete workflow
  - Automatic path management and result organization
  - Better error handling and logging
  - Easier to use than running individual scripts
"""

import sys
import argparse
import logging
import json
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

# Add src to path if running as script
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from diana.utils.config import setup_logging

logger = logging.getLogger(__name__)


class MultiTaskTrainingPipeline:
    """
    Orchestrates multi-task classifier training workflow.
    
    Handles:
      - Hyperparameter optimization (cross-validation + Optuna)
      - Final model training
      - Model evaluation on test set
      - Result collection and aggregation
      - SLURM job submission for GPU training
    """
    
    def __init__(
        self,
        features_path: Path,
        metadata_path: Path,
        output_dir: Path,
        use_gpu: bool = True,
        use_slurm: bool = False
    ):
        """
        Initialize training pipeline.
        
        Args:
            features_path: Path to k-mer matrix
            metadata_path: Path to metadata TSV
            output_dir: Output directory for all results
            use_gpu: Whether to use GPU (if available)
            use_slurm: Whether to submit SLURM jobs for parallel training
        """
        self.features_path = Path(features_path)
        self.metadata_path = Path(metadata_path)
        self.output_dir = Path(output_dir)
        self.use_gpu = use_gpu
        self.use_slurm = use_slurm
        
        # Create output directory structure
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cv_dir = self.output_dir / "cv_results"
        self.cv_dir.mkdir(exist_ok=True)
        
    def optimize_hyperparameters(
        self,
        n_folds: int = 5,
        n_trials: int = 50,
        max_epochs: int = 200,
        n_inner_splits: int = 3
    ) -> Dict[str, Any]:
        """
        Run hyperparameter optimization with cross-validation.
        
        Args:
            n_folds: Number of outer CV folds
            n_trials: Number of Optuna trials per fold
            max_epochs: Maximum epochs per trial
            n_inner_splits: Number of inner CV splits
            
        Returns:
            Dictionary with best hyperparameters and CV results
        """
        logger.info("Starting hyperparameter optimization...")
        logger.info(f"  Folds: {n_folds}, Trials: {n_trials}, Max Epochs: {max_epochs}")
        
        if self.use_slurm:
            return self._optimize_slurm(n_folds, n_trials, max_epochs, n_inner_splits)
        else:
            return self._optimize_local(n_folds, n_trials, max_epochs, n_inner_splits)
    
    def _optimize_slurm(
        self,
        n_folds: int,
        n_trials: int,
        max_epochs: int,
        n_inner_splits: int
    ) -> Dict[str, Any]:
        """Submit SLURM array job for parallel hyperparameter optimization."""
        logger.info("Submitting SLURM array job...")
        
        # Set environment variables for SLURM script
        env = {
            "FEATURES": str(self.features_path),
            "METADATA": str(self.metadata_path),
            "OUTPUT_DIR": str(self.cv_dir),
            "TOTAL_FOLDS": str(n_folds),
            "N_TRIALS": str(n_trials),
            "MAX_EPOCHS": str(max_epochs),
            "N_INNER_SPLITS": str(n_inner_splits)
        }
        
        # Submit job
        sbatch_script = Path(__file__).parent.parent.parent.parent / "scripts" / "training" / "run_multitask_gpu.sbatch"
        cmd = ["sbatch", f"--array=0-{n_folds-1}", str(sbatch_script)]
        
        result = subprocess.run(cmd, env={**subprocess.os.environ, **env}, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"SLURM submission failed: {result.stderr}")
        
        job_id = result.stdout.strip().split()[-1]
        logger.info(f"Submitted SLURM job: {job_id}")
        logger.info(f"Monitor with: squeue -j {job_id}")
        logger.info(f"After completion, run with --mode train to train final model")
        
        return {"job_id": job_id, "status": "submitted"}
    
    def _optimize_local(
        self,
        n_folds: int,
        n_trials: int,
        max_epochs: int,
        n_inner_splits: int
    ) -> Dict[str, Any]:
        """Run hyperparameter optimization locally (sequential folds)."""
        from diana.training.multitask_trainer import train_single_fold
        
        logger.info("Running local hyperparameter optimization (sequential)...")
        
        results = []
        for fold_id in range(n_folds):
            logger.info(f"Training fold {fold_id + 1}/{n_folds}...")
            
            fold_result = train_single_fold(
                fold_id=fold_id,
                total_folds=n_folds,
                features_path=self.features_path,
                metadata_path=self.metadata_path,
                output_dir=self.cv_dir,
                n_trials=n_trials,
                max_epochs=max_epochs,
                n_inner_splits=n_inner_splits,
                use_gpu=self.use_gpu
            )
            results.append(fold_result)
        
        # Aggregate results
        best_params = self._aggregate_cv_results(results)
        
        return best_params
    
    def _aggregate_cv_results(self, results: list) -> Dict[str, Any]:
        """Aggregate cross-validation results and identify best hyperparameters."""
        # This would call the logic from 08_collect_multitask_results.py
        logger.info("Aggregating cross-validation results...")
        # TODO: Implement aggregation logic
        return {}
    
    def train_final_model(
        self,
        hyperparams: Optional[Dict[str, Any]] = None,
        hyperparams_file: Optional[Path] = None
    ) -> Path:
        """
        Train final model on full training set.
        
        Args:
            hyperparams: Dictionary of hyperparameters
            hyperparams_file: Path to JSON file with hyperparameters
            
        Returns:
            Path to saved model
        """
        logger.info("Training final model on full training set...")
        
        if hyperparams_file:
            with open(hyperparams_file) as f:
                hyperparams = json.load(f)
        
        if not hyperparams:
            raise ValueError("Must provide hyperparameters or hyperparams_file")
        
        # TODO: Implement final model training
        logger.info(f"Using hyperparameters: {hyperparams}")
        
        model_path = self.output_dir / "final_model.pth"
        logger.info(f"Model saved to: {model_path}")
        
        return model_path
    
    def evaluate_model(
        self,
        model_path: Path,
        test_features: Path,
        test_metadata: Path
    ) -> Dict[str, Any]:
        """
        Evaluate trained model on test set.
        
        Args:
            model_path: Path to trained model
            test_features: Path to test k-mer matrix
            test_metadata: Path to test metadata
            
        Returns:
            Dictionary of test metrics
        """
        logger.info("Evaluating model on test set...")
        
        # TODO: Implement evaluation logic
        
        metrics = {}
        logger.info(f"Test results: {metrics}")
        
        return metrics


def main():
    """Command-line interface entry point."""
    parser = argparse.ArgumentParser(
        description="DIANA Multi-Task Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='task', help='Training task')
    
    # Multi-task classifier
    multitask_parser = subparsers.add_parser('multitask', help='Train multi-task classifier')
    multitask_parser.add_argument('--features', type=Path, required=True,
                                   help='Path to k-mer matrix (.pa.mat)')
    multitask_parser.add_argument('--metadata', type=Path, required=True,
                                   help='Path to metadata TSV')
    multitask_parser.add_argument('--output', type=Path, required=True,
                                   help='Output directory for results')
    multitask_parser.add_argument('--mode', choices=['optimize', 'train', 'evaluate', 'full'],
                                   default='full', help='Training mode')
    multitask_parser.add_argument('--n-folds', type=int, default=5,
                                   help='Number of CV folds for optimization')
    multitask_parser.add_argument('--n-trials', type=int, default=50,
                                   help='Number of Optuna trials per fold')
    multitask_parser.add_argument('--max-epochs', type=int, default=200,
                                   help='Maximum epochs per trial')
    multitask_parser.add_argument('--use-slurm', action='store_true',
                                   help='Submit SLURM jobs for parallel training')
    multitask_parser.add_argument('--hyperparams', type=Path,
                                   help='Path to hyperparameters JSON (for train mode)')
    multitask_parser.add_argument('--model', type=Path,
                                   help='Path to trained model (for evaluate mode)')
    multitask_parser.add_argument('--test-features', type=Path,
                                   help='Path to test matrix (for evaluate mode)')
    multitask_parser.add_argument('--test-metadata', type=Path,
                                   help='Path to test metadata (for evaluate mode)')
    
    args = parser.parse_args()
    
    if not args.task:
        parser.print_help()
        return
    
    # Setup logging
    setup_logging(args.output / "diana_train.log")
    
    # Initialize pipeline
    pipeline = MultiTaskTrainingPipeline(
        features_path=args.features,
        metadata_path=args.metadata,
        output_dir=args.output,
        use_slurm=args.use_slurm
    )
    
    # Run requested mode
    if args.mode in ['optimize', 'full']:
        results = pipeline.optimize_hyperparameters(
            n_folds=args.n_folds,
            n_trials=args.n_trials,
            max_epochs=args.max_epochs
        )
        
        if args.use_slurm:
            logger.info("SLURM job submitted. Exiting.")
            return
    
    if args.mode in ['train', 'full']:
        model_path = pipeline.train_final_model(
            hyperparams_file=args.hyperparams
        )
    
    if args.mode in ['evaluate', 'full']:
        if not args.test_features or not args.test_metadata:
            logger.error("--test-features and --test-metadata required for evaluation")
            return
        
        metrics = pipeline.evaluate_model(
            model_path=args.model or model_path,
            test_features=args.test_features,
            test_metadata=args.test_metadata
        )
    
    logger.info("Pipeline complete!")


if __name__ == '__main__':
    main()
