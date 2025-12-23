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
  - K-mer matrix (.mat format - any matrix type)
  - Metadata (.tsv)

USAGE:
------
# Full workflow (hyperparameter optimization + final training):
diana-train multitask \\
    --config configs/experiment_001.yaml \\
    --output results/experiments/multitask/run_001 \\
    --mode full

# Hyperparameter optimization only:
diana-train multitask \\
    --config configs/experiment_001.yaml \\
    --output results/experiments/multitask/run_001 \\
    --mode optimize

# Train final model with specific hyperparameters:
diana-train multitask \\
    --config configs/experiment_001.yaml \\
    --output results/experiments/multitask/final_model \\
    --mode train \\
    --hyperparams results/experiments/multitask/run_001/cv_results/best_hyperparameters.json

MODES:
------
optimize  - Run cross-validation with hyperparameter search
train     - Train final model with given hyperparameters
full      - Complete workflow: optimize → train

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

3. Full mode:
   - Automatically runs optimize → train
   - Produces production-ready model
   - For evaluation, use diana-test CLI (separate)

COMPARISON TO SCRIPTS:
----------------------
This CLI is a high-level wrapper that calls:
  - scripts/training/07_train_multitask_single_fold.py (via SLURM or local)
  - scripts/evaluation/08_collect_multitask_results.py
  - Additional training logic

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

from diana.utils.config import setup_logging, load_config

logger = logging.getLogger(__name__)


class MultiTaskTrainingPipeline:
    """
    Orchestrates multi-task classifier training workflow.
    
    Handles:
      - Hyperparameter optimization (cross-validation + Optuna)
      - Final model training with validation-based early stopping
      - Result collection and aggregation
      - SLURM job submission for GPU training (config-driven)
    
    Note on Hyperparameter Aggregation:
      The best hyperparameters are determined by taking the mean (numeric) or mode
      (categorical) across CV folds. While this is a reasonable heuristic, it's not
      guaranteed to be optimal. For production, consider running a final focused
      Optuna study on the full training set with validation split to select the
      single best hyperparameter combination.
    """
    
    def __init__(
        self,
        output_dir: Path,
        config: Optional[Dict[str, Any]] = None,
        config_path: Optional[Path] = None
    ):
        """
        Initialize training pipeline.
        
        Args:
            output_dir: Output directory for all results
            config: Config dict with all parameters
            config_path: Path to config YAML file (alternative to config)
        """
        import torch
        
        self.output_dir = Path(output_dir)
        
        # Load configuration
        if config is not None:
            self.config = config
        elif config_path is not None:
            self.config = load_config(config_path)
        else:
            raise ValueError("Must provide either config or config_path")
        
        # Extract data paths from config
        data_config = self.config.get('data', {})
        self.features_path = Path(data_config.get('features_path', ''))
        self.metadata_path = Path(data_config.get('metadata_path', ''))
        
        if not self.features_path or not self.features_path.exists():
            raise ValueError(f"Invalid features_path in config: {self.features_path}")
        if not self.metadata_path or not self.metadata_path.exists():
            raise ValueError(f"Invalid metadata_path in config: {self.metadata_path}")
        
        # Extract training parameters from config
        training_config = self.config.get('training', {})
        self.use_gpu = training_config.get('use_gpu', True)
        self.use_slurm = training_config.get('use_slurm', False)
        
        # Setup device
        if self.use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device('cpu')
            logger.info("Using CPU")
        
        # Optional split path (for future use)
        self.split_path = None
        
        # Create output directory structure
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cv_dir = self.output_dir / "cv_results"
        self.cv_dir.mkdir(exist_ok=True)
        
    def optimize_hyperparameters(self) -> Dict[str, Any]:
        """
        Run hyperparameter optimization with cross-validation.
        
        All parameters are read from config:
        - training.n_folds: Number of outer CV folds
        - training.n_trials: Number of Optuna trials per fold
        - training.max_epochs: Maximum epochs per trial
        - training.n_inner_splits: Number of inner CV splits
            
        Returns:
            Dictionary with best hyperparameters and CV results
        """
        training_config = self.config.get('training', {})
        n_folds = training_config.get('n_folds', 5)
        n_trials = training_config.get('n_trials', 50)
        max_epochs = training_config.get('max_epochs', 200)
        n_inner_splits = training_config.get('n_inner_splits', 3)
        
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
        
        # Save run configuration for SLURM jobs
        run_config = {
            'features_path': str(self.features_path),
            'metadata_path': str(self.metadata_path),
            'output_dir': str(self.cv_dir),
            'n_folds': n_folds,
            'n_trials': n_trials,
            'max_epochs': max_epochs,
            'n_inner_splits': n_inner_splits,
            'task_names': self.config.get('model', {}).get('task_names', []),
            'task_weights': self.config.get('training', {}).get('task_weights', {})
        }
        
        config_file = self.cv_dir / "slurm_run_config.json"
        with open(config_file, 'w') as f:
            json.dump(run_config, f, indent=2)
        logger.info(f"Saved SLURM run config to {config_file}")
        
        # Get sbatch script path from config or use default
        sbatch_script = self.config.get('training', {}).get('sbatch_script')
        if sbatch_script is None:
            sbatch_script = Path(__file__).parent.parent.parent.parent / "scripts" / "training" / "run_multitask_gpu.sbatch"
        else:
            sbatch_script = Path(sbatch_script)
        
        import os
        
        # Submit job with config file path
        env = {"RUN_CONFIG": str(config_file)}
        cmd = ["sbatch", f"--array=0-{n_folds-1}", str(sbatch_script)]
        
        result = subprocess.run(cmd, env={**os.environ, **env}, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"SLURM submission failed: {result.stderr}")
        
        job_id = result.stdout.strip().split()[-1]
        logger.info(f"Submitted SLURM job: {job_id}")
        logger.info(f"Monitor with: squeue -j {job_id}")
        logger.info(f"Check logs in: {self.cv_dir.parent.parent / 'logs'}")
        logger.info("")
        logger.info("After job completion, run:")
        logger.info(f"  diana-train multitask --output {self.output_dir} --mode train")
        logger.info("")
        logger.info(f"To check job status: sacct -j {job_id} --format=JobID,State,ExitCode,Elapsed")
        
        return {"job_id": job_id, "status": "submitted", "config_file": str(config_file)}
    
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
        logger.info("Aggregating cross-validation results...")
        
        import numpy as np
        
        # Collect hyperparameters and metrics from all folds
        all_hyperparams = []
        all_metrics = []
        
        for fold_dir in self.cv_dir.glob("fold_*"):
            # Load results JSON
            result_files = list(fold_dir.glob("multitask_fold_*_results_*.json"))
            if not result_files:
                continue
                
            with open(result_files[0]) as f:
                fold_data = json.load(f)
            
            # Try both 'best_params' and 'best_hyperparameters' keys
            params = fold_data.get('best_params', fold_data.get('best_hyperparameters', {}))
            all_hyperparams.append(params)
            all_metrics.append(fold_data.get('test_metrics', {}))
        
        # Aggregate metrics (mean ± std across folds)
        aggregated_metrics = {}
        if all_metrics:
            for task in all_metrics[0].keys():
                aggregated_metrics[task] = {}
                for metric in all_metrics[0][task].keys():
                    values = [m[task][metric] for m in all_metrics]
                    aggregated_metrics[task][metric] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'values': values
                    }
        
        # Find most common hyperparameters (mode across folds)
        best_hyperparams = {}
        if all_hyperparams:
            # For each hyperparameter, use the most common value
            for key in all_hyperparams[0].keys():
                values = [h.get(key) for h in all_hyperparams if key in h]
                if values:
                    # For numeric values, use mean; for categorical, use mode
                    if isinstance(values[0], (int, float)):
                        best_hyperparams[key] = float(np.mean(values))
                    else:
                        # Mode for categorical
                        best_hyperparams[key] = max(set(values), key=values.count)
        
        # Save aggregated results
        output = {
            'aggregated_metrics': aggregated_metrics,
            'best_hyperparameters': best_hyperparams,
            'n_folds': len(all_metrics)
        }
        
        output_file = self.cv_dir / "aggregated_results.json"
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        
        # Save best hyperparameters separately
        hyperparam_file = self.cv_dir / "best_hyperparameters.json"
        with open(hyperparam_file, 'w') as f:
            json.dump(best_hyperparams, f, indent=2)
        
        logger.info(f"Saved aggregated results to {output_file}")
        logger.info(f"Saved best hyperparameters to {hyperparam_file}")
        
        return output
    
    def _train_final_slurm(self, hyperparams: Dict[str, Any]) -> Dict[str, Any]:
        """Submit SLURM job for final model training."""
        logger.info("Submitting SLURM job for final training...")
        
        # Save training configuration
        train_config = {
            'features_path': str(self.features_path),
            'metadata_path': str(self.metadata_path),
            'output_dir': str(self.output_dir),
            'hyperparameters': hyperparams,
            'task_names': self.config.get('model', {}).get('task_names', []),
            'task_weights': self.config.get('training', {}).get('task_weights', {}),
            'validation_split': self.config.get('training', {}).get('validation_split', 0.1),
            'max_epochs': self.config.get('training', {}).get('max_epochs', 200),
            'early_stopping_patience': self.config.get('training', {}).get('early_stopping_patience', 20)
        }
        
        config_file = self.output_dir / "final_training_config.json"
        with open(config_file, 'w') as f:
            json.dump(train_config, f, indent=2)
        logger.info(f"Saved training config to {config_file}")
        
        # Get sbatch script path
        sbatch_script = self.config.get('training', {}).get('final_training_sbatch')
        if sbatch_script is None:
            sbatch_script = Path(__file__).parent.parent.parent.parent / "scripts" / "training" / "run_final_training_gpu.sbatch"
        else:
            sbatch_script = Path(sbatch_script)
        
        if not sbatch_script.exists():
            raise FileNotFoundError(f"SLURM script not found: {sbatch_script}")
        
        import os
        
        # Submit job with config file path
        env = {"TRAIN_CONFIG": str(config_file)}
        cmd = ["sbatch", str(sbatch_script)]
        
        result = subprocess.run(cmd, env={**os.environ, **env}, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"SLURM submission failed: {result.stderr}")
        
        job_id = result.stdout.strip().split()[-1]
        logger.info(f"Submitted SLURM job: {job_id}")
        logger.info(f"Monitor with: squeue -j {job_id}")
        logger.info(f"Check logs in: {self.output_dir.parent / 'logs'}")
        logger.info(f"Model will be saved to: {self.output_dir / 'best_model.pth'}")
        
        return {"job_id": job_id, "status": "submitted", "config_file": str(config_file)}
    
    def train_final_model(
        self,
        hyperparams: Optional[Dict[str, Any]] = None,
        hyperparams_file: Optional[Path] = None
    ) -> Path:
        """
        Train final model on full training set with validation-based early stopping.
        
        IMPORTANT: This implements proper validation-based early stopping to prevent
        overfitting. The training data is split into sub-train (90%) and validation (10%)
        sets. The model is trained on sub-train and evaluated on validation after each
        epoch. Early stopping monitors validation loss, not training loss.
        
        Args:
            hyperparams: Dictionary of hyperparameters
            hyperparams_file: Path to JSON file with hyperparameters
            
        Returns:
            Path to saved model (best checkpoint based on validation loss)
        """
        from diana.training.trainer import MultiTaskTrainer
        from diana.models.multitask_mlp import MultiTaskMLP
        from diana.data.loader import MatrixLoader
        from sklearn.model_selection import train_test_split
        import torch
        import polars as pl
        import numpy as np
        
        logger.info("=" * 50)
        logger.info("Training final model with validation-based early stopping")
        logger.info("=" * 50)
        
        # Load hyperparameters
        if hyperparams_file:
            logger.info(f"Loading hyperparameters from {hyperparams_file}")
            with open(hyperparams_file) as f:
                hyperparams = json.load(f)
        elif hyperparams is None:
            # Try to load from CV results
            hyperparam_file = self.cv_dir / "best_hyperparameters.json"
            if hyperparam_file.exists():
                logger.info(f"Loading best hyperparameters from {hyperparam_file}")
                with open(hyperparam_file) as f:
                    hyperparams = json.load(f)
            else:
                # Aggregate CV results to generate best_hyperparameters.json
                logger.info("best_hyperparameters.json not found, aggregating CV results...")
                fold_results = []
                for fold_dir in self.cv_dir.glob("fold_*"):
                    result_files = list(fold_dir.glob("multitask_fold_*_results_*.json"))
                    if result_files:
                        with open(result_files[0]) as f:
                            fold_results.append(json.load(f))
                
                if not fold_results:
                    raise ValueError("No CV results found in cv_results directory. Run optimization first.")
                
                aggregated = self._aggregate_cv_results(fold_results)
                hyperparams = aggregated['best_hyperparameters']
        
        logger.info(f"Hyperparameters: {json.dumps(hyperparams, indent=2)}")
        
        # If using SLURM, submit job instead of training locally
        if self.use_slurm:
            return self._train_final_slurm(hyperparams)
        
        # Load training data
        logger.info(f"Loading training data from {self.features_path}")
        loader = MatrixLoader(self.features_path, self.metadata_path)
        
        # Load data based on split
        if self.split_path:
            logger.info(f"Using split file: {self.split_path}")
            split_df = pl.read_csv(self.split_path)
            train_df = split_df.filter(pl.col('split') == 'train')
            train_ids = train_df['sample_id'].to_list()
            X_full, y_full = loader.load_data(sample_ids=train_ids)
        else:
            logger.info("No split file provided, using all data")
            X_full, y_full = loader.load_data()
        
        logger.info(f"Full training data shape: {X_full.shape}")
        logger.info(f"Total samples: {len(X_full)}")
        
        # Get task names from config
        task_names = self.config.get('model', {}).get('task_names', [])
        if not task_names:
            raise ValueError("No task_names found in config")
        
        # Split into sub-train and validation sets for proper early stopping
        validation_split = self.config.get('training', {}).get('validation_split', 0.1)
        logger.info(f"Creating validation split: {validation_split * 100:.0f}% for validation")
        indices = np.arange(len(X_full))
        train_idx, val_idx = train_test_split(
            indices, 
            test_size=validation_split, 
            random_state=42,
            stratify=y_full[task_names[0]]  # Stratify on first task
        )
        
        X_train, X_val = X_full[train_idx], X_full[val_idx]
        y_train = {task: y_full[task][train_idx] for task in task_names}
        y_val = {task: y_full[task][val_idx] for task in task_names}
        
        logger.info(f"Sub-train samples: {len(X_train)}")
        logger.info(f"Validation samples: {len(X_val)}")
        
        # Get task info from metadata
        task_info = {}
        for task_name in task_names:
            n_classes = len(loader.metadata[task_name].unique())
            task_info[task_name] = n_classes
            logger.info(f"Task '{task_name}': {n_classes} classes")
        
        # Initialize model with hyperparameters
        model = MultiTaskMLP(
            input_dim=X_full.shape[1],
            task_info=task_info,
            hidden_dims=hyperparams.get('hidden_dims', [256, 128]),
            dropout_rate=hyperparams.get('dropout_rate', 0.2),
            use_batch_norm=hyperparams.get('use_batch_norm', False),
            activation=hyperparams.get('activation', 'relu')
        )
        
        # Initialize trainer
        trainer = MultiTaskTrainer(
            model=model,
            task_names=task_names,
            device=self.device,
            learning_rate=hyperparams.get('learning_rate', 1e-3),
            task_weights=self.config.get('training', {}).get('task_weights', {})
        )
        
        # Use encapsulated training loop with validation-based early stopping
        logger.info("Starting training with validation monitoring...")
        max_epochs = self.config.get('training', {}).get('max_epochs', 200)
        patience = self.config.get('training', {}).get('early_stopping_patience', 20)
        
        history = trainer.fit(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            max_epochs=max_epochs,
            batch_size=hyperparams.get('batch_size', 32),
            patience=patience,
            checkpoint_dir=self.output_dir,
            verbose=True
        )
        
        # The trainer.fit() method saves the best model automatically
        model_path = self.output_dir / "best_model.pth"
        if not model_path.exists():
            # Fallback: save current model state
            logger.warning("Best model not found, saving current state")
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'hyperparameters': hyperparams,
                'task_info': task_info,
                'training_history': history,
                'input_dim': X_full.shape[1]
            }
            torch.save(checkpoint, model_path)
        
        logger.info(f"Best model saved to {model_path}")
        
        # Save training history
        history_file = self.output_dir / "training_history.json"
        with open(history_file, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            serializable_history = {}
            for key, value in history.items():
                if isinstance(value, dict):
                    serializable_history[key] = {k: [float(v) for v in vals] for k, vals in value.items()}
                else:
                    serializable_history[key] = [float(v) for v in value]
            json.dump(serializable_history, f, indent=2)
        logger.info(f"Saved training history to {history_file}")
        
        # Log final performance
        if 'val_loss' in history and len(history['val_loss']) > 0:
            best_epoch = np.argmin(history['val_loss'])
            logger.info(f"")
            logger.info(f"Training complete!")
            logger.info(f"Best epoch: {best_epoch + 1}")
            logger.info(f"Best validation loss: {history['val_loss'][best_epoch]:.4f}")
            for task in task_names:
                if task in history.get('val_acc', {}):
                    logger.info(f"  {task} val accuracy: {history['val_acc'][task][best_epoch]:.4f}")
        
        return model_path


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
    multitask_parser.add_argument('--config', type=Path, required=True,
                                   help='Path to configuration YAML file (contains all data paths and training parameters)')
    multitask_parser.add_argument('--output', type=Path, required=True,
                                   help='Output directory for results')
    multitask_parser.add_argument('--mode', choices=['optimize', 'train', 'full'],
                                   default='full', help='Training mode (default: full)')
    multitask_parser.add_argument('--hyperparams', type=Path,
                                   help='Path to hyperparameters JSON (for train mode, optional if using CV results)')
    
    args = parser.parse_args()
    
    if not args.task:
        parser.print_help()
        return
    
    # Setup logging
    setup_logging(args.output / "diana_train.log")
    
    # Initialize pipeline
    pipeline = MultiTaskTrainingPipeline(
        output_dir=args.output,
        config_path=args.config
    )
    
    # Run requested mode
    if args.mode in ['optimize', 'full']:
        results = pipeline.optimize_hyperparameters()
        
        if pipeline.use_slurm:
            logger.info("="*50)
            logger.info("SLURM job submitted successfully")
            logger.info("="*50)
            logger.info("The pipeline will exit now. Jobs are running on SLURM.")
            logger.info("After jobs complete, continue with:")
            logger.info(f"  diana-train multitask --config {args.config} --output {args.output} --mode train")
            return
    
    if args.mode in ['train', 'full']:
        model_path = pipeline.train_final_model(
            hyperparams_file=args.hyperparams
        )
    
    logger.info("Pipeline complete!")


if __name__ == '__main__':
    main()
