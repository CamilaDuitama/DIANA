"""
End-to-end smoke tests for DIANA training pipeline.

These tests verify the complete pipeline runs without crashing on minimal test data.
They do NOT test accuracy - only that code executes successfully.
"""

import pytest
import subprocess
import sys
import json
import torch


class TestEndToEnd:
    """End-to-end integration tests - most critical for production readiness."""
    
    def test_full_training_pipeline(self, temp_dir, dummy_matrix_path, dummy_metadata_path):
        """
        CRITICAL TEST: Full training pipeline with Optuna hyperparameter search.
        
        Validates:
        - Data loading from real file formats
        - Optuna optimization (2 trials)
        - Multi-task model training (2 epochs)
        - Checkpoint saving
        - Best model tracking
        - Results JSON creation
        
        This is the most important test - if this passes, the pipeline works.
        """
        output_dir = temp_dir / "training_output"
        output_dir.mkdir(exist_ok=True)
        
        cmd = [
            sys.executable,
            "scripts/training/07_train_multitask_single_fold.py",
            "--fold_id", "0",
            "--total_folds", "2",
            "--features", str(dummy_matrix_path),
            "--metadata", str(dummy_metadata_path),
            "--output", str(output_dir),
            "--n_trials", "2",
            "--max_epochs", "2",
            "--n_inner_splits", "2",
            "--checkpoint_freq", "1",
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        assert result.returncode == 0, (
            f"Training failed with code {result.returncode}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
        
        # Verify outputs
        fold_dir = output_dir / "fold_0"
        assert fold_dir.exists(), "Fold directory not created"
        
        # Check results file
        results_files = list(fold_dir.glob("multitask_fold_0_results_*.json"))
        assert len(results_files) > 0, "Results JSON not created"
        
        with open(results_files[0]) as f:
            results = json.load(f)
        
        # Verify structure
        assert "test_metrics" in results
        assert "best_params" in results
        
        # Verify all tasks
        for task in ["sample_type", "community_type", "sample_host", "material"]:
            assert task in results["test_metrics"]
            assert "accuracy" in results["test_metrics"][task]
        
        # Verify model files exist
        assert (fold_dir / "best_model.pth").exists()
        
        # Verify checkpoint can be loaded
        checkpoint = torch.load(fold_dir / "best_model.pth", map_location="cpu", weights_only=False)
        assert "model_state_dict" in checkpoint


class TestImports:
    """Test that all modules can be imported without errors."""
    
    def test_all_modules_import(self):
        """Verify critical DIANA modules can be imported."""
        from diana.data.loader import MatrixLoader
        from diana.data.dataset import DianaDataset
        from diana.models.multitask_mlp import MultiTaskMLP
        from diana.config.manager import ConfigManager
        from diana.utils.config import setup_logging
        from diana.utils.checkpointing import CheckpointManager
        assert True  # If we got here, imports worked
    
    def test_pytorch_available(self):
        """Verify PyTorch is installed and working."""
        import torch
        x = torch.tensor([1.0, 2.0, 3.0])
        assert x.shape == (3,)
