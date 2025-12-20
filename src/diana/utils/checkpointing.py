"""
Model Checkpointing Utilities
==============================

Handles saving and loading model checkpoints during training.

Features:
  - Save best model (based on validation loss)
  - Save periodic checkpoints
  - Save final model
  - Resume training from checkpoint
  - Load models for inference

DEPENDENCIES:
-------------
Python packages:
  - torch (PyTorch)
  - pathlib
  - logging

Usage:
  # During training
  checkpoint_manager = CheckpointManager(output_dir, save_best=True, save_frequency=10)
  
  # Save checkpoint
  checkpoint_manager.save_checkpoint(
      model=model,
      optimizer=optimizer,
      epoch=epoch,
      metrics={"val_loss": 0.123},
      is_best=True
  )
  
  # Resume training
  checkpoint = checkpoint_manager.load_checkpoint("best_model.pth")
  model.load_state_dict(checkpoint['model_state_dict'])
"""

import torch
import logging
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manages model checkpointing during training.
    
    Responsibilities:
      - Save best model (lowest validation loss)
      - Save periodic checkpoints (every N epochs)
      - Save final model after training
      - Track and load checkpoints
      - Clean up old checkpoints (optional)
    
    Example:
        manager = CheckpointManager("results/experiment1")
        
        for epoch in range(n_epochs):
            # Training loop...
            val_loss = validate(model)
            
            # Save checkpoint
            manager.save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics={"val_loss": val_loss},
                is_best=(val_loss < best_loss)
            )
    """
    
    def __init__(
        self,
        output_dir: Path,
        save_best: bool = True,
        save_frequency: int = 10,
        save_final: bool = True,
        keep_last_n: Optional[int] = None
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            output_dir: Directory to save checkpoints
            save_best: Whether to save best model
            save_frequency: Save checkpoint every N epochs (0 to disable)
            save_final: Whether to save final model
            keep_last_n: Keep only last N checkpoints (None = keep all)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.save_best = save_best
        self.save_frequency = save_frequency
        self.save_final = save_final
        self.keep_last_n = keep_last_n
        
        self.best_metric = float('inf')
        self.checkpoints = []
        
        logger.info(f"CheckpointManager initialized: {self.output_dir}")
    
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        hyperparams: Optional[Dict[str, Any]] = None,
        is_best: bool = False,
        checkpoint_name: Optional[str] = None
    ) -> Path:
        """
        Save model checkpoint.
        
        Args:
            model: PyTorch model
            optimizer: PyTorch optimizer (optional)
            epoch: Current epoch number
            metrics: Dictionary of metrics (e.g., {"val_loss": 0.123})
            hyperparams: Hyperparameters used for this model
            is_best: Whether this is the best model so far
            checkpoint_name: Custom checkpoint name (auto-generated if None)
        
        Returns:
            Path to saved checkpoint
        """
        if metrics is None:
            metrics = {}
        
        # Prepare checkpoint data
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'metrics': metrics,
            'hyperparams': hyperparams,
            'timestamp': datetime.now().isoformat(),
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        # Generate checkpoint name
        if checkpoint_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_name = f"checkpoint_epoch_{epoch}_{timestamp}.pth"
        
        checkpoint_path = self.output_dir / checkpoint_name
        
        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        # Track checkpoint
        self.checkpoints.append({
            'path': checkpoint_path,
            'epoch': epoch,
            'metrics': metrics
        })
        
        # Save best model
        if is_best and self.save_best:
            val_loss = metrics.get('val_loss', float('inf'))
            if val_loss < self.best_metric:
                self.best_metric = val_loss
                best_path = self.output_dir / "best_model.pth"
                shutil.copy2(checkpoint_path, best_path)
                logger.info(f"New best model saved: {best_path} (val_loss: {val_loss:.4f})")
        
        # Clean up old checkpoints
        if self.keep_last_n is not None and len(self.checkpoints) > self.keep_last_n:
            old_checkpoint = self.checkpoints.pop(0)
            if old_checkpoint['path'].exists():
                old_checkpoint['path'].unlink()
                logger.debug(f"Removed old checkpoint: {old_checkpoint['path']}")
        
        return checkpoint_path
    
    def save_final_model(
        self,
        model: torch.nn.Module,
        metrics: Optional[Dict[str, float]] = None,
        hyperparams: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Save final model after training completion.
        
        Args:
            model: Trained PyTorch model
            metrics: Final metrics
            hyperparams: Hyperparameters used
        
        Returns:
            Path to saved model
        """
        if not self.save_final:
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_path = self.output_dir / f"final_model_{timestamp}.pth"
        
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'metrics': metrics or {},
            'hyperparams': hyperparams or {},
            'timestamp': datetime.now().isoformat(),
            'final': True
        }
        
        torch.save(checkpoint, final_path)
        logger.info(f"Final model saved: {final_path}")
        
        return final_path
    
    def load_checkpoint(self, checkpoint_path: Path) -> Dict[str, Any]:
        """
        Load checkpoint from file.
        
        Args:
            checkpoint_path: Path to checkpoint file
        
        Returns:
            Checkpoint dictionary
        
        Raises:
            FileNotFoundError: If checkpoint doesn't exist
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        return checkpoint
    
    def load_best_model(self) -> Optional[Dict[str, Any]]:
        """
        Load best model checkpoint.
        
        Returns:
            Best model checkpoint or None if not found
        """
        best_path = self.output_dir / "best_model.pth"
        
        if not best_path.exists():
            logger.warning(f"Best model not found: {best_path}")
            return None
        
        return self.load_checkpoint(best_path)
    
    def resume_from_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        checkpoint_path: Optional[Path] = None
    ) -> int:
        """
        Resume training from checkpoint.
        
        Args:
            model: Model to load weights into
            optimizer: Optimizer to load state into (optional)
            checkpoint_path: Path to checkpoint (uses best if None)
        
        Returns:
            Epoch number to resume from
        """
        if checkpoint_path is None:
            checkpoint_path = self.output_dir / "best_model.pth"
        
        checkpoint = self.load_checkpoint(checkpoint_path)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("Model state loaded")
        
        # Load optimizer state
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info("Optimizer state loaded")
        
        epoch = checkpoint.get('epoch', 0)
        logger.info(f"Resuming from epoch {epoch}")
        
        return epoch
    
    def list_checkpoints(self) -> list:
        """Return list of all saved checkpoints."""
        return sorted(self.output_dir.glob("checkpoint_*.pth"))
    
    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get path to most recent checkpoint."""
        checkpoints = self.list_checkpoints()
        return checkpoints[-1] if checkpoints else None
