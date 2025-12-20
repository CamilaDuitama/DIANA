"""
Unit tests for model checkpointing.

Tests CheckpointManager for saving/loading model checkpoints and tracking best models.
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
from diana.utils.checkpointing import CheckpointManager


class DummyModel(nn.Module):
    """Simple model for testing checkpointing."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)
    
    def forward(self, x):
        return self.fc(x)


class TestCheckpointing:
    """Test the CheckpointManager class."""
    
    def test_initialization(self, temp_dir):
        """Test creating a CheckpointManager."""
        manager = CheckpointManager(output_dir=temp_dir)
        assert manager.output_dir == temp_dir
        assert temp_dir.exists()
    
    def test_save_checkpoint(self, temp_dir):
        """Test saving a training checkpoint."""
        model = DummyModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        manager = CheckpointManager(output_dir=temp_dir)
        
        checkpoint_path = manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=5,
            metrics={'val_loss': 0.123}
        )
        
        assert checkpoint_path.exists()
        assert checkpoint_path.suffix == '.pth'
    
    def test_load_checkpoint(self, temp_dir):
        """Test loading a saved checkpoint."""
        model = DummyModel()
        optimizer = torch.optim.Adam(model.parameters())
        
        manager = CheckpointManager(output_dir=temp_dir)
        
        # Save
        checkpoint_path = manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=5,
            metrics={'val_loss': 0.123}
        )
        
        # Load
        checkpoint = manager.load_checkpoint(str(checkpoint_path))
        
        assert checkpoint['epoch'] == 5
        assert checkpoint['metrics']['val_loss'] == 0.123
        assert 'model_state_dict' in checkpoint
        assert 'optimizer_state_dict' in checkpoint
    
    def test_best_model_tracking(self, temp_dir):
        """Test saving and loading best model."""
        model = DummyModel()
        optimizer = torch.optim.Adam(model.parameters())
        
        manager = CheckpointManager(output_dir=temp_dir, save_best=True)
        
        # Save checkpoint marked as best
        manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=5,
            metrics={'val_loss': 0.25},
            is_best=True
        )
        
        # Verify best_model.pth exists
        best_path = temp_dir / 'best_model.pth'
        assert best_path.exists()
        
        # Load best model
        best_checkpoint = manager.load_best_model()
        
        assert best_checkpoint is not None
        assert best_checkpoint['metrics']['val_loss'] == 0.25
        assert best_checkpoint['epoch'] == 5
    
    def test_weights_preserved(self, temp_dir):
        """Verify model weights are preserved exactly through save/load."""
        model = DummyModel()
        initial_weight = model.fc.weight.data.clone()
        
        manager = CheckpointManager(output_dir=temp_dir)
        optimizer = torch.optim.Adam(model.parameters())
        
        # Save
        checkpoint_path = manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=1,
            metrics={'val_loss': 0.5}
        )
        
        # Load into new model
        new_model = DummyModel()
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        new_model.load_state_dict(checkpoint['model_state_dict'])
        
        # Verify weights match
        assert torch.allclose(new_model.fc.weight.data, initial_weight)
