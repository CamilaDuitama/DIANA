# Migration Guide: From Scripts to Production Code

This guide shows how to update existing scripts to use the new production-ready features.

## Quick Start: Minimal Changes

### Before (Current)
```python
# scripts/training/07_train_multitask_single_fold.py
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Hardcoded parameters
MAX_EPOCHS = 200
BATCH_SIZE = 32
```

### After (With New Features)
```python
# scripts/training/07_train_multitask_single_fold.py
from diana.utils.config import setup_logging
from diana.config import ConfigManager
from diana.utils.checkpointing import CheckpointManager

# Setup logging
logger = setup_logging(
    log_file=args.output / "training.log",
    level="INFO",
    log_to_console=True
)

# Load configuration
if args.config:
    config = ConfigManager.from_yaml(args.config)
    max_epochs = config.get("training.max_epochs")
    batch_size = config.get("training.batch_size")
else:
    max_epochs = args.max_epochs
    batch_size = args.batch_size

# Initialize checkpointing
checkpoint_mgr = CheckpointManager(
    output_dir=args.output / f"fold_{args.fold_id}",
    save_best=True,
    save_frequency=10
)
```

---

## Feature-by-Feature Migration

### 1. Configuration Files

**Add to argparse:**
```python
parser.add_argument('--config', type=Path, help='Path to YAML configuration file')
```

**Load and merge with args:**
```python
def get_param(config, args, key, default=None):
    """Get parameter from config or args, with fallback to default."""
    if config:
        return config.get(key, getattr(args, key.replace('.', '_'), default))
    return getattr(args, key.replace('.', '_'), default)

# Usage
max_epochs = get_param(config, args, "training.max_epochs", 200)
```

### 2. Logging

**Replace all print() statements:**
```python
# Before
print(f"Starting training fold {fold_id}")
print(f"Epoch {epoch}/{max_epochs}, Loss: {loss:.4f}")

# After
logger.info(f"Starting training fold {fold_id}")
logger.info(f"Epoch {epoch}/{max_epochs}, Loss: {loss:.4f}")
logger.debug(f"Batch {batch_idx}: detailed metrics")
logger.warning(f"Validation loss increased for {n_epochs} epochs")
```

### 3. Checkpointing

**Replace manual model saving:**
```python
# Before
model_path = output_dir / f"best_model_fold_{fold_id}.pth"
torch.save(model.state_dict(), model_path)

# After
checkpoint_mgr.save_checkpoint(
    model=model,
    optimizer=optimizer,
    epoch=epoch,
    metrics={"val_loss": val_loss, "f1_macro": f1_macro},
    hyperparams=trial.params,
    is_best=(val_loss < best_val_loss)
)
```

**Add resume capability:**
```python
# At start of training
if args.resume_from:
    start_epoch = checkpoint_mgr.resume_from_checkpoint(model, optimizer, args.resume_from)
else:
    start_epoch = 0
```

### 4. Error Handling

**Wrap file operations:**
```python
# Before
features, metadata = load_matrix_data(matrix_path, metadata_path)

# After
try:
    logger.info(f"Loading data from {matrix_path}")
    features, metadata = load_matrix_data(matrix_path, metadata_path)
    logger.info(f"Loaded {features.shape[0]} samples with {features.shape[1]} features")
except FileNotFoundError as e:
    logger.error(f"Data file not found: {e}")
    sys.exit(1)
except ValueError as e:
    logger.error(f"Invalid data format: {e}")
    sys.exit(1)
except Exception as e:
    logger.error(f"Unexpected error loading data: {e}", exc_info=True)
    raise
```

---

## Complete Example: Updated Training Script

```python
#!/usr/bin/env python3
"""
Multi-Task MLP Training with Production Features
=================================================

Supports:
  - YAML configuration files
  - Structured logging
  - Model checkpointing
  - Resume from checkpoint
  - Comprehensive error handling
"""

import sys
import argparse
from pathlib import Path
import torch

from diana.config import ConfigManager
from diana.utils.config import setup_logging
from diana.utils.checkpointing import CheckpointManager
from diana.data.loader import MatrixLoader
from diana.models.multitask_mlp import MultiTaskMLP


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, help='YAML configuration file')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume-from', type=Path, help='Resume from checkpoint')
    
    # Optional overrides (if no config provided)
    parser.add_argument('--features', type=Path)
    parser.add_argument('--metadata', type=Path)
    parser.add_argument('--max-epochs', type=int, default=200)
    
    args = parser.parse_args()
    
    # Load configuration
    config = ConfigManager.from_yaml(args.config) if args.config else None
    
    # Setup logging
    logger = setup_logging(
        log_file=args.output / "training.log",
        level=config.get("logging.level", "INFO") if config else "INFO",
        log_to_console=True
    )
    
    logger.info("=" * 80)
    logger.info("DIANA Multi-Task Training")
    logger.info("=" * 80)
    
    # Get parameters
    features_path = config.get("data.train_matrix") if config else args.features
    metadata_path = config.get("data.train_metadata") if config else args.metadata
    max_epochs = config.get("training.max_epochs") if config else args.max_epochs
    
    # Validate inputs
    if not features_path or not metadata_path:
        logger.error("Must provide --features and --metadata, or --config")
        sys.exit(1)
    
    # Save configuration for reproducibility
    if config:
        config.save(args.output / "config_used.yaml")
        logger.info(f"Configuration saved to {args.output / 'config_used.yaml'}")
    
    # Initialize checkpointing
    checkpoint_mgr = CheckpointManager(
        output_dir=args.output,
        save_best=True,
        save_frequency=config.get("output.checkpoint_frequency", 10) if config else 10
    )
    
    # Load data
    try:
        logger.info(f"Loading data from {features_path}")
        loader = MatrixLoader(features_path)
        features, metadata = loader.load_with_metadata(
            metadata_path=metadata_path,
            align_to_matrix=True
        )
        logger.info(f"Loaded {features.shape[0]} samples × {features.shape[1]} features")
    except Exception as e:
        logger.error(f"Failed to load data: {e}", exc_info=True)
        sys.exit(1)
    
    # Initialize model
    model = MultiTaskMLP(
        input_dim=features.shape[1],
        hidden_dims=config.get("model.hidden_dims", [256, 128, 64]) if config else [256, 128, 64],
        num_classes_per_task={"sample_type": 2, "community_type": 6, "sample_host": 12, "material": 13}
    )
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.get("optimizer.learning_rate", 0.001) if config else 0.001
    )
    
    # Resume from checkpoint if requested
    start_epoch = 0
    if args.resume_from:
        try:
            start_epoch = checkpoint_mgr.resume_from_checkpoint(model, optimizer, args.resume_from)
            logger.info(f"Resumed from checkpoint, starting at epoch {start_epoch}")
        except Exception as e:
            logger.error(f"Failed to resume from checkpoint: {e}", exc_info=True)
            sys.exit(1)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, max_epochs):
        logger.info(f"Epoch {epoch + 1}/{max_epochs}")
        
        # Train (simplified for example)
        model.train()
        train_loss = 0.0  # Actual training code here
        
        # Validate
        model.eval()
        val_loss = 0.0  # Actual validation code here
        
        logger.info(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save checkpoint
        checkpoint_mgr.save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch + 1,
            metrics={"train_loss": train_loss, "val_loss": val_loss},
            is_best=(val_loss < best_val_loss)
        )
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
    
    # Save final model
    checkpoint_mgr.save_final_model(model, metrics={"best_val_loss": best_val_loss})
    logger.info("Training complete!")


if __name__ == '__main__':
    main()
```

---

## Testing the Migration

### 1. Test with dummy data
```bash
# Create test config
cat > configs/test.yaml <<EOF
data:
  train_matrix: data/test_data/splits/train_matrix_100feat.pa.mat
  train_metadata: data/test_data/splits/train_metadata.tsv
training:
  max_epochs: 5
  n_trials: 2
logging:
  level: DEBUG
output:
  checkpoint_frequency: 2
EOF

# Run with config
python scripts/training/07_train_multitask_single_fold.py \
    --config configs/test.yaml \
    --output results/test_migration
```

### 2. Check outputs
```bash
# Verify logs
cat results/test_migration/training.log

# Verify checkpoints
ls -lh results/test_migration/*.pth

# Verify config saved
cat results/test_migration/config_used.yaml
```

### 3. Test resume
```bash
# Interrupt training (Ctrl+C)
# Resume
python scripts/training/07_train_multitask_single_fold.py \
    --config configs/test.yaml \
    --output results/test_migration \
    --resume-from results/test_migration/best_model.pth
```

---

## Backwards Compatibility

The new features are **opt-in** and **backwards compatible**:

- ✅ Scripts work without `--config` (use command-line args)
- ✅ Checkpointing can be disabled
- ✅ Old model files still loadable
- ✅ Logging works with or without config

**Migration Strategy:**
1. Start with logging (replace print statements)
2. Add checkpointing (improves robustness)
3. Add config support (easier experimentation)
4. Add error handling (production readiness)

You can adopt features incrementally without breaking existing workflows!
