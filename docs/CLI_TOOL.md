# DIANA CLI Tool: `diana-train`

## Overview

The `diana-train` CLI provides a high-level interface for the complete multi-task training workflow, encapsulating hyperparameter optimization, final model training, and evaluation in a single command.

## Installation

```bash
# Install in development mode
cd /pasteur/appa/scratch/cduitama/EDID/decOM-classify
pip install -e .

# Verify installation
diana-train --help
```

## Design Philosophy

### Current Approach (Scripts)
```bash
# Step 1: Submit SLURM jobs for hyperparameter optimization
FEATURES="data/splits/train_matrix.pa.mat" \
METADATA="data/splits/train_metadata.tsv" \
sbatch --array=0-4 scripts/training/run_multitask_gpu.sbatch

# Step 2: Wait for jobs to complete, then collect results
python scripts/evaluation/08_collect_multitask_results.py \
    --results-dir results/experiments/multitask/run_001/cv_results

# Step 3: Train final model (manual script creation needed)
# Step 4: Evaluate on test set (manual script creation needed)
```

### New Approach (CLI)
```bash
# Complete workflow in one command
diana-train multitask \
    --features data/splits/train_matrix.pa.mat \
    --metadata data/splits/train_metadata.tsv \
    --test-features data/splits/test_matrix.pa.mat \
    --test-metadata data/splits/test_metadata.tsv \
    --output results/experiments/multitask/run_001 \
    --mode full \
    --use-slurm
```

## Usage Examples

### 1. Hyperparameter Optimization Only

**Local (CPU/GPU, sequential folds):**
```bash
diana-train multitask \
    --features data/splits/train_matrix.pa.mat \
    --metadata data/splits/train_metadata.tsv \
    --output results/experiments/multitask/run_001 \
    --mode optimize \
    --n-folds 5 \
    --n-trials 50 \
    --max-epochs 200
```

**SLURM (parallel GPU folds):**
```bash
diana-train multitask \
    --features data/splits/train_matrix.pa.mat \
    --metadata data/splits/train_metadata.tsv \
    --output results/experiments/multitask/run_001 \
    --mode optimize \
    --use-slurm \
    --n-folds 5 \
    --n-trials 50 \
    --max-epochs 200
```

### 2. Train Final Model with Best Hyperparameters

```bash
diana-train multitask \
    --features data/splits/train_matrix.pa.mat \
    --metadata data/splits/train_metadata.tsv \
    --output results/experiments/multitask/final_model \
    --mode train \
    --hyperparams results/experiments/multitask/run_001/best_hyperparameters.json
```

### 3. Evaluate Trained Model

```bash
diana-train multitask \
    --features data/splits/test_matrix.pa.mat \
    --metadata data/splits/test_metadata.tsv \
    --output results/experiments/multitask/final_model \
    --mode evaluate \
    --model results/experiments/multitask/final_model/best_model.pth
```

### 4. Full Workflow (Optimize → Train → Evaluate)

```bash
diana-train multitask \
    --features data/splits/train_matrix.pa.mat \
    --metadata data/splits/train_metadata.tsv \
    --test-features data/splits/test_matrix.pa.mat \
    --test-metadata data/splits/test_metadata.tsv \
    --output results/experiments/multitask/production \
    --mode full \
    --use-slurm
```

## Benefits

### For Users
- **Single command**: No need to remember multiple scripts and their order
- **Automatic path management**: Handles output directory structure
- **Better error handling**: Validates inputs before starting long jobs
- **Progress tracking**: Clear logging of pipeline stages
- **Flexibility**: Can run complete workflow or individual steps

### For Developers
- **Maintainability**: Business logic centralized in `diana.cli.train`
- **Testability**: Pipeline logic can be unit tested
- **Extensibility**: Easy to add new modes or tasks
- **Consistency**: All training uses same interface

## Relationship to Scripts

The CLI **uses** the existing scripts internally:

```
diana-train (CLI wrapper)
    ├── calls: scripts/training/run_multitask_gpu.sbatch (for SLURM)
    ├── calls: scripts/training/07_train_multitask_single_fold.py (for local)
    ├── calls: scripts/evaluation/08_collect_multitask_results.py (for aggregation)
    └── uses: diana.models, diana.data, diana.training (for final model)
```

**Scripts remain useful for:**
- Direct SLURM submission with custom parameters
- Debugging individual pipeline stages
- Integration with other workflows
- Advanced users who need fine-grained control

## Implementation Status

### ✅ Completed
- CLI structure and argument parsing
- Pipeline class architecture
- SLURM submission wrapper
- Documentation

### 🚧 To Implement
- `_optimize_local()`: Sequential local optimization
- `_aggregate_cv_results()`: Call collect results script
- `train_final_model()`: Final model training logic
- `evaluate_model()`: Test set evaluation logic

### 📋 Future Enhancements
- Single-task training: `diana-train singletask --task sample_type`
- Inference mode: `diana-train predict --model X --features Y`
- Model comparison: `diana-train compare --models A B C`
- Interactive mode: Guide user through parameter selection

## Recommendation

**For now**: Use scripts directly (fully implemented and tested)

**Next steps**: 
1. Implement missing methods in `MultiTaskTrainingPipeline`
2. Test CLI with dummy data
3. Once validated, recommend CLI as primary interface
4. Keep scripts as alternative for advanced users

**Final state**: Users choose based on needs:
- `diana-train`: Recommended for most users (simple, automated)
- Scripts: Available for custom workflows, debugging, SLURM experts
