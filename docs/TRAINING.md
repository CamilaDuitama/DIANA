# Full Model Training Guide

This guide explains how to train the DIANA multi-task classifier from hyperparameter optimization to final production model.

## Overview

The training pipeline consists of 3 automatic steps:

1. **Hyperparameter Optimization** - 5-fold cross-validation with Optuna
2. **Result Aggregation** - Automatic aggregation of best hyperparameters
3. **Final Training** - Train production model on all data using best hyperparameters

## Prerequisites

- Configured environment: `./env` with all dependencies installed
- Training data: K-mer matrix and metadata
- Configuration file: `configs/full_training.yaml`
- SLURM cluster with GPU nodes

## Step 1: Hyperparameter Optimization

Run cross-validation to find optimal hyperparameters:

```bash
./env/bin/diana-train multitask \
    --config configs/full_training.yaml \
    --output results/full_training \
    --mode full
```

**What happens:**
- Submits 5 parallel SLURM jobs (array job) to GPU partition
- Each fold trains on 80% of data (2,456 samples), tests on 20% (614 samples)
- Optuna runs 50 trials per fold to optimize hyperparameters
- Duration: ~3-11 minutes per fold
- Resources per job: 8 CPUs, 64GB RAM, 1 GPU

**Outputs:**
- `results/full_training/cv_results/fold_0/` through `fold_4/`
  - `best_model.pth` - Best model for this fold
  - `multitask_fold_X_results_TIMESTAMP.json` - Contains `best_params` and `test_metrics`
- `logs/multitask/hyperopt/hyperopt_JOBID_X.{out,err}` - Training logs

**Monitor progress:**
```bash
# Check job status
squeue -u $USER

# Check specific job
sacct -j JOBID --format=JobID,State,ExitCode,Elapsed

# View live output
tail -f logs/multitask/hyperopt/hyperopt_JOBID_0.out

# Check resource efficiency after completion
reportseff JOBID
```

**Verify completion:**
```bash
# Check all folds completed
ls results/full_training/cv_results/fold_*/multitask_fold_*_results_*.json

# Should show 5 files (one per fold)
```

## Step 2 + 3: Aggregation + Final Training

After CV jobs complete, train the final model:

```bash
./env/bin/diana-train multitask \
    --config configs/full_training.yaml \
    --output results/full_training \
    --mode train
```

**What happens automatically:**

### 2a. Result Aggregation (if needed)
- Checks if `results/full_training/cv_results/best_hyperparameters.json` exists
- If not: reads all 5 fold result JSONs
- Aggregates hyperparameters across folds:
  - **Numeric params** (learning_rate, dropout, etc.): Takes mean across folds
  - **Categorical params** (activation, etc.): Takes mode (most common value)
- Saves aggregated results

**Outputs:**
- `results/full_training/cv_results/best_hyperparameters.json`
- `results/full_training/cv_results/aggregated_results.json`

### 2b. Final Training Submission
- Submits single SLURM GPU job for final training
- Uses aggregated best hyperparameters
- Trains on **all 3,070 samples**
- Duration: ~30-60 minutes (estimated)
- Resources: 4 CPUs, 64GB RAM, 1 GPU, 2h time limit

**Outputs:**
- `results/full_training/best_model.pth` - **Production model** (best checkpoint)
- `results/full_training/training_history.json` - Training/validation metrics per epoch
- `logs/final_training/diana_final_JOBID.{out,err}` - Training logs

**Monitor final training:**
```bash
# Check job status
squeue -u $USER

# View live training progress
tail -f logs/final_training/diana_final_JOBID.out

# After completion, check results
cat results/full_training/training_history.json
```

## Training Configuration

Edit `configs/full_training.yaml` to customize training:

```yaml
data:
  features_path: data/matrices/large_matrix_3070_with_frac/unitigs.frac.mat
  metadata_path: data/metadata/DIANA_metadata.tsv

model:
  task_names: [sample_type, community_type, sample_host, material]

training:
  # Hyperparameter optimization settings
  n_folds: 5                      # Number of CV folds
  n_trials: 50                    # Optuna trials per fold
  max_epochs: 200                 # Max epochs per trial
  n_inner_splits: 3               # Inner CV for Optuna
  
  # Final training settings
  validation_split: 0.1           # 10% held out for early stopping
  early_stopping_patience: 20     # Epochs without improvement before stopping
  
  # Task loss weights (optional, defaults to 1.0)
  task_weights:
    sample_type: 1.0
    community_type: 1.0
    sample_host: 1.0
    material: 1.0
  
  # Execution settings
  use_gpu: true                   # Use GPU for training
  use_slurm: true                 # Submit to SLURM (vs local execution)
```

## Advanced: Manual Aggregation

If you want to aggregate results without training:

```bash
./env/bin/python scripts/evaluation/08_collect_multitask_results.py \
    --results-dir results/full_training/cv_results \
    --output results/full_training/cv_results/aggregated_results.json
```

## Advanced: Provide Custom Hyperparameters

Skip optimization and use custom hyperparameters:

```bash
# Create custom hyperparameters JSON
cat > custom_hyperparams.json << EOF
{
  "n_layers": 3,
  "hidden_dim_0": 512,
  "hidden_dim_1": 256,
  "hidden_dim_2": 128,
  "dropout": 0.2,
  "learning_rate": 0.001,
  "batch_size": 64,
  "activation": "relu",
  "use_batch_norm": false
}
EOF

# Train with custom hyperparameters
./env/bin/diana-train multitask \
    --config configs/full_training.yaml \
    --output results/custom_training \
    --mode train \
    --hyperparams custom_hyperparams.json
```

## Training Modes Explained

The `--mode` parameter controls what gets executed:

| Mode | Optimization | Aggregation | Final Training | Use Case |
|------|-------------|-------------|----------------|----------|
| `optimize` | ✅ | ❌ | ❌ | Only run CV hyperparameter search |
| `train` | ❌ | ✅ (auto) | ✅ | Only train final model (assumes CV done) |
| `full` | ✅ | ❌ | ❌ | Run CV then **exit** (need manual `train` after) |

**Note:** When `use_slurm: true`, the `full` mode only submits CV jobs and exits. You must run `--mode train` separately after CV completes.

## Expected Cross-Validation Performance

Based on the 3,070 sample training set:

| Task | Mean F1 | Std | Min | Max |
|------|---------|-----|-----|-----|
| sample_type | 97.01% | 1.05% | 95.36% | 98.23% |
| community_type | 88.47% | 2.95% | 83.75% | 91.51% |
| sample_host | 92.77% | 2.38% | 88.03% | 94.65% |
| material | 84.84% | 7.60% | 70.79% | 91.67% |
| **Overall** | **90.77%** | **3.74%** | **84.48%** | **93.20%** |

*Note: Higher variance in `material` task indicates it's the most challenging classification.*

## Troubleshooting

### Jobs fail immediately
```bash
# Check error logs
tail -30 logs/multitask/hyperopt/hyperopt_JOBID_0.err
tail -30 logs/final_training/diana_final_JOBID.err

# Common issues:
# - Wrong partition/QoS for your cluster
# - Insufficient resources requested
# - Python environment not activated properly
```

### Import errors (ModuleNotFoundError)
```bash
# Verify environment has diana package
./env/bin/python -c "import diana; print(diana.__file__)"

# Check that sbatch uses 'mamba run -p ./env' not 'conda activate'
```

### Out of memory errors
```bash
# Check actual memory usage
reportseff JOBID

# Increase memory in sbatch script:
#SBATCH --mem=128G  # for hyperopt jobs
#SBATCH --mem=96G   # for final training
```

### Job times out
```bash
# Check elapsed time
sacct -j JOBID --format=JobID,Elapsed,Timelimit

# Increase time limit in configs or sbatch:
#SBATCH --time=24:00:00
```

## Output Files Reference

```
results/full_training/
├── cv_results/
│   ├── fold_0/
│   │   ├── best_model.pth                          # Best model for fold 0
│   │   ├── multitask_fold_0_results_*.json         # Fold 0 results & hyperparams
│   │   └── checkpoints/                            # Epoch checkpoints
│   ├── ... (fold_1 through fold_4)
│   ├── best_hyperparameters.json                   # Aggregated best hyperparams
│   ├── aggregated_results.json                     # Full aggregation with metrics
│   └── slurm_run_config.json                       # CV job configuration
├── best_model.pth                                  # **PRODUCTION MODEL**
├── training_history.json                           # Final training metrics
├── final_training_config.json                      # Final training job config
└── diana_train.log                                 # CLI execution log

logs/
├── multitask/hyperopt/
│   ├── hyperopt_JOBID_0.out                        # CV fold 0 stdout
│   ├── hyperopt_JOBID_0.err                        # CV fold 0 stderr
│   └── ... (folds 1-4)
└── final_training/
    ├── diana_final_JOBID.out                       # Final training stdout
    └── diana_final_JOBID.err                       # Final training stderr
```

## Next Steps

After training completes:

1. **Validate model** on held-out test set (Step 10 in project TODO)
2. **Compare performance** with single-task baselines (Task 4)
3. **Deploy model** for inference on new samples
4. **Document results** in paper/report

For inference/evaluation documentation, see [EVALUATION.md](EVALUATION.md) (to be created).
