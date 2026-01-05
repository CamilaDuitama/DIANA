# Quick Reference - Multi-Task Training

## Directory Organization

```
results/
├── multitask/                    # Multi-task classifier (4 tasks together)
│   ├── hyperopt/                # Production hyperparameter search
│   ├── hyperopt_test/           # Test runs on dummy data  
│   └── final/                   # Final model (after hyperopt)
└── singletask/                   # Single-task classifiers (future)
    ├── sample_type/
    ├── community_type/
    ├── sample_host/
    └── material/

logs/
├── multitask/                    # SLURM logs organized by experiment
│   ├── hyperopt/
│   ├── hyperopt_test/
│   └── final/
└── singletask/
```

## Common Commands

### Testing (Quick verification)
```bash
./scripts/training/submit_multitask.sh test
# Results → results/multitask/hyperopt_test/
# Logs → logs/multitask/hyperopt_test/
```

### Production (Full hyperparameter search)
```bash
./scripts/training/submit_multitask.sh prod
# Results → results/multitask/hyperopt/
# Logs → logs/multitask/hyperopt/
```

### Aggregate Results
```bash
python scripts/evaluation/collect_multitask_results.py \
    --results-dir results/multitask/hyperopt \
    --save-config
# Output → results/multitask/hyperopt/aggregated_results.json
```

### Monitor Jobs
```bash
squeue -u $USER                                    # Job status
tail -f logs/multitask/hyperopt/hyperopt_*.out    # Watch progress
ls -lh results/multitask/hyperopt/fold_*/         # Check outputs
```

## Script Dependencies

### 07_train_multitask_single_fold.py
**Purpose**: Hyperparameter optimization for one CV fold  
**Requires**:
- `diana.models.multitask_mlp.MultiTaskMLP`
- `diana.data.loader.MatrixLoader`
- torch, optuna, scikit-learn, polars

**Input**:
- K-mer matrix (.pa.mat)
- Metadata (.tsv)

**Output** (per fold):
- `best_multitask_model_fold_X_*.pth`
- `multitask_fold_X_results_*.json`
- `fold_X_training_log_*.txt`

### collect_multitask_results.py
**Purpose**: Aggregate metrics across all folds  
**Requires**: pandas, numpy  
**Input**: Fold results from hyperopt  
**Output**: 
- `aggregated_results.json` (summary)
- `best_config_for_final_training.json` (best hyperparams)

### run_multitask_gpu.sbatch
**Purpose**: SLURM GPU array job wrapper  
**Requires**: SLURM, mamba environment  
**Calls**: `07_train_multitask_single_fold.py`  
**Configurable via**: Environment variables (FEATURES, METADATA, OUTPUT_DIR, LOG_DIR, etc.)

### submit_multitask.sh
**Purpose**: Convenient launcher with presets  
**Modes**: test (dummy), prod (full), custom  
**Sets**: All environment variables for run_multitask_gpu.sbatch
