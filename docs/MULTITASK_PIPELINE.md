# Multi-Task Classification Pipeline

Complete pipeline for training multi-task MLP classifiers on ancient DNA samples.

## Overview

**Objective**: Classify ancient DNA samples across 4 tasks simultaneously:
1. `sample_type` (2 classes): ancient_metagenome vs modern_metagenome
2. `community_type` (6 classes): oral, skeletal tissue, gut, plant tissue, soft tissue, env sample
3. `sample_host` (12 classes): Homo sapiens, Ursus arctos, environmental, etc.
4. `material` (13 classes): dental calculus, tooth, bone, sediment, etc.

**Approach**: Shared neural network encoder with task-specific heads (multi-task learning)

## Pipeline Steps

### Step 1: Data Preparation

#### 1.1 Create Train/Test Splits
```bash
python scripts/data_prep/01_create_splits.py \
    --metadata data/metadata/DIANA_metadata.tsv \
    --output data/splits \
    --test-size 0.15 \
    --stratify-by sample_type \
    --random-state 42
```

**Dependencies**:
- Input: `data/metadata/DIANA_metadata.tsv`
- Requires: pandas, scikit-learn
- Module: `diana.data.splitter.StratifiedSplitter`

**Output**:
- `data/splits/train_ids.txt` (2,609 sample IDs)
- `data/splits/test_ids.txt` (461 sample IDs)
- `data/splits/train_metadata.tsv` (metadata for training samples)
- `data/splits/test_metadata.tsv` (metadata for test samples)

#### 1.2 Extract K-mer Matrices
```bash
python scripts/data_prep/05_extract_and_split_matrices.py \
    --metadata data/splits/train_metadata.tsv \
    --test-metadata data/splits/test_metadata.tsv \
    --fof data/diana_samples.fof \
    --matrix-dir data/matrices/large_matrix_3070_with_frac/kmer_matrix \
    --output-dir data/splits \
    --kmer-size 31
```

**Dependencies**:
- Input: Split metadata files, k-mer matrix directory
- Requires: polars, numpy
- Module: `diana.data.loader.MatrixLoader`

**Output**:
- `data/splits/train_matrix.pa.mat` (2,609 × 104,565 features, ~4GB)
- `data/splits/test_matrix.pa.mat` (461 × 104,565 features)

---

### Step 2: Hyperparameter Optimization

#### 2.1 Quick Test (Recommended First)
```bash
./scripts/training/submit_multitask.sh test
```

**Configuration**:
- Data: 100 samples from dummy dataset
- Folds: 2
- Trials: 5 Optuna trials per fold
- Epochs: 20 max per trial
- Runtime: ~3-5 minutes

**Dependencies**:
- Scripts: `scripts/training/07_train_multitask_single_fold.py`
- Modules: `diana.models.multitask_mlp.MultiTaskMLP`, `diana.data.loader.MatrixLoader`
- Packages: torch, optuna, scikit-learn, polars

**Output**:
- Results: `results/multitask/hyperopt_test/fold_0/`, `fold_1/`
- Logs: `logs/multitask/hyperopt_test/hyperopt_*.out/err`

#### 2.2 Production Run (Full Dataset)
```bash
./scripts/training/submit_multitask.sh prod
```

**Configuration**:
- Data: 2,609 training samples, 104,565 features
- Folds: 5-fold cross-validation
- Trials: 50 Optuna trials per fold (Bayesian optimization)
- Epochs: 200 max per trial (with early stopping)
- Inner CV: 3-fold nested cross-validation
- Runtime: ~2-4 hours per fold on GPU

**Search Space**:
- Hidden layers: 2-4 layers, 64-512 neurons each
- Activation: ReLU, LeakyReLU, GELU
- Dropout: 0.0-0.5
- Batch normalization: True/False
- Learning rate: 1e-5 to 1e-2
- Batch size: 32, 64, 128, 256
- Task weights: Optimized per task

**Output**:
```
results/multitask/hyperopt/
├── fold_0/
│   ├── best_multitask_model_fold_0_*.pth      (52MB - trained model)
│   ├── multitask_fold_0_results_*.json        (metrics + hyperparameters)
│   └── fold_0_training_log_*.txt              (detailed training log)
├── fold_1/
├── fold_2/
├── fold_3/
└── fold_4/

logs/multitask/hyperopt/
├── hyperopt_JOBID_0.out/err
├── hyperopt_JOBID_1.out/err
├── hyperopt_JOBID_2.out/err
├── hyperopt_JOBID_3.out/err
└── hyperopt_JOBID_4.out/err
```

#### 2.3 Monitor Progress
```bash
# Check job status
squeue -u $USER

# Check latest results
tail -f logs/multitask/hyperopt/hyperopt_*.out

# Check fold completion
ls -lh results/multitask/hyperopt/fold_*/
```

---

### Step 3: Aggregate Results

```bash
python scripts/evaluation/collect_multitask_results.py \
    --results-dir results/multitask/hyperopt \
    --save-config
```

**Dependencies**:
- Input: Fold results from Step 2.2
- Requires: pandas, numpy

**Output**:
- `results/multitask/hyperopt/aggregated_results.json` - Full summary
- `results/multitask/hyperopt/best_config_for_final_training.json` - Best hyperparameters

**Displays**:
- Mean ± std metrics across all folds for each task
- Best hyperparameters (most common or best performing)
- Per-fold breakdown

---

### Step 4: Train Final Model (Future)

Train final model on all training data using best hyperparameters:

```bash
python scripts/training/08_train_final_multitask.py \
    --config results/multitask/hyperopt/best_config_for_final_training.json \
    --features data/splits/train_matrix.pa.mat \
    --metadata data/splits/train_metadata.tsv \
    --output results/multitask/final
```

**Output**:
- `results/multitask/final/final_multitask_model.pth`
- `results/multitask/final/final_results.json`

---

### Step 5: Evaluate on Test Set (Future)

```bash
python scripts/evaluation/evaluate_multitask_test.py \
    --model results/multitask/final/final_multitask_model.pth \
    --features data/splits/test_matrix.pa.mat \
    --metadata data/splits/test_metadata.tsv \
    --output results/multitask/final
```

**Output**:
- `results/multitask/final/test_predictions.csv`
- `results/multitask/final/test_metrics.json`
- Confusion matrices, classification reports

---

## Directory Structure

See [RESULTS_STRUCTURE.md](../RESULTS_STRUCTURE.md) for complete organization.

```
results/multitask/
├── hyperopt/          # Cross-validation results
├── hyperopt_test/     # Test run results
└── final/             # Final model on full training data

logs/multitask/
├── hyperopt/          # CV SLURM logs
├── hyperopt_test/     # Test SLURM logs
└── final/             # Final training logs
```

---

## Comparison with Single-Task Models (Future Task 4)

After multi-task training, compare with 4 separate single-task models:

```bash
# Train separate models
for task in sample_type community_type sample_host material; do
    ./scripts/training/submit_singletask.sh $task prod
done

# Compare performance
python scripts/evaluation/compare_multitask_vs_singletask.py \
    --multitask-results results/multitask/hyperopt \
    --singletask-results results/singletask \
    --output results/comparison
```

---

## Dependencies Summary

### Python Packages
- **Data**: polars (≥0.20), pandas, numpy
- **ML**: torch (≥2.0), scikit-learn
- **Optimization**: optuna (≥4.0)
- **Utils**: scipy, logging, json

### Internal Modules
- `diana.data.loader.MatrixLoader` - Fast data loading
- `diana.data.splitter.StratifiedSplitter` - Data splitting
- `diana.models.multitask_mlp.MultiTaskMLP` - Model architecture
- `diana.models.multitask_mlp.MultiTaskLoss` - Multi-task loss function

### External Tools
- **muset**: K-mer matrix generation (external/muset/)
- **SLURM**: Job scheduling

### Input Files
- `data/metadata/DIANA_metadata.tsv` - Sample metadata
- `data/diana_samples.fof` - File of files (sample paths)
- `data/matrices/large_matrix_3070_with_frac/kmer_matrix/` - K-mer matrices

---

## Troubleshooting

### Out of Memory
- Reduce batch size in hyperparameter search
- Use smaller `--n_trials` or `--max_epochs`
- Request more memory in SLURM: `#SBATCH --mem=128G`

### Slow Loading
- Ensure using polars (not pandas) in MatrixLoader
- Check matrix file size: `ls -lh data/splits/train_matrix.pa.mat`
- Expected: ~4GB for 2,609 × 104,565 matrix

### Poor Performance
- Check class imbalance: Some tasks have very few samples per class
- Review Optuna search space: May need wider ranges
- Increase `--n_trials` for better hyperparameter search

### SLURM Job Failed
- Check error logs: `tail -100 logs/multitask/hyperopt/hyperopt_*.err`
- Verify GPU availability: Job needs `--gres=gpu:1`
- Check environment: `mamba run -p ./env python --version`

---

## Next Steps

1. ✅ Complete hyperparameter optimization (Step 2.2)
2. ✅ Aggregate results (Step 3)
3. ⏳ Create final training script (Step 4) - **NEXT**
4. ⏳ Create test evaluation script (Step 5)
5. ⏳ Implement single-task baselines (Task 4)
6. ⏳ Compare multi-task vs single-task performance
