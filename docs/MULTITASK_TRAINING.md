# Multi-Task MLP Training Workflow

Complete workflow for training a multi-task deep learning classifier on the DIANA dataset.

## Classification Tasks

The model simultaneously predicts 4 biological characteristics:

1. **sample_type** (Binary): Ancient vs Modern metagenome
   - Ancient: 2,410 samples (92.4%)
   - Modern: 199 samples (7.6%)

2. **community_type** (6 classes): Biological community type
   - Oral, skeletal tissue, gut, plant tissue, soft tissue, environmental

3. **sample_host** (12 classes): Host organism
   - Homo sapiens, Ursus arctos, environmental samples, etc.

4. **material** (13 classes): Sample material
   - Dental calculus, tooth, bone, sediment, etc.

## Workflow Steps

### 1. Data Preparation

Ensure train/test splits and matrices are generated:

```bash
# Create stratified splits (85% train, 15% test)
mamba run -p ./env python scripts/data_prep/01_create_splits.py

# Extract DIANA samples and split matrices
mamba run -p ./env python scripts/data_prep/05_extract_and_split_matrices.py
```

This creates:
- `data/splits/train_metadata.tsv` (2,609 samples)
- `data/splits/test_metadata.tsv` (461 samples)
- `data/splits/train_matrix.pa.mat` (2,609 × 104,565)
- `data/splits/test_matrix.pa.mat` (461 × 104,565)

### 2. Hyperparameter Optimization

Run 5-fold cross-validation with Bayesian hyperparameter optimization:

```bash
# Submit to GPU queue
./scripts/training/launch_multitask.sh

# Monitor progress
squeue -u $USER
watch -n 30 'ls -lh results/multitask_gpu/fold_*/multitask_fold_*.json 2>/dev/null'
```

Each fold:
- Performs nested CV with 3 inner folds
- Runs 50 Optuna trials (Bayesian optimization)
- Optimizes: architecture, learning rate, dropout, task weights, etc.
- Trains final model with best hyperparameters
- Evaluates on held-out test fold

**Resources:**
- Time: ~2-4 hours per fold (depends on GPU and n_trials)
- GPU: 1 × A40/A100 (16-40GB VRAM)
- Memory: 64GB RAM
- CPUs: 8 cores

**Output structure:**
```
results/multitask_gpu/
├── fold_0/
│   ├── multitask_fold_0_results_20251219_203045.json
│   ├── best_multitask_model_fold_0_20251219_203045.pth
│   └── fold_0_training_log_20251219_203045.txt
├── fold_1/
│   └── ...
...
└── fold_4/
    └── ...
```

### 3. Collect and Analyze Results

After all folds complete:

```bash
# Summarize results across folds
mamba run -p ./env python scripts/evaluation/collect_multitask_results.py \
    --results-dir results/multitask_gpu \
    --save-config

# View summary
cat results/multitask_gpu/summary.json
```

Output includes:
- Per-task performance (mean ± std across folds)
- Best hyperparameters for final training
- Configuration file: `best_config_for_final_training.json`

### 4. Final Model Training (TODO)

Train final model on full training set with optimal hyperparameters:

```bash
# Coming soon: train_final_multitask.py
mamba run -p ./env python scripts/training/train_final_multitask.py \
    --config results/multitask_gpu/best_config_for_final_training.json \
    --features data/splits/train_matrix.pa.mat \
    --metadata data/splits/train_metadata.tsv \
    --output models/multitask/final_model.pth
```

### 5. Evaluation on Test Set (TODO)

```bash
# Coming soon: evaluate_multitask.py
mamba run -p ./env python scripts/evaluation/evaluate_multitask.py \
    --model models/multitask/final_model.pth \
    --features data/splits/test_matrix.pa.mat \
    --metadata data/splits/test_metadata.tsv \
    --output results/multitask/test_evaluation.json
```

## Model Architecture

**Multi-Task MLP with Shared Backbone:**

```
Input (104,565 k-mers)
    ↓
Shared Encoder
    - 2-4 hidden layers (64-512 units)
    - Batch normalization (optional)
    - Dropout (0.1-0.5)
    - Activation: ReLU/GELU/LeakyReLU
    ↓
Task-Specific Heads (4 heads)
    - sample_type head → 2 classes
    - community_type head → 6 classes  
    - sample_host head → 12 classes
    - material head → 13 classes
```

**Loss Function:**
- Weighted sum of cross-entropy losses
- Task weights optimized by Optuna
- Class weights for imbalanced classes (especially sample_type)

## Hyperparameters Optimized

**Architecture:**
- Number of hidden layers: 2-4
- Hidden layer dimensions: 64-512
- Activation function: ReLU, GELU, LeakyReLU
- Batch normalization: True/False
- Dropout rate: 0.1-0.5

**Training:**
- Learning rate: 1e-5 to 1e-2 (log scale)
- Weight decay: 1e-6 to 1e-3 (log scale)
- Batch size: 32, 64, 128, 256

**Multi-Task:**
- Task weight for sample_type: 0.5-2.0
- Task weight for community_type: 0.5-2.0
- Task weight for sample_host: 0.5-2.0
- Task weight for material: 0.5-2.0

## Key Features

✅ **Multi-task learning** - Shared representations across related tasks  
✅ **Class imbalance handling** - Weighted losses for rare classes  
✅ **Bayesian optimization** - Efficient hyperparameter search with Optuna  
✅ **Nested cross-validation** - Unbiased performance estimation  
✅ **GPU acceleration** - Fast training with PyTorch CUDA  
✅ **Early stopping** - Prevents overfitting  
✅ **Pruning** - Terminates unpromising trials early  

## Troubleshooting

**Job fails with CUDA out of memory:**
- Reduce batch size in Optuna search space
- Request more GPU memory (A100 40GB/80GB)
- Reduce max hidden layer size

**Jobs take too long:**
- Reduce `n_trials` (default: 50)
- Reduce `max_epochs` (default: 200)
- Use fewer inner CV folds

**Poor performance on rare classes:**
- Check class weights are applied
- Increase task weight for that target
- Consider oversampling minority classes

## Next Steps

1. ✅ Hyperparameter optimization (current)
2. ⏳ Final model training on full dataset
3. ⏳ Test set evaluation
4. ⏳ Single-task models for comparison (Task 4)
5. ⏳ Feature importance analysis
6. ⏳ Results visualization
