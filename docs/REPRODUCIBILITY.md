# DIANA: Reproducibility Guide

**Multi-task classification of ancient DNA samples using unitig k-mer features**

---

## Table of Contents

1. [Environment Setup](#environment-setup)
2. [Data Preparation](#data-preparation)
3. [Train/Test Split](#traintest-split)
4. [Model Training](#model-training)
5. [Model Evaluation](#model-evaluation)
6. [Feature Analysis](#feature-analysis)
7. [Validation](#validation)
8. [Output Structure](#output-structure)

---

## Environment Setup

```bash
# Create mamba environment from specification
mamba env create -f environment.yml -p ./env

# Activate environment
mamba activate ./env

# Verify installation
python -c "import torch, polars, plotly; print('✓ Environment ready')"
```

**Requirements:** Python 3.10, PyTorch, Polars, Plotly, scikit-learn, Optuna, ETE3, Biopython, seqkit

---

## Data Preparation

### 1. Build k-mer Matrix with muset

```bash
# Build muset tool (one-time setup)
bash scripts/create_umat/01_build_muset.sh

# Generate unitig matrix from FASTQ files
# Input: data/diana_samples.fof (list of sample FASTQ paths)
# Output: data/matrices/large_matrix_3070_with_frac/
sbatch scripts/create_umat/02_regenerate_matrix_with_frac.sbatch
```

**Output:** `unitigs.frac.mat` (3070 samples × 107,480 features, 1.6GB)

### 2. Prepare Metadata

Metadata file: `data/metadata/DIANA_metadata.tsv`

Required columns:
- `Run_accession`: Sample identifier
- `sample_type`: ancient_metagenome | modern_metagenome
- `community_type`: oral | gut | skeletal tissue | plant tissue | soft tissue | env sample
- `sample_host`: Homo sapiens | Ursus arctos | environmental | etc. (12 classes)
- `material`: dental calculus | tooth | bone | sediment | etc. (13 classes)

---

## Train/Test Split

```bash
# Create stratified 85/15 train/test split
mamba run -p ./env python scripts/data_prep/01_create_splits.py \
  --metadata data/metadata/DIANA_metadata.tsv \
  --output data/splits \
  --train-size 0.85 \
  --test-size 0.15 \
  --random-state 42
```

**Output:**
- `data/splits/train_ids.txt` (2609 samples)
- `data/splits/test_ids.txt` (461 samples)
- `data/splits/train_metadata.tsv`
- `data/splits/test_metadata.tsv`

**Critical:** Test set is held out for final evaluation only. Never used during training or hyperparameter optimization.

---

## Model Training

**Configuration:** Edit `configs/train_config.yaml` with training parameters and hyperparameters:

```yaml
# Data paths
data:
  matrix: "data/matrices/large_matrix_3070_with_frac/unitigs.frac.mat"
  metadata: "data/splits/train_metadata.tsv"  # ← Uses train set only
  
# Training settings
training:
  n_folds: 5              # Outer CV folds for hyperparameter optimization
  n_trials: 50            # Optuna trials per fold
  max_epochs: 200
  n_inner_splits: 3       # Inner CV splits
  validation_split: 0.1   # For final model early stopping
  early_stopping_patience: 20
  use_gpu: true

# Model hyperparameters (from previous optimization or defaults)
hyperparameters:
  model_params:
    hidden_dims: [307, 474, 272]
    dropout: 0.1916894511329029
    activation: "relu"
    use_batch_norm: false
  trainer_params:
    learning_rate: 0.001723369322527049
    weight_decay: 3.75062679509968e-05
    task_weights:
      sample_type: 0.9307526054899867
      community_type: 1.6222031999590159
      sample_host: 1.1909289698749999
      material: 1.810458861595924
  batch_size: 96

# Classification tasks
tasks:
  - sample_type
  - community_type
  - sample_host
  - material
```

### Step 1: Hyperparameter Optimization

```bash
# Submit 5-fold CV optimization jobs to SLURM
mamba run -p ./env diana-train multitask \
  --config configs/train_config.yaml \
  --output results/training \
  --mode optimize
```

This submits 5 SLURM array jobs (one per fold). Monitor with `squeue -u $USER`.

**Wait for all jobs to complete** before proceeding to Step 2.

### Step 2: Aggregate Results

```bash
# After all fold jobs complete, aggregate the results
mamba run -p ./env diana-train multitask \
  --config configs/train_config.yaml \
  --output results/training \
  --mode optimize
```

### Step 3: Train Final Model

```bash
# Train on full training set with aggregated best hyperparameters
mamba run -p ./env diana-train multitask \
  --config configs/train_config.yaml \
  --output results/training \
  --mode train \
  --hyperparams results/training/cv_results/best_hyperparameters.json
```

**Output:**
- `results/training/best_model.pth`
- `results/training/training_history.json`
- `results/training/label_encoders.json`

**Trained on:** 2609 train samples (90% sub-train, 10% validation for early stopping)

---

## Model Evaluation

### Test on Held-Out Set

```bash
# Evaluate on test set (461 samples, never seen during training)
mamba run -p ./env diana-test \
  --model results/training/best_model.pth \
  --config results/training/final_training_config.json \
  --matrix data/matrices/large_matrix_3070_with_frac/unitigs.frac.mat \
  --metadata data/splits/test_metadata.tsv \
  --test-ids data/splits/test_ids.txt \
  --output results/test_evaluation
```

**Output:**
- `results/test_evaluation/test_metrics.json`
- `results/test_evaluation/test_predictions.tsv`

### Generate Plots and Tables

```bash
# Create all publication figures and tables
mamba run -p ./env python scripts/evaluation/04_model_performance_metrics.py \
  --metrics results/test_evaluation/test_metrics.json \
  --history results/training/training_history.json \
  --config results/training/final_training_config.json \
  --predictions results/test_evaluation/test_predictions.tsv \
  --label-encoders results/training/label_encoders.json \
  --output-dir paper
```

**Output:**
- `paper/figures/model_evaluation/` (HTML + PNG)
  - `test_set_multitask_performance_summary`
  - `test_set_confusion_matrix_{task}`
  - `test_set_roc_curves_{task}`
  - `test_set_pr_curves_{task}`
  - `training_set_loss_curves`
- `paper/tables/model_evaluation/`
  - `test_set_performance_summary.csv`
  - `test_set_per_class_metrics_{task}.csv`
  - `hyperparameters.csv`

---

## Feature Analysis

### 1. Extract Feature Importance

```bash
# Compute gradient-based and weight-based importance
mamba run -p ./env python scripts/feature_analysis/01_extract_feature_importance.py \
  --model results/training/best_model.pth \
  --config results/training/final_training_config.json \
  --matrix data/matrices/large_matrix_3070_with_frac/unitigs.frac.mat \
  --metadata data/splits/test_metadata.tsv \
  --output paper/tables/feature_analyses
```

**Output:** Top 50 features per task with importance scores

### 2. Analyze Sequences

```bash
# Extract sequences and compute GC content, length, complexity
mamba run -p ./env python scripts/feature_analysis/02_analyze_feature_sequences.py \
  --importance-dir paper/tables/feature_analyses \
  --unitigs-fa data/matrices/large_matrix_3070_with_frac/unitigs.fa \
  --unitigs-mat data/matrices/large_matrix_3070_with_frac/unitigs.frac.mat \
  --output paper
```

**Output:** Sequence statistics, distribution plots, fraction prevalence

### 3. Annotate with BLAST

```bash
# BLAST against NCBI nt database
sbatch scripts/feature_analysis/run_blast_annotation.sbatch

# Parse results and create taxonomic visualizations
mamba run -p ./env python scripts/feature_analysis/03_annotate_features.py \
  --importance-dir paper/tables/feature_analyses \
  --blast-results paper/blast_results/blast_results.tsv \
  --unitigs-fa data/matrices/large_matrix_3070_with_frac/unitigs.fa \
  --output paper
```

**Output:** 
- Annotated feature tables with species names
- Taxonomic composition plots (Phylum → Family → Genus)
- Sunburst charts with hierarchical taxonomy

---

## Validation

### Download External Dataset

```bash
# Expand metadata with ENA run accessions
mamba run -p ./env python scripts/validation/01_expand_metadata.py \
  --input data/validation/validation_metadata_v25.09.0.tsv \
  --output data/validation/validation_metadata_expanded.tsv

# Prefetch .sra files (880 samples)
bash scripts/validation/03_prefetch_all.sh

# Convert to FASTQ
sbatch --array=1-879%20 scripts/validation/04_convert_sra_to_fastq.sbatch
```

**Output:** `data/validation/raw/{accession}/*.fastq.gz`

### Run Inference

```bash
# Extract k-mers and predict
bash scripts/inference/inference_pipeline.sh \
  --input data/validation/raw \
  --model results/training/best_model.pth \
  --output results/validation_predictions
```

---

## Output Structure

```
decOM-classify/
├── data/
│   ├── matrices/large_matrix_3070_with_frac/
│   │   ├── unitigs.frac.mat          # 3070×107480 feature matrix
│   │   └── unitigs.fa                # Unitig sequences
│   ├── metadata/DIANA_metadata.tsv   # Full metadata
│   └── splits/
│       ├── train_ids.txt             # 2609 samples
│       ├── test_ids.txt              # 461 samples
│       ├── train_metadata.tsv
│       └── test_metadata.tsv
├── results/
│   ├── training/
│   │   ├── cv_results/
│   │   │   └── best_hyperparameters.json
│   │   ├── best_model.pth
│   │   ├── training_history.json
│   │   └── label_encoders.json
│   └── test_evaluation/
│       ├── test_metrics.json
│       └── test_predictions.tsv
└── paper/
    ├── figures/
    │   ├── model_evaluation/         # Test set performance
    │   ├── feature_analyses/         # Sequence & taxonomy plots
    │   └── data_distribution/        # Metadata distributions
    └── tables/
        ├── model_evaluation/         # Metrics & per-class stats
        └── feature_analyses/         # Top features & annotations
```

---

**Last Updated:** December 24, 2025  
**Contact:** cduitama@pasteur.fr
