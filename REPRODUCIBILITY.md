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
9. [Script Organization](#script-organization)

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

**Configuration:** `configs/train_config.yaml`

Key settings:
- **Data:** Uses `train_metadata.tsv` (2609 samples only)
- **Tasks:** sample_type, community_type, sample_host, material
- **CV:** 5-fold outer CV, 3-fold inner CV
- **Optimization:** 50 Optuna trials per fold
- **Execution:** SLURM GPU array jobs (`use_slurm: true`)

### Step 1: Hyperparameter Optimization

```bash
# Submit 5-fold CV hyperparameter search (SLURM array job)
mamba run -p ./env diana-train multitask \
  --config configs/train_config.yaml \
  --output results/training \
  --mode optimize
```

**What happens:**
- Submits SLURM array job with 5 tasks (fold_0 through fold_4)
- Each fold runs 50 Optuna trials with 3-fold inner CV
- Jobs run in parallel on GPU nodes
- Command exits immediately after submission

**Monitor jobs:**
```bash
# Check job status (use job ID from diana-train output)
squeue -j <job_id>
sacct -j <job_id> --format=JobID,State,ExitCode,Elapsed

# Check logs (created in logs/ directory)
tail -f logs/*.err
grep "Best trial" logs/*.err  # See best hyperparams per fold
```

**Expected outputs:**
```
results/training/cv_results/
├── fold_0/
│   ├── best_hyperparameters.json
│   ├── optuna_study.db
│   └── training_log.txt
├── fold_1/
├── fold_2/
├── fold_3/
├── fold_4/
└── best_hyperparameters.json  # Aggregated best params
```

**⚠️ WAIT:** All 5 SLURM jobs must complete before Step 2! Check with `squeue -j <job_id>`

### Step 2: Train Final Model

```bash
# Train on full training set with aggregated best hyperparameters
mamba run -p ./env diana-train multitask \
  --config configs/train_config.yaml \
  --output results/training \
  --mode train
```

**What happens:**
- Automatically loads `results/training/cv_results/best_hyperparameters.json`
- Trains on 90% of 2609 train samples (2348 samples)
- Uses 10% for validation and early stopping (261 samples)
- Saves model when validation loss plateaus

**Expected outputs:**
```
results/training/
├── best_model.pth                    # Trained model weights
├── training_history.json             # Loss/accuracy curves
├── label_encoders.json               # Class mappings for each task
├── final_training_config.json        # Full config used for training
└── cv_results/                       # From Step 1
    └── best_hyperparameters.json
```

**Training time:** ~10-30 minutes on GPU (depends on early stopping)

---

## Model Evaluation

### Step 3: Test on Held-Out Set

```bash
# Evaluate on test set (461 samples, never seen during training or optimization)
mamba run -p ./env diana-test \
  --model results/training/best_model.pth \
  --config results/training/final_training_config.json \
  --matrix data/matrices/large_matrix_3070_with_frac/unitigs.frac.mat \
  --metadata data/splits/test_metadata.tsv \
  --test-ids data/splits/test_ids.txt \
  --output results/test_evaluation
```

**Expected outputs:**
```
results/test_evaluation/
├── test_metrics.json                 # Per-task accuracy, F1, etc.
├── test_predictions.tsv              # Predictions for all 461 samples
├── confusion_matrices/               # Per-task confusion matrices
└── classification_reports/           # Detailed per-class metrics
```

### Step 4: Generate Performance Plots and Tables

```bash
# Create publication-ready figures and tables
mamba run -p ./env python scripts/evaluation/04_model_performance_metrics.py \
  --metrics results/test_evaluation/test_metrics.json \
  --history results/training/training_history.json \
  --config results/training/final_training_config.json \
  --predictions results/test_evaluation/test_predictions.tsv \
  --label-encoders results/training/label_encoders.json \
  --output-dir results/figures
```

**Expected outputs:**
```
results/figures/model_evaluation/
├── test_set_multitask_performance_summary.html
├── test_set_confusion_matrix_sample_type.png
├── test_set_confusion_matrix_community_type.png
├── test_set_confusion_matrix_sample_host.png
├── test_set_confusion_matrix_material.png
├── test_set_roc_curves_sample_type.html
├── test_set_pr_curves_sample_type.html
└── training_loss_curves.html

results/tables/model_evaluation/
├── test_set_performance_summary.csv
├── test_set_per_class_metrics_sample_type.csv
├── test_set_per_class_metrics_community_type.csv
├── test_set_per_class_metrics_sample_host.csv
├── test_set_per_class_metrics_material.csv
└── hyperparameters.csv
```

---

## Feature Analysis

### Step 5: Extract Feature Importance

```bash
# Compute gradient-based importance scores for top features
mamba run -p ./env python scripts/feature_analysis/01_extract_feature_importance.py \
  --model results/training/best_model.pth \
  --config results/training/final_training_config.json \
  --matrix data/matrices/large_matrix_3070_with_frac/unitigs.frac.mat \
  --metadata data/splits/test_metadata.tsv \
  --output results/feature_analysis
```

**Expected outputs:**
```
results/feature_analysis/
├── importance_sample_type.csv        # Top 50 features for sample_type
├── importance_community_type.csv
├── importance_sample_host.csv
├── importance_material.csv
└── importance_scores_summary.json    # Statistics across all tasks
```

### Step 6: Analyze Feature Sequences

```bash
# Compute sequence properties (GC content, length, complexity)
mamba run -p ./env python scripts/feature_analysis/02_analyze_feature_sequences.py \
  --importance-dir results/feature_analysis \
  --unitigs-fa data/matrices/large_matrix_3070_with_frac/unitigs.fa \
  --unitigs-mat data/matrices/large_matrix_3070_with_frac/unitigs.frac.mat \
  --output results/feature_analysis
```

**Expected outputs:**
```
results/feature_analysis/
├── sequence_properties_sample_type.csv       # GC%, length, complexity
├── sequence_properties_community_type.csv
├── sequence_properties_sample_host.csv
├── sequence_properties_material.csv
└── figures/
    ├── gc_content_distribution.html
    ├── length_distribution.html
    └── fraction_prevalence.html
```

### Step 7: Taxonomic Annotation with BLAST

```bash
# Run BLAST against NCBI nt database (takes several hours)
sbatch scripts/feature_analysis/run_blast_annotation.sbatch

# Wait for BLAST jobs to complete, then parse results
mamba run -p ./env python scripts/feature_analysis/03_annotate_features.py \
  --importance-dir results/feature_analysis \
  --blast-results results/blast/blast_results.tsv \
  --unitigs-fa data/matrices/large_matrix_3070_with_frac/unitigs.fa \
  --output results/feature_analysis
```

**Expected outputs:**
```
results/feature_analysis/
├── annotated_features_sample_type.csv        # With taxonomic assignments
├── annotated_features_community_type.csv
├── annotated_features_sample_host.csv
├── annotated_features_material.csv
└── figures/
    ├── taxonomy_phylum_sample_type.html
    ├── taxonomy_family_sample_type.html
    ├── taxonomy_genus_sample_type.html
    └── taxonomy_sunburst_sample_type.html    # Interactive hierarchy
```

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

After completing all steps, you should have:

```
decOM-classify/
├── data/
│   ├── matrices/large_matrix_3070_with_frac/
│   │   ├── unitigs.frac.mat          # 3070×107480 k-mer matrix
│   │   └── unitigs.fa                # Unitig sequences (FASTA)
│   ├── metadata/DIANA_metadata.tsv   # Full metadata (3070 samples)
│   └── splits/                       # 85/15 train/test split
│       ├── train_ids.txt             # 2609 sample IDs
│       ├── test_ids.txt              # 461 sample IDs
│       ├── train_metadata.tsv        # Training metadata
│       └── test_metadata.tsv         # Test metadata
│
├── results/
│   ├── training/                     # Model training outputs
│   │   ├── best_model.pth            # Final trained model
│   │   ├── training_history.json     # Loss/accuracy curves
│   │   ├── label_encoders.json       # Class label mappings
│   │   ├── final_training_config.json
│   │   └── cv_results/               # Hyperparameter optimization
│   │       ├── best_hyperparameters.json  # Aggregated best params
│   │       └── fold_{0-4}/           # Per-fold results
│   │
│   ├── test_evaluation/              # Test set performance
│   │   ├── test_metrics.json         # Accuracy, F1, etc.
│   │   ├── test_predictions.tsv      # Model predictions
│   │   ├── confusion_matrices/
│   │   └── classification_reports/
│   │
│   ├── feature_analysis/             # Important features
│   │   ├── importance_{task}.csv     # Top 50 features per task
│   │   ├── annotated_features_{task}.csv  # With taxonomy
│   │   ├── sequence_properties_{task}.csv
│   │   └── figures/
│   │
│   └── figures/                      # Publication figures
│       └── model_evaluation/
│           ├── test_set_confusion_matrix_{task}.png
│           ├── test_set_roc_curves_{task}.html
│           └── training_loss_curves.html
│
└── logs/                             # SLURM job logs
    ├── diana_train_*.err
    └── blast_*.err
```

---

## Script Organization

All scripts are organized in numbered sequence within functional subdirectories:

```
scripts/
├── create_umat/                      # K-mer matrix generation
│   ├── 01_build_muset.sh            # Build MUSET tool
│   └── 02_regenerate_matrix_with_frac.sbatch  # Generate unitig matrix
│
├── data_prep/                        # Data preparation
│   ├── 01_create_splits.py          # Train/test split
│   ├── 02_extract_and_split_matrices.py
│   └── 03_analyze_unitigs.py
│
├── training/                         # Model training
│   ├── 01_train_multitask_single_fold.py  # CV fold training
│   ├── 02_train_final_model.py      # Final model training
│   ├── run_multitask_gpu.sbatch     # SLURM submission for CV
│   └── run_final_training_gpu.sbatch
│
├── evaluation/                       # Model evaluation
│   ├── 01_plot_data_distribution.py
│   ├── 02_statistical_tests_splits.py
│   ├── 03_collect_multitask_results.py
│   └── 04_model_performance_metrics.py  # Main evaluation script
│
├── feature_analysis/                 # Feature importance
│   ├── 01_extract_feature_importance.py
│   ├── 02_analyze_feature_sequences.py
│   ├── 03_annotate_features.py
│   └── run_blast_annotation.sbatch
│
├── inference/                        # Production inference
│   ├── 00_extract_reference_kmers.sh
│   ├── 01_count_kmers.sh
│   ├── 02_aggregate_to_unitigs.sh
│   └── inference_pipeline.sh
│
└── validation/                       # External validation
    ├── 01_expand_metadata.py
    ├── 02_prepare_download.py
    ├── 03_prefetch_all.sh
    └── 04_convert_sra_to_fastq.sbatch
```

**Configuration files:**
```
configs/
├── train_config.yaml                 # Main training configuration
├── data_config.yaml                  # Data processing config
└── feature_analysis.yaml             # Feature analysis config
```

---

## Key Points for Reproducibility

1. **No Data Leakage:** Test set (461 samples) completely held out until final evaluation
2. **Train Set Filtering:** All scripts use `train_metadata.tsv` which filters the full matrix to 2609 training samples
3. **Fixed Random Seed:** `random_state=42` for all splits and cross-validation
4. **Stratification:** Train/test split stratified by `sample_type` to maintain class balance
5. **Hyperparameter Search:** Nested CV on training set only (5 outer folds × 3 inner folds)
6. **Early Stopping:** 10% of training set used as validation to prevent overfitting

---

**Last Updated:** December 24, 2025  
**Contact:** cduitama@pasteur.fr
