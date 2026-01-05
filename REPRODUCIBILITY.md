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

### 1. Build Unitig Matrix with muset

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
│   ├── multitask_fold_0_results_<timestamp>.json
│   ├── optuna_study.db
│   └── training_log.txt
├── fold_1/
├── fold_2/
├── fold_3/
└── fold_4/
```

**⚠️ WAIT:** All 5 SLURM jobs must complete before Step 2! Check with `squeue -j <job_id>`

### Step 2: Aggregate Fold Results

After all folds complete, aggregate results to determine best hyperparameters:

```bash
# Aggregate 5 fold results into best hyperparameters
mamba run -p ./env diana-train multitask \
  --config configs/train_config.yaml \
  --output results/training \
  --mode aggregate
```

**What happens:**
- Reads results from all 5 folds in `results/training/cv_results/fold_*/`
- Aggregates hyperparameters (mean for numeric, mode for categorical)
- Computes mean ± std for metrics across folds
- Saves aggregated results

**Expected outputs:**
```
results/training/cv_results/
├── fold_0/ ... fold_4/               # From Step 1
├── best_hyperparameters.json         # Aggregated best params
└── aggregated_results.json           # Metrics summary
```

**Example aggregated hyperparameters:**
```json
{
  "learning_rate": 0.0021,
  "weight_decay": 0.0001,
  "hidden_dims": [512, 256, 128],
  "dropout": 0.3,
  "batch_size": 64
}
```

### Step 3: Train Final Model

```bash
# Train on full training set with aggregated best hyperparameters
mamba run -p ./env diana-train multitask \
  --config configs/train_config.yaml \
  --output results/full_training \
  --mode train
```

**What happens:**
- Automatically loads `results/training/cv_results/best_hyperparameters.json`
- Trains on 90% of 2609 train samples (2348 samples)
- Uses 10% for validation and early stopping (261 samples)
- Saves model when validation loss plateaus

**Expected outputs:**
```
results/full_training/
├── best_model.pth                    # Trained model weights
├── training_history.json             # Loss/accuracy curves
├── label_encoders.json               # Class mappings for each task
├── final_training_config.json        # Full config used for training
└── cv_results/                       # From Steps 1-2 (symlink or copied)
    ├── best_hyperparameters.json
    └── aggregated_results.json
```

**Training time:** ~10-30 minutes on GPU (depends on early stopping)

---

## Model Evaluation

### Step 4: Test on Held-Out Set

```bash
# Evaluate on test set (461 samples, never seen during training or optimization)
mamba run -p ./env diana-test \
  --model results/full_training/best_model.pth \
  --config results/full_training/final_training_config.json \
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

### Step 5: Generate Performance Plots and Tables

```bash
# Create publication-ready figures and tables
mamba run -p ./env python scripts/evaluation/04_model_performance_metrics.py \
  --metrics results/test_evaluation/test_metrics.json \
  --history results/full_training/training_history.json \
  --config results/full_training/final_training_config.json \
  --predictions results/test_evaluation/test_predictions.tsv \
  --label-encoders results/full_training/label_encoders.json \
  --output-dir paper/figures
```

**Expected outputs:**
```
paper/figures/model_evaluation/
├── test_set_multitask_performance_summary.html
├── test_set_confusion_matrix_sample_type.png
├── test_set_confusion_matrix_community_type.png
├── test_set_confusion_matrix_sample_host.png
├── test_set_confusion_matrix_material.png
├── test_set_roc_curves_sample_type.html
├── test_set_pr_curves_sample_type.html
└── training_loss_curves.html

paper/tables/model_evaluation/
├── test_set_performance_summary.csv
├── test_set_per_class_metrics_sample_type.csv
├── test_set_per_class_metrics_community_type.csv
├── test_set_per_class_metrics_sample_host.csv
├── test_set_per_class_metrics_material.csv
└── hyperparameters.csv
```

---

## Feature Analysis

### Step 6: Extract Feature Importance

```bash
# Compute gradient-based and weight-based importance scores
mamba run -p ./env python scripts/feature_analysis/01_extract_feature_importance.py \
  --config configs/feature_analysis.yaml
```

**Expected outputs:**
```
paper/tables/feature_analysis/
├── top_100_features_weight_based.csv    # Top 100 features per task (weight method)
├── top_100_features_weight_based.md     # Markdown summary
├── top_100_features_gradient_based.csv  # Top 100 features per task (gradient method)
└── top_100_features_gradient_based.md   # Markdown summary

paper/figures/feature_analysis/
├── feature_importance_heatmap_weight_based.html
├── feature_importance_heatmap_weight_based.png
├── feature_importance_heatmap_gradient_based.html
├── feature_importance_heatmap_gradient_based.png
├── feature_overlap_weight_based.html
├── feature_overlap_weight_based.png
├── feature_overlap_gradient_based.html
├── feature_overlap_gradient_based.png
├── feature_importance_comparison.html
└── feature_importance_comparison.png
```

### Step 7: Analyze Feature Sequences

```bash
# Compute sequence properties (GC content, length, complexity)
mamba run -p ./env python scripts/feature_analysis/02_analyze_feature_sequences.py \
  --config configs/feature_analysis.yaml
```

**Expected outputs:**
```
paper/tables/feature_analysis/
├── sequence_properties_sample_type.csv       # GC%, length, complexity
├── sequence_properties_community_type.csv
├── sequence_properties_sample_host.csv
└── sequence_properties_material.csv

paper/figures/feature_analysis/
├── gc_content_distribution.html
├── length_distribution.html
└── fraction_prevalence.html
```

### Step 8: Taxonomic Annotation with BLAST

```bash
# Run BLAST against NCBI nt database (takes several hours)
sbatch scripts/feature_analysis/run_blast_annotation.sbatch

# Wait for BLAST jobs to complete, then parse results
mamba run -p ./env python scripts/feature_analysis/03_annotate_features.py \
  --config configs/feature_analysis.yaml
```

**Expected outputs:**
```
paper/tables/feature_analysis/
├── annotated_features_sample_type.csv        # With taxonomic assignments
├── annotated_features_community_type.csv
├── annotated_features_sample_host.csv
└── annotated_features_material.csv

paper/figures/feature_analysis/
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
  --output data/validation/validation_metadata_expanded_raw.tsv

# Prefetch .sra files (880 samples)
bash scripts/validation/03_prefetch_all.sh

# Convert to FASTQ
sbatch --array=1-879%20 scripts/validation/04_convert_sra_to_fastq.sbatch
```

**Output:** `data/validation/raw/{accession}/*.fastq.gz`

### Run Inference on Validation Set

```bash
sbatch --array=1-629%10 scripts/validation/05_run_predictions.sbatch
```

---

**Last Updated:** January 2026
**Contact:** cduitama@pasteur.fr
