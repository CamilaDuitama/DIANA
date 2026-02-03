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

#Install diana-train
mamba run -p ./env pip install -e .

# Verify installation
python -c "import torch, polars, plotly; print('✓ Environment ready')"
```

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

**Output:** `unitigs.frac.mat` (107,480 features × 3,071 samples, 1.6GB)

> **Note:** Matrix is stored in transposed format (samples as rows). The 107,480 rows represent unitigs, and 3,071 columns represent samples. When loaded by `MatrixLoader`, it's automatically transposed to (3,070 samples × 107,480 features).

### 2. Prepare Metadata

Metadata files are located in `paper/metadata/`:
- `train_metadata.tsv` (2,609 samples)
- `test_metadata.tsv` (461 samples)
- `validation_metadata.tsv` (1,029 samples)

**All three files have identical 48 columns** (standardized format).

**Training + Test combined: 3,070 samples**

**Task columns and classes (train/test):**
- `sample_type`: 2 classes (ancient_metagenome, modern_metagenome)
- `material`: 13 classes (dental calculus, sediment, tooth, bone, digestive_contents, etc.)
- `sample_host`: 12 classes (Homo sapiens, Not applicable - env sample, Ursus arctos, Gorilla sp., etc.)
- `community_type`: 6 classes (oral, Not applicable - env sample, skeletal tissue, soft tissue, gut, plant tissue)

**Key columns:**
- `Run_accession`: Sample identifier
- `sample_type`, `material`, `sample_host`, `community_type`: Target labels
- Plus 44 additional metadata columns (SRA fields, sequence stats, etc.)

> **Note:** Validation set contains additional classes not in training (20 material types, 18 host species) due to broader ancient sample diversity.

---

## Train/Test Split

The train/test split is **already prepared** in `data/splits/`:
- `train_ids.txt` (2,609 samples, 85%)
- `test_ids.txt` (461 samples, 15%)

Metadata files in `paper/metadata/` are filtered versions:
- `train_metadata.tsv` - Contains only training samples
- `test_metadata.tsv` - Contains only test samples

**Critical:** Test set is held out for final evaluation only. Never used during training or hyperparameter optimization.

<details>
<summary>To regenerate splits from scratch (optional)</summary>

```bash
# Create stratified 85/15 train/test split
mamba run -p ./env python scripts/data_prep/01_create_splits.py \
  --metadata data/metadata/DIANA_metadata.tsv \
  --output data/splits \
  --train-size 0.85 \
  --test-size 0.15 \
  --random-state 42
```

This will regenerate `train_ids.txt`, `test_ids.txt`, and metadata files.

</details>

---

## Model Training

**Configuration:** `configs/train_config.yaml`

Key settings:
- **Data:** Uses `paper/metadata/train_metadata.tsv` (2,609 samples only)
- **Tasks:** sample_type, material, sample_host, community_type
- **Class imbalance:** Automatic class-weighted loss (minority classes weighted higher)
- **CV:** 5-fold outer CV, 3-fold inner CV
- **Optimization:** 50 Optuna trials per fold
- **Execution:** SLURM GPU array jobs (`use_slurm: true`)

### Two Training Approaches

You can use either the **CLI workflow** (recommended for reproducibility) or the **direct SBATCH submission** (for manual control).

---

#### Approach A: CLI Workflow (Recommended)

**Step 1: Hyperparameter Optimization**

```bash
# Submit 5-fold CV hyperparameter search (SLURM array job)
mamba run -p ./env diana-train multitask \
  --config configs/train_config.yaml \
  --output results/training \
  --mode optimize
```

This internally submits `scripts/training/run_multitask_gpu.sbatch` as a SLURM array job.

**Step 2: Train Final Model**

After all folds complete (`squeue -j <job_id>` shows no jobs):

```bash
# Train on full training set with best hyperparameters from CV
mamba run -p ./env diana-train multitask \
  --config configs/train_config.yaml \
  --output results/training \
  --mode train
```

**What happens:**
- Automatically aggregates fold results if `best_hyperparameters.json` doesn't exist
- Trains on 90% of training set (2,348 samples)
- Uses 10% for validation and early stopping (261 samples)
- Saves final model when validation loss plateaus

**Optional:** To manually aggregate fold results before training:
```bash
mamba run -p ./env diana-train multitask \
  --config configs/train_config.yaml \
  --output results/training \
  --mode aggregate
```

## Model Evaluation

### Step 3: Test on Held-Out Set

```bash
# Evaluate on test set (461 samples, never seen during training or optimization)
mamba run -p ./env diana-test \
  --model results/training/best_model.pth \
  --config results/training/final_training_config.json \
  --matrix data/matrices/large_matrix_3070_with_frac/unitigs.frac.mat \
  --metadata paper/metadata/test_metadata.tsv \
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
  --output-dir paper
```

**Expected outputs:**
```
paper/figures/
├── test_set_multitask_performance_summary.html + .png
├── test_set_confusion_matrix_{task}.html + .png       # 4 tasks
├── test_set_per_class_metrics_{task}.html + .png      # 4 tasks
├── test_set_roc_curves_{task}.html + .png             # 4 tasks
├── test_set_pr_curves_{task}.html + .png              # 4 tasks
└── training_set_loss_curves.html + .png

paper/tables/
├── test_set_performance_summary.csv + .tex + .html + .png
├── test_set_per_class_metrics_{task}.csv + .tex       # 4 tasks
└── hyperparameters.csv + .tex + .html + .png
```
---

## Feature Analysis

### Step 5: Extract Feature Importance

```bash
# Compute gradient-based and weight-based importance scores
mamba run -p ./env python scripts/feature_analysis/01_extract_feature_importance.py \
  --config configs/feature_analysis.yaml
```

### Step 6: Analyze Feature Sequences

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

### Step 7: Taxonomic Annotation with BLAST

```bash
# Run BLAST against NCBI nt database for all 107,480 unitig features (4-12 hours)
sbatch scripts/feature_analysis/run_blast_all_features.sbatch

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

### Step 8: Aggregate Feature Analysis for Paper Figures

```bash
# Create aggregated TSV files for validation comparison plots
mamba run -p ./env python scripts/feature_analysis/05_aggregate_for_validation.py
```

**Expected outputs:**
```
results/feature_analysis/
├── feature_importance_by_genus.tsv     # Genus counts per task (for main figure)
└── blast_annotations.tsv               # BLAST hit rates (for main figure)
```

---

## Validation

### Prepare Validation Dataset

The validation set combines:
- **Ancient samples** from AncientMetagenomeDir (863 samples, highly curated)
- **Modern samples** from interactive review (166 metagenomics samples, improved distribution)

**Key principle:** Match the training set's sample type distribution (84% ancient, 16% modern)
while testing model generalization to completely unseen samples.

> **Starting fresh vs continuing:** 
> - If you have existing downloads: Scripts will **skip existing files** automatically
> - If starting from scratch: Follow all steps below  
> - If you only want to re-query modern samples: Skip to [step 2](#2-get-modern-samples-from-mgnify-recommended)

#### 1. Get Ancient Samples from AncientMetagenomeDir

```bash
# Combine host-associated and environmental samples
mamba run -p ./env python scripts/validation/00_prepare_combined_metadata.py

# Expand to run accessions via ENA API
mamba run -p ./env python scripts/validation/01_expand_metadata.py

# Remove train/test overlaps
mamba run -p ./env python scripts/validation/01b_remove_overlap.py

# Prepare download list
mamba run -p ./env python scripts/validation/02_prepare_download.py
```

**Output:** `data/validation/accessions.txt` (863 ancient samples initially)

#### 2. Add Modern Samples via Interactive Review

Modern samples were added through manual curation using interactive review:

```bash
# Interactive review of modern samples with SRA metadata
# (Already completed - 166 modern samples approved and merged into validation_metadata.tsv)
mamba run -p ./env python scripts/validation/interactive_label_review.py

# Merge reviewed samples into validation_metadata.tsv
# (Already completed - validation_metadata.tsv now contains 1,029 samples)
mamba run -p ./env python scripts/validation/merge_reviewed_samples.py
```

**Modern sample distribution (166 samples):**
- soil: 65 (39%)
- plaque: 24 (14%)
- faeces: 24 (14%)
- skin: 24 (14%)
- saliva: 17 (10%)
- Other oral/environmental: 12 (9%)

> **Note:** Modern samples exclude RNA-based sequencing (transcriptomics, miRNA-seq, etc.) but include AMPLICON (16S/ITS) and WGS metagenomics.

#### 3. Download All Samples

> **Note:** If you already have validation samples downloaded, these scripts will **skip existing files**
> automatically. They check for:
> - Existing SRA files in `data/validation/sra/`
> The accessions.txt file is automatically updated when merging reviewed samples
# It now contains all 1,010 unique run accessions from validation_metadata.tsv

# Will only download the ~89 newly added modern samples
bash scripts/validation/03_prefetch_all.sh

# Convert SRA → FASTQ (auto-skips existing)
# Update array size to match total accessions
sbatch --array=1-1171%20 scripts/validation/04_convert_sra_to_fastq.sbatch
```

**Output:** `data/validation/sra/{accession}/*.sra` and `data/validation/raw/{accession}/*.fastq.gz`

```

**Output:** `data/validation/sra/{accession}/*.sra` and `data/validation/raw/{accession}/*.fastq.gz`


**To check download progress:**
```bash
# Count downloaded SRA files
find data/validation/sra -name "*.sra" | wc -l
1,010 unique run accessions (some samples have multiple runs)
```

> **Distribution Summary:**
> - Ancient samples: 84% (863 from AncientMetagenomeDir, highly curated)
> - Modern samples: 16% (166 from interactive review, improved balance)
> - Modern material distribution: soil 39%, plaque 14%, faeces 14%, skin 14%, saliva 10%
> **Distribution Summary:**
> - Ancient samples: 93% (from AncientMetagenomeDir, high quality curated)
> - Modern samples: 7% (from MGnify, matches training 7.6% modern proportion)
> - Material distribution in modern samples matches training exactly

### Run Inference on Validation Set

**Automated retry system with memory scaling:**

```bash
# Submit validation predictions with automatic OOM retry
bash scripts/validation/submit_validation_with_retry.sh
```

**How it works:**
1. **Initial run:** All samples start with 32GB memory
2. **Monitoring:** Each sample tracks job metadata (`.jobinfo`) and memory history (`.memory_history`)
3. **Cache check:** Skips samples with `"status": "SUCCESS"` in `.jobinfo`
4. **OOM detection:** Failed jobs are retried with doubled memory (32→64→128→256→512GB)
5. **Re-run:** Simply execute the script again after jobs complete to retry OOM failures

**Example workflow:**
```bash
# First submission (all samples @ 32GB)
bash scripts/validation/submit_validation_with_retry.sh
# → Submits array jobs for each memory tier

# Monitor progress
squeue -u $USER
reportseff <job_id>

# After completion, retry OOM failures @ doubled memory
bash scripts/validation/submit_validation_with_retry.sh

# Continue until all samples complete or hit 512GB limit
bash scripts/validation/submit_validation_with_retry.sh
```

**Output structure per sample:**
```
results/validation_predictions/
├── {accession}/
│   ├── {accession}_predictions.json   # Prediction output
│   ├── .jobinfo                       # Job metadata (job_id, memory, runtime, status)
│   └── .memory_history                # Memory allocations tried (MB, one per line)
└── ...

logs/validation/
├── diana_predict_{JOB_ID}_{TASK_ID}.out   # stdout
├── diana_predict_{JOB_ID}_{TASK_ID}.err   # stderr
└── ...
```

**Monitor progress:**
```bash
# Check job status
squeue -u $USER

# Count completed predictions
find results/validation_predictions -name "*_predictions.json" | wc -l

# Expected: ~1000 samples from paper/metadata/validation_metadata.tsv
1,029
# Check recent logs
tail -f logs/validation/diana_predict_*.err
```

**After all predictions complete, generate validation metrics and figures:**
```bash
# Load validation predictions with metadata (shared utility)
# This creates a standardized DataFrame used by all paper scripts
mamba run -p ./env python scripts/validation/load_validation_data.py

# Output: DataFrame with columns - sample_id, task, true_label, pred_label, 
#         confidence, is_correct, is_seen

# Generate all publication-ready figures and tables
bash scripts/paper/generate_all_paper_materials.sh

# Or generate individual outputs:
# - Figures: main_01-04 (confusion, ROC/PR, features, BLAST)
# - Figures: sup_01-02 (runtime, data split)  
# - Tables: main_table_01-02 (performance summary, computational resources)
# - Tables: sup_table_01-06 (class distribution, unseen labels, per-class performance, 
#                             hyperparameters, wrong predictions, BLAST summary)

# Legacy comparison script (still functional):
mamba run -p ./env python scripts/validation/06_compare_predictions.py
```

---

**Last Updated:** February 2026

