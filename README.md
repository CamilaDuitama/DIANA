# DIANA: Reproducibility Guide

**Multi-task classification of ancient DNA samples using unitig features**

---

## Model Versions

> **v9 is the current version. Everything else on this page is historical.**
> Do not train, evaluate or report from v5 or v7 artefacts — their splits, labels
> and feature vocabulary are all superseded, and their numbers are not comparable
> with v9's. See `PROJECT.md` for the full rationale.

### v9 — current

| | |
|---|---|
| labels | `data/v9_labels_prepartition/` (corrected targets, pre-split) |
| split | `data/splits_v9/` — BioProject-disjoint, asserted in code |
| features | `data/matrices/matrix_v9_train/` — 110,202 unitigs |
| config | `configs/train_config_bioproject_v9.json` |
| results | `results/training_bioproject_v9/` |

**Four classification heads:** `community_type` (host-associated vocabulary),
`feature` (environmental vocabulary), `sample_host`, `material`. No regression:
`sample_age`, `latitude` and `longitude` were dropped as future work, and
`sample_type` has been out since v7 (every sample is an ancient metagenome).

What makes v9 different, and why the older numbers cannot be reused:

1. **The split is genuinely BioProject-disjoint and checked.** v7's
   `split_config.json` claimed disjointness across all three splits; only
   train↔test actually held. Disjointness is now asserted in the split script,
   which refuses to write if it fails.
2. **The feature vocabulary is built from training runs only** (R3.4). v7 built it
   over train *and* test together, so the test set influenced which unitigs exist.
   Val and test are scored against the fixed v9 vocabulary afterwards, using the
   same feature-extraction path as `diana-predict`.
3. **Absent labels are masked, not encoded as a class.** An environmental run has
   no `sample_host`; that is now excluded from the loss and from every metric,
   rather than trained as a `nan` category.
4. **Class imbalance is handled by logit adjustment** (τ=1) rather than
   inverse-frequency weights, which taught v7 a balanced posterior that was then
   argmaxed on an imbalanced split.
5. **Hyperparameters are selected on BioProject-grouped folds.** v7's were tuned
   with a plain `StratifiedKFold`, so R3.2 applied to the search as well as to the
   split.

**There is no v8 model.** `05_build_v9_labels.py` writes the corrected label table
and `06_create_bioproject_splits_v9.py` partitions it; the two landed on different
days and the intermediate briefly carried a v8 name. Nothing is trained from it.

### Historical versions

Kept for provenance only. **Not recommended for any new work.**

| Version | Path | Note |
|---------|------|------|
| v1 / v2 | `results/training_bioproject/` | broken HPO: full-batch, test-set leakage |
| v3 | `results/training_bioproject_v3/` | HPO fixed; label smoothing collapsed `sample_host` |
| v4 | `results/training_bioproject_v4/` | v3 architecture without label smoothing |
| v5 | `results/training_bioproject_v5/` | 4 tasks incl. `sample_type` |
| v7 | `results/training_bioproject_v7/` | 6 tasks; leaked split and vocabulary — see above |

The `paper` branch reproduces the submitted manuscript from v5-era artefacts and
must not be updated with new results.

### v5 pipeline (run from project root)

```bash
# 1. Train  (~1-2 h on GPU)
sbatch scripts/training/run_final_train_edid_v5.sbatch

# 2. Evaluate on test set
sbatch scripts/evaluation/run_test_eval_v5.sbatch

# 3. Validation inference (array jobs)
sbatch scripts/validation/run_inference_bioproject_v5.sbatch

# 4. Training-set metrics
mamba run -p ./env python scripts/evaluation/06_compute_training_metrics.py

# 5. Patch baseline comparison JSON with current DIANA values
mamba run -p ./env python scripts/evaluation/patch_diana_in_metrics.py

# 6. Regenerate all paper figures and tables
bash scripts/paper/generate_all_paper_materials.sh
```

---

## Key Methodological Improvements

This implementation includes several critical improvements over the initial version:

1. **BioProject-Disjoint Split:** Train/test split stratified by BioProject to prevent data leakage from multi-sample studies. No research project appears in both training (2,514 samples) and test (523 samples) sets.

2. **No Label Smoothing (v4):** Per-task label smoothing (v3, ε = 0.043–0.064) was evaluated and dropped. Despite correct HPO, it caused `sample_host` validation balanced accuracy to collapse (80.9% → 41.8%) and worsened test ECE for 3/4 tasks. v4 uses ε = 0 for all tasks.

3. **Corrected Nested Cross-Validation:**
   - Each hyperparameter trial evaluated on **all 3 inner folds** (not split across trials)
   - Mini-batch training with DataLoader (batch_size: 32-256)
   - No test set leakage: separate sub-validation split for early stopping
   - Combined stratification: `sample_type + community_type` (12 combinations)
   - Balanced metric: average of (balanced_accuracy + macro_F1) / 2 across tasks

4. **Robust Model Selection:** 5-fold outer CV for final evaluation, hyperparameters averaged across folds for final model training.

---

## Table of Contents

1. [Environment Setup](#environment-setup)
2. [Data Preparation](#data-preparation)
3. [Train/Test Split](#traintest-split)
4. [Model Training](#model-training)
5. [Model Evaluation](#model-evaluation)
6. [Baseline Comparison](#baseline-comparison)
7. [Feature Analysis](#feature-analysis)
8. [Validation](#validation)
9. [Output Structure](#output-structure)
10. [Script Organization](#script-organization)

**Model Evaluation steps:** Setup → Step 1 (hyperopt) → Step 2 (aggregate CV) → Step 3 (train final) → Step 4 (test eval) → Step 5 (plots) → Step 6 (val predictions) → Step 7 (calibration) → Step 8 (training metrics) → Step 9 (all paper materials) → Baseline Comparison

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

**Output:** `unitigs.frac.mat` (107,480 features × 3,038 samples, 1.6GB)

> **Note:** Matrix is stored in transposed format (samples as rows). The 107,480 rows represent unitigs, and 3,038 columns represent samples. When loaded by `MatrixLoader`, it's automatically transposed to (3,037 samples × 107,480 features).

### 2. Prepare Metadata

Metadata files are located in `data/splits_v5/`:
- `train_metadata.tsv` (2,515 samples)
- `test_metadata.tsv` (523 samples)
- `validation_metadata.tsv` (360 samples)

**All three files have identical 48 columns** (standardized format).

**Training + Test combined: 3,037 samples**

**Task columns and classes (train/test):**
- `sample_type`: 2 classes (ancient_metagenome, modern_metagenome)
- `material`: 13 classes (dental calculus, sediment, tooth, bone, digestive_contents, etc.)
- `sample_host`: 12 classes (Homo sapiens, Not applicable - env sample, Ursus arctos, Gorilla sp., etc.)
- `community_type`: 6 classes (oral, Not applicable - env sample, skeletal tissue, soft tissue, gut, plant tissue)

**Key columns:**
- `Run_accession`: Sample identifier
- `sample_type`, `material`, `sample_host`, `community_type`: Target labels
- Plus 44 additional metadata columns (SRA fields, sequence stats, etc.)

> **Note:** Validation set is a subset of training classes (9/17 material types, 2/11 host species) due to its focus on ancient samples only.

---

## Train/Test Split

The train/test split is **already prepared** in `data/splits_v5/` using **BioProject-disjoint stratification** to prevent data leakage from multi-sample studies:
- `train_ids.txt` (2,515 samples, 82.8%)
- `test_ids.txt` (523 samples, 17.2%)

Metadata files in `data/splits_v5/` contain only the respective split samples:
- `train_metadata.tsv` - Contains only training samples (2,515)
- `test_metadata.tsv` - Contains only test samples (523)

**BioProject-disjoint split ensures:**
- No BioProject appears in both train and test sets
- Prevents data leakage from multi-sample studies
- Tests true generalization to unseen research projects

**Critical:** Test set is held out for final evaluation only. Never used during training or hyperparameter optimization.

<details>
<summary>To regenerate splits from scratch (optional)</summary>

```bash
# Regenerate BioProject-disjoint v5 split
mamba run -p ./env python scripts/data_prep/03_create_bioproject_splits_v7_fixed.py
```

This will regenerate `train_ids.txt`, `test_ids.txt`, and metadata files.

</details>

---

## Model Training

**Configuration files:**
- v5 (4 tasks): `configs/train_config_bioproject_v5.yaml`
- v7 (6 tasks): `configs/train_config_bioproject_v7.json`

Key settings (v5 example):
- **Data:** Uses `data/splits_v5/train_metadata.tsv` (2,515 samples, BioProject-disjoint)
- **Tasks:** sample_type, material, sample_host, community_type
- **Class imbalance:** Automatic class-weighted loss (minority classes weighted higher)
- **CV:** 5-fold outer CV, 3-fold inner CV
- **Optimization:** 50 Optuna trials per fold
- **Execution:** SLURM GPU array jobs (`use_slurm: true`)

> **Note:** v7 uses 6 tasks (community_type, sample_host, material, sample_age, latitude, longitude). The `sample_type` task is excluded because all v7 samples are `ancient_metagenome`.

### Training Workflow (Version-Agnostic)

The training pipeline uses config files to specify data paths, tasks, and hyperparameters. The same scripts work for any model version.

**Step 1: Hyperparameter Optimization**

```bash
# Submit 5-fold CV hyperparameter search (SLURM array job)
# v5 example:
sbatch scripts/training/run_hyperopt_bioproject_v5.sbatch

# v7 example:
sbatch scripts/training/run_hyperopt_bioproject_v7.sbatch
```

Monitor progress:
```bash
squeue -u $USER
tail -f logs/hyperopt_v5/fold_0_*.out  # or hyperopt_v7/
```

**Step 2: Aggregate CV Results**

After all folds complete:

```bash
python scripts/training/aggregate_cv_results.py \
  --cv_dir results/training_bioproject_v5/cv_results \
  --n_folds 5
```

Creates `results/training_bioproject_v5/final_training_config.json` (or v7 equivalent) for step 3.

**Step 3: Train Final Model**

```bash
python scripts/training/02_train_final_model.py \
  results/training_bioproject_v5/final_training_config.json
```

Or via SLURM:
```bash
sbatch scripts/training/run_final_train_edid_v5.sbatch  # or v7
```

**What happens:**
- Trains on 90% of training set (e.g., 2,263 samples for v5)
- Uses 10% for validation and early stopping (e.g., 251 samples for v5)
- Saves final model when validation loss plateaus



---

## Model Evaluation

The evaluation pipeline is version-agnostic — paths are determined by the model and split config used.

### Step 4: Test on Held-Out Set

```bash
# Evaluate on test set (BioProject-disjoint, never seen during training or optimization)
# v5 example (523 samples):
mamba run -p ./env diana-test \
  --model results/training_bioproject_v5/best_model.pth \
  --config results/training_bioproject_v5/final_training_config.json \
  --matrix data/matrices/large_matrix_3070_with_frac/unitigs.frac.mat \
  --metadata data/splits_v5/test_metadata.tsv \
  --test-ids data/splits_v5/test_ids.txt \
  --output results/test_evaluation_bioproject_v5

# v7 example (352 samples):
mamba run -p ./env diana-test \
  --model results/training_bioproject_v7/best_model.pth \
  --config results/training_bioproject_v7/final_training_config.json \
  --matrix data/matrices/matrix_v7_3190/unitigs.frac.mat \
  --metadata data/splits_v7/test_metadata.tsv \
  --test-ids data/splits_v7/test_ids.txt \
  --output results/test_evaluation_bioproject_v7
```

**Expected outputs:**
```
results/test_evaluation_bioproject_<version>/
├── test_metrics.json                 # Per-task accuracy, F1, etc.
├── test_predictions.tsv              # Predictions for all test samples
├── confusion_matrices/               # Per-task confusion matrices
└── classification_reports/           # Detailed per-class metrics
```

### Step 5: Generate Performance Plots and Tables

```bash
# Create publication-ready figures and tables
bash scripts/paper/generate_all_paper_materials.sh
```

**Expected outputs:**
```
paper/figures/final/
├── test_set_multitask_performance_summary.html + .png
├── test_set_confusion_matrix_{task}.html + .png       # one per task
├── test_set_per_class_metrics_{task}.html + .png      # one per task
├── test_set_roc_curves_{task}.html + .png             # one per task
├── test_set_pr_curves_{task}.html + .png              # one per task
└── training_set_loss_curves.html + .png

paper/tables/final/
├── test_set_performance_summary.csv + .tex + .html + .png
├── test_set_per_class_metrics_{task}.csv + .tex       # one per task
└── hyperparameters.csv + .tex + .html + .png
```

### Step 6: Generate Validation Predictions

Run inference on validation samples using a trained model. The same script works for any model version.

```bash
# Submit SLURM array job (version-agnostic script)
# Adjust array size to match number of validation samples
sbatch --array=1-702 scripts/validation/run_sample_inference.sbatch
```

The script reads configuration from `configs/validation_inference_config.yaml` by default, or accepts a custom config as argument:

```bash
sbatch --array=1-500 scripts/validation/run_sample_inference.sbatch configs/my_custom_config.yaml
```

**Expected outputs:**
```
results/validation_predictions_<version>/
├── {accession}/
│   ├── {accession}_predictions.json   # Task predictions and confidence
│   └── {accession}_unitig_abundance.txt  # K-mer abundance vector
└── ...
```

> **Note:** Only labeled samples in the validation metadata are used for performance evaluation. Unlabeled samples can be used for exploratory predictions.

### Step 7: Confidence Calibration Analysis

Evaluates whether predicted confidence scores are reliable (correct predictions should have higher confidence than incorrect ones). Generates reliability diagrams, ECE/MCE metrics, and precision-recall flagging curves.

```bash
mamba run -p ./env python scripts/paper/32_confidence_calibration_analysis.py \
  --predictions results/test_evaluation_bioproject_v5/test_predictions.tsv \
  --val-pred-dir results/validation_predictions_bioproject_v5_full \
  --label-encoders results/training_bioproject_v5/label_encoders.json
```

**Expected outputs:**
```
paper/figures/final/sup_calibration_A_confidence_distributions.png/.html
paper/figures/final/sup_calibration_B_precision_recall_flagging.png/.html
paper/figures/final/sup_calibration_C_reliability_diagrams.png/.html
results/calibration_analysis/calibration_metrics.json
```

### Step 8: Evaluate Model on Training Set

Required before generating the main performance table (Table 1). Runs a single forward pass of the trained model on all 2,515 training samples.

```bash
mamba run -p ./env python scripts/evaluation/06_compute_training_metrics.py
```

```
Output: results/training_bioproject_v5/training_set_metrics.json
```

### Step 9: Generate All Paper Materials

Regenerates all publication figures and tables from the results above. Requires Steps 4–8 to be complete.

```bash
bash scripts/paper/generate_all_paper_materials.sh
```

> **Note:** This script does **not** run calibration (Step 7) or generalisation gap plots. Run those separately:
> ```bash
> # Calibration supplementary figures
> mamba run -p ./env python scripts/paper/32_confidence_calibration_analysis.py \
>   --predictions results/test_evaluation_bioproject_v5/test_predictions.tsv \
>   --val-pred-dir results/validation_predictions_bioproject_v5_full \
>   --label-encoders results/training_bioproject_v5/label_encoders.json
>
> # Generalisation gap slopegraphs (requires baseline comparison to be done first)
> mamba run -p ./env python scripts/paper/34_generate_generalisation_gap_plots.py
> ```

---

## Baseline Comparison

Compares DIANA against classical ML baselines (MajorityClass, LogisticRegression, LinearSVM, RidgeClassifier, RandomForest) on the held-out test set and the labeled validation set. For each model × task combination, Balanced Accuracy and macro F1 are reported with **95% bootstrap confidence intervals** (1,000 resamples). Results feed directly into the comparison figures.

The baseline comparison script is **version-agnostic** — all paths, tasks, and hyperparameters are specified in a YAML config file.

### Running Baseline Comparison

```bash
# v7 example (6 tasks, 78K features):
sbatch scripts/evaluation/run_test_baseline_comparison_v7.sbatch

# Custom config:
python scripts/evaluation/08_test_set_baseline_comparison.py \
  --config configs/baseline_config_custom.yaml
```

The script accepts command-line overrides:
```bash
# Override tasks only
python scripts/evaluation/08_test_set_baseline_comparison.py \
  --config configs/baseline_config_v7.yaml \
  --tasks community_type,sample_host,material

# Override output directory
python scripts/evaluation/08_test_set_baseline_comparison.py \
  --config configs/baseline_config_v7.yaml \
  --output-dir results/baseline_comparison_custom
```

**Expected outputs:**
```
results/baseline_comparison_<version>/
├── metrics.json              # All model metrics with 95% bootstrap CIs
├── summary.csv               # Human-readable performance table
├── summary.tex               # LaTeX table for paper
└── baseline_comparison.log   # Full execution log
```

> **After retraining DIANA** (baselines unchanged), update only the DIANA rows in `metrics.json` without redoing baseline training (~5 sec):
> ```bash
> mamba run -p ./env python scripts/evaluation/patch_diana_in_metrics.py
> ```

Then regenerate the comparison figures:

```bash
# Supplementary Figure 6: DIANA vs baselines bar chart
mamba run -p ./env python scripts/paper/21_generate_baseline_comparison.py

# Supplementary Figure 7: Generalisation gap (Test → Validation slopegraphs with 95% CIs)
mamba run -p ./env python scripts/paper/34_generate_generalisation_gap_plots.py
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
- **Ancient samples** from AncientMetagenomeDir (360 samples after removing train/test overlaps)
  - Completely independent of training/test BioProjects
  - Curated ancient metagenomes for robust validation

**Key principle:** Test model generalization to completely unseen samples, covering both ancient (87%) and modern (13%) metagenomes
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

**Output:** `data/validation/accessions.txt` (360 ancient samples after overlap removal)

#### 2. Download All Samples

> **Note:** Scripts will **skip existing files** automatically. They check for existing SRA files in `data/validation/sra/`. The `accessions.txt` file is updated automatically when merging reviewed samples and now contains all 360 unique run accessions from `validation_metadata.tsv` (ancient samples only, train/test overlaps removed).

```bash
# Will only download newly added samples (skips existing)
bash scripts/validation/03_prefetch_all.sh

# Convert SRA → FASTQ (auto-skips existing); update array size to match total accessions
sbatch --array=1-360%20 scripts/validation/04_convert_sra_to_fastq.sbatch
```

**Output:** `data/validation/sra/{accession}/*.sra` and `data/validation/raw/{accession}/*.fastq.gz`

**To check download progress:**
```bash
# Count downloaded SRA files
find data/validation/sra -name "*.sra" | wc -l
# Expect ~360 unique run accessions (ancient samples only)
```

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

**After all predictions complete, generate validation metrics and figures:**
```bash
# Load validation predictions with metadata (shared utility)
# This creates a standardized DataFrame used by all paper scripts
mamba run -p ./env python scripts/validation/load_validation_data.py

# Output: DataFrame with columns - sample_id, task, true_label, pred_label, 
#         confidence, is_correct, is_seen

# Generate all publication-ready figures and tables
bash scripts/paper/generate_all_paper_materials.sh

# Legacy comparison script (still functional):
mamba run -p ./env python scripts/validation/06_compare_predictions.py
```

---

**Last Updated:** June 2026

