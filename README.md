# DIANA: Deep Integration of Ancient DNA

**Multi-task classification of ancient DNA samples using unitig k-mer features**

DIANA is a deep learning tool that predicts four key characteristics of ancient DNA samples:
- **Sample Type**: ancient vs. modern metagenome
- **Community Type**: oral, gut, skeletal tissue, plant tissue, soft tissue, or environmental sample
- **Sample Host**: Homo sapiens, Ursus arctos, environmental, and 9 other host species
- **Material**: dental calculus, tooth, bone, sediment, and 9 other material types

---

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Predict on New Samples](#predict-on-new-samples)
  - [Training](#training)
- [FAQ](#faq)
  - [Memory Requirements](#memory-requirements)
  - [Out-of-Memory (OOM) Errors](#out-of-memory-oom-errors)
  - [Measuring K-mer Complexity](#measuring-k-mer-complexity)
- [License](#license)

---

## Overview

DIANA uses unitig k-mer features extracted from raw sequencing data to classify ancient DNA samples across multiple taxonomic and contextual dimensions. The model is trained on 3,070 samples from the AncientMetagenomeDir database, making it the largest ancient DNA classification model to date.

**Key Features:**
- Multi-task learning architecture for joint prediction of 4 sample characteristics
- Unitig-based feature representation using muset for efficient k-mer matrix generation
- Pre-trained model available for immediate inference
- Fast predictions: ~30-40 seconds per sample

---

## Installation

### Prerequisites
- Linux operating system
- [Mamba](https://mamba.readthedocs.io/) or Conda package manager
- 12GB RAM minimum
- 6 CPU cores recommended

### Setup

```bash
# Clone the repository
git clone https://github.com/CamilaDuitama/DIANA.git
cd DIANA

# Create and activate the environment
mamba env create -f environment.yml -p ./env
mamba activate ./env

# Install dependencies and download reference k-mers (179MB)
bash install.sh
```

The installation script will:
1. Build the muset tool for k-mer matrix generation
2. Download reference k-mers from Zenodo (749MB uncompressed)
3. Verify installation integrity

---

## Quick Start

Predict sample characteristics for a new FASTQ file:

```bash
# Example: Predict on a single-end or paired-end sample
mamba run -p ./env diana-predict \
  --sample path/to/sample.fastq.gz \
  --model results/full_training/best_model.pth \
  --muset-matrix data/matrices/large_matrix_3070_with_frac \
  --output results/my_predictions \
  --threads 6

# For paired-end samples, provide both files:
mamba run -p ./env diana-predict \
  --sample path/to/sample_R1.fastq.gz path/to/sample_R2.fastq.gz \
  --model results/full_training/best_model.pth \
  --muset-matrix data/matrices/large_matrix_3070_with_frac \
  --output results/my_predictions \
  --threads 6
```

**Output:**
```
results/my_predictions/sample/
├── predictions.json           # Predictions and probabilities for all 4 tasks
├── predictions_plot.html      # Interactive visualization
├── predictions_plot.png       # Static plot
└── muset_output/              # Intermediate k-mer matrix
```

---

## Usage

### Predict on New Samples

```bash
diana-predict \
  --sample <FASTQ_FILE(S)> \
  --model <MODEL_PATH> \
  --muset-matrix <MATRIX_DIR> \
  --output <OUTPUT_DIR> \
  --threads <NUM_THREADS> \
  [--verbose]
```

**Required Arguments:**
- `--sample`: Path to FASTQ file(s). For paired-end, provide both files separated by space
- `--model`: Path to trained model (`.pth` file)
- `--muset-matrix`: Directory containing reference k-mers and unitig matrix
- `--output`: Output directory for predictions
- `--threads`: Number of CPU threads to use

**Optional Arguments:**
- `--verbose`: Enable detailed logging

**Example with pre-trained model:**
```bash
# Download the pre-trained model (if not already available)
# Model location: results/full_training/best_model.pth

# Run prediction
mamba run -p ./env diana-predict \
  --sample data/raw/ERR3003613/ERR3003613.fastq.gz \
  --model results/full_training/best_model.pth \
  --muset-matrix data/matrices/large_matrix_3070_with_frac \
  --output results/test_prediction \
  --threads 6 \
  --verbose
```

### Training

To train DIANA on your own dataset or reproduce the published results, see [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the complete training pipeline.

**Quick training overview:**
```bash
# Step 1: Hyperparameter optimization (5-fold CV)
mamba run -p ./env diana-train multitask \
  --config configs/train_config.yaml \
  --output results/training \
  --mode optimize

# Step 2: Aggregate results
mamba run -p ./env diana-train multitask \
  --config configs/train_config.yaml \
  --output results/training \
  --mode aggregate

# Step 3: Train final model
mamba run -p ./env diana-train multitask \
  --config configs/train_config.yaml \
  --output results/full_training \
  --mode train
```

For detailed training instructions, see [docs/TRAINING.md](docs/TRAINING.md).

---

## FAQ

### Memory Requirements

**Q: How much RAM do I need?**

Memory usage depends on k-mer complexity during the `back_to_sequences` indexing step, not just input file size.

**Typical requirements:**
- **Ancient DNA (low diversity):** 64-128 GB
- **Modern/environmental samples:** 128-256 GB  
- **High-diversity oral metagenomes:** >128 GB (3% of samples may need >128 GB)

**Important:** A small file (260 MB) may require >128 GB if it has high k-mer diversity, while a large file (7 GB) may need only 64 GB if it has lower diversity.

### Out-of-Memory (OOM) Errors

**Q: My job failed with "OUT_OF_MEMORY" or "oom_kill" error. What should I do?**

OOM failures occur during **Step 1 (k-mer counting)** when `back_to_sequences` (from SSHash) exhausts available RAM while building the in-memory k-mer index.

**Diagnosis:**
1. Check memory usage:
   ```bash
   seff <job_id>  # or reportseff <job_id> on SLURM systems
   ```
2. If memory efficiency is >95%, the job truly needs more RAM

**Solutions:**
- **Retry with 2× RAM:** If job used 64 GB at 100%, retry with 128 GB
- **High-diversity samples:** Dental calculus and oral metagenomes often need >256 GB
- **Check k-mer complexity:** Samples with high microbial diversity require more memory (see below)

**Technical details:**
- Failure point: Step 1 - Counting k-mers in sample
- Tool: `back_to_sequences` (SSHash library)
- Bottleneck: In-memory k-mer deduplication during indexing
- Memory = f(reference k-mers + sample k-mer diversity)

### Measuring K-mer Complexity

**Q: How can I estimate if my sample will need high memory before running?**

**Suggested approaches to measure k-mer complexity:**

1. **Pre-screen with k-mer counting tools:**
   - Use lightweight tools like `jellyfish`, `KMC`, or `ntCard` to count unique k-mers
   - Higher unique k-mer count → higher memory needs
   - Example:
     ```bash
     jellyfish count -m 31 -s 100M -t 4 sample.fastq.gz
     jellyfish stats mer_counts.jf  # Check "Unique" count
     ```

2. **Estimate from sample metadata:**
   - **Oral/dental calculus samples:** Typically highest diversity → >128 GB
   - **Gut samples:** Moderate diversity → 64-128 GB
   - **Ancient bone/tooth:** Lower diversity → 64 GB usually sufficient
   - **Modern environmental:** High diversity → 128-256 GB

3. **Quick test run:**
   - Monitor memory usage during Step 1 with `top` or `htop`
   - If memory climbs rapidly toward limit, kill and restart with more RAM

4. **K-mer entropy/diversity metrics:**
   - Calculate Shannon entropy of k-mer distribution
   - Use tools like `GenomeScope` or `smudgeplot` to profile k-mer spectra
   - Higher complexity → steeper memory requirements

**Validation data insights:**
- 97% of ancient metagenomes succeeded with ≤128 GB
- Memory expansion ranges from 10× to 1513× input file size (median: 37×)
- File size is **not** a reliable predictor of memory needs

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
