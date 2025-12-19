# DIANA Project Structure

**DIANA**: DNA Identification for Ancient Authenticity

## Overview
Multi-task classification of ancient DNA samples using genomic sequence features from muset/kmtricks output (Logan's dataset).

## Key Features
- **Multi-task learning**: One model predicting 4 targets simultaneously
- **Alternative approach**: Separate models for each target
- **Large-scale genomic features**: Optimized for sparse sequence feature matrices
- **3,070 curated samples** from metadata_decOM_classify.tsv
- **Largest aDNA dataset**: Trained on Logan's comprehensive ancient DNA collection

## Prediction Targets
1. `sample_type` (2 classes): ancient vs modern
2. `community_type` (6 classes): oral, environmental, skeletal, gut, plant, soft tissue
3. `sample_host` (12 classes): Homo sapiens, Ursus arctos, environmental, etc.
4. `material` (13 classes): dental calculus, sediment, tooth, bone, etc.

## Directory Structure
```
diana/
├── configs/              # YAML configuration files
│   ├── data_config.yaml
│   ├── model_multitask.yaml
│   └── training_config.yaml
├── data/                 # Data directory (gitignored)
│   ├── metadata/         # Links to metadata_decOM_classify.tsv
│   ├── splits/           # Train/val/test sample IDs
│   └── raw/              # Optional raw data
├── models/               # Saved models (gitignored)
│   ├── multitask/        # Multi-task model checkpoints
│   ├── separate/         # Individual task models
│   └── unsupervised/     # Autoencoders
├── results/              # Experiment outputs (gitignored)
│   ├── experiments/      # Experiment logs and configs
│   ├── figures/          # Plots and visualizations
│   └── metrics/          # Performance metrics
├── scripts/              # Executable scripts
│   ├── 01_create_splits.py
│   ├── training/
│   ├── evaluation/
│   └── data_prep/
├── src/diana/            # Core library (importable)
│   ├── data/             # Data loading and preprocessing
│   ├── models/           # Model architectures
│   ├── training/         # Training loops and callbacks
│   ├── evaluation/       # Metrics and visualization
│   └── utils/            # Configuration and logging
├── tests/                # Unit tests
├── notebooks/            # Jupyter notebooks
└── logs/                 # Training logs

## Workflow
1. **Data preparation**: Create stratified splits (70/15/15)
2. **Optional**: Train sparse autoencoder for feature learning
3. **Training**: Train multi-task or separate models
4. **Evaluation**: Compute metrics, generate visualizations
5. **Prediction**: Classify new samples

## Setup
```bash
# Create environment
conda env create -f environment.yml
conda activate diana

# Install package in development mode
pip install -e .

# Run tests
pytest tests/
```

## Quick Start
```bash
# 1. Create splits
python scripts/01_create_splits.py

# 2. Train multi-task model
python scripts/03_train_multitask.py  # TODO: create this

# 3. Evaluate
python scripts/04_evaluate.py  # TODO: create this
```

## Integration with muset
- Muset processes genomic sequences and generates feature matrices (kmtricks output)
- Matrices stored in: /pasteur/appa/scratch/cduitama/decOM/data/unitigs/matrices/large_matrix_3116
- Sample IDs matched via kmtricks.fof file
- Features represent genomic k-mer content from Logan's aDNA dataset

## Next Steps
1. Implement matrix loading from kmtricks format
2. Create training scripts (02, 03)
3. Create evaluation script (04)
4. Add hyperparameter search script (05)
5. Develop notebooks for data exploration
6. Fill out README with detailed documentation
