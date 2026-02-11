# DIANA: Deep Learning Identification and Assessment of Ancient DNA

**Multi-task classification of ancient DNA samples using unitigs**

DIANA uses unitig sequence features from raw FASTQ files to compare new samples against the whole plethora of existing ancient DNA samples in the SRA, simultaneously predicting four characteristics:
- **Sample Type**: Ancient vs. modern metagenome
- **Community Type**: Oral, gut, skeletal tissue, plant tissue, soft tissue, or environmental sample  
- **Sample Host**: Homo sapiens, Ursus arctos, and 10 other host species
- **Material**: Dental calculus, tooth, bone, sediment, and 9 other material types

The model is trained on 2,609 samples from the [AncientMetagenomeDir database](https://github.com/SPAAM-community/AncientMetagenomeDir).

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [diana-predict](#diana-predict)
  - [diana-project](#diana-project)
- [FAQ](#faq)
- [License](#license)
- [Citation](#citation)

---

## Installation

### Prerequisites
- **Operating System**: Linux (tested on Ubuntu 20.04+)
- **Package Manager**: [Mamba](https://mamba.readthedocs.io/) or Conda

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

**What gets installed:**
- Python dependencies (PyTorch, scikit-learn, polars, etc.)
- External tools: `back_to_sequences` and `MUSET`
- Reference k-mers file (~179MB, downloaded from Zenodo)
- Pre-trained model checkpoint

---

## Quick Start

Test the installation with a small sample:

```bash
# Download test sample (ancient oral metagenome, ~10MB)
mkdir -p test_data && cd test_data
wget https://ftp.sra.ebi.ac.uk/vol1/fastq/ERR360/004/ERR3609654/ERR3609654_{1,2}.fastq.gz
cd ..

# Run prediction
mamba run -p ./env diana-predict \
  --sample test_data/ERR3609654_1.fastq.gz test_data/ERR3609654_2.fastq.gz \
  --model results/training/best_model.pth \
  --muset-matrix data/matrices/large_matrix_3070_with_frac \
  --output test_results

# Visualize similarity to training data
mamba run -p ./env diana-project --sample test_results/ERR3609654/

# View predictions
cat test_results/ERR3609654/ERR3609654_predictions.json
```

Expected: Predictions for sample_type (Ancient), community_type (oral), sample_host (Homo sapiens), and material (dental calculus).

**Example outputs:**

*Prediction probabilities:*
<p align="center">
  <img src="test_results/ERR3609654/plots/ERR3609654_sample_type_barplot.png" width="45%"/>
  <img src="test_results/ERR3609654/plots/ERR3609654_material_barplot.png" width="45%"/>
</p>

*PCA projection showing sample similarity to training data:*
<p align="center">
  <img src="results/pca_projection/ERR3609654/pca_projection_sample_type.png" width="45%"/>
  <img src="results/pca_projection/ERR3609654/species_abundance.png" width="45%"/>
</p>

---

## Usage

### diana-predict

Predict sample characteristics from FASTQ files. The tool extracts unitig features and runs the neural network classifier.

**Basic usage:**

```bash
# Single-end reads
mamba run -p ./env diana-predict \
  --sample sample.fastq.gz \
  --model results/training/best_model.pth \
  --muset-matrix data/matrices/large_matrix_3070_with_frac \
  --output results/predictions

# Paired-end reads
mamba run -p ./env diana-predict \
  --sample sample_R1.fastq.gz sample_R2.fastq.gz \
  --model results/training/best_model.pth \
  --muset-matrix data/matrices/large_matrix_3070_with_frac \
  --output results/predictions
```

**Arguments:**
- `--sample`: FASTQ file(s). For paired-end, provide both files separated by space
- `--model`: Trained model checkpoint (`.pth` file)
- `--muset-matrix`: Reference matrix directory (requires `unitigs.fa` and `reference_kmers.fasta` for feature extraction)
- `--output`: Output directory
- `--threads`: Number of threads (default: 4)

**Note:** The model predicts from unitig features. The `--muset-matrix` directory contains reference data needed to transform your FASTQ into the same feature space the model was trained on.

**Outputs:**
```
results/predictions/sample_id/
├── sample_id_predictions.json         # Predictions and probabilities (JSON)
├── sample_id_predictions_plot.html    # Interactive visualization
├── sample_id_predictions_plot.png     # Static plot for publication
├── sample_id_kmer_counts.txt         # K-mer counts from sample
├── sample_id_unitig_abundance.txt    # Unitig abundance
└── sample_id_unitig_fraction.txt     # Unitig fractions (model features)
```

**Prediction plots:**

Generates 4 bar plots showing predicted probabilities for each task (sample_type, community_type, sample_host, material).

---

### diana-project

Visualize where your sample sits relative to 3,070 training samples in PCA space.

**Usage:**

```bash
diana-project --sample results/predictions/sample_id/
```

**Outputs:** 6 plots (HTML + PNG) showing sample position in PCA space colored by task labels, unitig species composition, and species abundance bar chart.

---

## FAQ

### Out-of-Memory (OOM) Errors

**Q: My job failed with "OUT_OF_MEMORY". What should I do?**

OOM failures occur during k-mer indexing when the sample has high microbial diversity.

**Solutions:**
1. **Check actual memory usage:**
   ```bash
   seff <job_id>  # On SLURM systems
   ```
2. **If >95% memory used:** Retry with 2× RAM
3. **High-diversity samples:** Dental calculus/oral samples often need >256 GB

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Citation

If you use DIANA in your research, please cite:

```bibtex
@article{diana2025,
  title={DIANA: Deep Learning Identification and Assessment of Ancient DNA
},
  author={Duitama, Camila and [Authors]},
  journal={[Journal]},
  year={2025},
  doi={[DOI]}
}
```


