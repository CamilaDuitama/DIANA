# DIANA: Deep Learning Identification and Assessment of Ancient DNA

**Multi-task classification of ancient DNA samples using unitigs**

DIANA uses unitig sequence features from raw FASTQ/FASTA files to compare new samples against the whole plethora of existing ancient DNA samples in the SRA, simultaneously predicting four characteristics:
- **Sample Type**: Ancient vs. modern metagenome
- **Community Type**: Oral, gut, skeletal tissue, plant tissue, soft tissue, or environmental sample  
- **Sample Host**: Homo sapiens, Ursus arctos, and 10 other host species
- **Material**: Dental calculus, tooth, bone, sediment, and 9 other material types

The model is trained on 2,597 samples from the [AncientMetagenomeDir database](https://github.com/SPAAM-community/AncientMetagenomeDir).

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
# (also installs the DIANA package and registers diana-predict / diana-project)
mamba env create -f environment.yml -p ./env
mamba activate ./env

# Download the trained model and PCA reference (~382 MB from Hugging Face) and
# reference k-mers (~179 MB from Zenodo), and build external tools
bash install.sh
```

**What gets installed:**
- Python dependencies (PyTorch, scikit-learn, polars, huggingface_hub, etc.)
- External tools: `back_to_sequences` (compiled from source) and `MUSET` (via conda)
- Trained model checkpoint (`results/training/best_model.pth`, ~336 MB, from [Hugging Face](https://huggingface.co/cduitamag/DIANA))
- PCA reference (`models/pca_reference.pkl`, ~46 MB, from [Hugging Face](https://huggingface.co/cduitamag/DIANA)) — required by `diana-project`
- Reference k-mers file (`data/matrices/large_matrix_3070_with_frac/reference_kmers.fasta`, ~179 MB, from Zenodo)

The following files are included directly in the repository (no download needed):
- `data/matrices/large_matrix_3070_with_frac/unitigs.fa` — 107,480 reference unitig sequences
- `results/training/label_encoders.json` — class label mappings for predictions

---

## Quick Start

Test the installation with a small sample:

```bash
# Download test sample (ancient oral metagenome, ~10MB)
mkdir -p test_data
wget -P test_data \
  https://ftp.sra.ebi.ac.uk/vol1/fastq/ERR360/004/ERR3609654/ERR3609654_1.fastq.gz \
  https://ftp.sra.ebi.ac.uk/vol1/fastq/ERR360/004/ERR3609654/ERR3609654_2.fastq.gz

# Run prediction (activate the environment first)
mamba activate ./env
diana-predict \
  --sample test_data/ERR3609654_1.fastq.gz test_data/ERR3609654_2.fastq.gz \
  --model results/training/best_model.pth \
  --muset-matrix data/matrices/large_matrix_3070_with_frac \
  --output test_results

# Visualize similarity to training data
diana-project --sample test_results/ERR3609654/

# View predictions
cat test_results/ERR3609654/ERR3609654_predictions.json
```

Expected: Predictions for sample_type (Ancient), community_type (oral), sample_host (Homo sapiens), and material (dental calculus).

`diana-predict` produces one interactive bar chart per task under `test_results/ERR3609654/plots/` (HTML + PNG).
`diana-project` produces PCA plots under `results/pca_projection/ERR3609654/`.

---

## Usage

### diana-predict

Predict sample characteristics from FASTQ or FASTA files. The tool extracts unitig features and runs the neural network classifier.

**Basic usage:**

```bash
# Single-end reads
diana-predict \
  --sample sample.fastq.gz \
  --model results/training/best_model.pth \
  --muset-matrix data/matrices/large_matrix_3070_with_frac \
  --output results/predictions

# Paired-end reads
diana-predict \
  --sample sample_R1.fastq.gz sample_R2.fastq.gz \
  --model results/training/best_model.pth \
  --muset-matrix data/matrices/large_matrix_3070_with_frac \
  --output results/predictions
```

**Arguments:**
- `--sample`: Gzipped FASTQ or FASTA file(s) (`*.fastq.gz`, `*.fq.gz`, `*.fasta.gz`, `*.fa.gz`, `*.fna.gz`). For paired-end, provide both files separated by a space.
- `--model`: Trained model checkpoint (`.pth` file)
- `--muset-matrix`: Reference matrix directory (must contain `unitigs.fa` and `reference_kmers.fasta`)
- `--output`: Output directory
- `--threads`: Number of threads (default: 10)

**Note:** The model predicts from unitig features. The `--muset-matrix` directory contains reference data needed to transform your reads into the same feature space the model was trained on.

**Outputs:**
```
results/predictions/sample_id/
├── sample_id_predictions.json              # Predictions and probabilities (JSON)
├── sample_id_kmer_counts.txt               # Per-reference-kmer counts
├── sample_id_unitig_abundance.txt          # Raw unitig-level counts
├── sample_id_unitig_fraction.txt           # Unitig fractions (model input features)
└── plots/
    ├── sample_id_sample_type_barplot.html  # Interactive probability bar chart
    ├── sample_id_sample_type_barplot.png   # Static version
    ├── sample_id_community_type_barplot.*
    ├── sample_id_sample_host_barplot.*
    └── sample_id_material_barplot.*
```

---

### diana-project

Visualize where your sample sits relative to 3,070 training samples in PCA space.

**Usage:**

```bash
diana-project --sample results/predictions/sample_id/
```

**Outputs:** Plots (HTML + PNG) showing sample position in PCA space colored by task labels, and a species abundance bar chart.

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

### `diana-predict: command not found`

The `diana-predict` and `diana-project` commands are registered by the DIANA package itself, which is installed as part of `mamba env create -f environment.yml -p ./env`. If the commands are missing, re-run that command with the environment activated, or run `pip install -e .` from the DIANA directory.

### HPC / cluster usage

If `mamba run` is unavailable or broken on your cluster, activate the environment and run commands directly:

```bash
mamba activate ./env   # or: source activate ./env
diana-predict --sample ...
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Citation

If you use DIANA in your research, please cite:

```bibtex
@article{diana2026,
  title={DIANA: Deep Learning Identification and Assessment of Ancient DNA},
  author={Duitama, Camila and others},
  journal={},
  year={2026},
  doi={}
}
```


