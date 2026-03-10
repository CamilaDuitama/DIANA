# DIANA: Deep Learning Identification and Assessment of Ancient DNA

Multi-task classification of ancient DNA samples using unitig k-mer features. Given raw sequencing reads, DIANA simultaneously predicts:

| Task | Labels |
|---|---|
| **Sample type** | Ancient / Modern |
| **Community type** | Oral, gut, skeletal tissue, soft tissue, environmental, … |
| **Sample host** | *Homo sapiens*, *Ursus arctos*, and 10 others |
| **Material** | Dental calculus, bone, sediment, and 9 others |

Trained on 2,597 samples from the [AncientMetagenomeDir](https://github.com/SPAAM-community/AncientMetagenomeDir) database.

<p align="center">
  <img src="images/barplot_example.png" width="48%" alt="Prediction bar chart"/>
  <img src="images/pca_example.png" width="48%" alt="PCA projection"/>
</p>

---

## Installation

**Requirements:** Linux, [Mamba](https://mamba.readthedocs.io/) or Conda.

```bash
git clone https://github.com/CamilaDuitama/DIANA.git
cd DIANA
mamba env create -f environment.yml -p ./env
mamba activate ./env
bash install.sh
```

`install.sh` builds `back_to_sequences`, then downloads the trained model and PCA reference (~382 MB) from [Hugging Face](https://huggingface.co/cduitamag/DIANA) and the reference k-mers (~179 MB) from Zenodo.

---

## Quick Start

```bash
# Download a test sample (ancient oral metagenome, ~10 MB paired-end)
mkdir -p test_data
wget -P test_data \
  https://ftp.sra.ebi.ac.uk/vol1/fastq/ERR360/004/ERR3609654/ERR3609654_1.fastq.gz \
  https://ftp.sra.ebi.ac.uk/vol1/fastq/ERR360/004/ERR3609654/ERR3609654_2.fastq.gz

# Run prediction
diana-predict \
  --sample test_data/ERR3609654_1.fastq.gz test_data/ERR3609654_2.fastq.gz \
  --model results/training/best_model.pth \
  --muset-matrix data/matrices/large_matrix_3070_with_frac \
  --output test_results

# Project onto training PCA space
diana-project --sample test_results/ERR3609654/

# View predictions
cat test_results/ERR3609654/ERR3609654_predictions.json
```

Expected result: Ancient · oral · *Homo sapiens* · dental calculus.

Outputs are written to `test_results/ERR3609654/`:
- `ERR3609654_predictions.json` — predicted class and probability for each task
- `plots/ERR3609654_*_barplot.{html,png}` — one bar chart per task
- PCA plots → `results/pca_projection/ERR3609654/`

---

## Usage

### diana-predict

```bash
# Single-end
diana-predict --sample sample.fastq.gz \
  --model results/training/best_model.pth \
  --muset-matrix data/matrices/large_matrix_3070_with_frac \
  --output results/predictions

# Paired-end
diana-predict --sample sample_R1.fastq.gz sample_R2.fastq.gz \
  --model results/training/best_model.pth \
  --muset-matrix data/matrices/large_matrix_3070_with_frac \
  --output results/predictions
```

| Argument | Description |
|---|---|
| `--sample` | Gzipped FASTQ or FASTA (`*.fastq.gz`, `*.fq.gz`, `*.fasta.gz`, `*.fa.gz`, `*.fna.gz`). Provide two files for paired-end. |
| `--model` | Path to `best_model.pth` |
| `--muset-matrix` | Directory containing `unitigs.fa` and `reference_kmers.fasta` |
| `--output` | Output directory |
| `--threads` | Number of threads (default: 10) |

### diana-project

```bash
diana-project --sample results/predictions/sample_id/
```

Projects the sample onto the training PCA space and saves interactive HTML + PNG plots.

---

## FAQ

**`diana-predict: command not found`** — Make sure the environment is activated (`mamba activate ./env`). The commands are registered when the environment is created.

**Out-of-memory errors** — OOM during k-mer counting is common for high-diversity samples (dental calculus, oral metagenomes). Retry with more RAM (`--mem=32G` on SLURM). Calculus samples can require >256 GB.

**HPC / broken `mamba run`** — Activate the environment first (`mamba activate ./env` or `source activate ./env`) and call `diana-predict` directly.

---

## License

MIT — see [LICENSE](LICENSE).

---

## Citation

```bibtex
@article{diana2026,
  title={DIANA: Deep Learning Identification and Assessment of Ancient DNA},
  author={Duitama, Camila and others},
  year={2026}
}
```


