# DIANA: Deep Learning Identification and Assessment of Ancient DNA

Multi-task classification of ancient DNA samples using unitig abundances as features. A **unitig** is a maximal non-branching path in a de Bruijn graph — it compacts overlapping k-mers into a single sequence, reducing redundancy while preserving genomic diversity.

<p align="center">
  <img src="images/unitigs.png" width="60%" alt="Unitig concept: k-mers form de Bruijn graph nodes; unitigs are maximal non-branching paths"/>
</p>

Given raw sequencing reads, DIANA counts how many reference unitigs are present in the sample and feeds the resulting abundance vector into a multi-task neural network that simultaneously predicts:

| Task | Labels |
|---|---|
| **Sample type** | Ancient / Modern |
| **Community type** | Oral, gut, skeletal tissue, soft tissue, environmental, … |
| **Sample host** | *Homo sapiens*, *Ursus arctos*, and 10 others |
| **Material** | Dental calculus, bone, sediment, and 9 others |

Trained on 2,597 samples from the [AncientMetagenomeDir](https://github.com/SPAAM-community/AncientMetagenomeDir) database.

<p align="center">
  <img src="images/pipeline.png" width="80%" alt="DIANA pipeline: raw reads → k-mer counting → unitig abundance vector → multi-task neural network → metadata predictions"/>
</p>

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [diana-predict](#diana-predict)
  - [diana-project *(optional)*](#diana-project-optional)
- [FAQ](#faq)
- [License](#license)
- [Citation](#citation)

---

## Installation

**Requirements:** Linux, [Mamba](https://mamba.readthedocs.io/) or Conda.

```bash
git clone --recurse-submodules https://github.com/CamilaDuitama/DIANA.git
cd DIANA
mamba env create -f environment.yml -p ./env
mamba activate ./env
bash install.sh
```

---

## Quick Start

A small bundled test sample is included in `test_data/` — a 1 % random subsample (seed 42, ~182 k read pairs, 9 MB each) of [ERR3609654](https://www.ebi.ac.uk/ena/browser/view/ERR3609654), an ancient oral metagenome. Use it to verify the installation without downloading the full 1.6 GB dataset.

```bash
diana-predict \
  --sample test_data/ERR3609654_1_small.fastq.gz test_data/ERR3609654_2_small.fastq.gz \
  --model results/training/best_model.pth \
  --muset-matrix data/matrices/large_matrix_3070_with_frac \
  --output test_results
```

View the predictions:

```bash
cat test_results/ERR3609654/ERR3609654_predictions.json
```

`diana-predict` writes results to `test_results/ERR3609654/`:
- `ERR3609654_predictions.json` — predicted class and probability for each task
- `plots/ERR3609654_*_barplot.{html,png}` — one interactive bar chart per task

Each bar chart shows every class on the y-axis and its predicted probability on the x-axis; the most probable class is highlighted. The `.html` version is fully interactive (hover for exact values). Below are the four charts produced for ERR3609654:

**Sample type** — Is the sample ancient or modern?
<p align="center"><img src="images/example_sample_type.png" width="70%" alt="Sample type bar chart: Ancient 79%, Modern 21%"/></p>

**Community type** — What microbial community does the sample come from?
<p align="center"><img src="images/example_community_type.png" width="70%" alt="Community type bar chart: oral community predicted with 63% confidence"/></p>

**Sample host** — Which host species does the sample originate from?
<p align="center"><img src="images/example_sample_host.png" width="70%" alt="Sample host bar chart: Homo sapiens predicted with 48% confidence"/></p>

**Material** — What physical material was the sample extracted from?
<p align="center"><img src="images/example_material.png" width="70%" alt="Material bar chart: dental calculus predicted with 25% confidence"/></p>

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

### diana-project *(optional)*

`diana-project` is an optional companion tool that projects a sample onto the training PCA space, finds its nearest neighbours among the 2,597 training samples, and saves interactive HTML + PNG scatter plots.

```bash
diana-project --sample results/predictions/sample_id/
```

For each prediction task it produces a `pca_projection_<task>.html/png`: training samples are coloured by label, the five nearest neighbours are highlighted in yellow, and the new sample is shown as a red star.

**PCA projection (sample type)** — ERR3609654 (red star) lands among ancient samples, consistent with the prediction. Its five nearest neighbours (yellow diamonds) are all ancient oral metagenomes.
<p align="center"><img src="images/example_pca_projection.png" width="80%" alt="PCA projection plot: ERR3609654 projects into the ancient oral cluster"/></p>
<p align="center"><em><a href="results/pca_projection/ERR3609654/pca_projection_sample_type.html">Open interactive version</a></em></p>

**Species abundance** — Top microbial species detected in the sample's unitigs, giving a quick taxonomic overview.
<p align="center"><img src="images/example_species_abundance.png" width="70%" alt="Species abundance bar chart for ERR3609654"/></p>
<p align="center"><em><a href="results/pca_projection/ERR3609654/species_abundance.html">Open interactive version</a></em></p>

---

## FAQ

**`diana-predict: command not found`** — Make sure the environment is activated (`mamba activate ./env`). The commands are registered when the environment is created.

**Out-of-memory errors** — OOM during k-mer counting is common for high-diversity samples (dental calculus, oral metagenomes). Retry with more RAM (`--mem=32G` on SLURM). Calculus samples can require >256 GB.

**HPC / broken `mamba run`** — Activate the environment first (`mamba activate ./env` or `source activate ./env`) and call `diana-predict` directly.

---

## License

Apache 2.0 — see [LICENSE](LICENSE).

---

## Citation

```bibtex
@article{diana2026,
  title   = {{DIANA}: Deep Learning Identification and Assessment of Ancient {DNA}},
  author  = {Duitama Gonz{\'{a}}lez, Camila and Lopopolo, Maria and Nishimura, Luca
             and Faure, Roland and Duchene, Sebastian},
  year    = {2026},
  note    = {Correspondence: cduitama@pasteur.fr}
}
```


