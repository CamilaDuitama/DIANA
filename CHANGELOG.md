# Changelog

All notable changes to DIANA are documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0] — 2026-03-20

First public release of DIANA (**D**eep-learning **I**dentification and **A**ssessment of a**N**cient DN**A**).

### Added

- `diana-predict` CLI entry point — runs the full FASTQ → k-mer counting →
  unitig aggregation → multi-task inference pipeline on one or more samples.
- `diana-project` CLI entry point — project a sample's unitig abundance vector
  onto the reference PCA space.
- Multi-task MLP classifier predicting four labels simultaneously:
  `sample_type`, `community_type`, `sample_host`, `material`.
- Structured JSON output per sample with per-task top predictions and
  confidence scores for all classes.
- Optional bar-chart visualisations per task (`--plot` flag).
- `install.sh` — downloads the trained model (~336 MB) and PCA reference
  (~46 MB) from Hugging Face Hub (`cduitamag/DIANA`) and the reference k-mer
  index (~179 MB compressed) from Zenodo
  ([10.5281/zenodo.18157419](https://doi.org/10.5281/zenodo.18157419)).
  Builds `back_to_sequences` (Rust) from source.
- `training_matrix/unitigs.fa` (18 MB) — reference unitigs bundled in the
  repository; no separate download required.
- `tests/test_integration.sh` — end-to-end smoke test on bundled test FASTQ
  data (ERR3609654, paired-end); validates all four task predictions.
- `environment.yml` — reproducible conda/mamba environment specification
  (Python 3.10, PyTorch, kmat\_tools v0.5.1, muset, Rust).

### Dependencies

- Python ≥ 3.10
- PyTorch ≥ 2.0
- `kmat_tools` v0.5.1 (via `conda install -c camiladuitama muset`)
- `back_to_sequences` (built from source during `install.sh`)
- `muset` (k-mer matrix construction)

[0.1.0]: https://github.com/CamilaDuitama/DIANA/releases/tag/v0.1.0
