# TODO

## Coverage analysis (fraction feature depth-dependence)

- [ ] Quantify whether k-mer fraction values are correlated with sequencing depth across training samples
  - Compute mean fraction per sample (row mean of `unitigs.frac.mat`) as a proxy for "detection rate"
  - Compute sequencing depth per sample from the raw FASTQ stats (e.g., `seqkit stats` output already in `validation_seqkit_stats.tsv` or similar)
  - Plot mean fraction vs. depth (log scale) — expect positive correlation if depth-dependence is real
  - Report Spearman correlation; if strong (r > 0.5), flag as a potential confound for the paper
  - Potential response: model is trained on same depth variability it will see at inference (coverage-aware training)
  - Script location: `scripts/analysis/coverage_fraction_correlation.py` (to be created)

## Reviewer experiment (cross-task consistency)

- [ ] Run mislabelling experiment: artificially mislabel N% of training samples and measure model degradation
  - Script ready: `scripts/calibration/04_cross_task_consistency.py` — flags biologically implausible
    prediction combinations (e.g. oral community_type + fish host)
  - Use this to respond to reviewers asking about robustness to label noise

## v7 model (future)

### Eriksen2025 / missing validation samples
- [ ] 133 samples (`LV7001887*` lanes + `calculus_lib1`) are **zero-shot validation samples** not on SRA/EBI.
  - Source: Eriksen2025 paper — Greenland *Rangifer tarandus* ribs (skeletal tissue, 200 BP) + 1 human calculus
  - Archive DOI: `10.17894/ucph.b159460b-6491-4951-b1f4-8282d627f229` (UCPH data repository)
  - **These are intentional zero-shot classes**: `rib` material and *Rangifer tarandus* host appear
    exclusively in these samples. Both are absent from train AND test. Do NOT remove from split metadata.
  - **Option A**: Contact Eriksen2025 authors to obtain raw FASTQs (preferred for complete validation)
  - **Option B**: Run validation on the 569 remaining samples; report zero-shot classes separately
  - Current status: excluded from inference (no FASTQs available)

### v7 reference k-mers (Zenodo upload)
- [ ] Upload to Zenodo before v7 model release:
  - `data/matrices/matrix_v7_3190/matrix.filtered.fasta.gz` — 15,654,596 k-mers (v7 reference_kmers)
  - `data/matrices/matrix_v7_3190/unitigs.fa` — 78,430 unitig sequences
  - `data/matrices/matrix_v7_3190/unitigs.sshash.dict` — SSHash dictionary
  - `data/matrices/matrix_v7_3190/unitigs.frac.mat` — unitig fraction matrix
  - Symlink in place: `reference_kmers.fasta → matrix.filtered.fasta` (for diana-predict compatibility)

### Intermediate data cleanup
- [x] Delete intermediate kmtricks matrices to reclaim ~29T (completed)
  ```bash
  rm -rf data/matrices/matrix_v7_3190/kmer_matrix/matrices/
  ```

### Validation inference (once FASTQs ready)
- [ ] 402 SRA samples: conversion running (job 52707920, seqbio partition)
- [ ] 3 EBI samples downloading in background (ERR10493279, ERR2269774, ERR7919461)
- [ ] ERR4593927: prefetched via prefetch-orig, needs fasterq-dump conversion
- [ ] After v7 model trained: create `scripts/validation/run_inference_bioproject_v7.sbatch`
  - `--muset-matrix data/matrices/matrix_v7_3190` (reference_kmers.fasta already symlinked)
  - `--model results/training_bioproject_v7/best_model.pth`
  - Skip LV* samples until FASTQs available

### Training
- [ ] Create `configs/train_config_bioproject_v7.yaml` with 6 tasks:
  - `community_type`, `sample_host`, `material`, `sample_age`, `latitude`, `longitude`
  - Note: `sample_type` is trivial in v7 (all samples are `ancient_metagenome`) — drop it or keep as sanity check
  - New model goes to `results/training_bioproject_v7/`
- [ ] Create `scripts/training/run_final_train_edid_v7.sbatch`
  - Update paths: `data/splits_v7/`, `data/matrices/matrix_v7_3190/unitigs.frac.mat`

### Baselines
- [ ] Resubmit 7-task baseline once 4-task run confirmed: job 52667753 done, results in `results/baseline_comparison_v7/`
  - Submit: `sbatch scripts/evaluation/run_test_baseline_comparison_v7.sbatch` (already has `--tasks` flag)

### Paper figures
- [ ] Generate figures/tables for the 3 new v7 tasks: `sample_age`, `latitude`, `longitude`
  - Label distribution figures (see `scripts/analysis/legacy/analyze_v7_label_distribution.py`)
  - Per-class performance tables (analogous to `scripts/paper/04_generate_perclass_performance_table.py`)
  - Update `scripts/paper/generate_all_paper_materials.sh` to include v7 scripts
