# DIANA Paper — Pending Tasks

## High Priority (before submission)

- [x] **Runtime/Memory figure**: Job 51296281 running — full pipeline re-run from FASTQs using v5 model, generating `.jobinfo` files. When done: re-run `submit_validation_with_retry_v5.sh` for OOM retries, then update `config.py` `predictions_dir` → `results/validation_predictions_bioproject_v5_full` and re-run `generate_all`.

- [x] **Calibration table (main_table_02)**: Will be regenerated once job 51296281 finishes (same `.jobinfo` fix as Runtime figure).

- [x] **`sup_08_calibration_*` numbering** — done.

- [x] **`sup_09_class_imbalance_overview` numbering** — done.

- [x] **sup_10 color scheme**: Keeping paper-enhanced version (task colors, `✓`/`★` markers, 2×2 grid). Docstring update pending.

- [x] **sup_table_02 removed**: Removed from `generate_all`, `supplementary.tex`, and `main_table_01` caption.

- [x] **PRJNA433935 & PRJNA706195 — amplicon contamination (paper note)**: Both confirmed as soil amplicon data (16S/ITS). Root cause: 80 Skin training samples from PRJEB5758 are also amplicon data — model learned amplicon k-mer signature = Skin. Add scope statement to paper; exclude PRJEB5758 in v6.

- [x] **PRJEB64128 material task**: `tooth` is in training vocabulary (238 samples). Model predicts `dental calculus` with ~99% confidence — true generalisation failure, not a labelling mismatch.

## Medium Priority

- [ ] **v5 calibration for anomaly detection**: Confidence thresholding at ≥0.90 works reasonably well (see calibration summary), but permafrost/saliva/tongue classes get ~80/52/69% confidence when wrong. Consider per-task confidence thresholds tuned on validation instead of a single global threshold.

- [ ] **BioProject offenders table**: PRJNA433935 and PRJNA706195 have no `project_name` in the table (NaN). Look up publication metadata and add manually.

- [x] **`sup_table_02` (unseen labels)**: Removed — 0 unseen labels in v5 validation, table was empty. Removed from `generate_all`, `supplementary.tex`, and the `\ref{tab:zero_support}` citation in `main_table_01`.

## DIANA v6 — Future Work

- [ ] **Download AncientMetagenomeDir and evaluate for v6 training**: Download the latest AncientMetagenomeDir release (host-associated + environmental ancient metagenomes only — no single genomes) from https://www.spaam-community.org/AncientMetagenomeDir/#/docs/using/download. Audit available label columns to identify candidate tasks beyond the current four (sample_type, community_type, sample_host, material). For each candidate task: count non-empty labels, assess class balance, and flag tasks with >20% missing labels as unsuitable. Exclude amplicon libraries (`LibrarySelection=PCR/AMPLICON`). Goal: define a cleaner, larger training set and possibly additional classification heads for v6.

## Low Priority / Nice-to-have

- [ ] **Upgrade Kaleido**: All figure scripts emit `DeprecationWarning` about Kaleido <1.0.0. Run `pip install 'kaleido>=1.0.0'` in the env to silence.

- [ ] **v5 full calibration analysis**: Run `32_confidence_calibration_analysis.py` results through a proper anomaly detection benchmark — compute AUROC of confidence score for "is this prediction correct?" on the validation set.
