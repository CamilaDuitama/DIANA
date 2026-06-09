# DIANA Paper — Pending Tasks

## High Priority (before submission)

- [ ] **Runtime/Memory figure (Supplementary Figure 1)**: `sup_01_runtime_memory.png` is missing because `results/validation_predictions_bioproject_v5/` has no `.jobinfo` files. Fix: either re-run validation inference with jobinfo tracking enabled for v5, or point `16_generate_runtime_memory.py` at the v4 validation directory which does have `.jobinfo` files (since the model architecture hasn't changed the runtime profile meaningfully).

- [ ] **Calibration table (main_table_02)**: The computational resources table is empty because `.jobinfo` files are missing (same issue as above). Same fix applies.

- [ ] **Number `sup_calibration_*` files**: Currently named `sup_calibration_A/B/C_*.png`. Should be renamed to `sup_08_calibration_A/B/C_*.png` and scripts updated accordingly.

- [ ] **Number `sup_class_imbalance_overview.png`**: Should be renamed to `sup_09_class_imbalance_overview.png`.

- [ ] **sup_10 color scheme**: `33_plot_single_sample_confidence.py` currently uses the paper's Vivid palette with per-task colors. The actual `diana-predict` (`scripts/inference/04_plot_results.py`) uses a single uniform `skyblue` for all bars and no ground-truth annotation. Decide whether sup_10 should faithfully reproduce the diana-predict UI (skyblue, 4 separate subplots) or keep the paper-enhanced version (task colors, `✓`/`★` markers, 2×2 grid) — and update the docstring accordingly.

- [ ] **Missing sup_table_02**: `main_table_02_computational_resources.tex` is empty. Same `.jobinfo` fix as above.

- [ ] **PRJNA433935 & PRJNA706195 — amplicon contamination (paper note)**: Both projects are amplicon data (16S/ITS, `LibrarySelection=PCR`) correctly labelled as soil/env but predicted as Homo sapiens Skin with ~100% confidence. Root cause: the 80 Skin training samples from PRJEB5758 are also amplicon data (`Assay Type=AMPLICON`, PRJEB5758) — the model learned to associate the PCR amplicon k-mer signature with Skin. The validation failures are internally consistent but stem from training data contamination. **Action**: add an explicit scope statement in the paper (DIANA is designed for shotgun metagenomes; amplicon data is unsupported input). Optionally exclude PRJEB5758 from training in v6.

- [ ] **material task on validation**: PRJEB64128 (Jackson2024, n=73, 68% error) accounts for 32% of all material errors. Investigate whether this is a labeling convention mismatch (e.g. "tooth" vs "dental calculus") or a true generalization failure.

## Medium Priority

- [ ] **v5 calibration for anomaly detection**: Confidence thresholding at ≥0.90 works reasonably well (see calibration summary), but permafrost/saliva/tongue classes get ~80/52/69% confidence when wrong. Consider per-task confidence thresholds tuned on validation instead of a single global threshold.

- [ ] **BioProject offenders table**: PRJNA433935 and PRJNA706195 have no `project_name` in the table (NaN). Look up publication metadata and add manually.

- [ ] **`sup_table_02` numbering**: The unseen labels table currently has nothing (0 unseen labels in validation). Decide whether to keep as placeholder or remove from paper.

## DIANA v6 — Future Work

- [ ] **Download AncientMetagenomeDir and evaluate for v6 training**: Download the latest AncientMetagenomeDir release (host-associated + environmental ancient metagenomes only — no single genomes) from https://www.spaam-community.org/AncientMetagenomeDir/#/docs/using/download. Audit available label columns to identify candidate tasks beyond the current four (sample_type, community_type, sample_host, material). For each candidate task: count non-empty labels, assess class balance, and flag tasks with >20% missing labels as unsuitable. Exclude amplicon libraries (`LibrarySelection=PCR/AMPLICON`). Goal: define a cleaner, larger training set and possibly additional classification heads for v6.

## Low Priority / Nice-to-have

- [ ] **Upgrade Kaleido**: All figure scripts emit `DeprecationWarning` about Kaleido <1.0.0. Run `pip install 'kaleido>=1.0.0'` in the env to silence.

- [ ] **v5 full calibration analysis**: Run `32_confidence_calibration_analysis.py` results through a proper anomaly detection benchmark — compute AUROC of confidence score for "is this prediction correct?" on the validation set.
