# Project TODOs

- [x] 1. Prepare the script for splitting the data into train and test set. (Ref: `/pasteur/appa/scratch/cduitama/decOM/scripts/create_train_test_split.py`)
- [x] 2. Create the basic plots for the input data, the train and test set. (Ref: `/pasteur/appa/scratch/cduitama/decOM/scripts/preparation_presentation.py`)
- [x] 3. Prepare the multi-task learning classifier.
- [ ] 4. Prepare the separate models on each target.
- [x] 5. Filter existing matrix (`/pasteur/appa/scratch/cduitama/decOM/data/unitigs/matrices/large_matrix_3116/`) for testing scripts.
- [x] 6. Build pipeline to extract new features for inference from a new fasta file or set of fasta files.
- [x] 7. Revise results from data/matrices/large_matrix_3070_with_frac/ and compress with zstd the larges subdirectory there
- [x] 8. Create plots for the unitigs that will be used as input to the model (unitigs inside unitigs.frac.mat, use their ids to extract them from unitigs.fa), for their sekqit stats and for their sparsity or distribution across samples. Results should go inside paper/figures and paper/tables.
- [x] 9. Start training the multi-task learning classifier on the unitig matrix from data/matrices/large_matrix_3070_with_frac/
- [ ] 10. Create the script to download the validation fasta files. Start downloading them.
- [x] 11. Test the multi-task classifier
- [x] 12. Interpret the model, find what are the most discriminant features.
- [ ] 13. Deploy diana-predict.
- [x] 14. Create the following plots inside the paper folder:Multi-Task Performance Summary Bar Chart (test set), Confusion Matrix for material Task (test set), Confusion Matrix for sample_host Task (test set), Per-Class Metrics Bar Chart for material Task (test set), Training & Validation Loss Curves (training set & validation set), ROC Curves for sample_type Task (test set), Precision-Recall Curves for sample_type Task (test set), ROC Curves for community_type Task (test set), Precision-Recall Curves for community_type Task (test set), Final Performance Metrics Summary Table (test set), Per-Class Metrics Table for material Task (test set), Model Hyperparameters Table (N/A)
- [ ] 15. Add input file validation in diana-predict to detect insufficient data:
  - Check minimum sequence count (current: only checks if file size > 0 bytes)
  - Use `seqkit stats` to verify files have enough reads (e.g., >10,000 sequences)
  - Detect files with very short reads or no k-mer matches before running full pipeline
  - Currently files with ~4,000-5,000 sequences (e.g., SRR867035, SRR867040) fail late in pipeline with "Loaded 0 unitig fractions" error
  - Should fail early with clear message: "Insufficient sequences (4,088 < 10,000 minimum)"
- [ ] 16. Fix Skin/Oral material sample labeling inconsistency (requires re-running full pipeline):
  - ISSUE: Train/test modern Skin samples (80+11) incorrectly labeled as environmental
    * Current: sample_host="Not applicable - env sample", community_type="Not applicable - env sample"
    * Should be: sample_host="Homo sapiens", community_type="skin" (they are human skin microbiome studies)
  - Original decOM metadata shows Organism="human skin metagenome" confirming these are microbiome samples
  - Same issue likely affects Oral material samples
  - Validation Skin samples (20) currently have correct sample_host="Homo sapiens" but community_type="Not applicable - env sample" (inconsistent)
  - Decision needed: Fix train/test metadata to proper labels OR document as limitation
  - Fixing requires: update metadata → retrain models → rerun all validation → regenerate all tables/figures
