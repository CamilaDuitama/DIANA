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
