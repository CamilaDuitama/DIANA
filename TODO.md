# Project TODOs

- [x] 1. Prepare the script for splitting the data into train and test set. (Ref: `/pasteur/appa/scratch/cduitama/decOM/scripts/create_train_test_split.py`)
- [x] 2. Create the basic plots for the input data, the train and test set. (Ref: `/pasteur/appa/scratch/cduitama/decOM/scripts/preparation_presentation.py`)
- [x] 3. Prepare the multi-task learning classifier.
- [x] 5. Filter existing matrix (`/pasteur/appa/scratch/cduitama/decOM/data/unitigs/matrices/large_matrix_3116/`) for testing scripts.
- [x] 6. Build pipeline to extract new features for inference from a new fasta file or set of fasta files.
- [x] 7. Revise results from data/matrices/large_matrix_3070_with_frac/ and compress with zstd the larges subdirectory there
- [x] 8. Create plots for the unitigs that will be used as input to the model (unitigs inside unitigs.frac.mat, use their ids to extract them from unitigs.fa), for their sekqit stats and for their sparsity or distribution across samples. Results should go inside paper/figures and paper/tables.
- [x] 9. Start training the multi-task learning classifier on the unitig matrix from data/matrices/large_matrix_3070_with_frac/
- [x] 10. Create the script to download the validation fasta files. Start downloading them.
- [ ] 11. Add read count/length statistics to paper/metadata/validation_metadata.tsv using seqkit stats (need to create/identify script)
- [x] 12. Re-run validation predictions with resource tracking to estimate runtime and CPU/memory efficiency vs input data size (performance data lost from deleted logs)
- [ ] 13. After validation predictions complete, track results in local-results branch: results/validation_predictions/, logs/validation/diana_predict_*, data/validation/sra/ (do NOT push to remote - large files)
- [x] 14. Test the multi-task classifier
- [x] 15. Interpret the model, find what are the most discriminant features.
- [x] 16. Deploy diana-predict.
- [x] 17. Create the following plots inside the paper folder:Multi-Task Performance Summary Bar Chart (test set), Confusion Matrix for material Task (test set), Confusion Matrix for sample_host Task (test set), Per-Class Metrics Bar Chart for material Task (test set), Training & Validation Loss Curves (training set & validation set), ROC Curves for sample_type Task (test set), Precision-Recall Curves for sample_type Task (test set), ROC Curves for community_type Task (test set), Precision-Recall Curves for community_type Task (test set), Final Performance Metrics Summary Table (test set), Per-Class Metrics Table for material Task (test set), Model Hyperparameters Table (N/A)

## Temperature Scaling for Model Calibration (Target: master branch)

**Problem**: Model shows high confidence scores for incorrect predictions, indicating poor calibration.

**Solution**: Implement post-hoc temperature scaling to align confidence scores with actual accuracy.

### Phase 1: Learn the Calibration Parameters

- [ ] **18. Create calibration script** (`scripts/training/04_calibrate_model.py`)
  - Load full training data and split into 90% train / 10% internal validation (same split used for final model training)
  - Load trained model (`models/checkpoints/best_model.pth`)
  - Extract raw logits (pre-softmax outputs) for validation set
  - For each of the four classification tasks separately:
    - Find optimal temperature T that minimizes Expected Calibration Error (ECE)
    - Temperature scaling: `softmax(logits / T)` instead of `softmax(logits)`
  - Save optimal temperatures to `models/optimal_temperatures.json` (one per task)
  - Log calibration metrics (ECE before/after, reliability diagrams)

### Phase 2: Apply the Calibration

- [ ] **19. Update inference scripts to support temperature scaling**
  - Modify `src/diana/cli/predict.py` (diana-predict):
    - Add `--calibration-file` argument (default: `models/optimal_temperatures.json`)
    - Load temperature values from JSON file
    - Apply temperature scaling: divide logits by T before softmax
  - Modify `src/diana/cli/test.py` (diana-test):
    - Add `--calibration-file` argument
    - Apply same temperature scaling logic
  - Ensure backward compatibility (optional calibration file)

### Phase 3: Document and Verify

- [ ] **20. Update documentation**
  - Add "Step 3.5: Calibrate Model" to `REPRODUCIBILITY.md`:
    - Command: `python scripts/training/04_calibrate_model.py`
    - Input: trained model, training data splits
    - Output: `models/optimal_temperatures.json`
  - Update "Step 4: Testing" commands to include `--calibration-file models/optimal_temperatures.json`
  - Update validation prediction commands to use calibration file
  
- [ ] **21. Verify calibration improves confidence scores**
  - Re-run test evaluation with calibrated model
  - Re-run validation predictions with calibration
  - Regenerate confidence distribution plots
  - Verify: confidence scores for incorrect predictions are now lower
  - Compare ECE metrics before/after calibration
  - Add calibration results to paper (reliability diagrams, ECE comparison)
