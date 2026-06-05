# Legacy Training Scripts

**These scripts are superseded by the v3 training workflow and retained for reference only.**

---

## Why These Scripts Are Deprecated

The current DIANA training workflow (v3) uses a corrected nested cross-validation implementation with per-task label smoothing. These legacy scripts contain bugs or use outdated approaches:

1. **Inner CV bug** - Trials split across folds instead of evaluated on all folds
2. **No batch training** - Used full-batch gradient descent
3. **Test set leakage** - Used test set for early stopping
4. **Incorrect stratification** - Only stratified by sample_type
5. **Broken pruning** - MedianPruner fired on incomplete fold evaluations

---

## Legacy Scripts

### Hyperparameter Optimization (Buggy Versions)

- **`run_hyperopt_bioproject_v2.sbatch`**
  - v2 hyperparameter optimization (had inner CV bug)
  - **Replaced by:** `run_hyperopt_bioproject_v3.sbatch`

- **`run_hyperopt_edid.sbatch`**
  - Non-BioProject split version
  - **Replaced by:** `run_hyperopt_bioproject_v3.sbatch`

### Final Training (Outdated Versions)

- **`run_final_train_ls.sbatch`**
  - Old label smoothing-specific training
  - **Replaced by:** `run_final_train_edid.sbatch` (supports per-task label smoothing)

- **`run_final_training_gpu.sbatch`**
  - Duplicate/old training script
  - **Replaced by:** `run_final_train_edid.sbatch`

### Old Workflow Scripts

- **`run_multitask_gpu.sbatch`**
  - Old training script from pre-v3 workflow
  - **Replaced by:** `01_train_multitask_single_fold.py` + v3 sbatch

- **`run_test_eval_edid.sbatch`**
  - Test set evaluation script (may still be useful for batch evaluations)
  - Not currently used in README workflow

- **`submit_multitask.sh`**
  - Launcher for `run_multitask_gpu.sbatch`
  - **Replaced by:** Direct sbatch submission in README

---

## Current v3 Workflow

**Active scripts in `scripts/training/`:**

1. `01_train_multitask_single_fold.py` - Single-fold hyperopt with corrected nested CV
2. `02_train_final_model.py` - Final model training with aggregated hyperparams
3. `aggregate_cv_results.py` - Aggregates results across 5 CV folds
4. `run_hyperopt_bioproject_v3.sbatch` - SLURM submission for v3 hyperopt
5. `run_final_train_edid.sbatch` - SLURM submission for final training

**See README.md for complete v3 workflow.**

---

**Last Updated:** June 2026
