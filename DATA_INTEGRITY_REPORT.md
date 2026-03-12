# DIANA Data Integrity Report

**Date:** January 7, 2026  
**Status:** ✅ **VERIFIED - No data leakage detected**

---

## Summary

All critical data integrity checks passed. The model training, testing, and validation datasets are properly isolated with no overlap.

---

## 1. Data Splits - No Leakage ✅

### Training/Test Split
```
Training set:    2,609 samples
Test set:          461 samples
Overlap:             0 samples (0.0%)
```
**Result:** ✅ Complete separation between train and test sets

### Validation Independence
```
Validation set:    616 samples (from validation_metadata_expanded.tsv)
Train overlap:       0 samples (0.0%)
Test overlap:        0 samples (0.0%)
```
**Result:** ✅ Validation set is completely independent from training data

**Data Sources:**
- Training/Test: DIANA samples from `data/diana_samples.fof` (3,070 total)
- Validation: AncientMetagenomeDir v25.09.0 (independent external dataset)

---

## 2. Validation Results

### Prediction Coverage
```
Metadata samples:     616
Predictions made:     608 successful
Failed (OOM):          ~8 samples
Missing/skipped:        2 samples (ERR11413738, ERR6178929)
Success rate:        98.7%
```

### Results Location
- Predictions: `results/validation_predictions/{sample_id}/{sample_id}_predictions.json`
- Summary: `results/validation_predictions/validation_comparison.json`
- Figures: `paper/figures/validation/`

---

## 3. Hardcoded Paths - None Found ✅

**Search performed:**
```bash
grep -rn "/pasteur\|/home\|/data/" src/ scripts/ --include="*.py"
```

**Result:** ✅ No hardcoded absolute paths in code  
**Configuration:** All paths use relative paths or config files (`configs/*.yaml`)

---

## 4. TODO Comments - 6 Found

| File | Count | Type |
|------|-------|------|
| `src/diana/data/preprocessing.py` | 3 | Optional features (sparse normalization) |
| `src/diana/inference/feature_extraction.py` | 1 | Optional parallel implementation |
| `src/diana/inference/feature_extractor.py` | 2 | Alternative k-mer counting methods |

**Status:** All TODOs are for optional/future features, not blocking issues

**Details:**
- Sparse matrix normalization: Currently unused (model works without it)
- Parallel feature extraction: Performance optimization (not critical)
- Alternative k-mer counters: Current implementation (back_to_sequences) works

---

## 5. Key File Clarifications

### Accessions File
- **File:** `data/validation/accessions.txt` (879 samples)
- **Purpose:** Initial download list for ALL AncientMetagenomeDir samples
- **Usage:** Used by download scripts (`02_prepare_download.py`, `03_prefetch_all.sh`)
- **Note:** More samples than actually validated (some may have failed download/QC)

### Actual Validation Metadata
- **File:** `data/validation/validation_metadata_expanded.tsv` (616 samples)
- **Purpose:** Final curated validation set after QC and filtering
- **Usage:** Used by comparison script (`06_compare_predictions.py`)
- **Note:** This is the authoritative validation set

**Recommendation:** Keep `accessions.txt` for reproducibility (shows download process)

---

## 6. Cross-Validation During Training

The 5-fold cross-validation properly isolates folds:
- Each fold uses 80% train / 20% validation
- No sample appears in multiple folds simultaneously
- Final model uses 90% train / 10% validation for early stopping

**Files:**
- Split definitions: `data/splits/train_ids.txt`, `data/splits/test_ids.txt`
- Validation during training: Generated dynamically from train split

---

## Conclusion

✅ **All data integrity checks passed**

The DIANA model training pipeline properly isolates:
1. Training data (2,609 samples) - used for model parameter learning
2. Test data (461 samples) - held out for final performance evaluation
3. Validation data (616 samples) - completely independent external dataset

No data leakage detected. Results are scientifically valid.

---

## Recommendations

1. ✅ **No action needed** - Data integrity is sound
2. 📝 Document that `accessions.txt` ≠ `validation_metadata_expanded.tsv` in README
3. 🔍 Investigate 2 missing validation predictions (ERR11413738, ERR6178929)
4. 🧹 Consider documenting TODOs as GitHub issues for future work
