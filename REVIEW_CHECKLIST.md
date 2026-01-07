# DIANA Code Review - Action Checklist

## ✅ Completed

### 1. Fix Type Errors in Validation Script
**Files**: `scripts/validation/06_compare_predictions.py` (line 542)  
**Fix**: Convert sparse matrix to dense before operations  
**Result**: ✅ Script runs without errors

### 2. Create Inference Pipeline Tests  
**Files**: `tests/test_inference.py`  
**Result**: ✅ 18 tests passing (model loading, predictions, label encoding, validation)

### 3. Create Preprocessing Tests
**Files**: `tests/test_preprocessing.py`  
**Result**: ✅ 10 tests passing (data integrity, metadata alignment)

### 4. Create Error Handling Tests
**Files**: `tests/test_error_handling.py`  
**Result**: ✅ 12 tests passing (missing files, corrupted data, edge cases)

### 5. Fix Predictor Architecture Bug
**Files**: `src/diana/inference/predictor.py` (lines 76-90)  
**Fix**: Filter BatchNorm layers when inferring hidden dimensions  
**Result**: ✅ All 68 tests passing, production model verified

---

## 🔴 Critical - Data Integrity

### 6. Review Data Leakage & Splits
**Goal**: Verify no overlap between train/val/test sets  
**Check**:
- [ ] No sample IDs appear in multiple splits
- [ ] Validation samples not in training data
- [ ] Test evaluation uses correct data splits
- [ ] Cross-validation folds properly isolated

**Files to check**:
- `data/splits/*.txt` - train/val/test IDs
- `scripts/training/*.py` - data loading logic
- `scripts/evaluation/*.py` - evaluation paths
- `data/validation/accessions.txt` - validation sample IDs

**Test**: Compare sample IDs across all splits, check for duplicates

---

### 7. Review Hardcoded Paths
**Goal**: Find and document all hardcoded paths  
**Search**: `grep -r "'/pasteur\|/home\|/data" src/ scripts/ --include="*.py"`  
**Files to check**:
- Config files: `configs/*.yaml`
- Training scripts: `scripts/training/`
- Inference scripts: `scripts/inference/`
- Validation scripts: `scripts/validation/`

**Action**: Create path configuration or use environment variables

---

### 8. Review TODOs in Code
**Goal**: Identify unfinished work or technical debt  
**Search**: `grep -r "TODO\|FIXME\|HACK\|XXX" src/ scripts/ --include="*.py"`  
**Categorize**:
- Critical: Blocks functionality
- Important: Should be done soon
- Optional: Nice to have
- Remove: No longer relevant

**Action**: Create GitHub issues or remove obsolete TODOs

---

## 🟡 Performance

### 9. Profile Memory Usage (OOM Issues)
**Goal**: Why ~40 validation samples failed with OOM  
**Files**: Add profiling to `src/diana/cli/predict.py`  
**Test**: Run on failed samples from `data/validation/oom_failed_tasks.txt`

### 10. Implement Batch Prediction
**Goal**: Process multiple samples efficiently  
**Files**: `src/diana/inference/predictor.py`  
**Benefit**: 2-3x faster inference

### 11. Analyze OOM Failure Patterns
**Goal**: Understand sample characteristics causing OOM  
**Create**: `scripts/validation/analyze_oom_failures.py`  
**Output**: Summary table of failed samples

---

## 🟢 Code Quality

### 12. Remove Unimplemented Stubs
**Files**: `src/diana/inference/` - NotImplementedError placeholders  
**Action**: Implement, remove, or document as unsupported

### 13. Add Type Hints
**Files**: `src/diana/evaluation/metrics.py`, `src/diana/evaluation/plotting.py`  
**Test**: `mypy src/diana/ --ignore-missing-imports`

---

## Summary

**Tests**: 68 passing (40 new tests created)  
**Bugs Fixed**: 2 (validation script, predictor architecture)  
**Training**: 85-92% accuracy (5-fold CV)  
**Validation**: 608/650 samples successful (93%)  

**Next Priority**: Items #6-8 (data integrity verification)
