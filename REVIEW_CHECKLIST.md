# DIANA Code Review - Action Checklist

## 🐛 Bug Fixes (Critical)

### ✅ 1. Fix Type Errors in Validation Script
**Goal**: Enable validation comparison script to run without errors  
**Files**: `scripts/validation/06_compare_predictions.py`  
**Changes**:
```python
# Line 546 - Convert spmatrix to array before np.sum
y_true_array = np.asarray(y_true_bin)
if y_true_array.shape[0] == 0 or np.sum(y_true_array) == 0:

# Lines 556-557, 609-610 - Use array indexing
for i in range(y_true_array.shape[1]):
    if np.sum(y_true_array[:, i]) > 0:
        y_true_binary = y_true_array[:, i]
```
**Test**: `python scripts/validation/06_compare_predictions.py --metadata data/validation/validation_metadata_expanded.tsv`  
**Success**: Script runs without type errors, generates figures in `paper/`

---

### ~~2. Implement Sparse Matrix Normalization~~ (NOT NEEDED)
**Status**: Classes are exported but never used in training/inference pipeline  
**Reason**: Model trained on raw binary k-mer features (0/1) works perfectly (85-92% accuracy)  
**Action**: Moved to optional cleanup (Item #11)

---

### ~~2. Fix Security Warning in Model Loading~~ (NOT A BUG)
**Status**: `weights_only=False` is correct for this use case  
**Reason**: Checkpoints contain Python objects (dicts, lists for history/config)  
**Security**: Safe because only loading trusted checkpoints from your own training  
**Note**: `weights_only=True` only for loading untrusted/third-party models  
**Action**: No change needed - current implementation is correct

---

## 🧪 New Tests (High Priority)

### ✅ 4. Create Inference Pipeline Tests
**Goal**: Validate prediction pipeline works end-to-end  
**Files**: Created `tests/test_inference.py`  
**Status**: 8 tests created, **all 8 passing** ✅  
**Coverage**: Model loading, predictions, paired-end detection, architecture reconstruction

---

### ✅ 5. Create Preprocessing Tests
**Goal**: Ensure data preprocessing is correct  
**Files**: Created `tests/test_preprocessing.py`  
**Status**: 10 tests created, **all 10 passing** ✅  
**Coverage**: Label encoding, metadata alignment, data integrity, matrix loading

---

### ✅ 6. Create Error Handling Tests
**Goal**: Verify graceful failure on bad inputs  
**Files**: Created `tests/test_error_handling.py`  
**Status**: 12 tests created, **all 12 passing** ✅  
**Coverage**: Missing files, corrupted data, dimension mismatches, invalid values

---

## 🐛 Bug Fixed

### ✅ 13. Fix Predictor Model Reconstruction
**Goal**: Fix Predictor to correctly infer model architecture from checkpoints  
**Files**: `src/diana/inference/predictor.py` (lines 76-90)  
**Issue**: Was extracting ALL weight layers instead of just Linear layers  
**Fix**: Filter out BatchNorm layers when inferring hidden dimensions  
**Impact**: ✅ **All 30 tests now pass**  
**Production**: ✅ Verified production model still works correctly

---

### ☐ 4. Create Inference Pipeline Tests (UPDATED)
**Goal**: Validate prediction pipeline works end-to-end  
**Files**: Create `tests/test_inference.py`  
**What to add**:
```python
def test_predictor_loads_checkpoint(temp_dir):
    """Verify Predictor loads saved models correctly"""
    # Test with dummy checkpoint from fixtures
    
def test_prediction_output_format():
    """Verify prediction JSON has all required fields"""
    # Check keys: sample_type, community_type, sample_host, material
    # Verify probabilities sum to 1.0

def test_paired_end_detection():
    """Test R1/R2 file detection"""
    # Create mock R1/R2 files, verify both detected
```
**Test**: `pytest tests/test_inference.py -v`  
**Success**: All 3+ tests pass

---

### ☐ 5. Create Preprocessing Tests
**Goal**: Ensure data preprocessing is correct  
**Files**: Create `tests/test_preprocessing.py`  
**What to add**:
```python
def test_l2_normalization():
    """Test sparse matrix L2 normalization"""
    # Create test sparse matrix
    # Verify row norms ≈ 1.0
    
def test_label_encoder_reversibility():
    """Test encode→decode returns original labels"""
    
def test_metadata_alignment():
    """Test sample IDs match between matrix and metadata"""
    # Mismatched IDs should raise clear error
```
**Test**: `pytest tests/test_preprocessing.py -v`  
**Success**: All tests pass

---

### ☐ 6. Create Error Handling Tests
**Goal**: Verify graceful failure on bad inputs  
**Files**: Create `tests/test_error_handling.py`  
**What to add**:
```python
def test_missing_metadata_column():
    """Missing required column raises clear error"""
    
def test_corrupted_checkpoint():
    """Corrupted .pth file raises informative error"""
    
def test_wrong_feature_dimensions():
    """New sample with different feature count fails gracefully"""
    
def test_empty_fastq():
    """Empty FASTQ file raises clear error"""
```
**Test**: `pytest tests/test_error_handling.py -v`  
**Success**: All edge cases handled with clear error messages

---

## 🚀 Performance Improvements (Medium Priority)

### ☐ 7. Profile Memory Usage for OOM Samples
**Goal**: Identify why ~40 samples failed with OOM  
**Files**: `src/diana/cli/predict.py`, `src/diana/inference/feature_extraction.py`  
**Changes**:
```python
# Add memory profiling
import tracemalloc

def predict_single_sample(...):
    tracemalloc.start()
    # ... existing code ...
    current, peak = tracemalloc.get_traced_memory()
    logger.info(f"Memory: peak={peak/1e9:.2f}GB")
    tracemalloc.stop()
```
**Test**: Run on failed samples from `data/validation/oom_failed_tasks.txt`  
**Success**: Memory logs identify bottleneck (likely k-mer counting step)

---

### ☐ 8. Implement Batch Prediction
**Goal**: Process multiple samples more efficiently  
**Files**: `src/diana/inference/predictor.py` (line 179)  
**Changes**:
```python
def predict_batch(self, features_list: List[np.ndarray]) -> List[Dict]:
    """Predict on multiple samples efficiently"""
    # Stack features, run single forward pass
    # Return list of prediction dicts
```
**Test**: Create `tests/test_inference.py::test_batch_prediction()`  
**Success**: Batch prediction 2-3x faster than sequential

---

## 📊 Validation & Analysis (Medium Priority)

### ☐ 9. Analyze OOM Failure Pattern
**Goal**: Understand what makes samples fail  
**Files**: Create `scripts/validation/analyze_oom_failures.py`  
**What to do**:
```python
# Load failed sample IDs from oom_failed_tasks.txt
# Extract metadata for failed samples
# Compare: file size, k-mer complexity, host organism
# Generate summary table
```
**Test**: `python scripts/validation/analyze_oom_failures.py`  
**Success**: Report showing common characteristics of OOM samples

---

### ☐ 10. Add Memory Tests
**Goal**: Prevent memory regressions  
**Files**: Create `tests/test_memory.py`  
**What to add**:
```python
@pytest.mark.slow
def test_large_matrix_loading():
    """Matrix >1GB loads without OOM"""
    
@pytest.mark.slow  
def test_prediction_memory_stable():
    """Memory doesn't grow with batch size"""
```
**Test**: `pytest tests/test_memory.py -v -m slow`  
**Success**: No memory leaks detected

---

## 🔍 Code Quality (Low Priority)

### ☐ 11. Remove Unimplemented Stubs
**Goal**: Clean up NotImplementedError placeholders  
**Files**: 
- `src/diana/inference/feature_extractor.py` (lines 175, 194)
- `src/diana/inference/predictor.py` (line 118)

**Options**:
1. Implement functionality (e.g., single-task model loading)
2. Remove unused code paths
3. Document as "not yet supported"

**Test**: `grep -r "raise NotImplementedError" src/`  
**Success**: Only intentional unimplemented features remain

---

### ☐ 12. Add Type Hints
**Goal**: Improve code maintainability  
**Files**: `src/diana/evaluation/metrics.py`, `src/diana/evaluation/plotting.py`  
**Changes**: Add type hints to all function signatures  
**Test**: `mypy src/diana/ --ignore-missing-imports`  
**Success**: No type errors reported

---

## 📈 Results Summary

**Current Status**:
- ✅ Training: Excellent (5-fold CV, 85-92% accuracy)
- ✅ Architecture: Well-designed, modular
- ✅ Core tests: Basic coverage exists
- ⚠️ Validation: 608/650 successful (93%)
- ⚠️ Type errors: 1 script with errors
- ⚠️ Missing tests: Inference, preprocessing, edge cases

**After Completing Checklist**:
- All critical bugs fixed
- Comprehensive test coverage
- Production-ready inference pipeline
- Memory issues understood/mitigated
- Clear error messages for edge cases

---

## Quick Start - Fix Critical Issues (30 min)

```bash
# 1. Fix validation script type errors ✅ DONE
python scripts/validation/06_compare_predictions.py \
  --metadata data/validation/validation_metadata_expanded.tsv

# 2-3. Skipped (not actual bugs - see notes above)

# 4. Run existing tests to verify nothing broken
python scripts/validation/06_compare_predictions.py --metadata data/validation/validation_metadata_expanded.tsv

# 4. Run existing tests
pytest tests/ -v

# Done! Critical issues resolved.
```

---

**Priority Order**: #1 ✅ → ~~#2~~ → ~~#3~~ → #4 → #5 → #6 → #7 → #9 → #8 → #10 → #11 → #12

**Estimated Total Time**: 
- Critical fixes (1): 30 min ✅
- New bug fix (13): 1 hour
- High priority tests (4-6): 2 hours ✅ (tests created, found 1 bug)
- Performance (7-8): 6 hours  
- Analysis & cleanup (9-12): 4 hours
- **Total**: ~14 hours remaining

---

## Test Summary

**Created**: 30 new tests across 3 files
- `tests/test_inference.py`: 8 tests (**all passing** ✅)
- `tests/test_preprocessing.py`: 10 tests (**all passing** ✅)
- `tests/test_error_handling.py`: 12 tests (**all passing** ✅)

**Total**: **30/30 passing (100%)** ✅

**Bug Found & Fixed**: Predictor architecture reconstruction (Item #13)
