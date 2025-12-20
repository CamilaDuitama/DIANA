# DIANA Testing Suite

Automated tests for validating DIANA's training pipeline and components.

## Quick Start

```bash
# Install package first
pip install -e .

# Run all tests
pytest tests/ -v

# Run only critical end-to-end test
pytest tests/test_smoke.py::TestEndToEnd::test_full_training_pipeline -v

# Run specific category
pytest tests/test_data_loading.py -v
```

## Test Files

| File | Purpose | Key Tests |
|------|---------|-----------|
| `test_smoke.py` | **End-to-end pipeline** | Full training with Optuna (CRITICAL) |
| `test_data_loading.py` | Data loading & metadata | MatrixLoader, label encoding |
| `test_config.py` | Configuration management | YAML/JSON loading, defaults |
| `test_checkpointing.py` | Model checkpointing | Save/load, best model tracking |
| `test_models.py` | Neural network models | Forward passes, output shapes |

## Test Data

**Location**: `tests/fixtures/`
- `dummy_data.pa.mat`: 50 samples × 20 features (space-separated)
- `dummy_metadata.tsv`: Balanced metadata (2-5 classes per task)
- `test_config.yaml`: Minimal test configuration

## What Tests Validate

✅ **End-to-End Pipeline** (Most Important)
- Data loads from real file formats
- Optuna hyperparameter optimization runs (2 trials)
- Multi-task training completes (2 epochs)
- Checkpoints saved correctly
- Results JSON created with proper structure

✅ **Data Integrity**
- Sample IDs match between matrix and metadata
- No NaN/inf values in loaded data
- Binary presence/absence values (0 or 1)
- Label encoding is reversible

✅ **Configuration**
- YAML configs load and override defaults
- Nested access with dot notation works
- Configs can be saved for reproducibility

✅ **Model Checkpointing**
- Checkpoints save with all necessary state
- Model weights preserved exactly through save/load
- Best model tracking based on validation loss

✅ **Models**
- All model architectures initialize correctly
- Forward passes produce expected output shapes
- Multi-task outputs contain all required tasks

## Test Markers

```bash
# Run by test file
pytest tests/test_smoke.py -v          # End-to-end tests
pytest tests/test_data_loading.py -v   # Data tests
pytest tests/test_config.py -v         # Config tests
pytest tests/test_checkpointing.py -v  # Checkpoint tests
pytest tests/test_models.py -v         # Model tests
```

## Understanding Test Results

**PASS**: Component works correctly
**FAIL**: Check error message - usually indicates:
  - API mismatch (parameter name changed)
  - Missing dependency
  - File path issue

The **most critical test** is `test_full_training_pipeline` - if this passes, the complete pipeline is functional.

## Adding New Tests

```python
def test_my_feature(self, dummy_matrix_path):
    \"\"\"
    Brief description of what this test validates.
    
    Explain why this test is important and what failure indicates.
    \"\"\"
    # Arrange
    input_data = create_test_input()
    
    # Act
    result = my_function(input_data)
    
    # Assert
    assert result is not None, "Descriptive error message"
```

## Dependencies

Tests require:
- pytest
- pytorch
- numpy, pandas
- scikit-learn
- pyyaml

All installed with: `pip install -e .`

---

**Note**: Tests use minimal dummy data (50 samples) for speed. They validate code correctness, not model accuracy.
