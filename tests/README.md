# DIANA Testing Suite

Automated tests for validating DIANA's inference pipeline and components.

## Quick Start

```bash
# Install package first
pip install -e ".[dev]"

# Run all unit/component tests (fast, no model artifacts needed)
pytest tests/ -v

# Run the full end-to-end integration test (requires model artifacts from install.sh)
bash tests/test_integration.sh --verbose

# Run a specific test module
pytest tests/test_models.py -v
```

## Test Files

| File | Purpose | Key Tests |
|------|---------|-----------|
| `test_smoke.py` | Component smoke tests | Model init, forward pass |
| `test_inference.py` | Inference pipeline | predict\_single\_sample, output format |
| `test_preprocessing.py` | Input preprocessing | PCA transform, feature alignment |
| `test_data_loading.py` | Data loading & metadata | MatrixLoader, label encoding |
| `test_config.py` | Configuration management | YAML/JSON loading, defaults |
| `test_checkpointing.py` | Model checkpointing | Save/load, best model tracking |
| `test_models.py` | Neural network models | Forward passes, output shapes |
| `test_error_handling.py` | Error handling | Invalid inputs, missing files |

**End-to-end test** (separate from pytest):

| Script | Purpose |
|--------|---------|
| `test_integration.sh` | Full pipeline: k-mer counting → unitig aggregation → `diana-predict` → validate JSON output |

## Test Data

**Location**: `tests/fixtures/`
- `dummy_data.pa.mat`: 50 samples × 20 features (space-separated)
- `dummy_metadata.tsv`: Balanced metadata (2-5 classes per task)
- `test_config.yaml`: Minimal test configuration

## What Tests Validate

✅ **End-to-End Prediction** (Primary — `test_integration.sh`)
- FASTQ input → k-mer counting → unitig aggregation
- `diana-predict` produces predictions JSON
- All four tasks predicted: `sample_type`, `community_type`, `sample_host`, `material`
- Output format matches expected structure

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
pytest tests/test_inference.py -v      # Inference pipeline tests
pytest tests/test_models.py -v         # Model architecture tests
pytest tests/test_data_loading.py -v   # Data loading tests
pytest tests/test_config.py -v         # Config tests
pytest tests/test_checkpointing.py -v  # Checkpoint tests
```

## Understanding Test Results

**PASS**: Component works correctly
**FAIL**: Check error message — usually indicates:
  - API mismatch (parameter name changed)
  - Missing dependency
  - File path issue

The **primary end-to-end test** is `tests/test_integration.sh` — run it after `install.sh` to confirm the full prediction pipeline is functional on real data.

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

Unit/component tests (`pytest tests/`):
- Installed with `pip install -e ".[dev]"`

Integration test (`bash tests/test_integration.sh`):
- Requires the full install: `mamba env create -f environment.yml -p ./env && mamba activate ./env && bash install.sh`
- Needs test FASTQ files in `test_data/` (bundled in the repo)
- Needs model artifacts in `results/training/best_model.pth` and `models/pca_reference.pkl`

> **Note:** `test_smoke.py::TestEndToEnd::test_full_training_pipeline` is skipped in CI.
> It validates the retraining pipeline and requires `pip install -e ".[optim]"`.
> Run it manually if you need to validate retraining: `pytest tests/test_smoke.py -v -k test_full_training_pipeline --no-header`

---

**Note**: Tests use minimal dummy data (50 samples) for speed. They validate code correctness, not model accuracy.
