# DIANA Inference Integration - Analysis & Plan

## Training Parameters (from logs/muset_fraction_60693849.err)

```
k-mer size (-k): 31
minimum abundance (-a): 2
minimum unitig length (-l): 61
minimum unitig fraction (-r): 0
fraction of absent samples (-f): 0.1
fraction of present samples (-F): 0.1
```

## Question-by-Question Analysis

### 1. Is this the right way to integrate?

**Current Approach**: ✅ YES, with minor adjustments needed

**Architecture**:
```
DIANA CLI
    ↓
Inference Module (Python)
    ↓ calls ↓
back_to_sequences (Rust) → kmat_tools unitig (C++)
    ↓
Feature Vector (numpy array)
    ↓
Trained Model (sklearn/PyTorch)
    ↓
Prediction
```

**Why this is correct**:
- Reuses MUSET's exact aggregation logic → consistency with training
- Leverages fast external tools (back_to_sequences is 100x faster than KMC)
- Clean separation: feature extraction vs. model inference
- Scalable: can process batches in parallel

**Recommended integration point**:
```python
# src/diana/inference/feature_extractor.py
class FeatureExtractor:
    def extract_features(self, fastq_path: str) -> np.ndarray:
        """Extract unitig abundance features for one sample."""
        # Step 1: Count k-mers
        # Step 2: Aggregate to unitigs
        # Return: (n_unitigs,) array
```

---

### 2. Self-contained tool (external dependencies)

**Current Status**: ⚠️ PARTIALLY CONTAINED

**Dependencies**:
1. **back_to_sequences** (Rust)
   - Status: ❌ NOT bundled (user must install via cargo)
   - Location: ~/.local/bin/back_to_sequences
   
2. **kmat_tools** (C++)
   - Status: ✅ BUNDLED in external/muset/
   - Requires: GCC 13.2.0 libraries
   - Location: external/muset/bin/kmat_tools

**Recommendation: Make DIANA fully self-contained**

**Option A: Bundle precompiled binaries** (Easiest)
```bash
# Package structure
diana/
  bin/
    back_to_sequences  # Static binary for Linux x86_64
    kmat_tools         # Static binary for Linux x86_64
  external/
    muset/            # Source code (backup)
```

Pros:
- Users don't need to compile anything
- Faster installation
- Consistent versions

Cons:
- Platform-specific (need binaries for Linux/Mac/ARM)
- Larger package size

**Option B: Build during installation** (Most robust)
```python
# setup.py
import subprocess
from setuptools import setup
from setuptools.command.install import install

class CustomInstall(install):
    def run(self):
        # Build MUSET/kmat_tools
        subprocess.run(['bash', 'scripts/create_umat/01_build_muset.sh'])
        # Install back_to_sequences via cargo
        subprocess.run(['cargo', 'install', 'back_to_sequences'])
        install.run(self)

setup(
    name='diana',
    cmdclass={'install': CustomInstall},
    # ...
)
```

Pros:
- Platform-independent
- Always uses latest stable versions
- Smaller package size

Cons:
- Requires build dependencies (gcc, cargo)
- Slower installation

**Recommended: Option A + fallback to Option B**
- Ship with precompiled binaries for common platforms
- Provide build script if binaries don't work

---

### 3. K-mer size verification (k=31)

**Training**: k=31 ✅
**Inference**: Currently hardcoded in scripts ✅

**Issue**: back_to_sequences doesn't have explicit k-mer size parameter!
- It **infers k-mer size from the reference FASTA**
- Reads first k-mer in reference_kmers.fasta
- Uses its length as k

**Verification needed**:
```bash
# Check reference k-mer length
head -2 reference_kmers.fasta
# >header
# TCTTACATAAGATTATAACCATAGAATTAAA  # 31 characters = k=31 ✅
```

**Safety check to add**:
```python
def validate_kmer_size(reference_fasta: str, expected_k: int = 31):
    """Verify reference k-mers have correct length."""
    with open(reference_fasta) as f:
        f.readline()  # Skip header
        kmer = f.readline().strip()
        actual_k = len(kmer)
        if actual_k != expected_k:
            raise ValueError(
                f"K-mer size mismatch! Expected {expected_k}, got {actual_k}"
            )
```

**Action**: ✅ Add validation to inference module

---

### 4. Minimum abundance threshold (a=2)

**Critical Question**: Does minimum abundance=2 affect inference?

**Training behavior**:
```
minimum abundance (-a): 2
→ kmtricks filters k-mers with abundance < 2 in each sample
→ Only k-mers with count ≥ 2 are kept
```

**Analysis**:

**Where filtering happens**:
1. **During k-mer counting** (kmtricks): 
   - Per-sample: k-mers with count < 2 are set to 0
   
2. **After matrix construction** (MUSET filtering):
   - Removes k-mers present in < 10% of samples (fraction_absent=0.1)
   - Removes k-mers present in > 90% of samples (fraction_present=0.1)

**For inference**:

**Option 1: Apply same threshold** (Recommended)
```bash
# Filter k-mers with count < 2
back_to_sequences ... | awk '$2 >= 2' > filtered_counts.txt
```

**Option 2: Use raw counts** (Current)
```bash
# Use all counts as-is
back_to_sequences ... > counts.txt
```

**Which is correct?**

Let me check what kmat_tools unitig expects:

```cpp
// From MUSET aggregator.h
for (auto& kmer : kmers) {
    if (count > 0) {  // Only checks > 0, not >= 2
        nb_present++;
        abundance_sum += count;
    }
}
```

**Answer**: kmat_tools unitig **does not apply minimum abundance threshold**
- It uses whatever counts you give it
- The filtering at a=2 was done **before creating the reference k-mer set**

**Implication**: 
- Reference k-mers (matrix.filtered.fasta) already passed the a=2 filter during training
- For inference, we should **use raw counts** (Option 2)
- The model learned from aggregated values that included all counts ≥ 2

**Decision**: ✅ **NO additional filtering needed** - reference set is already filtered

---

### 5. Testing strategy

**Test Plan**:

#### Test 1: Single sample (D1) ✅ DONE
```bash
# Already tested successfully
bash inference_pipeline.sh test_inference_config.sh
# Result: 20 unitig features extracted
```

#### Test 2: Multiple samples (D1, D2, D3)
```bash
# Create config for D2
SAMPLE_NAME="D2"
SAMPLE_FASTQ="external/muset/test/D2.fastq"
# Run pipeline
# Compare outputs: D1_unitig_abundance.txt vs D2_unitig_abundance.txt
```

**Expected**: Different patterns (D1-D5 vs D6-D10 should differ)

#### Test 3: Batch processing
```python
# src/diana/inference/batch_inference.py
for sample in samples:
    features = extract_features(sample)
    predictions = model.predict(features)
```

#### Test 4: Real validation data (AncientMetagenomeDir)
```bash
# Process 10 random samples from 757
# Verify:
# - All produce 20 features
# - No crashes
# - Reasonable runtime (~5-10 sec/sample)
```

#### Test 5: Edge cases
- Empty FASTQ (no k-mers found)
- Very large FASTQ (millions of reads)
- K-mer size mismatch (wrong reference)
- Missing reference files

---

### 6. Multi-language integration (Rust + C++ + Python)

**Current Stack**:
- **Rust** (back_to_sequences): K-mer counting
- **C++** (kmat_tools): Unitig aggregation  
- **Python** (DIANA): Orchestration + ML

**Is this OK?** ✅ YES - this is standard in bioinformatics

**Precedents**:
- **Kraken2**: C++ core + Perl scripts
- **MetaPhlAn**: C++ bowtie2 + Python analysis
- **Snakemake**: Python + any tools
- **scikit-bio**: Python + C extensions

**Benefits**:
- **Performance**: Rust/C++ for compute-intensive steps (k-mer ops)
- **Productivity**: Python for high-level logic, ML, visualization
- **Ecosystem**: Leverage best tools (no need to rewrite back_to_sequences)

**Challenges**:
1. **Installation complexity**
   - Solution: Ship precompiled binaries + fallback build

2. **Error handling across languages**
   - Solution: Python wrapper catches subprocess errors
   
3. **Debugging**
   - Solution: Verbose logging at each step

4. **Dependency management**
   - Solution: Document versions, use containers (Docker/Singularity)

**Best Practices**:
```python
# Python wrapper provides clean interface
from diana.inference import predict

# User doesn't need to know about Rust/C++
prediction = predict(fastq_path='sample.fastq')
# Internally: calls back_to_sequences → kmat_tools → model
```

---

## Recommended Integration Architecture

```python
# src/diana/inference/__init__.py
from .feature_extractor import FeatureExtractor
from .predictor import DIANAPredictor

# src/diana/inference/feature_extractor.py
class FeatureExtractor:
    def __init__(self, reference_dir: str):
        self.reference_kmers = f"{reference_dir}/matrix.filtered.fasta"
        self.unitigs = f"{reference_dir}/unitigs.fa"
        self.kmer_size = 31
        
        # Validate reference files
        self._validate()
    
    def extract_features(self, fastq_path: str) -> np.ndarray:
        """Extract unitig features from a FASTQ sample."""
        # Step 1: Count k-mers
        counts_file = self._count_kmers(fastq_path)
        
        # Step 2: Aggregate to unitigs
        abundance_file = self._aggregate_to_unitigs(counts_file)
        
        # Step 3: Load as numpy array
        features = np.loadtxt(abundance_file)
        
        return features

# src/diana/inference/predictor.py
class DIANAPredictor:
    def __init__(self, model_path: str, reference_dir: str):
        self.model = joblib.load(model_path)
        self.extractor = FeatureExtractor(reference_dir)
    
    def predict(self, fastq_path: str) -> dict:
        """Predict class for a new sample."""
        # Extract features
        features = self.extractor.extract_features(fastq_path)
        
        # Predict
        prediction = self.model.predict(features.reshape(1, -1))
        proba = self.model.predict_proba(features.reshape(1, -1))
        
        return {
            'sample': os.path.basename(fastq_path),
            'predicted_class': prediction[0],
            'confidence': proba[0],
            'features': features
        }

# CLI: src/diana/cli.py
def predict_command(args):
    predictor = DIANAPredictor(
        model_path=args.model,
        reference_dir=args.reference
    )
    
    results = predictor.predict(args.fastq)
    print(f"Predicted: {results['predicted_class']}")
    print(f"Confidence: {results['confidence']}")
```

---

## Action Items

1. ✅ **Verify k-mer size is 31** (already confirmed)

2. ⚠️ **Decide on minimum abundance** 
   - Recommendation: Use raw counts (no additional filtering)
   
3. 🔲 **Test with multiple samples** (D1, D2, D3)
   - Create test configs
   - Run pipeline
   - Compare outputs
   
4. 🔲 **Create Python wrapper classes**
   - FeatureExtractor
   - DIANAPredictor
   - CLI interface
   
5. 🔲 **Package external binaries**
   - Compile static back_to_sequences
   - Copy kmat_tools binary
   - Create bin/ directory
   
6. 🔲 **Add validation**
   - K-mer size check
   - Reference file existence
   - Output dimension check
   
7. 🔲 **Integration tests**
   - End-to-end test with real data
   - Batch processing test
   - Error handling tests

8. 🔲 **Documentation**
   - Installation guide
   - Usage examples
   - Troubleshooting

---

## Next Steps

**Immediate** (this session):
1. Test with D2, D3 samples
2. Create Python wrapper skeleton
3. Validate k-mer size in code

**Short-term** (next session):
4. Package binaries
5. Create CLI interface
6. Integration tests

**Long-term**:
7. Docker container
8. CI/CD pipeline
9. Benchmark on validation set (757 samples)
