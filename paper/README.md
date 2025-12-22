# Paper Figures and Tables Reproduction

This directory contains all figures and tables for the DIANA manuscript. All outputs are fully reproducible using the scripts in this repository.

## Directory Structure

```
paper/
├── README.md                 # This file - reproduction instructions
├── reproduce.sh              # Master script to regenerate all outputs
├── figures/                  # All manuscript figures (PNG + HTML interactive)
│   ├── data_distribution/    # Metadata visualizations
│   ├── unitig_*.png/html     # Feature analysis plots
│   └── ...
└── tables/                   # All manuscript tables (CSV + Markdown)
    ├── unitig_*.csv          # Feature statistics
    ├── train_test_*.md       # Split comparison
    └── ...
```

## Requirements

All scripts use the same environment as the DIANA tool:

```bash
# Create environment (if not already done)
mamba env create -f environment.yml -p ./env

# Or activate existing environment
mamba activate diana  # or: mamba run -p ./env <command>
```

**Key dependencies:**
- Python 3.10+
- polars, plotly, biopython (for data/visualization)
- scikit-learn, scipy (for statistics)
- All DIANA package dependencies (torch, etc.)

## Quick Start

**Option 1: Run everything at once**
```bash
cd paper/
./reproduce.sh
```

**Option 2: Run individual steps** (see sections below)

## Reproduction Workflow

### Prerequisites

Ensure you have the final feature matrix ready:
```bash
# Check that matrix exists
ls -lh data/matrices/large_matrix_3070_with_frac/unitigs.frac.mat

# Expected: ~1.6GB file with 3,070 samples × 107,480 unitig features
```

---

### Step 1: Create Train/Test Splits

**Script:** `scripts/data_prep/01_create_splits.py`  
**Input:** `data/metadata/DIANA_metadata.tsv`  
**Output:** `data/splits/{train,test,val}_ids.txt`, `split_config.json`

Creates stratified 85%/15% train/test split based on community type.

```bash
python scripts/data_prep/01_create_splits.py \
    --config configs/data_config.yaml
```

**Outputs:**
- Train: ~2,610 samples
- Test: ~460 samples
- Stratified by community_type to ensure balanced representation

---

### Step 2: Analyze Unitig Features

**Script:** `scripts/data_prep/06_analyze_unitigs.py`  
**Input:** `data/matrices/large_matrix_3070_with_frac/unitigs.{fa,frac.mat}`  
**Output:** Figures 1-4 + Tables 1-3

Analyzes the 107,480 unitig features used for classification.

```bash
python scripts/data_prep/06_analyze_unitigs.py \
    --matrix-dir data/matrices/large_matrix_3070_with_frac \
    --output-figures paper/figures \
    --output-tables paper/tables
```

**Generated figures:**
- `unitig_length_distribution.{png,html}` - Length histogram + boxplot
- `unitig_gc_content.{png,html}` - GC content distribution
- `unitig_sparsity_distribution.{png,html}` - 4-panel sparsity analysis
- `unitig_length_vs_sparsity.{png,html}` - Length-sparsity scatter plots

**Generated tables:**
- `unitig_sequence_stats.csv` - Sequence statistics summary
- `unitig_sparsity_stats.csv` - Sparsity metrics
- `top20_common_unitigs.csv` - Most prevalent features

---

### Step 3: Plot Metadata Distributions

**Script:** `scripts/evaluation/02_plot_data_distribution.py`  
**Input:** `data/metadata/DIANA_metadata.tsv`, `data/splits/`  
**Output:** Figures in `paper/figures/data_distribution/`

Visualizes metadata distributions (geography, time, sample types) for full dataset and train/test splits.

```bash
python scripts/evaluation/02_plot_data_distribution.py \
    --config configs/data_config.yaml
```

**Generated figures:**
- `full_distribution_*.png` - Distributions for each metadata column
- `full_dataset_world_map.{png,html}` - Geographic distribution
- `splits_world_map.{png,html}` - Train/test geographic comparison
- `split_comparison_*.png` - Train vs test for each variable

---

### Step 4: Statistical Tests for Split Balance

**Script:** `scripts/evaluation/03_statistical_tests_splits.py`  
**Input:** `data/metadata/DIANA_metadata.tsv`, `data/splits/`  
**Output:** `paper/tables/train_test_statistical_comparison.md`

Performs Chi-squared and Kolmogorov-Smirnov tests to verify train/test splits maintain similar distributions.

```bash
python scripts/evaluation/03_statistical_tests_splits.py \
    --config configs/data_config.yaml
```

**Generated table:**
- `train_test_statistical_comparison.md` - Statistical test results for all variables
  - Reports p-values for each metadata column
  - Identifies any significant differences between splits
  - Uses α = 0.05 significance level

---

## Customization

All scripts accept command-line arguments to customize paths:

**Analyze different matrix:**
```bash
python scripts/data_prep/06_analyze_unitigs.py \
    --matrix-dir /path/to/alternative/matrix \
    --output-figures paper/figures/alternative \
    --output-tables paper/tables/alternative
```

**Different metadata file:**
```bash
python scripts/evaluation/02_plot_data_distribution.py \
    --config /path/to/custom_config.yaml
```

**Custom output directories:**
```bash
python scripts/data_prep/06_analyze_unitigs.py \
    --output-figures output/figures \
    --output-tables output/tables
```

---

## Expected Runtime

On a standard workstation (16 cores, 32GB RAM):

| Step | Script | Runtime | Notes |
|------|--------|---------|-------|
| 1 | 01_create_splits.py | ~5 seconds | Fast - just metadata |
| 2 | 06_analyze_unitigs.py | ~2-3 minutes | Loads 1.6GB matrix + generates plots |
| 3 | 02_plot_data_distribution.py | ~30 seconds | Metadata only |
| 4 | 03_statistical_tests_splits.py | ~10 seconds | Statistical tests |

**Total: ~4 minutes**

---

## Troubleshooting

**"Matrix not found" error:**
```bash
# Ensure you have the correct matrix directory
ls data/matrices/large_matrix_3070_with_frac/
# Should contain: unitigs.fa, unitigs.frac.mat, etc.
```

**"Kaleido version warning":**
```bash
# Safe to ignore - PNG export still works with older Kaleido
# Or update: mamba install -c conda-forge kaleido
```

**Memory errors:**
```bash
# Loading 1.6GB matrix requires ~8GB RAM
# Close other applications or use a machine with more memory
```

**Missing dependencies:**
```bash
# Reinstall environment
mamba env remove -p ./env
mamba env create -f environment.yml -p ./env
```

---

## Citation

If you use these figures/tables, please cite:

> Duitama et al. (2025). DIANA: Multi-task classification of ancient DNA samples 
> using the largest reference database. [Journal Name].

---

## Files Manifest

This section lists all expected outputs after running the reproduction workflow.

### Figures (paper/figures/)

**Unitig feature analysis:**
- `unitig_length_distribution.png` + `.html`
- `unitig_gc_content.png` + `.html`
- `unitig_sparsity_distribution.png` + `.html`
- `unitig_length_vs_sparsity.png` + `.html`

**Metadata distributions (paper/figures/data_distribution/):**
- `full_distribution_<column>.png` - One per metadata column
- `full_dataset_world_map.png` + `.html`
- `splits_world_map.png` + `.html`
- `split_comparison_<column>.png` - Train vs test for each column

### Tables (paper/tables/)

**Unitig statistics:**
- `unitig_sequence_stats.csv` - Length, GC, N-content summary
- `unitig_sparsity_stats.csv` - Sparsity metrics
- `top20_common_unitigs.csv` - Most prevalent features

**Split validation:**
- `train_test_statistical_comparison.md` - Statistical tests

---

## Contact

For questions about reproduction:
- Open an issue on GitHub: https://github.com/CamilaDuitama/DIANA/issues
- Email: camiladuitama@gmail.com
